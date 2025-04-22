import os
import numpy as np
import torch
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class ActiveLearner:
    def __init__(
        self,
        model_builder,
        uncertainty_sampler,
        diversity_selector=None,
        train_kwargs=None,
        output_dir="checkpoints",
        initial_samples_ratio=0.05,
        budget_per_round=0.05,
        max_rounds=5,
        eval_metric="macro_f1",
        seed=42,
        device=None
    ):
        # 设置设备
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 模型和采样器
        self.model_builder = model_builder  # 一个函数，用于构建和训练模型
        self.uncertainty_sampler = uncertainty_sampler
        self.diversity_selector = diversity_selector
        
        # 训练参数
        self.train_kwargs = train_kwargs or {}
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 主动学习参数
        self.initial_samples_ratio = initial_samples_ratio
        self.budget_per_round = budget_per_round
        self.max_rounds = max_rounds
        self.eval_metric = eval_metric
        self.seed = seed
        
        # TensorBoard日志
        self.tb_writer = SummaryWriter(os.path.join(output_dir, "active_learning_logs"))
        
        # 存储主动学习过程中的信息
        self.history = {
            "labeled_indices": [],
            "train_metrics": [],
            "val_metrics": [],
            "test_metrics": []
        }
    
    def run(self, train_df, val_df=None, test_df=None, initial_indices=None, augmented_df=None):
        """
        运行主动学习循环
        
        参数:
        train_df: 训练数据
        val_df: 验证数据（可选）
        test_df: 测试数据（可选）
        initial_indices: 初始标注样本的索引列表（可选，如果不提供，则使用k-means++选择）
        augmented_df: 增强后的数据（可选，用于UDA）
        
        返回:
        训练好的最终模型和学习历史记录
        """
        logger.info("开始主动学习循环")
        
        # 检查是否有保存的恢复点
        checkpoint_path = os.path.join(self.output_dir, "active_learning_state.pt")
        start_round = 0
        labeled_indices = None
        best_val_metric = 0.0
        best_model = None
        
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path)
                start_round = checkpoint.get("round", 0)
                saved_indices = checkpoint.get("labeled_indices")
                best_val_metric = checkpoint.get("best_val_metric", 0.0)
                
                if saved_indices and len(saved_indices) > 0:
                    logger.info(f"从检查点恢复：轮次 {start_round}，已标注样本: {len(saved_indices)}")
                    
                    # 验证索引在有效范围内
                    if all(0 <= idx < len(train_df) for idx in saved_indices):
                        labeled_indices = saved_indices
                        self.history["labeled_indices"] = checkpoint.get("history_labeled_indices", [])
                        self.history["train_metrics"] = checkpoint.get("train_metrics", [])
                        self.history["val_metrics"] = checkpoint.get("val_metrics", [])
                        self.history["test_metrics"] = checkpoint.get("test_metrics", [])
                    else:
                        logger.warning("保存的索引无效，将重新开始主动学习过程")
                        start_round = 0
                        
                # 尝试加载最佳模型
                best_model_path = os.path.join(self.output_dir, "best_model")
                if os.path.exists(best_model_path):
                    try:
                        # 调用模型构建器创建基础模型结构
                        temp_model, _ = self.model_builder(
                            pd.DataFrame([train_df.iloc[0]]),  # 用一个样本初始化
                            output_dir=None,
                            **{k: v for k, v in self.train_kwargs.items() if k != 'augmented_df'}
                        )
                        
                        # 加载权重
                        if hasattr(temp_model, "load_state_dict"):
                            state_dict_path = os.path.join(best_model_path, "pytorch_model.bin")
                            if os.path.exists(state_dict_path):
                                temp_model.load_state_dict(torch.load(state_dict_path))
                                best_model = temp_model
                                logger.info("成功加载最佳模型权重")
                    except Exception as e:
                        logger.warning(f"加载最佳模型失败: {str(e)}")
            except Exception as e:
                logger.error(f"恢复检查点失败: {str(e)}")
                logger.info("将重新开始主动学习过程")
                start_round = 0
        
        # 如果提供了增强数据，记录其大小
        if augmented_df is not None:
            logger.info(f"使用增强数据集，大小: {len(augmented_df)}")
            
            # 更新train_kwargs，加入augmented_df
            self.train_kwargs['augmented_df'] = augmented_df
        
        # 1. 选择/使用初始标注样本
        if labeled_indices is None:
            if initial_indices is None and self.diversity_selector:
                logger.info("使用k-means++选择初始种子集...")
                n_initial = int(len(train_df) * self.initial_samples_ratio)
                initial_indices = self.diversity_selector.select_seeds(train_df, seed_ratio=self.initial_samples_ratio)
                logger.info(f"已选择 {len(initial_indices)} 个初始种子样本")
            elif initial_indices is not None:
                logger.info(f"使用提供的 {len(initial_indices)} 个初始种子样本")
            else:
                # 随机选择初始样本
                n_initial = int(len(train_df) * self.initial_samples_ratio)
                initial_indices = np.random.choice(len(train_df), n_initial, replace=False).tolist()
                logger.info(f"随机选择了 {len(initial_indices)} 个初始种子样本")
            
            # 复制一份，确保不修改原始参数
            labeled_indices = initial_indices.copy()
            self.history["labeled_indices"].append(labeled_indices.copy())
        
        # 记录总样本数和初始标注比例
        total_samples = len(train_df)
        logger.info(f"总样本数: {total_samples}, 初始标注比例: {len(labeled_indices)/total_samples:.2%}")
        
        # 2. 迭代主动学习循环
        current_model = None
        
        for round_idx in range(start_round, self.max_rounds):
            logger.info(f"开始主动学习轮次 {round_idx + 1}/{self.max_rounds}")
            logger.info(f"当前已标注样本数: {len(labeled_indices)}/{total_samples} ({len(labeled_indices)/total_samples:.2%})")
            
            # 2.1 使用当前标注样本训练模型
            logger.info("使用当前标注样本训练模型...")
            labeled_df = train_df.iloc[labeled_indices].copy()
            
            # 调用模型构建器训练模型
            model, train_metrics = self.model_builder(
                labeled_df, 
                val_df=val_df,
                output_dir=os.path.join(self.output_dir, f"round_{round_idx}"),
                **self.train_kwargs
            )
            current_model = model
            
            # 记录训练指标
            self.history["train_metrics"].append(train_metrics)
            self.tb_writer.add_scalar("train/labeled_ratio", len(labeled_indices)/total_samples, round_idx)
            for metric_name, metric_value in train_metrics.items():
                self.tb_writer.add_scalar(f"train/{metric_name}", metric_value, round_idx)
            
            # 2.2 在验证集上评估模型
            if val_df is not None:
                logger.info("在验证集上评估模型...")
                val_metrics = self._evaluate_model(current_model, val_df)
                self.history["val_metrics"].append(val_metrics)
                
                # 记录验证指标
                for metric_name, metric_value in val_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        self.tb_writer.add_scalar(f"val/{metric_name}", metric_value, round_idx)
                
                # 更新最佳模型
                if val_metrics.get(self.eval_metric, 0) > best_val_metric:
                    best_val_metric = val_metrics.get(self.eval_metric, 0)
                    best_model = model
                    # 保存最佳模型
                    model_path = os.path.join(self.output_dir, "best_model")
                    if hasattr(model, "save_pretrained"):
                        model.save_pretrained(model_path)
                    else:
                        torch.save(model.state_dict(), os.path.join(model_path, "model.pt"))
            
            # 2.3 在测试集上评估模型（可选）
            if test_df is not None:
                logger.info("在测试集上评估模型...")
                test_metrics = self._evaluate_model(current_model, test_df)
                self.history["test_metrics"].append(test_metrics)
                
                # 记录测试指标
                for metric_name, metric_value in test_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        self.tb_writer.add_scalar(f"test/{metric_name}", metric_value, round_idx)
            
            # 保存当前状态用于恢复
            try:
                checkpoint = {
                    "round": round_idx + 1,  # 保存下一轮的索引
                    "labeled_indices": labeled_indices,
                    "best_val_metric": best_val_metric,
                    "history_labeled_indices": self.history["labeled_indices"],
                    "train_metrics": self.history["train_metrics"],
                    "val_metrics": self.history["val_metrics"],
                    "test_metrics": self.history["test_metrics"]
                }
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"保存主动学习状态到 {checkpoint_path}")
            except Exception as e:
                logger.error(f"保存检查点失败: {str(e)}")
            
            # 2.4 如果还有更多轮次，选择新样本
            if round_idx < self.max_rounds - 1:
                # 计算本轮要增加的样本数
                budget = int(total_samples * self.budget_per_round)
                logger.info(f"为下一轮选择 {budget} 个新样本...")
                
                # 使用不确定性采样器选择新样本
                self.uncertainty_sampler.model = current_model
                
                # 确保uncertainty_sampler有tokenizer
                if not hasattr(self.uncertainty_sampler, 'tokenizer') or self.uncertainty_sampler.tokenizer is None:
                    logger.warning("不确定性采样器缺少tokenizer，尝试使用模型的tokenizer")
                    if hasattr(current_model, 'tokenizer') and current_model.tokenizer is not None:
                        self.uncertainty_sampler.tokenizer = current_model.tokenizer
                    else:
                        logger.warning("模型也缺少tokenizer，尝试创建新的tokenizer")
                        try:
                            from transformers import BertTokenizer
                            self.uncertainty_sampler.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
                        except Exception as e:
                            logger.error(f"创建tokenizer失败: {str(e)}")
                            raise RuntimeError("无法为不确定性采样器创建tokenizer")
                
                new_indices = self.uncertainty_sampler.select_samples(
                    train_df, 
                    labeled_indices, 
                    budget, 
                    self.diversity_selector
                )
                
                # 更新已标注样本集
                labeled_indices.extend(new_indices)
                self.history["labeled_indices"].append(labeled_indices.copy())
                logger.info(f"已选择 {len(new_indices)} 个新样本，总共有 {len(labeled_indices)} 个标注样本")
        
        # 3. 使用所有标注样本进行最终训练（如果需要）
        if len(labeled_indices) < total_samples:
            logger.info("使用所有标注样本进行最终训练...")
            labeled_df = train_df.iloc[labeled_indices].copy()
            
            final_model, final_train_metrics = self.model_builder(
                labeled_df, 
                val_df=val_df,
                output_dir=os.path.join(self.output_dir, "final_model"),
                **self.train_kwargs
            )
            
            # 在验证集上评估最终模型
            if val_df is not None:
                final_val_metrics = self._evaluate_model(final_model, val_df)
                
                # 决定返回哪个模型（最佳验证性能或最终模型）
                if final_val_metrics.get(self.eval_metric, 0) > best_val_metric:
                    best_model = final_model
                    best_val_metric = final_val_metrics.get(self.eval_metric, 0)
        
        # 4. 绘制主动学习曲线
        self._plot_learning_curves()
        
        logger.info("主动学习循环完成")
        return best_model if best_model is not None else current_model, self.history
    
    def _evaluate_model(self, model, df):
        """
        评估模型在给定数据集上的性能
        """
        from src.utils.data_utils import ACSADataset, FLAT_ASPECTS
        
        model.eval()
        dataset = ACSADataset(df, model.tokenizer if hasattr(model, "tokenizer") else None)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        all_preds = []
        all_labels = []
        aspect_preds = {aspect: [] for aspect in range(len(FLAT_ASPECTS))}
        aspect_labels = {aspect: [] for aspect in range(len(FLAT_ASPECTS))}
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].cpu().numpy()
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # 处理不同类型的模型输出
                if isinstance(outputs, dict):
                    logits = outputs.get("logits", None)
                    # 如果没有logits键，尝试其他可能的键
                    if logits is None and len(outputs) > 0:
                        # 尝试获取第一个值，可能是logits
                        logits = list(outputs.values())[0]
                elif isinstance(outputs, tuple):
                    # 如果是元组，通常第二个元素是logits (loss, logits)
                    logits = outputs[1] if len(outputs) > 1 else outputs[0]
                else:
                    # 假设outputs本身就是logits
                    logits = outputs
                
                # 确保logits是张量
                if not isinstance(logits, torch.Tensor):
                    logger.error(f"无法获取有效的logits，输出类型: {type(outputs)}")
                    logger.error(f"模型输出内容: {outputs}")
                    raise TypeError(f"模型输出无法转换为有效的logits tensor: {type(outputs)}")
                
                # 获取预测类别
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                
                # 筛选出有效标签（非0的标签，0表示不适用）
                for i in range(len(labels)):
                    for j in range(len(FLAT_ASPECTS)):
                        if labels[i][j] != 0:  # 0表示"不适用"，不计入评估
                            all_labels.append(labels[i][j])
                            all_preds.append(preds[i][j])
                            # 记录每个方面的预测和标签
                            aspect_preds[j].append(preds[i][j])
                            aspect_labels[j].append(labels[i][j])
        
        # 计算性能指标
        metrics = {}
        
        # 检查是否有有效预测
        if not all_preds:
            logger.warning("没有有效的预测结果，无法计算性能指标")
            return {"accuracy": 0.0, "macro_f1": 0.0}
        
        # 计算整体准确率
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds) if all_preds else 0
        metrics["accuracy"] = accuracy
        
        # 计算整体F1分数 - 添加zero_division参数
        try:
            macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            metrics["macro_f1"] = macro_f1
        except Exception as e:
            logger.error(f"计算F1分数时出错: {str(e)}")
            metrics["macro_f1"] = 0.0
        
        # 分析每个方面的表现
        aspect_metrics = {}
        for j, aspect in enumerate(FLAT_ASPECTS):
            if aspect_labels[j]:
                aspect_acc = sum(p == l for p, l in zip(aspect_preds[j], aspect_labels[j])) / len(aspect_labels[j])
                try:
                    aspect_f1 = f1_score(aspect_labels[j], aspect_preds[j], average="macro", zero_division=0)
                    # 记录样本数和类别分布
                    aspect_counts = {}
                    for label in aspect_labels[j]:
                        aspect_counts[int(label)] = aspect_counts.get(int(label), 0) + 1
                    
                    aspect_metrics[aspect] = {
                        "accuracy": aspect_acc,
                        "f1": aspect_f1,
                        "samples": len(aspect_labels[j]),
                        "label_distribution": aspect_counts
                    }
                except Exception as e:
                    logger.warning(f"计算方面 {aspect} 的F1分数时出错: {str(e)}")
        
        # 记录有明显性能差异的方面
        poor_aspects = [a for a, m in aspect_metrics.items() if m.get("f1", 0) < 0.3 and m.get("samples", 0) > 10]
        if poor_aspects:
            logger.warning(f"以下方面性能较差 (F1 < 0.3): {', '.join(poor_aspects)}")
            
        # 记录样本数量很少的方面
        rare_aspects = [a for a, m in aspect_metrics.items() if m.get("samples", 0) < 10]
        if rare_aspects:
            logger.warning(f"以下方面样本数量很少 (< 10): {', '.join(rare_aspects)}")
        
        # 添加详细指标到返回结果
        metrics["aspect_metrics"] = aspect_metrics
        
        logger.info(f"评估结果 - 准确率: {accuracy:.4f}, F1: {metrics['macro_f1']:.4f}")
        logger.info(f"各方面平均F1: {np.mean([m['f1'] for m in aspect_metrics.values() if 'f1' in m]):.4f}")
        
        return metrics
    
    def _plot_learning_curves(self):
        """
        绘制主动学习曲线
        """
        if not self.history["val_metrics"]:
            return
        
        # 准备数据
        rounds = list(range(len(self.history["val_metrics"])))
        labeled_ratios = [len(indices) / len(self.history["labeled_indices"][0]) * self.initial_samples_ratio 
                          for indices in self.history["labeled_indices"]]
        
        # 提取性能指标
        train_performance = [metrics.get(self.eval_metric, 0) for metrics in self.history["train_metrics"]]
        val_performance = [metrics.get(self.eval_metric, 0) for metrics in self.history["val_metrics"]]
        
        # 1. 绘制性能随轮次的变化
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, train_performance, "b-o", label="Training")
        plt.plot(rounds, val_performance, "r-o", label="Validation")
        plt.xlabel("Active Learning Round")
        plt.ylabel(self.eval_metric.capitalize())
        plt.title(f"{self.eval_metric.capitalize()} vs. Active Learning Round")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "learning_curve_rounds.png"))
        
        # 2. 绘制性能随标注样本比例的变化
        plt.figure(figsize=(10, 6))
        plt.plot(labeled_ratios, train_performance, "b-o", label="Training")
        plt.plot(labeled_ratios, val_performance, "r-o", label="Validation")
        plt.xlabel("Labeled Data Ratio")
        plt.ylabel(self.eval_metric.capitalize())
        plt.title(f"{self.eval_metric.capitalize()} vs. Labeled Data Ratio")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "learning_curve_ratio.png"))
        
        # 关闭图形，释放内存
        plt.close("all") 