import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import random
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from src.utils.data_utils import ACSADataset, build_data_loaders, FLAT_ASPECTS

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(
        self,
        model,
        tokenizer=None,
        output_dir="checkpoints",
        learning_rate=2e-5,
        weight_decay=0.01,
        batch_size=16,
        epochs=10,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        seed=42,
        device=None,
        use_consistency_loss=False,
        consistency_alpha=0.1
    ):
        # 设置随机种子
        self._set_seed(seed)
        
        # 设置设备
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 模型和tokenizer
        self.model = model
        self.model.to(self.device)
        self.tokenizer = tokenizer
        
        # 训练参数
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        
        # 一致性正则化（用于UDA）
        self.use_consistency_loss = use_consistency_loss
        self.consistency_alpha = consistency_alpha
        
        # TensorBoard日志
        self.tb_writer = SummaryWriter(os.path.join(output_dir, "runs"))
    
    def _set_seed(self, seed):
        """设置随机种子以确保结果可重现"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def train(self, train_df, val_df=None, augmented_df=None):
        """
        训练模型
        
        参数:
        train_df: 训练数据
        val_df: 验证数据
        augmented_df: 增强数据
        
        返回:
        训练指标
        """
        logger.info("准备数据加载器...")
        
        # 计算类别权重以处理不平衡
        has_labels = hasattr(train_df, 'labels') or any(col.startswith('label_') for col in train_df.columns)
        if has_labels:
            # 为每个方面收集标签
            aspect_labels = [[] for _ in range(len(FLAT_ASPECTS))]
            
            for i, row in train_df.iterrows():
                for aspect_idx in range(len(FLAT_ASPECTS)):
                    label_col = f'label_{aspect_idx}' if f'label_{aspect_idx}' in train_df.columns else 'labels'
                    if label_col == 'labels' and isinstance(row[label_col], (list, np.ndarray)):
                        # 处理当labels是一个列表或数组的情况
                        label = row[label_col][aspect_idx]
                    else:
                        label = row[label_col]
                    
                    if label != 0:  # 排除"不适用"标签
                        aspect_labels[aspect_idx].append(int(label))
            
            # 计算每个方面的类别权重
            all_aspect_weights = []
            valid_aspects = 0
            
            for aspect_idx, labels in enumerate(aspect_labels):
                if len(labels) > 0:
                    unique_labels = np.unique(labels)
                    if len(unique_labels) > 1:  # 至少有两个类别
                        try:
                            # 计算这个方面的类别权重
                            class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
                            
                            # 创建完整的权重张量（包括"不适用"类别）
                            weights = torch.zeros(max(unique_labels) + 1, dtype=torch.float)
                            for label, weight in zip(unique_labels, class_weights):
                                weights[label] = weight
                            
                            logger.info(f"方面 '{FLAT_ASPECTS[aspect_idx]}' 的类别权重: {dict(zip(unique_labels, class_weights))}")
                            all_aspect_weights.append(weights)
                            valid_aspects += 1
                        except Exception as e:
                            logger.warning(f"计算方面 '{FLAT_ASPECTS[aspect_idx]}' 的类别权重时出错: {str(e)}")
                            # 使用均匀权重作为回退
                            weights = torch.ones(4, dtype=torch.float)  # 假设有4个类别（0,1,2,3）
                            weights[0] = 0  # "不适用"类别的权重为0
                            all_aspect_weights.append(weights)
                    else:
                        logger.warning(f"方面 '{FLAT_ASPECTS[aspect_idx]}' 仅有一个类别，使用均匀权重")
                        # 使用均匀权重
                        weights = torch.ones(4, dtype=torch.float)
                        weights[0] = 0
                        all_aspect_weights.append(weights)
                else:
                    logger.warning(f"方面 '{FLAT_ASPECTS[aspect_idx]}' 没有有效标签，使用均匀权重")
                    # 使用均匀权重
                    weights = torch.ones(4, dtype=torch.float)
                    weights[0] = 0
                    all_aspect_weights.append(weights)
            
            # 检查增强型模型或基础模型，并设置相应的权重
            if valid_aspects > 0:
                if hasattr(self.model, 'set_aspect_weights') and callable(getattr(self.model, 'set_aspect_weights')):
                    # 如果模型支持方面特定权重，传递所有方面的权重
                    aspect_weights_tensor = torch.stack(all_aspect_weights).to(self.device)
                    self.model.set_aspect_weights(aspect_weights_tensor)
                    logger.info(f"为模型设置了方面特定的类别权重，形状: {aspect_weights_tensor.shape}")
                elif hasattr(self.model, 'set_class_weights') and callable(getattr(self.model, 'set_class_weights')):
                    # 如果模型只支持一组权重，计算所有方面的平均权重
                    # 先确保所有权重张量大小一致
                    max_size = max(w.size(0) for w in all_aspect_weights)
                    padded_weights = []
                    for w in all_aspect_weights:
                        if w.size(0) < max_size:
                            # 使用零填充扩展权重张量
                            padded = torch.zeros(max_size, dtype=torch.float)
                            padded[:w.size(0)] = w
                            padded_weights.append(padded)
                        else:
                            padded_weights.append(w)
                    
                    # 计算平均权重
                    avg_weights = torch.stack(padded_weights).mean(dim=0).to(self.device)
                    self.model.set_class_weights(avg_weights)
                    logger.info(f"使用所有方面的平均类别权重: {avg_weights}")
                else:
                    logger.warning("模型不支持设置类别权重或方面特定权重")
        
        # 创建数据集和数据加载器
        train_dataset = ACSADataset(train_df, self.tokenizer)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # 处理增强数据
        augmented_loader = None
        augmented_dataset = None
        if augmented_df is not None:
            logger.info(f"使用增强数据，大小: {len(augmented_df)}")
            augmented_dataset = ACSADataset(augmented_df, self.tokenizer)
            
            # 使用与训练数据相同的批次大小
            aug_batch_size = min(self.batch_size, len(augmented_dataset))
            
            # 确保批次大小至少为1
            aug_batch_size = max(1, aug_batch_size)
            
            if aug_batch_size != self.batch_size:
                logger.info(f"调整增强数据批次大小为: {aug_batch_size} (训练批次大小: {self.batch_size})")
            
            # 预先生成所有可能的增强批次，以便动态选择最匹配的批次
            if self.use_consistency_loss:
                logger.info("预处理增强数据批次以优化一致性训练...")
                all_aug_batches = []
                # 使用不同的批次大小生成多组批次
                batch_sizes = [self.batch_size]
                if self.batch_size > 4:
                    batch_sizes.extend([self.batch_size // 2, self.batch_size * 2])
                
                for bs in batch_sizes:
                    bs = min(bs, len(augmented_dataset))
                    temp_loader = DataLoader(augmented_dataset, batch_size=bs, shuffle=True)
                    for batch in temp_loader:
                        all_aug_batches.append({
                            'size': batch["input_ids"].size(0),
                            'batch': batch
                        })
                
                logger.info(f"生成了 {len(all_aug_batches)} 个预处理增强批次")
                # 按大小排序便于查找
                all_aug_batches.sort(key=lambda x: x['size'])
            else:
                # 如果不使用一致性损失，使用标准加载器
                augmented_loader = DataLoader(
                    augmented_dataset, 
                    batch_size=aug_batch_size, 
                    shuffle=True
                )
        
        # 验证数据
        if val_df is not None:
            val_dataset = ACSADataset(val_df, self.tokenizer)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.batch_size
            )
        else:
            val_loader = None
        
        # 训练准备
        # 设置优化器
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        # 学习率调度器
        total_steps = len(train_loader) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 训练循环
        logger.info("开始训练...")
        global_step = 0
        best_val_f1 = 0.0
        train_metrics = {}
        
        # 初始化增强数据迭代器
        aug_iter = iter(augmented_loader) if augmented_loader else None
        
        # 跟踪一致性损失跳过情况
        consistency_skip_count = 0
        consistency_apply_count = 0
        
        for epoch in range(self.epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_steps = 0
            all_preds = []
            all_labels = []
            
            # 使用tqdm创建进度条
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for batch in progress_bar:
                # 将数据移至设备
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # 清除梯度
                optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # 获取损失和logits
                loss = outputs[0] if isinstance(outputs, tuple) else outputs["loss"]
                logits = outputs[1] if isinstance(outputs, tuple) else outputs["logits"]
                
                # 如果使用一致性正则化且有增强数据
                if self.use_consistency_loss and (augmented_loader or 'all_aug_batches' in locals()):
                    try:
                        # 动态选择最匹配当前批次大小的增强批次
                        if 'all_aug_batches' in locals() and all_aug_batches:
                            current_batch_size = input_ids.size(0)
                            
                            # 寻找大小最接近的批次
                            best_batch_idx = None
                            min_size_diff = float('inf')
                            
                            for idx, item in enumerate(all_aug_batches):
                                size_diff = abs(item['size'] - current_batch_size)
                                if size_diff < min_size_diff:
                                    min_size_diff = size_diff
                                    best_batch_idx = idx
                                # 如果找到完全匹配的，立即使用
                                if size_diff == 0:
                                    break
                            
                            # 使用找到的最佳匹配批次
                            aug_batch = all_aug_batches[best_batch_idx]['batch']
                            
                            # 检查批次大小是否匹配
                            if aug_batch["input_ids"].size(0) != input_ids.size(0):
                                # 如果批次大小不匹配且差异显著，跳过一致性损失
                                if abs(aug_batch["input_ids"].size(0) - input_ids.size(0)) > current_batch_size * 0.3:
                                    consistency_skip_count += 1
                                    if consistency_skip_count % 20 == 0:
                                        logger.warning(f"增强批次大小差异太大: {aug_batch['input_ids'].size(0)} vs {input_ids.size(0)}，跳过一致性损失")
                                    continue
                                
                                # 确保增强批次与原始批次大小匹配
                                if aug_batch["input_ids"].size(0) > input_ids.size(0):
                                    # 随机截取子批次，而不是简单截断最后部分
                                    indices = torch.randperm(aug_batch["input_ids"].size(0))[:input_ids.size(0)]
                                    aug_batch = {k: v[indices] for k, v in aug_batch.items() if isinstance(v, torch.Tensor)}
                                else:
                                    # 批次过小，采用随机抽样而不是重复最后样本
                                    # 使用现有样本随机抽样填充到目标大小
                                    needed = input_ids.size(0) - aug_batch["input_ids"].size(0)
                                    
                                    # 如果需要填充的数量超过原批次的50%，跳过一致性损失
                                    if needed > aug_batch["input_ids"].size(0) * 0.5:
                                        consistency_skip_count += 1
                                        logger.debug(f"增强批次过小，需填充比例过高，跳过一致性损失")
                                        continue
                                    
                                    # 从现有样本中随机抽样
                                    indices = torch.randint(0, aug_batch["input_ids"].size(0), (needed,))
                                    for k, v in aug_batch.items():
                                        if isinstance(v, torch.Tensor) and v.dim() > 0:
                                            sampled = v[indices]
                                            aug_batch[k] = torch.cat([v, sampled], dim=0)
                        else:
                            # 使用常规迭代器
                            try:
                                aug_batch = next(aug_iter)
                            except (StopIteration, NameError):
                                # 重新初始化迭代器
                                aug_iter = iter(augmented_loader)
                                aug_batch = next(aug_iter)
                            
                            # 检查批次大小
                            if aug_batch["input_ids"].size(0) != input_ids.size(0):
                                # 如果批次大小差异过大，跳过一致性损失
                                diff_ratio = abs(aug_batch["input_ids"].size(0) - input_ids.size(0)) / input_ids.size(0)
                                if diff_ratio > 0.3:  # 30%以上的差异被视为显著
                                    consistency_skip_count += 1
                                    logger.debug(f"增强批次大小差异过大: {diff_ratio:.1%}，跳过一致性损失")
                                    continue
                                    
                                # 调整批次大小
                                if aug_batch["input_ids"].size(0) > input_ids.size(0):
                                    # 随机抽样而不是简单截断
                                    indices = torch.randperm(aug_batch["input_ids"].size(0))[:input_ids.size(0)]
                                    aug_batch = {k: v[indices] for k, v in aug_batch.items() if isinstance(v, torch.Tensor)}
                                else:
                                    # 批次过小，跳过一致性损失
                                    logger.debug("增强批次过小，跳过一致性损失")
                                    consistency_skip_count += 1
                                    continue
                        
                        # 获取原始和增强样本的预测
                        aug_input_ids = aug_batch["input_ids"].to(self.device)
                        aug_attention_mask = aug_batch["attention_mask"].to(self.device)
                        
                        # 获取增强样本的预测
                        with torch.no_grad():
                            aug_outputs = self.model(
                                input_ids=aug_input_ids,
                                attention_mask=aug_attention_mask
                            )
                            aug_logits = aug_outputs[1] if isinstance(aug_outputs, tuple) else aug_outputs["logits"]
                        
                        # 计算原始和增强样本之间的一致性损失
                        # 确保两个logits张量形状匹配
                        if logits.shape == aug_logits.shape:
                            consistency_loss = nn.MSELoss()(logits, aug_logits)
                            
                            # 结合标准损失和一致性损失
                            loss = loss + self.consistency_alpha * consistency_loss
                            
                            # 记录一致性损失
                            self.tb_writer.add_scalar("train/consistency_loss", consistency_loss.item(), global_step)
                            consistency_apply_count += 1
                        else:
                            logger.warning(f"Logits形状不匹配: {logits.shape} vs {aug_logits.shape}，跳过一致性损失")
                            consistency_skip_count += 1
                    except Exception as e:
                        logger.error(f"计算一致性损失时出错: {str(e)}")
                        # 错误不应该中断训练，所以我们继续使用原始损失
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # 优化器和调度器步骤
                optimizer.step()
                scheduler.step()
                
                # 更新进度信息
                train_loss += loss.item()
                train_steps += 1
                global_step += 1
                
                # 收集预测和标签
                preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
                label_ids = labels.detach().cpu().numpy()
                
                # 只考虑非零标签（非"不适用"）
                for i in range(preds.shape[0]):
                    for j in range(preds.shape[1]):
                        if label_ids[i][j] != 0:  # 0 = 不适用
                            all_preds.append(preds[i][j])
                            all_labels.append(label_ids[i][j])
                
                # 更新进度条
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # 记录到TensorBoard
                self.tb_writer.add_scalar("train/loss", loss.item(), global_step)
            
            # 计算训练指标
            avg_train_loss = train_loss / train_steps
            
            # 计算训练集的F1分数等
            train_accuracy = accuracy_score(all_labels, all_preds)
            train_precision = precision_score(all_labels, all_preds, average="macro")
            train_recall = recall_score(all_labels, all_preds, average="macro")
            train_f1 = f1_score(all_labels, all_preds, average="macro")
            
            # 记录一致性损失应用情况
            total_consistency = consistency_apply_count + consistency_skip_count
            if total_consistency > 0:
                logger.info(f"一致性损失统计 - 应用: {consistency_apply_count}/{total_consistency} ({consistency_apply_count/total_consistency:.1%}), "
                           f"跳过: {consistency_skip_count}/{total_consistency} ({consistency_skip_count/total_consistency:.1%})")
            
            # 重置一致性计数器
            consistency_skip_count = 0
            consistency_apply_count = 0
            
            logger.info(f"Epoch {epoch+1}/{self.epochs} - "
                       f"训练损失: {avg_train_loss:.4f}, "
                       f"准确率: {train_accuracy:.4f}, "
                       f"F1: {train_f1:.4f}")
            
            # 记录到TensorBoard
            self.tb_writer.add_scalar("train/accuracy", train_accuracy, epoch)
            self.tb_writer.add_scalar("train/precision", train_precision, epoch)
            self.tb_writer.add_scalar("train/recall", train_recall, epoch)
            self.tb_writer.add_scalar("train/f1", train_f1, epoch)
            
            # 保存训练指标
            train_metrics = {
                "loss": avg_train_loss,
                "accuracy": train_accuracy,
                "precision": train_precision,
                "recall": train_recall,
                "macro_f1": train_f1
            }
            
            # 验证阶段
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                
                logger.info(f"Epoch {epoch+1}/{self.epochs} - "
                           f"验证损失: {val_metrics['loss']:.4f}, "
                           f"准确率: {val_metrics['accuracy']:.4f}, "
                           f"F1: {val_metrics['macro_f1']:.4f}")
                
                # 记录到TensorBoard
                for key, value in val_metrics.items():
                    self.tb_writer.add_scalar(f"val/{key}", value, epoch)
                
                # 保存最佳模型
                if val_metrics["macro_f1"] > best_val_f1:
                    best_val_f1 = val_metrics["macro_f1"]
                    self.save_model(os.path.join(self.output_dir, "best_model"))
            
            # 保存checkpoint
            self.save_model(os.path.join(self.output_dir, f"checkpoint-{epoch}"))
        
        # 保存最终模型
        self.save_model(os.path.join(self.output_dir, "final_model"))
        logger.info("训练完成")
        
        return train_metrics
    
    def evaluate(self, val_loader):
        """
        在验证集上评估模型
        
        参数:
        val_loader: 验证数据加载器
        
        返回:
        验证指标字典
        """
        self.model.eval()
        val_loss = 0.0
        val_steps = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                # 将数据移至设备
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # 获取损失和logits
                loss = outputs[0] if isinstance(outputs, tuple) else outputs["loss"]
                logits = outputs[1] if isinstance(outputs, tuple) else outputs["logits"]
                
                val_loss += loss.item()
                val_steps += 1
                
                # 收集预测和标签
                preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
                label_ids = labels.detach().cpu().numpy()
                
                # 只考虑非零标签（非"不适用"）
                for i in range(preds.shape[0]):
                    for j in range(preds.shape[1]):
                        if label_ids[i][j] != 0:  # 0 = 不适用
                            all_preds.append(preds[i][j])
                            all_labels.append(label_ids[i][j])
        
        # 计算验证指标
        avg_val_loss = val_loss / val_steps
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds, average="macro")
        val_recall = recall_score(all_labels, all_preds, average="macro")
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        
        return {
            "loss": avg_val_loss,
            "accuracy": val_accuracy,
            "precision": val_precision,
            "recall": val_recall,
            "macro_f1": val_f1
        }
    
    def save_model(self, output_dir):
        """
        保存模型和tokenizer
        
        参数:
        output_dir: 输出目录
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        logger.info(f"保存模型到 {output_dir}")
        
        # 保存模型
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(output_dir)
        
        # 保存tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)
    
    def load_model(self, model_path):
        """
        从路径加载模型
        
        参数:
        model_path: 模型路径
        """
        logger.info(f"从 {model_path} 加载模型")
        self.model.from_pretrained(model_path)
        self.model.to(self.device) 