import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from transformers import BertTokenizer
import os

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class UncertaintySampler:
    def __init__(
        self,
        model,
        tokenizer=None,
        max_length=128,
        batch_size=32,
        device=None,
        alpha=0.5,  # 不确定性和多样性的权重平衡参数
    ):
        # 设置设备
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 模型参数
        self.model = model
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.alpha = alpha
    
    def compute_uncertainty(self, df, unlabeled_indices):
        """
        计算未标注样本的预测不确定性分数
        
        参数:
        df: 包含'review'列的DataFrame
        unlabeled_indices: 未标注样本的索引列表
        
        返回:
        不确定性分数列表，与unlabeled_indices顺序对应
        """
        logger.info("计算预测不确定性...")
        uncertainty_scores = []
        unlabeled_texts = df.iloc[unlabeled_indices]['review'].tolist()
        
        # 分批处理文本
        for i in tqdm(range(0, len(unlabeled_texts), self.batch_size)):
            batch_texts = unlabeled_texts[i:i+self.batch_size]
            
            # 使用tokenizer处理文本
            encoding = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # 将编码移至设备
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            
            with torch.no_grad():
                # 获取模型输出
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
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
                
                # 计算每个方面的预测不确定性
                # 使用预测概率的熵作为不确定性度量
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                
                # 对所有方面的不确定性取平均
                batch_uncertainty = entropy.mean(dim=1).cpu().numpy()
                uncertainty_scores.extend(batch_uncertainty.tolist())
        
        return uncertainty_scores
    
    def select_samples(self, df, labeled_indices, n_samples, diversity_selector=None):
        """
        基于不确定性和多样性选择最有价值的样本
        
        参数:
        df: 包含'review'列的DataFrame
        labeled_indices: 已标注样本的索引列表
        n_samples: 要选择的样本数量
        diversity_selector: 用于计算多样性的选择器（如KMeansSeedSelector）
        
        返回:
        选择的样本索引列表
        """
        # 1. 获取未标注样本的索引
        all_indices = set(range(len(df)))
        labeled_indices_set = set(labeled_indices)
        unlabeled_indices = list(all_indices - labeled_indices_set)
        
        # 如果没有未标注的样本或不需要选择新样本，直接返回空列表
        if not unlabeled_indices or n_samples <= 0:
            return []
        
        # 2. 计算不确定性分数
        uncertainty_scores = self.compute_uncertainty(df, unlabeled_indices)
        
        # 检查是否有方面优先级配置
        try:
            import yaml
            with open('configs/pipeline.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            has_aspect_priority = False
            aspect_priority_multiplier = 1.0
            high_priority_aspects = []
            
            if 'aspect_weights' in config and 'high_priority_aspects' in config['aspect_weights']:
                high_priority_aspects = config['aspect_weights']['high_priority_aspects']
                if 'sampling_weight_multiplier' in config['aspect_weights']:
                    aspect_priority_multiplier = config['aspect_weights']['sampling_weight_multiplier']
                has_aspect_priority = True
                logger.info(f"应用方面优先级，优先选择方面: {high_priority_aspects}")
                logger.info(f"优先方面的权重倍数: {aspect_priority_multiplier}")
        except Exception as e:
            logger.warning(f"读取方面优先级配置时出错: {str(e)}")
            has_aspect_priority = False
        
        # 3. 如果提供了多样性选择器，考虑样本多样性
        if diversity_selector:
            logger.info("计算多样性分数...")
            diversity_indices = diversity_selector.select_diverse_samples(
                df, labeled_indices, min(n_samples * 3, len(unlabeled_indices))
            )
            
            # 创建diversity_mask
            diversity_mask = np.zeros(len(unlabeled_indices))
            for idx in range(len(unlabeled_indices)):
                if unlabeled_indices[idx] in diversity_indices:
                    diversity_mask[idx] = 1
            
            # 结合不确定性和多样性
            # 归一化不确定性分数
            max_score = max(uncertainty_scores)
            min_score = min(uncertainty_scores)
            norm_uncertainty = [(s - min_score) / (max_score - min_score + 1e-10) for s in uncertainty_scores]
            
            # 应用方面优先级（如果有）
            if has_aspect_priority:
                # 预测每个未标注样本可能包含的方面
                from src.utils.data_utils import FLAT_ASPECTS
                
                # 创建方面优先级掩码
                aspect_priority_mask = np.ones(len(unlabeled_indices))
                
                # 从配置中读取方面关键词
                aspect_keywords = {}
                try:
                    if 'aspect_keywords' in config['aspect_weights']:
                        aspect_keywords = config['aspect_weights']['aspect_keywords']
                        logger.info(f"使用配置文件中定义的方面关键词")
                    else:
                        # 使用默认关键词集合
                        aspect_keywords = {
                            "Location#Easy_to_find": ["位置", "地方", "找到", "方便", "地址", "地理位置"],
                            "Service#Queue": ["排队", "等待", "等候", "人多", "拥挤", "队伍"],
                            "Service#Hospitality": ["服务", "态度", "热情", "客气", "服务员", "招待"],
                            "Service#Timely": ["及时", "速度", "快", "慢", "送餐", "配送", "等待时间"],
                            "Price#Discount": ["优惠", "折扣", "促销", "便宜", "实惠", "划算", "特价"],
                            "Ambience#Noise": ["噪音", "嘈杂", "安静", "吵", "环境", "声音"],
                            "Food#Taste": ["味道", "口感", "美味", "好吃", "难吃", "鲜美", "可口"],
                            "Food#Recommend": ["推荐", "建议", "值得", "再来", "介绍", "喜欢"]
                        }
                        logger.info(f"使用默认方面关键词")
                except Exception as e:
                    logger.warning(f"读取方面关键词时出错: {str(e)}，使用默认关键词")
                    aspect_keywords = {
                        "Location#Easy_to_find": ["位置", "地方", "找到", "方便", "地址", "地理位置"],
                        "Service#Queue": ["排队", "等待", "等候", "人多", "拥挤", "队伍"],
                        "Service#Hospitality": ["服务", "态度", "热情", "客气", "服务员", "招待"],
                        "Service#Timely": ["及时", "速度", "快", "慢", "送餐", "配送", "等待时间"],
                        "Price#Discount": ["优惠", "折扣", "促销", "便宜", "实惠", "划算", "特价"],
                        "Ambience#Noise": ["噪音", "嘈杂", "安静", "吵", "环境", "声音"],
                        "Food#Taste": ["味道", "口感", "美味", "好吃", "难吃", "鲜美", "可口"],
                        "Food#Recommend": ["推荐", "建议", "值得", "再来", "介绍", "喜欢"]
                    }
                
                # 跟踪每个方面匹配的样本数
                aspect_match_count = {aspect: 0 for aspect in high_priority_aspects}
                
                for i, idx in enumerate(unlabeled_indices):
                    text = df.iloc[idx]['review']
                    
                    # 检查文本是否包含高优先级方面的关键词
                    for aspect in high_priority_aspects:
                        if aspect in aspect_keywords:
                            matched = False
                            # 使用配置的关键词集合
                            for keyword in aspect_keywords[aspect]:
                                if keyword in text:
                                    aspect_priority_mask[i] = aspect_priority_multiplier
                                    matched = True
                                    aspect_match_count[aspect] += 1
                                    break
                            if matched:
                                break
                
                # 详细记录每个方面匹配的样本数
                match_info = ", ".join([f"{aspect.split('#')[1]}: {count}" for aspect, count in aspect_match_count.items()])
                logger.info(f"方面关键词匹配统计: {match_info}")
                logger.info(f"应用方面优先级后，{sum(aspect_priority_mask > 1)} 个样本获得优先级提升 ({sum(aspect_priority_mask > 1)/len(unlabeled_indices):.1%})")
                
                # 结合分数 = α * 不确定性 * 方面优先级 + (1-α) * 多样性
                combined_scores = [self.alpha * u * p + (1 - self.alpha) * d 
                                 for u, d, p in zip(norm_uncertainty, diversity_mask, aspect_priority_mask)]
            else:
                # 结合分数 = α * 不确定性 + (1-α) * 多样性
                combined_scores = [self.alpha * u + (1 - self.alpha) * d for u, d in zip(norm_uncertainty, diversity_mask)]
            
            # 选择分数最高的样本
            top_indices = np.argsort(combined_scores)[-n_samples:]
            selected_indices = [unlabeled_indices[i] for i in top_indices]
        else:
            # 仅基于不确定性选择
            top_indices = np.argsort(uncertainty_scores)[-n_samples:]
            selected_indices = [unlabeled_indices[i] for i in top_indices]
        
        return selected_indices 