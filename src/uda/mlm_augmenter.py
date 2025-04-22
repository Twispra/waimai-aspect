import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM
import random
import re
import os
import pandas as pd

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class MLMAugmenter:
    def __init__(
        self,
        model_name="bert-base-chinese",
        pretrained_model_path=None,
        mask_ratio=0.15,
        max_length=128,
        max_predictions=20,
        temperature=1.0,
        num_augmentations=3,
        retain_aspect_tokens=True,
        seed=42,
        device=None
    ):
        # 设置设备
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 设置随机种子
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # 加载tokenizer和model
        logger.info(f"加载MLM模型: {pretrained_model_path if pretrained_model_path else model_name}")
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
            self.model = BertForMaskedLM.from_pretrained(pretrained_model_path)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForMaskedLM.from_pretrained(model_name)
        
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        
        # 参数设置
        self.mask_ratio = mask_ratio
        self.max_length = max_length
        self.max_predictions = max_predictions
        self.temperature = temperature
        self.num_augmentations = num_augmentations
        self.retain_aspect_tokens = retain_aspect_tokens
    
    def _identify_aspect_tokens(self, text):
        """
        识别文本中的方面词，避免在数据增强过程中修改这些词
        
        参数:
        text: 原始文本
        
        返回:
        方面词在文本中的位置列表，以tokenizer的token index表示
        """
        from src.utils.data_utils import ASPECT_CATEGORIES
        
        # 从配置中提取所有可能的方面词
        aspect_words = []
        for category, aspects in ASPECT_CATEGORIES.items():
            aspect_words.append(category)
            aspect_words.extend(aspects)
            
        # 直接使用分词器将文本分词
        tokens = self.tokenizer.tokenize(text)
        
        # 将分词结果转换为小写，方便匹配
        tokens_lower = [t.lower() for t in tokens]
        
        # 识别方面词位置
        aspect_token_positions = []
        for aspect in aspect_words:
            # 将方面词分词
            aspect_tokens = self.tokenizer.tokenize(aspect)
            aspect_tokens_lower = [t.lower() for t in aspect_tokens]
            
            # 特殊情况：空分词
            if not aspect_tokens:
                continue
                
            # 在文本分词结果中查找方面词的连续子序列
            for i in range(len(tokens_lower) - len(aspect_tokens_lower) + 1):
                # 检查是否是连续匹配
                is_match = True
                for j in range(len(aspect_tokens_lower)):
                    if not (tokens_lower[i+j] == aspect_tokens_lower[j] or 
                            aspect_tokens_lower[j] in tokens_lower[i+j] or 
                            tokens_lower[i+j] in aspect_tokens_lower[j]):
                        is_match = False
                        break
                
                # 如果匹配，记录所有匹配位置
                if is_match:
                    for j in range(len(aspect_tokens_lower)):
                        # 添加+1是为了考虑[CLS]令牌
                        aspect_token_positions.append(i + j + 1)
        
        # 删除重复项并升序排列
        aspect_token_positions = sorted(list(set(aspect_token_positions)))
        
        # 确保不超出模型最大长度
        aspect_token_positions = [pos for pos in aspect_token_positions if pos < self.max_length - 1]
        
        if not aspect_token_positions:
            logger.debug(f"未识别出任何方面词: {text[:50]}...")
            
        return aspect_token_positions
    
    def _mask_tokens(self, inputs, aspect_token_positions=None):
        """
        随机掩盖输入中的一些token以进行MLM预测
        
        参数:
        inputs: 模型输入的token ids
        aspect_token_positions: 方面词token的位置列表，这些token不会被掩盖
        
        返回:
        掩盖后的输入和掩盖位置
        """
        # 确保aspect_token_positions是一个列表
        if aspect_token_positions is not None and not isinstance(aspect_token_positions, list):
            logger.warning(f"aspect_token_positions不是列表类型: {type(aspect_token_positions)}")
            aspect_token_positions = list(aspect_token_positions) if hasattr(aspect_token_positions, '__iter__') else []
            
        # 复制输入以避免修改原始数据
        inputs = inputs.clone()
        
        # 创建掩码概率矩阵
        probability_matrix = torch.full(inputs.shape, self.mask_ratio)
        
        # 将特殊tokens的掩码概率设为0
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in inputs.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        # 将方面词token的掩码概率设为0（如果提供）
        if aspect_token_positions and self.retain_aspect_tokens:
            for pos in aspect_token_positions:
                if pos < inputs.size(0):
                    probability_matrix[pos] = 0.0
        
        # 确定要掩盖的token
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 限制掩盖的token数量
        if self.max_predictions > 0:
            indices_count = min(masked_indices.sum().item(), self.max_predictions)
            if indices_count < masked_indices.sum().item():
                # 随机选择max_predictions个token进行掩盖
                excess_masks = masked_indices.sum() - indices_count
                masked_indices = masked_indices & ~torch.multinomial(
                    masked_indices.float(), 
                    excess_masks, 
                    replacement=False
                )
        
        # 确保至少掩盖一个token
        if masked_indices.sum() == 0:
            # 随机选择一个非特殊token进行掩盖
            non_special_indices = [
                i for i, m in enumerate(special_tokens_mask[0])
                if m == 0 and (not aspect_token_positions or i not in aspect_token_positions)
            ]
            if non_special_indices:
                masked_index = random.choice(non_special_indices)
                masked_indices[masked_index] = True
        
        # 记录原始标签
        labels = inputs.clone()
        
        # 创建掩盖后的输入
        inputs[masked_indices] = self.tokenizer.mask_token_id
        
        return inputs, masked_indices, labels
    
    def generate_augmentations(self, texts, labels=None):
        """
        使用MLM生成文本增强样本
        
        参数:
        texts: 要增强的文本列表
        labels: 原始文本对应的标签列表（可选）
        
        返回:
        增强后的文本列表和对应的标签列表（如果提供了标签）
        """
        augmented_texts = []
        augmented_labels = [] if labels is not None else None
        
        total_augmentations = len(texts) * self.num_augmentations
        logger.info(f"正在生成 {total_augmentations} 个增强样本...")
        
        # 更详细的进度条
        progress_bar = tqdm(texts, desc="生成增强样本")
        
        # 记录错误统计
        error_count = 0
        success_count = 0
        
        for i, text in enumerate(progress_bar):
            try:
                # 更新进度条信息
                progress_desc = f"生成样本 [{i+1}/{len(texts)}] - 成功:{success_count} 错误:{error_count}"
                progress_bar.set_description(progress_desc)
                
                # 获取方面词位置 - 使用更可靠的方法
                aspect_token_positions = self._identify_aspect_tokens(text) if self.retain_aspect_tokens else None
                
                # 生成多个增强版本
                text_augmentations = []
                for j in range(self.num_augmentations):
                    try:
                        augmented_text = self._augment_text(text, aspect_token_positions)
                        text_augmentations.append(augmented_text)
                    except Exception as e:
                        logger.warning(f"生成第{j+1}个增强样本时出错: {str(e)}")
                        # 出错时使用原始文本代替
                        text_augmentations.append(text)
                
                augmented_texts.extend(text_augmentations)
                success_count += 1
                
                # 如果提供了标签，为每个增强样本复制原始标签
                if labels is not None:
                    augmented_labels.extend([labels[i]] * self.num_augmentations)
                    
            except Exception as e:
                error_count += 1
                logger.error(f"处理文本 #{i+1} 时出错: {str(e)}")
                # 出错时使用原始文本作为增强样本
                for _ in range(self.num_augmentations):
                    augmented_texts.append(text)
                
                # 如果提供了标签，也需要添加
                if labels is not None:
                    augmented_labels.extend([labels[i]] * self.num_augmentations)
                
                # 如果太多错误，则提前退出
                if error_count > min(100, len(texts) * 0.1):  # 错误超过10%或100个时
                    logger.warning(f"错误过多，提前结束增强过程。成功：{success_count}，失败：{error_count}")
                    break
            
            # 每处理100个样本清理一次CUDA缓存
            if i % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info(f"数据增强完成。共生成 {len(augmented_texts)} 个样本，成功率: {success_count/len(texts)*100:.1f}%")
        
        return augmented_texts, augmented_labels
    
    def _augment_text(self, text, aspect_token_positions=None):
        """
        使用MLM增强单个文本
        
        参数:
        text: 要增强的原始文本
        aspect_token_positions: 方面词token的位置列表
        
        返回:
        增强后的文本
        """
        # 安全检查
        if not text or len(text.strip()) == 0:
            logger.warning("收到空文本，返回原文")
            return text
            
        try:
            # 编码输入文本
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            input_ids = encoding.input_ids.to(self.device)
            attention_mask = encoding.attention_mask.to(self.device)
            
            # 掩盖部分token
            masked_input, masked_indices, original_labels = self._mask_tokens(input_ids[0], aspect_token_positions)
            
            # 检查是否有被掩盖的token
            if masked_indices.sum() == 0:
                logger.warning(f"没有token被掩盖，返回原文: {text[:30]}...")
                return text
                
            masked_input = masked_input.unsqueeze(0)
            
            # 预测掩盖的token
            with torch.no_grad():
                outputs = self.model(
                    input_ids=masked_input,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
            
            # 应用温度
            if self.temperature != 1.0:
                logits = logits / self.temperature
            
            # 为掩盖的位置采样新token
            probs = F.softmax(logits, dim=-1)
            masked_token_probs = probs[0, masked_indices, :]
            
            # 对每个掩盖位置采样一个token
            sampled_tokens = torch.multinomial(masked_token_probs, 1).squeeze(-1)
            
            # 将采样的token替换到输入中
            augmented_input = input_ids.clone()
            augmented_input[0, masked_indices] = sampled_tokens
            
            # 解码并返回增强后的文本
            augmented_text = self.tokenizer.decode(augmented_input[0].tolist(), skip_special_tokens=True)
            
            # 确保增强后的文本不为空
            if not augmented_text or len(augmented_text.strip()) == 0:
                logger.warning("增强后文本为空，返回原文")
                return text
                
            # 如果增强后的文本变化太小，也使用原文
            if len(augmented_text) < 0.5 * len(text):
                logger.warning(f"增强后文本太短 ({len(augmented_text)} vs {len(text)})，返回原文")
                return text
                
            return augmented_text
            
        except Exception as e:
            logger.error(f"_augment_text 出错: {str(e)}")
            # 出错时返回原始文本
            return text
    
    def augment_dataset(self, df, text_column='review', augmentation_ratio=1.0):
        """
        增强整个数据集，生成伪标签
        
        参数:
        df: 包含文本的DataFrame
        text_column: 文本列的名称
        augmentation_ratio: 生成的增强样本占原始样本的比例
        
        返回:
        包含原始样本和增强样本的新DataFrame
        """
        try:
            # 1. 从DataFrame中提取文本和标签
            texts = df[text_column].tolist()
            
            # 2. 确定要增强的样本数
            n_samples = min(int(len(texts) * augmentation_ratio), len(texts))
            logger.info(f"将增强 {n_samples} 个样本 (总共 {len(texts)} 个)")
            
            indices = np.random.choice(len(texts), n_samples, replace=False)
            selected_texts = [texts[i] for i in indices]
            
            # 3. 生成增强样本，捕获整体异常
            try:
                augmented_texts, _ = self.generate_augmentations(selected_texts)
                
                # 检查生成的样本数量是否符合预期
                expected_count = n_samples * self.num_augmentations
                if len(augmented_texts) < expected_count:
                    logger.warning(f"生成的增强样本数量少于预期: {len(augmented_texts)} vs {expected_count}")
                    
                    # 如果差距太大，使用原始文本填充
                    if len(augmented_texts) < expected_count * 0.5:
                        logger.warning("增强样本数量不足预期的50%，使用原始文本填充")
                        while len(augmented_texts) < expected_count:
                            idx = len(augmented_texts) % len(selected_texts)
                            augmented_texts.append(selected_texts[idx])
                
            except Exception as e:
                logger.error(f"生成增强样本时发生错误: {str(e)}")
                # 出错时使用原始文本作为增强样本
                augmented_texts = []
                for text in selected_texts:
                    for _ in range(self.num_augmentations):
                        augmented_texts.append(text)
            
            # 4. 创建新的DataFrame包含原始样本和增强样本
            augmented_df = df.copy()
            
            # 5. 将增强样本添加到DataFrame中
            added_count = 0
            for i, text in enumerate(augmented_texts):
                try:
                    # 获取原始样本的索引
                    original_idx = indices[i // self.num_augmentations]
                    
                    # 创建新行，复制原始样本的所有字段
                    new_row = df.iloc[original_idx].copy()
                    
                    # 更新文本字段
                    new_row[text_column] = text
                    
                    # 添加新行
                    augmented_df = pd.concat([augmented_df, pd.DataFrame([new_row])], ignore_index=True)
                    added_count += 1
                except Exception as e:
                    logger.error(f"添加增强样本 #{i} 时出错: {str(e)}")
            
            logger.info(f"成功添加 {added_count} 个增强样本到数据集")
            return augmented_df
            
        except Exception as e:
            logger.error(f"增强数据集时发生严重错误: {str(e)}")
            logger.info("返回原始数据集，不进行增强")
            return df 