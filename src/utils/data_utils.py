import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

# 情感极性映射
POLARITY_MAP = {
    -2: "不适用",  # 不适用/未提及
    -1: "负面",    # 负面情感
    0: "中性",     # 中性情感
    1: "正面"      # 正面情感
}

# 情感类别映射
ASPECT_CATEGORIES = {
    "Location": ["Transportation", "Downtown", "Easy_to_find"],
    "Service": ["Queue", "Hospitality", "Parking", "Timely"],
    "Price": ["Level", "Cost_effective", "Discount"],
    "Ambience": ["Decoration", "Noise", "Space", "Sanitary"],
    "Food": ["Portion", "Taste", "Appearance", "Recommend"]
}

# 扁平化的方面列表
FLAT_ASPECTS = []
for category, aspects in ASPECT_CATEGORIES.items():
    for aspect in aspects:
        FLAT_ASPECTS.append(f"{category}#{aspect}")

# 加载原始数据
def load_raw_data(file_path):
    """
    加载原始CSV数据文件
    """
    df = pd.read_csv(file_path)
    return df

# 预处理数据
def preprocess_data(df):
    """
    预处理数据:
    1. 过滤空评论
    2. 添加方面和极性的统计信息
    """
    # 过滤掉空评论
    df = df[df['review'].notna() & (df['review'] != '')].copy()
    
    # 统计每个样本中提到的方面数量
    mentioned_aspects = (df[FLAT_ASPECTS] != -2).sum(axis=1)
    df['mentioned_aspects_count'] = mentioned_aspects
    
    # 统计正面、负面和中性情感的数量
    df['positive_count'] = (df[FLAT_ASPECTS] == 1).sum(axis=1)
    df['neutral_count'] = (df[FLAT_ASPECTS] == 0).sum(axis=1)
    df['negative_count'] = (df[FLAT_ASPECTS] == -1).sum(axis=1)
    
    # 计算情感分布比例
    df['positive_ratio'] = df['positive_count'] / df['mentioned_aspects_count'].replace(0, 1)
    df['neutral_ratio'] = df['neutral_count'] / df['mentioned_aspects_count'].replace(0, 1)
    df['negative_ratio'] = df['negative_count'] / df['mentioned_aspects_count'].replace(0, 1)
    
    return df

# 提取未标注数据
def extract_unlabeled_data(df):
    """
    提取所有评论文本用于无监督预训练
    """
    return df['review'].tolist()

# 将数据分割为训练集和验证集
def split_data(df, test_size=0.2, random_state=42):
    """
    将数据分割为训练集和验证集
    """
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, val_df

# 创建用于MLM预训练的数据集
class MLMDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # 使用tokenizer处理文本
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_special_tokens_mask=True
        )
        
        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'special_tokens_mask': torch.tensor(encoding['special_tokens_mask'])
        }

# 创建用于方面级情感分析的数据集
class ACSADataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.aspect_columns = FLAT_ASPECTS
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['review']
        
        # 使用tokenizer处理文本
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 获取方面极性值
        aspect_labels = []
        for aspect in self.aspect_columns:
            # 将-2（不适用）映射为0（不预测），其他情感值映射为1-3
            # -1 -> 1, 0 -> 2, 1 -> 3
            if row[aspect] == -2:
                aspect_labels.append(0)  # 0表示不适用
            else:
                aspect_labels.append(row[aspect] + 2)  # 映射为1,2,3
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(aspect_labels)
        }

# 构建DataLoader
def build_data_loaders(train_df, val_df, tokenizer, batch_size=16, max_length=128):
    """
    构建用于方面级情感分析的数据加载器
    """
    train_dataset = ACSADataset(train_df, tokenizer, max_length)
    val_dataset = ACSADataset(val_df, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

# 构建用于MLM预训练的DataLoader
def build_mlm_data_loader(texts, tokenizer, batch_size=16, max_length=128):
    """
    构建用于MLM预训练的数据加载器
    """
    dataset = MLMDataset(texts, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 保存处理后的数据
def save_processed_data(df, file_path):
    """
    保存处理后的数据
    """
    df.to_csv(file_path, index=False)
    print(f"数据已保存至 {file_path}")

# 加载处理后的数据
def load_processed_data(file_path):
    """
    加载处理后的数据
    """
    return pd.read_csv(file_path)

# 准备种子数据集
def prepare_seed_data(df, seed_indices):
    """
    根据提供的索引准备种子数据集
    """
    return df.iloc[seed_indices].copy() 