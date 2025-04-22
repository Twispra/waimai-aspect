#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import logging
import pandas as pd
import yaml
import random
import numpy as np
from transformers import BertTokenizer
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 导入自定义模块
from src.utils.data_utils import (
    load_raw_data,
    preprocess_data,
    extract_unlabeled_data,
    split_data,
    save_processed_data,
    load_processed_data,
    prepare_seed_data,
    FLAT_ASPECTS
)
from src.pretraining.mlm_trainer import MLMTrainer
from src.clustering.kmeans_seed import KMeansSeedSelector
from src.active_learning.uncertainty_sampler import UncertaintySampler
from src.active_learning.active_learner import ActiveLearner
from src.uda.mlm_augmenter import MLMAugmenter
from src.models.acsa_model import build_model
from src.models.model_trainer import ModelTrainer

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 确保数值参数为正确的类型
    try:
        # 处理pretraining部分
        if 'pretraining' in config:
            for key in ['learning_rate', 'weight_decay', 'mlm_probability', 'warmup_ratio']:
                if key in config['pretraining']:
                    config['pretraining'][key] = float(config['pretraining'][key])
                    logger.info(f"转换 pretraining.{key} 为浮点数: {config['pretraining'][key]}")
            for key in ['batch_size', 'epochs', 'max_length']:
                if key in config['pretraining']:
                    config['pretraining'][key] = int(config['pretraining'][key])
                    logger.info(f"转换 pretraining.{key} 为整数: {config['pretraining'][key]}")
        
        # 处理clustering部分
        if 'clustering' in config:
            for key in ['seed_ratio']:
                if key in config['clustering']:
                    config['clustering'][key] = float(config['clustering'][key])
                    logger.info(f"转换 clustering.{key} 为浮点数: {config['clustering'][key]}")
            for key in ['batch_size', 'max_length']:
                if key in config['clustering']:
                    config['clustering'][key] = int(config['clustering'][key])
                    logger.info(f"转换 clustering.{key} 为整数: {config['clustering'][key]}")
        
        # 处理active_learning部分
        if 'active_learning' in config:
            for key in ['learning_rate', 'weight_decay', 'warmup_ratio', 'initial_samples_ratio', 
                       'budget_per_round', 'alpha', 'consistency_alpha', 'augmentation_ratio']:
                if key in config['active_learning']:
                    config['active_learning'][key] = float(config['active_learning'][key])
                    logger.info(f"转换 active_learning.{key} 为浮点数: {config['active_learning'][key]}")
            for key in ['batch_size', 'epochs', 'max_length', 'max_rounds']:
                if key in config['active_learning']:
                    config['active_learning'][key] = int(config['active_learning'][key])
                    logger.info(f"转换 active_learning.{key} 为整数: {config['active_learning'][key]}")
        
        # 处理uda部分
        if 'uda' in config:
            for key in ['mask_ratio', 'temperature']:
                if key in config['uda']:
                    config['uda'][key] = float(config['uda'][key])
                    logger.info(f"转换 uda.{key} 为浮点数: {config['uda'][key]}")
            for key in ['max_length', 'max_predictions', 'num_augmentations']:
                if key in config['uda']:
                    config['uda'][key] = int(config['uda'][key])
                    logger.info(f"转换 uda.{key} 为整数: {config['uda'][key]}")
        
        # 确保全局seed是整数
        if 'seed' in config:
            config['seed'] = int(config['seed'])
            logger.info(f"转换 seed 为整数: {config['seed']}")
    
    except (ValueError, TypeError) as e:
        logger.error(f"配置文件中的参数类型转换错误: {e}")
        logger.error("请检查配置文件中的数值参数是否正确")
        raise
        
    return config

def model_builder_func(train_df, val_df=None, output_dir=None, **kwargs):
    """
    构建和训练模型的函数
    这个函数被主动学习循环调用
    
    参数:
    train_df: 训练数据
    val_df: 验证数据
    output_dir: 输出目录
    
    返回:
    训练好的模型和训练指标
    """
    # 从kwargs中获取模型和训练参数
    pretrained_model_path = kwargs.get('pretrained_model_path')
    model_type = kwargs.get('model_type', 'basic')
    augmented_df = kwargs.get('augmented_df')  # 获取增强数据
    
    # 构建tokenizer - 增加错误处理
    try:
        logger.info(f"尝试从路径加载tokenizer: {pretrained_model_path}")
        
        # 首先尝试在本地路径中加载tokenizer
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            # 检查vocab.txt是否存在
            if os.path.exists(os.path.join(pretrained_model_path, "vocab.txt")):
                logger.info(f"在 {pretrained_model_path} 找到vocab.txt，使用本地tokenizer")
                tokenizer = BertTokenizer.from_pretrained(pretrained_model_path, local_files_only=True)
            else:
                # 如果vocab.txt不存在，尝试在其他可能的子目录中查找
                possible_locations = [
                    os.path.join(pretrained_model_path, "tokenizer"),
                    os.path.join(os.path.dirname(pretrained_model_path), "tokenizer"),
                    os.path.dirname(pretrained_model_path)
                ]
                
                for location in possible_locations:
                    if os.path.exists(os.path.join(location, "vocab.txt")):
                        logger.info(f"在 {location} 找到vocab.txt，使用该tokenizer")
                        tokenizer = BertTokenizer.from_pretrained(location, local_files_only=True)
                        break
                else:
                    raise FileNotFoundError(f"在相关目录中未找到vocab.txt: {possible_locations}")
        else:
            # 如果路径不存在，使用备用路径
            fallback_path = os.path.join("data", "pretrained", "waimai-bert", "best_model")
            logger.info(f"指定的模型路径不存在，尝试备用路径: {fallback_path}")
            
            if os.path.exists(fallback_path) and os.path.exists(os.path.join(fallback_path, "vocab.txt")):
                logger.info(f"在备用路径找到vocab.txt，使用本地tokenizer")
                tokenizer = BertTokenizer.from_pretrained(fallback_path, local_files_only=True)
            else:
                # 如果仍然失败，尝试创建一个新的中文分词器
                logger.warning("无法找到本地tokenizer，创建基本的中文分词器")
                
                # 为中文定义基本词表（有限的应急措施）
                from transformers import BertTokenizerFast
                
                # 获取当前目录以建立临时词表
                temp_dir = os.path.join(os.getcwd(), "temp_tokenizer")
                os.makedirs(temp_dir, exist_ok=True)
                
                # 创建最小词表文件
                with open(os.path.join(temp_dir, "vocab.txt"), "w", encoding="utf-8") as f:
                    # 写入最基本的特殊token和一些中文字符
                    f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n")
                    # 添加一些基本的中文字符和标点
                    for i in range(0x4e00, 0x4e50):  # 一小部分中文字符
                        f.write(f"{chr(i)}\n")
                    f.write("。\n，\n！\n？\n")
                
                logger.info(f"创建了临时词表: {temp_dir}")
                tokenizer = BertTokenizer.from_pretrained(temp_dir, local_files_only=True)
                
    except Exception as e:
        logger.error(f"加载tokenizer时出错: {str(e)}")
        logger.error("创建备用词表分词器")
        
        # 创建基本词表作为备用
        from collections import OrderedDict
        basic_tokens = OrderedDict([
            ("[PAD]", 0),
            ("[UNK]", 1),
            ("[CLS]", 2),
            ("[SEP]", 3),
            ("[MASK]", 4)
        ])
        
        # 创建极简tokenizer
        class MinimalTokenizer:
            def __init__(self):
                self.vocab = basic_tokens
                self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
            
            def tokenize(self, text):
                # 简单按字符分词
                return list(text)
            
            def convert_tokens_to_ids(self, tokens):
                # 转换为id，未知字符映射为[UNK]
                return [self.vocab.get(token, 1) for token in tokens]
            
            def encode_plus(self, text, max_length=128, padding="max_length", truncation=True, return_tensors=None):
                tokens = self.tokenize(text)
                if truncation and len(tokens) > max_length - 2:  # 考虑[CLS]和[SEP]
                    tokens = tokens[:max_length-2]
                
                # 添加特殊token
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                input_ids = self.convert_tokens_to_ids(tokens)
                
                # padding
                if padding == "max_length" and len(input_ids) < max_length:
                    input_ids = input_ids + [0] * (max_length - len(input_ids))
                
                # 创建attention_mask
                attention_mask = [1] * len(input_ids)
                
                result = {"input_ids": input_ids, "attention_mask": attention_mask}
                
                # 转换为tensor
                if return_tensors == "pt":
                    import torch
                    result = {k: torch.tensor([v]) for k, v in result.items()}
                
                return result
        
        tokenizer = MinimalTokenizer()
        logger.warning("使用最小化tokenizer，可能影响模型性能")
    
    # 构建模型
    model = build_model(
        model_name="bert-base-chinese",  # 始终提供基础模型名称作为后备
        pretrained_model_path=pretrained_model_path,
        model_type=model_type,
        num_aspects=len(FLAT_ASPECTS),
        num_labels=4  # 0: 不适用, 1: 负面, 2: 中性, 3: 正面
    )
    
    # 模型关联tokenizer，方便后续使用
    model.tokenizer = tokenizer
    
    # 构建训练器
    trainer = ModelTrainer(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir if output_dir else "checkpoints/model",
        **{k: v for k, v in kwargs.items() if k not in ['pretrained_model_path', 'model_type', 'augmented_df']}
    )
    
    # 训练模型，传递增强数据
    train_metrics = trainer.train(train_df, val_df, augmented_df)
    
    return model, train_metrics

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="运行方面级情感分析完整流程")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--skip_pretraining", action="store_true", help="跳过预训练阶段")
    parser.add_argument("--skip_clustering", action="store_true", help="跳过聚类阶段")
    parser.add_argument("--skip_active_learning", action="store_true", help="跳过主动学习阶段")
    parser.add_argument("--skip_uda", action="store_true", help="跳过无监督数据增强阶段")
    parser.add_argument("--debug", action="store_true", help="启用调试日志")
    args = parser.parse_args()
    
    # 设置日志级别
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("调试模式已启用")
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置全局随机种子
    set_seed(config.get("seed", 42))
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 1. 数据加载和预处理
    logger.info("开始数据加载和预处理...")
    
    raw_train_path = config["data"]["raw_train_path"]
    raw_dev_path = config["data"]["raw_dev_path"]
    raw_test_path = config["data"]["raw_test_path"]
    processed_data_dir = config["data"]["processed_dir"]
    
    # 创建处理后的数据目录
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # 检查是否有处理好的数据
    processed_train_path = os.path.join(processed_data_dir, "train.csv")
    processed_dev_path = os.path.join(processed_data_dir, "dev.csv")
    processed_test_path = os.path.join(processed_data_dir, "test.csv")
    
    if (os.path.exists(processed_train_path) and
        os.path.exists(processed_dev_path) and
        os.path.exists(processed_test_path)):
        logger.info("加载已处理的数据...")
        train_df = load_processed_data(processed_train_path)
        dev_df = load_processed_data(processed_dev_path)
        test_df = load_processed_data(processed_test_path)
    else:
        logger.info("处理原始数据...")
        # 加载原始数据
        train_df = load_raw_data(raw_train_path)
        dev_df = load_raw_data(raw_dev_path)
        test_df = load_raw_data(raw_test_path)
        
        # 预处理数据
        train_df = preprocess_data(train_df)
        dev_df = preprocess_data(dev_df)
        test_df = preprocess_data(test_df)
        
        # 保存处理后的数据
        save_processed_data(train_df, processed_train_path)
        save_processed_data(dev_df, processed_dev_path)
        save_processed_data(test_df, processed_test_path)
    
    logger.info(f"训练集大小: {len(train_df)}")
    logger.info(f"开发集大小: {len(dev_df)}")
    logger.info(f"测试集大小: {len(test_df)}")
    
    # 2. 领域自适应预训练
    pretrained_model_path = None
    pretrained_model_type = config["pretraining"]["base_model"]
    if not args.skip_pretraining:
        logger.info("开始领域自适应预训练...")
        
        # 提取所有文本用于预训练
        all_texts = extract_unlabeled_data(train_df)
        if config["pretraining"].get("use_dev_texts", False):
            all_texts.extend(extract_unlabeled_data(dev_df))
        
        dev_texts = extract_unlabeled_data(dev_df) if config["pretraining"].get("validate", False) else None
        
        # 创建MLM预训练器
        mlm_trainer = MLMTrainer(
            model_name=config["pretraining"]["base_model"],
            output_dir=config["pretraining"]["output_dir"],
            mlm_probability=config["pretraining"].get("mlm_probability", 0.15),
            max_length=config["pretraining"].get("max_length", 128),
            batch_size=config["pretraining"].get("batch_size", 32),
            learning_rate=config["pretraining"].get("learning_rate", 5e-5),
            weight_decay=config["pretraining"].get("weight_decay", 0.01),
            epochs=config["pretraining"].get("epochs", 5),
            warmup_ratio=config["pretraining"].get("warmup_ratio", 0.1),
            seed=config["seed"],
            device=device
        )
        
        # 执行MLM预训练
        mlm_trainer.train(all_texts, dev_texts)
        
        # 使用最佳模型路径
        pretrained_model_path = os.path.join(config["pretraining"]["output_dir"], "best_model")
        logger.info(f"预训练完成，使用模型路径: {pretrained_model_path}")
    else:
        # 如果跳过预训练，使用指定的预训练模型
        if config.get("pretrained_model_path"):
            pretrained_model_path = config["pretrained_model_path"]
            logger.info(f"跳过预训练，使用指定模型路径: {pretrained_model_path}")
        else:
            logger.info(f"跳过预训练，使用原始模型: {pretrained_model_type}")
    
    # 验证预训练模型路径
    if pretrained_model_path:
        # 处理相对路径和绝对路径
        if not os.path.isabs(pretrained_model_path):
            # 如果是相对路径，转换为绝对路径
            abs_path = os.path.abspath(pretrained_model_path)
            logger.info(f"将相对路径转换为绝对路径: {pretrained_model_path} -> {abs_path}")
            pretrained_model_path = abs_path
            
        if os.path.exists(pretrained_model_path):
            logger.info(f"已验证预训练模型路径: {pretrained_model_path}")
        else:
            logger.warning(f"指定的预训练模型路径不存在: {pretrained_model_path}")
            logger.warning(f"将回退到原始预训练模型: {pretrained_model_type}")
            pretrained_model_path = None
    
    # 3. k-means++种子集选择
    seed_indices = None
    if not args.skip_clustering:
        logger.info("开始k-means++种子集选择...")
        
        # 确定使用哪个模型基础
        clustering_model = pretrained_model_path if pretrained_model_path else config["clustering"].get("base_model", "bert-base-chinese")
        logger.info(f"聚类使用模型: {clustering_model}")
        
        # 创建种子选择器
        seed_selector = KMeansSeedSelector(
            model_name=config["clustering"].get("base_model", "bert-base-chinese"),
            pretrained_model_path=pretrained_model_path,  # 优先使用预训练好的模型
            max_length=config["clustering"].get("max_length", 128),
            batch_size=config["clustering"].get("batch_size", 32),
            random_state=config["seed"],
            device=device
        )
        
        # 选择种子集
        seed_indices = seed_selector.select_seeds(
            train_df,
            seed_ratio=config["clustering"].get("seed_ratio", 0.05)
        )
        
        # 保存种子集索引
        seed_indices_path = os.path.join(processed_data_dir, "seed_indices.txt")
        with open(seed_indices_path, 'w') as f:
            f.write('\n'.join(map(str, seed_indices)))
        
        logger.info(f"已选择 {len(seed_indices)} 个种子样本，比例为 {len(seed_indices)/len(train_df):.2%}")
    else:
        # 如果跳过聚类，检查是否有保存的种子集索引
        seed_indices_path = os.path.join(processed_data_dir, "seed_indices.txt")
        if os.path.exists(seed_indices_path):
            logger.info("加载已保存的种子集索引...")
            with open(seed_indices_path, 'r') as f:
                seed_indices = [int(line.strip()) for line in f]
    
    # 4. UDA-MLM数据增强（可选）
    augmenter = None
    if not args.skip_uda:
        logger.info("准备UDA-MLM数据增强器...")
        
        # 确定使用哪个模型基础
        uda_model = pretrained_model_path if pretrained_model_path else config["uda"].get("base_model", "bert-base-chinese")
        logger.info(f"UDA使用模型: {uda_model}")
        
        # 创建数据增强器
        augmenter = MLMAugmenter(
            model_name=config["uda"].get("base_model", "bert-base-chinese"),
            pretrained_model_path=pretrained_model_path,  # 优先使用预训练好的模型
            mask_ratio=config["uda"].get("mask_ratio", 0.15),
            max_length=config["uda"].get("max_length", 128),
            max_predictions=config["uda"].get("max_predictions", 20),
            temperature=config["uda"].get("temperature", 1.0),
            num_augmentations=config["uda"].get("num_augmentations", 3),
            retain_aspect_tokens=config["uda"].get("retain_aspect_tokens", True),
            seed=config["seed"],
            device=device
        )
    
    # 5. 主动学习循环
    if not args.skip_active_learning:
        logger.info("开始主动学习循环...")
        
        # 确定使用哪个模型基础
        al_model = pretrained_model_path if pretrained_model_path else config["active_learning"].get("base_model", "bert-base-chinese")
        logger.info(f"主动学习使用模型: {al_model}")
        
        # 构建模型
        initial_model = build_model(
            model_name=config["active_learning"].get("base_model", "bert-base-chinese"),
            pretrained_model_path=pretrained_model_path,  # 优先使用预训练好的模型
            model_type=config["active_learning"].get("model_type", "basic"),
            num_aspects=len(FLAT_ASPECTS),
            num_labels=4  # 0: 不适用, 1: 负面, 2: 中性, 3: 正面
        )
        
        # 创建tokenizer - 使用与模型相同的来源
        tokenizer_source = pretrained_model_path if pretrained_model_path else config["active_learning"].get("base_model", "bert-base-chinese")
        logger.info(f"使用tokenizer: {tokenizer_source}")
        tokenizer = BertTokenizer.from_pretrained(tokenizer_source)
        
        # 创建不确定性采样器
        uncertainty_sampler = UncertaintySampler(
            model=initial_model,
            tokenizer=tokenizer,
            max_length=config["active_learning"].get("max_length", 128),
            batch_size=config["active_learning"].get("batch_size", 16),
            alpha=config["active_learning"].get("alpha", 0.5),
            device=device
        )
        
        # 创建种子选择器（如果未使用之前创建的）
        if not seed_selector and not args.skip_clustering:
            seed_selector = KMeansSeedSelector(
                model_name=config["clustering"].get("base_model", "bert-base-chinese"),
                pretrained_model_path=pretrained_model_path,
                max_length=config["clustering"].get("max_length", 128),
                batch_size=config["clustering"].get("batch_size", 32),
                random_state=config["seed"],
                device=device
            )
        
        # 准备训练参数
        train_kwargs = {
            "learning_rate": config["active_learning"].get("learning_rate", 2e-5),
            "weight_decay": config["active_learning"].get("weight_decay", 0.01),
            "batch_size": config["active_learning"].get("batch_size", 16),
            "epochs": config["active_learning"].get("epochs", 10),
            "warmup_ratio": config["active_learning"].get("warmup_ratio", 0.1),
            "pretrained_model_path": pretrained_model_path,  # 确保使用正确的预训练模型路径
            "model_type": config["active_learning"].get("model_type", "basic"),
        }
        
        logger.info(f"使用预训练模型路径: {pretrained_model_path}")
        
        # 确保模型路径存在
        if pretrained_model_path and not os.path.exists(pretrained_model_path):
            logger.warning(f"预训练模型路径不存在: {pretrained_model_path}")
            # 尝试备用路径
            alternate_path = os.path.abspath(os.path.join("data", "pretrained", "waimai-bert", "best_model"))
            if os.path.exists(alternate_path):
                logger.info(f"使用备用模型路径: {alternate_path}")
                train_kwargs["pretrained_model_path"] = alternate_path
                pretrained_model_path = alternate_path
            else:
                logger.warning(f"备用模型路径也不存在: {alternate_path}")
        
        # 如果使用UDA，添加一致性损失
        if not args.skip_uda:
            train_kwargs.update({
                "use_consistency_loss": True,
                "consistency_alpha": config["active_learning"].get("consistency_alpha", 0.1)
            })
        
        # 创建主动学习器
        active_learner = ActiveLearner(
            model_builder=model_builder_func,
            uncertainty_sampler=uncertainty_sampler,
            diversity_selector=seed_selector,
            train_kwargs=train_kwargs,
            output_dir=config["active_learning"]["output_dir"],
            initial_samples_ratio=config["active_learning"].get("initial_samples_ratio", 0.05),
            budget_per_round=config["active_learning"].get("budget_per_round", 0.05),
            max_rounds=config["active_learning"].get("max_rounds", 5),
            eval_metric=config["active_learning"].get("eval_metric", "macro_f1"),
            seed=config["seed"],
            device=device
        )
        
        # 如果使用UDA进行数据增强
        augmented_df = None
        if not args.skip_uda and "use_uda" in config["active_learning"] and config["active_learning"]["use_uda"]:
            logger.info("使用UDA-MLM进行数据增强...")
            
            # 确定要增强的数据
            if seed_indices:
                # 如果只想增强有标注的数据
                if config["active_learning"].get("augment_only_labeled", True):
                    subset_to_augment = train_df.iloc[seed_indices].copy()
                else:
                    # 增强所有数据
                    subset_to_augment = train_df.copy()
                
                # 进行数据增强
                augmented_df = augmenter.augment_dataset(
                    subset_to_augment,
                    text_column='review',
                    augmentation_ratio=config["active_learning"].get("augmentation_ratio", 1.0)
                )
                
                logger.info(f"生成了 {len(augmented_df) - len(subset_to_augment)} 个增强样本")
        
        # 运行主动学习循环
        best_model, history = active_learner.run(
            train_df=train_df,
            val_df=dev_df,
            test_df=test_df,
            initial_indices=seed_indices,
            augmented_df=augmented_df  # 传递增强后的数据
        )
        
        # 保存最终模型
        final_model_path = os.path.join(config["active_learning"]["output_dir"], "final_model")
        logger.info(f"最终模型已保存到 {final_model_path}")
    
    logger.info("所有任务完成!")

if __name__ == "__main__":
    main() 