import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    BertConfig,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import numpy as np
import random
import logging
from torch.utils.tensorboard import SummaryWriter

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class MLMTrainer:
    def __init__(
        self,
        model_name="bert-base-chinese",
        output_dir="data/pretrained",
        mlm_probability=0.15,
        max_length=128,
        batch_size=32,
        learning_rate=5e-5,
        weight_decay=0.01,
        epochs=5,
        warmup_ratio=0.1,
        seed=42,
        device=None
    ):
        # 设置随机种子
        self.seed = seed
        self._set_seed(seed)
        
        # 设置设备
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 训练参数
        self.output_dir = output_dir
        self.mlm_probability = mlm_probability
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        
        # 加载tokenizer和model
        logger.info(f"加载预训练模型: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        # 数据整理器，用于动态掩码
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=mlm_probability
        )
        
        # TensorBoard日志
        self.tb_writer = SummaryWriter(os.path.join(output_dir, "runs"))
    
    def _set_seed(self, seed):
        """设置随机种子以确保结果可重现"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def train(self, train_texts, val_texts=None):
        """训练MLM模型"""
        from src.utils.data_utils import MLMDataset
        
        # 创建数据集和数据加载器
        logger.info("创建训练数据集")
        train_dataset = MLMDataset(train_texts, self.tokenizer, self.max_length)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator
        )
        
        # 如果有验证集，也创建验证集的数据加载器
        val_dataloader = None
        if val_texts:
            logger.info("创建验证数据集")
            val_dataset = MLMDataset(val_texts, self.tokenizer, self.max_length)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                collate_fn=self.data_collator
            )
        
        # 优化器
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
        total_steps = len(train_dataloader) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        
        # 开始训练
        logger.info("开始训练")
        global_step = 0
        best_val_loss = float("inf")
        
        for epoch in range(self.epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_steps = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for batch in progress_bar:
                # 将数据移动到设备
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                
                # 反向传播
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # 更新进度条
                train_loss += loss.item()
                train_steps += 1
                global_step += 1
                
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # 记录到TensorBoard
                self.tb_writer.add_scalar("train/loss", loss.item(), global_step)
            
            avg_train_loss = train_loss / train_steps
            logger.info(f"Epoch {epoch+1}/{self.epochs} - 平均训练损失: {avg_train_loss:.4f}")
            
            # 验证阶段
            if val_dataloader:
                self.model.eval()
                val_loss = 0.0
                val_steps = 0
                
                with torch.no_grad():
                    for batch in tqdm(val_dataloader, desc="Validation"):
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        labels = batch["labels"].to(self.device)
                        
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss
                        
                        val_loss += loss.item()
                        val_steps += 1
                
                avg_val_loss = val_loss / val_steps
                logger.info(f"Epoch {epoch+1}/{self.epochs} - 平均验证损失: {avg_val_loss:.4f}")
                
                # 记录到TensorBoard
                self.tb_writer.add_scalar("val/loss", avg_val_loss, epoch)
                
                # 保存最佳模型
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.save_model(os.path.join(self.output_dir, "best_model"))
            
            # 保存每个epoch的模型
            model_path = os.path.join(self.output_dir, f"checkpoint-{epoch}")
            self.save_model(model_path)
        
        # 保存最终模型
        self.save_model(os.path.join(self.output_dir, "final_model"))
        logger.info("训练完成")
    
    def save_model(self, output_dir):
        """保存模型和tokenizer"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        logger.info(f"保存模型到 {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir) 