import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, BertConfig
import logging
import os

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class ACSAModel(BertPreTrainedModel):
    """
    方面级情感分析模型，基于BERT
    """
    def __init__(self, config, num_aspects=18, num_labels=4):
        super().__init__(config)
        self.num_labels = num_labels
        self.num_aspects = num_aspects
        
        # BERT编码器
        self.bert = BertModel(config)
        
        # 方面情感分类头
        self.classifier = nn.Linear(config.hidden_size, num_aspects * num_labels)
        
        # 类别权重（由外部设置）
        self.class_weights = None
        
        # 初始化
        self.init_weights()
    
    def set_class_weights(self, weights):
        """设置类别权重，用于处理类别不平衡问题"""
        self.class_weights = weights
        logger.info(f"设置了类别权重: shape={weights.shape}")
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        前向传播
        
        参数:
        input_ids: 输入token ids
        attention_mask: 注意力掩码
        token_type_ids: token类型ids
        position_ids: 位置ids
        head_mask: 头部掩码
        inputs_embeds: 输入嵌入
        labels: 标签，形状为[batch_size, num_aspects]
        output_attentions: 是否输出注意力
        output_hidden_states: 是否输出隐藏状态
        return_dict: 是否返回字典
        
        返回:
        包含损失和logits的元组
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 获取BERT输出
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 使用[CLS]标记的表示作为句子表示
        pooled_output = outputs[1]
        
        # 分类头
        logits = self.classifier(pooled_output)
        
        # 重塑logits以适应多方面情感分类
        logits = logits.view(-1, self.num_aspects, self.num_labels)
        
        loss = None
        if labels is not None:
            # 使用类别权重（如果可用）
            if self.class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=0, weight=self.class_weights)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # 忽略标签为0的样本（不适用）
            
            # 重塑标签以方便计算损失
            active_labels = labels.view(-1)  # [batch_size * num_aspects]
            active_logits = logits.view(-1, self.num_labels)  # [batch_size * num_aspects, num_labels]
            
            # 使用掩码忽略标签为0的样本
            active_mask = active_labels != 0
            if active_mask.sum() > 0:  # 确保有有效样本
                masked_labels = active_labels[active_mask]
                masked_logits = active_logits[active_mask]
                loss = loss_fct(masked_logits, masked_labels)
            else:
                # 如果没有有效样本，使用零损失
                loss = torch.tensor(0.0, device=logits.device)
        
        # 如果return_dict为True，返回一个字典
        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": outputs.hidden_states,
                "attentions": outputs.attentions,
            }
        
        # 否则返回元组
        return (loss, logits) if loss is not None else (logits,)


class EnhancedACSAModel(BertPreTrainedModel):
    """
    增强版方面级情感分析模型，使用方面特定的表示
    """
    def __init__(self, config, aspect_embeddings=None, num_aspects=18, num_labels=4):
        super().__init__(config)
        self.num_labels = num_labels
        self.num_aspects = num_aspects
        
        # BERT编码器
        self.bert = BertModel(config)
        
        # 方面嵌入
        if aspect_embeddings is not None:
            self.aspect_embeddings = nn.Parameter(aspect_embeddings)
        else:
            # 随机初始化方面嵌入
            self.aspect_embeddings = nn.Parameter(torch.randn(num_aspects, config.hidden_size))
        
        # 注意力层，用于融合方面信息
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            batch_first=True
        )
        
        # 分类头，每个方面一个
        self.classifiers = nn.ModuleList([
            nn.Linear(config.hidden_size * 2, num_labels) for _ in range(num_aspects)
        ])
        
        # 正则化层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 统一类别权重（由外部设置）
        self.class_weights = None
        
        # 方面特定类别权重（由外部设置）
        self.aspect_weights = None
        
        # 方面特定正则化强度
        self.aspect_reg_strength = nn.Parameter(torch.ones(num_aspects) * 0.001)
        
        # 添加方面特定偏差
        self.aspect_bias = nn.Parameter(torch.zeros(num_aspects))
        
        # 初始化
        self.init_weights()
    
    def set_class_weights(self, weights):
        """设置统一类别权重，用于处理类别不平衡问题"""
        self.class_weights = weights
        logger.info(f"设置了增强模型的统一类别权重: shape={weights.shape}")
    
    def set_aspect_weights(self, weights):
        """设置方面特定类别权重，用于处理各方面的类别不平衡问题"""
        self.aspect_weights = weights
        logger.info(f"设置了增强模型的方面特定类别权重: shape={weights.shape}")
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        前向传播
        
        参数:
        input_ids: 输入token ids
        attention_mask: 注意力掩码
        token_type_ids: token类型ids
        position_ids: 位置ids
        head_mask: 头部掩码
        inputs_embeds: 输入嵌入
        labels: 标签，形状为[batch_size, num_aspects]
        output_attentions: 是否输出注意力
        output_hidden_states: 是否输出隐藏状态
        return_dict: 是否返回字典
        
        返回:
        包含损失和logits的元组
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 获取BERT输出
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 获取所有token的隐藏状态
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        
        # 获取[CLS]标记的表示
        cls_output = outputs[1]  # [batch_size, hidden_size]
        
        batch_size = sequence_output.size(0)
        
        # 准备logits存储
        all_logits = []
        l2_reg_total = torch.tensor(0., device=cls_output.device)
        
        # 为每个方面计算情感预测
        for aspect_idx in range(self.num_aspects):
            # 获取当前方面的嵌入
            aspect_emb = self.aspect_embeddings[aspect_idx].unsqueeze(0).expand(batch_size, -1)  # [batch_size, hidden_size]
            
            # 使用多头注意力机制融合方面信息
            # 将方面嵌入作为query，序列输出作为key和value
            aspect_emb = aspect_emb.unsqueeze(1)  # [batch_size, 1, hidden_size]
            attn_output, _ = self.attention(
                aspect_emb,
                sequence_output,
                sequence_output,
                key_padding_mask=(1 - attention_mask).bool() if attention_mask is not None else None
            )
            
            # 合并方面特定的表示与[CLS]表示
            attn_output = attn_output.squeeze(1)  # [batch_size, hidden_size]
            combined = torch.cat([cls_output, attn_output], dim=1)  # [batch_size, hidden_size*2]
            combined = self.dropout(combined)
            
            # 使用方面特定的分类器
            logits = self.classifiers[aspect_idx](combined)  # [batch_size, num_labels]
            
            # 应用方面偏置，使稀有方面更可能被预测
            # 偏置基于方面的整体频率(通过aspect_bias参数)
            logits = logits + self.aspect_bias[aspect_idx]
            
            all_logits.append(logits)
            
            # 计算方面特定的L2正则化，使用学习的正则化强度
            l2_reg = torch.tensor(0., device=logits.device)
            for param in self.classifiers[aspect_idx].parameters():
                l2_reg += torch.norm(param)
            l2_reg_total += self.aspect_reg_strength[aspect_idx] * l2_reg
        
        # 合并所有方面的logits
        logits = torch.stack(all_logits, dim=1)  # [batch_size, num_aspects, num_labels]
        
        loss = None
        if labels is not None:
            loss_total = torch.tensor(0.0, device=logits.device)
            valid_aspect_count = 0
            
            # 处理每个方面的损失
            for aspect_idx in range(self.num_aspects):
                # 确定使用哪种权重
                if self.aspect_weights is not None:
                    aspect_weight = self.aspect_weights[aspect_idx]
                    loss_fct = nn.CrossEntropyLoss(ignore_index=0, weight=aspect_weight)
                elif self.class_weights is not None:
                    loss_fct = nn.CrossEntropyLoss(ignore_index=0, weight=self.class_weights)
                else:
                    loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # 忽略标签为0的样本（不适用）
                
                # 获取当前方面的标签和logits
                aspect_labels = labels[:, aspect_idx]  # [batch_size]
                aspect_logits = logits[:, aspect_idx, :]  # [batch_size, num_labels]
                
                # 使用掩码忽略标签为0的样本
                active_mask = aspect_labels != 0
                if active_mask.sum() > 0:  # 确保有有效样本
                    masked_labels = aspect_labels[active_mask]
                    masked_logits = aspect_logits[active_mask]
                    aspect_loss = loss_fct(masked_logits, masked_labels)
                    loss_total += aspect_loss
                    valid_aspect_count += 1
            
            # 计算平均损失
            if valid_aspect_count > 0:
                loss = loss_total / valid_aspect_count
                # 添加学习的L2正则化
                loss += l2_reg_total
            else:
                # 如果没有有效样本，使用零损失
                loss = torch.tensor(0.0, device=logits.device)
        
        # 如果return_dict为True，返回一个字典
        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": outputs.hidden_states,
                "attentions": outputs.attentions,
            }
        
        # 否则返回元组
        return (loss, logits) if loss is not None else (logits,)


def build_model(model_name="bert-base-chinese", pretrained_model_path=None, model_type="basic", **kwargs):
    """
    构建方面级情感分析模型
    
    参数:
    model_name: 预训练模型名称
    pretrained_model_path: 预训练模型路径
    model_type: 模型类型，"basic"或"enhanced"
    
    返回:
    构建好的模型
    """
    logger.info(f"构建方面级情感分析模型，类型: {model_type}")
    
    # 检查预训练模型路径
    if pretrained_model_path:
        if os.path.exists(pretrained_model_path):
            logger.info(f"使用预训练模型: {pretrained_model_path}")
            model_path_valid = True
        else:
            logger.warning(f"预训练模型路径不存在: {pretrained_model_path}")
            logger.warning(f"将使用Hugging Face模型: {model_name}")
            model_path_valid = False
    else:
        logger.info(f"未指定预训练模型路径，使用Hugging Face模型: {model_name}")
        model_path_valid = False
    
    try:
        # 加载配置
        if model_path_valid:
            try:
                config = BertConfig.from_pretrained(pretrained_model_path)
                logger.info(f"从路径加载配置成功: {pretrained_model_path}")
            except Exception as e:
                logger.error(f"从路径加载配置失败: {str(e)}")
                logger.info(f"尝试从Hugging Face加载配置: {model_name}")
                config = BertConfig.from_pretrained(model_name)
        else:
            config = BertConfig.from_pretrained(model_name)
        
        # 创建模型
        if model_type == "basic":
            model = ACSAModel(config, **kwargs)
            logger.info("创建基础ACSA模型")
        elif model_type == "enhanced":
            model = EnhancedACSAModel(config, **kwargs)
            logger.info("创建增强ACSA模型")
        else:
            logger.error(f"不支持的模型类型: {model_type}，使用基础模型")
            model = ACSAModel(config, **kwargs)
        
        # 加载预训练权重
        if model_path_valid:
            try:
                # 检查是否存在模型文件
                if os.path.exists(os.path.join(pretrained_model_path, "pytorch_model.bin")):
                    # 加载整个模型
                    pretrained_dict = torch.load(
                        os.path.join(pretrained_model_path, "pytorch_model.bin"),
                        map_location="cpu"
                    )
                    
                    # 筛选出BERT部分的权重
                    bert_dict = {k: v for k, v in pretrained_dict.items() if k.startswith("bert.")}
                    
                    # 加载BERT部分的权重
                    load_result = model.load_state_dict(bert_dict, strict=False)
                    
                    # 打印加载结果
                    if load_result.missing_keys:
                        logger.info(f"未加载的键: {len(load_result.missing_keys)} 个")
                    if load_result.unexpected_keys:
                        logger.warning(f"意外的键: {len(load_result.unexpected_keys)} 个")
                    
                    logger.info(f"从 {pretrained_model_path} 加载了预训练的BERT权重")
                else:
                    logger.warning(f"在 {pretrained_model_path} 中找不到pytorch_model.bin文件")
                    logger.info("模型将使用随机初始化权重")
            except Exception as e:
                logger.error(f"加载预训练权重时出错: {str(e)}")
                logger.info("模型将使用随机初始化权重")
        
        return model
        
    except Exception as e:
        logger.error(f"构建模型时发生错误: {str(e)}")
        logger.info("尝试创建带有默认配置的基础模型")
        
        # 创建默认配置和基础模型作为最后的回退选项
        default_config = BertConfig.from_pretrained("bert-base-chinese", local_files_only=False)
        return ACSAModel(default_config, **kwargs) 