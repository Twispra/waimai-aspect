# 外卖评论的方面级情感分析框架

这个项目实现了一个方面级情感分析框架，用于分析中文外卖评论数据。框架包含四个主要组件：

1. 领域自适应BERT预训练
2. k-means++种子集选择
3. 主动学习循环
4. 基于MLM的无监督数据增强(UDA-MLM)

## 项目结构

```
├── data/
│   ├── raw/                # 原始数据
│   ├── processed/          # 处理后的数据
│   └── pretrained/         # 预训练模型权重
├── src/
│   ├── pretraining/        # 领域适应性预训练相关代码
│   ├── clustering/         # k-means++种子集选择相关代码
│   ├── active_learning/    # 主动学习循环相关代码
│   ├── uda/               # 基于MLM的无监督数据增强相关代码
│   ├── models/            # 模型定义
│   └── utils/             # 工具函数
├── notebooks/             # 实验和可视化笔记本
├── scripts/               # 训练和评估脚本
└── configs/               # 配置文件
```

## 数据格式

数据集包含以下内容：
- `id`: 评论ID
- `review`: 评论文本
- `star`: 总体评分（1.0-5.0）
- 方面评分: 总共18个方面，每个格式为`{类别}#{特性}`
  - 评分值: 1(正面), 0(中性), -1(负面), -2(不适用)

## 使用方法

### 环境设置

```bash
pip install -r requirements.txt
```

### 领域自适应预训练

```bash
python scripts/run_domain_adaptation.py --config configs/domain_adaptation.yaml
```

### 种子集选择

```bash
python scripts/run_clustering.py --config configs/clustering.yaml
```

### 主动学习训练

```bash
python scripts/run_active_learning.py --config configs/active_learning.yaml
```

### 无监督数据增强

```bash
python scripts/run_uda.py --config configs/uda.yaml
```

### 完整训练流程

```bash
python scripts/run_pipeline.py --config configs/pipeline.yaml
```

## 评估

```bash
python scripts/evaluate.py --model_path checkpoints/final_model/ --test_file data/processed/asap_test.csv
``` 