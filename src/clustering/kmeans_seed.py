import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm
import logging
from transformers import BertModel, BertTokenizer
import os
import psutil
import gc

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class KMeansSeedSelector:
    def __init__(
        self,
        model_name="bert-base-chinese",
        pretrained_model_path=None,
        max_length=128,
        batch_size=32,
        random_state=42,
        device=None
    ):
        # 设置设备
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 模型参数
        self.max_length = max_length
        self.batch_size = batch_size
        self.random_state = random_state
        
        # 加载预训练模型或指定模型
        logger.info(f"加载模型: {pretrained_model_path if pretrained_model_path else model_name}")
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
            self.model = BertModel.from_pretrained(pretrained_model_path)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
        
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
    
    def extract_features(self, texts):
        """
        为文本提取BERT特征向量
        """
        logger.info("正在提取特征向量...")
        total_texts = len(texts)
        logger.info(f"总文本数: {total_texts}")
        
        # 报告初始内存使用情况
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)
        logger.info(f"初始内存使用: {initial_memory:.2f} MB")
        
        embeddings = []
        
        # 优化：对于大型数据集，使用分批和垃圾收集来管理内存
        batch_size = min(self.batch_size, 32)  # 使用较小的批次以减少内存占用
        
        # 分批处理文本
        progress_bar = tqdm(range(0, len(texts), batch_size), desc="提取特征")
        for i in progress_bar:
            batch_texts = texts[i:i+batch_size]
            current_batch = i // batch_size + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            # 更新进度条描述
            progress_desc = f"提取特征 [批次 {current_batch}/{total_batches}]"
            progress_bar.set_description(progress_desc)
            
            # 使用tokenizer处理文本
            encoding = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # 将编码移至设备并获取BERT特征
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # 使用[CLS]令牌的输出作为句子表示
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
            
            # 定期清理缓存，防止CUDA内存泄漏
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 每处理50个批次报告一次内存使用情况
            if current_batch % 50 == 0:
                current_memory = process.memory_info().rss / (1024 * 1024)
                memory_diff = current_memory - initial_memory
                logger.info(f"当前内存使用: {current_memory:.2f} MB (增加了 {memory_diff:.2f} MB)")
                
                # 如果内存使用量增长过快，主动触发垃圾收集
                if memory_diff > 1000:  # 如果内存增加超过1GB
                    logger.info("执行垃圾收集...")
                    gc.collect()
        
        # 报告最终内存使用情况
        final_memory = process.memory_info().rss / (1024 * 1024)
        logger.info(f"特征提取完成。最终内存使用: {final_memory:.2f} MB (增加了 {final_memory - initial_memory:.2f} MB)")
        
        return np.array(embeddings)
    
    def select_seeds(self, df, seed_ratio=0.05):
        """
        使用K-means++算法选择种子数据
        
        参数:
        df: 包含'review'列的DataFrame
        seed_ratio: 选择的种子样本比例
        
        返回:
        选择的种子样本的索引列表
        """
        texts = df['review'].tolist()
        n_seeds = int(len(texts) * seed_ratio)
        logger.info(f"正在选择 {n_seeds} 个种子样本 ({seed_ratio * 100:.1f}%)...")
        
        # 1. 提取特征向量
        features = self.extract_features(texts)
        
        # 2. 使用K-means++进行聚类
        logger.info(f"正在进行K-means++聚类 (k={n_seeds})，这可能需要较长时间...")
        logger.info("较大的聚类数可能需要几分钟到几十分钟不等，请耐心等待...")
        
        # 考虑聚类数量，选择合适的算法
        if n_seeds > 1000:
            # 对于大规模聚类，使用MiniBatchKMeans以提高速度
            from sklearn.cluster import MiniBatchKMeans
            logger.info("使用MiniBatchKMeans加速聚类过程...")
            
            # 显示进度条
            from tqdm import tqdm
            n_iterations = 100  # MiniBatchKMeans的默认迭代次数
            
            with tqdm(total=n_iterations, desc="MiniBatchKMeans聚类") as pbar:
                def update_progress(*args, **kwargs):
                    pbar.update(1)
                
                kmeans = MiniBatchKMeans(
                    n_clusters=n_seeds,
                    init='k-means++',
                    batch_size=1000,  # 每批处理的样本数
                    max_iter=n_iterations,
                    random_state=self.random_state,
                    verbose=0  # 禁用内置日志
                )
                
                # 手动拆分批次并更新进度
                batch_size = 1000
                for i in range(0, n_iterations):
                    # 使用所有数据，但每次迭代只处理一个批次的中心点更新
                    kmeans.partial_fit(features)
                    pbar.update(1)
        else:
            # 对于较小规模聚类，使用标准KMeans
            from sklearn.cluster import KMeans
            
            # 估计运行时间
            complexity = n_seeds * len(features)
            if complexity > 50000000:  # 这是一个经验阈值
                estimated_time = "可能需要10-30分钟"
            elif complexity > 10000000:
                estimated_time = "可能需要3-10分钟"
            else:
                estimated_time = "可能需要1-3分钟"
            
            logger.info(f"聚类规模: {complexity}，{estimated_time}")
            
            kmeans = KMeans(
                n_clusters=n_seeds,
                init='k-means++',
                n_init=10,
                random_state=self.random_state,
                verbose=1  # 启用进度输出
            )
        
        kmeans.fit(features)
        logger.info("K-means++聚类完成")
        
        # 3. 为每个聚类找到最近的样本
        logger.info("为每个聚类选择最具代表性的样本...")
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, features)
        
        # 4. 返回种子集索引
        seed_indices = sorted(closest.tolist())
        logger.info(f"已选择 {len(seed_indices)} 个种子样本")
        
        return seed_indices
    
    def select_diverse_samples(self, df, seed_indices, n_samples):
        """
        在已有种子集的基础上，选择最多样化的新样本
        用于主动学习循环中增加新样本
        
        参数:
        df: 包含'review'列的DataFrame
        seed_indices: 已有种子样本的索引
        n_samples: 要选择的新样本数量
        
        返回:
        新选择的样本索引列表
        """
        # 1. 获取所有样本和已标注样本的索引
        all_indices = set(range(len(df)))
        labeled_indices = set(seed_indices)
        unlabeled_indices = list(all_indices - labeled_indices)
        
        # 如果没有未标注的样本或不需要选择新样本，直接返回空列表
        if not unlabeled_indices or n_samples <= 0:
            return []
        
        # 2. 提取所有未标注样本的特征
        unlabeled_texts = df.iloc[unlabeled_indices]['review'].tolist()
        unlabeled_features = self.extract_features(unlabeled_texts)
        
        # 3. 提取已标注样本的特征
        labeled_texts = df.iloc[list(labeled_indices)]['review'].tolist()
        labeled_features = self.extract_features(labeled_texts)
        
        # 4. 计算每个未标注样本到最近的已标注样本的距离
        # 使用余弦距离，值越大表示越不相似
        distances = []
        for i, unlabeled_feature in enumerate(unlabeled_features):
            # 将特征向量重塑为二维数组
            unlabeled_feature = unlabeled_feature.reshape(1, -1)
            
            # 计算与所有已标注样本的余弦相似度
            similarities = np.dot(unlabeled_feature, labeled_features.T) / (
                np.linalg.norm(unlabeled_feature) * np.linalg.norm(labeled_features, axis=1)
            )
            
            # 取最大相似度，并转换为距离
            max_similarity = np.max(similarities)
            distance = 1 - max_similarity
            distances.append(distance)
        
        # 5. 选择距离最大的n_samples个样本
        top_indices = np.argsort(distances)[-n_samples:]
        selected_indices = [unlabeled_indices[i] for i in top_indices]
        
        return selected_indices 