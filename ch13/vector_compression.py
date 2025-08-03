"""
向量压缩与存储优化
包含产品量化(PQ)技术和存储成本优化策略
"""

import numpy as np
from typing import List, Tuple, Dict
import pickle

class ProductQuantizer:
    def __init__(self, dimension: int, num_subvectors: int = 8, num_centroids: int = 256):
        """
        产品量化器初始化
        Args:
            dimension: 原始向量维度
            num_subvectors: 子向量数量，通常选择8或16
            num_centroids: 每个子空间的聚类中心数，通常选择256
        """
        self.dimension = dimension
        self.num_subvectors = num_subvectors
        self.num_centroids = num_centroids
        
        # 确保维度可以被子向量数整除
        assert dimension % num_subvectors == 0, "维度必须被子向量数整除"
        self.subvector_dimension = dimension // num_subvectors
        
        # 每个子空间的聚类中心
        self.codebooks = None
        self.trained = False
        
    def train(self, training_vectors: np.ndarray):
        """训练量化器"""
        print(f"开始训练PQ量化器，输入数据形状: {training_vectors.shape}")
        
        self.codebooks = []
        
        for i in range(self.num_subvectors):
            start_dim = i * self.subvector_dimension
            end_dim = (i + 1) * self.subvector_dimension
            
            # 提取子向量
            subvectors = training_vectors[:, start_dim:end_dim]
            
            # K-means聚类
            centroids = self._kmeans_clustering(subvectors, self.num_centroids)
            self.codebooks.append(centroids)
            
            print(f"子空间 {i+1}/{self.num_subvectors} 训练完成")
        
        self.trained = True
        print("PQ量化器训练完成")
    
    def _kmeans_clustering(self, data: np.ndarray, k: int, max_iters: int = 100) -> np.ndarray:
        """K-means聚类实现"""
        n_samples, n_features = data.shape
        
        # 随机初始化聚类中心
        centroids = data[np.random.choice(n_samples, k, replace=False)]
        
        for iteration in range(max_iters):
            # 计算每个点到聚类中心的距离
            distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
            
            # 分配到最近的聚类中心
            closest_cluster = np.argmin(distances, axis=0)
            
            # 更新聚类中心
            new_centroids = np.array([
                data[closest_cluster == i].mean(axis=0) 
                for i in range(k)
            ])
            
            # 检查收敛
            if np.allclose(centroids, new_centroids, rtol=1e-4):
                break
                
            centroids = new_centroids
        
        return centroids
    
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """向量编码为PQ码"""
        assert self.trained, "量化器尚未训练"
        
        n_vectors = vectors.shape[0]
        codes = np.zeros((n_vectors, self.num_subvectors), dtype=np.uint8)
        
        for i in range(self.num_subvectors):
            start_dim = i * self.subvector_dimension
            end_dim = (i + 1) * self.subvector_dimension
            
            # 提取子向量
            subvectors = vectors[:, start_dim:end_dim]
            
            # 计算到每个聚类中心的距离
            distances = np.sqrt(((subvectors[:, np.newaxis, :] - 
                                self.codebooks[i][np.newaxis, :, :])**2).sum(axis=2))
            
            # 找到最近的聚类中心索引
            codes[:, i] = np.argmin(distances, axis=1)
        
        return codes
    
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """PQ码解码为近似向量"""
        assert self.trained, "量化器尚未训练"
        
        n_vectors, n_subvectors = codes.shape
        decoded_vectors = np.zeros((n_vectors, self.dimension))
        
        for i in range(self.num_subvectors):
            start_dim = i * self.subvector_dimension
            end_dim = (i + 1) * self.subvector_dimension
            
            # 从codebook中查找对应的向量
            decoded_vectors[:, start_dim:end_dim] = self.codebooks[i][codes[:, i]]
        
        return decoded_vectors
    
    def compute_distances_asymmetric(self, query_vector: np.ndarray, 
                                   database_codes: np.ndarray) -> np.ndarray:
        """不对称距离计算（查询向量vs编码数据库）"""
        assert self.trained, "量化器尚未训练"
        
        n_database_vectors = database_codes.shape[0]
        distances = np.zeros(n_database_vectors)
        
        for i in range(self.num_subvectors):
            start_dim = i * self.subvector_dimension
            end_dim = (i + 1) * self.subvector_dimension
            
            # 查询向量的子向量
            query_subvector = query_vector[start_dim:end_dim]
            
            # 计算查询子向量到所有聚类中心的距离
            subvector_distances = np.sqrt(((query_subvector[np.newaxis, :] - 
                                          self.codebooks[i])**2).sum(axis=1))
            
            # 累加到总距离
            distances += subvector_distances[database_codes[:, i]]
        
        return distances
    
    def save(self, filepath: str):
        """保存训练好的量化器"""
        save_data = {
            'dimension': self.dimension,
            'num_subvectors': self.num_subvectors,
            'num_centroids': self.num_centroids,
            'subvector_dimension': self.subvector_dimension,
            'codebooks': self.codebooks,
            'trained': self.trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    @classmethod
    def load(cls, filepath: str):
        """加载训练好的量化器"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        pq = cls(
            dimension=save_data['dimension'],
            num_subvectors=save_data['num_subvectors'],
            num_centroids=save_data['num_centroids']
        )
        
        pq.subvector_dimension = save_data['subvector_dimension']
        pq.codebooks = save_data['codebooks']
        pq.trained = save_data['trained']
        
        return pq

class CompressedVectorIndex:
    def __init__(self, dimension: int, compression_config: Dict = None):
        self.dimension = dimension
        self.compression_config = compression_config or {
            'method': 'pq',
            'num_subvectors': 8,
            'num_centroids': 256
        }
        
        # 初始化压缩器
        if self.compression_config['method'] == 'pq':
            self.compressor = ProductQuantizer(
                dimension=dimension,
                num_subvectors=self.compression_config['num_subvectors'],
                num_centroids=self.compression_config['num_centroids']
            )
        
        self.compressed_vectors = None
        self.vector_ids = []
        
    def train_compressor(self, training_vectors: np.ndarray):
        """训练压缩器"""
        print(f"开始训练压缩器，训练数据: {training_vectors.shape}")
        self.compressor.train(training_vectors)
        print("压缩器训练完成")
    
    def add_vectors(self, vectors: np.ndarray, vector_ids: List[str]):
        """添加压缩向量"""
        if not self.compressor.trained:
            print("警告：压缩器尚未训练，使用输入向量进行训练")
            self.train_compressor(vectors)
        
        # 压缩向量
        compressed = self.compressor.encode(vectors)
        
        if self.compressed_vectors is None:
            self.compressed_vectors = compressed
        else:
            self.compressed_vectors = np.vstack([self.compressed_vectors, compressed])
        
        self.vector_ids.extend(vector_ids)
        
        print(f"添加了 {len(vectors)} 个压缩向量")
    
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """搜索最相似的向量"""
        if self.compressed_vectors is None:
            return []
        
        # 计算不对称距离
        distances = self.compressor.compute_distances_asymmetric(
            query_vector, self.compressed_vectors
        )
        
        # 找到top-k最近的向量
        top_indices = np.argpartition(distances, top_k)[:top_k]
        top_indices = top_indices[np.argsort(distances[top_indices])]
        
        results = []
        for idx in top_indices:
            vector_id = self.vector_ids[idx]
            distance = distances[idx]
            results.append((vector_id, distance))
        
        return results
    
    def get_compression_stats(self) -> Dict:
        """获取压缩统计信息"""
        if self.compressed_vectors is None:
            return {}
        
        original_size = len(self.vector_ids) * self.dimension * 4  # float32
        compressed_size = self.compressed_vectors.nbytes
        compression_ratio = original_size / compressed_size
        
        return {
            'original_vectors': len(self.vector_ids),
            'original_size_mb': original_size / (1024 * 1024),
            'compressed_size_mb': compressed_size / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'space_saving_percent': (1 - 1/compression_ratio) * 100
        }

class StorageOptimizedVectorDB:
    def __init__(self, config: Dict):
        self.config = config
        self.hot_storage = {}   # 内存中的热数据
        self.warm_storage = {}  # SSD中的温数据
        self.cold_storage = {}  # 归档存储中的冷数据
        
        # 分层存储配置
        self.hot_threshold = config.get('hot_threshold', 1000)  # 最近1000次访问
        self.warm_threshold = config.get('warm_threshold', 100)  # 最近100次访问
        
        # 压缩配置
        self.compression_enabled = config.get('compression_enabled', True)
        self.pq_compressor = None
        
    def optimize_storage_cost(self, vectors: np.ndarray, access_patterns: Dict):
        """基于访问模式优化存储成本"""
        
        # 1. 分析访问模式
        hot_vectors, warm_vectors, cold_vectors = self._classify_vectors_by_access(
            vectors, access_patterns
        )
        
        # 2. 分层存储
        storage_plan = {
            'hot_storage': {
                'vectors': hot_vectors,
                'storage_type': 'memory',
                'compression': False,
                'cost_per_gb': 100  # 内存成本高
            },
            'warm_storage': {
                'vectors': warm_vectors, 
                'storage_type': 'ssd',
                'compression': True,
                'cost_per_gb': 10   # SSD成本中等
            },
            'cold_storage': {
                'vectors': cold_vectors,
                'storage_type': 'archive',
                'compression': True,
                'cost_per_gb': 1    # 归档成本低
            }
        }
        
        # 3. 计算成本效益
        cost_analysis = self._calculate_storage_costs(storage_plan)
        
        return storage_plan, cost_analysis
    
    def _classify_vectors_by_access(self, vectors: np.ndarray, 
                                  access_patterns: Dict) -> Tuple[List, List, List]:
        """根据访问模式分类向量"""
        hot_vectors = []
        warm_vectors = []
        cold_vectors = []
        
        for i, vector in enumerate(vectors):
            vector_id = f"vec_{i}"
            access_count = access_patterns.get(vector_id, 0)
            
            if access_count >= self.hot_threshold:
                hot_vectors.append((vector_id, vector))
            elif access_count >= self.warm_threshold:
                warm_vectors.append((vector_id, vector))
            else:
                cold_vectors.append((vector_id, vector))
        
        return hot_vectors, warm_vectors, cold_vectors
    
    def _calculate_storage_costs(self, storage_plan: Dict) -> Dict:
        """计算存储成本"""
        total_cost = 0
        cost_breakdown = {}
        
        for tier, plan in storage_plan.items():
            vectors = plan['vectors']
            cost_per_gb = plan['cost_per_gb']
            compression = plan['compression']
            
            # 计算数据大小
            vector_count = len(vectors)
            if vector_count > 0:
                vector_size = len(vectors[0][1]) * 4  # float32
                total_size_gb = (vector_count * vector_size) / (1024**3)
                
                # 如果启用压缩，减少75%存储
                if compression:
                    total_size_gb *= 0.25
                
                tier_cost = total_size_gb * cost_per_gb
                total_cost += tier_cost
                
                cost_breakdown[tier] = {
                    'vector_count': vector_count,
                    'size_gb': total_size_gb,
                    'cost': tier_cost
                }
        
        return {
            'total_cost': total_cost,
            'breakdown': cost_breakdown,
            'estimated_savings': self._calculate_savings(cost_breakdown)
        }
    
    def _calculate_savings(self, cost_breakdown: Dict) -> Dict:
        """计算成本节省"""
        # 计算如果全部使用热存储的成本
        total_vectors = sum(tier['vector_count'] for tier in cost_breakdown.values())
        total_size = sum(tier['size_gb'] for tier in cost_breakdown.values())
        
        hot_storage_cost = total_size * 100  # 全部使用内存的成本
        actual_cost = sum(tier['cost'] for tier in cost_breakdown.values())
        
        savings = hot_storage_cost - actual_cost
        savings_percent = (savings / hot_storage_cost) * 100
        
        return {
            'absolute_savings': savings,
            'percentage_savings': savings_percent,
            'hot_storage_cost': hot_storage_cost,
            'optimized_cost': actual_cost
        }

# 使用示例
def demonstrate_vector_compression():
    """演示向量压缩的使用"""
    
    # 生成示例数据
    dimension = 768  # 典型的嵌入维度
    num_vectors = 10000
    
    print("=== 向量压缩技术演示 ===")
    
    # 生成随机向量数据
    vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
    vector_ids = [f"doc_{i}" for i in range(num_vectors)]
    
    # 创建压缩索引
    compressed_index = CompressedVectorIndex(
        dimension=dimension,
        compression_config={
            'method': 'pq',
            'num_subvectors': 8,
            'num_centroids': 256
        }
    )
    
    # 训练并添加向量
    print("\n1. 训练压缩器...")
    compressed_index.train_compressor(vectors[:1000])  # 使用部分数据训练
    
    print("\n2. 添加压缩向量...")
    compressed_index.add_vectors(vectors, vector_ids)
    
    # 显示压缩统计
    print("\n3. 压缩效果统计:")
    stats = compressed_index.get_compression_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # 测试搜索
    print("\n4. 搜索测试...")
    query_vector = np.random.randn(dimension).astype(np.float32)
    results = compressed_index.search(query_vector, top_k=5)
    
    print("Top-5 搜索结果:")
    for i, (vector_id, distance) in enumerate(results):
        print(f"  {i+1}. {vector_id}: 距离 = {distance:.4f}")

if __name__ == "__main__":
    demonstrate_vector_compression()