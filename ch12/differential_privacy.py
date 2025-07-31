"""
差分隐私RAG实现
Deep RAG Notes Chapter 12 - Privacy Protection Technologies
"""

import numpy as np
from typing import List, Dict, Any, Optional
import hashlib
import logging

class DifferentialPrivacyRAG:
    """差分隐私RAG系统"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        初始化差分隐私RAG
        
        Args:
            epsilon: 隐私预算，越小越安全
            delta: 失败概率
        """
        self.epsilon = epsilon
        self.delta = delta
        self.noise_scale = self.calculate_noise_scale()
        self.query_count = 0
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def calculate_noise_scale(self) -> float:
        """计算噪声规模"""
        # 基于高斯机制的噪声规模
        sensitivity = 1.0  # L2敏感度
        return sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def add_noise_to_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """为向量嵌入添加差分隐私噪声"""
        if embeddings.size == 0:
            return embeddings
            
        noise = np.random.normal(
            loc=0.0, 
            scale=self.noise_scale, 
            size=embeddings.shape
        )
        
        # 添加噪声并归一化
        noisy_embeddings = embeddings + noise
        
        # L2归一化，保持向量空间的几何性质
        norms = np.linalg.norm(noisy_embeddings, axis=1, keepdims=True)
        normalized_embeddings = noisy_embeddings / (norms + 1e-8)
        
        self.logger.info(f"Added noise with scale {self.noise_scale:.4f}")
        return normalized_embeddings
    
    def private_similarity_search(self, 
                                 query_embedding: np.ndarray, 
                                 doc_embeddings: np.ndarray, 
                                 top_k: int = 5) -> List[Dict[str, Any]]:
        """差分隐私的相似度搜索"""
        self.query_count += 1
        
        if doc_embeddings.size == 0:
            return []
            
        # 1. 为查询向量添加噪声
        noisy_query = self.add_noise_to_embeddings(
            query_embedding.reshape(1, -1)
        )[0]
        
        # 2. 为文档向量添加噪声
        noisy_docs = self.add_noise_to_embeddings(doc_embeddings)
        
        # 3. 计算噪声化的相似度
        similarities = np.dot(noisy_docs, noisy_query)
        
        # 4. 添加指数机制噪声进行排序
        sensitivity = 1.0
        exponential_weights = np.exp(
            self.epsilon * similarities / (2 * sensitivity)
        )
        
        # 5. 概率采样返回top-k结果
        probabilities = exponential_weights / np.sum(exponential_weights)
        selected_indices = np.random.choice(
            len(doc_embeddings), 
            size=min(top_k, len(doc_embeddings)),
            replace=False,
            p=probabilities
        )
        
        results = []
        for idx in selected_indices:
            results.append({
                'document_id': int(idx),
                'similarity_score': float(similarities[idx]),
                'privacy_noise_level': self.noise_scale,
                'query_count': self.query_count
            })
        
        return sorted(results, key=lambda x: x['similarity_score'], reverse=True)
    
    def get_privacy_metrics(self) -> Dict[str, Any]:
        """获取隐私保护指标"""
        return {
            'epsilon': self.epsilon,
            'delta': self.delta,
            'noise_scale': self.noise_scale,
            'query_count': self.query_count,
            'privacy_loss': self.query_count * self.epsilon
        }


class PrivacyBudgetManager:
    """隐私预算管理器"""
    
    def __init__(self, total_budget: float = 10.0):
        """
        初始化预算管理器
        
        Args:
            total_budget: 总隐私预算
        """
        self.total_budget = total_budget
        self.used_budget = 0.0
        self.query_history = []
        self.budget_allocation_strategy = 'adaptive'
        
    def allocate_budget_for_query(self, query_metadata: Dict[str, Any]) -> float:
        """为查询分配隐私预算"""
        remaining_budget = self.total_budget - self.used_budget
        
        if remaining_budget <= 0:
            return 0.0  # 预算用完，拒绝查询
        
        # 基于查询特征动态分配预算
        base_allocation = self.calculate_base_allocation(query_metadata)
        
        # 根据剩余预算调整
        if self.budget_allocation_strategy == 'exponential_decay':
            # 指数衰减策略：早期查询获得更多预算
            decay_factor = np.exp(-len(self.query_history) / 100)
            allocated_budget = base_allocation * decay_factor
        elif self.budget_allocation_strategy == 'uniform':
            # 均匀分配策略
            expected_total_queries = query_metadata.get('expected_total_queries', 1000)
            remaining_queries = expected_total_queries - len(self.query_history)
            if remaining_queries > 0:
                allocated_budget = remaining_budget / remaining_queries
            else:
                allocated_budget = base_allocation
        else:
            # 自适应策略
            allocated_budget = min(base_allocation, remaining_budget * 0.1)
        
        # 确保不超过剩余预算
        allocated_budget = min(allocated_budget, remaining_budget)
        
        return allocated_budget
    
    def calculate_base_allocation(self, query_metadata: Dict[str, Any]) -> float:
        """计算基础预算分配"""
        base_budget = 0.1
        
        # 根据查询敏感度调整
        sensitivity_level = query_metadata.get('sensitivity_level', 'medium')
        sensitivity_multipliers = {'low': 0.5, 'medium': 1.0, 'high': 2.0, 'critical': 3.0}
        base_budget *= sensitivity_multipliers.get(sensitivity_level, 1.0)
        
        # 根据查询复杂度调整
        complexity = query_metadata.get('complexity', 'medium')
        complexity_multipliers = {'simple': 0.8, 'medium': 1.0, 'complex': 1.5}
        base_budget *= complexity_multipliers.get(complexity, 1.0)
        
        # 根据数据敏感性调整
        data_sensitivity = query_metadata.get('data_sensitivity', 'medium')
        data_multipliers = {'public': 0.5, 'internal': 1.0, 'confidential': 1.5, 'secret': 2.0}
        base_budget *= data_multipliers.get(data_sensitivity, 1.0)
        
        return base_budget
    
    def record_query_usage(self, allocated_budget: float, query_result: Dict[str, Any]):
        """记录查询的预算使用"""
        self.used_budget += allocated_budget
        self.query_history.append({
            'timestamp': np.datetime64('now').astype(str),
            'budget_used': allocated_budget,
            'query_success': query_result.get('success', False),
            'privacy_level_achieved': query_result.get('privacy_level', 'unknown'),
            'result_count': query_result.get('result_count', 0)
        })
    
    def get_budget_status(self) -> Dict[str, Any]:
        """获取预算使用状态"""
        return {
            'total_budget': self.total_budget,
            'used_budget': self.used_budget,
            'remaining_budget': self.total_budget - self.used_budget,
            'query_count': len(self.query_history),
            'budget_utilization': self.used_budget / self.total_budget * 100,
            'average_budget_per_query': self.used_budget / max(len(self.query_history), 1)
        }
    
    def reset_budget(self, new_budget: Optional[float] = None):
        """重置预算"""
        if new_budget is not None:
            self.total_budget = new_budget
        self.used_budget = 0.0
        self.query_history = []


def demo_differential_privacy():
    """差分隐私演示"""
    print("=== 差分隐私RAG演示 ===")
    
    # 创建模拟数据
    np.random.seed(42)
    doc_embeddings = np.random.randn(100, 768)  # 100个文档，768维向量
    query_embedding = np.random.randn(768)
    
    # 创建差分隐私RAG系统
    dp_rag = DifferentialPrivacyRAG(epsilon=1.0)
    
    # 创建预算管理器
    budget_manager = PrivacyBudgetManager(total_budget=10.0)
    
    # 查询元数据
    query_metadata = {
        'sensitivity_level': 'high',
        'complexity': 'medium',
        'data_sensitivity': 'confidential'
    }
    
    # 分配预算
    allocated_budget = budget_manager.allocate_budget_for_query(query_metadata)
    print(f"分配的隐私预算: {allocated_budget:.4f}")
    
    if allocated_budget > 0:
        # 设置系统的epsilon为分配的预算
        dp_rag.epsilon = allocated_budget
        dp_rag.noise_scale = dp_rag.calculate_noise_scale()
        
        # 执行隐私保护搜索
        results = dp_rag.private_similarity_search(
            query_embedding, 
            doc_embeddings, 
            top_k=5
        )
        
        # 记录预算使用
        query_result = {
            'success': True,
            'privacy_level': 'high',
            'result_count': len(results)
        }
        budget_manager.record_query_usage(allocated_budget, query_result)
        
        # 显示结果
        print(f"找到 {len(results)} 个结果:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. 文档ID: {result['document_id']}, "
                  f"相似度: {result['similarity_score']:.4f}")
        
        # 显示隐私指标
        privacy_metrics = dp_rag.get_privacy_metrics()
        print(f"\n隐私保护指标:")
        print(f"  Epsilon: {privacy_metrics['epsilon']:.4f}")
        print(f"  噪声水平: {privacy_metrics['noise_scale']:.4f}")
        print(f"  隐私损失: {privacy_metrics['privacy_loss']:.4f}")
        
        # 显示预算状态
        budget_status = budget_manager.get_budget_status()
        print(f"\n预算使用状态:")
        print(f"  总预算: {budget_status['total_budget']:.4f}")
        print(f"  已使用: {budget_status['used_budget']:.4f}")
        print(f"  剩余预算: {budget_status['remaining_budget']:.4f}")
        print(f"  使用率: {budget_status['budget_utilization']:.2f}%")
    else:
        print("预算不足，拒绝查询")


if __name__ == "__main__":
    demo_differential_privacy()