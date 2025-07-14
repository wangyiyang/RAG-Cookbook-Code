"""
高性能向量嵌入模块
实现向量嵌入计算、优化和质量验证功能
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
import logging


class HighPerformanceEmbedding:
    """高性能嵌入计算器"""
    
    def __init__(self, model_name: str = "BAAI/bge-large-zh", device: str = "cuda"):
        """
        初始化高性能嵌入计算器
        
        Args:
            model_name: 嵌入模型名称
            device: 计算设备 ('cuda' 或 'cpu')
        """
        # 选择优秀的中文嵌入模型，使用GPU加速
        self.model = SentenceTransformer(model_name, device=device)
        self.cache = {}  # 内存缓存，避免重复计算
        self.model_name = model_name
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def batch_encode_with_optimization(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        企业级高性能批量编码实现
        
        Args:
            texts: 待编码的文本列表
            batch_size: 批量大小，根据GPU内存调整
            
        Returns:
            编码后的向量数组
        """
        start_time = time.time()
        
        # 步骤1：智能去重，减少计算量
        unique_texts = list(set(texts))
        text_to_index = {text: i for i, text in enumerate(unique_texts)}
        
        self.logger.info(f"去重前: {len(texts)}个文本, 去重后: {len(unique_texts)}个文本")
        
        # 步骤2：缓存检查，跳过已计算的文本
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(unique_texts):
            text_hash = self._get_text_hash(text)  # 使用哈希快速匹配
            if text_hash not in self.cache:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        self.logger.info(f"缓存命中: {len(unique_texts) - len(uncached_texts)}个文本, 需要计算: {len(uncached_texts)}个文本")
        
        # 步骤3：批量计算新文本的嵌入向量
        if uncached_texts:
            new_embeddings = self.model.encode(
                uncached_texts, 
                batch_size=batch_size,           # 批量大小根据GPU内存调整
                show_progress_bar=True,          # 显示处理进度
                normalize_embeddings=True        # 归一化向量，提升检索精度
            )
            
            # 将新计算的结果存入缓存
            for text, embedding in zip(uncached_texts, new_embeddings):
                text_hash = self._get_text_hash(text)
                self.cache[text_hash] = embedding
        
        # 步骤4：组装最终结果，保持原始顺序
        all_embeddings = []
        for text in texts:
            text_hash = self._get_text_hash(text)
            all_embeddings.append(self.cache[text_hash])
        
        processing_time = time.time() - start_time
        self.logger.info(f"批量编码完成，耗时: {processing_time:.2f}秒")
        
        return np.array(all_embeddings)
    
    def _get_text_hash(self, text: str) -> str:
        """生成文本的哈希值用于缓存"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        return {
            'cache_size': len(self.cache),
            'model_name': self.model_name,
            'total_cached_vectors': len(self.cache)
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.logger.info("缓存已清空")


class EmbeddingQualityValidator:
    """嵌入质量验证器"""
    
    def __init__(self, embedding_model: HighPerformanceEmbedding):
        """
        初始化质量验证器
        
        Args:
            embedding_model: 嵌入模型实例
        """
        self.embedding_model = embedding_model
    
    def validate_embedding_quality(self, embeddings: np.ndarray, texts: List[str]) -> Dict:
        """
        全面的嵌入质量评估体系
        
        Args:
            embeddings: 嵌入向量数组
            texts: 对应的文本列表
            
        Returns:
            质量评估结果
        """
        # 测试用例：精心设计的语义对比组
        test_pairs = [
            ("机器学习", "人工智能"),  # 相关概念，期望相似度 > 0.7
            ("机器学习", "苹果"),      # 无关概念，期望相似度 < 0.3
            ("深度学习", "神经网络"),  # 相关概念
            ("汽车", "飞机"),          # 无关概念
        ]
        
        similarity_scores = []
        for text1, text2 in test_pairs:
            emb1 = self.embedding_model.batch_encode_with_optimization([text1])[0]
            emb2 = self.embedding_model.batch_encode_with_optimization([text2])[0]
            # 计算余弦相似度，范围[-1, 1]，越接近1越相似
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            similarity_scores.append(similarity)
        
        # 计算向量质量指标
        vector_norms = [np.linalg.norm(emb) for emb in embeddings]
        
        # 综合质量指标计算
        quality_metrics = {
            # 语义准确性指标
            'semantic_precision_related': np.mean([similarity_scores[0], similarity_scores[2]]),  # 相关概念平均相似度
            'semantic_discrimination': 1 - np.mean([similarity_scores[1], similarity_scores[3]]),  # 无关概念区分度
            
            # 向量技术指标
            'vector_dimension': len(embeddings[0]) if len(embeddings) > 0 else 0,
            'vector_norm_mean': np.mean(vector_norms),
            'vector_norm_std': np.std(vector_norms),
            
            # 整体质量评分
            'overall_score': 0.0
        }
        
        # 计算整体质量分数
        semantic_score = (quality_metrics['semantic_precision_related'] + quality_metrics['semantic_discrimination']) / 2
        technical_score = 1.0 if 0.8 <= quality_metrics['vector_norm_mean'] <= 1.2 else 0.5
        
        quality_metrics['overall_score'] = (semantic_score + technical_score) / 2
        quality_metrics['quality_grade'] = self._get_quality_grade(quality_metrics['overall_score'])
        
        return quality_metrics
    
    def _get_quality_grade(self, score: float) -> str:
        """根据分数获取质量等级"""
        if score >= 0.9:
            return "A级优秀"
        elif score >= 0.8:
            return "B级良好"
        elif score >= 0.7:
            return "C级需改进"
        else:
            return "D级不合格"
    
    def generate_quality_report(self, quality_metrics: Dict) -> str:
        """生成质量报告"""
        report = f"""
=== 嵌入质量评估报告 ===

语义准确性:
- 相关概念相似度: {quality_metrics['semantic_precision_related']:.3f} (期望 > 0.7)
- 无关概念区分度: {quality_metrics['semantic_discrimination']:.3f} (期望 > 0.7)

技术指标:
- 向量维度: {quality_metrics['vector_dimension']}
- 向量长度均值: {quality_metrics['vector_norm_mean']:.3f}
- 向量长度标准差: {quality_metrics['vector_norm_std']:.3f}

综合评估:
- 整体质量分数: {quality_metrics['overall_score']:.3f}
- 质量等级: {quality_metrics['quality_grade']}

建议:
"""
        
        # 根据指标给出建议
        if quality_metrics['semantic_precision_related'] < 0.7:
            report += "- 相关概念相似度偏低，建议更换更适合的嵌入模型\n"
        
        if quality_metrics['semantic_discrimination'] < 0.7:
            report += "- 无关概念区分度不足，模型可能存在过拟合问题\n"
        
        if quality_metrics['vector_norm_std'] > 0.3:
            report += "- 向量长度方差较大，建议启用向量归一化\n"
        
        if quality_metrics['overall_score'] >= 0.8:
            report += "- 整体质量良好，可以用于生产环境\n"
        
        return report


def benchmark_embedding_models(texts: List[str], models: List[str]) -> Dict:
    """
    对比不同嵌入模型的性能
    
    Args:
        texts: 测试文本列表
        models: 待测试的模型列表
        
    Returns:
        性能对比结果
    """
    results = {}
    
    for model_name in models:
        print(f"测试模型: {model_name}")
        
        try:
            # 初始化模型
            embedding_model = HighPerformanceEmbedding(model_name)
            validator = EmbeddingQualityValidator(embedding_model)
            
            # 计算嵌入
            start_time = time.time()
            embeddings = embedding_model.batch_encode_with_optimization(texts)
            encoding_time = time.time() - start_time
            
            # 质量评估
            quality_metrics = validator.validate_embedding_quality(embeddings, texts)
            
            results[model_name] = {
                'encoding_time': encoding_time,
                'quality_score': quality_metrics['overall_score'],
                'quality_grade': quality_metrics['quality_grade'],
                'vector_dimension': quality_metrics['vector_dimension']
            }
            
        except Exception as e:
            print(f"模型 {model_name} 测试失败: {e}")
            results[model_name] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # 使用示例
    sample_texts = [
        "机器学习是人工智能的重要分支",
        "深度学习使用神经网络进行训练",
        "自然语言处理帮助计算机理解文本",
        "计算机视觉让机器能够看懂图像",
        "强化学习通过奖励机制学习策略"
    ]
    
    # 创建高性能嵌入器
    embedding_model = HighPerformanceEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    
    # 批量编码
    embeddings = embedding_model.batch_encode_with_optimization(sample_texts)
    print(f"生成嵌入向量: {embeddings.shape}")
    
    # 质量验证
    validator = EmbeddingQualityValidator(embedding_model)
    quality_metrics = validator.validate_embedding_quality(embeddings, sample_texts)
    
    # 生成报告
    report = validator.generate_quality_report(quality_metrics)
    print(report)
    
    # 缓存统计
    cache_stats = embedding_model.get_cache_stats()
    print(f"缓存统计: {cache_stats}")