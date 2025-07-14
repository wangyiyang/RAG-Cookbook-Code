"""
语义感知分割模块
实现基于语义相似度的智能文档分割功能
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import re


class SemanticAwareChunker:
    """语义感知分割器"""
    
    def __init__(self, target_size: int = 512, overlap_ratio: float = 0.1):
        """
        初始化语义分割器
        
        Args:
            target_size: 目标chunk长度（词数）
            overlap_ratio: 重叠比例，防止信息丢失
        """
        self.target_size = target_size        # 目标chunk长度
        self.overlap_ratio = overlap_ratio    # 重叠比例，防止信息丢失
        # 使用轻量级语义模型，平衡效果和速度
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def intelligent_chunking(self, text: str) -> List[str]:
        """
        智能语义分割的完整流程
        
        Args:
            text: 待分割的文本
            
        Returns:
            分割后的文本片段列表
        """
        # 第1步：句子级别分割
        sentences = self.split_into_sentences(text)
        
        if len(sentences) <= 1:
            return [text]  # 文本太短，直接返回
        
        # 第2步：为每个句子生成语义向量
        embeddings = self.sentence_model.encode(sentences)
        
        # 第3步：计算相邻句子的语义相似度
        similarity_scores = self.calculate_semantic_similarity(embeddings)
        
        # 第4步：基于相似度动态确定分割边界
        chunks = self.find_semantic_boundaries(sentences, similarity_scores)
        
        return chunks
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        将文本分割为句子
        
        Args:
            text: 输入文本
            
        Returns:
            句子列表
        """
        # 使用正则表达式进行句子分割（简化版）
        # 实际应用中可以使用更复杂的NLP库如spacy或nltk
        sentence_endings = r'[.!?。！？；;]'
        sentences = re.split(sentence_endings, text)
        
        # 清理和过滤句子
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # 过滤过短的片段
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def calculate_semantic_similarity(self, embeddings: np.ndarray) -> List[float]:
        """
        语义相似度计算：核心算法
        
        Args:
            embeddings: 句子嵌入向量数组
            
        Returns:
            相邻句子间的相似度列表
        """
        similarities = []
        for i in range(len(embeddings) - 1):
            # 使用余弦相似度衡量语义接近程度
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            similarities.append(sim)
        return similarities
    
    def find_semantic_boundaries(self, sentences: List[str], similarities: List[float]) -> List[str]:
        """
        智能边界识别：关键决策逻辑
        
        Args:
            sentences: 句子列表
            similarities: 相似度列表
            
        Returns:
            分割后的文本块列表
        """
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            current_length += len(sentence.split())
            
            # 分割决策：同时考虑长度和语义
            should_split = (
                current_length >= self.target_size and  # 长度约束：达到目标大小
                i < len(similarities) and               # 边界检查：不是最后一句
                similarities[i] < 0.7                   # 语义约束：相似度低于阈值
            )
            
            if should_split:
                chunks.append(self.join_sentences(current_chunk))
                
                # 重叠策略：保留部分内容，确保上下文连贯
                overlap_size = int(len(current_chunk) * self.overlap_ratio)
                current_chunk = current_chunk[-overlap_size:] if overlap_size > 0 else []
                current_length = sum(len(s.split()) for s in current_chunk)
        
        # 处理文档末尾的剩余内容
        if current_chunk:
            chunks.append(self.join_sentences(current_chunk))
        
        return chunks
    
    def join_sentences(self, sentences: List[str]) -> str:
        """
        将句子列表合并为完整文本
        
        Args:
            sentences: 句子列表
            
        Returns:
            合并后的文本
        """
        return '。'.join(sentences) + '。'


class AdaptiveChunker:
    """自适应分割器，支持多种分割策略"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        初始化自适应分割器
        
        Args:
            chunk_size: 目标chunk大小
            overlap: 重叠长度
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_strategy(self, text: str, strategy: str = 'semantic') -> List[str]:
        """
        根据策略进行文本分割
        
        Args:
            text: 待分割文本
            strategy: 分割策略 ('fixed', 'semantic', 'sliding_window')
            
        Returns:
            分割后的文本块列表
        """
        if strategy == 'semantic':
            chunker = SemanticAwareChunker(self.chunk_size, self.overlap / self.chunk_size)
            return chunker.intelligent_chunking(text)
        elif strategy == 'sliding_window':
            return self.sliding_window_chunking(text)
        else:
            return self.fixed_size_chunking(text)
    
    def fixed_size_chunking(self, text: str) -> List[str]:
        """固定长度分割"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append(' '.join(chunk_words))
        
        return chunks
    
    def sliding_window_chunking(self, text: str) -> List[str]:
        """滑动窗口分割"""
        words = text.split()
        chunks = []
        
        step_size = self.chunk_size - self.overlap
        
        for i in range(0, len(words) - self.chunk_size + 1, step_size):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append(' '.join(chunk_words))
        
        # 处理最后一个窗口
        if len(words) > self.chunk_size:
            last_chunk = words[-self.chunk_size:]
            chunks.append(' '.join(last_chunk))
        
        return chunks


def evaluate_chunking_quality(chunks: List[str]) -> dict:
    """
    评估分割质量
    
    Args:
        chunks: 文本块列表
        
    Returns:
        质量评估指标
    """
    if not chunks:
        return {'quality_score': 0.0, 'uniformity': 0.0, 'count': 0}
    
    # 计算长度分布
    lengths = [len(chunk.split()) for chunk in chunks]
    avg_length = sum(lengths) / len(lengths)
    
    # 计算长度方差（均匀性指标）
    length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
    uniformity = 1 - min(length_variance / (avg_length ** 2), 1.0)
    
    # 计算内容质量（简化指标）
    content_quality = sum(1 for chunk in chunks if len(chunk.strip()) > 50) / len(chunks)
    
    overall_quality = (uniformity + content_quality) / 2
    
    return {
        'quality_score': overall_quality,
        'uniformity': uniformity,
        'content_quality': content_quality,
        'avg_length': avg_length,
        'count': len(chunks),
        'length_std': length_variance ** 0.5
    }


if __name__ == "__main__":
    # 使用示例
    sample_text = """
    机器学习是人工智能的一个重要分支。它通过算法让计算机从数据中自动学习模式和规律。
    深度学习是机器学习的一个子集。它使用多层神经网络来模拟人脑的工作方式。
    自然语言处理是另一个重要领域。它专注于让计算机理解和生成人类语言。
    计算机视觉技术发展迅速。它能够让机器理解和分析图像内容。
    """
    
    # 语义感知分割
    semantic_chunker = SemanticAwareChunker(target_size=50)
    semantic_chunks = semantic_chunker.intelligent_chunking(sample_text)
    
    print("语义分割结果:")
    for i, chunk in enumerate(semantic_chunks):
        print(f"Chunk {i+1}: {chunk}")
    
    # 评估分割质量
    quality = evaluate_chunking_quality(semantic_chunks)
    print(f"\n分割质量评估: {quality}")
    
    # 自适应分割对比
    adaptive_chunker = AdaptiveChunker(chunk_size=30, overlap=5)
    
    print("\n不同策略对比:")
    for strategy in ['fixed', 'semantic', 'sliding_window']:
        chunks = adaptive_chunker.chunk_by_strategy(sample_text, strategy)
        quality = evaluate_chunking_quality(chunks)
        print(f"{strategy}: {len(chunks)}个片段, 质量分数: {quality['quality_score']:.3f}")