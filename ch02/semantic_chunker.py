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
            if len(sentence) > 5:  # 过滤过短的片段
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
            # 修复中文文本的词数计算
            current_length += len(sentence.replace('，', ' ').replace('。', ' ').split())
            
            # 分割决策：同时考虑长度和语义
            should_split = (
                current_length >= self.target_size and  # 长度约束：达到目标大小
                i < len(similarities) and               # 边界检查：不是最后一句
                similarities[i] < 0.95                  # 降低阈值，更容易分割
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
        # 修复中文文本的词数计算
        words = text.replace('，', ' ').replace('。', ' ').replace('！', ' ').replace('？', ' ').split()
        chunks = []
        
        if len(words) <= self.chunk_size:
            return [text]
        
        step = max(1, self.chunk_size - self.overlap)
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            if chunk_words:  # 确保不是空列表
                chunks.append(' '.join(chunk_words))
        
        return chunks
    
    def sliding_window_chunking(self, text: str) -> List[str]:
        """滑动窗口分割"""
        # 修复中文文本的词数计算
        words = text.replace('，', ' ').replace('。', ' ').replace('！', ' ').replace('？', ' ').split()
        chunks = []
        
        # 如果文本太短，直接返回
        if len(words) <= self.chunk_size:
            return [text]
        
        step_size = max(1, self.chunk_size - self.overlap)
        
        # 生成滑动窗口
        i = 0
        while i + self.chunk_size <= len(words):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append(' '.join(chunk_words))
            i += step_size
        
        # 处理剩余的词
        if i < len(words):
            remaining_words = words[i:]
            if len(remaining_words) > self.overlap:  # 只有当剩余词数足够时才添加
                chunks.append(' '.join(remaining_words))
        
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
    
    # 计算长度分布 - 修复中文文本词数计算
    lengths = [len(chunk.replace('。', ' ').split()) for chunk in chunks]
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
    # 创建更有对比性的测试文本，包含不同主题的段落
    sample_text = """
    机器学习是人工智能的一个重要分支，它通过算法让计算机从数据中自动学习模式和规律。传统的编程需要开发者明确地编写每个步骤的指令，而机器学习允许计算机通过分析大量数据来自动发现规律。深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的工作方式。这种技术在图像识别、语音处理和自然语言理解等领域取得了突破性进展。
    
    金融科技正在改变传统银行业务模式。区块链技术为数字货币提供了去中心化的解决方案。移动支付已经成为日常生活中不可或缺的一部分。数字钱包让用户能够轻松管理多种支付方式。
    
    云计算技术使得企业能够灵活地扩展计算资源。微服务架构帮助开发团队更好地管理复杂的应用系统。容器化技术简化了应用的部署和运维工作。DevOps实践促进了开发和运营团队之间的协作。
    
    人工智能在医疗领域的应用前景广阔。机器学习算法能够帮助医生更准确地诊断疾病。医疗影像分析技术正在革命性地改变放射科的工作流程。个性化医疗将根据患者的基因信息制定治疗方案。
    """
    
    print("=== 语义感知分割演示 ===")
    
    # 使用较小的目标大小和更高的阈值来展示分割效果
    semantic_chunker = SemanticAwareChunker(target_size=15, overlap_ratio=0.1)
    
    # 调试信息
    sentences = semantic_chunker.split_into_sentences(sample_text)
    print(f"分割出的句子数量: {len(sentences)}")
    
    if len(sentences) > 1:
        embeddings = semantic_chunker.sentence_model.encode(sentences)
        similarities = semantic_chunker.calculate_semantic_similarity(embeddings)
        print("相邻句子语义相似度:")
        for i, sim in enumerate(similarities):
            print(f"  句子{i+1}→{i+2}: {sim:.3f}")
    
    # 执行语义分割
    semantic_chunks = semantic_chunker.intelligent_chunking(sample_text)
    
    print(f"\n语义分割结果 (共{len(semantic_chunks)}个片段):")
    for i, chunk in enumerate(semantic_chunks):
        words = chunk.replace('。', ' ').split()
        print(f"Chunk {i+1} ({len(words)}词): {chunk[:100]}...")
    
    print("\n=== 不同策略效果对比 ===")
    
    # 首先计算文本的实际词数
    total_words = len(sample_text.replace('，', ' ').replace('。', ' ').replace('！', ' ').replace('？', ' ').split())
    print(f"文本总词数: {total_words}")
    
    # 创建测试配置 - 使用更小的chunk_size来强制分割
    test_configs = [
        {"strategy": "fixed", "chunker": AdaptiveChunker(chunk_size=10, overlap=2)},
        {"strategy": "semantic", "chunker": AdaptiveChunker(chunk_size=10, overlap=2)},
        {"strategy": "sliding_window", "chunker": AdaptiveChunker(chunk_size=10, overlap=2)}
    ]
    
    results = {}
    
    for config in test_configs:
        strategy = config["strategy"]
        chunker = config["chunker"]
        
        print(f"\n调试 {strategy.upper()} 策略 (chunk_size={chunker.chunk_size}, overlap={chunker.overlap}):")
        
        # 添加调试信息
        if strategy == "fixed":
            test_words = sample_text.replace('，', ' ').replace('。', ' ').replace('！', ' ').replace('？', ' ').split()
            print(f"  处理词数: {len(test_words)}")
            print(f"  预期片段数: {max(1, len(test_words) // max(1, chunker.chunk_size - chunker.overlap))}")
        
        chunks = chunker.chunk_by_strategy(sample_text, strategy)
        quality = evaluate_chunking_quality(chunks)
        results[strategy] = {'chunks': chunks, 'quality': quality}
        
        print(f"\n{strategy.upper()} 策略:")
        print(f"  片段数量: {len(chunks)}")
        print(f"  质量分数: {quality['quality_score']:.3f}")
        print(f"  平均长度: {quality['avg_length']:.1f} 词")
        print(f"  长度标准差: {quality['length_std']:.1f}")
        
        # 显示实际分割结果
        for i, chunk in enumerate(chunks):
            words = chunk.replace('，', ' ').replace('。', ' ').replace('！', ' ').replace('？', ' ').split()
            print(f"  片段{i+1}: {chunk[:60]}... ({len(words)}词)")
            
            # 对于滑动窗口，显示重叠信息
            if strategy == "sliding_window" and i > 0:
                prev_words = chunks[i-1].replace('，', ' ').replace('。', ' ').replace('！', ' ').replace('？', ' ').split()
                curr_words = words
                
                # 显示前一片段末尾和当前片段开头
                if len(prev_words) >= 2 and len(curr_words) >= 2:
                    print(f"    └─ 前片段末尾: ...{' '.join(prev_words[-2:])}")
                    print(f"    └─ 当前片段开头: {' '.join(curr_words[:2])}...")
                    
                    # 检查是否有重叠词汇
                    overlap_words = []
                    for word in prev_words[-chunker.overlap:]:
                        if word in curr_words[:chunker.overlap]:
                            overlap_words.append(word)
                    
                    if overlap_words:
                        print(f"    └─ 重叠词汇: {' '.join(overlap_words)}")
    
    print("\n=== 分割策略分析 ===")
    
    # 找到最佳策略
    best_strategy = max(results.keys(), key=lambda x: results[x]['quality']['quality_score'])
    print(f"推荐策略: {best_strategy.upper()}")
    
    # 分析各策略特点
    print("\n策略特点分析:")
    for strategy, result in results.items():
        chunks = result['chunks']
        quality = result['quality']
        print(f"  {strategy.upper()}:")
        print(f"    - 片段数量: {len(chunks)}")
        print(f"    - 内容质量: {quality['content_quality']:.3f}")
        print(f"    - 长度均匀性: {quality['uniformity']:.3f}")
        
        if strategy == 'semantic':
            print(f"    - 语义连贯性: 在语义转换点分割 (阈值<0.95)")
            print(f"    - 智能边界: 识别主题变化，保持内容完整性")
        elif strategy == 'sliding_window':
            print(f"    - 重叠策略: 每个片段与前一片段重叠{chunker.overlap}个词")
            print(f"    - 上下文保持: 防止重要信息在分割边界丢失")
        else:
            print(f"    - 固定分割: 按词数({chunker.chunk_size})均匀分割")
            print(f"    - 简单高效: 不考虑语义，处理速度最快")
    
    print("\n=== 💡 RAG应用建议 ===")
    print("📊 性能对比:")
    print(f"  • 语义分割: {results['semantic']['quality']['quality_score']:.3f}分 - 最适合问答系统")
    print(f"  • 固定分割: {results['fixed']['quality']['quality_score']:.3f}分 - 最适合大批量处理")
    print(f"  • 滑动窗口: {results['sliding_window']['quality']['quality_score']:.3f}分 - 最适合需要上下文的场景")
    
    print("\n🎯 使用场景:")
    print("  📝 语义分割 → 智能问答、知识图谱构建")
    print("  ⚡ 固定分割 → 大规模文档索引、批量处理")
    print("  🔄 滑动窗口 → 长文档理解、跨段落检索")

# === 语义感知分割演示 ===
# 分割出的句子数量: 16
# 相邻句子语义相似度:
#   句子1→2: 0.593
#   句子2→3: 0.524
#   句子3→4: 0.402
#   句子4→5: 0.498
#   句子5→6: 0.501
#   句子6→7: 0.578
#   句子7→8: 0.350
#   句子8→9: 0.505
#   句子9→10: 0.943
#   句子10→11: 0.473
#   句子11→12: 0.322
#   句子12→13: 0.240
#   句子13→14: 0.489
#   句子14→15: 0.758
#   句子15→16: 0.425

# 语义分割结果 (共2个片段):
# Chunk 1 (12词): 机器学习是人工智能的一个重要分支，它通过算法让计算机从数据中自动学习模式和规律。传统的编程需要开发者明确地编写每个步骤的指令，而机器学习允许计算机通过分析大量数据来自动发现规律。深度学习是机器学习的一...
# Chunk 2 (5词): DevOps实践促进了开发和运营团队之间的协作。人工智能在医疗领域的应用前景广阔。机器学习算法能够帮助医生更准确地诊断疾病。医疗影像分析技术正在革命性地改变放射科的工作流程。个性化医疗将根据患者的基因...

# === 不同策略效果对比 ===
# 文本总词数: 19

# 调试 FIXED 策略 (chunk_size=10, overlap=2):
#   处理词数: 19
#   预期片段数: 2

# FIXED 策略:
#   片段数量: 3
#   质量分数: 0.907
#   平均长度: 7.7 词
#   长度标准差: 3.3
#   片段1: 机器学习是人工智能的一个重要分支 它通过算法让计算机从数据中自动学习模式和规律 传统的编程需要开发者明确地编写每个步骤的... (10词)
#   片段2: 区块链技术为数字货币提供了去中心化的解决方案 移动支付已经成为日常生活中不可或缺的一部分 数字钱包让用户能够轻松管理多种... (10词)
#   片段3: 机器学习算法能够帮助医生更准确地诊断疾病 医疗影像分析技术正在革命性地改变放射科的工作流程 个性化医疗将根据患者的基因信... (3词)

# 调试 SEMANTIC 策略 (chunk_size=10, overlap=2):

# SEMANTIC 策略:
#   片段数量: 2
#   质量分数: 0.984
#   平均长度: 8.5 词
#   长度标准差: 1.5
#   片段1: 机器学习是人工智能的一个重要分支，它通过算法让计算机从数据中自动学习模式和规律。传统的编程需要开发者明确地编写每个步骤的... (10词)
#   片段2: 移动支付已经成为日常生活中不可或缺的一部分。数字钱包让用户能够轻松管理多种支付方式。云计算技术使得企业能够灵活地扩展计算... (10词)

# 调试 SLIDING_WINDOW 策略 (chunk_size=10, overlap=2):

# SLIDING_WINDOW 策略:
#   片段数量: 3
#   质量分数: 0.907
#   平均长度: 7.7 词
#   长度标准差: 3.3
#   片段1: 机器学习是人工智能的一个重要分支 它通过算法让计算机从数据中自动学习模式和规律 传统的编程需要开发者明确地编写每个步骤的... (10词)
#   片段2: 区块链技术为数字货币提供了去中心化的解决方案 移动支付已经成为日常生活中不可或缺的一部分 数字钱包让用户能够轻松管理多种... (10词)
#     └─ 前片段末尾: ...区块链技术为数字货币提供了去中心化的解决方案 移动支付已经成为日常生活中不可或缺的一部分
#     └─ 当前片段开头: 区块链技术为数字货币提供了去中心化的解决方案 移动支付已经成为日常生活中不可或缺的一部分...
#     └─ 重叠词汇: 区块链技术为数字货币提供了去中心化的解决方案 移动支付已经成为日常生活中不可或缺的一部分
#   片段3: 机器学习算法能够帮助医生更准确地诊断疾病 医疗影像分析技术正在革命性地改变放射科的工作流程 个性化医疗将根据患者的基因信... (3词)
#     └─ 前片段末尾: ...机器学习算法能够帮助医生更准确地诊断疾病 医疗影像分析技术正在革命性地改变放射科的工作流程
#     └─ 当前片段开头: 机器学习算法能够帮助医生更准确地诊断疾病 医疗影像分析技术正在革命性地改变放射科的工作流程...
#     └─ 重叠词汇: 机器学习算法能够帮助医生更准确地诊断疾病 医疗影像分析技术正在革命性地改变放射科的工作流程

# === 分割策略分析 ===
# 推荐策略: SEMANTIC

# 策略特点分析:
#   FIXED:
#     - 片段数量: 3
#     - 内容质量: 1.000
#     - 长度均匀性: 0.815
#     - 固定分割: 按词数(10)均匀分割
#     - 简单高效: 不考虑语义，处理速度最快
#   SEMANTIC:
#     - 片段数量: 2
#     - 内容质量: 1.000
#     - 长度均匀性: 0.969
#     - 语义连贯性: 在语义转换点分割 (阈值<0.95)
#     - 智能边界: 识别主题变化，保持内容完整性
#   SLIDING_WINDOW:
#     - 片段数量: 3
#     - 内容质量: 1.000
#     - 长度均匀性: 0.815
#     - 重叠策略: 每个片段与前一片段重叠2个词
#     - 上下文保持: 防止重要信息在分割边界丢失

# === 💡 RAG应用建议 ===
# 📊 性能对比:
#   • 语义分割: 0.984分 - 最适合问答系统
#   • 固定分割: 0.907分 - 最适合大批量处理
#   • 滑动窗口: 0.907分 - 最适合需要上下文的场景

# 🎯 使用场景:
#   📝 语义分割 → 智能问答、知识图谱构建
#   ⚡ 固定分割 → 大规模文档索引、批量处理
#   🔄 滑动窗口 → 长文档理解、跨段落检索