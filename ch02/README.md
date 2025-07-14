# 深度RAG笔记02 - 数据索引代码实现

本目录包含深度RAG笔记第02篇《数据索引阶段深度解析》的完整代码实现。

## 📁 文件结构

```
code/ch02/
├── README.md                    # 代码说明文档
├── semantic_chunker.py         # 智能分割模块
├── embedding_optimizer.py      # 向量嵌入优化模块
└── hnsw_index.py               # HNSW向量索引实现
```

## 🚀 快速开始

### 1. 环境准备

```bash
pip install numpy pandas scikit-learn sentence-transformers
pip install paddleocr python-docx beautifulsoup4 psutil
```

### 2. 基本使用

```python
from semantic_chunker import SemanticAwareChunker
from embedding_optimizer import HighPerformanceEmbedding
from quality_monitor import IndexQualityMonitor

# 1. 智能分割
text = "你的文档内容"  # 假设已有文档文本
chunker = SemanticAwareChunker(target_size=512)
chunks = chunker.intelligent_chunking(text)

# 2. 向量嵌入
embedder = HighPerformanceEmbedding()
embeddings = embedder.batch_encode_with_optimization(chunks)

# 3. 质量监控
monitor = IndexQualityMonitor()
quality = monitor.evaluate_quality([text], chunks)
print(f"质量评级: {quality['quality_grade']}")
```

## 📋 模块说明

### semantic_chunker.py
- **功能**: 基于语义的智能文档分割
- **核心算法**: 语义相似度分析、边界识别、重叠处理
- **支持策略**: 固定长度、语义感知、滑动窗口

### embedding_optimizer.py
- **功能**: 高性能向量嵌入计算和质量验证
- **优化特性**: 批量处理、缓存机制、去重优化
- **质量评估**: 语义准确性、技术指标、综合评分

### hnsw_index.py
- **功能**: HNSW算法的完整实现
- **核心特性**: 多层图结构、启发式邻居选择、动态插入
- **性能优势**: O(log N)搜索复杂度、高召回率

### quality_monitor.py
- **功能**: 全面的索引质量监控和性能分析
- **监控维度**: 信息完整性、语义一致性、检索精度
- **报告功能**: 质量等级评定、优化建议生成

## 🎯 使用场景

### 场景1: 企业文档知识库构建
```python
# 处理企业内部文档
chunker = SemanticAwareChunker(target_size=1024, overlap_ratio=0.1)

for text in document_texts:  # 假设已有文档文本
    chunks = chunker.intelligent_chunking(text)
    # 后续处理...
```

### 场景2: 大规模文档批量处理
```python
# 高性能批量处理
embedder = HighPerformanceEmbedding("BAAI/bge-large-zh")
all_chunks = []  # 收集所有文档片段

# 批量计算嵌入，自动优化性能
embeddings = embedder.batch_encode_with_optimization(all_chunks, batch_size=64)
```

### 场景3: 质量监控和优化
```python
# 持续质量监控
monitor = IndexQualityMonitor()
quality_result = monitor.evaluate_quality(original_docs, chunks)

if quality_result['overall_score'] < 0.8:
    print("质量不达标，需要优化:")
    for recommendation in quality_result['recommendations']:
        print(f"- {recommendation}")
```

## 📊 性能基准

在测试环境下的性能表现：

| 指标 | 性能 | 说明 |
|------|------|------|
| 文档处理速度 | 100+ docs/min | PDF/DOCX混合文档 |
| 嵌入计算速度 | 1000+ chunks/min | BGE-large-zh模型 |
| HNSW搜索延迟 | <5ms | 100万向量规模 |
| 内存占用 | 原数据1.2倍 | 包含索引结构 |

## ⚙️ 配置选项

### 分割配置
```python
chunker = SemanticAwareChunker(
    target_size=512,           # 目标片段长度
    overlap_ratio=0.1,         # 重叠比例
    similarity_threshold=0.7   # 语义相似度阈值
)
```

### 嵌入配置
```python
embedder = HighPerformanceEmbedding(
    model_name="BAAI/bge-large-zh",  # 嵌入模型
    device="cuda",                   # 计算设备
    batch_size=64                    # 批处理大小
)
```

### HNSW配置
```python
index = HNSWIndex(
    dimension=768,           # 向量维度
    max_m=16,               # 最大连接数
    ef_construction=200     # 构建时搜索宽度
)
```

## 🔧 自定义扩展

### 实现自定义分割策略
```python
class CustomChunker(SemanticAwareChunker):
    def domain_specific_chunking(self, text):
        # 实现领域特定的分割逻辑
        pass
```

### 集成其他向量数据库
```python
class PineconeIntegration:
    def store_vectors(self, embeddings, metadata):
        # 实现Pinecone存储逻辑
        pass
```

## 📈 监控和调优

### 性能监控
```python
from quality_monitor import PerformanceMonitor

monitor = PerformanceMonitor()
metrics = monitor.collect_metrics(processing_stats)
report = monitor.generate_performance_report()
```

### 质量调优
1. **分割策略调优**: 根据文档类型调整target_size和overlap_ratio
2. **嵌入模型选择**: 根据语言和领域选择合适的预训练模型
3. **索引参数优化**: 根据数据规模和精度要求调整HNSW参数

## 🔗 相关资源

- [深度RAG笔记01: 核心概念与诞生背景](../ch01/)
- [深度RAG笔记03: 智能检索核心技术](../ch03/)
- [BGE嵌入模型](https://huggingface.co/BAAI/bge-large-zh)
- [HNSW论文](https://arxiv.org/abs/1603.09320)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进代码实现。

## 📄 许可证

本代码遵循MIT许可证。