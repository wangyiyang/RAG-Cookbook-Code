# RAG框架混合架构实现

深度RAG笔记10配套代码：LangChain + LlamaIndex 强强联合实战

## 文件结构

```
ch10/
├── README.md                          # 说明文档
├── langchain_llamaindex_hybrid.py     # 混合架构核心实现
├── llama_packs_integration.py         # Llama Packs集成示例
├── hybrid_performance_optimizer.py    # 性能优化模块
├── customer_service_raptor_example.py # 客服系统RAPTOR改造实例
└── demo.py                           # 完整使用示例
```

## 核心功能

### 1. 混合架构设计
- **LangChain**: 负责工作流控制、链式组合、Agent协调
- **LlamaIndex**: 负责文档索引、智能检索、查询优化
- **协同优势**: 发挥各自技术优势，实现1+1>2的效果

### 2. Llama Packs生态利用
- Agent Search Retriever: 智能搜索代理
- Fusion Retriever: 多路检索融合
- Corrective RAG: 自动纠错机制
- Self RAG: 自我反思优化
- **RAPTOR**: 递归抽象压缩文档树 (客服项目实证：74%→91%准确率)

### 3. 性能优化策略
- 缓存机制优化
- 并行检索处理
- 动态参数调整
- 实时监控反馈

## 安装依赖

```bash
pip install langchain llamaindex openai chromadb
pip install llama-index-packs-agent-search-retriever
pip install llama-index-packs-fusion-retriever
pip install llama-index-packs-raptor
```

## 快速开始

### 基础混合架构
```python
from demo import HybridRAGDemo

# 初始化混合RAG系统
demo = HybridRAGDemo()
demo.build_knowledge_base("./documents/")

# 智能问答
result = demo.smart_query("什么是深度学习？")
print(f"答案: {result['answer']}")
print(f"置信度: {result['confidence']}")
```

### 客服系统RAPTOR实例
```python
from customer_service_raptor_example import CustomerServiceRAG

# 基于真实项目：74%→91%准确率提升
cs_rag = CustomerServiceRAG()
cs_rag.load_customer_service_kb("./customer_service_kb/")

# 客服查询
result = cs_rag.answer_customer_query("如何申请退款？")
print(f"客服回答: {result['answer']}")
print(f"RAPTOR层数: {result['raptor_info']['layers_searched']}")
```

## 技术特点

1. **架构灵活**: 支持不同场景的动态切换
2. **性能优异**: 结合两个框架的技术优势
3. **实战验证**: 客服项目真实数据，准确率提升17%
4. **易于扩展**: 模块化设计，便于功能扩展
5. **生产就绪**: 包含完整的监控和优化机制

## 注意事项

- 确保OpenAI API密钥已正确配置
- 建议使用Python 3.8+版本
- 首次运行需要下载模型，请确保网络连接正常