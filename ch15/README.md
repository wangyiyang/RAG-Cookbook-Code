# RAG+Agent融合系统 - 深度RAG笔记15

本章节代码配套第15篇文章 **《RAG+Agent融合实战，打造自主决策的智能问答系统》**。

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行完整演示

```bash
python demo.py
```

### 运行简化演示

```bash
python demo.py --simple
```

## 文件结构

- `graph_rag.py`: GraphRAG知识图谱增强检索系统
- `raptor_tree.py`: RAPTOR递归抽象处理技术
- `agent_enhanced_rag.py`: Agent增强RAG系统，支持自主决策和工具调用
- `demo.py`: 完整演示脚本，展示三种技术的集成应用
- `requirements.txt`: 依赖包清单
- `README.md`: 项目说明文档

## 核心技术模块

### 1. `graph_rag.py` - GraphRAG系统

**功能**: 基于知识图谱的增强检索，支持复杂关系推理。

**核心类**:
- `GraphRAGSystem`: 主系统类，整合图构建和检索
- `EntityExtractor`: 实体提取器，从文本中识别关键实体
- `RelationExtractor`: 关系提取器，发现实体间的关联
- `CommunityDetector`: 社区检测器，识别图中的知识聚群
- `GraphRetriever`: 图检索器，执行多跳推理检索

**核心功能**:
- 从文档构建知识图谱
- 实体和关系自动提取
- 社区检测和摘要生成
- 图遍历推理检索
- 多维度证据收集

**使用示例**:
```python
from graph_rag import GraphRAGSystem, MockLLM, MockEmbeddingModel

# 初始化系统
llm = MockLLM()
embedding_model = MockEmbeddingModel()
graph_rag = GraphRAGSystem(llm, embedding_model)

# 构建知识图谱
documents = [{"content": "文档内容...", "title": "标题"}]
graph_rag.build_knowledge_graph(documents)

# 执行查询
result = graph_rag.generate_graph_augmented_answer("你的问题")
print(result['answer'])
```

### 2. `raptor_tree.py` - RAPTOR分层系统

**功能**: 递归构建文档的分层抽象结构，支持多粒度检索。

**核心类**:
- `RAPTORTree`: 主系统类，管理分层树结构
- `RAPTORNode`: 树节点，包含内容、摘要和嵌入
- 聚类和摘要生成组件

**核心功能**:
- 文档自动分块和嵌入
- 递归聚类构建树结构
- 多层次摘要生成
- 分层检索策略(树遍历/分层/扁平化)
- 多策略结果融合

**检索策略**:
- `tree_traversal`: 从根节点开始的树遍历检索
- `layer_wise`: 按层级进行的全面检索
- `collapsed_tree`: 扁平化的全节点检索

**使用示例**:
```python
from raptor_tree import RAPTORTree, MockLLM, MockEmbeddingModel

# 初始化RAPTOR系统
llm = MockLLM()
embedding_model = MockEmbeddingModel()
raptor = RAPTORTree(llm, embedding_model, max_cluster_size=5)

# 构建分层树
documents = [{"content": "文档内容...", "title": "标题"}]
raptor.build_raptor_tree(documents)

# 多策略检索
tree_results = raptor.raptor_retrieval(query, traverse_strategy='tree_traversal')
layer_results = raptor.raptor_retrieval(query, traverse_strategy='layer_wise')

# 生成答案
result = raptor.generate_raptor_answer("你的问题")
print(result['answer'])
```

### 3. `agent_enhanced_rag.py` - Agent增强系统

**功能**: 集成Agent能力的RAG系统，支持自主决策、记忆管理和工具调用。

**核心类**:
- `AgentEnhancedRAG`: 主系统，整合Agent各项能力
- `ActionPlanner`: 动作规划器，制定执行计划
- `ActionExecutor`: 动作执行器，执行具体步骤
- `MemoryManager`: 记忆管理器，管理用户交互历史
- `ToolRegistry`: 工具注册中心，管理可用工具
- `ContinualLearningEngine`: 持续学习引擎

**Agent能力**:
- **自主规划**: 基于意图分析制定多步执行计划
- **记忆管理**: 存储和检索用户交互历史
- **工具调用**: 支持搜索、分析、计算、验证等工具
- **持续学习**: 从交互中学习优化响应质量
- **动态调整**: 根据执行结果动态调整计划

**工具系统**:
- `SearchTool`: 信息检索工具
- `AnalysisTool`: 信息分析工具  
- `CalculatorTool`: 数学计算工具
- `FactCheckTool`: 事实验证工具

**使用示例**:
```python
from agent_enhanced_rag import AgentEnhancedRAG, MockLLM, MockRetriever, MockEmbeddingModel

# 初始化Agent系统
llm = MockLLM()
retriever = MockRetriever()
embedding_model = MockEmbeddingModel()
agent_rag = AgentEnhancedRAG(llm, retriever, embedding_model)

# 用户上下文
user_context = {
    'user_id': 'user_123',
    'session_id': 'session_456',
    'original_query': '你的问题'
}

# 增强查询处理
result = agent_rag.enhanced_query_processing("你的问题", user_context)
print(result['answer'])
```

### 4. `demo.py` - 集成演示系统

**功能**: 展示三种技术的完整集成和协同工作。

**核心类**:
- `IntegratedRAGSystem`: 集成系统，统一管理三种RAG技术

**演示功能**:
- 统一的知识库构建
- 多模式并行查询
- 智能结果融合
- 性能对比分析
- 系统状态监控

**运行模式**:
- **完整演示** (`python demo.py`): 展示所有功能和性能对比
- **简化演示** (`python demo.py --simple`): 快速体验核心功能
- **帮助信息** (`python demo.py --help`): 显示使用说明

**使用示例**:
```python
from demo import IntegratedRAGSystem

# 初始化集成系统
system = IntegratedRAGSystem()

# 设置知识库
documents = [...]  # 你的文档
setup_result = system.setup_knowledge_base(documents)

# 多模式查询
result = system.multi_modal_query(
    "你的问题",
    use_graph_rag=True,
    use_raptor=True, 
    use_agent=True
)

print(result['final_answer'])
```

## 技术特色

### GraphRAG优势
- **关系推理**: 支持复杂的多跳关系推理
- **全局理解**: 通过社区检测获得全局知识视图
- **证据路径**: 提供清晰的推理证据链
- **结构化知识**: 将非结构化文档转为结构化图谱

### RAPTOR优势  
- **多层抽象**: 支持不同粒度的信息检索
- **递归聚类**: 自动发现文档内在结构
- **分层摘要**: 提供从细节到概要的多层次信息
- **灵活检索**: 支持多种检索策略

### Agent增强优势
- **自主决策**: 智能分析意图并制定执行计划
- **记忆能力**: 记住用户历史交互，提供个性化服务
- **工具集成**: 扩展系统能力边界
- **持续优化**: 从交互中学习，不断改进性能

## 性能优化

### 内存优化
- 使用嵌入向量缓存减少重复计算
- 限制图谱和树结构的规模
- 实现记忆容量管理机制

### 计算优化
- 并行处理多个检索任务
- 优化聚类算法参数
- 缓存中间计算结果

### 错误处理
- 完善的异常捕获和处理
- 优雅的降级策略
- 详细的错误日志记录

## 扩展指南

### 添加新工具
```python
class CustomTool(BaseTool):
    def __init__(self):
        super().__init__("custom_tool", "自定义工具描述")
    
    def execute(self, **kwargs) -> Dict:
        # 工具实现逻辑
        return {"status": "success", "result": "..."}

# 注册工具
agent_rag.tool_registry.register(CustomTool())
```

### 自定义检索策略
```python
def custom_retrieval_strategy(self, query_embedding, query, top_k):
    # 自定义检索逻辑
    pass

# 添加到RAPTOR系统
raptor_tree._custom_retrieval = custom_retrieval_strategy
```

### 集成外部模型
```python
class CustomLLM:
    def generate(self, prompt: str) -> str:
        # 调用实际的LLM服务
        pass

# 替换模拟LLM
graph_rag.llm = CustomLLM()
```

## 注意事项

1. **依赖安装**: 确保安装所有必需依赖，特别是numpy、networkx、scikit-learn
2. **内存使用**: 大规模文档处理时注意内存占用
3. **计算性能**: 图构建和树构建可能需要较长时间
4. **模型替换**: 示例使用模拟模型，实际应用需替换为真实模型
5. **错误处理**: 生产环境中需要增强错误处理和监控

## 故障排除

### 常见问题

**Q: 导入模块失败**  
A: 检查依赖安装：`pip install -r requirements.txt`

**Q: 内存不足**  
A: 减少文档数量或调整`max_cluster_size`参数

**Q: 构建速度慢**  
A: 检查文档大小，考虑增加分块粒度

**Q: 检索结果质量差**  
A: 调整嵌入模型或优化文档预处理

### 调试模式
设置环境变量启用详细日志：
```bash
export DEBUG_MODE=1
python demo.py
```

## 进阶应用

### 生产环境部署
- 使用实际的LLM和嵌入模型
- 配置持久化存储
- 添加API接口和监控
- 实现负载均衡

### 性能调优
- 根据数据特点调整聚类参数
- 优化图构建策略
- 实现增量更新机制

### 功能扩展
- 添加多语言支持
- 集成更多工具类型
- 实现用户个性化
- 支持实时学习更新

---

**本代码是RAG+Agent融合技术的完整实现，展示了下一代智能问答系统的核心能力。通过GraphRAG的关系推理、RAPTOR的分层抽象和Agent的自主决策，构建了一个功能强大、智能化程度高的RAG系统。**