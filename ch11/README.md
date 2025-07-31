# 第11篇代码：RAGAS实战指南

本目录包含RAGAS RAG系统评估的完整代码实现，让你20分钟完成RAG系统体检。

## 文件说明

### ragas_evaluator.py
RAGAS评估系统，提供：
- 快速环境配置和数据准备
- 4个核心指标评估（忠实度、答案相关性、上下文相关性、上下文精确度）
- 智能结果解读和优化建议
- 批量评估和版本对比功能

### demo.py
使用示例和演示代码：
- 基础评估流程演示
- 常见问题解决方案
- 进阶使用技巧

## 快速开始

### 1. 安装依赖

```bash
# 基础安装
pip install ragas datasets openai

# 如果需要本地模型支持
pip install transformers torch
```

### 2. 配置API密钥

```bash
export OPENAI_API_KEY="your-api-key"
```

### 3. 运行评估

```python
from ragas_evaluator import RAGASEvaluator

# 创建评估器
evaluator = RAGASEvaluator()

# 准备测试数据
test_data = [
    {
        'question': '你的问题',
        'answer': 'RAG系统的回答',
        'contexts': ['检索到的上下文1', '检索到的上下文2'],
        'ground_truth': '标准答案（可选）'
    }
]

# 执行评估
result = evaluator.evaluate(test_data)
print(f"综合评分: {result['overall_score']:.3f}")
```

## 核心功能

### 支持的评估指标
- **faithfulness**: 忠实度评估，检查答案是否忠实于上下文
- **answer_relevancy**: 答案相关性，评估回答是否切题
- **context_relevancy**: 上下文相关性，检查检索质量
- **context_precision**: 上下文精确度，评估排序质量

### 结果解读
- 自动判断系统质量等级（优秀/良好/需改进/不合格）
- 针对性优化建议生成
- 详细的评估报告输出

### 实用功能
- 版本对比评估
- 批量自动化评估
- 自定义评估阈值
- 评估结果可视化

## 使用场景

1. **开发阶段**: 快速验证RAG系统基础性能
2. **优化阶段**: 对比不同版本的改进效果
3. **上线前**: 全面评估系统准备度
4. **运维阶段**: 定期监控系统质量

## 注意事项

- 确保测试数据质量，garbage in garbage out
- 首次运行会较慢，因为需要调用LLM进行评估
- 建议先用小批次数据验证，再进行大规模评估
- 根据业务场景调整评估阈值