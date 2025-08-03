# RAG准确性提升系统 - 深度RAG笔记14

本章节代码配套第14篇文章 **《20分钟解决RAG胡说八道问题》**。

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行演示

```bash
python demo.py
```

## 文件结构

- `accuracy_checker.py`: RAG准确性诊断工具，快速发现回答中的问题
- `smart_reasoner.py`: 智能推理引擎，实现多步推理和自适应策略  
- `self_corrector.py`: 自我修正机制，自动检查并修正事实、逻辑等错误
- `confidence_scorer.py`: 置信度评估器，为AI的回答提供可信度分数
- `self_rag.py`: Self-RAG实现，让RAG具备自我反思的能力
- `corrective_rag.py`: Corrective RAG实现，当检索质量不佳时自动进行纠正
- `demo.py`: 完整演示脚本，展示端到端的准确性提升流程
- `requirements.txt`: 依赖包清单

### 模块说明

#### 1. `accuracy_checker.py`
- **功能**: 快速诊断RAG输出的质量。
- **核心类**: `AccuracyChecker`
- **主要方法**: `quick_diagnosis()`
- **诊断维度**: 幻觉、逻辑、完整性、一致性。

#### 2. `smart_reasoner.py`
- **功能**: 实现多步推理，让AI“思考”更深入。
- **核心类**: `SmartReasoner`, `StrategySelector`
- **主要方法**: `think_step_by_step()`
- **特色**: 能根据问题复杂度自动选择推理策略（直接回答、分步思考、深度推理）。

#### 3. `self_corrector.py`
- **功能**: 对生成的答案进行多轮自我检查和修正。
- **核心类**: `SelfCorrector`
- **主要方法**: `auto_fix_errors()`
- **修正类型**: 事实错误、逻辑错误、完整性错误。

#### 4. `confidence_scorer.py`
- **功能**: 为最终答案生成一个置信度分数。
- **核心类**: `ConfidenceScorer`
- **主要方法**: `rate_answer_confidence()`
- **评估维度**: 来源可靠性、信息一致性、回答完整性等。

#### 5. `self_rag.py`
- **功能**: 实现Self-RAG框架，在生成过程中加入反思环节。
- **核心类**: `SelfRAG`
- **主要方法**: `generate_with_reflection()`
- **反思点**: 是否需要检索、检索质量、生成内容质量。

#### 6. `corrective_rag.py`
- **功能**: 实现Corrective RAG框架，在检索效果不佳时自动采取纠正措施。
- **核心类**: `CorrectiveRAG`
- **主要方法**: `smart_retrieve_and_generate()`
- **纠正措施**: 查询扩展、Web搜索等。

#### 7. `demo.py`
- **功能**: 将以上所有组件集成，提供一个端到端的演示。
- **运行**: 直接运行此文件即可看到所有模块协同工作的效果。
