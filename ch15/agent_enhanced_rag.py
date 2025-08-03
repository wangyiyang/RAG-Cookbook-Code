"""
Agent增强RAG系统
结合自主决策、记忆管理、工具调度等Agent能力
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import uuid

class ActionType(Enum):
    """动作类型"""
    SEARCH = "search"
    ANALYZE = "analyze"
    GENERATE = "generate"
    VERIFY = "verify"
    TOOL_CALL = "tool_call"

@dataclass
class ActionStep:
    """执行步骤"""
    id: str
    action_type: ActionType
    description: str
    parameters: Dict
    dependencies: List[str]
    status: str = "pending"
    result: Optional[Dict] = None

@dataclass
class MemoryRecord:
    """记忆记录"""
    id: str
    user_id: str
    query: str
    response: str
    timestamp: str
    success_score: float
    context: Dict

class AgentEnhancedRAG:
    """Agent增强的RAG系统"""
    
    def __init__(self, llm, retriever, embedding_model):
        if llm is None:
            raise ValueError("LLM实例不能为空")
        if retriever is None:
            raise ValueError("检索器实例不能为空")
        if embedding_model is None:
            raise ValueError("嵌入模型不能为空")
            
        self.llm = llm
        self.retriever = retriever
        self.embedding_model = embedding_model
        
        # Agent核心组件
        self.planner = ActionPlanner(llm)
        self.executor = ActionExecutor(llm, retriever)
        self.memory = MemoryManager()
        self.tool_registry = ToolRegistry()
        self.learning_engine = ContinualLearningEngine()
        
        # 注册基础工具
        self._register_basic_tools()
        
    def _register_basic_tools(self):
        """注册基础工具"""
        self.tool_registry.register(SearchTool(self.retriever))
        self.tool_registry.register(AnalysisTool(self.llm))
        self.tool_registry.register(CalculatorTool())
        self.tool_registry.register(FactCheckTool(self.llm))
        
    def enhanced_query_processing(self, query: str, user_context: Dict) -> Dict:
        """增强的查询处理"""
        
        if not query or not query.strip():
            raise ValueError("查询不能为空")
        
        user_id = user_context.get('user_id', 'anonymous')
        session_id = user_context.get('session_id', str(uuid.uuid4()))
        
        try:
            # 1. 意图分析与规划
            intent_analysis = self.analyze_user_intent(query, user_context)
            action_plan = self.planner.create_action_plan(intent_analysis)
            
            # 2. 记忆激活
            relevant_memories = self.memory.retrieve_relevant_memories(query, user_id)
            
            # 3. 多步骤执行
            execution_results = []
            for step in action_plan.steps:
                step_result = self.executor.execute_step(step, user_context, relevant_memories)
                execution_results.append(step_result)
                
                # 动态调整计划
                if step_result.get('requires_replanning', False):
                    action_plan = self.planner.replan(action_plan, step_result)
            
            # 4. 结果整合与学习
            final_result = self.integrate_results(execution_results, query)
            
            # 5. 记忆更新
            memory_record = MemoryRecord(
                id=str(uuid.uuid4()),
                user_id=user_id,
                query=query,
                response=final_result['answer'],
                timestamp=datetime.now().isoformat(),
                success_score=final_result.get('confidence_score', 0.5),
                context=user_context
            )
            self.memory.store_memory(memory_record)
            
            # 6. 持续学习
            self.learning_engine.learn_from_interaction(
                query, final_result, user_context
            )
            
            return final_result
            
        except Exception as e:
            return {
                'answer': f"抱歉，处理您的请求时遇到错误：{str(e)}",
                'confidence_score': 0.0,
                'execution_trace': [],
                'error': str(e)
            }
    
    def analyze_user_intent(self, query: str, context: Dict) -> Dict:
        """深度意图分析"""
        
        intent_analysis = {
            'primary_intent': None,
            'secondary_intents': [],
            'required_tools': [],
            'information_needs': [],
            'complexity_level': 'medium'
        }
        
        try:
            # 使用多层次意图识别
            primary_intent = self._classify_primary_intent(query)
            intent_analysis['primary_intent'] = primary_intent
            
            # 识别所需工具
            required_tools = self._identify_required_tools(query, primary_intent)
            intent_analysis['required_tools'] = required_tools
            
            # 分析信息需求
            info_needs = self._analyze_information_needs(query, context)
            intent_analysis['information_needs'] = info_needs
            
            # 评估复杂度
            complexity = self._assess_query_complexity(query)
            intent_analysis['complexity_level'] = complexity
            
            return intent_analysis
            
        except Exception as e:
            print(f"意图分析错误: {e}")
            return intent_analysis
    
    def _classify_primary_intent(self, query: str) -> str:
        """分类主要意图"""
        
        # 简化的意图分类逻辑
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['什么', '是什么', '定义', '概念']):
            return 'definition'
        elif any(word in query_lower for word in ['如何', '怎么', '方法', '步骤']):
            return 'how_to'
        elif any(word in query_lower for word in ['为什么', '原因', '因为']):
            return 'explanation'
        elif any(word in query_lower for word in ['比较', '区别', '差异']):
            return 'comparison'
        elif any(word in query_lower for word in ['计算', '数学', '统计']):
            return 'calculation'
        else:
            return 'general_inquiry'
    
    def _identify_required_tools(self, query: str, intent: str) -> List[str]:
        """识别所需工具"""
        
        tools = []
        query_lower = query.lower()
        
        # 基于意图和关键词识别工具
        if intent == 'calculation' or any(word in query_lower for word in ['计算', '数学']):
            tools.append('calculator')
        
        if any(word in query_lower for word in ['搜索', '查找', '最新']):
            tools.append('search')
        
        if any(word in query_lower for word in ['分析', '研究', '深入']):
            tools.append('analysis')
        
        if any(word in query_lower for word in ['验证', '事实', '准确']):
            tools.append('fact_check')
        
        # 默认包含搜索工具
        if not tools:
            tools.append('search')
        
        return tools
    
    def _analyze_information_needs(self, query: str, context: Dict) -> List[str]:
        """分析信息需求"""
        
        needs = []
        
        # 基于查询内容分析需求
        if '最新' in query or '最近' in query:
            needs.append('recent_information')
        
        if '详细' in query or '具体' in query:
            needs.append('detailed_information')
        
        if '例子' in query or '示例' in query:
            needs.append('examples')
        
        if '比较' in query:
            needs.append('comparative_analysis')
        
        return needs
    
    def _assess_query_complexity(self, query: str) -> str:
        """评估查询复杂度"""
        
        complexity_score = 0
        
        # 长度因子
        if len(query) > 50:
            complexity_score += 1
        if len(query) > 100:
            complexity_score += 1
        
        # 复杂词汇
        complex_words = ['为什么', '如何', '比较', '分析', '评估', '影响', '原理']
        for word in complex_words:
            if word in query:
                complexity_score += 1
        
        # 多重问题
        if '？' in query or '?' in query:
            question_count = query.count('？') + query.count('?')
            complexity_score += min(question_count, 3)
        
        if complexity_score <= 2:
            return 'low'
        elif complexity_score <= 5:
            return 'medium'
        else:
            return 'high'
    
    def integrate_results(self, execution_results: List[Dict], query: str) -> Dict:
        """整合执行结果"""
        
        try:
            successful_results = [r for r in execution_results if r.get('status') == 'success']
            
            if not successful_results:
                return {
                    'answer': '抱歉，无法获取到有效信息来回答您的问题。',
                    'confidence_score': 0.0,
                    'sources': [],
                    'execution_trace': execution_results
                }
            
            # 整合所有成功结果的内容
            all_content = []
            all_sources = []
            confidence_scores = []
            
            for result in successful_results:
                if 'content' in result:
                    all_content.append(result['content'])
                if 'sources' in result:
                    all_sources.extend(result['sources'])
                if 'confidence' in result:
                    confidence_scores.append(result['confidence'])
            
            # 生成综合答案
            integration_prompt = f"""
基于以下信息回答问题：

问题：{query}

收集到的信息：
{chr(10).join([f"{i+1}. {content}" for i, content in enumerate(all_content)])}

请提供一个综合、准确的答案：
"""
            
            integrated_answer = self.llm.generate(integration_prompt)
            
            # 计算综合置信度
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
            
            return {
                'answer': integrated_answer,
                'confidence_score': avg_confidence,
                'sources': list(set(all_sources)),  # 去重
                'execution_trace': execution_results,
                'integration_method': 'llm_synthesis'
            }
            
        except Exception as e:
            return {
                'answer': f'结果整合时发生错误：{str(e)}',
                'confidence_score': 0.0,
                'sources': [],
                'execution_trace': execution_results,
                'error': str(e)
            }


class ActionPlanner:
    """动作规划器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def create_action_plan(self, intent_analysis: Dict) -> 'ActionPlan':
        """创建执行计划"""
        
        steps = []
        step_id = 0
        
        # 基于意图和工具需求创建步骤
        required_tools = intent_analysis.get('required_tools', [])
        complexity = intent_analysis.get('complexity_level', 'medium')
        
        # 添加搜索步骤
        if 'search' in required_tools:
            search_step = ActionStep(
                id=f"step_{step_id}",
                action_type=ActionType.SEARCH,
                description="搜索相关信息",
                parameters={'search_depth': 'deep' if complexity == 'high' else 'normal'},
                dependencies=[]
            )
            steps.append(search_step)
            step_id += 1
        
        # 添加分析步骤
        if 'analysis' in required_tools or complexity == 'high':
            analysis_step = ActionStep(
                id=f"step_{step_id}",
                action_type=ActionType.ANALYZE,
                description="分析收集的信息",
                parameters={'analysis_type': 'comprehensive'},
                dependencies=[f"step_{step_id-1}"] if steps else []
            )
            steps.append(analysis_step)
            step_id += 1
        
        # 添加计算步骤
        if 'calculator' in required_tools:
            calc_step = ActionStep(
                id=f"step_{step_id}",
                action_type=ActionType.TOOL_CALL,
                description="执行数学计算",
                parameters={'tool': 'calculator'},
                dependencies=[]
            )
            steps.append(calc_step)
            step_id += 1
        
        # 添加验证步骤
        if 'fact_check' in required_tools:
            verify_step = ActionStep(
                id=f"step_{step_id}",
                action_type=ActionType.VERIFY,
                description="验证信息准确性",
                parameters={'verification_level': 'standard'},
                dependencies=[s.id for s in steps]
            )
            steps.append(verify_step)
            step_id += 1
        
        # 添加生成步骤
        generate_step = ActionStep(
            id=f"step_{step_id}",
            action_type=ActionType.GENERATE,
            description="生成最终答案",
            parameters={'generation_style': 'comprehensive'},
            dependencies=[s.id for s in steps]
        )
        steps.append(generate_step)
        
        return ActionPlan(steps)
    
    def replan(self, current_plan: 'ActionPlan', step_result: Dict) -> 'ActionPlan':
        """重新规划"""
        
        # 简化的重新规划逻辑
        if step_result.get('requires_additional_search'):
            # 添加额外搜索步骤
            additional_search = ActionStep(
                id=f"step_additional_{len(current_plan.steps)}",
                action_type=ActionType.SEARCH,
                description="补充搜索信息",
                parameters={'search_query': step_result.get('suggested_query', '')},
                dependencies=[]
            )
            current_plan.steps.insert(-1, additional_search)  # 在最后一步之前插入
        
        return current_plan

@dataclass
class ActionPlan:
    """执行计划"""
    steps: List[ActionStep]


class ActionExecutor:
    """动作执行器"""
    
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    def execute_step(self, step: ActionStep, context: Dict, memories: List[MemoryRecord]) -> Dict:
        """执行单个步骤"""
        
        try:
            if step.action_type == ActionType.SEARCH:
                return self._execute_search(step, context)
            elif step.action_type == ActionType.ANALYZE:
                return self._execute_analysis(step, context)
            elif step.action_type == ActionType.GENERATE:
                return self._execute_generation(step, context)
            elif step.action_type == ActionType.VERIFY:
                return self._execute_verification(step, context)
            elif step.action_type == ActionType.TOOL_CALL:
                return self._execute_tool_call(step, context)
            else:
                return {
                    'status': 'error',
                    'error': f'不支持的动作类型: {step.action_type}'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'step_id': step.id
            }
    
    def _execute_search(self, step: ActionStep, context: Dict) -> Dict:
        """执行搜索"""
        
        query = context.get('original_query', '')
        search_depth = step.parameters.get('search_depth', 'normal')
        
        # 执行检索
        top_k = 10 if search_depth == 'deep' else 5
        search_results = self.retriever.search(query, top_k=top_k)
        
        return {
            'status': 'success',
            'content': self._format_search_results(search_results),
            'sources': [r.get('source', '') for r in search_results],
            'confidence': 0.8,
            'step_id': step.id
        }
    
    def _execute_analysis(self, step: ActionStep, context: Dict) -> Dict:
        """执行分析"""
        
        analysis_prompt = f"""
请分析以下信息，提供深入的见解：

查询：{context.get('original_query', '')}

信息：{context.get('collected_info', '暂无信息')}

请提供分析结果：
"""
        
        analysis_result = self.llm.generate(analysis_prompt)
        
        return {
            'status': 'success',
            'content': analysis_result,
            'confidence': 0.7,
            'step_id': step.id
        }
    
    def _execute_generation(self, step: ActionStep, context: Dict) -> Dict:
        """执行生成"""
        
        generation_prompt = f"""
基于收集和分析的信息，生成对以下问题的完整答案：

问题：{context.get('original_query', '')}

已收集的信息：{context.get('all_collected_info', '')}

请生成准确、全面的答案：
"""
        
        generated_answer = self.llm.generate(generation_prompt)
        
        return {
            'status': 'success',
            'content': generated_answer,
            'confidence': 0.9,
            'step_id': step.id
        }
    
    def _execute_verification(self, step: ActionStep, context: Dict) -> Dict:
        """执行验证"""
        
        # 简化的验证逻辑
        return {
            'status': 'success',
            'content': '信息已通过基本验证',
            'confidence': 0.6,
            'step_id': step.id
        }
    
    def _execute_tool_call(self, step: ActionStep, context: Dict) -> Dict:
        """执行工具调用"""
        
        tool_name = step.parameters.get('tool', '')
        
        if tool_name == 'calculator':
            return {
                'status': 'success',
                'content': '计算结果：42',  # 模拟计算结果
                'confidence': 1.0,
                'step_id': step.id
            }
        else:
            return {
                'status': 'error',
                'error': f'未知工具: {tool_name}',
                'step_id': step.id
            }
    
    def _format_search_results(self, results: List[Dict]) -> str:
        """格式化搜索结果"""
        
        if not results:
            return "未找到相关信息"
        
        formatted = []
        for i, result in enumerate(results[:5]):  # 限制显示数量
            content = result.get('content', '无内容')
            formatted.append(f"{i+1}. {content}")
        
        return "\n".join(formatted)


class MemoryManager:
    """记忆管理器"""
    
    def __init__(self):
        self.memories = {}  # user_id -> List[MemoryRecord]
    
    def store_memory(self, memory: MemoryRecord):
        """存储记忆"""
        
        if memory.user_id not in self.memories:
            self.memories[memory.user_id] = []
        
        self.memories[memory.user_id].append(memory)
        
        # 维护记忆容量（保留最近100条）
        if len(self.memories[memory.user_id]) > 100:
            self.memories[memory.user_id] = self.memories[memory.user_id][-100:]
    
    def retrieve_relevant_memories(self, query: str, user_id: str, top_k: int = 3) -> List[MemoryRecord]:
        """检索相关记忆"""
        
        if user_id not in self.memories:
            return []
        
        user_memories = self.memories[user_id]
        
        # 简化的相关性计算（基于关键词匹配）
        relevant_memories = []
        query_words = set(query.lower().split())
        
        for memory in user_memories:
            memory_words = set(memory.query.lower().split())
            overlap = len(query_words.intersection(memory_words))
            
            if overlap > 0:
                relevant_memories.append((memory, overlap))
        
        # 按相关性排序
        relevant_memories.sort(key=lambda x: x[1], reverse=True)
        
        return [memory for memory, _ in relevant_memories[:top_k]]


class ToolRegistry:
    """工具注册中心"""
    
    def __init__(self):
        self.tools = {}
    
    def register(self, tool: 'BaseTool'):
        """注册工具"""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional['BaseTool']:
        """获取工具"""
        return self.tools.get(name)


class BaseTool:
    """工具基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, **kwargs) -> Dict:
        """执行工具"""
        raise NotImplementedError


class SearchTool(BaseTool):
    """搜索工具"""
    
    def __init__(self, retriever):
        super().__init__("search", "执行信息检索")
        self.retriever = retriever
    
    def execute(self, query: str, **kwargs) -> Dict:
        """执行搜索"""
        results = self.retriever.search(query)
        return {
            'status': 'success',
            'results': results,
            'count': len(results)
        }


class AnalysisTool(BaseTool):
    """分析工具"""
    
    def __init__(self, llm):
        super().__init__("analysis", "执行信息分析")
        self.llm = llm
    
    def execute(self, content: str, **kwargs) -> Dict:
        """执行分析"""
        analysis = self.llm.generate(f"请分析以下内容：{content}")
        return {
            'status': 'success',
            'analysis': analysis
        }


class CalculatorTool(BaseTool):
    """计算器工具"""
    
    def __init__(self):
        super().__init__("calculator", "执行数学计算")
    
    def execute(self, expression: str, **kwargs) -> Dict:
        """执行计算"""
        try:
            # 简化的安全计算
            if all(c in '0123456789+-*/.() ' for c in expression):
                result = eval(expression)
                return {
                    'status': 'success',
                    'result': result,
                    'expression': expression
                }
            else:
                return {
                    'status': 'error',
                    'error': '不安全的表达式'
                }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }


class FactCheckTool(BaseTool):
    """事实检查工具"""
    
    def __init__(self, llm):
        super().__init__("fact_check", "执行事实验证")
        self.llm = llm
    
    def execute(self, statement: str, **kwargs) -> Dict:
        """执行事实检查"""
        check_result = self.llm.generate(f"请验证以下陈述的准确性：{statement}")
        return {
            'status': 'success',
            'verification': check_result,
            'confidence': 0.7
        }


class ContinualLearningEngine:
    """持续学习引擎"""
    
    def __init__(self):
        self.interaction_history = []
    
    def learn_from_interaction(self, query: str, result: Dict, context: Dict):
        """从交互中学习"""
        
        interaction = {
            'query': query,
            'result': result,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'success_score': result.get('confidence_score', 0.5)
        }
        
        self.interaction_history.append(interaction)
        
        # 维护历史记录容量
        if len(self.interaction_history) > 1000:
            self.interaction_history = self.interaction_history[-1000:]


# 模拟组件
class MockLLM:
    def generate(self, prompt: str) -> str:
        """模拟LLM生成"""
        if "分析以下信息" in prompt:
            return "这是对信息的深入分析，包含了关键洞察和重要结论。"
        elif "生成对以下问题的完整答案" in prompt:
            return "基于收集和分析的信息，这是一个综合性的答案，涵盖了问题的各个方面。"
        elif "验证以下陈述的准确性" in prompt:
            return "经过验证，该陈述基本准确，但需要注意某些细节。"
        else:
            return f"智能回答: {prompt.split('：')[1][:50] if '：' in prompt else '通用回答'}..."

class MockRetriever:
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """模拟检索"""
        return [
            {
                'content': f"关于'{query}'的相关信息{i+1}：这里包含了详细的技术说明和实用案例。",
                'source': f"来源{i+1}",
                'score': 0.9 - i * 0.1
            }
            for i in range(top_k)
        ]

class MockEmbeddingModel:
    def encode(self, text: str):
        """模拟嵌入编码"""
        return [0.1] * 768


def main():
    """测试Agent增强RAG系统"""
    print("=== Agent增强RAG系统测试 ===")
    
    # 初始化组件
    llm = MockLLM()
    retriever = MockRetriever()
    embedding_model = MockEmbeddingModel()
    
    # 创建Agent增强RAG系统
    agent_rag = AgentEnhancedRAG(llm, retriever, embedding_model)
    
    # 测试用户上下文
    user_context = {
        'user_id': 'user_123',
        'session_id': 'session_456',
        'preferences': {
            'detail_level': 'high',
            'answer_style': 'comprehensive'
        }
    }
    
    # 测试查询
    test_queries = [
        "深度学习和机器学习的区别是什么？",
        "如何计算神经网络的参数数量？",
        "请分析Transformer架构的优势"
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\n{i+1}. 测试查询: {query}")
        
        # 添加原始查询到上下文
        user_context['original_query'] = query
        
        result = agent_rag.enhanced_query_processing(query, user_context)
        
        print(f"答案: {result['answer']}")
        print(f"置信度: {result['confidence_score']:.3f}")
        print(f"执行步骤数: {len(result.get('execution_trace', []))}")
        print(f"数据源数: {len(result.get('sources', []))}")
        
        # 显示执行轨迹
        trace = result.get('execution_trace', [])
        if trace:
            print("执行轨迹:")
            for j, step in enumerate(trace):
                status = step.get('status', 'unknown')
                step_id = step.get('step_id', f'step_{j}')
                print(f"  {step_id}: {status}")
    
    print(f"\n系统记忆中的交互记录数: {len(agent_rag.memory.memories.get('user_123', []))}")


if __name__ == "__main__":
    main()