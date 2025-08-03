"""
智能推理引擎
让AI学会慢慢思考，而不是急着给答案
"""

import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ReasoningStrategy(Enum):
    """推理策略枚举"""
    DIRECT = "direct_answer"          # 直接回答
    STEP_BY_STEP = "step_by_step"     # 分步思考
    DEEP_REASONING = "deep_reasoning"  # 深度推理


@dataclass
class ReasoningStep:
    """推理步骤数据结构"""
    step_number: int
    question: str
    retrieved_docs: List[Dict]
    thinking: str
    confidence: float
    next_action: str


class SmartReasoner:
    """智能推理引擎：让AI学会慢慢想"""
    
    def __init__(self, llm, retriever, max_steps: int = 5):
        if llm is None:
            raise ValueError("LLM实例不能为空")
        if retriever is None:
            raise ValueError("Retriever实例不能为空")
        if max_steps <= 0:
            raise ValueError("最大步数必须大于0")
            
        self.llm = llm
        self.retriever = retriever
        self.max_steps = max_steps
        self.strategy_selector = StrategySelector()
    
    def think_step_by_step(self, question: str, strategy: Optional[str] = None) -> Dict:
        """让AI一步步思考，而不是直接给答案"""
        
        if not question or not question.strip():
            raise ValueError("问题不能为空")
            
        # 自动选择策略
        if strategy is None:
            try:
                strategy = self.strategy_selector.choose_strategy(question)
            except Exception as e:
                print(f"策略选择失败，使用默认策略: {e}")
                strategy = ReasoningStrategy.STEP_BY_STEP.value
        
        reasoning_trace = {
            "original_question": question,
            "strategy": strategy,
            "steps": [],
            "final_answer": "",
            "confidence": 0.0,
            "reasoning_time": 0
        }
        
        start_time = time.time()
        
        if strategy == ReasoningStrategy.DIRECT.value:
            # 直接回答模式
            answer = self._direct_answer(question)
            reasoning_trace["final_answer"] = answer
            reasoning_trace["confidence"] = 0.8
        
        elif strategy == ReasoningStrategy.STEP_BY_STEP.value:
            # 分步思考模式
            reasoning_trace = self._step_by_step_reasoning(question, reasoning_trace)
        
        elif strategy == ReasoningStrategy.DEEP_REASONING.value:
            # 深度推理模式
            reasoning_trace = self._deep_reasoning(question, reasoning_trace)
        
        reasoning_trace["reasoning_time"] = time.time() - start_time
        return reasoning_trace
    
    def _direct_answer(self, question: str) -> str:
        """直接回答模式"""
        docs = self.retriever.search(question)
        context = self._build_context(docs)
        
        prompt = f"""
        问题: {question}
        
        参考信息: {context}
        
        请直接回答问题，保持简洁准确。
        """
        
        return self.llm.generate(prompt)
    
    def _step_by_step_reasoning(self, question: str, reasoning_trace: Dict) -> Dict:
        """分步思考模式"""
        current_question = question
        steps = []
        
        for step_num in range(1, self.max_steps + 1):
            try:
                # 检索相关信息
                docs = self.retriever.search(current_question)
                context = self._build_context(docs)
            except Exception as e:
                print(f"检索失败 (步骤 {step_num}): {e}")
                context = "检索失败，使用内部知识"
                docs = []
            
            # 生成这一步的思考
            thinking_prompt = f"""
            当前问题: {current_question}
            
            参考信息: {context}
            
            请进行一步推理思考。如果需要进一步分析，请提出下一个需要思考的问题。
            如果可以给出最终答案，请明确标注"最终答案："。
            
            思考格式:
            分析: [你的分析过程]
            结论: [这一步的结论]
            下一步: [如果需要继续，下一步要思考什么]
            """
            
            thinking_result = self.llm.generate(thinking_prompt)
            
            # 解析思考结果
            step = ReasoningStep(
                step_number=step_num,
                question=current_question,
                retrieved_docs=docs,
                thinking=thinking_result,
                confidence=self._estimate_step_confidence(thinking_result),
                next_action=self._extract_next_action(thinking_result)
            )
            
            steps.append(step.__dict__)
            
            # 判断是否结束
            if "最终答案" in thinking_result or step.next_action == "完成":
                final_answer = self._extract_final_answer(thinking_result)
                reasoning_trace["final_answer"] = final_answer
                break
            
            # 准备下一步的问题
            current_question = self._extract_next_question(thinking_result)
            if not current_question:
                # 如果无法提取下一个问题，结束推理
                reasoning_trace["final_answer"] = self._summarize_thinking(steps)
                break
        
        reasoning_trace["steps"] = steps
        reasoning_trace["confidence"] = self._calculate_overall_confidence(steps)
        
        return reasoning_trace
    
    def _deep_reasoning(self, question: str, reasoning_trace: Dict) -> Dict:
        """深度推理模式"""
        # 深度推理包含多轮迭代和自我验证
        initial_steps = self._step_by_step_reasoning(question, reasoning_trace.copy())
        
        # 自我验证
        verification_result = self._verify_reasoning(initial_steps)
        
        if verification_result["needs_revision"]:
            # 如果需要修正，进行第二轮推理
            revised_question = verification_result["revised_question"]
            reasoning_trace = self._step_by_step_reasoning(revised_question, reasoning_trace)
            reasoning_trace["verification"] = verification_result
        else:
            reasoning_trace = initial_steps
        
        return reasoning_trace
    
    def _build_context(self, docs: List[Dict]) -> str:
        """构建上下文"""
        if not docs:
            return "暂无相关信息"
        
        context_parts = []
        for i, doc in enumerate(docs[:3]):  # 最多使用3个文档
            content = doc.get("content", "")[:200]  # 限制长度
            context_parts.append(f"文档{i+1}: {content}")
        
        return "\n\n".join(context_parts)
    
    def _estimate_step_confidence(self, thinking: str) -> float:
        """估算步骤置信度"""
        confidence_indicators = {
            "确定": 0.9,
            "明确": 0.8,
            "可能": 0.6,
            "也许": 0.4,
            "不确定": 0.3
        }
        
        base_confidence = 0.7
        for indicator, score in confidence_indicators.items():
            if indicator in thinking:
                return score
        
        return base_confidence
    
    def _extract_next_action(self, thinking: str) -> str:
        """提取下一步动作"""
        if "最终答案" in thinking or "结论" in thinking:
            return "完成"
        elif "下一步" in thinking or "需要" in thinking:
            return "继续"
        else:
            return "完成"
    
    def _extract_final_answer(self, thinking: str) -> str:
        """提取最终答案"""
        if "最终答案：" in thinking:
            return thinking.split("最终答案：")[1].strip()
        elif "结论：" in thinking:
            return thinking.split("结论：")[1].strip()
        else:
            return thinking.strip()
    
    def _extract_next_question(self, thinking: str) -> str:
        """提取下一个问题"""
        if "下一步：" in thinking:
            next_part = thinking.split("下一步：")[1].strip()
            return next_part.split("\n")[0]  # 取第一行
        return ""
    
    def _summarize_thinking(self, steps: List[Dict]) -> str:
        """总结思考过程"""
        if not steps:
            return "无法得出结论"
        
        conclusions = []
        for step in steps:
            thinking = step.get("thinking", "")
            if "结论：" in thinking:
                conclusion = thinking.split("结论：")[1].split("\n")[0]
                conclusions.append(conclusion.strip())
        
        return "。".join(conclusions) if conclusions else "综合分析后的结论"
    
    def _calculate_overall_confidence(self, steps: List[Dict]) -> float:
        """计算整体置信度"""
        if not steps:
            return 0.5
        
        confidences = [step.get("confidence", 0.5) for step in steps]
        return sum(confidences) / len(confidences)
    
    def _verify_reasoning(self, reasoning_result: Dict) -> Dict:
        """验证推理结果"""
        final_answer = reasoning_result.get("final_answer", "")
        steps = reasoning_result.get("steps", [])
        
        verification_prompt = f"""
        请验证以下推理过程是否合理:
        
        推理步骤: {json.dumps(steps, ensure_ascii=False, indent=2)}
        最终答案: {final_answer}
        
        请判断:
        1. 推理逻辑是否一致
        2. 结论是否有充分支撑
        3. 是否需要修正
        
        回答格式:
        验证结果: [合理/需要修正]
        问题: [如果有问题，指出具体问题]
        建议: [修正建议]
        """
        
        verification = self.llm.generate(verification_prompt)
        
        needs_revision = "需要修正" in verification
        revised_question = reasoning_result["original_question"]
        
        if needs_revision and "建议" in verification:
            suggestion = verification.split("建议：")[1].strip()
            revised_question = f"{reasoning_result['original_question']} (考虑: {suggestion})"
        
        return {
            "needs_revision": needs_revision,
            "verification_text": verification,
            "revised_question": revised_question
        }


class StrategySelector:
    """策略选择器：根据问题特征选择合适的推理方式"""
    
    def choose_strategy(self, question: str) -> str:
        """根据问题难度自动选择合适的思考方式"""
        complexity = self.analyze_complexity(question)
        
        if complexity < 3:
            return ReasoningStrategy.DIRECT.value
        elif complexity < 7:
            return ReasoningStrategy.STEP_BY_STEP.value
        else:
            return ReasoningStrategy.DEEP_REASONING.value
    
    def analyze_complexity(self, question: str) -> int:
        """分析问题复杂度 (1-10分)"""
        complexity_score = 1
        
        # 长度因素
        if len(question) > 50:
            complexity_score += 1
        if len(question) > 100:
            complexity_score += 1
        
        # 复杂词汇
        complex_words = ["为什么", "如何", "比较", "分析", "评估", "原理", "机制", "影响"]
        for word in complex_words:
            if word in question:
                complexity_score += 1
        
        # 多重问题
        if "?" in question or "？" in question:
            question_count = question.count("?") + question.count("？")
            if question_count > 1:
                complexity_score += 2
        
        # 逻辑连接词
        logic_words = ["并且", "或者", "但是", "然而", "因此", "所以"]
        for word in logic_words:
            if word in question:
                complexity_score += 1
        
        return min(complexity_score, 10)


# 模拟的LLM和检索器类，用于测试
class MockLLM:
    def generate(self, prompt: str) -> str:
        return f"模拟回答: {prompt[:50]}..."


class MockRetriever:
    def search(self, query: str) -> List[Dict]:
        return [
            {"content": f"关于'{query}'的相关信息1"},
            {"content": f"关于'{query}'的相关信息2"}
        ]


def main():
    """测试智能推理引擎"""
    llm = MockLLM()
    retriever = MockRetriever()
    reasoner = SmartReasoner(llm, retriever)
    
    # 测试问题
    question = "为什么Transformer模型的注意力机制比RNN更有效？"
    
    print("=== 智能推理测试 ===")
    print(f"问题: {question}")
    print()
    
    # 自动选择策略并推理
    result = reasoner.think_step_by_step(question)
    
    print(f"选择策略: {result['strategy']}")
    print(f"推理步数: {len(result['steps'])}")
    print(f"推理时间: {result['reasoning_time']:.2f}秒")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"最终答案: {result['final_answer']}")


if __name__ == "__main__":
    main()
