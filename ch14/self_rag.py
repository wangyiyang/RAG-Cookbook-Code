"""
Self-RAG: AI学会自我反思
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ReflectionAction(Enum):
    """反思动作枚举"""
    RETRIEVE = "retrieve"          # 需要检索
    GENERATE = "generate"          # 直接生成
    CRITICIZE = "criticize"        # 评估生成内容
    CORRECT = "correct"            # 修正生成内容


@dataclass
class ReflectionToken:
    """反思Token"""
    action: ReflectionAction
    value: Any


class SelfRAG:
    """Self-RAG：带自我反思的RAG生成"""
    
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    def generate_with_reflection(self, question: str) -> Dict:
        """带自我反思的RAG生成"""
        
        reflection_trace = []
        
        # Step 1: 判断是否需要检索
        should_retrieve_token = self._decide_retrieval(question)
        reflection_trace.append(should_retrieve_token)
        
        if should_retrieve_token.value:
            # Step 2: 如果需要，执行检索
            docs = self.retriever.search(question)
            
            # Step 3: 评估检索质量
            retrieval_quality_token = self._evaluate_retrieval(question, docs)
            reflection_trace.append(retrieval_quality_token)
            
            if retrieval_quality_token.value == "good":
                # 质量好，直接生成
                answer = self._generate_answer(question, docs)
            else:
                # 质量差，尝试修正检索或使用内部知识
                answer = self._fallback_generate(question, docs)
        else:
            # 不需要检索，直接生成
            answer = self._generate_without_retrieval(question)
        
        # Step 4: 评估和修正生成内容
        criticism_token = self._criticize_generation(question, answer)
        reflection_trace.append(criticism_token)
        
        if criticism_token.value == "needs_correction":
            final_answer = self._correct_generation(question, answer)
            correction_token = ReflectionToken(action=ReflectionAction.CORRECT, value="修正完成")
            reflection_trace.append(correction_token)
        else:
            final_answer = answer
        
        return {
            "final_answer": final_answer,
            "reflection_trace": [token.__dict__ for token in reflection_trace]
        }
    
    def _decide_retrieval(self, question: str) -> ReflectionToken:
        """判断是否需要检索"""
        prompt = f"""
        问题: "{question}"
        
        请判断回答这个问题是否需要检索外部知识。
        
        回答 "是" 或 "否"。
        """
        
        decision = self.llm.generate(prompt)
        needs_retrieval = "是" in decision.lower()
        
        return ReflectionToken(action=ReflectionAction.RETRIEVE, value=needs_retrieval)
    
    def _evaluate_retrieval(self, question: str, docs: List[Dict]) -> ReflectionToken:
        """评估检索质量"""
        context = self._build_context(docs)
        
        prompt = f"""
        问题: "{question}"
        
        检索到的信息:
        {context}
        
        请评估检索到的信息是否与问题相关，并且足够回答问题。
        
        回答 "good" 或 "bad"。
        """
        
        quality = self.llm.generate(prompt).lower()
        return ReflectionToken(action=ReflectionAction.CRITICIZE, value=quality)
    
    def _generate_answer(self, question: str, docs: List[Dict]) -> str:
        """基于文档生成答案"""
        context = self._build_context(docs)
        
        prompt = f"""
        问题: "{question}"
        
        参考信息:
        {context}
        
        请根据参考信息回答问题。
        """
        
        return self.llm.generate(prompt)
    
    def _fallback_generate(self, question: str, docs: List[Dict]) -> str:
        """检索质量差时的备用生成方案"""
        # 尝试使用更通用的提示
        context = self._build_context(docs)
        
        prompt = f"""
        问题: "{question}"
        
        参考信息可能不完全相关:
        {context}
        
        请结合参考信息和你的内部知识，谨慎地回答问题。
        如果信息不足，请说明。
        """
        
        return self.llm.generate(prompt)
    
    def _generate_without_retrieval(self, question: str) -> str:
        """不使用检索直接生成"""
        prompt = f"""
        问题: "{question}"
        
        请直接回答这个问题。
        """
        
        return self.llm.generate(prompt)
    
    def _criticize_generation(self, question: str, answer: str) -> ReflectionToken:
        """评估生成内容的质量"""
        prompt = f"""
        问题: "{question}"
        回答: "{answer}"
        
        请评估这个回答是否准确、完整、并且没有幻觉。
        
        回答 "ok" 或 "needs_correction"。
        """
        
        criticism = self.llm.generate(prompt).lower()
        return ReflectionToken(action=ReflectionAction.CRITICIZE, value=criticism)
    
    def _correct_generation(self, question: str, answer: str) -> str:
        """修正生成的内容"""
        prompt = f"""
        问题: "{question}"
        有问题的回答: "{answer}"
        
        请修正这个回答，使其更加准确和完整。
        """
        
        return self.llm.generate(prompt)
    
    def _build_context(self, docs: List[Dict]) -> str:
        """构建上下文"""
        if not docs:
            return "无相关信息"
        
        return "\n\n".join([f"文档 {i+1}: {doc.get('content', '')[:200]}" for i, doc in enumerate(docs)])


# 模拟的LLM和检索器类
class MockLLM:
    def generate(self, prompt: str) -> str:
        if "判断回答这个问题是否需要检索" in prompt:
            return "是"
        elif "评估检索到的信息是否与问题相关" in prompt:
            return "good"
        elif "评估这个回答是否准确" in prompt:
            return "needs_correction"
        else:
            return f"模拟回答: {prompt[:50]}..."


class MockRetriever:
    def search(self, query: str) -> List[Dict]:
        return [
            {"content": f"关于'{query}'的相关信息1"},
            {"content": f"关于'{query}'的相关信息2"}
        ]


def main():
    """测试Self-RAG"""
    llm = MockLLM()
    retriever = MockRetriever()
    self_rag = SelfRAG(llm, retriever)
    
    question = "什么是人工智能？"
    
    print("=== Self-RAG 测试 ===")
    print(f"问题: {question}")
    print()
    
    result = self_rag.generate_with_reflection(question)
    
    print(f"最终答案: {result['final_answer']}")
    print()
    
    print("反思过程:")
    for step in result['reflection_trace']:
        print(f"  - 动作: {step['action']}, 值: {step['value']}")


if __name__ == "__main__":
    main()
