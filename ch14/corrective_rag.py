"""
Corrective RAG: 错了就重来
智能检索+生成，发现问题立即纠正
"""

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class RetrievalQuality(Enum):
    """检索质量枚举"""
    HIGH = "high"        # 高质量
    MEDIUM = "medium"    # 中等质量
    LOW = "low"          # 低质量


@dataclass
class RetrievalEvaluation:
    """检索评估结果"""
    quality: RetrievalQuality
    issues: List[str]
    relevance_score: float
    completeness_score: float


class CorrectiveRAG:
    """Corrective RAG：智能检索+生成，发现问题立即纠正"""
    
    def __init__(self, llm, retriever, web_search_retriever=None):
        self.llm = llm
        self.retriever = retriever
        self.web_search_retriever = web_search_retriever or self.retriever  # 备用检索器
    
    def smart_retrieve_and_generate(self, question: str) -> Dict:
        """智能检索+生成，发现问题立即纠正"""
        
        correction_log = []
        
        # 第一次检索
        docs = self.retriever.search(question)
        correction_log.append(f"Initial retrieval found {len(docs)} documents.")
        
        # 评估检索质量
        evaluation = self._evaluate_retrieval_quality(question, docs)
        correction_log.append(f"Initial retrieval quality: {evaluation.quality.value}, issues: {evaluation.issues}")
        
        if evaluation.quality == RetrievalQuality.LOW:
            # 质量太差，尝试修正
            correction_log.append("Retrieval quality is low. Applying correction.")
            
            # 尝试查询扩展
            expanded_question = self._expand_query(question)
            correction_log.append(f"Expanded query: {expanded_question}")
            
            docs = self.retriever.search(expanded_question)
            
            # 再次评估
            evaluation = self._evaluate_retrieval_quality(expanded_question, docs)
            correction_log.append(f"Retrieval quality after query expansion: {evaluation.quality.value}")
            
            # 如果还是不行，尝试Web搜索
            if evaluation.quality == RetrievalQuality.LOW and self.web_search_retriever:
                correction_log.append("Still low quality. Trying web search.")
                docs = self.web_search_retriever.search(question)
        
        # 基于最终的文档生成答案
        final_answer = self._generate_final_answer(question, docs)
        
        return {
            "final_answer": final_answer,
            "final_docs_count": len(docs),
            "correction_log": correction_log
        }
    
    def _evaluate_retrieval_quality(self, question: str, docs: List[Dict]) -> RetrievalEvaluation:
        """评估检索质量"""
        issues = []
        
        if not docs:
            return RetrievalEvaluation(
                quality=RetrievalQuality.LOW,
                issues=["No documents found"],
                relevance_score=0.0,
                completeness_score=0.0
            )
        
        # 评估相关性
        relevance_score = self._calculate_relevance(question, docs)
        if relevance_score < 0.5:
            issues.append("Low relevance to the question")
        
        # 评估完整性
        completeness_score = self._calculate_completeness(question, docs)
        if completeness_score < 0.5:
            issues.append("Information seems incomplete")
        
        # 评估信息冲突
        if self._has_conflicts(docs):
            issues.append("Conflicting information found in documents")
        
        # 综合判断质量
        overall_score = (relevance_score * 0.6) + (completeness_score * 0.4)
        
        if overall_score >= 0.75 and not issues:
            quality = RetrievalQuality.HIGH
        elif overall_score >= 0.5:
            quality = RetrievalQuality.MEDIUM
        else:
            quality = RetrievalQuality.LOW
        
        return RetrievalEvaluation(
            quality=quality,
            issues=issues,
            relevance_score=relevance_score,
            completeness_score=completeness_score
        )
    
    def _calculate_relevance(self, question: str, docs: List[Dict]) -> float:
        """计算相关性"""
        question_keywords = set(self._extract_keywords(question))
        
        if not question_keywords:
            return 0.5
        
        total_relevance = 0.0
        for doc in docs:
            doc_keywords = set(self._extract_keywords(doc.get("content", "")))
            overlap = question_keywords.intersection(doc_keywords)
            relevance = len(overlap) / len(question_keywords)
            total_relevance += relevance
        
        return total_relevance / len(docs) if docs else 0.0
    
    def _calculate_completeness(self, question: str, docs: List[Dict]) -> float:
        """计算完整性"""
        # 这是一个简化的完整性评估
        # 实际应用中可能需要更复杂的模型
        
        # 检查是否覆盖了问题的不同方面
        question_aspects = self._get_question_aspects(question)
        doc_content = " ".join([d.get("content", "") for d in docs])
        
        covered_aspects = 0
        for aspect in question_aspects:
            if aspect in doc_content:
                covered_aspects += 1
        
        return covered_aspects / len(question_aspects) if question_aspects else 0.8
    
    def _has_conflicts(self, docs: List[Dict]) -> bool:
        """检查文档间是否有冲突"""
        # 简化的冲突检测
        texts = [doc.get("content", "") for doc in docs]
        
        # 检查数字冲突
        all_numbers = " ".join(re.findall(r'\d+', " ".join(texts)))
        if len(set(re.findall(r'\d+', all_numbers))) != len(re.findall(r'\d+', all_numbers)):
            return True
        
        # 检查肯定/否定冲突
        has_positive = any("是" in text or "可以" in text for text in texts)
        has_negative = any("不是" in text or "不可以" in text for text in texts)
        
        return has_positive and has_negative
    
    def _expand_query(self, question: str) -> str:
        """扩展查询"""
        prompt = f"""
        原始问题: "{question}"
        
        请将这个问题扩展为更详细、更具体的查询，以便更好地检索信息。
        例如，可以添加关键词、同义词或相关概念。
        
        只返回扩展后的问题。
        """
        
        return self.llm.generate(prompt)
    
    def _generate_final_answer(self, question: str, docs: List[Dict]) -> str:
        """生成最终答案"""
        context = "\n\n".join([f"文档 {i+1}: {doc.get('content', '')[:300]}" for i, doc in enumerate(docs)])
        
        prompt = f"""
        问题: "{question}"
        
        参考信息:
        {context}
        
        请根据以上信息，全面、准确地回答问题。
        """
        
        return self.llm.generate(prompt)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        words = re.findall(r'\w+', text.lower())
        stop_words = {'的', '了', '在', '是', '和', '与', '或', '但'}
        return [w for w in words if len(w) > 1 and w not in stop_words]
    
    def _get_question_aspects(self, question: str) -> List[str]:
        """获取问题方面"""
        aspects = []
        if "为什么" in question: aspects.append("原因")
        if "如何" in question: aspects.append("方法")
        if "比较" in question: aspects.append("对比")
        if not aspects: aspects.append(question)
        return aspects


# 模拟的LLM和检索器类
class MockLLM:
    def generate(self, prompt: str) -> str:
        if "扩展为更详细" in prompt:
            return f"扩展后的问题: {prompt.split('\"')[1]}"
        return f"模拟回答: {prompt[:50]}..."


class MockRetriever:
    def __init__(self, quality: str = "high"):
        self.quality = quality
    
    def search(self, query: str) -> List[Dict]:
        if self.quality == "low":
            return [{"content": "不相关的信息"}]
        return [
            {"content": f"关于'{query}'的相关信息1"},
            {"content": f"关于'{query}'的相关信息2"}
        ]


def main():
    """测试Corrective RAG"""
    llm = MockLLM()
    
    # 测试高质量检索
    print("=== 测试高质量检索 ===")
    retriever_high = MockRetriever(quality="high")
    crag_high = CorrectiveRAG(llm, retriever_high)
    result_high = crag_high.smart_retrieve_and_generate("什么是AI？")
    print(f"最终答案: {result_high['final_answer']}")
    print("修正日志:")
    for log in result_high['correction_log']:
        print(f"  - {log}")
    print("-" * 20)
    
    # 测试低质量检索
    print("=== 测试低质量检索 ===")
    retriever_low = MockRetriever(quality="low")
    crag_low = CorrectiveRAG(llm, retriever_low)
    result_low = crag_low.smart_retrieve_and_generate("什么是AI？")
    print(f"最终答案: {result_low['final_answer']}")
    print("修正日志:")
    for log in result_low['correction_log']:
        print(f"  - {log}")


if __name__ == "__main__":
    main()
