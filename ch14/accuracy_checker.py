"""
RAG准确性诊断工具
快速发现RAG系统的准确性问题
"""

import re
import json
from typing import List, Dict, Any
from datetime import datetime
import numpy as np


class AccuracyChecker:
    """RAG准确性诊断工具：10秒钟找出问题所在"""
    
    def __init__(self):
        self.issue_patterns = {
            "hallucination": [
                r"据我了解", r"根据常识", r"一般来说", 
                r"众所周知", r"通常情况下", r"我认为"
            ],
            "uncertainty": [
                r"可能", r"也许", r"大概", r"估计", 
                r"应该", r"似乎", r"看起来"
            ],
            "incomplete": [
                r"等等", r"之类的", r"诸如此类", 
                r"更多信息请", r"详情请"
            ]
        }
    
    def quick_diagnosis(self, query: str, answer: str, sources: List[Dict]) -> Dict:
        """快速诊断RAG回答质量，告诉你哪里有问题"""
        issues = []
        scores = {}
        
        # 检查是否在瞎编
        hallucination_score = self._check_hallucination(answer, sources)
        if hallucination_score > 0.3:
            issues.append("AI在瞎编，没有依据")
        scores["hallucination"] = hallucination_score
        
        # 检查逻辑是否合理  
        logic_score = self._check_logic(query, answer)
        if logic_score < 0.7:
            issues.append("推理逻辑有问题")
        scores["logic"] = logic_score
        
        # 检查信息是否完整
        completeness_score = self._check_completeness(query, answer)
        if completeness_score < 0.8:
            issues.append("回答不够完整")
        scores["completeness"] = completeness_score
        
        # 检查答案一致性
        consistency_score = self._check_consistency(answer, sources)
        if consistency_score < 0.7:
            issues.append("答案与源文档不一致")
        scores["consistency"] = consistency_score
        
        overall_score = np.mean(list(scores.values()))
        
        return {
            "issues": issues,
            "scores": scores,
            "overall_score": overall_score,
            "diagnosis_time": datetime.now().isoformat(),
            "recommendation": self._get_recommendation(issues)
        }
    
    def _check_hallucination(self, answer: str, sources: List[Dict]) -> float:
        """检查AI是否在瞎编内容"""
        if not sources:
            return 0.8  # 没有源文档，可能在瞎编
        
        # 检查幻觉模式
        hallucination_count = 0
        for pattern_type, patterns in self.issue_patterns.items():
            if pattern_type == "hallucination":
                for pattern in patterns:
                    if re.search(pattern, answer):
                        hallucination_count += 1
        
        # 检查答案中的事实是否有源文档支持
        source_text = " ".join([doc.get("content", "") for doc in sources])
        answer_sentences = answer.split("。")
        unsupported_count = 0
        
        for sentence in answer_sentences:
            if len(sentence.strip()) > 5:  # 忽略太短的句子
                if not self._has_source_support(sentence, source_text):
                    unsupported_count += 1
        
        if len(answer_sentences) > 0:
            unsupported_ratio = unsupported_count / len(answer_sentences)
        else:
            unsupported_ratio = 0
        
        return min(1.0, (hallucination_count * 0.2 + unsupported_ratio * 0.8))
    
    def _check_logic(self, query: str, answer: str) -> float:
        """检查推理逻辑是否合理"""
        logic_issues = 0
        
        # 检查答案是否回答了问题
        if not self._answers_question(query, answer):
            logic_issues += 1
        
        # 检查答案是否自相矛盾
        if self._has_contradiction(answer):
            logic_issues += 1
        
        # 检查推理链是否合理
        if not self._has_reasonable_flow(answer):
            logic_issues += 1
        
        return max(0.0, 1.0 - logic_issues * 0.3)
    
    def _check_completeness(self, query: str, answer: str) -> float:
        """检查回答是否完整"""
        completeness_score = 1.0
        
        # 检查是否包含不完整的标识
        for pattern in self.issue_patterns["incomplete"]:
            if re.search(pattern, answer):
                completeness_score -= 0.2
        
        # 检查答案长度是否合理
        if len(answer) < 20:
            completeness_score -= 0.3
        
        # 检查是否回答了查询的所有部分
        query_keywords = self._extract_keywords(query)
        answer_coverage = self._calculate_keyword_coverage(query_keywords, answer)
        completeness_score *= answer_coverage
        
        return max(0.0, completeness_score)
    
    def _check_consistency(self, answer: str, sources: List[Dict]) -> float:
        """检查答案与源文档的一致性"""
        if not sources:
            return 0.5
        
        source_text = " ".join([doc.get("content", "") for doc in sources])
        
        # 简单的一致性检查：关键信息是否匹配
        answer_facts = self._extract_facts(answer)
        source_facts = self._extract_facts(source_text)
        
        matching_facts = 0
        for fact in answer_facts:
            if any(self._facts_similar(fact, sf) for sf in source_facts):
                matching_facts += 1
        
        if len(answer_facts) == 0:
            return 0.5
        
        return matching_facts / len(answer_facts)
    
    def _has_source_support(self, sentence: str, source_text: str) -> bool:
        """检查句子是否有源文档支持"""
        sentence_keywords = self._extract_keywords(sentence)
        source_keywords = self._extract_keywords(source_text)
        
        # 简单的关键词匹配
        matches = sum(1 for kw in sentence_keywords if kw in source_keywords)
        return matches >= len(sentence_keywords) * 0.3
    
    def _answers_question(self, query: str, answer: str) -> bool:
        """检查答案是否回答了问题"""
        query_keywords = self._extract_keywords(query)
        answer_keywords = self._extract_keywords(answer)
        
        # 检查关键词覆盖度
        coverage = sum(1 for kw in query_keywords if kw in answer_keywords)
        return coverage >= len(query_keywords) * 0.5
    
    def _has_contradiction(self, text: str) -> bool:
        """检查文本是否自相矛盾"""
        contradiction_patterns = [
            (r"正确", r"错误"),
            (r"是", r"不是"),
            (r"有", r"没有"),
            (r"可以", r"不可以"),
            (r"能", r"不能")
        ]
        
        for pos_pattern, neg_pattern in contradiction_patterns:
            if re.search(pos_pattern, text) and re.search(neg_pattern, text):
                return True
        
        return False
    
    def _has_reasonable_flow(self, text: str) -> bool:
        """检查推理流程是否合理"""
        sentences = text.split("。")
        if len(sentences) < 2:
            return True
        
        # 检查是否有适当的连接词
        flow_indicators = ["因此", "所以", "由于", "因为", "首先", "其次", "最后"]
        has_flow = any(indicator in text for indicator in flow_indicators)
        
        return has_flow or len(sentences) <= 3
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取
        words = re.findall(r'\w+', text.lower())
        # 过滤停用词
        stop_words = {"的", "了", "在", "是", "和", "与", "或", "但", "如果", "那么"}
        keywords = [w for w in words if len(w) > 1 and w not in stop_words]
        return keywords
    
    def _calculate_keyword_coverage(self, query_keywords: List[str], answer: str) -> float:
        """计算关键词覆盖度"""
        if not query_keywords:
            return 1.0
        
        answer_keywords = self._extract_keywords(answer)
        covered = sum(1 for kw in query_keywords if kw in answer_keywords)
        return covered / len(query_keywords)
    
    def _extract_facts(self, text: str) -> List[str]:
        """提取事实性陈述"""
        sentences = text.split("。")
        facts = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 5 and not self._is_uncertain_statement(sentence):
                facts.append(sentence)
        
        return facts
    
    def _is_uncertain_statement(self, sentence: str) -> bool:
        """判断是否为不确定陈述"""
        for pattern in self.issue_patterns["uncertainty"]:
            if re.search(pattern, sentence):
                return True
        return False
    
    def _facts_similar(self, fact1: str, fact2: str) -> bool:
        """判断两个事实是否相似"""
        keywords1 = set(self._extract_keywords(fact1))
        keywords2 = set(self._extract_keywords(fact2))
        
        if not keywords1 or not keywords2:
            return False
        
        intersection = keywords1.intersection(keywords2)
        return len(intersection) >= min(len(keywords1), len(keywords2)) * 0.5
    
    def _get_recommendation(self, issues: List[str]) -> str:
        """根据发现的问题给出建议"""
        if not issues:
            return "回答质量良好，无明显问题"
        
        recommendations = {
            "AI在瞎编，没有依据": "建议启用事实检查机制，确保答案基于源文档",
            "推理逻辑有问题": "建议使用多步推理，让AI逐步分析问题",
            "回答不够完整": "建议改进检索策略，获取更全面的信息",
            "答案与源文档不一致": "建议加强答案与源文档的对齐检查"
        }
        
        return "; ".join([recommendations.get(issue, "需要人工检查") for issue in issues])


def main():
    """测试诊断工具"""
    checker = AccuracyChecker()
    
    # 测试用例
    query = "什么是Transformer的注意力机制？"
    answer = "Transformer的注意力机制是一种让模型能够关注输入序列中不同位置的技术。据我了解，它主要包括自注意力和交叉注意力两种。"
    sources = [
        {"content": "Transformer模型使用了多头自注意力机制，能够并行处理序列中的所有位置。"},
        {"content": "注意力机制通过计算查询、键和值的相似度来确定各位置的重要性。"}
    ]
    
    result = checker.quick_diagnosis(query, answer, sources)
    
    print("=== RAG准确性诊断结果 ===")
    print(f"整体得分: {result['overall_score']:.2f}")
    print(f"发现问题: {result['issues']}")
    print(f"各维度得分: {result['scores']}")
    print(f"建议: {result['recommendation']}")


if __name__ == "__main__":
    main()
