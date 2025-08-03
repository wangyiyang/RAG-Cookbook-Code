"""
置信度评估器
给AI回答打分，告诉用户答案有多靠谱
"""

import re
import json
import math
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class ConfidenceFactors:
    """置信度因素"""
    source_reliability: float = 0.0    # 来源可靠性
    information_consistency: float = 0.0  # 信息一致性
    answer_completeness: float = 0.0   # 回答完整性
    logical_coherence: float = 0.0     # 逻辑连贯性
    factual_support: float = 0.0       # 事实支撑度


class ConfidenceScorer:
    """置信度评分器：给AI回答打分"""
    
    def __init__(self):
        # 权重配置
        self.weights = {
            "source_reliability": 0.25,    # 来源可靠性权重
            "information_consistency": 0.25, # 信息一致性权重
            "answer_completeness": 0.20,   # 回答完整性权重
            "logical_coherence": 0.15,     # 逻辑连贯性权重
            "factual_support": 0.15        # 事实支撑度权重
        }
    
    def rate_answer_confidence(self, answer: str, sources: List[Dict], 
                              question: str = "") -> Dict[str, Any]:
        """给AI回答打分，告诉用户这个答案有多靠谱"""
        
        # 计算各维度得分
        factors = ConfidenceFactors()
        factors.source_reliability = self._assess_source_reliability(sources)
        factors.information_consistency = self._assess_information_consistency(answer, sources)
        factors.answer_completeness = self._assess_answer_completeness(answer, question)
        factors.logical_coherence = self._assess_logical_coherence(answer)
        factors.factual_support = self._assess_factual_support(answer, sources)
        
        # 计算加权总分
        overall_confidence = self._calculate_weighted_score(factors)
        
        # 生成可信度级别和建议
        confidence_level = self._get_confidence_level(overall_confidence)
        recommendations = self._get_recommendations(factors, overall_confidence)
        
        return {
            "overall_confidence": overall_confidence,
            "confidence_level": confidence_level,
            "factors": {
                "source_reliability": factors.source_reliability,
                "information_consistency": factors.information_consistency,
                "answer_completeness": factors.answer_completeness,
                "logical_coherence": factors.logical_coherence,
                "factual_support": factors.factual_support
            },
            "recommendations": recommendations,
            "user_guidance": self._get_user_guidance(overall_confidence)
        }
    
    def _assess_source_reliability(self, sources: List[Dict]) -> float:
        """评估来源可靠性（占25%）"""
        if not sources:
            return 0.3  # 无来源时给较低分
        
        reliability_score = 0.0
        total_weight = 0.0
        
        for source in sources:
            source_score = 0.5  # 基础分
            
            # 来源质量指标
            if source.get("title"):
                source_score += 0.1  # 有标题
            
            if source.get("author"):
                source_score += 0.1  # 有作者
            
            if source.get("url"):
                url = source.get("url", "")
                # 权威域名加分
                if any(domain in url for domain in [".edu", ".gov", ".org"]):
                    source_score += 0.2
                elif any(domain in url for domain in [".com", ".net"]):
                    source_score += 0.1
            
            if source.get("publish_date"):
                source_score += 0.1  # 有发布日期
            
            # 内容长度和质量
            content = source.get("content", "")
            if len(content) > 100:
                source_score += 0.1
            if len(content) > 500:
                source_score += 0.1
            
            weight = min(1.0, len(content) / 1000)  # 根据内容长度确定权重
            reliability_score += source_score * weight
            total_weight += weight
        
        if total_weight > 0:
            return min(1.0, reliability_score / total_weight)
        return 0.5
    
    def _assess_information_consistency(self, answer: str, sources: List[Dict]) -> float:
        """评估信息一致性（占25%）"""
        if not sources:
            return 0.5
        
        source_content = " ".join([s.get("content", "") for s in sources])
        
        # 提取关键事实
        answer_facts = self._extract_key_facts(answer)
        source_facts = self._extract_key_facts(source_content)
        
        if not answer_facts:
            return 0.6
        
        # 计算事实匹配度
        matching_facts = 0
        for fact in answer_facts:
            if self._fact_matches_sources(fact, source_facts):
                matching_facts += 1
        
        consistency_score = matching_facts / len(answer_facts)
        
        # 检查冲突信息
        conflicts = self._detect_conflicts(answer, source_content)
        if conflicts > 0:
            consistency_score *= (1 - conflicts * 0.1)  # 每个冲突降低10%
        
        return max(0.0, min(1.0, consistency_score))
    
    def _assess_answer_completeness(self, answer: str, question: str) -> float:
        """评估回答完整性（占20%）"""
        completeness_score = 0.6  # 基础分
        
        # 回答长度评估
        if len(answer) > 50:
            completeness_score += 0.1
        if len(answer) > 150:
            completeness_score += 0.1
        if len(answer) > 300:
            completeness_score += 0.1
        
        # 问题关键词覆盖度
        if question:
            question_keywords = self._extract_keywords(question)
            answer_keywords = self._extract_keywords(answer)
            
            if question_keywords:
                coverage = len(set(question_keywords) & set(answer_keywords)) / len(question_keywords)
                completeness_score += coverage * 0.2
        
        # 结构化程度
        if self._has_good_structure(answer):
            completeness_score += 0.1
        
        return min(1.0, completeness_score)
    
    def _assess_logical_coherence(self, answer: str) -> float:
        """评估逻辑连贯性（占15%）"""
        coherence_score = 0.7  # 基础分
        
        # 检查逻辑连接词
        logical_connectors = ['因此', '所以', '由于', '因为', '然而', '但是', '而且', '此外']
        has_connectors = any(conn in answer for conn in logical_connectors)
        if has_connectors:
            coherence_score += 0.1
        
        # 检查自相矛盾
        contradictions = self._find_contradictions(answer)
        if contradictions > 0:
            coherence_score -= contradictions * 0.2
        
        # 检查推理流畅性
        if self._has_smooth_flow(answer):
            coherence_score += 0.1
        
        return max(0.0, min(1.0, coherence_score))
    
    def _assess_factual_support(self, answer: str, sources: List[Dict]) -> float:
        """评估事实支撑度（占15%）"""
        if not sources:
            return 0.4
        
        source_text = " ".join([s.get("content", "") for s in sources])
        
        # 检查数字、日期等关键事实
        answer_numbers = re.findall(r'\d+(?:\.\d+)?%?', answer)
        source_numbers = re.findall(r'\d+(?:\.\d+)?%?', source_text)
        
        number_support = 0.0
        if answer_numbers:
            supported_numbers = sum(1 for num in answer_numbers if num in source_numbers)
            number_support = supported_numbers / len(answer_numbers)
        
        # 检查专有名词支撑
        proper_nouns = re.findall(r'[A-Z][a-zA-Z]+', answer)
        noun_support = 0.0
        if proper_nouns:
            supported_nouns = sum(1 for noun in proper_nouns if noun in source_text)
            noun_support = supported_nouns / len(proper_nouns)
        
        # 综合评分
        support_score = (number_support * 0.6 + noun_support * 0.4) if (answer_numbers or proper_nouns) else 0.7
        
        return min(1.0, support_score)
    
    def _calculate_weighted_score(self, factors: ConfidenceFactors) -> float:
        """计算加权总分"""
        total_score = (
            factors.source_reliability * self.weights["source_reliability"] +
            factors.information_consistency * self.weights["information_consistency"] +
            factors.answer_completeness * self.weights["answer_completeness"] +
            factors.logical_coherence * self.weights["logical_coherence"] +
            factors.factual_support * self.weights["factual_support"]
        )
        
        return min(1.0, max(0.0, total_score))
    
    def _extract_key_facts(self, text: str) -> List[str]:
        """提取关键事实"""
        sentences = re.split(r'[。！？]', text)
        facts = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                # 过滤不确定表述
                uncertain_patterns = ['可能', '也许', '大概', '似乎', '看起来']
                if not any(pattern in sentence for pattern in uncertain_patterns):
                    facts.append(sentence)
        
        return facts
    
    def _fact_matches_sources(self, fact: str, source_facts: List[str]) -> bool:
        """检查事实是否与来源匹配"""
        fact_keywords = set(self._extract_keywords(fact))
        
        for source_fact in source_facts:
            source_keywords = set(self._extract_keywords(source_fact))
            overlap = fact_keywords & source_keywords
            if len(overlap) >= min(3, len(fact_keywords) * 0.5):
                return True
        
        return False
    
    def _detect_conflicts(self, answer: str, source_content: str) -> int:
        """检测冲突信息"""
        conflicts = 0
        
        # 简单的冲突检测模式
        conflict_patterns = [
            (r'正确', r'错误'),
            (r'是', r'不是'),
            (r'有', r'没有'),
            (r'可以', r'不可以')
        ]
        
        for pos_pattern, neg_pattern in conflict_patterns:
            if (re.search(pos_pattern, answer) and re.search(neg_pattern, source_content)) or \
               (re.search(neg_pattern, answer) and re.search(pos_pattern, source_content)):
                conflicts += 1
        
        return conflicts
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        words = re.findall(r'\w+', text.lower())
        stop_words = {'的', '了', '在', '是', '和', '与', '或', '但', '如果', '那么', '这个', '那个'}
        return [w for w in words if len(w) > 1 and w not in stop_words]
    
    def _has_good_structure(self, answer: str) -> bool:
        """检查是否有良好的结构"""
        # 检查是否有列表、分点等结构
        structure_indicators = ['首先', '其次', '最后', '第一', '第二', '第三', '：', '1.', '2.', '•']
        return any(indicator in answer for indicator in structure_indicators)
    
    def _find_contradictions(self, text: str) -> int:
        """查找自相矛盾"""
        contradictions = 0
        
        contradiction_pairs = [
            ('正确', '错误'),
            ('是', '不是'),
            ('有', '没有'),
            ('可以', '不可以'),
            ('能', '不能')
        ]
        
        for pos, neg in contradiction_pairs:
            if re.search(pos, text) and re.search(neg, text):
                contradictions += 1
        
        return contradictions
    
    def _has_smooth_flow(self, answer: str) -> bool:
        """检查推理流畅性"""
        sentences = re.split(r'[。！？]', answer)
        if len(sentences) < 2:
            return True
        
        # 检查句子间的连接
        transitions = ['因此', '所以', '然而', '但是', '而且', '此外', '另外', '同时']
        return any(transition in answer for transition in transitions)
    
    def _get_confidence_level(self, score: float) -> str:
        """获取置信度级别"""
        if score >= 0.85:
            return "高可信度"
        elif score >= 0.70:
            return "较高可信度"
        elif score >= 0.55:
            return "中等可信度"
        elif score >= 0.40:
            return "较低可信度"
        else:
            return "低可信度"
    
    def _get_recommendations(self, factors: ConfidenceFactors, overall_score: float) -> List[str]:
        """获取改进建议"""
        recommendations = []
        
        if factors.source_reliability < 0.6:
            recommendations.append("建议使用更可靠的信息源")
        
        if factors.information_consistency < 0.6:
            recommendations.append("需要解决信息不一致问题")
        
        if factors.answer_completeness < 0.6:
            recommendations.append("回答需要更加完整和详细")
        
        if factors.logical_coherence < 0.6:
            recommendations.append("需要改善逻辑连贯性")
        
        if factors.factual_support < 0.6:
            recommendations.append("需要更多事实依据支撑")
        
        if overall_score < 0.5:
            recommendations.append("建议人工验证答案准确性")
        
        return recommendations
    
    def _get_user_guidance(self, confidence: float) -> str:
        """获取用户指导"""
        if confidence >= 0.85:
            return "此答案可信度很高，可以放心使用"
        elif confidence >= 0.70:
            return "此答案可信度较高，建议谨慎使用"
        elif confidence >= 0.55:
            return "此答案可信度中等，建议核实关键信息"
        elif confidence >= 0.40:
            return "此答案可信度较低，建议多方验证"
        else:
            return "此答案可信度很低，建议寻求其他信息源"


def main():
    """测试置信度评估器"""
    scorer = ConfidenceScorer()
    
    # 测试用例
    question = "什么是深度学习？"
    answer = "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的复杂模式。它在图像识别、自然语言处理等领域取得了显著成功。深度学习需要大量数据和计算资源。"
    sources = [
        {
            "content": "深度学习是机器学习的子集，使用具有多个隐层的神经网络。它能够自动提取特征，在计算机视觉、语音识别等任务中表现出色。",
            "title": "深度学习入门",
            "url": "https://example.edu/deep-learning"
        },
        {
            "content": "神经网络通过反向传播算法训练，深度学习需要GPU等高性能计算设备支持。",
            "author": "张三",
            "publish_date": "2023-01-01"
        }
    ]
    
    print("=== 置信度评估测试 ===")
    print(f"问题: {question}")
    print(f"回答: {answer}")
    print()
    
    result = scorer.rate_answer_confidence(answer, sources, question)
    
    print(f"整体置信度: {result['overall_confidence']:.2f}")
    print(f"置信度级别: {result['confidence_level']}")
    print()
    
    print("各维度得分:")
    for factor, score in result['factors'].items():
        print(f"  {factor}: {score:.2f}")
    
    print()
    print("改进建议:")
    for rec in result['recommendations']:
        print(f"  - {rec}")
    
    print()
    print(f"用户指导: {result['user_guidance']}")


if __name__ == "__main__":
    main()
