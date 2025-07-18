"""
法律质量验证器
实现多重验证机制、风险评估和质量保证
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import jieba
from collections import Counter, defaultdict
import numpy as np


class ValidationLevel(Enum):
    """验证级别"""
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    CRITICAL_RISK = "critical_risk"


class ValidationAspect(Enum):
    """验证方面"""
    FACT_ACCURACY = "fact_accuracy"
    CITATION_CORRECTNESS = "citation_correctness"
    LOGIC_CONSISTENCY = "logic_consistency"
    APPLICABILITY = "applicability"
    COMPLETENESS = "completeness"


@dataclass
class ValidationResult:
    """验证结果"""
    aspect: ValidationAspect
    score: float
    risk_level: ValidationLevel
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """质量报告"""
    overall_score: float
    overall_risk_level: ValidationLevel
    validation_results: List[ValidationResult]
    risk_summary: Dict[str, Any]
    recommendations: List[str]
    confidence_score: float


class LegalQualityValidator:
    """法律质量验证器"""
    
    def __init__(self):
        self.legal_terms = self._load_legal_terms()
        self.law_database = self._load_law_database()
        self.risk_patterns = self._load_risk_patterns()
        self.quality_metrics = self._load_quality_metrics()
        
    def validate_legal_content_like_expert(
        self, 
        legal_content: str, 
        query_context: str = "", 
        sources: List[Dict] = None
    ) -> QualityReport:
        """像法律专家一样验证内容质量"""
        
        sources = sources or []
        validation_results = []
        
        # 1. 事实准确性验证
        fact_result = self._validate_fact_accuracy(legal_content, sources)
        validation_results.append(fact_result)
        
        # 2. 引用正确性验证
        citation_result = self._validate_citation_correctness(legal_content)
        validation_results.append(citation_result)
        
        # 3. 逻辑一致性验证
        logic_result = self._validate_logic_consistency(legal_content)
        validation_results.append(logic_result)
        
        # 4. 适用性验证
        applicability_result = self._validate_applicability(legal_content, query_context)
        validation_results.append(applicability_result)
        
        # 5. 完整性验证
        completeness_result = self._validate_completeness(legal_content, query_context)
        validation_results.append(completeness_result)
        
        # 6. 综合评估
        quality_report = self._generate_quality_report(validation_results, legal_content)
        
        return quality_report
    
    def _validate_fact_accuracy(self, content: str, sources: List[Dict]) -> ValidationResult:
        """验证事实准确性"""
        issues = []
        suggestions = []
        evidence = []
        
        # 提取法律事实陈述
        fact_statements = self._extract_fact_statements(content)
        
        # 验证每个事实陈述
        accuracy_scores = []
        for statement in fact_statements:
            # 与源材料对比
            source_support = self._check_source_support(statement, sources)
            
            # 事实一致性检查
            consistency_score = self._check_fact_consistency(statement, content)
            
            # 法律事实的准确性
            legal_accuracy = self._check_legal_fact_accuracy(statement)
            
            statement_score = (source_support * 0.4 + 
                             consistency_score * 0.3 + 
                             legal_accuracy * 0.3)
            accuracy_scores.append(statement_score)
            
            if statement_score < 0.7:
                issues.append(f"事实陈述可能不准确: {statement[:50]}...")
                suggestions.append("建议核实相关法律事实的准确性")
            
            if source_support > 0.8:
                evidence.append(f"事实陈述有充分源材料支持: {statement[:30]}...")
        
        # 计算总体事实准确性分数
        overall_score = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.5
        
        # 判断风险级别
        if overall_score >= 0.9:
            risk_level = ValidationLevel.LOW_RISK
        elif overall_score >= 0.7:
            risk_level = ValidationLevel.MEDIUM_RISK
        elif overall_score >= 0.5:
            risk_level = ValidationLevel.HIGH_RISK
        else:
            risk_level = ValidationLevel.CRITICAL_RISK
        
        return ValidationResult(
            aspect=ValidationAspect.FACT_ACCURACY,
            score=overall_score,
            risk_level=risk_level,
            issues=issues,
            suggestions=suggestions,
            evidence=evidence
        )
    
    def _validate_citation_correctness(self, content: str) -> ValidationResult:
        """验证引用正确性"""
        issues = []
        suggestions = []
        evidence = []
        
        # 提取所有法律引用
        citations = self._extract_legal_citations(content)
        
        if not citations:
            issues.append("未发现法律引用，可能缺乏法律依据")
            suggestions.append("建议添加相关法律条文引用")
            return ValidationResult(
                aspect=ValidationAspect.CITATION_CORRECTNESS,
                score=0.3,
                risk_level=ValidationLevel.HIGH_RISK,
                issues=issues,
                suggestions=suggestions
            )
        
        # 验证每个引用
        citation_scores = []
        for citation in citations:
            # 格式正确性
            format_score = self._check_citation_format(citation)
            
            # 法条存在性
            existence_score = self._check_law_existence(citation)
            
            # 引用适当性
            appropriateness_score = self._check_citation_appropriateness(citation, content)
            
            citation_score = (format_score * 0.3 + 
                            existence_score * 0.4 + 
                            appropriateness_score * 0.3)
            citation_scores.append(citation_score)
            
            if format_score < 0.8:
                issues.append(f"引用格式可能不规范: {citation}")
                suggestions.append("建议检查法条引用格式")
            
            if existence_score < 0.8:
                issues.append(f"引用法条可能不存在或已失效: {citation}")
                suggestions.append("建议核实法条的现行有效性")
            
            if appropriateness_score > 0.8:
                evidence.append(f"引用恰当: {citation}")
        
        # 计算总体引用正确性分数
        overall_score = sum(citation_scores) / len(citation_scores) if citation_scores else 0.5
        
        # 判断风险级别
        if overall_score >= 0.9:
            risk_level = ValidationLevel.LOW_RISK
        elif overall_score >= 0.7:
            risk_level = ValidationLevel.MEDIUM_RISK
        elif overall_score >= 0.5:
            risk_level = ValidationLevel.HIGH_RISK
        else:
            risk_level = ValidationLevel.CRITICAL_RISK
        
        return ValidationResult(
            aspect=ValidationAspect.CITATION_CORRECTNESS,
            score=overall_score,
            risk_level=risk_level,
            issues=issues,
            suggestions=suggestions,
            evidence=evidence
        )
    
    def _validate_logic_consistency(self, content: str) -> ValidationResult:
        """验证逻辑一致性"""
        issues = []
        suggestions = []
        evidence = []
        
        # 提取逻辑关系
        logical_statements = self._extract_logical_statements(content)
        
        # 检查逻辑矛盾
        contradictions = self._detect_logical_contradictions(logical_statements)
        
        # 检查推理链完整性
        reasoning_chains = self._extract_reasoning_chains(content)
        incomplete_chains = self._check_reasoning_completeness(reasoning_chains)
        
        # 检查前提结论一致性
        premise_conclusion_consistency = self._check_premise_conclusion_consistency(content)
        
        # 计算逻辑一致性分数
        contradiction_penalty = len(contradictions) * 0.2
        incomplete_penalty = len(incomplete_chains) * 0.15
        
        logic_score = max(0.0, 1.0 - contradiction_penalty - incomplete_penalty)
        logic_score = logic_score * premise_conclusion_consistency
        
        # 生成问题报告
        for contradiction in contradictions:
            issues.append(f"发现逻辑矛盾: {contradiction}")
            suggestions.append("建议检查并解决逻辑矛盾")
        
        for incomplete_chain in incomplete_chains:
            issues.append(f"推理链不完整: {incomplete_chain}")
            suggestions.append("建议补充完整的推理过程")
        
        if premise_conclusion_consistency > 0.8:
            evidence.append("前提与结论逻辑一致")
        
        # 判断风险级别
        if logic_score >= 0.9:
            risk_level = ValidationLevel.LOW_RISK
        elif logic_score >= 0.7:
            risk_level = ValidationLevel.MEDIUM_RISK
        elif logic_score >= 0.5:
            risk_level = ValidationLevel.HIGH_RISK
        else:
            risk_level = ValidationLevel.CRITICAL_RISK
        
        return ValidationResult(
            aspect=ValidationAspect.LOGIC_CONSISTENCY,
            score=logic_score,
            risk_level=risk_level,
            issues=issues,
            suggestions=suggestions,
            evidence=evidence
        )
    
    def _validate_applicability(self, content: str, query_context: str) -> ValidationResult:
        """验证适用性"""
        issues = []
        suggestions = []
        evidence = []
        
        # 提取适用条件
        applicable_conditions = self._extract_applicable_conditions(content)
        
        # 分析查询上下文
        context_elements = self._analyze_query_context(query_context)
        
        # 检查适用性匹配
        applicability_score = self._calculate_applicability_match(
            applicable_conditions, context_elements
        )
        
        # 检查法条适用范围
        law_scope_match = self._check_law_scope_applicability(content, query_context)
        
        # 检查时效性
        temporal_applicability = self._check_temporal_applicability(content)
        
        # 综合适用性评分
        overall_applicability = (applicability_score * 0.4 + 
                               law_scope_match * 0.4 + 
                               temporal_applicability * 0.2)
        
        # 生成适用性报告
        if applicability_score < 0.6:
            issues.append("内容与查询情况的适用性较低")
            suggestions.append("建议确认法律建议是否适用于具体情况")
        
        if law_scope_match < 0.6:
            issues.append("引用法条的适用范围可能不匹配")
            suggestions.append("建议核实法条的适用范围")
        
        if temporal_applicability < 0.8:
            issues.append("法律内容的时效性可能存在问题")
            suggestions.append("建议确认法律法规的现行有效性")
        
        if overall_applicability > 0.8:
            evidence.append("法律建议与查询情况适用性良好")
        
        # 判断风险级别
        if overall_applicability >= 0.9:
            risk_level = ValidationLevel.LOW_RISK
        elif overall_applicability >= 0.7:
            risk_level = ValidationLevel.MEDIUM_RISK
        elif overall_applicability >= 0.5:
            risk_level = ValidationLevel.HIGH_RISK
        else:
            risk_level = ValidationLevel.CRITICAL_RISK
        
        return ValidationResult(
            aspect=ValidationAspect.APPLICABILITY,
            score=overall_applicability,
            risk_level=risk_level,
            issues=issues,
            suggestions=suggestions,
            evidence=evidence
        )
    
    def _validate_completeness(self, content: str, query_context: str) -> ValidationResult:
        """验证完整性"""
        issues = []
        suggestions = []
        evidence = []
        
        # 分析查询需求
        query_requirements = self._analyze_query_requirements(query_context)
        
        # 检查内容覆盖度
        content_coverage = self._check_content_coverage(content, query_requirements)
        
        # 检查必要要素
        essential_elements = self._check_essential_legal_elements(content)
        
        # 检查风险提示
        risk_disclosure = self._check_risk_disclosure(content)
        
        # 综合完整性评分
        completeness_score = (content_coverage * 0.5 + 
                            essential_elements * 0.3 + 
                            risk_disclosure * 0.2)
        
        # 生成完整性报告
        if content_coverage < 0.7:
            issues.append("内容可能未充分回应查询需求")
            suggestions.append("建议补充相关法律分析")
        
        if essential_elements < 0.8:
            issues.append("缺少必要的法律要素")
            suggestions.append("建议补充法律适用条件、法律后果等要素")
        
        if risk_disclosure < 0.6:
            issues.append("缺少必要的风险提示")
            suggestions.append("建议添加相关法律风险提示")
        
        if completeness_score > 0.8:
            evidence.append("内容覆盖较为全面")
        
        # 判断风险级别
        if completeness_score >= 0.9:
            risk_level = ValidationLevel.LOW_RISK
        elif completeness_score >= 0.7:
            risk_level = ValidationLevel.MEDIUM_RISK
        elif completeness_score >= 0.5:
            risk_level = ValidationLevel.HIGH_RISK
        else:
            risk_level = ValidationLevel.CRITICAL_RISK
        
        return ValidationResult(
            aspect=ValidationAspect.COMPLETENESS,
            score=completeness_score,
            risk_level=risk_level,
            issues=issues,
            suggestions=suggestions,
            evidence=evidence
        )
    
    def _generate_quality_report(
        self, 
        validation_results: List[ValidationResult], 
        content: str
    ) -> QualityReport:
        """生成质量报告"""
        
        # 计算总体分数
        aspect_weights = {
            ValidationAspect.FACT_ACCURACY: 0.25,
            ValidationAspect.CITATION_CORRECTNESS: 0.25,
            ValidationAspect.LOGIC_CONSISTENCY: 0.20,
            ValidationAspect.APPLICABILITY: 0.20,
            ValidationAspect.COMPLETENESS: 0.10
        }
        
        weighted_scores = []
        for result in validation_results:
            weight = aspect_weights.get(result.aspect, 0.2)
            weighted_scores.append(result.score * weight)
        
        overall_score = sum(weighted_scores)
        
        # 确定总体风险级别
        overall_risk_level = self._determine_overall_risk_level(validation_results)
        
        # 生成风险摘要
        risk_summary = self._generate_risk_summary(validation_results)
        
        # 生成建议
        recommendations = self._generate_recommendations(validation_results, overall_score)
        
        # 计算置信度
        confidence_score = self._calculate_confidence_score(validation_results, content)
        
        return QualityReport(
            overall_score=overall_score,
            overall_risk_level=overall_risk_level,
            validation_results=validation_results,
            risk_summary=risk_summary,
            recommendations=recommendations,
            confidence_score=confidence_score
        )
    
    def _extract_fact_statements(self, content: str) -> List[str]:
        """提取事实陈述"""
        # 使用正则表达式提取事实性陈述
        fact_patterns = [
            r'[^。]*事实[^。]*。',
            r'[^。]*情况[^。]*。',
            r'[^。]*发生[^。]*。',
            r'[^。]*约定[^。]*。',
            r'[^。]*规定[^。]*。'
        ]
        
        statements = []
        for pattern in fact_patterns:
            matches = re.findall(pattern, content)
            statements.extend(matches)
        
        return statements
    
    def _check_source_support(self, statement: str, sources: List[Dict]) -> float:
        """检查源材料支持度"""
        if not sources:
            return 0.3
        
        # 计算陈述与源材料的相似度
        support_scores = []
        for source in sources:
            source_content = source.get('content', '')
            similarity = self._calculate_text_similarity(statement, source_content)
            support_scores.append(similarity)
        
        return max(support_scores) if support_scores else 0.3
    
    def _check_fact_consistency(self, statement: str, content: str) -> float:
        """检查事实一致性"""
        # 检查陈述与内容的一致性
        consistency_indicators = ['一致', '符合', '相符', '吻合']
        inconsistency_indicators = ['矛盾', '冲突', '不符', '相悖']
        
        consistency_score = 0.5
        
        for indicator in consistency_indicators:
            if indicator in content:
                consistency_score += 0.1
        
        for indicator in inconsistency_indicators:
            if indicator in content:
                consistency_score -= 0.2
        
        return max(0.0, min(1.0, consistency_score))
    
    def _check_legal_fact_accuracy(self, statement: str) -> float:
        """检查法律事实准确性"""
        # 检查法律术语使用的准确性
        accuracy_score = 0.5
        
        # 检查法律术语
        legal_terms_used = [term for term in self.legal_terms if term in statement]
        if legal_terms_used:
            accuracy_score += 0.3
        
        # 检查明显错误
        error_patterns = ['错误', '不当', '违法', '无效']
        for pattern in error_patterns:
            if pattern in statement:
                accuracy_score -= 0.2
        
        return max(0.0, min(1.0, accuracy_score))
    
    def _extract_legal_citations(self, content: str) -> List[str]:
        """提取法律引用"""
        citation_patterns = [
            r'《[^》]+》第\d+条',
            r'《[^》]+》第\d+条第\d+款',
            r'[^《》]+法第\d+条',
            r'最高人民法院.*?第\d+号'
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, content)
            citations.extend(matches)
        
        return citations
    
    def _check_citation_format(self, citation: str) -> float:
        """检查引用格式"""
        # 检查引用格式的规范性
        format_score = 0.5
        
        # 书名号检查
        if '《' in citation and '》' in citation:
            format_score += 0.3
        
        # 条文号检查
        if re.search(r'第\d+条', citation):
            format_score += 0.2
        
        return min(1.0, format_score)
    
    def _check_law_existence(self, citation: str) -> float:
        """检查法律存在性"""
        # 简化实现：检查是否在法律数据库中
        for law in self.law_database:
            if law['name'] in citation:
                return 0.9
        
        return 0.5  # 无法确认
    
    def _check_citation_appropriateness(self, citation: str, content: str) -> float:
        """检查引用适当性"""
        # 检查引用是否与内容相关
        appropriateness_score = 0.5
        
        # 上下文相关性
        context_words = content.split()
        citation_words = citation.split()
        
        common_words = set(context_words) & set(citation_words)
        if common_words:
            appropriateness_score += min(len(common_words) * 0.1, 0.4)
        
        return min(1.0, appropriateness_score)
    
    def _extract_logical_statements(self, content: str) -> List[str]:
        """提取逻辑陈述"""
        logical_patterns = [
            r'因此[^。]*。',
            r'所以[^。]*。',
            r'由此[^。]*。',
            r'综上[^。]*。',
            r'根据[^。]*。'
        ]
        
        statements = []
        for pattern in logical_patterns:
            matches = re.findall(pattern, content)
            statements.extend(matches)
        
        return statements
    
    def _detect_logical_contradictions(self, statements: List[str]) -> List[str]:
        """检测逻辑矛盾"""
        contradictions = []
        
        # 检查否定表达
        positive_statements = [s for s in statements if not any(neg in s for neg in ['不', '非', '无', '否'])]
        negative_statements = [s for s in statements if any(neg in s for neg in ['不', '非', '无', '否'])]
        
        # 简化的矛盾检测
        for pos in positive_statements:
            for neg in negative_statements:
                if self._calculate_text_similarity(pos, neg) > 0.7:
                    contradictions.append(f"{pos} 与 {neg} 可能存在矛盾")
        
        return contradictions
    
    def _extract_reasoning_chains(self, content: str) -> List[str]:
        """提取推理链"""
        reasoning_patterns = [
            r'由于[^。]*，因此[^。]*。',
            r'根据[^。]*，所以[^。]*。',
            r'鉴于[^。]*，故[^。]*。'
        ]
        
        chains = []
        for pattern in reasoning_patterns:
            matches = re.findall(pattern, content)
            chains.extend(matches)
        
        return chains
    
    def _check_reasoning_completeness(self, chains: List[str]) -> List[str]:
        """检查推理完整性"""
        incomplete_chains = []
        
        for chain in chains:
            # 检查是否包含前提和结论
            if '由于' in chain or '根据' in chain:
                if not ('因此' in chain or '所以' in chain or '故' in chain):
                    incomplete_chains.append(chain)
        
        return incomplete_chains
    
    def _check_premise_conclusion_consistency(self, content: str) -> float:
        """检查前提结论一致性"""
        # 简化实现：检查逻辑连接词的使用
        logical_connectors = ['因此', '所以', '故', '由此', '综上']
        connector_count = sum(1 for connector in logical_connectors if connector in content)
        
        if connector_count > 0:
            return 0.8
        else:
            return 0.5
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        # 简化的相似度计算
        words1 = set(jieba.cut(text1))
        words2 = set(jieba.cut(text2))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _determine_overall_risk_level(self, validation_results: List[ValidationResult]) -> ValidationLevel:
        """确定总体风险级别"""
        risk_levels = [result.risk_level for result in validation_results]
        
        # 如果有任何关键风险，总体为关键风险
        if ValidationLevel.CRITICAL_RISK in risk_levels:
            return ValidationLevel.CRITICAL_RISK
        
        # 如果高风险项目大于等于2个，总体为高风险
        high_risk_count = risk_levels.count(ValidationLevel.HIGH_RISK)
        if high_risk_count >= 2:
            return ValidationLevel.HIGH_RISK
        
        # 如果有高风险项目，总体为中风险
        if ValidationLevel.HIGH_RISK in risk_levels:
            return ValidationLevel.MEDIUM_RISK
        
        # 如果中风险项目大于等于3个，总体为中风险
        medium_risk_count = risk_levels.count(ValidationLevel.MEDIUM_RISK)
        if medium_risk_count >= 3:
            return ValidationLevel.MEDIUM_RISK
        
        # 否则为低风险
        return ValidationLevel.LOW_RISK
    
    def _generate_risk_summary(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """生成风险摘要"""
        risk_counts = Counter([result.risk_level for result in validation_results])
        
        total_issues = sum(len(result.issues) for result in validation_results)
        total_suggestions = sum(len(result.suggestions) for result in validation_results)
        
        return {
            'risk_distribution': {
                'low_risk': risk_counts.get(ValidationLevel.LOW_RISK, 0),
                'medium_risk': risk_counts.get(ValidationLevel.MEDIUM_RISK, 0),
                'high_risk': risk_counts.get(ValidationLevel.HIGH_RISK, 0),
                'critical_risk': risk_counts.get(ValidationLevel.CRITICAL_RISK, 0)
            },
            'total_issues': total_issues,
            'total_suggestions': total_suggestions,
            'most_problematic_aspects': [
                result.aspect.value for result in validation_results
                if result.risk_level in [ValidationLevel.HIGH_RISK, ValidationLevel.CRITICAL_RISK]
            ]
        }
    
    def _generate_recommendations(self, validation_results: List[ValidationResult], overall_score: float) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于总体分数的建议
        if overall_score < 0.5:
            recommendations.append("内容质量偏低，强烈建议咨询专业律师")
        elif overall_score < 0.7:
            recommendations.append("内容存在一定风险，建议谨慎参考并咨询专业律师")
        elif overall_score < 0.9:
            recommendations.append("内容质量良好，但建议结合具体情况谨慎应用")
        else:
            recommendations.append("内容质量很高，可以作为参考依据")
        
        # 基于具体验证结果的建议
        for result in validation_results:
            if result.risk_level in [ValidationLevel.HIGH_RISK, ValidationLevel.CRITICAL_RISK]:
                recommendations.extend(result.suggestions)
        
        return list(set(recommendations))  # 去重
    
    def _calculate_confidence_score(self, validation_results: List[ValidationResult], content: str) -> float:
        """计算置信度分数"""
        # 基于各项验证结果计算置信度
        scores = [result.score for result in validation_results]
        average_score = sum(scores) / len(scores) if scores else 0.5
        
        # 基于内容长度的置信度调整
        content_length = len(content)
        if content_length < 100:
            length_factor = 0.8
        elif content_length < 500:
            length_factor = 0.9
        else:
            length_factor = 1.0
        
        # 基于证据数量的置信度调整
        total_evidence = sum(len(result.evidence) for result in validation_results)
        evidence_factor = min(1.0, total_evidence / 10.0)
        
        confidence_score = average_score * length_factor * evidence_factor
        
        return confidence_score
    
    def _load_legal_terms(self) -> Set[str]:
        """加载法律术语"""
        return {
            '合同', '协议', '违约', '侵权', '赔偿', '责任', '权利', '义务',
            '法律', '法规', '条例', '办法', '规定', '判决', '裁定', '调解',
            '诉讼', '仲裁', '执行', '上诉', '申请', '原告', '被告', '第三人',
            '证据', '证明', '举证', '质证', '认定', '民事', '刑事', '行政'
        }
    
    def _load_law_database(self) -> List[Dict[str, str]]:
        """加载法律数据库"""
        return [
            {'name': '中华人民共和国民法典', 'status': 'active'},
            {'name': '中华人民共和国刑法', 'status': 'active'},
            {'name': '中华人民共和国行政法', 'status': 'active'},
            {'name': '中华人民共和国合同法', 'status': 'repealed'},
            {'name': '中华人民共和国侵权责任法', 'status': 'repealed'}
        ]
    
    def _load_risk_patterns(self) -> List[str]:
        """加载风险模式"""
        return [
            r'一定[^。]*',
            r'必须[^。]*',
            r'绝对[^。]*',
            r'肯定[^。]*'
        ]
    
    def _load_quality_metrics(self) -> Dict[str, float]:
        """加载质量指标"""
        return {
            'min_fact_accuracy': 0.8,
            'min_citation_correctness': 0.9,
            'min_logic_consistency': 0.7,
            'min_applicability': 0.8,
            'min_completeness': 0.7
        }
    
    # 以下方法为简化实现，实际应用中需要更复杂的逻辑
    def _extract_applicable_conditions(self, content: str) -> List[str]:
        """提取适用条件"""
        return re.findall(r'适用于[^。]*', content)
    
    def _analyze_query_context(self, query_context: str) -> Dict[str, Any]:
        """分析查询上下文"""
        return {'keywords': jieba.lcut(query_context)}
    
    def _calculate_applicability_match(self, conditions: List[str], context: Dict[str, Any]) -> float:
        """计算适用性匹配"""
        return 0.8  # 简化实现
    
    def _check_law_scope_applicability(self, content: str, query_context: str) -> float:
        """检查法律范围适用性"""
        return 0.8  # 简化实现
    
    def _check_temporal_applicability(self, content: str) -> float:
        """检查时效性适用性"""
        return 0.9  # 简化实现
    
    def _analyze_query_requirements(self, query_context: str) -> List[str]:
        """分析查询需求"""
        return ['法律依据', '适用条件', '法律后果']
    
    def _check_content_coverage(self, content: str, requirements: List[str]) -> float:
        """检查内容覆盖度"""
        coverage_count = sum(1 for req in requirements if req in content)
        return coverage_count / len(requirements) if requirements else 0.5
    
    def _check_essential_legal_elements(self, content: str) -> float:
        """检查必要法律要素"""
        essential_elements = ['法律依据', '适用条件', '法律后果', '责任承担']
        present_count = sum(1 for element in essential_elements if element in content)
        return present_count / len(essential_elements)
    
    def _check_risk_disclosure(self, content: str) -> float:
        """检查风险披露"""
        risk_keywords = ['风险', '注意', '可能', '建议', '谨慎']
        risk_count = sum(1 for keyword in risk_keywords if keyword in content)
        return min(1.0, risk_count / 3.0)


# 使用示例
if __name__ == "__main__":
    validator = LegalQualityValidator()
    
    # 测试法律内容
    test_content = """
    根据《中华人民共和国民法典》第464条规定，合同是民事主体之间设立、变更、终止民事法律关系的协议。
    在本案中，双方当事人签订的房屋买卖合同合法有效。
    被告违反合同约定，构成违约，应当依照《中华人民共和国民法典》第577条承担违约责任。
    因此，被告应当赔偿原告因违约造成的损失。
    """
    
    query_context = "房屋买卖合同违约责任问题"
    
    # 执行质量验证
    quality_report = validator.validate_legal_content_like_expert(
        test_content, query_context
    )
    
    print("=== 法律内容质量验证报告 ===")
    print(f"总体评分: {quality_report.overall_score:.2f}")
    print(f"风险级别: {quality_report.overall_risk_level.value}")
    print(f"置信度: {quality_report.confidence_score:.2f}")
    
    print("\n=== 各项验证结果 ===")
    for result in quality_report.validation_results:
        print(f"{result.aspect.value}: {result.score:.2f} ({result.risk_level.value})")
        if result.issues:
            print(f"  问题: {result.issues[0]}")
        if result.suggestions:
            print(f"  建议: {result.suggestions[0]}")
    
    print("\n=== 风险摘要 ===")
    risk_summary = quality_report.risk_summary
    print(f"总问题数: {risk_summary['total_issues']}")
    print(f"总建议数: {risk_summary['total_suggestions']}")
    
    print("\n=== 最终建议 ===")
    for recommendation in quality_report.recommendations:
        print(f"- {recommendation}")