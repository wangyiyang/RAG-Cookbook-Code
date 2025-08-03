"""
自我修正机制
让AI学会发现错误并自动改正
"""

import re
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class ErrorType(Enum):
    """错误类型枚举"""
    FACTUAL = "factual_error"          # 事实错误
    LOGICAL = "logical_error"          # 逻辑错误
    COMPLETENESS = "completeness_error" # 完整性错误
    CONSISTENCY = "consistency_error"   # 一致性错误


@dataclass
class ErrorDetection:
    """错误检测结果"""
    error_type: ErrorType
    description: str
    location: str
    confidence: float
    suggestion: str


class SelfCorrector:
    """自我修正机制：AI自动检查并修正错误"""
    
    def __init__(self, llm, fact_checker=None):
        if llm is None:
            raise ValueError("LLM实例不能为空")
            
        self.llm = llm
        self.fact_checker = fact_checker or FactChecker()
        self.logic_checker = LogicChecker()
        self.completeness_checker = CompletenessChecker()
    
    def auto_fix_errors(self, question: str, answer: str, sources: List[Dict] = None) -> Dict:
        """AI自动检查并修正回答中的错误"""
        
        if not question or not question.strip():
            raise ValueError("问题不能为空")
        if not answer or not answer.strip():
            raise ValueError("答案不能为空")
            
        correction_history = []
        current_answer = answer
        
        # 第一轮：检查事实错误
        fact_errors = self.fact_checker.check_facts(current_answer, sources or [])
        if fact_errors:
            current_answer = self._fix_factual_errors(current_answer, fact_errors)
            correction_history.append({
                "round": 1,
                "type": "factual_correction",
                "errors_found": len(fact_errors),
                "corrected_answer": current_answer
            })
        
        # 第二轮：检查逻辑错误
        logic_errors = self.logic_checker.check_logic(question, current_answer)
        if logic_errors:
            current_answer = self._fix_logical_errors(current_answer, logic_errors)
            correction_history.append({
                "round": 2,
                "type": "logical_correction",
                "errors_found": len(logic_errors),
                "corrected_answer": current_answer
            })
        
        # 第三轮：检查完整性
        completeness_errors = self.completeness_checker.check_completeness(question, current_answer)
        if completeness_errors:
            current_answer = self._fix_completeness_errors(question, current_answer, completeness_errors)
            correction_history.append({
                "round": 3,
                "type": "completeness_correction",
                "errors_found": len(completeness_errors),
                "corrected_answer": current_answer
            })
        
        return {
            "original_answer": answer,
            "corrected_answer": current_answer,
            "corrections_made": len(correction_history),
            "correction_history": correction_history,
            "is_improved": current_answer != answer
        }
    
    def _fix_factual_errors(self, answer: str, errors: List[ErrorDetection]) -> str:
        """修正事实错误"""
        corrected_answer = answer
        
        for error in errors:
            if error.confidence > 0.7:  # 只修正高置信度的错误
                correction_prompt = f"""
                原始回答: {corrected_answer}
                
                检测到的事实错误: {error.description}
                位置: {error.location}
                修正建议: {error.suggestion}
                
                请修正这个事实错误，保持其他内容不变。
                """
                
                corrected_answer = self.llm.generate(correction_prompt)
        
        return corrected_answer
    
    def _fix_logical_errors(self, answer: str, errors: List[ErrorDetection]) -> str:
        """修正逻辑错误"""
        corrected_answer = answer
        
        for error in errors:
            if error.confidence > 0.6:
                correction_prompt = f"""
                原始回答: {corrected_answer}
                
                检测到的逻辑错误: {error.description}
                修正建议: {error.suggestion}
                
                请修正逻辑错误，确保推理过程合理。
                """
                
                corrected_answer = self.llm.generate(correction_prompt)
        
        return corrected_answer
    
    def _fix_completeness_errors(self, question: str, answer: str, errors: List[ErrorDetection]) -> str:
        """修正完整性错误"""
        missing_aspects = [error.suggestion for error in errors]
        
        completion_prompt = f"""
        原始问题: {question}
        当前回答: {answer}
        
        缺失的方面: {'; '.join(missing_aspects)}
        
        请补充缺失的信息，使回答更加完整。
        """
        
        return self.llm.generate(completion_prompt)


class FactChecker:
    """事实检查器"""
    
    def check_facts(self, answer: str, sources: List[Dict]) -> List[ErrorDetection]:
        """检查事实错误"""
        errors = []
        
        # 检查无依据陈述
        unsupported_statements = self._find_unsupported_statements(answer, sources)
        for statement in unsupported_statements:
            errors.append(ErrorDetection(
                error_type=ErrorType.FACTUAL,
                description=f"无依据陈述: {statement}",
                location=statement,
                confidence=0.8,
                suggestion="删除无依据陈述或添加来源支持"
            ))
        
        # 检查数字错误
        number_errors = self._check_numbers(answer, sources)
        errors.extend(number_errors)
        
        # 检查日期错误
        date_errors = self._check_dates(answer, sources)
        errors.extend(date_errors)
        
        return errors
    
    def _find_unsupported_statements(self, answer: str, sources: List[Dict]) -> List[str]:
        """找出无依据的陈述"""
        if not sources:
            return []
        
        source_text = " ".join([doc.get("content", "") for doc in sources])
        sentences = re.split(r'[。！？]', answer)
        
        unsupported = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and not self._has_support(sentence, source_text):
                unsupported.append(sentence)
        
        return unsupported
    
    def _has_support(self, statement: str, source_text: str) -> bool:
        """检查陈述是否有源文档支持"""
        # 简单的关键词匹配
        statement_words = set(re.findall(r'\w+', statement.lower()))
        source_words = set(re.findall(r'\w+', source_text.lower()))
        
        common_words = statement_words.intersection(source_words)
        return len(common_words) >= len(statement_words) * 0.3
    
    def _check_numbers(self, answer: str, sources: List[Dict]) -> List[ErrorDetection]:
        """检查数字是否正确"""
        errors = []
        
        # 提取答案中的数字
        answer_numbers = re.findall(r'\d+(?:\.\d+)?', answer)
        
        if not sources:
            return errors
        
        source_text = " ".join([doc.get("content", "") for doc in sources])
        source_numbers = re.findall(r'\d+(?:\.\d+)?', source_text)
        
        for num in answer_numbers:
            if num not in source_numbers:
                errors.append(ErrorDetection(
                    error_type=ErrorType.FACTUAL,
                    description=f"数字 {num} 在源文档中找不到对应",
                    location=f"数字: {num}",
                    confidence=0.7,
                    suggestion="验证数字准确性或添加来源说明"
                ))
        
        return errors
    
    def _check_dates(self, answer: str, sources: List[Dict]) -> List[ErrorDetection]:
        """检查日期是否正确"""
        errors = []
        
        # 提取日期模式
        date_patterns = [
            r'\d{4}年',
            r'\d{1,2}月',
            r'\d{1,2}日',
            r'\d{4}-\d{1,2}-\d{1,2}'
        ]
        
        answer_dates = []
        for pattern in date_patterns:
            answer_dates.extend(re.findall(pattern, answer))
        
        if not sources or not answer_dates:
            return errors
        
        source_text = " ".join([doc.get("content", "") for doc in sources])
        
        for date in answer_dates:
            if date not in source_text:
                errors.append(ErrorDetection(
                    error_type=ErrorType.FACTUAL,
                    description=f"日期 {date} 在源文档中找不到",
                    location=f"日期: {date}",
                    confidence=0.6,
                    suggestion="验证日期准确性"
                ))
        
        return errors


class LogicChecker:
    """逻辑检查器"""
    
    def check_logic(self, question: str, answer: str) -> List[ErrorDetection]:
        """检查逻辑错误"""
        errors = []
        
        # 检查自相矛盾
        contradictions = self._find_contradictions(answer)
        errors.extend(contradictions)
        
        # 检查推理跳跃
        reasoning_gaps = self._find_reasoning_gaps(answer)
        errors.extend(reasoning_gaps)
        
        # 检查答非所问
        relevance_errors = self._check_relevance(question, answer)
        errors.extend(relevance_errors)
        
        return errors
    
    def _find_contradictions(self, answer: str) -> List[ErrorDetection]:
        """找出自相矛盾的地方"""
        errors = []
        
        contradiction_patterns = [
            (r'正确', r'错误'),
            (r'是', r'不是'),
            (r'有', r'没有'),
            (r'可以', r'不可以'),
            (r'能', r'不能'),
            (r'会', r'不会')
        ]
        
        for pos_pattern, neg_pattern in contradiction_patterns:
            if re.search(pos_pattern, answer) and re.search(neg_pattern, answer):
                errors.append(ErrorDetection(
                    error_type=ErrorType.LOGICAL,
                    description=f"检测到矛盾：同时出现 '{pos_pattern}' 和 '{neg_pattern}'",
                    location="全文",
                    confidence=0.8,
                    suggestion="解决矛盾陈述，保持逻辑一致性"
                ))
        
        return errors
    
    def _find_reasoning_gaps(self, answer: str) -> List[ErrorDetection]:
        """找出推理跳跃"""
        errors = []
        
        sentences = re.split(r'[。！？]', answer)
        if len(sentences) < 2:
            return errors
        
        # 检查是否有推理连接词
        reasoning_connectors = ['因此', '所以', '由于', '因为', '从而', '导致', '结果']
        has_connectors = any(conn in answer for conn in reasoning_connectors)
        
        if len(sentences) > 3 and not has_connectors:
            errors.append(ErrorDetection(
                error_type=ErrorType.LOGICAL,
                description="缺乏推理连接，逻辑跳跃过大",
                location="推理过程",
                confidence=0.6,
                suggestion="添加推理连接词，说明逻辑关系"
            ))
        
        return errors
    
    def _check_relevance(self, question: str, answer: str) -> List[ErrorDetection]:
        """检查答案是否回答了问题"""
        errors = []
        
        question_keywords = set(re.findall(r'\w+', question.lower()))
        answer_keywords = set(re.findall(r'\w+', answer.lower()))
        
        # 去除停用词
        stop_words = {'的', '了', '在', '是', '和', '与', '或', '但'}
        question_keywords -= stop_words
        answer_keywords -= stop_words
        
        if question_keywords:
            overlap = question_keywords.intersection(answer_keywords)
            relevance_score = len(overlap) / len(question_keywords)
            
            if relevance_score < 0.3:
                errors.append(ErrorDetection(
                    error_type=ErrorType.LOGICAL,
                    description="答案与问题相关性较低",
                    location="整体回答",
                    confidence=0.7,
                    suggestion="确保回答直接针对问题"
                ))
        
        return errors


class CompletenessChecker:
    """完整性检查器"""
    
    def check_completeness(self, question: str, answer: str) -> List[ErrorDetection]:
        """检查回答完整性"""
        errors = []
        
        # 检查问题要求的所有方面是否都有回答
        question_aspects = self._extract_question_aspects(question)
        answer_coverage = self._check_aspect_coverage(question_aspects, answer)
        
        for aspect, covered in answer_coverage.items():
            if not covered:
                errors.append(ErrorDetection(
                    error_type=ErrorType.COMPLETENESS,
                    description=f"未充分回答问题的 '{aspect}' 方面",
                    location=aspect,
                    confidence=0.6,
                    suggestion=f"补充关于 {aspect} 的信息"
                ))
        
        return errors
    
    def _extract_question_aspects(self, question: str) -> List[str]:
        """提取问题的各个方面"""
        aspects = []
        
        # 检查疑问词对应的方面
        question_words = {
            '什么': '定义/概念',
            '为什么': '原因',
            '如何': '方法/过程',
            '哪些': '列举',
            '比较': '对比分析',
            '分析': '深入分析',
            '影响': '影响因素',
            '优缺点': '优势和劣势'
        }
        
        for word, aspect in question_words.items():
            if word in question:
                aspects.append(aspect)
        
        # 如果没有明显的疑问词，添加基本方面
        if not aspects:
            aspects = ['基本信息']
        
        return aspects
    
    def _check_aspect_coverage(self, aspects: List[str], answer: str) -> Dict[str, bool]:
        """检查各方面是否被覆盖"""
        coverage = {}
        
        aspect_keywords = {
            '定义/概念': ['是', '指', '定义', '概念', '含义'],
            '原因': ['因为', '由于', '原因', '导致'],
            '方法/过程': ['步骤', '方法', '过程', '如何', '通过'],
            '列举': ['包括', '有', '例如', '主要'],
            '对比分析': ['比较', '对比', '差异', '不同', '相同'],
            '深入分析': ['分析', '研究', '探讨', '说明'],
            '影响因素': ['影响', '作用', '效果', '结果'],
            '优势和劣势': ['优点', '缺点', '优势', '劣势', '好处', '问题']
        }
        
        for aspect in aspects:
            keywords = aspect_keywords.get(aspect, [aspect])
            has_coverage = any(keyword in answer for keyword in keywords)
            coverage[aspect] = has_coverage
        
        return coverage


# 模拟LLM类
class MockLLM:
    def generate(self, prompt: str) -> str:
        """更真实的模拟LLM回答"""
        if "修正这个事实错误" in prompt:
            return "机器学习是人工智能的一个重要分支，它通过算法使计算机能够自动学习和改进性能。"
        elif "修正逻辑错误" in prompt:
            return "机器学习是人工智能的分支，通过数据训练算法来实现智能决策。它具有学习能力，能够从数据中发现模式。"
        elif "补充缺失的信息" in prompt:
            return "机器学习是人工智能的核心技术之一。它包括监督学习、无监督学习和强化学习等类型，广泛应用于图像识别、自然语言处理、推荐系统等领域。"
        else:
            return f"基于提示生成的智能回答: {prompt.split(':', 1)[-1].strip()[:50]}..."


def main():
    """测试自我修正机制"""
    llm = MockLLM()
    corrector = SelfCorrector(llm)
    
    # 测试用例
    question = "什么是机器学习？"
    answer = "机器学习是AI的一种。据我了解，它在1990年被发明。机器学习不能学习，但是可以学习。"
    sources = [
        {"content": "机器学习是人工智能的一个分支，通过算法让计算机系统自动学习和改进。"}
    ]
    
    print("=== 自我修正测试 ===")
    print(f"原始问题: {question}")
    print(f"原始回答: {answer}")
    print()
    
    result = corrector.auto_fix_errors(question, answer, sources)
    
    print(f"是否改进: {result['is_improved']}")
    print(f"修正次数: {result['corrections_made']}")
    print(f"修正后回答: {result['corrected_answer']}")
    print()
    
    for i, correction in enumerate(result['correction_history']):
        print(f"第{correction['round']}轮修正:")
        print(f"  类型: {correction['type']}")
        print(f"  发现错误: {correction['errors_found']}个")


if __name__ == "__main__":
    main()
