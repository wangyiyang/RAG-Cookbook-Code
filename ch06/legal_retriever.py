"""
专业法律检索引擎
实现五维度法律检索、案例类比分析和权威性排序
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict


class SearchDimension(Enum):
    """检索维度"""
    SEMANTIC = "semantic"           # 语义理解检索
    EXACT = "exact"                 # 精确法条匹配
    ANALOGOUS = "analogous"         # 类比案例检索
    CITATIONS = "citations"         # 引用关系检索
    REASONING = "reasoning"         # 知识图谱推理


@dataclass
class SearchResult:
    """检索结果"""
    document_id: str
    title: str
    content: str
    relevance_score: float
    legal_significance: float
    source_type: str
    metadata: Dict[str, Any]


@dataclass
class LegalQuery:
    """法律查询"""
    query_text: str
    case_elements: Dict[str, Any]
    query_type: str = "general"
    context: str = ""


class ProfessionalLegalRetriever:
    """专业级法律检索引擎"""
    
    def __init__(self, knowledge_base: Dict[str, Any] = None):
        self.knowledge_base = knowledge_base or {}
        self.case_type_indicators = self._load_case_type_indicators()
        self.legal_relations = self._load_legal_relations()
        self.importance_keywords = self._load_importance_keywords()
        
    def search_like_senior_lawyer(
        self, 
        legal_query: LegalQuery, 
        context: str = ""
    ) -> List[SearchResult]:
        """像资深律师一样全方位搜索"""
        
        # 五维检索：像律师一样全方位思考
        search_dimensions = {}
        
        # 1. 语义理解：理解法律概念的深层含义
        semantic_results = self.semantic_legal_search(legal_query)
        search_dimensions['semantic'] = semantic_results
        
        # 2. 精确匹配：找到明确相关的法条条文
        exact_matches = self.exact_legal_provision_search(legal_query)
        search_dimensions['exact'] = exact_matches
        
        # 3. 案例类比：寻找法院处理相似案件的方式
        similar_cases = self.find_analogous_legal_cases(legal_query)
        search_dimensions['analogous'] = similar_cases
        
        # 4. 权威引用：追踪具有指导意义的法律文件
        authoritative_refs = self.find_authoritative_citations(legal_query)
        search_dimensions['citations'] = authoritative_refs
        
        # 5. 法理逻辑：基于知识图谱发现深层关联
        legal_reasoning = self.knowledge_graph_reasoning(legal_query)
        search_dimensions['reasoning'] = legal_reasoning
        
        # 6. 专业融合：像律师一样综合分析各种信息
        comprehensive_results = self.synthesize_legal_analysis(
            search_dimensions, legal_query, context
        )
        
        return comprehensive_results
    
    def semantic_legal_search(self, query: LegalQuery) -> List[SearchResult]:
        """语义理解检索"""
        results = []
        
        # 提取查询中的法律概念
        legal_concepts = self._extract_legal_concepts(query.query_text)
        
        # 模拟语义搜索（实际应用中使用向量检索）
        for doc_id, doc_data in self.knowledge_base.get('documents', {}).items():
            similarity_score = self._calculate_semantic_similarity(
                legal_concepts, doc_data.get('content', '')
            )
            
            if similarity_score > 0.3:  # 语义相似度阈值
                result = SearchResult(
                    document_id=doc_id,
                    title=doc_data.get('title', ''),
                    content=doc_data.get('content', ''),
                    relevance_score=similarity_score,
                    legal_significance=0.7,
                    source_type='semantic',
                    metadata=doc_data.get('metadata', {})
                )
                results.append(result)
        
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)[:10]
    
    def exact_legal_provision_search(self, query: LegalQuery) -> List[SearchResult]:
        """精确法条匹配检索"""
        results = []
        
        # 提取法条引用
        law_references = self._extract_law_references(query.query_text)
        
        # 精确匹配法条
        for law_ref in law_references:
            matching_docs = self._find_exact_law_provisions(law_ref)
            for doc in matching_docs:
                result = SearchResult(
                    document_id=doc['id'],
                    title=doc['title'],
                    content=doc['content'],
                    relevance_score=0.95,  # 精确匹配高相关度
                    legal_significance=0.9,
                    source_type='exact',
                    metadata=doc.get('metadata', {})
                )
                results.append(result)
        
        return results
    
    def find_analogous_legal_cases(self, query: LegalQuery) -> List[SearchResult]:
        """案例类比检索"""
        
        # 案例要素提取：识别关键法律要素
        case_elements = self.extract_legal_case_elements(query)
        
        # 相似案例检索：基于要素匹配寻找类似案例
        similar_cases = self._search_similar_judgments(case_elements)
        
        # 相似度评分：多维度评估案例相似程度
        scored_cases = self._calculate_case_similarity_scores(
            case_elements, similar_cases
        )
        
        return scored_cases[:5]  # 返回最相似的5个案例
    
    def extract_legal_case_elements(self, query: LegalQuery) -> Dict[str, Any]:
        """专业法律要素识别：像法官一样分析案件"""
        elements = {}
        query_text = query.query_text
        
        # 案件性质智能识别
        for case_type, indicators in self.case_type_indicators.items():
            if any(indicator in query_text for indicator in indicators):
                elements['case_type'] = case_type
                break
        
        # 争议焦点提取
        if '争议焦点' in query_text or '核心问题' in query_text:
            elements['has_clear_dispute'] = True
        
        # 法律关系类型识别
        for relation_type, keywords in self.legal_relations.items():
            if any(keyword in query_text for keyword in keywords):
                elements['legal_relation'] = relation_type
                break
        
        # 涉案金额识别
        money_pattern = r'(\d+(?:,\d{3})*(?:\.\d{2})?[万千百十]?元)'
        money_matches = re.findall(money_pattern, query_text)
        if money_matches:
            elements['involved_amount'] = money_matches[0]
        
        # 当事人类型识别
        if '个人' in query_text or '自然人' in query_text:
            elements['party_type'] = 'individual'
        elif '公司' in query_text or '企业' in query_text:
            elements['party_type'] = 'corporate'
        
        return elements
    
    def find_authoritative_citations(self, query: LegalQuery) -> List[SearchResult]:
        """权威引用检索"""
        results = []
        
        # 查找权威法条和判例
        authoritative_sources = self.knowledge_base.get('authoritative_sources', [])
        
        for source in authoritative_sources:
            relevance = self._calculate_authority_relevance(query, source)
            if relevance > 0.5:
                result = SearchResult(
                    document_id=source['id'],
                    title=source['title'],
                    content=source['content'],
                    relevance_score=relevance,
                    legal_significance=source.get('authority_score', 0.8),
                    source_type='authoritative',
                    metadata=source.get('metadata', {})
                )
                results.append(result)
        
        return sorted(results, key=lambda x: x.legal_significance, reverse=True)[:10]
    
    def knowledge_graph_reasoning(self, query: LegalQuery) -> List[SearchResult]:
        """知识图谱推理检索"""
        results = []
        
        # 基于知识图谱的推理逻辑
        related_concepts = self._find_related_legal_concepts(query.query_text)
        
        for concept in related_concepts:
            related_docs = self._find_documents_by_concept(concept)
            for doc in related_docs:
                result = SearchResult(
                    document_id=doc['id'],
                    title=doc['title'],
                    content=doc['content'],
                    relevance_score=concept.get('relevance', 0.6),
                    legal_significance=0.75,
                    source_type='reasoning',
                    metadata=doc.get('metadata', {})
                )
                results.append(result)
        
        return results[:8]
    
    def synthesize_legal_analysis(
        self, 
        search_dimensions: Dict[str, List[SearchResult]], 
        query: LegalQuery, 
        context: str
    ) -> List[SearchResult]:
        """综合分析各维度检索结果"""
        
        # 合并所有维度的结果
        all_results = []
        for dimension, results in search_dimensions.items():
            for result in results:
                result.metadata['search_dimension'] = dimension
                all_results.append(result)
        
        # 去重（基于文档ID）
        unique_results = {}
        for result in all_results:
            doc_id = result.document_id
            if doc_id not in unique_results:
                unique_results[doc_id] = result
            else:
                # 合并来自不同维度的结果，取最高分
                existing = unique_results[doc_id]
                if result.relevance_score > existing.relevance_score:
                    unique_results[doc_id] = result
        
        # 重新评分和排序
        final_results = list(unique_results.values())
        for result in final_results:
            result.relevance_score = self._calculate_comprehensive_score(
                result, query, context
            )
        
        # 按综合分数排序
        final_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return final_results[:20]  # 返回最相关的20个结果
    
    def _extract_legal_concepts(self, text: str) -> List[str]:
        """提取法律概念"""
        concepts = []
        
        # 法律术语识别
        legal_terms = [
            '合同', '侵权', '违约', '赔偿', '责任', '权利', '义务',
            '诉讼', '仲裁', '调解', '执行', '上诉', '再审'
        ]
        
        for term in legal_terms:
            if term in text:
                concepts.append(term)
        
        return concepts
    
    def _calculate_semantic_similarity(self, concepts: List[str], content: str) -> float:
        """计算语义相似度"""
        if not concepts:
            return 0.0
        
        # 简单的概念匹配评分
        matches = sum(1 for concept in concepts if concept in content)
        return min(matches / len(concepts), 1.0)
    
    def _extract_law_references(self, text: str) -> List[str]:
        """提取法条引用"""
        patterns = [
            r'《[^》]+》第\d+条',
            r'[^《》]+法第\d+条',
            r'第\d+条'
        ]
        
        references = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            references.extend(matches)
        
        return references
    
    def _find_exact_law_provisions(self, law_ref: str) -> List[Dict]:
        """查找精确法条"""
        # 模拟查找法条
        provisions = self.knowledge_base.get('law_provisions', [])
        return [p for p in provisions if law_ref in p.get('content', '')]
    
    def _search_similar_judgments(self, case_elements: Dict) -> List[Dict]:
        """搜索相似判决"""
        judgments = self.knowledge_base.get('judgments', [])
        similar = []
        
        for judgment in judgments:
            similarity = self._calculate_case_similarity(case_elements, judgment)
            if similarity > 0.3:
                judgment['similarity'] = similarity
                similar.append(judgment)
        
        return similar
    
    def _calculate_case_similarity_scores(
        self, 
        case_elements: Dict, 
        similar_cases: List[Dict]
    ) -> List[SearchResult]:
        """计算案例相似度分数"""
        results = []
        
        for case in similar_cases:
            similarity = case.get('similarity', 0.5)
            
            result = SearchResult(
                document_id=case['id'],
                title=case['title'],
                content=case['content'],
                relevance_score=similarity,
                legal_significance=0.8,
                source_type='analogous',
                metadata=case.get('metadata', {})
            )
            results.append(result)
        
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)
    
    def _calculate_case_similarity(self, elements1: Dict, case2: Dict) -> float:
        """计算案例相似度"""
        similarity = 0.0
        weight_total = 0.0
        
        # 案件类型相似度
        if elements1.get('case_type') == case2.get('case_type'):
            similarity += 0.4
        weight_total += 0.4
        
        # 法律关系相似度
        if elements1.get('legal_relation') == case2.get('legal_relation'):
            similarity += 0.3
        weight_total += 0.3
        
        # 当事人类型相似度
        if elements1.get('party_type') == case2.get('party_type'):
            similarity += 0.2
        weight_total += 0.2
        
        # 金额范围相似度
        if elements1.get('involved_amount') and case2.get('involved_amount'):
            similarity += 0.1
        weight_total += 0.1
        
        return similarity / weight_total if weight_total > 0 else 0.0
    
    def _calculate_authority_relevance(self, query: LegalQuery, source: Dict) -> float:
        """计算权威性相关度"""
        relevance = 0.0
        
        # 关键词匹配
        query_keywords = set(query.query_text.split())
        source_keywords = set(source.get('keywords', []))
        
        if query_keywords & source_keywords:
            relevance += 0.5
        
        # 法律领域匹配
        if query.case_elements.get('case_type') == source.get('case_type'):
            relevance += 0.3
        
        return min(relevance, 1.0)
    
    def _find_related_legal_concepts(self, query_text: str) -> List[Dict]:
        """查找相关法律概念"""
        # 模拟知识图谱推理
        related_concepts = []
        
        # 基于关键词扩展
        if '合同' in query_text:
            related_concepts.append({'concept': '违约责任', 'relevance': 0.8})
            related_concepts.append({'concept': '合同履行', 'relevance': 0.7})
        
        if '侵权' in query_text:
            related_concepts.append({'concept': '损害赔偿', 'relevance': 0.8})
            related_concepts.append({'concept': '过错责任', 'relevance': 0.7})
        
        return related_concepts
    
    def _find_documents_by_concept(self, concept: Dict) -> List[Dict]:
        """根据概念查找文档"""
        documents = self.knowledge_base.get('documents', {})
        related_docs = []
        
        concept_name = concept['concept']
        for doc_id, doc_data in documents.items():
            if concept_name in doc_data.get('content', ''):
                doc_data['id'] = doc_id
                related_docs.append(doc_data)
        
        return related_docs
    
    def _calculate_comprehensive_score(
        self, 
        result: SearchResult, 
        query: LegalQuery, 
        context: str
    ) -> float:
        """计算综合分数"""
        base_score = result.relevance_score
        significance_bonus = result.legal_significance * 0.2
        
        # 根据检索维度调整权重
        dimension_weights = {
            'exact': 1.0,
            'authoritative': 0.9,
            'semantic': 0.8,
            'analogous': 0.7,
            'reasoning': 0.6
        }
        
        dimension = result.metadata.get('search_dimension', 'semantic')
        dimension_weight = dimension_weights.get(dimension, 0.5)
        
        final_score = (base_score * dimension_weight) + significance_bonus
        return min(final_score, 1.0)
    
    def _load_case_type_indicators(self) -> Dict[str, List[str]]:
        """加载案件类型指示词"""
        return {
            '合同纠纷': ['合同', '协议', '违约', '履行'],
            '侵权纠纷': ['侵权', '损害', '人身伤害', '财产损失'],
            '婚姻家庭': ['离婚', '抚养', '财产分割', '继承'],
            '劳动争议': ['劳动合同', '工伤', '加班费', '辞退'],
            '物权纠纷': ['物权', '所有权', '用益物权', '担保物权'],
            '知识产权': ['专利', '商标', '著作权', '商业秘密']
        }
    
    def _load_legal_relations(self) -> Dict[str, List[str]]:
        """加载法律关系类型"""
        return {
            '买卖关系': ['买卖', '购买', '销售', '出售'],
            '租赁关系': ['租赁', '出租', '承租'],
            '借贷关系': ['借款', '贷款', '债权', '债务'],
            '雇佣关系': ['雇佣', '聘用', '劳动'],
            '代理关系': ['代理', '委托', '授权'],
            '合伙关系': ['合伙', '共同经营', '联营']
        }
    
    def _load_importance_keywords(self) -> List[str]:
        """加载重要性关键词"""
        return [
            '依照', '根据', '违反', '适用', '判决', '裁定',
            '认定', '确认', '支持', '驳回', '赔偿', '承担',
            '责任', '义务', '权利', '法律后果', '法律责任'
        ]


# 使用示例
if __name__ == "__main__":
    # 创建模拟知识库
    knowledge_base = {
        'documents': {
            'doc1': {
                'title': '合同违约案例',
                'content': '甲方违反合同约定，应承担违约责任',
                'metadata': {'case_type': '合同纠纷'}
            },
            'doc2': {
                'title': '侵权赔偿案例',
                'content': '乙方侵权行为导致损害，应赔偿损失',
                'metadata': {'case_type': '侵权纠纷'}
            }
        },
        'law_provisions': [
            {
                'id': 'law1',
                'title': '民法典第464条',
                'content': '合同是民事主体之间设立、变更、终止民事法律关系的协议'
            }
        ],
        'authoritative_sources': [
            {
                'id': 'auth1',
                'title': '最高法院指导案例',
                'content': '合同违约责任的认定标准',
                'authority_score': 0.95,
                'case_type': '合同纠纷'
            }
        ]
    }
    
    # 创建检索器
    retriever = ProfessionalLegalRetriever(knowledge_base)
    
    # 测试查询
    query = LegalQuery(
        query_text="合同违约责任如何认定？",
        case_elements={'case_type': '合同纠纷'},
        query_type="legal_consultation"
    )
    
    # 执行检索
    results = retriever.search_like_senior_lawyer(query)
    
    print("=== 法律检索结果 ===")
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. {result.title}")
        print(f"   相关度: {result.relevance_score:.2f}")
        print(f"   法律重要性: {result.legal_significance:.2f}")
        print(f"   来源类型: {result.source_type}")
        print(f"   内容预览: {result.content[:100]}...")
        print("-" * 50)