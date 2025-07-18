"""
法律引用网络分析器
实现法条引用关系发现、权威性评估和引用网络构建
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from collections import defaultdict, Counter
import numpy as np


class CitationType(Enum):
    """引用类型"""
    LAW_ARTICLE = "law_article"      # 法条引用
    CASE_REFERENCE = "case_reference" # 案例引用
    JUDICIAL_INTERPRETATION = "judicial_interpretation" # 司法解释引用
    GUIDANCE_CASE = "guidance_case"   # 指导案例引用
    REGULATION = "regulation"         # 法规引用


@dataclass
class Citation:
    """引用关系"""
    source_id: str
    target_id: str
    citation_type: CitationType
    citation_text: str
    context: str
    confidence: float
    authority_weight: float = 0.0


@dataclass
class AuthorityScore:
    """权威性评分"""
    document_id: str
    authority_score: float
    citation_count: int
    quality_score: float
    recency_score: float
    source_reliability: float


class LegalCitationAnalyzer:
    """法律引用网络分析器"""
    
    def __init__(self):
        self.citation_patterns = self._load_citation_patterns()
        self.authority_indicators = self._load_authority_indicators()
        self.court_hierarchy = self._load_court_hierarchy()
        self.citation_network = nx.DiGraph()
        
    def analyze_legal_citations_like_scholar(
        self, 
        legal_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """像法学学者一样分析引用网络"""
        
        # 1. 引用关系提取：发现文档间的引用关系
        all_citations = self._extract_all_citations(legal_documents)
        
        # 2. 构建引用网络：将引用关系组织成网络结构
        self._build_citation_network(all_citations)
        
        # 3. 权威性评估：基于引用网络计算权威性分数
        authority_scores = self._calculate_authority_scores()
        
        # 4. 引用模式分析：发现引用趋势和模式
        citation_patterns = self._analyze_citation_patterns(all_citations)
        
        # 5. 影响力分析：识别最有影响力的法律文档
        influential_docs = self._identify_influential_documents()
        
        return {
            'citations': all_citations,
            'authority_scores': authority_scores,
            'citation_patterns': citation_patterns,
            'influential_documents': influential_docs,
            'network_metrics': self._calculate_network_metrics()
        }
    
    def _extract_all_citations(self, documents: List[Dict]) -> List[Citation]:
        """提取所有文档的引用关系"""
        all_citations = []
        
        for doc in documents:
            doc_citations = self._extract_document_citations(doc)
            all_citations.extend(doc_citations)
        
        return all_citations
    
    def _extract_document_citations(self, document: Dict) -> List[Citation]:
        """提取单个文档的引用关系"""
        citations = []
        content = document.get('content', '')
        doc_id = document.get('id', '')
        
        # 提取各类引用
        for citation_type, patterns in self.citation_patterns.items():
            type_citations = self._extract_citations_by_type(
                content, doc_id, citation_type, patterns
            )
            citations.extend(type_citations)
        
        return citations
    
    def _extract_citations_by_type(
        self, 
        content: str, 
        source_id: str, 
        citation_type: str, 
        patterns: List[str]
    ) -> List[Citation]:
        """按类型提取引用"""
        citations = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                citation_text = match.group(0)
                
                # 提取上下文
                context_start = max(0, match.start() - 100)
                context_end = min(len(content), match.end() + 100)
                context = content[context_start:context_end]
                
                # 解析目标文档ID
                target_id = self._parse_target_id(citation_text, citation_type)
                
                if target_id:
                    citation = Citation(
                        source_id=source_id,
                        target_id=target_id,
                        citation_type=CitationType(citation_type),
                        citation_text=citation_text,
                        context=context,
                        confidence=self._calculate_citation_confidence(
                            citation_text, context
                        )
                    )
                    citations.append(citation)
        
        return citations
    
    def _parse_target_id(self, citation_text: str, citation_type: str) -> Optional[str]:
        """解析引用目标ID"""
        if citation_type == "law_article":
            # 法条引用解析
            law_match = re.search(r'《([^》]+)》', citation_text)
            article_match = re.search(r'第(\d+)条', citation_text)
            
            if law_match and article_match:
                law_name = law_match.group(1)
                article_num = article_match.group(1)
                return f"{law_name}_art_{article_num}"
        
        elif citation_type == "case_reference":
            # 案例引用解析
            case_match = re.search(r'（(\d{4}）[^）]*(\d+)号', citation_text)
            if case_match:
                return case_match.group(0)
        
        elif citation_type == "judicial_interpretation":
            # 司法解释引用解析
            interp_match = re.search(r'最高人民法院.*?第(\d+)号', citation_text)
            if interp_match:
                return f"supreme_court_interp_{interp_match.group(1)}"
        
        return None
    
    def _calculate_citation_confidence(self, citation_text: str, context: str) -> float:
        """计算引用置信度"""
        confidence = 0.5
        
        # 引用格式完整性
        if '《' in citation_text and '》' in citation_text:
            confidence += 0.2
        
        # 条文号明确性
        if re.search(r'第\d+条', citation_text):
            confidence += 0.2
        
        # 上下文相关性
        legal_keywords = ['依照', '根据', '按照', '适用', '违反']
        context_score = sum(1 for keyword in legal_keywords if keyword in context) * 0.1
        confidence += min(context_score, 0.3)
        
        return min(confidence, 1.0)
    
    def _build_citation_network(self, citations: List[Citation]) -> None:
        """构建引用网络"""
        self.citation_network.clear()
        
        for citation in citations:
            # 添加节点
            self.citation_network.add_node(citation.source_id)
            self.citation_network.add_node(citation.target_id)
            
            # 添加边
            self.citation_network.add_edge(
                citation.source_id,
                citation.target_id,
                citation_type=citation.citation_type.value,
                confidence=citation.confidence,
                citation_text=citation.citation_text
            )
    
    def _calculate_authority_scores(self) -> Dict[str, AuthorityScore]:
        """计算权威性分数"""
        authority_scores = {}
        
        # 计算PageRank分数
        try:
            pagerank_scores = nx.pagerank(self.citation_network, alpha=0.85)
        except:
            pagerank_scores = {node: 0.0 for node in self.citation_network.nodes()}
        
        # 计算各节点的权威性指标
        for node in self.citation_network.nodes():
            # 被引用次数
            in_degree = self.citation_network.in_degree(node)
            
            # 引用质量评分
            quality_score = self._calculate_citation_quality(node)
            
            # 时效性评分
            recency_score = self._calculate_recency_score(node)
            
            # 来源可靠性
            source_reliability = self._calculate_source_reliability(node)
            
            # 综合权威性分数
            authority_score = (
                pagerank_scores.get(node, 0.0) * 0.4 +
                min(in_degree / 10.0, 1.0) * 0.3 +
                quality_score * 0.2 +
                recency_score * 0.1
            )
            
            authority_scores[node] = AuthorityScore(
                document_id=node,
                authority_score=authority_score,
                citation_count=in_degree,
                quality_score=quality_score,
                recency_score=recency_score,
                source_reliability=source_reliability
            )
        
        return authority_scores
    
    def _calculate_citation_quality(self, node_id: str) -> float:
        """计算引用质量"""
        if node_id not in self.citation_network.nodes():
            return 0.0
        
        # 获取所有引用该节点的边
        in_edges = self.citation_network.in_edges(node_id, data=True)
        
        if not in_edges:
            return 0.0
        
        # 计算引用者的权威性
        citing_authorities = []
        for source, target, data in in_edges:
            source_authority = self._get_source_authority(source)
            confidence = data.get('confidence', 0.5)
            citing_authorities.append(source_authority * confidence)
        
        # 取平均值
        return sum(citing_authorities) / len(citing_authorities)
    
    def _get_source_authority(self, source_id: str) -> float:
        """获取引用源的权威性"""
        # 基于法院层级判断权威性
        if '最高人民法院' in source_id:
            return 1.0
        elif '高级人民法院' in source_id:
            return 0.8
        elif '中级人民法院' in source_id:
            return 0.6
        elif '基层人民法院' in source_id:
            return 0.4
        else:
            return 0.3
    
    def _calculate_recency_score(self, node_id: str) -> float:
        """计算时效性评分"""
        # 从节点ID中提取年份信息
        year_match = re.search(r'(\d{4})', node_id)
        if year_match:
            year = int(year_match.group(1))
            current_year = 2024  # 当前年份
            
            # 计算年份衰减
            years_ago = current_year - year
            if years_ago <= 1:
                return 1.0
            elif years_ago <= 3:
                return 0.8
            elif years_ago <= 5:
                return 0.6
            elif years_ago <= 10:
                return 0.4
            else:
                return 0.2
        
        return 0.5  # 默认值
    
    def _calculate_source_reliability(self, node_id: str) -> float:
        """计算来源可靠性"""
        # 基于文档类型和来源判断可靠性
        if '最高人民法院' in node_id:
            return 1.0
        elif '高级人民法院' in node_id:
            return 0.9
        elif '中级人民法院' in node_id:
            return 0.8
        elif '基层人民法院' in node_id:
            return 0.7
        elif '司法解释' in node_id:
            return 0.95
        elif '法律' in node_id or '法规' in node_id:
            return 0.9
        else:
            return 0.5
    
    def _analyze_citation_patterns(self, citations: List[Citation]) -> Dict[str, Any]:
        """分析引用模式"""
        patterns = {}
        
        # 引用类型分布
        type_distribution = Counter([c.citation_type.value for c in citations])
        patterns['citation_type_distribution'] = dict(type_distribution)
        
        # 高频被引文档
        target_counter = Counter([c.target_id for c in citations])
        patterns['most_cited_documents'] = dict(target_counter.most_common(10))
        
        # 活跃引用者
        source_counter = Counter([c.source_id for c in citations])
        patterns['most_active_citers'] = dict(source_counter.most_common(10))
        
        # 引用网络密度
        if self.citation_network.number_of_nodes() > 0:
            density = nx.density(self.citation_network)
            patterns['network_density'] = density
        
        return patterns
    
    def _identify_influential_documents(self) -> List[Dict[str, Any]]:
        """识别最有影响力的文档"""
        influential_docs = []
        
        # 基于中心性指标识别影响力
        try:
            # 度中心性
            degree_centrality = nx.degree_centrality(self.citation_network)
            
            # 中介中心性
            betweenness_centrality = nx.betweenness_centrality(self.citation_network)
            
            # 接近中心性
            closeness_centrality = nx.closeness_centrality(self.citation_network)
            
            # 综合影响力评分
            for node in self.citation_network.nodes():
                influence_score = (
                    degree_centrality.get(node, 0.0) * 0.4 +
                    betweenness_centrality.get(node, 0.0) * 0.3 +
                    closeness_centrality.get(node, 0.0) * 0.3
                )
                
                influential_docs.append({
                    'document_id': node,
                    'influence_score': influence_score,
                    'degree_centrality': degree_centrality.get(node, 0.0),
                    'betweenness_centrality': betweenness_centrality.get(node, 0.0),
                    'closeness_centrality': closeness_centrality.get(node, 0.0)
                })
        
        except Exception as e:
            print(f"计算中心性指标时出错: {e}")
        
        # 按影响力排序
        influential_docs.sort(key=lambda x: x['influence_score'], reverse=True)
        
        return influential_docs[:20]  # 返回前20个最有影响力的文档
    
    def _calculate_network_metrics(self) -> Dict[str, Any]:
        """计算网络指标"""
        metrics = {}
        
        if self.citation_network.number_of_nodes() == 0:
            return metrics
        
        # 基本网络指标
        metrics['nodes_count'] = self.citation_network.number_of_nodes()
        metrics['edges_count'] = self.citation_network.number_of_edges()
        
        # 网络密度
        metrics['density'] = nx.density(self.citation_network)
        
        # 连通性
        metrics['is_connected'] = nx.is_connected(self.citation_network.to_undirected())
        
        # 平均度
        degrees = [d for n, d in self.citation_network.degree()]
        metrics['average_degree'] = sum(degrees) / len(degrees) if degrees else 0
        
        # 聚类系数
        try:
            metrics['clustering_coefficient'] = nx.average_clustering(
                self.citation_network.to_undirected()
            )
        except:
            metrics['clustering_coefficient'] = 0.0
        
        return metrics
    
    def find_authoritative_citations_for_query(
        self, 
        query_keywords: List[str], 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """为查询找到权威引用"""
        relevant_citations = []
        
        # 获取权威性分数
        authority_scores = self._calculate_authority_scores()
        
        # 筛选与查询相关的文档
        for doc_id, score in authority_scores.items():
            relevance = self._calculate_query_relevance(doc_id, query_keywords)
            if relevance > 0.3:  # 相关度阈值
                relevant_citations.append({
                    'document_id': doc_id,
                    'authority_score': score.authority_score,
                    'relevance_score': relevance,
                    'citation_count': score.citation_count,
                    'combined_score': score.authority_score * 0.6 + relevance * 0.4
                })
        
        # 按综合分数排序
        relevant_citations.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return relevant_citations[:top_k]
    
    def _calculate_query_relevance(self, doc_id: str, query_keywords: List[str]) -> float:
        """计算查询相关度"""
        # 简化实现：基于文档ID中的关键词匹配
        relevance = 0.0
        
        for keyword in query_keywords:
            if keyword in doc_id:
                relevance += 0.2
        
        return min(relevance, 1.0)
    
    def _load_citation_patterns(self) -> Dict[str, List[str]]:
        """加载引用模式"""
        return {
            'law_article': [
                r'《[^》]+》第\d+条',
                r'《[^》]+》第\d+条第\d+款',
                r'《[^》]+》第\d+条第\d+款第\d+项',
                r'[^《》]+法第\d+条',
                r'根据《[^》]+》第\d+条',
                r'依照《[^》]+》第\d+条'
            ],
            'case_reference': [
                r'（\d{4}）[^（）]*\d+号',
                r'\d{4}年[^第]*第\d+号',
                r'[^（）]*人民法院.*?（\d{4}）[^（）]*\d+号'
            ],
            'judicial_interpretation': [
                r'最高人民法院.*?解释.*?第\d+号',
                r'最高人民法院.*?规定.*?第\d+号',
                r'最高人民法院.*?意见.*?第\d+号'
            ],
            'guidance_case': [
                r'指导案例第\d+号',
                r'最高人民法院指导案例第\d+号',
                r'参考案例第\d+号'
            ],
            'regulation': [
                r'《[^》]*条例》',
                r'《[^》]*规定》',
                r'《[^》]*办法》',
                r'《[^》]*细则》'
            ]
        }
    
    def _load_authority_indicators(self) -> Dict[str, float]:
        """加载权威性指标"""
        return {
            '最高人民法院': 1.0,
            '高级人民法院': 0.8,
            '中级人民法院': 0.6,
            '基层人民法院': 0.4,
            '司法解释': 0.95,
            '指导案例': 0.9,
            '法律': 0.9,
            '行政法规': 0.8,
            '部门规章': 0.7
        }
    
    def _load_court_hierarchy(self) -> Dict[str, int]:
        """加载法院层级"""
        return {
            '最高人民法院': 4,
            '高级人民法院': 3,
            '中级人民法院': 2,
            '基层人民法院': 1
        }


# 使用示例
if __name__ == "__main__":
    analyzer = LegalCitationAnalyzer()
    
    # 测试文档
    test_documents = [
        {
            'id': 'judgment_001',
            'content': '''
            北京市朝阳区人民法院民事判决书（2023）京0105民初12345号
            本院认为，根据《中华人民共和国民法典》第464条规定，
            合同是民事主体之间设立、变更、终止民事法律关系的协议。
            参照最高人民法院关于适用《中华人民共和国民法典》合同编的解释第1号，
            当事人对合同条款的理解发生争议的，应当按照通常理解进行解释。
            '''
        },
        {
            'id': 'judgment_002', 
            'content': '''
            上海市第一中级人民法院民事判决书（2023）沪01民终5678号
            依照《中华人民共和国民法典》第577条，
            当事人一方不履行合同义务或者履行合同义务不符合约定的，
            应当承担继续履行、采取补救措施或者赔偿损失等违约责任。
            '''
        }
    ]
    
    # 分析引用网络
    analysis_result = analyzer.analyze_legal_citations_like_scholar(test_documents)
    
    print("=== 法律引用网络分析结果 ===")
    print(f"发现引用关系: {len(analysis_result['citations'])}个")
    print(f"权威性评分: {len(analysis_result['authority_scores'])}个文档")
    print(f"网络节点数: {analysis_result['network_metrics'].get('nodes_count', 0)}")
    print(f"网络边数: {analysis_result['network_metrics'].get('edges_count', 0)}")
    
    # 显示权威性最高的文档
    print("\n=== 权威性最高的文档 ===")
    for doc_id, score in list(analysis_result['authority_scores'].items())[:5]:
        print(f"{doc_id}: 权威性={score.authority_score:.3f}, 被引用={score.citation_count}次")
    
    # 显示引用模式
    print("\n=== 引用模式分析 ===")
    patterns = analysis_result['citation_patterns']
    if 'citation_type_distribution' in patterns:
        print("引用类型分布:")
        for ctype, count in patterns['citation_type_distribution'].items():
            print(f"  {ctype}: {count}次")
    
    # 测试查询相关引用
    print("\n=== 合同相关权威引用 ===")
    query_keywords = ['合同', '违约', '民法典']
    relevant_citations = analyzer.find_authoritative_citations_for_query(query_keywords)
    
    for citation in relevant_citations[:3]:
        print(f"文档: {citation['document_id']}")
        print(f"  权威性: {citation['authority_score']:.3f}")
        print(f"  相关度: {citation['relevance_score']:.3f}")
        print(f"  综合得分: {citation['combined_score']:.3f}")