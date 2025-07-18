"""
法律知识图谱构建器
实现法律实体关系挖掘、知识图谱构建和图谱推理查询
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from collections import defaultdict, Counter
import numpy as np


class EntityType(Enum):
    """实体类型"""
    LAW = "law"                    # 法律法规
    ARTICLE = "article"            # 法条
    CASE = "case"                  # 案例
    COURT = "court"                # 法院
    CONCEPT = "concept"            # 法律概念
    PERSON = "person"              # 人员
    ORGANIZATION = "organization"   # 组织机构
    LOCATION = "location"          # 地点


class RelationType(Enum):
    """关系类型"""
    CITES = "cites"                # 引用关系
    INTERPRETS = "interprets"      # 解释关系
    APPLIES = "applies"            # 适用关系
    CONFLICTS = "conflicts"        # 冲突关系
    SIMILAR_TO = "similar_to"      # 相似关系
    BELONGS_TO = "belongs_to"      # 归属关系
    DEPENDS_ON = "depends_on"      # 依赖关系
    EXTENDS = "extends"            # 扩展关系


@dataclass
class KnowledgeEntity:
    """知识实体"""
    entity_id: str
    entity_type: EntityType
    name: str
    description: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    importance_score: float = 0.0


@dataclass
class KnowledgeRelation:
    """知识关系"""
    relation_id: str
    source_entity: str
    target_entity: str
    relation_type: RelationType
    confidence: float
    evidence: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


class LegalKnowledgeGraphBuilder:
    """法律知识图谱构建器"""
    
    def __init__(self):
        self.knowledge_graph = nx.MultiDiGraph()
        self.entities = {}
        self.relations = {}
        self.entity_patterns = self._load_entity_patterns()
        self.relation_patterns = self._load_relation_patterns()
        self.legal_concepts = self._load_legal_concepts()
        
    def build_legal_knowledge_graph(
        self, 
        legal_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """构建法律知识图谱"""
        
        # 1. 实体识别：从文档中识别法律实体
        all_entities = self._extract_all_entities(legal_documents)
        
        # 2. 关系挖掘：发现实体间的关系
        all_relations = self._extract_all_relations(legal_documents, all_entities)
        
        # 3. 图谱构建：构建知识图谱结构
        self._build_graph_structure(all_entities, all_relations)
        
        # 4. 实体重要性评估
        self._calculate_entity_importance()
        
        # 5. 关系置信度评估
        self._calculate_relation_confidence()
        
        # 6. 图谱优化
        self._optimize_knowledge_graph()
        
        return {
            'entities': self.entities,
            'relations': self.relations,
            'graph_metrics': self._calculate_graph_metrics(),
            'entity_clusters': self._discover_entity_clusters(),
            'central_concepts': self._identify_central_concepts()
        }
    
    def _extract_all_entities(self, documents: List[Dict]) -> List[KnowledgeEntity]:
        """提取所有法律实体"""
        all_entities = []
        entity_counter = Counter()
        
        for doc in documents:
            doc_entities = self._extract_document_entities(doc)
            all_entities.extend(doc_entities)
            
            # 统计实体频次
            for entity in doc_entities:
                entity_counter[entity.entity_id] += 1
        
        # 去重并合并相同实体
        unique_entities = self._merge_duplicate_entities(all_entities)
        
        # 基于频次设置重要性
        for entity in unique_entities:
            entity.importance_score = min(entity_counter[entity.entity_id] / 10.0, 1.0)
        
        return unique_entities
    
    def _extract_document_entities(self, document: Dict) -> List[KnowledgeEntity]:
        """从单个文档提取实体"""
        entities = []
        content = document.get('content', '')
        doc_id = document.get('id', '')
        
        # 提取各类实体
        for entity_type, patterns in self.entity_patterns.items():
            type_entities = self._extract_entities_by_type(
                content, doc_id, entity_type, patterns
            )
            entities.extend(type_entities)
        
        return entities
    
    def _extract_entities_by_type(
        self, 
        content: str, 
        doc_id: str, 
        entity_type: str, 
        patterns: List[str]
    ) -> List[KnowledgeEntity]:
        """按类型提取实体"""
        entities = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                entity_text = match.group(0)
                
                # 生成实体ID
                entity_id = self._generate_entity_id(entity_text, entity_type)
                
                # 提取实体属性
                attributes = self._extract_entity_attributes(
                    entity_text, entity_type, content
                )
                
                entity = KnowledgeEntity(
                    entity_id=entity_id,
                    entity_type=EntityType(entity_type),
                    name=entity_text,
                    description=self._generate_entity_description(entity_text, entity_type),
                    attributes=attributes
                )
                entities.append(entity)
        
        return entities
    
    def _generate_entity_id(self, entity_text: str, entity_type: str) -> str:
        """生成实体ID"""
        # 清理实体文本
        clean_text = re.sub(r'[《》（）\s]', '', entity_text)
        return f"{entity_type}_{clean_text}"
    
    def _extract_entity_attributes(
        self, 
        entity_text: str, 
        entity_type: str, 
        content: str
    ) -> Dict[str, Any]:
        """提取实体属性"""
        attributes = {}
        
        if entity_type == "law":
            # 法律属性
            if "中华人民共和国" in entity_text:
                attributes["level"] = "national"
            attributes["full_name"] = entity_text
            
        elif entity_type == "article":
            # 法条属性
            article_match = re.search(r'第(\d+)条', entity_text)
            if article_match:
                attributes["article_number"] = article_match.group(1)
            
        elif entity_type == "court":
            # 法院属性
            if "最高人民法院" in entity_text:
                attributes["level"] = "supreme"
            elif "高级人民法院" in entity_text:
                attributes["level"] = "high"
            elif "中级人民法院" in entity_text:
                attributes["level"] = "intermediate"
            elif "基层人民法院" in entity_text:
                attributes["level"] = "basic"
            
        elif entity_type == "case":
            # 案例属性
            year_match = re.search(r'(\d{4})', entity_text)
            if year_match:
                attributes["year"] = year_match.group(1)
            
            if "民事" in entity_text:
                attributes["case_type"] = "civil"
            elif "刑事" in entity_text:
                attributes["case_type"] = "criminal"
            elif "行政" in entity_text:
                attributes["case_type"] = "administrative"
        
        return attributes
    
    def _generate_entity_description(self, entity_text: str, entity_type: str) -> str:
        """生成实体描述"""
        if entity_type == "law":
            return f"法律法规：{entity_text}"
        elif entity_type == "article":
            return f"法条：{entity_text}"
        elif entity_type == "court":
            return f"审判机关：{entity_text}"
        elif entity_type == "case":
            return f"法律案例：{entity_text}"
        elif entity_type == "concept":
            return f"法律概念：{entity_text}"
        else:
            return f"法律实体：{entity_text}"
    
    def _merge_duplicate_entities(self, entities: List[KnowledgeEntity]) -> List[KnowledgeEntity]:
        """合并重复实体"""
        entity_map = {}
        
        for entity in entities:
            if entity.entity_id in entity_map:
                # 合并属性
                existing = entity_map[entity.entity_id]
                for key, value in entity.attributes.items():
                    if key not in existing.attributes:
                        existing.attributes[key] = value
            else:
                entity_map[entity.entity_id] = entity
        
        return list(entity_map.values())
    
    def _extract_all_relations(
        self, 
        documents: List[Dict], 
        entities: List[KnowledgeEntity]
    ) -> List[KnowledgeRelation]:
        """提取所有关系"""
        all_relations = []
        
        # 构建实体索引
        entity_text_map = {entity.name: entity.entity_id for entity in entities}
        
        for doc in documents:
            doc_relations = self._extract_document_relations(doc, entity_text_map)
            all_relations.extend(doc_relations)
        
        return all_relations
    
    def _extract_document_relations(
        self, 
        document: Dict, 
        entity_text_map: Dict[str, str]
    ) -> List[KnowledgeRelation]:
        """从单个文档提取关系"""
        relations = []
        content = document.get('content', '')
        doc_id = document.get('id', '')
        
        # 提取各类关系
        for relation_type, patterns in self.relation_patterns.items():
            type_relations = self._extract_relations_by_type(
                content, doc_id, relation_type, patterns, entity_text_map
            )
            relations.extend(type_relations)
        
        return relations
    
    def _extract_relations_by_type(
        self, 
        content: str, 
        doc_id: str, 
        relation_type: str, 
        patterns: List[str], 
        entity_text_map: Dict[str, str]
    ) -> List[KnowledgeRelation]:
        """按类型提取关系"""
        relations = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                relation_text = match.group(0)
                
                # 查找关系中的实体
                source_entity, target_entity = self._identify_relation_entities(
                    relation_text, entity_text_map
                )
                
                if source_entity and target_entity:
                    relation_id = f"{relation_type}_{source_entity}_{target_entity}"
                    
                    # 计算关系置信度
                    confidence = self._calculate_relation_confidence(
                        relation_text, relation_type, content
                    )
                    
                    relation = KnowledgeRelation(
                        relation_id=relation_id,
                        source_entity=source_entity,
                        target_entity=target_entity,
                        relation_type=RelationType(relation_type),
                        confidence=confidence,
                        evidence=[relation_text]
                    )
                    relations.append(relation)
        
        return relations
    
    def _identify_relation_entities(
        self, 
        relation_text: str, 
        entity_text_map: Dict[str, str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """识别关系中的实体"""
        found_entities = []
        
        for entity_text, entity_id in entity_text_map.items():
            if entity_text in relation_text:
                found_entities.append(entity_id)
        
        if len(found_entities) >= 2:
            return found_entities[0], found_entities[1]
        
        return None, None
    
    def _calculate_relation_confidence(
        self, 
        relation_text: str, 
        relation_type: str, 
        context: str
    ) -> float:
        """计算关系置信度"""
        confidence = 0.5
        
        # 关系指示词强度
        strong_indicators = {
            'cites': ['依照', '根据', '按照'],
            'interprets': ['解释', '说明', '阐明'],
            'applies': ['适用', '应用', '运用'],
            'conflicts': ['冲突', '矛盾', '相悖'],
            'similar_to': ['类似', '相似', '如同']
        }
        
        if relation_type in strong_indicators:
            for indicator in strong_indicators[relation_type]:
                if indicator in relation_text:
                    confidence += 0.2
        
        # 上下文支持度
        context_window = 200
        extended_context = context[max(0, len(context)//2 - context_window):
                                   min(len(context), len(context)//2 + context_window)]
        
        if any(indicator in extended_context for indicators in strong_indicators.values() 
               for indicator in indicators):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _build_graph_structure(
        self, 
        entities: List[KnowledgeEntity], 
        relations: List[KnowledgeRelation]
    ) -> None:
        """构建图谱结构"""
        self.knowledge_graph.clear()
        self.entities = {}
        self.relations = {}
        
        # 添加实体节点
        for entity in entities:
            self.entities[entity.entity_id] = entity
            self.knowledge_graph.add_node(
                entity.entity_id,
                entity_type=entity.entity_type.value,
                name=entity.name,
                description=entity.description,
                attributes=entity.attributes,
                importance_score=entity.importance_score
            )
        
        # 添加关系边
        for relation in relations:
            if (relation.source_entity in self.entities and 
                relation.target_entity in self.entities):
                
                self.relations[relation.relation_id] = relation
                self.knowledge_graph.add_edge(
                    relation.source_entity,
                    relation.target_entity,
                    relation_type=relation.relation_type.value,
                    confidence=relation.confidence,
                    evidence=relation.evidence,
                    attributes=relation.attributes
                )
    
    def _calculate_entity_importance(self) -> None:
        """计算实体重要性"""
        # 基于度中心性
        try:
            degree_centrality = nx.degree_centrality(self.knowledge_graph)
            
            # 基于PageRank
            pagerank_scores = nx.pagerank(self.knowledge_graph)
            
            # 更新实体重要性
            for entity_id, entity in self.entities.items():
                degree_score = degree_centrality.get(entity_id, 0.0)
                pagerank_score = pagerank_scores.get(entity_id, 0.0)
                
                # 综合评分
                entity.importance_score = (
                    entity.importance_score * 0.4 +
                    degree_score * 0.3 +
                    pagerank_score * 0.3
                )
                
                # 更新图中的节点属性
                self.knowledge_graph.nodes[entity_id]['importance_score'] = entity.importance_score
        
        except Exception as e:
            print(f"计算实体重要性时出错: {e}")
    
    def _calculate_relation_confidence(self) -> None:
        """计算关系置信度"""
        # 基于共现频次和上下文强度调整关系置信度
        for relation_id, relation in self.relations.items():
            # 获取关系的边数据
            edge_data = self.knowledge_graph.get_edge_data(
                relation.source_entity, relation.target_entity
            )
            
            if edge_data:
                # 更新置信度
                for key, data in edge_data.items():
                    data['confidence'] = relation.confidence
    
    def _optimize_knowledge_graph(self) -> None:
        """优化知识图谱"""
        # 移除低置信度的关系
        edges_to_remove = []
        for u, v, data in self.knowledge_graph.edges(data=True):
            if data.get('confidence', 0.0) < 0.3:
                edges_to_remove.append((u, v))
        
        for u, v in edges_to_remove:
            self.knowledge_graph.remove_edge(u, v)
        
        # 移除孤立节点
        isolated_nodes = list(nx.isolates(self.knowledge_graph))
        for node in isolated_nodes:
            if self.entities[node].importance_score < 0.1:
                self.knowledge_graph.remove_node(node)
                del self.entities[node]
    
    def query_knowledge_graph(
        self, 
        query_entity: str, 
        relation_types: List[str] = None, 
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """查询知识图谱"""
        if query_entity not in self.knowledge_graph:
            return {"error": "实体不存在于知识图谱中"}
        
        # 查找相关实体
        related_entities = self._find_related_entities(
            query_entity, relation_types, max_depth
        )
        
        # 查找相关关系
        related_relations = self._find_related_relations(
            query_entity, related_entities
        )
        
        # 构建子图
        subgraph_nodes = [query_entity] + related_entities
        subgraph = self.knowledge_graph.subgraph(subgraph_nodes)
        
        return {
            'query_entity': self.entities[query_entity],
            'related_entities': [self.entities[e] for e in related_entities],
            'related_relations': related_relations,
            'subgraph_metrics': {
                'nodes': len(subgraph_nodes),
                'edges': subgraph.number_of_edges(),
                'density': nx.density(subgraph) if len(subgraph_nodes) > 1 else 0
            }
        }
    
    def _find_related_entities(
        self, 
        query_entity: str, 
        relation_types: List[str], 
        max_depth: int
    ) -> List[str]:
        """查找相关实体"""
        related_entities = []
        visited = set()
        
        def dfs(entity, depth):
            if depth > max_depth or entity in visited:
                return
            
            visited.add(entity)
            
            # 查找邻居
            neighbors = list(self.knowledge_graph.neighbors(entity))
            for neighbor in neighbors:
                if neighbor not in visited:
                    # 检查关系类型
                    edge_data = self.knowledge_graph.get_edge_data(entity, neighbor)
                    if edge_data:
                        for data in edge_data.values():
                            if (relation_types is None or 
                                data.get('relation_type') in relation_types):
                                related_entities.append(neighbor)
                                dfs(neighbor, depth + 1)
                                break
        
        dfs(query_entity, 0)
        return list(set(related_entities))
    
    def _find_related_relations(
        self, 
        query_entity: str, 
        related_entities: List[str]
    ) -> List[KnowledgeRelation]:
        """查找相关关系"""
        related_relations = []
        all_entities = [query_entity] + related_entities
        
        for relation_id, relation in self.relations.items():
            if (relation.source_entity in all_entities and 
                relation.target_entity in all_entities):
                related_relations.append(relation)
        
        return related_relations
    
    def _calculate_graph_metrics(self) -> Dict[str, Any]:
        """计算图谱指标"""
        metrics = {}
        
        if self.knowledge_graph.number_of_nodes() == 0:
            return metrics
        
        # 基本指标
        metrics['nodes_count'] = self.knowledge_graph.number_of_nodes()
        metrics['edges_count'] = self.knowledge_graph.number_of_edges()
        metrics['density'] = nx.density(self.knowledge_graph)
        
        # 连通性
        metrics['is_connected'] = nx.is_connected(self.knowledge_graph.to_undirected())
        metrics['connected_components'] = nx.number_connected_components(
            self.knowledge_graph.to_undirected()
        )
        
        # 度分布
        degrees = [d for n, d in self.knowledge_graph.degree()]
        metrics['average_degree'] = sum(degrees) / len(degrees) if degrees else 0
        metrics['max_degree'] = max(degrees) if degrees else 0
        
        # 路径长度
        try:
            if nx.is_connected(self.knowledge_graph.to_undirected()):
                metrics['average_path_length'] = nx.average_shortest_path_length(
                    self.knowledge_graph.to_undirected()
                )
            else:
                metrics['average_path_length'] = float('inf')
        except:
            metrics['average_path_length'] = float('inf')
        
        return metrics
    
    def _discover_entity_clusters(self) -> List[List[str]]:
        """发现实体聚类"""
        try:
            # 使用社区发现算法
            import networkx.algorithms.community as nx_comm
            
            undirected_graph = self.knowledge_graph.to_undirected()
            communities = nx_comm.greedy_modularity_communities(undirected_graph)
            
            return [list(community) for community in communities]
        
        except Exception as e:
            print(f"发现实体聚类时出错: {e}")
            return []
    
    def _identify_central_concepts(self) -> List[Dict[str, Any]]:
        """识别核心概念"""
        central_concepts = []
        
        try:
            # 计算中心性指标
            degree_centrality = nx.degree_centrality(self.knowledge_graph)
            betweenness_centrality = nx.betweenness_centrality(self.knowledge_graph)
            closeness_centrality = nx.closeness_centrality(self.knowledge_graph)
            
            # 综合评分
            for entity_id, entity in self.entities.items():
                centrality_score = (
                    degree_centrality.get(entity_id, 0.0) * 0.4 +
                    betweenness_centrality.get(entity_id, 0.0) * 0.3 +
                    closeness_centrality.get(entity_id, 0.0) * 0.3
                )
                
                central_concepts.append({
                    'entity_id': entity_id,
                    'name': entity.name,
                    'entity_type': entity.entity_type.value,
                    'centrality_score': centrality_score,
                    'importance_score': entity.importance_score
                })
            
            # 按中心性排序
            central_concepts.sort(key=lambda x: x['centrality_score'], reverse=True)
            
        except Exception as e:
            print(f"识别核心概念时出错: {e}")
        
        return central_concepts[:20]  # 返回前20个核心概念
    
    def _load_entity_patterns(self) -> Dict[str, List[str]]:
        """加载实体模式"""
        return {
            'law': [
                r'《[^》]*法》',
                r'《[^》]*条例》',
                r'《[^》]*规定》',
                r'《[^》]*办法》'
            ],
            'article': [
                r'《[^》]+》第\d+条',
                r'第\d+条',
                r'第\d+条第\d+款'
            ],
            'court': [
                r'最高人民法院',
                r'\w+高级人民法院',
                r'\w+中级人民法院',
                r'\w+基层人民法院',
                r'\w+人民法院'
            ],
            'case': [
                r'（\d{4}）[^（）]*\d+号',
                r'指导案例第\d+号'
            ],
            'concept': [
                r'合同违约',
                r'侵权责任',
                r'物权',
                r'债权',
                r'知识产权'
            ]
        }
    
    def _load_relation_patterns(self) -> Dict[str, List[str]]:
        """加载关系模式"""
        return {
            'cites': [
                r'依照[^。]*',
                r'根据[^。]*',
                r'按照[^。]*'
            ],
            'interprets': [
                r'解释[^。]*',
                r'说明[^。]*',
                r'阐明[^。]*'
            ],
            'applies': [
                r'适用[^。]*',
                r'应用[^。]*',
                r'运用[^。]*'
            ],
            'conflicts': [
                r'冲突[^。]*',
                r'矛盾[^。]*',
                r'相悖[^。]*'
            ],
            'similar_to': [
                r'类似[^。]*',
                r'相似[^。]*',
                r'如同[^。]*'
            ]
        }
    
    def _load_legal_concepts(self) -> Dict[str, str]:
        """加载法律概念"""
        return {
            '合同': '当事人之间设立、变更、终止民事法律关系的协议',
            '侵权': '行为人因过错侵害他人民事权益造成损害的行为',
            '物权': '权利人依法对特定的物享有直接支配和排他的权利',
            '债权': '权利人请求债务人为或者不为一定行为的权利',
            '知识产权': '权利人对其智力劳动成果依法享有的专有权利'
        }


# 使用示例
if __name__ == "__main__":
    builder = LegalKnowledgeGraphBuilder()
    
    # 测试文档
    test_documents = [
        {
            'id': 'judgment_001',
            'content': '''
            北京市朝阳区人民法院民事判决书（2023）京0105民初12345号
            本院认为，根据《中华人民共和国民法典》第464条规定，
            合同是民事主体之间设立、变更、终止民事法律关系的协议。
            被告违反合同约定，构成违约，应承担违约责任。
            '''
        },
        {
            'id': 'law_001',
            'content': '''
            《中华人民共和国民法典》第464条：
            合同是民事主体之间设立、变更、终止民事法律关系的协议。
            第577条：当事人一方不履行合同义务应当承担违约责任。
            '''
        }
    ]
    
    # 构建知识图谱
    kg_result = builder.build_legal_knowledge_graph(test_documents)
    
    print("=== 法律知识图谱构建结果 ===")
    print(f"实体数量: {len(kg_result['entities'])}")
    print(f"关系数量: {len(kg_result['relations'])}")
    
    # 显示图谱指标
    metrics = kg_result['graph_metrics']
    print(f"\n=== 图谱指标 ===")
    print(f"节点数: {metrics.get('nodes_count', 0)}")
    print(f"边数: {metrics.get('edges_count', 0)}")
    print(f"密度: {metrics.get('density', 0):.3f}")
    print(f"平均度: {metrics.get('average_degree', 0):.3f}")
    
    # 显示核心概念
    central_concepts = kg_result['central_concepts']
    print(f"\n=== 核心概念 ===")
    for concept in central_concepts[:5]:
        print(f"{concept['name']} ({concept['entity_type']}): "
              f"中心性={concept['centrality_score']:.3f}")
    
    # 测试查询
    print(f"\n=== 知识图谱查询测试 ===")
    if kg_result['entities']:
        sample_entity = list(kg_result['entities'].keys())[0]
        query_result = builder.query_knowledge_graph(sample_entity, max_depth=1)
        
        print(f"查询实体: {query_result['query_entity'].name}")
        print(f"相关实体数: {len(query_result['related_entities'])}")
        print(f"相关关系数: {len(query_result['related_relations'])}")
    
    # 显示实体聚类
    clusters = kg_result['entity_clusters']
    print(f"\n=== 实体聚类 ===")
    for i, cluster in enumerate(clusters[:3]):
        print(f"聚类 {i+1}: {len(cluster)} 个实体")
        for entity_id in cluster[:3]:
            if entity_id in kg_result['entities']:
                print(f"  - {kg_result['entities'][entity_id].name}")