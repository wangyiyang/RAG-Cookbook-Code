"""
GraphRAG：知识图谱增强检索系统
让RAG不再只是简单的向量搜索，而是具备复杂关系推理能力
"""

import networkx as nx
import numpy as np
import json
import re
import time
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class EntityNode:
    """实体节点"""
    id: str
    name: str
    type: str
    properties: Dict
    embeddings: np.ndarray
    
@dataclass
class RelationEdge:
    """关系边"""
    source: str
    target: str
    relation_type: str
    properties: Dict
    weight: float

class GraphRAGSystem:
    """GraphRAG核心系统"""
    
    def __init__(self, llm, embedding_model):
        if llm is None:
            raise ValueError("LLM实例不能为空")
        if embedding_model is None:
            raise ValueError("嵌入模型不能为空")
            
        self.llm = llm
        self.embedding_model = embedding_model
        
        # 知识图谱
        self.knowledge_graph = nx.MultiDiGraph()
        
        # 图构建组件
        self.entity_extractor = EntityExtractor(llm)
        self.relation_extractor = RelationExtractor(llm)
        self.graph_builder = GraphBuilder()
        
        # 社区检测器
        self.community_detector = CommunityDetector()
        
        # 图检索器
        self.graph_retriever = GraphRetriever()
        
        # 社区摘要缓存
        self.community_summaries = {}
        
    def build_knowledge_graph(self, documents: List[Dict]) -> None:
        """构建知识图谱"""
        
        if not documents:
            raise ValueError("文档列表不能为空")
            
        print(f"开始构建知识图谱，文档数量: {len(documents)}")
        
        all_entities = {}
        all_relations = []
        
        try:
            # 1. 实体和关系提取
            for i, doc in enumerate(documents):
                print(f"处理文档 {i+1}/{len(documents)}")
                
                if 'content' not in doc:
                    print(f"警告: 文档 {i+1} 缺少content字段，跳过")
                    continue
                
                # 提取实体
                entities = self.entity_extractor.extract_entities(doc['content'])
                for entity in entities:
                    entity_key = f"{entity.name}_{entity.type}"
                    if entity_key not in all_entities:
                        all_entities[entity_key] = entity
                    else:
                        # 合并实体属性
                        all_entities[entity_key] = self._merge_entities(
                            all_entities[entity_key], entity
                        )
                
                # 提取关系
                relations = self.relation_extractor.extract_relations(
                    doc['content'], entities
                )
                all_relations.extend(relations)
            
            # 2. 构建图结构
            self.graph_builder.build_graph(
                self.knowledge_graph, list(all_entities.values()), all_relations
            )
            
            # 3. 社区检测
            communities = self.community_detector.detect_communities(
                self.knowledge_graph
            )
            
            # 4. 生成社区摘要
            for community_id, nodes in communities.items():
                summary = self._generate_community_summary(community_id, nodes)
                self.community_summaries[community_id] = summary
            
            print(f"知识图谱构建完成: {len(all_entities)} 个实体, {len(all_relations)} 个关系")
            
        except Exception as e:
            print(f"构建知识图谱时发生错误: {e}")
            raise
    
    def _merge_entities(self, entity1: EntityNode, entity2: EntityNode) -> EntityNode:
        """合并实体属性"""
        merged_properties = entity1.properties.copy()
        merged_properties.update(entity2.properties)
        
        # 合并嵌入向量（取平均值）
        merged_embeddings = (entity1.embeddings + entity2.embeddings) / 2
        
        return EntityNode(
            id=entity1.id,
            name=entity1.name,
            type=entity1.type,
            properties=merged_properties,
            embeddings=merged_embeddings
        )
    
    def graph_augmented_retrieval(self, query: str, max_hops: int = 2) -> Dict:
        """图增强检索"""
        
        if not query or not query.strip():
            raise ValueError("查询不能为空")
            
        try:
            # 1. 查询实体识别
            query_entities = self.entity_extractor.extract_entities(query)
            
            if not query_entities:
                # 如果没有识别到实体，使用语义搜索
                return self._semantic_fallback_search(query)
            
            # 2. 图遍历检索
            relevant_subgraph = self.graph_retriever.retrieve_subgraph(
                self.knowledge_graph, query_entities, max_hops
            )
            
            # 3. 路径推理
            reasoning_paths = self.graph_retriever.find_reasoning_paths(
                relevant_subgraph, query_entities, query
            )
            
            # 4. 社区级别检索
            relevant_communities = self._find_relevant_communities(query_entities)
            
            # 5. 多级信息整合
            retrieval_context = {
                'local_context': self._format_subgraph_context(relevant_subgraph),
                'global_context': self._format_community_context(relevant_communities),
                'reasoning_paths': reasoning_paths,
                'entity_details': self._get_entity_details(query_entities)
            }
            
            return retrieval_context
            
        except Exception as e:
            print(f"图增强检索时发生错误: {e}")
            return self._semantic_fallback_search(query)
    
    def _semantic_fallback_search(self, query: str) -> Dict:
        """语义搜索备用方案"""
        return {
            'local_context': f"对查询'{query}'进行基础语义搜索",
            'global_context': "未找到相关实体，使用通用知识",
            'reasoning_paths': [],
            'entity_details': []
        }
    
    def _find_relevant_communities(self, query_entities: List[EntityNode]) -> List[str]:
        """找出相关社区"""
        relevant_communities = []
        
        for entity in query_entities:
            if self.knowledge_graph.has_node(entity.id):
                # 找到实体所属的社区
                for community_id, nodes in self.community_summaries.items():
                    if entity.id in nodes:
                        relevant_communities.append(community_id)
        
        return list(set(relevant_communities))
    
    def _format_subgraph_context(self, subgraph: nx.MultiDiGraph) -> str:
        """格式化子图上下文"""
        if not subgraph.nodes():
            return "无相关实体信息"
        
        context_parts = []
        for node_id in list(subgraph.nodes())[:10]:  # 限制输出数量
            node_data = subgraph.nodes.get(node_id, {})
            if node_data:
                context_parts.append(f"- {node_data.get('name', node_id)}: {node_data.get('type', 'UNKNOWN')}")
        
        return "相关实体:\n" + "\n".join(context_parts)
    
    def _format_community_context(self, communities: List[str]) -> str:
        """格式化社区上下文"""
        if not communities:
            return "无相关社区信息"
        
        context_parts = []
        for community_id in communities[:3]:  # 限制为前3个社区
            summary = self.community_summaries.get(community_id, "无摘要信息")
            context_parts.append(f"社区 {community_id}: {summary}")
        
        return "\n\n".join(context_parts)
    
    def _get_entity_details(self, entities: List[EntityNode]) -> List[Dict]:
        """获取实体详情"""
        details = []
        for entity in entities[:5]:  # 限制数量
            details.append({
                'name': entity.name,
                'type': entity.type,
                'properties': entity.properties
            })
        return details
    
    def generate_graph_augmented_answer(self, query: str) -> Dict:
        """生成图增强答案"""
        
        try:
            # 1. 图增强检索
            retrieval_context = self.graph_augmented_retrieval(query)
            
            # 2. 上下文构建
            context_prompt = self._build_graph_context_prompt(
                query, retrieval_context
            )
            
            # 3. 答案生成
            answer = self.llm.generate(context_prompt)
            
            # 4. 证据收集
            evidence_paths = retrieval_context['reasoning_paths']
            supporting_entities = retrieval_context['entity_details']
            
            return {
                'answer': answer,
                'evidence_paths': evidence_paths,
                'supporting_entities': supporting_entities,
                'graph_context': retrieval_context,
                'confidence_score': self._calculate_graph_confidence(retrieval_context)
            }
            
        except Exception as e:
            return {
                'answer': f"抱歉，处理您的问题时遇到错误：{str(e)}",
                'evidence_paths': [],
                'supporting_entities': [],
                'graph_context': {},
                'confidence_score': 0.0
            }
    
    def _generate_community_summary(self, community_id: str, nodes: List[str]) -> str:
        """生成社区摘要"""
        
        try:
            # 获取社区内的实体和关系
            community_entities = []
            community_relations = []
            
            for node in nodes:
                if self.knowledge_graph.has_node(node):
                    community_entities.append(self.knowledge_graph.nodes[node])
                    
                    # 获取节点的边
                    for neighbor in self.knowledge_graph.neighbors(node):
                        if neighbor in nodes:
                            edge_data = self.knowledge_graph.get_edge_data(node, neighbor)
                            if edge_data:
                                community_relations.append(edge_data)
            
            # 构建摘要提示
            summary_prompt = f"""
基于以下实体和关系信息，生成这个知识社区的综合摘要：

实体信息：
{self._format_entities_for_summary(community_entities)}

关系信息：
{self._format_relations_for_summary(community_relations)}

请生成一个300字左右的摘要，描述这个社区的主要主题和核心关联：
"""
            
            summary = self.llm.generate(summary_prompt)
            return summary.strip()
            
        except Exception as e:
            return f"社区{community_id}的摘要生成失败：{str(e)}"
    
    def _format_entities_for_summary(self, entities: List[Dict]) -> str:
        """格式化实体信息用于摘要"""
        if not entities:
            return "无实体信息"
        
        formatted = []
        for entity in entities[:10]:  # 限制数量
            name = entity.get('name', '未知')
            entity_type = entity.get('type', '未知')
            formatted.append(f"- {name} ({entity_type})")
        
        return "\n".join(formatted)
    
    def _format_relations_for_summary(self, relations: List[Dict]) -> str:
        """格式化关系信息用于摘要"""
        if not relations:
            return "无关系信息"
        
        formatted = []
        for relation in relations[:10]:  # 限制数量
            if isinstance(relation, dict):
                rel_type = relation.get('relation_type', '相关')
                formatted.append(f"- {rel_type}")
        
        return "\n".join(formatted)
    
    def _build_graph_context_prompt(self, query: str, context: Dict) -> str:
        """构建图上下文提示"""
        
        prompt = f"""
基于以下知识图谱信息回答问题：

问题：{query}

本地上下文（相关实体和关系）：
{context['local_context']}

全局上下文（相关社区摘要）：
{context['global_context']}

推理路径：
{self._format_reasoning_paths(context['reasoning_paths'])}

请基于以上信息提供准确、全面的答案：
"""
        
        return prompt
    
    def _format_reasoning_paths(self, paths: List[Dict]) -> str:
        """格式化推理路径"""
        if not paths:
            return "无明确推理路径"
        
        formatted_paths = []
        for i, path in enumerate(paths[:3]):  # 限制为前3条路径
            path_str = " -> ".join([
                f"{node.get('name', 'Unknown')}({node.get('type', 'Unknown')})" 
                for node in path.get('nodes', [])
            ])
            formatted_paths.append(f"路径{i+1}: {path_str}")
        
        return "\n".join(formatted_paths)
    
    def _calculate_graph_confidence(self, context: Dict) -> float:
        """计算图检索置信度"""
        confidence = 0.5  # 基础置信度
        
        # 基于实体数量调整
        if context['entity_details']:
            confidence += min(len(context['entity_details']) * 0.1, 0.3)
        
        # 基于推理路径调整
        if context['reasoning_paths']:
            confidence += min(len(context['reasoning_paths']) * 0.05, 0.2)
        
        return min(confidence, 1.0)


class EntityExtractor:
    """实体提取器"""
    
    def __init__(self, llm):
        self.llm = llm
        self.entity_types = [
            'PERSON', 'ORGANIZATION', 'LOCATION', 'EVENT', 
            'CONCEPT', 'PRODUCT', 'DATE', 'NUMBER'
        ]
    
    def extract_entities(self, text: str) -> List[EntityNode]:
        """提取实体"""
        
        if not text or not text.strip():
            return []
        
        try:
            extraction_prompt = f"""
从以下文本中提取所有重要实体，并分类：

文本：{text[:1000]}  # 限制文本长度

实体类型：{', '.join(self.entity_types)}

请以以下JSON格式返回：
[
  {{
    "name": "实体名称",
    "type": "实体类型",
    "properties": {{"description": "实体描述", "context": "上下文信息"}}
  }}
]

提取的实体：
"""
            
            response = self.llm.generate(extraction_prompt)
            
            # 解析响应
            entities = []
            try:
                entity_data = json.loads(response)
                
                for item in entity_data:
                    if isinstance(item, dict) and 'name' in item:
                        entity = EntityNode(
                            id=f"{item['name']}_{item.get('type', 'UNKNOWN')}",
                            name=item['name'],
                            type=item.get('type', 'UNKNOWN'),
                            properties=item.get('properties', {}),
                            embeddings=self._generate_entity_embedding(item['name'])
                        )
                        entities.append(entity)
            except json.JSONDecodeError:
                # 解析失败时的fallback处理
                entities = self._fallback_entity_extraction(text)
            
            return entities
            
        except Exception as e:
            print(f"实体提取错误: {e}")
            return self._fallback_entity_extraction(text)
    
    def _generate_entity_embedding(self, entity_name: str) -> np.ndarray:
        """生成实体嵌入"""
        # 这里应该使用实际的embedding模型
        # 目前使用随机向量作为示例
        return np.random.randn(768)
    
    def _fallback_entity_extraction(self, text: str) -> List[EntityNode]:
        """备用实体提取方法"""
        # 简化的实体提取逻辑
        words = text.split()
        entities = []
        
        for word in words:
            if word.istitle() and len(word) > 2:
                entity = EntityNode(
                    id=f"{word}_UNKNOWN",
                    name=word,
                    type="UNKNOWN",
                    properties={"context": text[:100]},
                    embeddings=self._generate_entity_embedding(word)
                )
                entities.append(entity)
        
        return entities[:10]  # 限制数量


class RelationExtractor:
    """关系提取器"""
    
    def __init__(self, llm):
        self.llm = llm
        self.relation_types = [
            'PART_OF', 'RELATED_TO', 'CAUSED_BY', 'LOCATED_IN',
            'WORKS_FOR', 'OWNS', 'CREATED_BY', 'HAPPENED_AT'
        ]
    
    def extract_relations(self, text: str, entities: List[EntityNode]) -> List[RelationEdge]:
        """提取关系"""
        
        if len(entities) < 2:
            return []
        
        try:
            entity_names = [entity.name for entity in entities]
            
            relation_prompt = f"""
基于以下文本和实体列表，提取实体之间的关系：

文本：{text[:1000]}  # 限制文本长度

实体列表：{', '.join(entity_names)}

关系类型：{', '.join(self.relation_types)}

请以以下JSON格式返回：
[
  {{
    "source": "源实体名称",
    "target": "目标实体名称", 
    "relation_type": "关系类型",
    "properties": {{"evidence": "支持证据文本"}},
    "weight": 0.8
  }}
]

提取的关系：
"""
            
            response = self.llm.generate(relation_prompt)
            
            relations = []
            try:
                relation_data = json.loads(response)
                
                for item in relation_data:
                    if isinstance(item, dict) and 'source' in item and 'target' in item:
                        relation = RelationEdge(
                            source=f"{item['source']}_UNKNOWN",  # 需要匹配实际实体ID
                            target=f"{item['target']}_UNKNOWN",
                            relation_type=item.get('relation_type', 'RELATED_TO'),
                            properties=item.get('properties', {}),
                            weight=item.get('weight', 0.5)
                        )
                        relations.append(relation)
            except json.JSONDecodeError:
                # 解析失败时的简化处理
                relations = self._fallback_relation_extraction(entities)
            
            return relations
            
        except Exception as e:
            print(f"关系提取错误: {e}")
            return self._fallback_relation_extraction(entities)
    
    def _fallback_relation_extraction(self, entities: List[EntityNode]) -> List[RelationEdge]:
        """备用关系提取"""
        relations = []
        
        # 基于实体共现创建简单关系
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                relation = RelationEdge(
                    source=entity1.id,
                    target=entity2.id,
                    relation_type="RELATED_TO",
                    properties={"confidence": 0.3},
                    weight=0.3
                )
                relations.append(relation)
        
        return relations[:10]  # 限制数量


class GraphBuilder:
    """图构建器"""
    
    def build_graph(self, graph: nx.MultiDiGraph, 
                   entities: List[EntityNode], 
                   relations: List[RelationEdge]) -> None:
        """构建知识图谱"""
        
        # 添加实体节点
        for entity in entities:
            graph.add_node(entity.id, 
                          name=entity.name,
                          type=entity.type,
                          properties=entity.properties,
                          embeddings=entity.embeddings)
        
        # 添加关系边
        for relation in relations:
            if graph.has_node(relation.source) and graph.has_node(relation.target):
                graph.add_edge(relation.source, relation.target,
                              relation_type=relation.relation_type,
                              properties=relation.properties,
                              weight=relation.weight)


class CommunityDetector:
    """社区检测器"""
    
    def detect_communities(self, graph: nx.MultiDiGraph) -> Dict[str, List[str]]:
        """检测图中的社区"""
        
        if not graph.nodes():
            return {}
        
        # 转换为无向图用于社区检测
        undirected_graph = graph.to_undirected()
        
        # 使用简化的聚类方法
        return self._simple_community_detection(undirected_graph)
    
    def _simple_community_detection(self, graph: nx.Graph) -> Dict[str, List[str]]:
        """简化的社区检测"""
        
        communities = {}
        visited = set()
        community_id = 0
        
        for node in graph.nodes():
            if node not in visited:
                # BFS找到连通分量
                component = self._bfs_component(graph, node, visited)
                communities[f"community_{community_id}"] = component
                community_id += 1
        
        return communities
    
    def _bfs_component(self, graph: nx.Graph, start_node: str, visited: Set[str]) -> List[str]:
        """BFS找连通分量"""
        component = []
        queue = [start_node]
        visited.add(start_node)
        
        while queue:
            node = queue.pop(0)
            component.append(node)
            
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return component


class GraphRetriever:
    """图检索器"""
    
    def retrieve_subgraph(self, graph: nx.MultiDiGraph, 
                         query_entities: List[EntityNode], 
                         max_hops: int = 2) -> nx.MultiDiGraph:
        """检索相关子图"""
        
        subgraph_nodes = set()
        
        # 从查询实体开始扩展
        for entity in query_entities:
            if graph.has_node(entity.id):
                # 添加实体节点
                subgraph_nodes.add(entity.id)
                
                # 多跳扩展
                current_nodes = {entity.id}
                for hop in range(max_hops):
                    next_nodes = set()
                    
                    for node in current_nodes:
                        # 添加邻居节点
                        neighbors = list(graph.neighbors(node))
                        next_nodes.update(neighbors[:5])  # 限制每个节点的邻居数量
                    
                    subgraph_nodes.update(next_nodes)
                    current_nodes = next_nodes
        
        # 创建子图
        if subgraph_nodes:
            return graph.subgraph(subgraph_nodes).copy()
        else:
            return nx.MultiDiGraph()
    
    def find_reasoning_paths(self, subgraph: nx.MultiDiGraph, 
                           query_entities: List[EntityNode], 
                           query: str) -> List[Dict]:
        """找到推理路径"""
        
        reasoning_paths = []
        
        if len(query_entities) < 2:
            return reasoning_paths
        
        # 在查询实体之间找路径
        for i, entity1 in enumerate(query_entities):
            for entity2 in query_entities[i+1:]:
                if (subgraph.has_node(entity1.id) and 
                    subgraph.has_node(entity2.id)):
                    
                    try:
                        # 找最短路径
                        path = nx.shortest_path(
                            subgraph, entity1.id, entity2.id
                        )
                        
                        if len(path) <= 4:  # 限制路径长度
                            path_info = {
                                'nodes': [
                                    {
                                        'id': node_id,
                                        'name': subgraph.nodes[node_id].get('name', node_id),
                                        'type': subgraph.nodes[node_id].get('type', 'UNKNOWN')
                                    }
                                    for node_id in path
                                ],
                                'length': len(path),
                                'relevance_score': self._calculate_path_relevance(
                                    path, query
                                )
                            }
                            reasoning_paths.append(path_info)
                    
                    except nx.NetworkXNoPath:
                        continue
        
        # 按相关性排序
        reasoning_paths.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return reasoning_paths[:5]  # 返回前5条路径
    
    def _calculate_path_relevance(self, path: List[str], query: str) -> float:
        """计算路径相关性"""
        # 简化的相关性计算
        query_terms = set(query.lower().split())
        path_terms = set()
        
        for node_id in path:
            path_terms.update(node_id.lower().split('_'))
        
        overlap = len(query_terms.intersection(path_terms))
        total_terms = len(query_terms.union(path_terms))
        
        return overlap / total_terms if total_terms > 0 else 0.0


# 模拟LLM类
class MockLLM:
    def generate(self, prompt: str) -> str:
        """模拟LLM生成"""
        if "提取所有重要实体" in prompt:
            return '''[
  {"name": "机器学习", "type": "CONCEPT", "properties": {"description": "人工智能的一个分支"}},
  {"name": "深度学习", "type": "CONCEPT", "properties": {"description": "机器学习的子领域"}}
]'''
        elif "提取实体之间的关系" in prompt:
            return '''[
  {"source": "深度学习", "target": "机器学习", "relation_type": "PART_OF", "properties": {"evidence": "深度学习是机器学习的一个子领域"}, "weight": 0.9}
]'''
        else:
            return f"基于图谱信息的智能回答: {prompt.split('问题：')[1].split('本地上下文')[0].strip() if '问题：' in prompt else '通用回答'}"


# 模拟嵌入模型
class MockEmbeddingModel:
    def encode(self, text: str) -> np.ndarray:
        """模拟嵌入编码"""
        return np.random.randn(768)


def main():
    """测试GraphRAG系统"""
    print("=== GraphRAG 系统测试 ===")
    
    # 初始化组件
    llm = MockLLM()
    embedding_model = MockEmbeddingModel()
    
    # 创建GraphRAG系统
    graph_rag = GraphRAGSystem(llm, embedding_model)
    
    # 测试文档
    documents = [
        {
            "id": "doc1",
            "title": "机器学习基础",
            "content": "机器学习是人工智能的一个重要分支。深度学习是机器学习的子领域，使用神经网络进行学习。",
            "source": "AI教程"
        },
        {
            "id": "doc2", 
            "title": "深度学习应用",
            "content": "深度学习在计算机视觉、自然语言处理等领域有广泛应用。卷积神经网络用于图像识别。",
            "source": "技术文档"
        }
    ]
    
    # 构建知识图谱
    print("\n1. 构建知识图谱")
    graph_rag.build_knowledge_graph(documents)
    
    print(f"图谱节点数: {graph_rag.knowledge_graph.number_of_nodes()}")
    print(f"图谱边数: {graph_rag.knowledge_graph.number_of_edges()}")
    
    # 测试查询
    print("\n2. 测试图增强检索")
    query = "深度学习和机器学习的关系是什么？"
    result = graph_rag.generate_graph_augmented_answer(query)
    
    print(f"查询: {query}")
    print(f"答案: {result['answer']}")
    print(f"置信度: {result['confidence_score']:.3f}")
    print(f"支持实体数: {len(result['supporting_entities'])}")


if __name__ == "__main__":
    main()