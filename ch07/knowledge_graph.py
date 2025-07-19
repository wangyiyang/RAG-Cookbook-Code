"""
医学知识图谱构建与推理
实现疾病-症状关联、药物相互作用等医学知识建模
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import networkx as nx
import numpy as np


class RelationType(Enum):
    """关系类型"""
    DISEASE_SYMPTOM = "disease_symptom"         # 疾病-症状
    DRUG_INDICATION = "drug_indication"         # 药物-适应症
    DRUG_INTERACTION = "drug_interaction"       # 药物相互作用
    SYMPTOM_TREATMENT = "symptom_treatment"     # 症状-治疗
    CONTRAINDICATION = "contraindication"       # 禁忌症
    ADVERSE_REACTION = "adverse_reaction"       # 不良反应


@dataclass
class MedicalEntity:
    """医学实体"""
    entity_id: str
    entity_type: str  # disease, symptom, drug, treatment
    name: str
    aliases: List[str]
    properties: Dict[str, Any]
    confidence: float = 1.0


@dataclass
class MedicalRelation:
    """医学关系"""
    relation_id: str
    source_entity: str
    target_entity: str
    relation_type: RelationType
    confidence: float
    evidence: List[str]
    metadata: Dict[str, Any]


class MedicalKnowledgeGraph:
    """医学知识图谱"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entities = {}
        self.relations = {}
        self.entity_index = {}
        
    def build_comprehensive_medical_kg_like_expert(
        self, 
        medical_literature: List[Dict]
    ) -> Dict[str, Any]:
        """像医学专家一样构建全面的医学知识图谱"""
        
        # 1. 实体识别与标准化
        extracted_entities = self.extract_and_normalize_medical_entities(
            medical_literature
        )
        
        # 2. 关系抽取与验证
        medical_relations = self.extract_and_validate_medical_relations(
            medical_literature, extracted_entities
        )
        
        # 3. 知识图谱构建
        kg_construction_result = self.construct_medical_knowledge_graph(
            extracted_entities, medical_relations
        )
        
        # 4. 图谱质量评估
        quality_metrics = self.assess_knowledge_graph_quality()
        
        # 5. 子图社区检测
        community_structure = self.detect_medical_communities()
        
        return {
            'entities': extracted_entities,
            'relations': medical_relations,
            'construction_result': kg_construction_result,
            'quality_metrics': quality_metrics,
            'communities': community_structure
        }
    
    def extract_and_normalize_medical_entities(
        self, 
        literature: List[Dict]
    ) -> Dict[str, MedicalEntity]:
        """提取并标准化医学实体"""
        entities = {}
        
        for doc in literature:
            content = doc.get('content', '')
            
            # 疾病实体提取
            diseases = self._extract_disease_entities(content)
            for disease in diseases:
                entity_id = f"disease_{len(entities)}"
                entities[entity_id] = MedicalEntity(
                    entity_id=entity_id,
                    entity_type="disease",
                    name=disease['name'],
                    aliases=disease.get('aliases', []),
                    properties={
                        'icd_code': disease.get('icd_code'),
                        'severity': disease.get('severity'),
                        'category': disease.get('category')
                    },
                    confidence=disease.get('confidence', 0.8)
                )
            
            # 症状实体提取
            symptoms = self._extract_symptom_entities(content)
            for symptom in symptoms:
                entity_id = f"symptom_{len(entities)}"
                entities[entity_id] = MedicalEntity(
                    entity_id=entity_id,
                    entity_type="symptom",
                    name=symptom['name'],
                    aliases=symptom.get('aliases', []),
                    properties={
                        'body_system': symptom.get('body_system'),
                        'severity_scale': symptom.get('severity_scale')
                    },
                    confidence=symptom.get('confidence', 0.8)
                )
            
            # 药物实体提取
            drugs = self._extract_drug_entities(content)
            for drug in drugs:
                entity_id = f"drug_{len(entities)}"
                entities[entity_id] = MedicalEntity(
                    entity_id=entity_id,
                    entity_type="drug",
                    name=drug['name'],
                    aliases=drug.get('aliases', []),
                    properties={
                        'drug_class': drug.get('drug_class'),
                        'mechanism': drug.get('mechanism'),
                        'dosage_form': drug.get('dosage_form')
                    },
                    confidence=drug.get('confidence', 0.8)
                )
            
            # 治疗方法实体提取
            treatments = self._extract_treatment_entities(content)
            for treatment in treatments:
                entity_id = f"treatment_{len(entities)}"
                entities[entity_id] = MedicalEntity(
                    entity_id=entity_id,
                    entity_type="treatment",
                    name=treatment['name'],
                    aliases=treatment.get('aliases', []),
                    properties={
                        'treatment_type': treatment.get('treatment_type'),
                        'duration': treatment.get('duration'),
                        'effectiveness': treatment.get('effectiveness')
                    },
                    confidence=treatment.get('confidence', 0.8)
                )
        
        return entities
    
    def extract_and_validate_medical_relations(
        self, 
        literature: List[Dict],
        entities: Dict[str, MedicalEntity]
    ) -> Dict[str, MedicalRelation]:
        """提取并验证医学关系"""
        relations = {}
        
        for doc in literature:
            content = doc.get('content', '')
            
            # 疾病-症状关系提取
            disease_symptom_relations = self._extract_disease_symptom_relations(
                content, entities
            )
            relations.update(disease_symptom_relations)
            
            # 药物-适应症关系提取
            drug_indication_relations = self._extract_drug_indication_relations(
                content, entities
            )
            relations.update(drug_indication_relations)
            
            # 药物相互作用关系提取
            drug_interaction_relations = self._extract_drug_interaction_relations(
                content, entities
            )
            relations.update(drug_interaction_relations)
            
            # 治疗方案关系提取
            treatment_relations = self._extract_treatment_relations(
                content, entities
            )
            relations.update(treatment_relations)
        
        # 关系验证和置信度评估
        validated_relations = self._validate_medical_relations(relations)
        
        return validated_relations
    
    def construct_medical_knowledge_graph(
        self, 
        entities: Dict[str, MedicalEntity],
        relations: Dict[str, MedicalRelation]
    ) -> Dict[str, Any]:
        """构建医学知识图谱"""
        
        # 添加实体节点
        for entity_id, entity in entities.items():
            self.graph.add_node(
                entity_id,
                entity_type=entity.entity_type,
                name=entity.name,
                aliases=entity.aliases,
                properties=entity.properties,
                confidence=entity.confidence
            )
            self.entities[entity_id] = entity
        
        # 添加关系边
        for relation_id, relation in relations.items():
            self.graph.add_edge(
                relation.source_entity,
                relation.target_entity,
                relation_id=relation_id,
                relation_type=relation.relation_type.value,
                confidence=relation.confidence,
                evidence=relation.evidence,
                metadata=relation.metadata
            )
            self.relations[relation_id] = relation
        
        # 构建实体索引
        self._build_entity_index()
        
        construction_stats = {
            'total_entities': len(entities),
            'total_relations': len(relations),
            'entity_types': self._count_entity_types(),
            'relation_types': self._count_relation_types(),
            'graph_density': nx.density(self.graph),
            'connected_components': nx.number_weakly_connected_components(self.graph)
        }
        
        return construction_stats
    
    def query_medical_knowledge_with_reasoning(
        self, 
        query: str,
        reasoning_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """基于推理的医学知识查询"""
        
        # 解析查询意图
        query_intent = self._parse_medical_query(query)
        
        # 实体识别
        query_entities = self._identify_query_entities(query)
        
        # 多跳推理查询
        reasoning_results = []
        
        for entity_id in query_entities:
            if entity_id in self.graph:
                # 执行多跳推理
                reasoning_paths = self._perform_multi_hop_reasoning(
                    entity_id, query_intent, reasoning_depth
                )
                reasoning_results.extend(reasoning_paths)
        
        # 结果排序和过滤
        filtered_results = self._filter_and_rank_results(
            reasoning_results, query_intent
        )
        
        return filtered_results
    
    def detect_medical_communities(self) -> Dict[str, Any]:
        """检测医学社区结构"""
        
        # 使用Louvain算法检测社区
        undirected_graph = self.graph.to_undirected()
        
        # 社区检测
        communities = {}
        community_id = 0
        
        # 基于实体类型的初始社区划分
        entity_type_communities = {}
        for node, data in self.graph.nodes(data=True):
            entity_type = data.get('entity_type', 'unknown')
            if entity_type not in entity_type_communities:
                entity_type_communities[entity_type] = []
            entity_type_communities[entity_type].append(node)
        
        # 基于关系强度的社区细分
        for entity_type, nodes in entity_type_communities.items():
            if len(nodes) > 1:
                subgraph = undirected_graph.subgraph(nodes)
                try:
                    # 如果networkx版本支持community模块
                    import networkx.algorithms.community as nx_comm
                    sub_communities = list(nx_comm.greedy_modularity_communities(subgraph))
                    
                    for sub_community in sub_communities:
                        communities[f"{entity_type}_{community_id}"] = {
                            'nodes': list(sub_community),
                            'size': len(sub_community),
                            'entity_type': entity_type,
                            'modularity': nx_comm.modularity(subgraph, [sub_community])
                        }
                        community_id += 1
                        
                except ImportError:
                    # 简化的社区检测
                    communities[f"{entity_type}_{community_id}"] = {
                        'nodes': nodes,
                        'size': len(nodes),
                        'entity_type': entity_type,
                        'modularity': 0.5
                    }
                    community_id += 1
        
        return {
            'communities': communities,
            'total_communities': len(communities),
            'modularity_score': self._calculate_overall_modularity(communities)
        }
    
    def assess_knowledge_graph_quality(self) -> Dict[str, float]:
        """评估知识图谱质量"""
        
        # 完整性评估
        completeness = self._assess_completeness()
        
        # 一致性评估
        consistency = self._assess_consistency()
        
        # 准确性评估
        accuracy = self._assess_accuracy()
        
        # 覆盖度评估
        coverage = self._assess_coverage()
        
        return {
            'completeness': completeness,
            'consistency': consistency,
            'accuracy': accuracy,
            'coverage': coverage,
            'overall_quality': (completeness + consistency + accuracy + coverage) / 4
        }
    
    def _extract_disease_entities(self, content: str) -> List[Dict]:
        """提取疾病实体"""
        disease_patterns = [
            r'患有([^，。；\s]+?)(病|症|炎|癌)',
            r'诊断为([^，。；\s]+?)(病|症|炎)',
            r'确诊([^，。；\s]+?)(病|症|癌)',
            r'([^，。；\s]+?)(综合征|综合症)'
        ]
        
        diseases = []
        for pattern in disease_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                disease_name = match.group(1).strip()
                if len(disease_name) > 1 and len(disease_name) < 20:
                    diseases.append({
                        'name': disease_name,
                        'confidence': 0.8,
                        'source_text': match.group(0)
                    })
        
        return diseases
    
    def _extract_symptom_entities(self, content: str) -> List[Dict]:
        """提取症状实体"""
        symptom_patterns = [
            r'出现([^，。；\s]+?)(症状|表现)',
            r'主诉([^，。；\s]+)',
            r'伴有([^，。；\s]+?)(疼痛|不适|症状)',
            r'患者诉([^，。；\s]+?)(疼痛|不适)'
        ]
        
        symptoms = []
        for pattern in symptom_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                symptom_name = match.group(1).strip()
                if len(symptom_name) > 1 and len(symptom_name) < 15:
                    symptoms.append({
                        'name': symptom_name,
                        'confidence': 0.8,
                        'source_text': match.group(0)
                    })
        
        return symptoms
    
    def _extract_drug_entities(self, content: str) -> List[Dict]:
        """提取药物实体"""
        drug_patterns = [
            r'服用([^，。；\s]+?)(片|胶囊|注射液)',
            r'给予([^，。；\s]+?)(治疗|处理)',
            r'使用([^，。；\s]+?)(药物|药)',
            r'口服([^，。；\s]+?)(\d+mg|\d+g)'
        ]
        
        drugs = []
        for pattern in drug_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                drug_name = match.group(1).strip()
                if len(drug_name) > 1 and len(drug_name) < 20:
                    drugs.append({
                        'name': drug_name,
                        'confidence': 0.8,
                        'source_text': match.group(0)
                    })
        
        return drugs
    
    def _extract_treatment_entities(self, content: str) -> List[Dict]:
        """提取治疗方法实体"""
        treatment_patterns = [
            r'(手术|放疗|化疗|理疗)治疗',
            r'进行([^，。；\s]+?)(手术|治疗)',
            r'采用([^，。；\s]+?)(疗法|方法)',
            r'(物理治疗|药物治疗|心理治疗)'
        ]
        
        treatments = []
        for pattern in treatment_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                if match.lastindex and match.lastindex > 1:
                    treatment_name = match.group(1).strip()
                else:
                    treatment_name = match.group(0).strip()
                
                if len(treatment_name) > 1 and len(treatment_name) < 15:
                    treatments.append({
                        'name': treatment_name,
                        'confidence': 0.8,
                        'source_text': match.group(0)
                    })
        
        return treatments
    
    def _extract_disease_symptom_relations(
        self, 
        content: str, 
        entities: Dict[str, MedicalEntity]
    ) -> Dict[str, MedicalRelation]:
        """提取疾病-症状关系"""
        relations = {}
        
        # 查找疾病和症状实体
        disease_entities = {k: v for k, v in entities.items() 
                          if v.entity_type == 'disease'}
        symptom_entities = {k: v for k, v in entities.items() 
                           if v.entity_type == 'symptom'}
        
        # 基于文本距离和语言模式识别关系
        for disease_id, disease in disease_entities.items():
            for symptom_id, symptom in symptom_entities.items():
                # 检查文本中是否存在关联
                if self._check_entity_relation_in_text(
                    disease.name, symptom.name, content
                ):
                    relation_id = f"rel_{len(relations)}"
                    relations[relation_id] = MedicalRelation(
                        relation_id=relation_id,
                        source_entity=disease_id,
                        target_entity=symptom_id,
                        relation_type=RelationType.DISEASE_SYMPTOM,
                        confidence=0.7,
                        evidence=[content[:200] + "..."],
                        metadata={'extraction_method': 'pattern_matching'}
                    )
        
        return relations
    
    def _check_entity_relation_in_text(
        self, 
        entity1: str, 
        entity2: str, 
        text: str
    ) -> bool:
        """检查文本中实体关系"""
        # 计算实体在文本中的位置
        pos1 = text.find(entity1)
        pos2 = text.find(entity2)
        
        if pos1 == -1 or pos2 == -1:
            return False
        
        # 如果两个实体距离较近，认为存在关系
        distance = abs(pos1 - pos2)
        return distance < 100
    
    def _build_entity_index(self):
        """构建实体索引"""
        for entity_id, entity in self.entities.items():
            # 名称索引
            if entity.name not in self.entity_index:
                self.entity_index[entity.name] = []
            self.entity_index[entity.name].append(entity_id)
            
            # 别名索引
            for alias in entity.aliases:
                if alias not in self.entity_index:
                    self.entity_index[alias] = []
                self.entity_index[alias].append(entity_id)


# 使用示例
if __name__ == "__main__":
    kg = MedicalKnowledgeGraph()
    
    # 测试医学文献
    test_literature = [
        {
            'title': '高血压的诊疗指南',
            'content': '''
            高血压患者常出现头痛、头晕等症状。确诊高血压后，
            可服用氨氯地平片进行降压治疗。患者应定期监测血压，
            避免高钠饮食。严重高血压可能导致心肌梗死等并发症。
            '''
        },
        {
            'title': '糖尿病管理要点',
            'content': '''
            糖尿病患者主诉多饮、多尿、多食症状。确诊糖尿病后，
            给予二甲双胍治疗。患者需要进行饮食控制和运动疗法。
            糖尿病并发症包括糖尿病肾病、视网膜病变等。
            '''
        }
    ]
    
    # 构建知识图谱
    kg_result = kg.build_comprehensive_medical_kg_like_expert(test_literature)
    
    print("=== 医学知识图谱构建结果 ===")
    print(f"实体数量: {kg_result['construction_result']['total_entities']}")
    print(f"关系数量: {kg_result['construction_result']['total_relations']}")
    print(f"图密度: {kg_result['construction_result']['graph_density']:.3f}")
    
    print(f"\n实体类型分布:")
    for entity_type, count in kg_result['construction_result']['entity_types'].items():
        print(f"  {entity_type}: {count}")
    
    print(f"\n社区数量: {kg_result['communities']['total_communities']}")
    
    # 测试知识查询
    query_results = kg.query_medical_knowledge_with_reasoning(
        "高血压的症状有哪些？", reasoning_depth=2
    )
    
    print(f"\n查询结果数量: {len(query_results)}")
    for i, result in enumerate(query_results[:3], 1):
        print(f"{i}. {result}")