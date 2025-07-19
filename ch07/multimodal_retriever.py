"""
多模态医疗检索器
实现文本、影像、数值数据的融合检索
"""

import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
# import torch
# import torch.nn as nn
# from transformers import AutoTokenizer, AutoModel
# 演示版本使用模拟实现，避免重依赖
from datetime import datetime


class ModalityType(Enum):
    """模态类型"""
    TEXT = "text"           # 文本数据
    IMAGE = "image"         # 影像数据
    NUMERIC = "numeric"     # 数值数据
    TEMPORAL = "temporal"   # 时序数据


@dataclass
class ModalityFeature:
    """模态特征"""
    modality_type: ModalityType
    feature_vector: np.ndarray
    confidence: float
    metadata: Dict[str, Any]
    timestamp: Optional[datetime] = None


@dataclass
class PatientCase:
    """患者病例"""
    case_id: str
    clinical_notes: str
    medical_images: List[Dict]
    lab_results: Dict[str, float]
    vital_signs: Dict[str, List[Tuple[datetime, float]]]
    diagnosis: Optional[str] = None
    treatment_plan: Optional[str] = None


class MedicalMultimodalRetriever:
    """医疗多模态检索器"""
    
    def __init__(self):
        self.text_encoder = self._load_biobert_encoder()
        self.image_encoder = self._load_medical_vision_model()
        self.fusion_network = self._build_fusion_network()
        self.vector_index = {}
        
    def retrieve_similar_cases_intelligently(
        self, 
        query_case: PatientCase, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """像临床专家一样智能检索相似病例"""
        
        # 1. 多模态特征提取
        multimodal_features = self.extract_comprehensive_features(query_case)
        
        # 2. 跨模态相似性计算
        similarity_scores = self.calculate_cross_modal_similarity(
            multimodal_features
        )
        
        # 3. 时序EHR匹配
        temporal_matches = self.match_temporal_patterns(
            query_case.vital_signs
        )
        
        # 4. 融合检索结果
        final_results = self.fuse_retrieval_results(
            similarity_scores, temporal_matches, top_k
        )
        
        return final_results
    
    def extract_comprehensive_features(
        self, 
        patient_case: PatientCase
    ) -> Dict[str, ModalityFeature]:
        """全面提取患者特征"""
        features = {}
        
        # 文本特征提取：病历、症状描述
        if patient_case.clinical_notes:
            text_features = self.extract_clinical_text_features(
                patient_case.clinical_notes
            )
            features['text'] = ModalityFeature(
                modality_type=ModalityType.TEXT,
                feature_vector=text_features,
                confidence=0.9,
                metadata={'source': 'clinical_notes'}
            )
        
        # 影像特征提取：CT、MRI、X光片
        if patient_case.medical_images:
            image_features = self.extract_medical_image_features(
                patient_case.medical_images
            )
            features['image'] = ModalityFeature(
                modality_type=ModalityType.IMAGE,
                feature_vector=image_features,
                confidence=0.85,
                metadata={'image_count': len(patient_case.medical_images)}
            )
        
        # 数值特征提取：检验指标、生命体征
        if patient_case.lab_results:
            numeric_features = self.extract_lab_numeric_features(
                patient_case.lab_results
            )
            features['numeric'] = ModalityFeature(
                modality_type=ModalityType.NUMERIC,
                feature_vector=numeric_features,
                confidence=0.95,
                metadata={'lab_count': len(patient_case.lab_results)}
            )
        
        return features
    
    def extract_clinical_text_features(self, clinical_text: str) -> np.ndarray:
        """提取临床文本特征"""
        # 使用BioBERT进行医学文本编码（演示版本）
        inputs = self.text_encoder.tokenizer(
            clinical_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        # 模拟特征提取
        outputs = self.text_encoder.model(**inputs)
        # 使用CLS token作为句子表示（模拟版本）
        features = outputs.last_hidden_state[:, 0, :]
        
        return features.flatten()
    
    def extract_medical_image_features(self, medical_images: List[Dict]) -> np.ndarray:
        """提取医学影像特征"""
        image_features = []
        
        for image_info in medical_images:
            modality = image_info.get('modality', 'unknown')  # CT, MRI, X-ray
            
            # 根据影像类型选择专门的特征提取器
            if modality.upper() == 'CT':
                features = self._extract_ct_features(image_info)
            elif modality.upper() == 'MRI':
                features = self._extract_mri_features(image_info)
            elif modality.upper() in ['X-RAY', 'XRAY']:
                features = self._extract_xray_features(image_info)
            else:
                features = self._extract_generic_image_features(image_info)
            
            image_features.append(features)
        
        # 聚合多张影像的特征
        if image_features:
            return np.mean(image_features, axis=0)
        else:
            return np.zeros(512)  # 默认特征维度
    
    def extract_lab_numeric_features(
        self, 
        lab_results: Dict[str, float]
    ) -> np.ndarray:
        """提取检验数值特征"""
        # 标准化实验室指标
        standardized_features = []
        
        # 常见检验指标的正常范围
        normal_ranges = {
            'white_blood_cell': (4.0, 10.0),
            'red_blood_cell': (4.2, 5.4),
            'hemoglobin': (120, 160),
            'platelet_count': (150, 400),
            'glucose': (3.9, 6.1),
            'creatinine': (53, 106),
            'urea': (2.5, 7.5),
            'alt': (7, 56),
            'ast': (13, 35)
        }
        
        for indicator, value in lab_results.items():
            if indicator in normal_ranges:
                min_val, max_val = normal_ranges[indicator]
                # 标准化为正常范围内的相对位置
                normalized_value = (value - min_val) / (max_val - min_val)
                standardized_features.append(normalized_value)
            else:
                # 未知指标使用原值标准化
                standardized_features.append(value / 100.0)
        
        # 补齐到固定维度
        while len(standardized_features) < 20:
            standardized_features.append(0.0)
        
        return np.array(standardized_features[:20])
    
    def calculate_cross_modal_similarity(
        self, 
        query_features: Dict[str, ModalityFeature]
    ) -> List[Dict[str, Any]]:
        """计算跨模态相似性"""
        similarity_results = []
        
        # 与索引中的每个病例计算相似度
        for case_id, indexed_features in self.vector_index.items():
            total_similarity = 0.0
            modality_count = 0
            
            for modality, query_feature in query_features.items():
                if modality in indexed_features:
                    indexed_feature = indexed_features[modality]
                    
                    # 计算余弦相似度
                    similarity = self._cosine_similarity(
                        query_feature.feature_vector,
                        indexed_feature.feature_vector
                    )
                    
                    # 加权平均（考虑置信度）
                    weight = (query_feature.confidence + indexed_feature.confidence) / 2
                    total_similarity += similarity * weight
                    modality_count += 1
            
            if modality_count > 0:
                avg_similarity = total_similarity / modality_count
                similarity_results.append({
                    'case_id': case_id,
                    'similarity_score': avg_similarity,
                    'matched_modalities': modality_count
                })
        
        # 按相似度排序
        similarity_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similarity_results
    
    def match_temporal_patterns(
        self, 
        vital_signs: Dict[str, List[Tuple[datetime, float]]]
    ) -> List[Dict[str, Any]]:
        """匹配时序模式"""
        temporal_matches = []
        
        # 分析生命体征变化趋势
        for indicator, time_series in vital_signs.items():
            if len(time_series) < 2:
                continue
            
            # 计算变化趋势
            values = [value for _, value in time_series]
            trend = self._calculate_trend(values)
            
            # 查找相似的时序模式
            similar_patterns = self._find_similar_temporal_patterns(
                indicator, trend, values
            )
            
            temporal_matches.extend(similar_patterns)
        
        return temporal_matches
    
    def fuse_retrieval_results(
        self, 
        similarity_scores: List[Dict],
        temporal_matches: List[Dict],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """融合检索结果"""
        # 合并相似度得分和时序匹配
        case_scores = {}
        
        # 处理相似度得分
        for result in similarity_scores:
            case_id = result['case_id']
            case_scores[case_id] = {
                'similarity_score': result['similarity_score'],
                'temporal_score': 0.0,
                'final_score': result['similarity_score'] * 0.7,
                'matched_modalities': result['matched_modalities']
            }
        
        # 处理时序匹配得分
        for match in temporal_matches:
            case_id = match['case_id']
            if case_id in case_scores:
                case_scores[case_id]['temporal_score'] = match['temporal_similarity']
                case_scores[case_id]['final_score'] += match['temporal_similarity'] * 0.3
            else:
                case_scores[case_id] = {
                    'similarity_score': 0.0,
                    'temporal_score': match['temporal_similarity'],
                    'final_score': match['temporal_similarity'] * 0.3,
                    'matched_modalities': 0
                }
        
        # 排序并返回top_k结果
        final_results = [
            {
                'case_id': case_id,
                'final_score': scores['final_score'],
                'similarity_score': scores['similarity_score'],
                'temporal_score': scores['temporal_score'],
                'matched_modalities': scores['matched_modalities']
            }
            for case_id, scores in case_scores.items()
        ]
        
        final_results.sort(key=lambda x: x['final_score'], reverse=True)
        return final_results[:top_k]
    
    def _load_biobert_encoder(self):
        """加载BioBERT编码器（演示版本使用模拟）"""
        class MockTokenizer:
            def __call__(self, text, **kwargs):
                return {'input_ids': list(range(10)), 'attention_mask': [1]*10}
        
        class MockModel:
            def __call__(self, **kwargs):
                import numpy as np
                
                class MockOutput:
                    def __init__(self):
                        self.last_hidden_state = np.random.normal(0, 1, (1, 10, 768))
                
                return MockOutput()
        
        class MockBioBERTEncoder:
            def __init__(self):
                print("使用模拟BioBERT编码器进行演示")
                self.tokenizer = MockTokenizer()
                self.model = MockModel()
        
        return MockBioBERTEncoder()
    
    def _load_medical_vision_model(self):
        """加载医学视觉模型"""
        # 这里应该加载专门的医学影像分析模型
        # 为演示目的，返回模拟模型
        return None
    
    def _build_fusion_network(self):
        """构建融合网络（演示版本使用简化实现）"""
        class MockFusionNetwork:
            def __init__(self, text_dim=768, image_dim=512, numeric_dim=20):
                self.text_dim = text_dim
                self.image_dim = image_dim
                self.numeric_dim = numeric_dim
                print("使用模拟融合网络进行演示")
                
            def forward(self, text_feat, image_feat, numeric_feat):
                # 简单的特征拼接作为演示
                import numpy as np
                return np.concatenate([
                    text_feat[:256] if len(text_feat) > 256 else text_feat,
                    image_feat[:256] if len(image_feat) > 256 else image_feat,
                    numeric_feat
                ])
        
        return MockFusionNetwork()
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _extract_ct_features(self, image_info: Dict) -> np.ndarray:
        """提取CT影像特征"""
        # 模拟CT特征提取
        return np.random.normal(0, 1, 512)
    
    def _extract_mri_features(self, image_info: Dict) -> np.ndarray:
        """提取MRI影像特征"""
        # 模拟MRI特征提取
        return np.random.normal(0, 1, 512)
    
    def _extract_xray_features(self, image_info: Dict) -> np.ndarray:
        """提取X光片特征"""
        # 模拟X光片特征提取
        return np.random.normal(0, 1, 512)
    
    def _extract_generic_image_features(self, image_info: Dict) -> np.ndarray:
        """提取通用影像特征"""
        return np.random.normal(0, 1, 512)
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算数值趋势"""
        if len(values) < 2:
            return "stable"
        
        # 简单线性趋势计算
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        slope = coeffs[0]
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _find_similar_temporal_patterns(
        self, 
        indicator: str, 
        trend: str, 
        values: List[float]
    ) -> List[Dict]:
        """查找相似的时序模式"""
        # 模拟时序模式匹配
        return [
            {
                'case_id': f'case_{i}',
                'temporal_similarity': np.random.uniform(0.5, 0.9),
                'matched_indicator': indicator,
                'pattern_type': trend
            }
            for i in range(3)
        ]


# 使用示例
if __name__ == "__main__":
    retriever = MedicalMultimodalRetriever()
    
    # 构造测试病例
    test_case = PatientCase(
        case_id="test_001",
        clinical_notes="患者男性，65岁，主诉胸痛3小时。心电图示ST段抬高，肌钙蛋白升高。",
        medical_images=[
            {'modality': 'X-ray', 'path': '/path/to/chest_xray.jpg'},
            {'modality': 'CT', 'path': '/path/to/cardiac_ct.jpg'}
        ],
        lab_results={
            'creatinine': 98.5,
            'glucose': 7.2,
            'hemoglobin': 135.0,
            'white_blood_cell': 8.5
        },
        vital_signs={
            'blood_pressure': [
                (datetime.now(), 140.0),
                (datetime.now(), 135.0)
            ],
            'heart_rate': [
                (datetime.now(), 85.0),
                (datetime.now(), 88.0)
            ]
        }
    )
    
    # 执行多模态检索
    similar_cases = retriever.retrieve_similar_cases_intelligently(test_case, top_k=5)
    
    print("=== 多模态医疗检索结果 ===")
    for i, case in enumerate(similar_cases, 1):
        print(f"{i}. 病例ID: {case['case_id']}")
        print(f"   综合得分: {case['final_score']:.3f}")
        print(f"   相似度得分: {case['similarity_score']:.3f}")
        print(f"   时序得分: {case['temporal_score']:.3f}")
        print(f"   匹配模态数: {case['matched_modalities']}")
        print("-" * 40)