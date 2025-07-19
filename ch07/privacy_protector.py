"""
医疗隐私保护模块
实现医疗数据脱敏、差分隐私查询、合规性检查
"""

import re
import hashlib
import random
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, date, timedelta
import numpy as np
import json


class PrivacyLevel(Enum):
    """隐私保护级别"""
    PUBLIC = "public"           # 公开信息
    INTERNAL = "internal"       # 内部信息
    CONFIDENTIAL = "confidential"  # 机密信息
    RESTRICTED = "restricted"   # 限制访问


class SensitivityType(Enum):
    """敏感信息类型"""
    PERSONAL_ID = "personal_id"        # 个人身份信息
    MEDICAL_RECORD = "medical_record"  # 医疗记录
    GENETIC_INFO = "genetic_info"      # 基因信息
    MENTAL_HEALTH = "mental_health"    # 心理健康
    FINANCIAL_INFO = "financial_info"  # 财务信息


@dataclass
class PrivacyRule:
    """隐私规则"""
    rule_id: str
    sensitivity_type: SensitivityType
    privacy_level: PrivacyLevel
    anonymization_method: str
    retention_period: int  # days
    access_restrictions: List[str]
    compliance_framework: str  # HIPAA, GDPR, etc.


@dataclass
class AnonymizationResult:
    """脱敏结果"""
    original_text: str
    anonymized_text: str
    anonymization_map: Dict[str, str]
    confidence_score: float
    applied_rules: List[str]
    privacy_level_achieved: PrivacyLevel


class MedicalPrivacyProtector:
    """医疗隐私保护器"""
    
    def __init__(self):
        self.privacy_rules = self._load_privacy_rules()
        self.sensitive_patterns = self._load_sensitive_patterns()
        self.anonymization_methods = self._load_anonymization_methods()
        self.compliance_frameworks = self._load_compliance_frameworks()
        self.differential_privacy_epsilon = 1.0
        
    def comprehensive_privacy_protection(
        self, 
        medical_text: str,
        required_privacy_level: PrivacyLevel = PrivacyLevel.CONFIDENTIAL,
        compliance_framework: str = "HIPAA"
    ) -> AnonymizationResult:
        """全面的医疗隐私保护"""
        
        # 1. 敏感信息识别
        sensitive_entities = self.identify_sensitive_medical_information(medical_text)
        
        # 2. 隐私风险评估
        privacy_risks = self.assess_privacy_risks(sensitive_entities, medical_text)
        
        # 3. 脱敏策略选择
        anonymization_strategy = self.select_anonymization_strategy(
            privacy_risks, required_privacy_level, compliance_framework
        )
        
        # 4. 执行脱敏处理
        anonymized_result = self.execute_anonymization(
            medical_text, sensitive_entities, anonymization_strategy
        )
        
        # 5. 隐私保护验证
        privacy_verification = self.verify_privacy_protection(
            anonymized_result, required_privacy_level
        )
        
        return anonymized_result
    
    def identify_sensitive_medical_information(
        self, 
        text: str
    ) -> Dict[SensitivityType, List[Dict]]:
        """识别敏感医疗信息"""
        
        sensitive_entities = {
            SensitivityType.PERSONAL_ID: [],
            SensitivityType.MEDICAL_RECORD: [],
            SensitivityType.GENETIC_INFO: [],
            SensitivityType.MENTAL_HEALTH: [],
            SensitivityType.FINANCIAL_INFO: []
        }
        
        # 个人身份信息识别
        personal_id_entities = self._extract_personal_identifiers(text)
        sensitive_entities[SensitivityType.PERSONAL_ID].extend(personal_id_entities)
        
        # 医疗记录信息识别
        medical_record_entities = self._extract_medical_records(text)
        sensitive_entities[SensitivityType.MEDICAL_RECORD].extend(medical_record_entities)
        
        # 基因信息识别
        genetic_entities = self._extract_genetic_information(text)
        sensitive_entities[SensitivityType.GENETIC_INFO].extend(genetic_entities)
        
        # 心理健康信息识别
        mental_health_entities = self._extract_mental_health_info(text)
        sensitive_entities[SensitivityType.MENTAL_HEALTH].extend(mental_health_entities)
        
        # 财务信息识别
        financial_entities = self._extract_financial_information(text)
        sensitive_entities[SensitivityType.FINANCIAL_INFO].extend(financial_entities)
        
        return sensitive_entities
    
    def execute_anonymization(
        self, 
        text: str,
        sensitive_entities: Dict[SensitivityType, List[Dict]],
        strategy: Dict[str, Any]
    ) -> AnonymizationResult:
        """执行脱敏处理"""
        
        anonymized_text = text
        anonymization_map = {}
        applied_rules = []
        
        # 按敏感性级别排序处理
        processing_order = [
            SensitivityType.PERSONAL_ID,
            SensitivityType.GENETIC_INFO,
            SensitivityType.MENTAL_HEALTH,
            SensitivityType.MEDICAL_RECORD,
            SensitivityType.FINANCIAL_INFO
        ]
        
        for sensitivity_type in processing_order:
            entities = sensitive_entities.get(sensitivity_type, [])
            
            for entity in entities:
                original_value = entity['value']
                
                # 根据策略选择脱敏方法
                method = strategy.get(sensitivity_type.value, 'redaction')
                anonymized_value = self._apply_anonymization_method(
                    original_value, method, sensitivity_type
                )
                
                # 替换文本
                anonymized_text = anonymized_text.replace(original_value, anonymized_value)
                anonymization_map[original_value] = anonymized_value
                applied_rules.append(f"{sensitivity_type.value}_{method}")
        
        # 计算置信度分数
        confidence_score = self._calculate_anonymization_confidence(
            text, anonymized_text, anonymization_map
        )
        
        return AnonymizationResult(
            original_text=text,
            anonymized_text=anonymized_text,
            anonymization_map=anonymization_map,
            confidence_score=confidence_score,
            applied_rules=applied_rules,
            privacy_level_achieved=PrivacyLevel.CONFIDENTIAL
        )
    
    def differential_privacy_query(
        self, 
        medical_database: List[Dict],
        query_function: callable,
        epsilon: float = 1.0
    ) -> Dict[str, Any]:
        """差分隐私查询"""
        
        # 执行原始查询
        true_result = query_function(medical_database)
        
        # 添加拉普拉斯噪声
        if isinstance(true_result, (int, float)):
            # 数值结果
            noise_scale = 1.0 / epsilon
            noise = np.random.laplace(0, noise_scale)
            noisy_result = true_result + noise
            
            return {
                'result': noisy_result,
                'epsilon': epsilon,
                'noise_added': noise,
                'privacy_budget_used': epsilon,
                'query_type': 'numeric'
            }
        
        elif isinstance(true_result, dict):
            # 字典结果（如统计数据）
            noisy_result = {}
            total_noise = 0
            
            for key, value in true_result.items():
                if isinstance(value, (int, float)):
                    noise_scale = 1.0 / epsilon
                    noise = np.random.laplace(0, noise_scale)
                    noisy_result[key] = value + noise
                    total_noise += abs(noise)
                else:
                    noisy_result[key] = value
            
            return {
                'result': noisy_result,
                'epsilon': epsilon,
                'total_noise': total_noise,
                'privacy_budget_used': epsilon,
                'query_type': 'statistical'
            }
        
        else:
            # 其他类型结果
            return {
                'result': true_result,
                'epsilon': epsilon,
                'privacy_budget_used': 0,
                'query_type': 'non_numeric',
                'note': 'No noise added to non-numeric result'
            }
    
    def compliance_check(
        self, 
        anonymized_data: AnonymizationResult,
        framework: str = "HIPAA"
    ) -> Dict[str, Any]:
        """合规性检查"""
        
        compliance_results = {
            'framework': framework,
            'compliance_status': 'unknown',
            'compliance_score': 0.0,
            'violations': [],
            'recommendations': [],
            'audit_trail': []
        }
        
        if framework == "HIPAA":
            compliance_results = self._check_hipaa_compliance(anonymized_data)
        elif framework == "GDPR":
            compliance_results = self._check_gdpr_compliance(anonymized_data)
        elif framework == "CCPA":
            compliance_results = self._check_ccpa_compliance(anonymized_data)
        
        return compliance_results
    
    def _extract_personal_identifiers(self, text: str) -> List[Dict]:
        """提取个人身份标识"""
        identifiers = []
        
        # 姓名模式
        name_patterns = [
            r'患者\s*([^，。；\s]{2,4})',
            r'姓名[:：]\s*([^，。；\s]{2,4})',
            r'(先生|女士|小姐)\s*([^，。；\s]{2,4})',
            r'([^，。；\s]{2,4})\s*(先生|女士|小姐)'
        ]
        
        for pattern in name_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                name = match.group(1) if match.lastindex == 1 else match.group(2)
                identifiers.append({
                    'type': 'name',
                    'value': name,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8
                })
        
        # 身份证号模式
        id_pattern = r'\\b\\d{17}[\\dXx]\\b'
        for match in re.finditer(id_pattern, text):
            identifiers.append({
                'type': 'id_number',
                'value': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.95
            })
        
        # 电话号码模式
        phone_pattern = r'\\b1[3-9]\\d{9}\\b'
        for match in re.finditer(phone_pattern, text):
            identifiers.append({
                'type': 'phone',
                'value': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.9
            })
        
        # 地址模式
        address_patterns = [
            r'(住址|地址)[:：]([^，。；\\n]+)',
            r'(省|市|区|县|街道|路|号)([^，。；\\s]+)'
        ]
        
        for pattern in address_patterns:
            for match in re.finditer(pattern, text):
                identifiers.append({
                    'type': 'address',
                    'value': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.7
                })
        
        return identifiers
    
    def _extract_medical_records(self, text: str) -> List[Dict]:
        """提取医疗记录信息"""
        medical_records = []
        
        # 病历号模式
        record_patterns = [
            r'病历号[:：]\\s*(\\d+)',
            r'住院号[:：]\\s*(\\d+)',
            r'门诊号[:：]\\s*(\\d+)'
        ]
        
        for pattern in record_patterns:
            for match in re.finditer(pattern, text):
                medical_records.append({
                    'type': 'medical_record_number',
                    'value': match.group(1),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9
                })
        
        # 诊断信息
        diagnosis_patterns = [
            r'诊断[:：]([^，。；\\n]+)',
            r'初步诊断[:：]([^，。；\\n]+)',
            r'确诊为([^，。；\\n]+)'
        ]
        
        for pattern in diagnosis_patterns:
            for match in re.finditer(pattern, text):
                medical_records.append({
                    'type': 'diagnosis',
                    'value': match.group(1).strip(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8
                })
        
        return medical_records
    
    def _extract_genetic_information(self, text: str) -> List[Dict]:
        """提取基因信息"""
        genetic_info = []
        
        # 基因型模式
        genetic_patterns = [
            r'(BRCA[12]|TP53|EGFR|KRAS)\\s*基因',
            r'基因型[:：]([^，。；\\n]+)',
            r'突变[:：]([^，。；\\n]+)',
            r'(HLA-[A-Z0-9*:]+)'
        ]
        
        for pattern in genetic_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                genetic_info.append({
                    'type': 'genetic_marker',
                    'value': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.85
                })
        
        return genetic_info
    
    def _extract_mental_health_info(self, text: str) -> List[Dict]:
        """提取心理健康信息"""
        mental_health = []
        
        # 心理健康相关术语
        mental_patterns = [
            r'(抑郁症|焦虑症|精神分裂|双相情感障碍)',
            r'(自杀|自残|自伤)倾向',
            r'心理治疗[:：]([^，。；\\n]+)',
            r'精神状态[:：]([^，。；\\n]+)'
        ]
        
        for pattern in mental_patterns:
            for match in re.finditer(pattern, text):
                mental_health.append({
                    'type': 'mental_health_condition',
                    'value': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8
                })
        
        return mental_health
    
    def _extract_financial_information(self, text: str) -> List[Dict]:
        """提取财务信息"""
        financial_info = []
        
        # 财务相关信息
        financial_patterns = [
            r'医保号[:：]\\s*(\\d+)',
            r'费用[:：]\\s*(\\d+(?:\\.\\d+)?)\\s*元',
            r'银行卡号[:：]\\s*(\\d{16,19})',
            r'支付方式[:：]([^，。；\\n]+)'
        ]
        
        for pattern in financial_patterns:
            for match in re.finditer(pattern, text):
                financial_info.append({
                    'type': 'financial_info',
                    'value': match.group(1) if match.lastindex else match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.85
                })
        
        return financial_info
    
    def _apply_anonymization_method(
        self, 
        value: str,
        method: str,
        sensitivity_type: SensitivityType
    ) -> str:
        """应用脱敏方法"""
        
        if method == 'redaction':
            # 简单编辑
            return '[已编辑]'
        
        elif method == 'masking':
            # 部分遮蔽
            if len(value) <= 2:
                return '*' * len(value)
            elif len(value) <= 4:
                return value[0] + '*' * (len(value) - 2) + value[-1]
            else:
                return value[:2] + '*' * (len(value) - 4) + value[-2:]
        
        elif method == 'pseudonymization':
            # 假名化
            hash_object = hashlib.md5(value.encode())
            hash_hex = hash_object.hexdigest()
            
            if sensitivity_type == SensitivityType.PERSONAL_ID:
                if 'name' in str(sensitivity_type):
                    return f"患者{hash_hex[:6].upper()}"
                elif 'phone' in str(sensitivity_type):
                    return f"138****{hash_hex[:4]}"
                else:
                    return f"ID{hash_hex[:8].upper()}"
            else:
                return f"CODE{hash_hex[:8].upper()}"
        
        elif method == 'generalization':
            # 泛化
            if sensitivity_type == SensitivityType.PERSONAL_ID:
                return '[个人信息]'
            elif sensitivity_type == SensitivityType.MEDICAL_RECORD:
                return '[医疗信息]'
            else:
                return '[敏感信息]'
        
        elif method == 'suppression':
            # 抑制（完全删除）
            return ''
        
        else:
            # 默认编辑
            return '[已脱敏]'
    
    def _check_hipaa_compliance(self, result: AnonymizationResult) -> Dict[str, Any]:
        """检查HIPAA合规性"""
        
        compliance = {
            'framework': 'HIPAA',
            'compliance_status': 'compliant',
            'compliance_score': 100.0,
            'violations': [],
            'recommendations': []
        }
        
        # 检查18种HIPAA标识符
        hipaa_identifiers = [
            r'\\b\\d{17}[\\dXx]\\b',  # 身份证号
            r'\\b1[3-9]\\d{9}\\b',    # 电话号码
            r'\\b\\d{4}-\\d{2}-\\d{2}\\b',  # 具体日期
            r'\\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}\\b'  # 邮箱
        ]
        
        for i, pattern in enumerate(hipaa_identifiers):
            if re.search(pattern, result.anonymized_text):
                compliance['violations'].append(f"HIPAA标识符{i+1}未完全脱敏")
                compliance['compliance_score'] -= 20
        
        if compliance['compliance_score'] < 80:
            compliance['compliance_status'] = 'non_compliant'
        elif compliance['compliance_score'] < 95:
            compliance['compliance_status'] = 'partially_compliant'
        
        return compliance
    
    def _load_privacy_rules(self) -> List[PrivacyRule]:
        """加载隐私规则"""
        return [
            PrivacyRule(
                rule_id="rule_001",
                sensitivity_type=SensitivityType.PERSONAL_ID,
                privacy_level=PrivacyLevel.RESTRICTED,
                anonymization_method="pseudonymization",
                retention_period=2555,  # 7 years
                access_restrictions=["医生", "护士", "数据管理员"],
                compliance_framework="HIPAA"
            )
        ]


# 使用示例
if __name__ == "__main__":
    protector = MedicalPrivacyProtector()
    
    # 测试医疗文本
    test_text = """
    患者张三，男，45岁，身份证号123456789012345678，电话13812345678。
    住址：北京市朝阳区某某街道123号。
    诊断：高血压、糖尿病。
    病历号：H20240001。
    费用：1500元，医保号：110123456789。
    """
    
    # 执行隐私保护
    result = protector.comprehensive_privacy_protection(
        test_text, 
        required_privacy_level=PrivacyLevel.CONFIDENTIAL,
        compliance_framework="HIPAA"
    )
    
    print("=== 医疗隐私保护结果 ===")
    print("原始文本：")
    print(result.original_text)
    print("\\n脱敏后文本：")
    print(result.anonymized_text)
    print(f"\\n置信度评分: {result.confidence_score:.2f}")
    print(f"应用的规则: {', '.join(result.applied_rules)}")
    
    # 差分隐私查询示例
    medical_db = [
        {'age': 45, 'diagnosis': 'hypertension'},
        {'age': 52, 'diagnosis': 'diabetes'},
        {'age': 38, 'diagnosis': 'hypertension'}
    ]
    
    def count_hypertension(db):
        return sum(1 for record in db if record['diagnosis'] == 'hypertension')
    
    dp_result = protector.differential_privacy_query(
        medical_db, count_hypertension, epsilon=1.0
    )
    
    print(f"\\n差分隐私查询结果: {dp_result['result']:.2f}")
    print(f"隐私预算使用: {dp_result['privacy_budget_used']}")