"""
医疗安全检查器
实现禁忌症检测、药物相互作用检查、剂量合理性验证等安全保障
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, date
import warnings


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"           # 低风险
    MEDIUM = "medium"     # 中风险
    HIGH = "high"         # 高风险
    CRITICAL = "critical" # 严重风险


class SafetyCheckType(Enum):
    """安全检查类型"""
    ALLERGY_CHECK = "allergy_check"           # 过敏史检查
    DRUG_INTERACTION = "drug_interaction"     # 药物相互作用
    CONTRAINDICATION = "contraindication"     # 禁忌症
    DOSAGE_SAFETY = "dosage_safety"          # 剂量安全性
    AGE_RESTRICTION = "age_restriction"       # 年龄限制
    PREGNANCY_SAFETY = "pregnancy_safety"     # 妊娠安全性


@dataclass
class PatientProfile:
    """患者档案"""
    patient_id: str
    age: int
    gender: str  # M/F
    weight: float  # kg
    height: float  # cm
    allergies: List[str]
    current_medications: List[str]
    medical_conditions: List[str]
    pregnancy_status: Optional[bool] = None
    kidney_function: Optional[str] = None  # normal/mild/moderate/severe
    liver_function: Optional[str] = None   # normal/mild/moderate/severe


@dataclass
class SafetyAlert:
    """安全警报"""
    alert_id: str
    check_type: SafetyCheckType
    risk_level: RiskLevel
    title: str
    description: str
    recommendation: str
    evidence: List[str]
    affected_medications: List[str]
    confidence: float


@dataclass
class MedicalAdvice:
    """医疗建议"""
    advice_id: str
    medications: List[Dict[str, Any]]
    treatments: List[str]
    diagnostic_tests: List[str]
    follow_up_plan: str


class MedicalSafetyChecker:
    """医疗安全检查器"""
    
    def __init__(self):
        self.drug_database = self._load_drug_safety_database()
        self.interaction_matrix = self._load_drug_interaction_matrix()
        self.allergy_database = self._load_allergy_database()
        self.contraindication_rules = self._load_contraindication_rules()
        self.dosage_guidelines = self._load_dosage_guidelines()
        
    def comprehensive_safety_check(
        self, 
        medical_advice: MedicalAdvice,
        patient_profile: PatientProfile
    ) -> Dict[str, Any]:
        """像临床药师一样进行全面安全检查"""
        
        safety_alerts = []
        
        # 1. 过敏史安全检查
        allergy_alerts = self.check_allergy_contraindications(
            medical_advice, patient_profile
        )
        safety_alerts.extend(allergy_alerts)
        
        # 2. 药物相互作用检查
        interaction_alerts = self.check_drug_interactions(
            medical_advice, patient_profile
        )
        safety_alerts.extend(interaction_alerts)
        
        # 3. 禁忌症检查
        contraindication_alerts = self.check_medical_contraindications(
            medical_advice, patient_profile
        )
        safety_alerts.extend(contraindication_alerts)
        
        # 4. 剂量安全性检查
        dosage_alerts = self.validate_dosage_safety(
            medical_advice, patient_profile
        )
        safety_alerts.extend(dosage_alerts)
        
        # 5. 特殊人群安全检查
        special_population_alerts = self.check_special_population_safety(
            medical_advice, patient_profile
        )
        safety_alerts.extend(special_population_alerts)
        
        # 6. 生成安全评估报告
        safety_report = self.generate_safety_assessment_report(
            safety_alerts, patient_profile
        )
        
        return {
            'safety_alerts': safety_alerts,
            'safety_report': safety_report,
            'overall_risk_level': self._calculate_overall_risk_level(safety_alerts),
            'recommendations': self._generate_safety_recommendations(safety_alerts),
            'approval_status': self._determine_approval_status(safety_alerts)
        }
    
    def check_allergy_contraindications(
        self, 
        medical_advice: MedicalAdvice,
        patient_profile: PatientProfile
    ) -> List[SafetyAlert]:
        """检查过敏史禁忌"""
        alerts = []
        
        for medication in medical_advice.medications:
            drug_name = medication.get('name', '').lower()
            
            # 检查直接过敏
            for allergy in patient_profile.allergies:
                if self._check_allergy_match(drug_name, allergy.lower()):
                    alert = SafetyAlert(
                        alert_id=f"allergy_{len(alerts)}",
                        check_type=SafetyCheckType.ALLERGY_CHECK,
                        risk_level=RiskLevel.CRITICAL,
                        title="严重过敏风险",
                        description=f"患者对{drug_name}过敏，禁止使用",
                        recommendation="立即停用该药物，考虑替代治疗方案",
                        evidence=[f"患者过敏史记录：{allergy}"],
                        affected_medications=[drug_name],
                        confidence=0.95
                    )
                    alerts.append(alert)
            
            # 检查交叉过敏
            cross_allergies = self._check_cross_allergies(drug_name, patient_profile.allergies)
            for cross_allergy in cross_allergies:
                alert = SafetyAlert(
                    alert_id=f"cross_allergy_{len(alerts)}",
                    check_type=SafetyCheckType.ALLERGY_CHECK,
                    risk_level=RiskLevel.HIGH,
                    title="交叉过敏风险",
                    description=f"{drug_name}可能与已知过敏药物{cross_allergy}存在交叉过敏",
                    recommendation="谨慎使用，密切观察过敏反应，备好抢救药物",
                    evidence=[f"交叉过敏数据库匹配：{cross_allergy}"],
                    affected_medications=[drug_name],
                    confidence=0.8
                )
                alerts.append(alert)
        
        return alerts
    
    def check_drug_interactions(
        self, 
        medical_advice: MedicalAdvice,
        patient_profile: PatientProfile
    ) -> List[SafetyAlert]:
        """检查药物相互作用"""
        alerts = []
        
        # 获取所有药物列表（新开药物 + 正在服用药物）
        new_medications = [med.get('name', '') for med in medical_advice.medications]
        all_medications = new_medications + patient_profile.current_medications
        
        # 两两检查药物相互作用
        for i, drug1 in enumerate(all_medications):
            for drug2 in all_medications[i+1:]:
                interaction = self._check_drug_pair_interaction(drug1, drug2)
                
                if interaction:
                    risk_level = self._assess_interaction_risk_level(interaction)
                    
                    alert = SafetyAlert(
                        alert_id=f"interaction_{len(alerts)}",
                        check_type=SafetyCheckType.DRUG_INTERACTION,
                        risk_level=risk_level,
                        title=f"{drug1} × {drug2} 相互作用",
                        description=interaction['description'],
                        recommendation=interaction['recommendation'],
                        evidence=interaction['evidence'],
                        affected_medications=[drug1, drug2],
                        confidence=interaction['confidence']
                    )
                    alerts.append(alert)
        
        return alerts
    
    def check_medical_contraindications(
        self, 
        medical_advice: MedicalAdvice,
        patient_profile: PatientProfile
    ) -> List[SafetyAlert]:
        """检查医学禁忌症"""
        alerts = []
        
        for medication in medical_advice.medications:
            drug_name = medication.get('name', '')
            
            # 检查疾病禁忌症
            for condition in patient_profile.medical_conditions:
                contraindication = self._check_disease_contraindication(
                    drug_name, condition
                )
                
                if contraindication:
                    alert = SafetyAlert(
                        alert_id=f"contraindication_{len(alerts)}",
                        check_type=SafetyCheckType.CONTRAINDICATION,
                        risk_level=contraindication['risk_level'],
                        title=f"{drug_name}在{condition}中的禁忌",
                        description=contraindication['description'],
                        recommendation=contraindication['recommendation'],
                        evidence=contraindication['evidence'],
                        affected_medications=[drug_name],
                        confidence=contraindication['confidence']
                    )
                    alerts.append(alert)
            
            # 检查器官功能禁忌症
            organ_alerts = self._check_organ_function_contraindications(
                drug_name, patient_profile
            )
            alerts.extend(organ_alerts)
        
        return alerts
    
    def validate_dosage_safety(
        self, 
        medical_advice: MedicalAdvice,
        patient_profile: PatientProfile
    ) -> List[SafetyAlert]:
        """验证剂量安全性"""
        alerts = []
        
        for medication in medical_advice.medications:
            drug_name = medication.get('name', '')
            dosage = medication.get('dosage', 0)
            frequency = medication.get('frequency', '')
            
            # 获取该药物的安全剂量范围
            dosage_guideline = self._get_dosage_guideline(drug_name, patient_profile)
            
            if dosage_guideline:
                # 检查剂量是否超出安全范围
                dosage_safety = self._validate_dosage_range(
                    dosage, frequency, dosage_guideline, patient_profile
                )
                
                if not dosage_safety['is_safe']:
                    risk_level = RiskLevel.HIGH if dosage_safety['severity'] == 'severe' else RiskLevel.MEDIUM
                    
                    alert = SafetyAlert(
                        alert_id=f"dosage_{len(alerts)}",
                        check_type=SafetyCheckType.DOSAGE_SAFETY,
                        risk_level=risk_level,
                        title=f"{drug_name}剂量安全性问题",
                        description=dosage_safety['description'],
                        recommendation=dosage_safety['recommendation'],
                        evidence=dosage_safety['evidence'],
                        affected_medications=[drug_name],
                        confidence=0.9
                    )
                    alerts.append(alert)
        
        return alerts
    
    def check_special_population_safety(
        self, 
        medical_advice: MedicalAdvice,
        patient_profile: PatientProfile
    ) -> List[SafetyAlert]:
        """检查特殊人群安全性"""
        alerts = []
        
        for medication in medical_advice.medications:
            drug_name = medication.get('name', '')
            
            # 年龄限制检查
            age_alerts = self._check_age_restrictions(drug_name, patient_profile.age)
            alerts.extend(age_alerts)
            
            # 妊娠安全性检查
            if patient_profile.pregnancy_status:
                pregnancy_alerts = self._check_pregnancy_safety(drug_name)
                alerts.extend(pregnancy_alerts)
            
            # 儿童用药安全
            if patient_profile.age < 18:
                pediatric_alerts = self._check_pediatric_safety(
                    drug_name, patient_profile.age, patient_profile.weight
                )
                alerts.extend(pediatric_alerts)
            
            # 老年人用药安全
            if patient_profile.age >= 65:
                geriatric_alerts = self._check_geriatric_safety(drug_name, patient_profile.age)
                alerts.extend(geriatric_alerts)
        
        return alerts
    
    def generate_safety_assessment_report(
        self, 
        safety_alerts: List[SafetyAlert],
        patient_profile: PatientProfile
    ) -> Dict[str, Any]:
        """生成安全评估报告"""
        
        # 按风险等级分类警报
        risk_distribution = {
            'critical': [alert for alert in safety_alerts if alert.risk_level == RiskLevel.CRITICAL],
            'high': [alert for alert in safety_alerts if alert.risk_level == RiskLevel.HIGH],
            'medium': [alert for alert in safety_alerts if alert.risk_level == RiskLevel.MEDIUM],
            'low': [alert for alert in safety_alerts if alert.risk_level == RiskLevel.LOW]
        }
        
        # 计算安全评分
        safety_score = self._calculate_safety_score(safety_alerts)
        
        # 生成建议措施
        recommended_actions = self._generate_recommended_actions(safety_alerts)
        
        return {
            'assessment_timestamp': datetime.now().isoformat(),
            'patient_id': patient_profile.patient_id,
            'total_alerts': len(safety_alerts),
            'risk_distribution': {k: len(v) for k, v in risk_distribution.items()},
            'safety_score': safety_score,
            'risk_summary': self._generate_risk_summary(risk_distribution),
            'recommended_actions': recommended_actions,
            'monitoring_requirements': self._generate_monitoring_requirements(safety_alerts),
            'follow_up_schedule': self._generate_follow_up_schedule(safety_alerts)
        }
    
    def _check_allergy_match(self, drug_name: str, allergy: str) -> bool:
        """检查过敏匹配"""
        # 直接匹配
        if drug_name in allergy or allergy in drug_name:
            return True
        
        # 成分匹配
        drug_ingredients = self._get_drug_ingredients(drug_name)
        for ingredient in drug_ingredients:
            if ingredient.lower() in allergy.lower():
                return True
        
        return False
    
    def _check_cross_allergies(self, drug_name: str, allergies: List[str]) -> List[str]:
        """检查交叉过敏"""
        cross_allergies = []
        
        for allergy in allergies:
            # 查询交叉过敏数据库
            if drug_name in self.allergy_database.get(allergy.lower(), {}).get('cross_allergies', []):
                cross_allergies.append(allergy)
        
        return cross_allergies
    
    def _check_drug_pair_interaction(self, drug1: str, drug2: str) -> Optional[Dict]:
        """检查药物对相互作用"""
        # 标准化药物名称
        drug1_clean = self._normalize_drug_name(drug1)
        drug2_clean = self._normalize_drug_name(drug2)
        
        # 查询相互作用矩阵
        interaction_key = f"{drug1_clean}_{drug2_clean}"
        reverse_key = f"{drug2_clean}_{drug1_clean}"
        
        if interaction_key in self.interaction_matrix:
            return self.interaction_matrix[interaction_key]
        elif reverse_key in self.interaction_matrix:
            return self.interaction_matrix[reverse_key]
        
        return None
    
    def _load_drug_safety_database(self) -> Dict:
        """加载药物安全数据库"""
        return {
            '阿司匹林': {
                'contraindications': ['消化性溃疡', '出血性疾病'],
                'pregnancy_category': 'C',
                'age_restrictions': {'min_age': 12},
                'interactions': ['华法林', '肝素']
            },
            '美托洛尔': {
                'contraindications': ['支气管哮喘', '严重心动过缓'],
                'pregnancy_category': 'C',
                'age_restrictions': {},
                'interactions': ['维拉帕米', '地尔硫卓']
            }
        }
    
    def _load_drug_interaction_matrix(self) -> Dict:
        """加载药物相互作用矩阵"""
        return {
            '阿司匹林_华法林': {
                'description': '增加出血风险',
                'mechanism': '协同抗凝作用',
                'recommendation': '调整华法林剂量，密切监测凝血功能',
                'evidence': ['药理学研究', '临床试验数据'],
                'confidence': 0.9,
                'severity': 'major'
            },
            '美托洛尔_维拉帕米': {
                'description': '可能导致严重心动过缓和房室传导阻滞',
                'mechanism': '双重阻断心脏传导系统',
                'recommendation': '避免联用，如必须使用需密切监测心电图',
                'evidence': ['心血管药理学研究'],
                'confidence': 0.95,
                'severity': 'major'
            }
        }


# 使用示例
if __name__ == "__main__":
    safety_checker = MedicalSafetyChecker()
    
    # 构造测试患者档案
    test_patient = PatientProfile(
        patient_id="patient_001",
        age=68,
        gender="M",
        weight=75.0,
        height=170.0,
        allergies=["青霉素", "磺胺类"],
        current_medications=["阿司匹林", "美托洛尔"],
        medical_conditions=["高血压", "冠心病"],
        kidney_function="mild"
    )
    
    # 构造测试医疗建议
    test_advice = MedicalAdvice(
        advice_id="advice_001",
        medications=[
            {
                'name': '华法林',
                'dosage': 5.0,
                'frequency': '每日一次',
                'duration': '长期'
            },
            {
                'name': '阿莫西林',
                'dosage': 500.0,
                'frequency': '每日三次',
                'duration': '7天'
            }
        ],
        treatments=["冠状动脉造影"],
        diagnostic_tests=["凝血功能检查"],
        follow_up_plan="一周后复查"
    )
    
    # 执行安全检查
    safety_result = safety_checker.comprehensive_safety_check(test_advice, test_patient)
    
    print("=== 医疗安全检查报告 ===")
    print(f"总风险等级: {safety_result['overall_risk_level'].value}")
    print(f"警报数量: {len(safety_result['safety_alerts'])}")
    print(f"批准状态: {safety_result['approval_status']}")
    
    print(f"\n安全警报:")
    for i, alert in enumerate(safety_result['safety_alerts'], 1):
        print(f"{i}. [{alert.risk_level.value.upper()}] {alert.title}")
        print(f"   描述: {alert.description}")
        print(f"   建议: {alert.recommendation}")
        print(f"   涉及药物: {', '.join(alert.affected_medications)}")
        print("-" * 50)
    
    print(f"\n安全评分: {safety_result['safety_report']['safety_score']:.2f}/100")
    print(f"建议措施: {len(safety_result['recommendations'])}项")