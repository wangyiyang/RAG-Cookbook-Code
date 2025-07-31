"""
隐私影响评估系统
Deep RAG Notes Chapter 12 - Privacy Protection Technologies
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

class RiskLevel(Enum):
    """风险级别枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceStatus(Enum):
    """合规状态枚举"""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    UNKNOWN = "unknown"

@dataclass
class RiskItem:
    """风险项"""
    category: str
    description: str
    severity: RiskLevel
    likelihood: str
    impact: str
    mitigation_required: bool
    recommendations: List[str]

@dataclass
class ComplianceItem:
    """合规项"""
    regulation: str
    requirement: str
    status: ComplianceStatus
    evidence: str
    gaps: List[str]
    actions_required: List[str]

@dataclass
class AssessmentReport:
    """评估报告"""
    assessment_id: str
    assessment_date: str
    system_name: str
    system_configuration: Dict[str, Any]
    privacy_risks: List[RiskItem]
    compliance_items: List[ComplianceItem]
    overall_privacy_score: float
    recommendations: List[str]
    next_review_date: str

class PrivacyImpactAssessment:
    """隐私影响评估系统"""
    
    def __init__(self):
        """初始化隐私影响评估系统"""
        self.assessment_framework = self.initialize_assessment_framework()
        self.regulation_requirements = self.load_regulation_requirements()
        self.risk_matrix = self.load_risk_matrix()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def initialize_assessment_framework(self) -> Dict[str, Any]:
        """初始化评估框架"""
        return {
            'assessment_criteria': {
                'data_collection': {
                    'description': '数据收集过程的隐私风险',
                    'weight': 0.2,
                    'checkpoints': [
                        '数据收集的合法性基础',
                        '数据最小化原则遵循',
                        '用户同意机制',
                        '数据收集透明度'
                    ]
                },
                'data_storage': {
                    'description': '数据存储阶段的隐私风险',
                    'weight': 0.25,
                    'checkpoints': [
                        '数据加密状态',
                        '访问控制机制',
                        '数据备份安全',
                        '存储位置合规性'
                    ]
                },
                'data_processing': {
                    'description': '数据处理过程的隐私风险',
                    'weight': 0.3,
                    'checkpoints': [
                        '处理目的限制',
                        '自动化决策保护',
                        '数据质量保证',
                        '处理记录完整性'
                    ]
                },
                'data_sharing': {
                    'description': '数据共享和传输的隐私风险',
                    'weight': 0.15,
                    'checkpoints': [
                        '第三方共享控制',
                        '跨境传输保护',
                        '数据接收方评估',
                        '共享协议合规性'
                    ]
                },
                'user_rights': {
                    'description': '用户权利保护情况',
                    'weight': 0.1,
                    'checkpoints': [
                        '访问权实现',
                        '更正权实现',
                        '删除权实现',
                        '可携带权实现'
                    ]
                }
            },
            'scoring_method': 'weighted_average',
            'minimum_score_threshold': 70.0
        }
    
    def load_regulation_requirements(self) -> Dict[str, Dict]:
        """加载法规要求"""
        return {
            'GDPR': {
                'region': ['EU', 'EEA'],
                'applicability_criteria': [
                    '在欧盟境内提供商品或服务',
                    '监控欧盟境内个人行为',
                    '处理欧盟居民个人数据'
                ],
                'key_requirements': {
                    'lawful_basis': {
                        'description': '数据处理的合法基础',
                        'compliance_checks': [
                            '明确的合法基础',
                            '同意机制（如适用）',
                            '合法利益评估（如适用）'
                        ]
                    },
                    'data_subject_rights': {
                        'description': '数据主体权利',
                        'compliance_checks': [
                            '访问权实现机制',
                            '更正和删除权实现',
                            '数据可携带权实现',
                            '反对权实现机制'
                        ]
                    },
                    'privacy_by_design': {
                        'description': '隐私设计和默认隐私',
                        'compliance_checks': [
                            '技术和组织措施',
                            '数据保护影响评估',
                            '数据保护官指定'
                        ]
                    },
                    'breach_notification': {
                        'description': '数据泄露通知',
                        'compliance_checks': [
                            '72小时内向监管机构通知',
                            '高风险时通知个人',
                            '泄露记录保持'
                        ]
                    }
                },
                'penalties': {
                    'administrative_fines': '最高2000万欧元或年营业额4%',
                    'other_measures': '处理限制、数据保护审计'
                }
            },
            'CCPA': {
                'region': ['California'],
                'applicability_criteria': [
                    '在加州开展业务',
                    '年收入超过2500万美元',
                    '处理5万以上消费者个人信息'
                ],
                'key_requirements': {
                    'transparency': {
                        'description': '透明度要求',
                        'compliance_checks': [
                            '隐私政策完整性',
                            '收集信息类别披露',
                            '信息使用目的说明'
                        ]
                    },
                    'consumer_rights': {
                        'description': '消费者权利',
                        'compliance_checks': [
                            '知情权实现',
                            '删除权实现',
                            '选择退出销售权',
                            '非歧视权保护'
                        ]
                    }
                },
                'penalties': {
                    'civil_penalties': '每次违规最高2500美元',
                    'intentional_violations': '每次违规最高7500美元'
                }
            },
            'PIPL': {
                'region': ['China'],
                'applicability_criteria': [
                    '在中国境内处理个人信息',
                    '境外处理境内个人信息',
                    '为向境内提供产品服务'
                ],
                'key_requirements': {
                    'consent_mechanism': {
                        'description': '个人信息处理同意',
                        'compliance_checks': [
                            '明确同意机制',
                            '敏感信息单独同意',
                            '同意撤回机制'
                        ]
                    },
                    'cross_border_transfer': {
                        'description': '跨境传输限制',
                        'compliance_checks': [
                            '安全评估通过',
                            '认证机构认证',
                            '标准合同条款'
                        ]
                    },
                    'data_localization': {
                        'description': '数据本地化要求',
                        'compliance_checks': [
                            '关键信息基础设施数据本地存储',
                            '达到标准的境内存储',
                            '出境评估程序'
                        ]
                    }
                },
                'penalties': {
                    'administrative_penalties': '最高5000万元或年营业额5%',
                    'criminal_liability': '严重情况可追究刑事责任'
                }
            }
        }
    
    def load_risk_matrix(self) -> Dict[str, Dict]:
        """加载风险矩阵"""
        return {
            'likelihood_levels': {
                'very_low': {'score': 1, 'description': '几乎不可能发生'},
                'low': {'score': 2, 'description': '不太可能发生'},
                'medium': {'score': 3, 'description': '可能发生'},
                'high': {'score': 4, 'description': '很可能发生'},
                'very_high': {'score': 5, 'description': '几乎确定发生'}
            },
            'impact_levels': {
                'negligible': {'score': 1, 'description': '影响微乎其微'},
                'minor': {'score': 2, 'description': '轻微影响'},
                'moderate': {'score': 3, 'description': '中等影响'},
                'major': {'score': 4, 'description': '重大影响'},
                'severe': {'score': 5, 'description': '严重影响'}
            },
            'risk_calculation': 'likelihood * impact',
            'risk_thresholds': {
                'low': (1, 6),
                'medium': (7, 12),
                'high': (13, 18),
                'critical': (19, 25)
            }
        }
    
    def conduct_comprehensive_assessment(self, 
                                       system_name: str,
                                       rag_system_config: Dict[str, Any],
                                       business_context: Dict[str, Any]) -> AssessmentReport:
        """进行全面的隐私影响评估"""
        
        assessment_id = f"PIA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"开始隐私影响评估: {assessment_id}")
        
        # 1. 数据流风险分析
        privacy_risks = []
        
        # 数据收集风险
        collection_risks = self.assess_data_collection_risks(rag_system_config, business_context)
        privacy_risks.extend(collection_risks)
        
        # 数据存储风险
        storage_risks = self.assess_data_storage_risks(rag_system_config, business_context)
        privacy_risks.extend(storage_risks)
        
        # 数据处理风险
        processing_risks = self.assess_data_processing_risks(rag_system_config, business_context)
        privacy_risks.extend(processing_risks)
        
        # 数据共享风险
        sharing_risks = self.assess_data_sharing_risks(rag_system_config, business_context)
        privacy_risks.extend(sharing_risks)
        
        # 用户权利保护风险
        rights_risks = self.assess_user_rights_risks(rag_system_config, business_context)
        privacy_risks.extend(rights_risks)
        
        # 2. 合规性评估
        compliance_items = self.assess_regulatory_compliance(rag_system_config, business_context)
        
        # 3. 计算整体隐私评分
        overall_score = self.calculate_overall_privacy_score(privacy_risks, compliance_items)
        
        # 4. 生成建议
        recommendations = self.generate_comprehensive_recommendations(privacy_risks, compliance_items)
        
        # 5. 创建评估报告
        report = AssessmentReport(
            assessment_id=assessment_id,
            assessment_date=datetime.now().isoformat(),
            system_name=system_name,
            system_configuration=rag_system_config,
            privacy_risks=privacy_risks,
            compliance_items=compliance_items,
            overall_privacy_score=overall_score,
            recommendations=recommendations,
            next_review_date=self.calculate_next_review_date(overall_score)
        )
        
        self.logger.info(f"隐私影响评估完成，总体评分: {overall_score:.1f}")
        return report
    
    def assess_data_collection_risks(self, config: Dict, context: Dict) -> List[RiskItem]:
        """评估数据收集风险"""
        risks = []
        
        # 检查数据收集范围
        data_sources = config.get('data_sources', [])
        sensitive_sources = [src for src in data_sources if src.get('sensitivity') in ['high', 'critical']]
        
        if sensitive_sources:
            risks.append(RiskItem(
                category='data_collection',
                description='收集敏感数据源存在隐私风险',
                severity=RiskLevel.HIGH,
                likelihood='medium',
                impact='major',
                mitigation_required=True,
                recommendations=[
                    '实施数据最小化原则',
                    '增强数据收集同意机制',
                    '定期审查数据收集必要性'
                ]
            ))
        
        # 检查用户同意机制
        consent_mechanism = config.get('consent_mechanism', {})
        if not consent_mechanism.get('explicit_consent', False):
            risks.append(RiskItem(
                category='data_collection',
                description='缺乏明确的用户同意机制',
                severity=RiskLevel.MEDIUM,
                likelihood='high',
                impact='moderate',
                mitigation_required=True,
                recommendations=[
                    '实施明确的用户同意流程',
                    '提供同意撤回机制',
                    '记录同意证据'
                ]
            ))
        
        return risks
    
    def assess_data_storage_risks(self, config: Dict, context: Dict) -> List[RiskItem]:
        """评估数据存储风险"""
        risks = []
        
        # 检查加密状态
        encryption_config = config.get('encryption', {})
        if not encryption_config.get('at_rest', False):
            risks.append(RiskItem(
                category='data_storage',
                description='数据静态加密未启用',
                severity=RiskLevel.HIGH,
                likelihood='high',
                impact='major',
                mitigation_required=True,
                recommendations=[
                    '启用数据静态加密',
                    '使用强加密算法',
                    '定期轮换加密密钥'
                ]
            ))
        
        # 检查访问控制
        access_control = config.get('access_control', {})
        if not access_control.get('role_based', False):
            risks.append(RiskItem(
                category='data_storage',
                description='缺乏基于角色的访问控制',
                severity=RiskLevel.MEDIUM,
                likelihood='medium',
                impact='moderate',
                mitigation_required=True,
                recommendations=[
                    '实施基于角色的访问控制',
                    '定期审查访问权限',
                    '实施最小权限原则'
                ]
            ))
        
        # 检查数据备份安全
        backup_config = config.get('backup', {})
        if backup_config.get('enabled', False) and not backup_config.get('encrypted', False):
            risks.append(RiskItem(
                category='data_storage',
                description='数据备份未加密',
                severity=RiskLevel.MEDIUM,
                likelihood='medium',
                impact='moderate',
                mitigation_required=True,
                recommendations=[
                    '启用备份数据加密',
                    '限制备份访问权限',
                    '定期测试备份恢复'
                ]
            ))
        
        return risks
    
    def assess_data_processing_risks(self, config: Dict, context: Dict) -> List[RiskItem]:
        """评估数据处理风险"""
        risks = []
        
        # 检查处理目的限制
        processing_purposes = config.get('processing_purposes', [])
        if len(processing_purposes) > 5:  # 假设超过5个用途可能存在过度处理
            risks.append(RiskItem(
                category='data_processing',
                description='数据处理目的过于广泛',
                severity=RiskLevel.MEDIUM,
                likelihood='medium',
                impact='moderate',
                mitigation_required=True,
                recommendations=[
                    '明确和限制处理目的',
                    '实施目的绑定控制',
                    '定期审查处理必要性'
                ]
            ))
        
        # 检查自动化决策
        automated_decision = config.get('automated_decision_making', {})
        if automated_decision.get('enabled', False) and not automated_decision.get('human_review', False):
            risks.append(RiskItem(
                category='data_processing',
                description='自动化决策缺乏人工审查',
                severity=RiskLevel.HIGH,
                likelihood='high',
                impact='major',
                mitigation_required=True,
                recommendations=[
                    '增加人工审查机制',
                    '提供决策解释功能',
                    '建立申诉渠道'
                ]
            ))
        
        # 检查数据质量保证
        data_quality = config.get('data_quality', {})
        if not data_quality.get('validation_enabled', False):
            risks.append(RiskItem(
                category='data_processing',
                description='缺乏数据质量验证机制',
                severity=RiskLevel.LOW,
                likelihood='medium',
                impact='minor',
                mitigation_required=False,
                recommendations=[
                    '实施数据质量验证',
                    '建立数据清洗流程',
                    '监控数据准确性'
                ]
            ))
        
        return risks
    
    def assess_data_sharing_risks(self, config: Dict, context: Dict) -> List[RiskItem]:
        """评估数据共享风险"""
        risks = []
        
        # 检查第三方共享
        third_party_sharing = config.get('third_party_sharing', {})
        if third_party_sharing.get('enabled', False):
            if not third_party_sharing.get('data_processing_agreement', False):
                risks.append(RiskItem(
                    category='data_sharing',
                    description='缺乏数据处理协议',
                    severity=RiskLevel.HIGH,
                    likelihood='high',
                    impact='major',
                    mitigation_required=True,
                    recommendations=[
                        '签署数据处理协议',
                        '评估第三方安全能力',
                        '定期审查共享必要性'
                    ]
                ))
        
        # 检查跨境传输
        cross_border_transfer = config.get('cross_border_transfer', {})
        if cross_border_transfer.get('enabled', False):
            if not cross_border_transfer.get('adequacy_decision', False) and \
               not cross_border_transfer.get('standard_contractual_clauses', False):
                risks.append(RiskItem(
                    category='data_sharing',
                    description='跨境传输缺乏适当保护措施',
                    severity=RiskLevel.CRITICAL,
                    likelihood='high',
                    impact='severe',
                    mitigation_required=True,
                    recommendations=[
                        '获得充分性认定',
                        '采用标准合同条款',
                        '实施补充保护措施'
                    ]
                ))
        
        return risks
    
    def assess_user_rights_risks(self, config: Dict, context: Dict) -> List[RiskItem]:
        """评估用户权利保护风险"""
        risks = []
        
        user_rights = config.get('user_rights', {})
        
        # 检查访问权
        if not user_rights.get('data_access', {}).get('enabled', False):
            risks.append(RiskItem(
                category='user_rights',
                description='未实现数据访问权',
                severity=RiskLevel.MEDIUM,
                likelihood='high',
                impact='moderate',
                mitigation_required=True,
                recommendations=[
                    '实现数据访问权接口',
                    '提供易用的访问机制',
                    '确保访问数据完整性'
                ]
            ))
        
        # 检查删除权
        if not user_rights.get('data_deletion', {}).get('enabled', False):
            risks.append(RiskItem(
                category='user_rights',
                description='未实现数据删除权',
                severity=RiskLevel.HIGH,
                likelihood='high',
                impact='major',
                mitigation_required=True,
                recommendations=[
                    '实现数据删除功能',
                    '确保删除的彻底性',
                    '处理删除例外情况'
                ]
            ))
        
        return risks
    
    def assess_regulatory_compliance(self, config: Dict, context: Dict) -> List[ComplianceItem]:
        """评估法规合规性"""
        compliance_items = []
        
        # 确定适用的法规
        applicable_regulations = self.determine_applicable_regulations(context)
        
        for regulation in applicable_regulations:
            reg_requirements = self.regulation_requirements[regulation]
            
            for req_name, requirement in reg_requirements['key_requirements'].items():
                # 评估每个要求的合规状态
                compliance_status = self.evaluate_compliance_status(
                    config, requirement, regulation, req_name
                )
                
                compliance_items.append(ComplianceItem(
                    regulation=regulation,
                    requirement=requirement['description'],
                    status=compliance_status['status'],
                    evidence=compliance_status['evidence'],
                    gaps=compliance_status['gaps'],
                    actions_required=compliance_status['actions']
                ))
        
        return compliance_items
    
    def determine_applicable_regulations(self, context: Dict) -> List[str]:
        """确定适用的法规"""
        applicable = []
        
        business_regions = set(context.get('business_regions', []))
        data_subject_regions = set(context.get('data_subject_regions', []))
        
        for regulation, reg_info in self.regulation_requirements.items():
            reg_regions = set(reg_info['region'])
            
            # 检查地理适用性
            if business_regions.intersection(reg_regions) or data_subject_regions.intersection(reg_regions):
                applicable.append(regulation)
        
        return applicable
    
    def evaluate_compliance_status(self, config: Dict, requirement: Dict, 
                                 regulation: str, req_name: str) -> Dict:
        """评估合规状态"""
        # 这里是简化的合规性评估逻辑
        # 实际应用中需要更复杂的规则引擎
        
        checks = requirement['compliance_checks']
        passed_checks = 0
        gaps = []
        actions = []
        evidence_items = []
        
        for check in checks:
            # 模拟检查逻辑
            if self.simulate_compliance_check(config, check, regulation):
                passed_checks += 1
                evidence_items.append(f"通过检查: {check}")
            else:
                gaps.append(check)
                actions.append(f"需要实施: {check}")
        
        # 计算合规状态
        compliance_ratio = passed_checks / len(checks)
        if compliance_ratio >= 0.8:
            status = ComplianceStatus.COMPLIANT
        elif compliance_ratio >= 0.5:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        return {
            'status': status,
            'evidence': '; '.join(evidence_items),
            'gaps': gaps,
            'actions': actions
        }
    
    def simulate_compliance_check(self, config: Dict, check: str, regulation: str) -> bool:
        """模拟合规性检查"""
        # 这是简化的检查逻辑，实际应用中需要更详细的实现
        check_lower = check.lower()
        
        if '同意' in check or 'consent' in check_lower:
            return config.get('consent_mechanism', {}).get('explicit_consent', False)
        elif '加密' in check or 'encryption' in check_lower:
            return config.get('encryption', {}).get('at_rest', False)
        elif '访问权' in check or 'access right' in check_lower:
            return config.get('user_rights', {}).get('data_access', {}).get('enabled', False)
        elif '删除' in check or 'deletion' in check_lower:
            return config.get('user_rights', {}).get('data_deletion', {}).get('enabled', False)
        elif '通知' in check or 'notification' in check_lower:
            return config.get('breach_response', {}).get('notification_enabled', False)
        else:
            # 默认50%的检查通过概率
            return np.random.random() > 0.5
    
    def calculate_overall_privacy_score(self, risks: List[RiskItem], 
                                      compliance_items: List[ComplianceItem]) -> float:
        """计算整体隐私评分"""
        # 风险评分（0-50分）
        risk_score = self.calculate_risk_score(risks)
        
        # 合规评分（0-50分）
        compliance_score = self.calculate_compliance_score(compliance_items)
        
        # 总分（0-100分）
        total_score = risk_score + compliance_score
        
        return min(100.0, max(0.0, total_score))
    
    def calculate_risk_score(self, risks: List[RiskItem]) -> float:
        """计算风险评分"""
        if not risks:
            return 50.0  # 无风险的满分
        
        # 风险权重
        risk_weights = {
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 4,
            RiskLevel.CRITICAL: 8
        }
        
        # 计算加权风险分数
        total_weighted_risk = sum(risk_weights[risk.severity] for risk in risks)
        max_possible_risk = len(risks) * risk_weights[RiskLevel.CRITICAL]
        
        # 转换为0-50分的评分（风险越高，分数越低）
        risk_ratio = total_weighted_risk / max_possible_risk if max_possible_risk > 0 else 0
        risk_score = 50.0 * (1 - risk_ratio)
        
        return max(0.0, risk_score)
    
    def calculate_compliance_score(self, compliance_items: List[ComplianceItem]) -> float:
        """计算合规评分"""
        if not compliance_items:
            return 50.0  # 无合规要求的满分
        
        # 合规状态权重
        compliance_weights = {
            ComplianceStatus.COMPLIANT: 1.0,
            ComplianceStatus.PARTIALLY_COMPLIANT: 0.6,
            ComplianceStatus.NON_COMPLIANT: 0.0,
            ComplianceStatus.UNKNOWN: 0.3
        }
        
        # 计算平均合规分数
        total_compliance = sum(compliance_weights[item.status] for item in compliance_items)
        average_compliance = total_compliance / len(compliance_items)
        
        # 转换为0-50分的评分
        compliance_score = 50.0 * average_compliance
        
        return compliance_score
    
    def generate_comprehensive_recommendations(self, risks: List[RiskItem], 
                                             compliance_items: List[ComplianceItem]) -> List[str]:
        """生成综合建议"""
        recommendations = []
        
        # 基于风险的建议
        critical_risks = [r for r in risks if r.severity == RiskLevel.CRITICAL]
        high_risks = [r for r in risks if r.severity == RiskLevel.HIGH]
        
        if critical_risks:
            recommendations.append("紧急处理关键隐私风险，立即实施相应的缓解措施")
            for risk in critical_risks:
                recommendations.extend(risk.recommendations)
        
        if high_risks:
            recommendations.append("优先处理高风险项目，制定详细的风险缓解计划")
        
        # 基于合规性的建议
        non_compliant_items = [c for c in compliance_items if c.status == ComplianceStatus.NON_COMPLIANT]
        if non_compliant_items:
            recommendations.append("立即解决不合规项目，避免监管处罚风险")
        
        # 通用建议
        recommendations.extend([
            "建立定期隐私影响评估流程",
            "加强员工隐私保护培训",
            "建立隐私事件响应流程",
            "定期审查和更新隐私政策"
        ])
        
        # 去重并返回
        return list(set(recommendations))
    
    def calculate_next_review_date(self, privacy_score: float) -> str:
        """计算下次审查日期"""
        from datetime import timedelta
        
        # 根据隐私评分确定审查频率
        if privacy_score >= 80:
            months = 12  # 年度审查
        elif privacy_score >= 60:
            months = 6   # 半年度审查
        else:
            months = 3   # 季度审查
        
        next_date = datetime.now() + timedelta(days=30 * months)
        return next_date.isoformat()
    
    def export_assessment_report(self, report: AssessmentReport, 
                               format: str = 'json') -> str:
        """导出评估报告"""
        if format.lower() == 'json':
            # 使用自定义序列化处理枚举类型
            return json.dumps(asdict(report), indent=2, ensure_ascii=False, default=str)
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    def generate_executive_summary(self, report: AssessmentReport) -> str:
        """生成执行摘要"""
        critical_risks = len([r for r in report.privacy_risks if r.severity == RiskLevel.CRITICAL])
        high_risks = len([r for r in report.privacy_risks if r.severity == RiskLevel.HIGH])
        non_compliant = len([c for c in report.compliance_items if c.status == ComplianceStatus.NON_COMPLIANT])
        
        summary = f"""
隐私影响评估执行摘要

系统名称: {report.system_name}
评估日期: {report.assessment_date}
整体隐私评分: {report.overall_privacy_score:.1f}/100

风险概览:
- 关键风险: {critical_risks} 项
- 高风险: {high_risks} 项
- 总风险项: {len(report.privacy_risks)} 项

合规状况:
- 不合规项目: {non_compliant} 项
- 总合规检查: {len(report.compliance_items)} 项

主要建议:
{chr(10).join(f"- {rec}" for rec in report.recommendations[:5])}

下次审查日期: {report.next_review_date}
        """
        
        return summary.strip()


def demo_privacy_assessment():
    """隐私影响评估演示"""
    print("=== 隐私影响评估演示 ===")
    
    # 创建评估系统
    pia_system = PrivacyImpactAssessment()
    
    # 模拟RAG系统配置
    rag_config = {
        'data_sources': [
            {'name': 'user_documents', 'sensitivity': 'medium'},
            {'name': 'customer_data', 'sensitivity': 'high'},
            {'name': 'financial_records', 'sensitivity': 'critical'}
        ],
        'encryption': {
            'at_rest': True,
            'in_transit': True,
            'algorithm': 'AES-256'
        },
        'access_control': {
            'role_based': True,
            'multi_factor_auth': True
        },
        'consent_mechanism': {
            'explicit_consent': True,
            'granular_consent': False
        },
        'user_rights': {
            'data_access': {'enabled': True},
            'data_deletion': {'enabled': False},
            'data_portability': {'enabled': False}
        },
        'third_party_sharing': {
            'enabled': True,
            'data_processing_agreement': False
        },
        'cross_border_transfer': {
            'enabled': True,
            'adequacy_decision': False,
            'standard_contractual_clauses': True
        },
        'automated_decision_making': {
            'enabled': True,
            'human_review': False
        },
        'backup': {
            'enabled': True,
            'encrypted': True
        },
        'breach_response': {
            'notification_enabled': True,
            'response_plan': True
        }
    }
    
    # 业务上下文
    business_context = {
        'business_regions': ['EU', 'US', 'China'],
        'data_subject_regions': ['EU', 'US', 'China'],
        'industry': 'financial_services',
        'data_volume': 'large',
        'user_base': 'international'
    }
    
    # 执行综合评估
    print("开始执行隐私影响评估...")
    assessment_report = pia_system.conduct_comprehensive_assessment(
        system_name="企业级RAG智能问答系统",
        rag_system_config=rag_config,
        business_context=business_context
    )
    
    # 显示执行摘要
    print("\n" + "="*50)
    print(pia_system.generate_executive_summary(assessment_report))
    print("="*50)
    
    # 详细风险分析
    print(f"\n详细风险分析:")
    risk_by_severity = {}
    for risk in assessment_report.privacy_risks:
        severity = risk.severity.value
        if severity not in risk_by_severity:
            risk_by_severity[severity] = []
        risk_by_severity[severity].append(risk)
    
    for severity in ['critical', 'high', 'medium', 'low']:
        if severity in risk_by_severity:
            print(f"\n{severity.upper()}风险 ({len(risk_by_severity[severity])}项):")
            for risk in risk_by_severity[severity]:
                print(f"  - {risk.description}")
                if risk.recommendations:
                    print(f"    建议: {risk.recommendations[0]}")
    
    # 合规状况分析
    print(f"\n合规状况分析:")
    compliance_by_regulation = {}
    for item in assessment_report.compliance_items:
        if item.regulation not in compliance_by_regulation:
            compliance_by_regulation[item.regulation] = []
        compliance_by_regulation[item.regulation].append(item)
    
    for regulation, items in compliance_by_regulation.items():
        compliant = len([i for i in items if i.status == ComplianceStatus.COMPLIANT])
        total = len(items)
        print(f"  {regulation}: {compliant}/{total} 项合规 ({compliant/total*100:.1f}%)")
        
        non_compliant = [i for i in items if i.status == ComplianceStatus.NON_COMPLIANT]
        if non_compliant:
            print(f"    不合规项目:")
            for item in non_compliant[:3]:  # 显示前3个不合规项目
                print(f"    - {item.requirement}")
    
    # 主要建议
    print(f"\n主要建议:")
    for i, recommendation in enumerate(assessment_report.recommendations[:8], 1):
        print(f"  {i}. {recommendation}")
    
    # 导出报告
    print(f"\n导出评估报告...")
    report_json = pia_system.export_assessment_report(assessment_report)
    
    # 保存到文件（可选）
    report_filename = f"privacy_assessment_{assessment_report.assessment_id}.json"
    try:
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report_json)
        print(f"评估报告已保存至: {report_filename}")
    except Exception as e:
        print(f"保存报告失败: {str(e)}")
    
    print(f"\n评估完成！")
    print(f"- 评估ID: {assessment_report.assessment_id}")
    print(f"- 整体评分: {assessment_report.overall_privacy_score:.1f}/100")
    print(f"- 下次审查: {assessment_report.next_review_date}")


if __name__ == "__main__":
    demo_privacy_assessment()