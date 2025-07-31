"""
数据脱敏与匿名化实现
Deep RAG Notes Chapter 12 - Privacy Protection Technologies
"""

import re
import hashlib
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class MaskingResult:
    """脱敏结果"""
    original_text: str
    masked_text: str
    masking_operations: List[Dict[str, Any]]
    sensitivity_analysis: Dict[str, Any]

class IntelligentDataMasking:
    """智能数据脱敏系统"""
    
    def __init__(self):
        """初始化数据脱敏系统"""
        self.sensitive_patterns = self.load_sensitive_patterns()
        self.masking_strategies = self.load_masking_strategies()
        self.anonymization_cache = {}
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_sensitive_patterns(self) -> Dict[str, Dict[str, Any]]:
        """加载敏感信息识别模式"""
        return {
            'personal_id': {
                'patterns': [
                    r'\b\d{17}[\dX]\b',  # 中国身份证号
                    r'\b\d{3}-?\d{2}-?\d{4}\b',  # 美国社会安全号
                    r'\b\d{9}\b',  # 简化身份证号
                ],
                'sensitivity_level': 'critical',
                'description': '身份证号、社会安全号等个人身份标识'
            },
            'phone_number': {
                'patterns': [
                    r'\b1[3-9]\d{9}\b',  # 中国手机号
                    r'\b\(\d{3}\)\s?\d{3}-?\d{4}\b',  # 美国电话格式
                    r'\b\d{3}-\d{3}-\d{4}\b',  # 标准电话格式
                ],
                'sensitivity_level': 'high',
                'description': '电话号码'
            },
            'email_address': {
                'patterns': [
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                ],
                'sensitivity_level': 'medium',
                'description': '电子邮箱地址'
            },
            'financial_account': {
                'patterns': [
                    r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # 信用卡号
                    r'\b\d{10,20}\b'  # 银行账号（简化）
                ],
                'sensitivity_level': 'critical',
                'description': '银行卡号、信用卡号等金融账户信息'
            },
            'ip_address': {
                'patterns': [
                    r'\b(?:\d{1,3}\.){3}\d{1,3}\b'  # IPv4地址
                ],
                'sensitivity_level': 'medium',
                'description': 'IP地址'
            },
            'chinese_name': {
                'patterns': [
                    r'[\u4e00-\u9fa5]{2,4}(?=\s|，|。|：|；)',  # 中文姓名（简化匹配）
                ],
                'sensitivity_level': 'high',
                'description': '中文姓名'
            }
        }
    
    def load_masking_strategies(self) -> Dict[str, Any]:
        """加载脱敏策略"""
        return {
            'hash_anonymization': {
                'description': '哈希匿名化，保持唯一性但不可逆',
                'reversible': False,
                'security_level': 'high'
            },
            'partial_masking': {
                'description': '部分遮蔽，保留部分信息便于理解',
                'reversible': False,
                'security_level': 'medium'
            },
            'domain_generalization': {
                'description': '域泛化，替换为更通用的类别',
                'reversible': False,
                'security_level': 'medium'
            },
            'tokenization': {
                'description': '令牌化，生成可逆的令牌',
                'reversible': True,
                'security_level': 'high'
            },
            'complete_removal': {
                'description': '完全移除敏感信息',
                'reversible': False,
                'security_level': 'very_high'
            }
        }
    
    def intelligent_masking(self, text: str, context: Optional[Dict] = None) -> MaskingResult:
        """智能数据脱敏"""
        if not text:
            return MaskingResult(text, text, [], {})
        
        masked_text = text
        masking_operations = []
        sensitivity_analysis = {}
        
        # 分析上下文
        context_info = self.analyze_context(text, context or {})
        
        # 检测并处理每种敏感信息类型
        for info_type, pattern_info in self.sensitive_patterns.items():
            matches = self.detect_sensitive_info(text, pattern_info['patterns'])
            
            if matches:
                # 根据上下文调整敏感度
                adjusted_sensitivity = self.adjust_sensitivity_by_context(
                    pattern_info['sensitivity_level'],
                    context_info,
                    info_type
                )
                
                # 选择脱敏方法
                masking_method = self.select_masking_method(
                    adjusted_sensitivity,
                    info_type,
                    context_info
                )
                
                # 执行脱敏
                for match in matches:
                    masked_value = self.apply_masking_method(
                        match['value'],
                        masking_method,
                        info_type
                    )
                    
                    # 替换文本中的敏感信息
                    masked_text = masked_text.replace(match['value'], masked_value)
                    
                    # 记录脱敏操作
                    masking_operations.append({
                        'original_value': match['value'],
                        'masked_value': masked_value,
                        'method': masking_method,
                        'position': match['position'],
                        'info_type': info_type,
                        'sensitivity_level': adjusted_sensitivity
                    })
                
                # 记录敏感性分析
                sensitivity_analysis[info_type] = {
                    'count': len(matches),
                    'original_sensitivity': pattern_info['sensitivity_level'],
                    'adjusted_sensitivity': adjusted_sensitivity,
                    'method_used': masking_method,
                    'description': pattern_info['description']
                }
        
        self.logger.info(f"文本脱敏完成，处理了 {len(masking_operations)} 个敏感信息")
        
        return MaskingResult(
            original_text=text,
            masked_text=masked_text,
            masking_operations=masking_operations,
            sensitivity_analysis=sensitivity_analysis
        )
    
    def detect_sensitive_info(self, text: str, patterns: List[str]) -> List[Dict[str, Any]]:
        """检测敏感信息"""
        matches = []
        
        for pattern in patterns:
            try:
                for match in re.finditer(pattern, text):
                    matches.append({
                        'value': match.group(),
                        'position': {
                            'start': match.start(),
                            'end': match.end()
                        },
                        'pattern': pattern
                    })
            except re.error as e:
                self.logger.warning(f"正则表达式错误: {pattern}, {str(e)}")
                continue
        
        return matches
    
    def analyze_context(self, text: str, context: Dict) -> Dict[str, Any]:
        """分析文本上下文"""
        context_info = {
            'text_length': len(text),
            'document_type': context.get('document_type', 'unknown'),
            'domain': context.get('domain', 'general'),
            'sensitivity_level': context.get('sensitivity_level', 'medium'),
            'is_public': context.get('is_public', False),
            'has_business_context': any(word in text.lower() for word in ['公司', '企业', '业务', 'company', 'business']),
            'has_personal_context': any(word in text.lower() for word in ['个人', '私人', 'personal', 'private']),
            'contains_financial_terms': any(word in text.lower() for word in ['银行', '账户', '支付', 'bank', 'account', 'payment'])
        }
        
        return context_info
    
    def adjust_sensitivity_by_context(self, 
                                     base_sensitivity: str, 
                                     context_info: Dict, 
                                     info_type: str) -> str:
        """根据上下文调整敏感度"""
        sensitivity_levels = ['low', 'medium', 'high', 'critical']
        base_level = sensitivity_levels.index(base_sensitivity) if base_sensitivity in sensitivity_levels else 1
        
        # 上下文调整因子
        adjustment = 0
        
        # 公开文档降低敏感度
        if context_info.get('is_public', False):
            adjustment -= 1
        
        # 金融领域提高敏感度
        if context_info.get('domain') == 'financial' or context_info.get('contains_financial_terms'):
            adjustment += 1
        
        # 个人上下文提高敏感度
        if context_info.get('has_personal_context'):
            adjustment += 1
        
        # 业务上下文可能降低某些敏感度
        if context_info.get('has_business_context') and info_type == 'phone_number':
            adjustment -= 1
        
        # 计算最终敏感度级别
        final_level = max(0, min(len(sensitivity_levels) - 1, base_level + adjustment))
        return sensitivity_levels[final_level]
    
    def select_masking_method(self, 
                             sensitivity_level: str, 
                             info_type: str, 
                             context_info: Dict) -> str:
        """选择脱敏方法"""
        # 基于敏感度的默认方法映射
        sensitivity_methods = {
            'low': 'partial_masking',
            'medium': 'domain_generalization',
            'high': 'hash_anonymization',
            'critical': 'tokenization'
        }
        
        base_method = sensitivity_methods.get(sensitivity_level, 'partial_masking')
        
        # 针对特定信息类型的优化
        if info_type == 'email_address':
            return 'domain_generalization'
        elif info_type == 'phone_number' and context_info.get('has_business_context'):
            return 'partial_masking'
        elif info_type == 'personal_id':
            return 'hash_anonymization'  # 身份证号总是使用哈希
        elif info_type == 'financial_account':
            return 'tokenization'  # 金融账户使用令牌化
        
        return base_method
    
    def apply_masking_method(self, value: str, method: str, info_type: str) -> str:
        """应用脱敏方法"""
        try:
            if method == 'hash_anonymization':
                return self.hash_anonymization(value)
            elif method == 'partial_masking':
                return self.partial_masking(value, info_type)
            elif method == 'domain_generalization':
                return self.domain_generalization(value, info_type)
            elif method == 'tokenization':
                return self.tokenization(value)
            elif method == 'complete_removal':
                return '[REMOVED]'
            else:
                return self.partial_masking(value, info_type)  # 默认方法
        except Exception as e:
            self.logger.error(f"脱敏方法应用失败: {method}, {str(e)}")
            return '[REDACTED]'
    
    def hash_anonymization(self, value: str) -> str:
        """哈希匿名化"""
        hash_value = hashlib.sha256(value.encode('utf-8')).hexdigest()[:8]
        return f"[HASH_{hash_value}]"
    
    def partial_masking(self, value: str, info_type: str) -> str:
        """部分遮蔽"""
        if not value:
            return value
        
        length = len(value)
        
        if info_type == 'phone_number':
            # 电话号码：保留前3位和后4位
            if length >= 7:
                return value[:3] + '*' * (length - 7) + value[-4:]
            else:
                return '*' * length
        elif info_type == 'email_address':
            # 邮箱：保留用户名前2位和域名
            if '@' in value:
                username, domain = value.split('@', 1)
                if len(username) >= 2:
                    masked_username = username[:2] + '*' * (len(username) - 2)
                else:
                    masked_username = '*' * len(username)
                return f"{masked_username}@{domain}"
            else:
                return '*' * length
        elif info_type == 'personal_id':
            # 身份证：保留前6位和后4位
            if length >= 10:
                return value[:6] + '*' * (length - 10) + value[-4:]
            else:
                return '*' * length
        elif info_type == 'financial_account':
            # 银行卡：只保留后4位
            if length >= 4:
                return '*' * (length - 4) + value[-4:]
            else:
                return '*' * length
        else:
            # 通用：保留前2位和后2位
            if length <= 4:
                return '*' * length
            else:
                return value[:2] + '*' * (length - 4) + value[-2:]
    
    def domain_generalization(self, value: str, info_type: str) -> str:
        """域泛化"""
        if info_type == 'email_address' and '@' in value:
            username, domain = value.split('@', 1)
            return f"user@{domain}"
        elif info_type == 'ip_address':
            # IP地址泛化到网段
            parts = value.split('.')
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.xxx.xxx"
        elif info_type == 'phone_number':
            # 电话号码泛化到地区码
            if len(value) >= 7:
                return value[:3] + "xxxxxxx"
        
        return f"[{info_type.upper()}]"
    
    def tokenization(self, value: str) -> str:
        """令牌化（可逆）"""
        # 生成确定性令牌
        token = hashlib.md5(value.encode('utf-8')).hexdigest()[:8]
        
        # 存储到可逆映射中（实际应用中应该使用安全的存储）
        self.anonymization_cache[token] = value
        
        return f"[TOKEN_{token}]"
    
    def reverse_tokenization(self, token: str) -> Optional[str]:
        """反向令牌化"""
        if token.startswith('[TOKEN_') and token.endswith(']'):
            token_id = token[7:-1]  # 提取TOKEN_和]之间的内容
            return self.anonymization_cache.get(token_id)
        return None
    
    def get_masking_statistics(self, results: List[MaskingResult]) -> Dict[str, Any]:
        """获取脱敏统计信息"""
        if not results:
            return {}
        
        total_operations = sum(len(result.masking_operations) for result in results)
        info_type_counts = {}
        method_counts = {}
        sensitivity_counts = {}
        
        for result in results:
            for operation in result.masking_operations:
                info_type = operation['info_type']
                method = operation['method']
                sensitivity = operation['sensitivity_level']
                
                info_type_counts[info_type] = info_type_counts.get(info_type, 0) + 1
                method_counts[method] = method_counts.get(method, 0) + 1
                sensitivity_counts[sensitivity] = sensitivity_counts.get(sensitivity, 0) + 1
        
        return {
            'total_documents': len(results),
            'total_masking_operations': total_operations,
            'average_operations_per_document': total_operations / len(results),
            'info_type_distribution': info_type_counts,
            'method_distribution': method_counts,
            'sensitivity_distribution': sensitivity_counts
        }


def demo_data_masking():
    """数据脱敏演示"""
    print("=== 智能数据脱敏演示 ===")
    
    # 创建脱敏系统
    masking_system = IntelligentDataMasking()
    
    # 测试文本
    test_texts = [
        "张三的身份证号是110101199001011234，电话是13812345678，邮箱是zhangsan@example.com",
        "请联系客户经理李四，手机号码：15987654321，工作邮箱：lisi@company.com，银行卡号：6222021234567890",
        "服务器IP地址：192.168.1.100，管理员联系方式：admin@server.com",
        "员工信息：王五，身份证：320101198505156789，联系电话：(021)1234-5678"
    ]
    
    # 不同的上下文场景
    contexts = [
        {'document_type': 'personal', 'domain': 'general', 'is_public': False},
        {'document_type': 'business', 'domain': 'financial', 'is_public': False},
        {'document_type': 'technical', 'domain': 'it', 'is_public': True},
        {'document_type': 'hr', 'domain': 'internal', 'is_public': False}
    ]
    
    results = []
    
    for i, (text, context) in enumerate(zip(test_texts, contexts), 1):
        print(f"\n--- 测试案例 {i} ---")
        print(f"原文: {text}")
        print(f"上下文: {context}")
        
        # 执行脱敏
        result = masking_system.intelligent_masking(text, context)
        results.append(result)
        
        print(f"脱敏后: {result.masked_text}")
        
        # 显示脱敏操作详情
        if result.masking_operations:
            print("脱敏操作:")
            for op in result.masking_operations:
                print(f"  - {op['info_type']}: {op['original_value']} -> {op['masked_value']} "
                      f"(方法: {op['method']}, 敏感度: {op['sensitivity_level']})")
        
        # 显示敏感性分析
        if result.sensitivity_analysis:
            print("敏感性分析:")
            for info_type, analysis in result.sensitivity_analysis.items():
                print(f"  - {info_type}: 发现{analysis['count']}个, "
                      f"敏感度: {analysis['original_sensitivity']} -> {analysis['adjusted_sensitivity']}")
    
    # 显示统计信息
    print(f"\n=== 脱敏统计 ===")
    stats = masking_system.get_masking_statistics(results)
    print(f"处理文档数: {stats['total_documents']}")
    print(f"总脱敏操作数: {stats['total_masking_operations']}")
    print(f"平均每文档操作数: {stats['average_operations_per_document']:.2f}")
    
    print(f"\n信息类型分布:")
    for info_type, count in stats['info_type_distribution'].items():
        print(f"  {info_type}: {count}")
    
    print(f"\n脱敏方法分布:")
    for method, count in stats['method_distribution'].items():
        print(f"  {method}: {count}")
    
    print(f"\n敏感度分布:")
    for sensitivity, count in stats['sensitivity_distribution'].items():
        print(f"  {sensitivity}: {count}")
    
    # 演示令牌化逆向
    print(f"\n=== 令牌化逆向演示 ===")
    for result in results:
        for op in result.masking_operations:
            if op['method'] == 'tokenization':
                original = masking_system.reverse_tokenization(op['masked_value'])
                print(f"令牌 {op['masked_value']} 对应原值: {original}")


if __name__ == "__main__":
    demo_data_masking()