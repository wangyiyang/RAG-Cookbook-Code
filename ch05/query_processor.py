"""
电商智能客服RAG系统 - 查询处理模块
实现查询意图识别和个性化检索
"""

import re
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class QueryIntent(Enum):
    """查询意图枚举"""
    PRICE_INQUIRY = "price_inquiry"
    PRODUCT_INFO = "product_info"
    ORDER_STATUS = "order_status"
    SHIPPING_INFO = "shipping_info"
    RETURN_REFUND = "return_refund"
    TECHNICAL_SUPPORT = "technical_support"
    GENERAL_INQUIRY = "general_inquiry"


class UrgencyLevel(Enum):
    """紧急程度枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QueryAnalysis:
    """查询分析结果"""
    intent: QueryIntent
    entities: Dict[str, Any]
    context_signals: Dict[str, Any]
    urgency: UrgencyLevel
    confidence: float


class QueryProcessor:
    """查询处理器"""
    
    def __init__(self):
        self.intent_patterns = {
            QueryIntent.PRICE_INQUIRY: [
                r'.*价格.*', r'.*多少钱.*', r'.*贵不贵.*', r'.*便宜.*', r'.*优惠.*'
            ],
            QueryIntent.PRODUCT_INFO: [
                r'.*怎么样.*', r'.*质量.*', r'.*规格.*', r'.*参数.*', r'.*介绍.*'
            ],
            QueryIntent.ORDER_STATUS: [
                r'.*订单.*', r'.*状态.*', r'.*发货.*', r'.*处理.*'
            ],
            QueryIntent.SHIPPING_INFO: [
                r'.*配送.*', r'.*物流.*', r'.*快递.*', r'.*运费.*'
            ],
            QueryIntent.RETURN_REFUND: [
                r'.*退货.*', r'.*退款.*', r'.*换货.*', r'.*售后.*'
            ],
            QueryIntent.TECHNICAL_SUPPORT: [
                r'.*故障.*', r'.*问题.*', r'.*修理.*', r'.*维修.*'
            ]
        }
        
        self.urgency_keywords = {
            UrgencyLevel.CRITICAL: ['紧急', '急', '故障', '坏了', '问题'],
            UrgencyLevel.HIGH: ['尽快', '马上', '立即', '重要'],
            UrgencyLevel.MEDIUM: ['希望', '需要', '想要'],
            UrgencyLevel.LOW: ['了解', '咨询', '查询']
        }
    
    def analyze_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> QueryAnalysis:
        """分析查询"""
        # 1. 意图识别
        intent = self.identify_intent(query)
        
        # 2. 实体提取
        entities = self.extract_entities(query)
        
        # 3. 上下文信号分析
        context_signals = self.analyze_context_signals(user_context or {})
        
        # 4. 紧急程度评估
        urgency = self.assess_urgency(query, entities)
        
        # 5. 计算置信度
        confidence = self.calculate_confidence(query, intent, entities)
        
        return QueryAnalysis(
            intent=intent,
            entities=entities,
            context_signals=context_signals,
            urgency=urgency,
            confidence=confidence
        )
    
    def identify_intent(self, query: str) -> QueryIntent:
        """识别查询意图"""
        query_lower = query.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        return QueryIntent.GENERAL_INQUIRY
    
    def extract_entities(self, query: str) -> Dict[str, Any]:
        """提取实体信息"""
        entities = {}
        
        # 提取产品名称
        product_patterns = [
            r'(iPhone\s*\d+[^，。]*)',
            r'(MacBook[^，。]*)',
            r'(iPad[^，。]*)',
            r'(\w+手机)',
            r'(\w+电脑)'
        ]
        
        for pattern in product_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entities['product'] = match.group(1)
                break
        
        # 提取价格相关
        price_match = re.search(r'(\d+)\s*元', query)
        if price_match:
            entities['price'] = int(price_match.group(1))
        
        # 提取订单号
        order_match = re.search(r'订单号?\s*[:：]?\s*(\w+)', query)
        if order_match:
            entities['order_id'] = order_match.group(1)
        
        return entities
    
    def analyze_context_signals(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """分析用户上下文信号"""
        signals = {}
        
        # VIP等级信号
        vip_level = user_context.get('vip_level', 0)
        if vip_level >= 3:
            signals['high_value_customer'] = True
        
        # 购买历史信号
        purchase_history = user_context.get('purchase_history', [])
        if purchase_history:
            signals['returning_customer'] = True
            signals['purchase_frequency'] = len(purchase_history)
        
        # 投诉历史信号
        complaint_history = user_context.get('complaint_history', [])
        if complaint_history:
            signals['has_complaints'] = True
            signals['complaint_count'] = len(complaint_history)
        
        return signals
    
    def assess_urgency(self, query: str, entities: Dict[str, Any]) -> UrgencyLevel:
        """评估紧急程度"""
        query_lower = query.lower()
        
        # 检查紧急关键词
        for urgency, keywords in self.urgency_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return urgency
        
        # 根据查询类型判断
        if '故障' in query_lower or '坏了' in query_lower:
            return UrgencyLevel.CRITICAL
        elif '退款' in query_lower or '投诉' in query_lower:
            return UrgencyLevel.HIGH
        elif '咨询' in query_lower or '了解' in query_lower:
            return UrgencyLevel.LOW
        
        return UrgencyLevel.MEDIUM
    
    def calculate_confidence(self, query: str, intent: QueryIntent, entities: Dict[str, Any]) -> float:
        """计算置信度"""
        confidence = 0.5  # 基础置信度
        
        # 根据关键词匹配程度调整
        if intent != QueryIntent.GENERAL_INQUIRY:
            confidence += 0.3
        
        # 根据实体提取情况调整
        if entities:
            confidence += 0.2
        
        # 根据查询长度调整
        if len(query) > 10:
            confidence += 0.1
        
        return min(confidence, 1.0)


class PersonalizedRetriever:
    """个性化检索器"""
    
    def __init__(self, vector_store, product_db):
        self.vector_store = vector_store
        self.product_db = product_db
    
    def retrieve_personalized(self, query: str, user_context: Dict[str, Any], 
                            query_analysis: QueryAnalysis) -> List[Dict[str, Any]]:
        """执行个性化检索"""
        # 1. 基础检索
        base_results = self.vector_store.similarity_search(query, k=20)
        
        # 2. 用户偏好过滤
        if user_context.get('purchase_history'):
            preference_filter = self.build_preference_filter(
                user_context['purchase_history']
            )
            base_results = self.apply_preference_filter(base_results, preference_filter)
        
        # 3. VIP用户特殊处理
        if user_context.get('vip_level', 0) >= 3:
            base_results = self.enhance_for_vip(base_results, query)
        
        # 4. 意图权重调整
        weighted_results = self.apply_intent_weighting(
            base_results, query_analysis.intent
        )
        
        return weighted_results[:10]
    
    def build_preference_filter(self, purchase_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """构建用户偏好过滤器"""
        brand_preferences = {}
        category_preferences = {}
        
        for order in purchase_history:
            for item in order.get('items', []):
                brand = item.get('brand')
                category = item.get('category')
                
                if brand:
                    brand_preferences[brand] = brand_preferences.get(brand, 0) + 1
                if category:
                    category_preferences[category] = category_preferences.get(category, 0) + 1
        
        return {
            'preferred_brands': sorted(brand_preferences.items(), 
                                     key=lambda x: x[1], reverse=True)[:5],
            'preferred_categories': sorted(category_preferences.items(), 
                                         key=lambda x: x[1], reverse=True)[:3]
        }
    
    def apply_preference_filter(self, results: List[Dict[str, Any]], 
                              preference_filter: Dict[str, Any]) -> List[Dict[str, Any]]:
        """应用偏好过滤器"""
        filtered_results = []
        
        preferred_brands = [brand for brand, _ in preference_filter['preferred_brands']]
        preferred_categories = [cat for cat, _ in preference_filter['preferred_categories']]
        
        for result in results:
            metadata = result.get('metadata', {})
            brand = metadata.get('brand')
            category = metadata.get('category')
            
            # 提升偏好品牌和类别的权重
            if brand in preferred_brands:
                result['score'] = result.get('score', 0) * 1.2
            if category in preferred_categories:
                result['score'] = result.get('score', 0) * 1.1
                
            filtered_results.append(result)
        
        return sorted(filtered_results, key=lambda x: x.get('score', 0), reverse=True)
    
    def enhance_for_vip(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """VIP用户增强处理"""
        # 为VIP用户添加专属信息
        vip_enhanced = []
        
        for result in results:
            enhanced_result = result.copy()
            
            # 添加VIP专属内容
            if 'VIP' not in enhanced_result.get('content', ''):
                enhanced_result['content'] += '\n\n作为VIP用户，您还可以享受专属优惠和服务。'
            
            # 提升VIP相关内容的权重
            if 'VIP' in enhanced_result.get('content', ''):
                enhanced_result['score'] = enhanced_result.get('score', 0) * 1.3
            
            vip_enhanced.append(enhanced_result)
        
        return vip_enhanced
    
    def apply_intent_weighting(self, results: List[Dict[str, Any]], 
                              intent: QueryIntent) -> List[Dict[str, Any]]:
        """应用意图权重"""
        # 根据意图类型调整权重
        intent_weights = {
            QueryIntent.PRICE_INQUIRY: {'price': 1.5, 'promotion': 1.3},
            QueryIntent.PRODUCT_INFO: {'technical': 1.4, 'basic_info': 1.2},
            QueryIntent.SHIPPING_INFO: {'logistics': 1.5, 'service': 1.2},
            QueryIntent.RETURN_REFUND: {'service': 1.4, 'basic_info': 1.1}
        }
        
        weights = intent_weights.get(intent, {})
        
        for result in results:
            chunk_type = result.get('metadata', {}).get('chunk_type', '')
            if chunk_type in weights:
                result['score'] = result.get('score', 0) * weights[chunk_type]
        
        return sorted(results, key=lambda x: x.get('score', 0), reverse=True)


# 使用示例
if __name__ == "__main__":
    processor = QueryProcessor()
    
    # 查询分析示例
    query = "iPhone 15 Pro价格贵不贵？"
    user_context = {
        'vip_level': 3,
        'purchase_history': [
            {
                'items': [
                    {'brand': 'Apple', 'category': 'smartphone'}
                ]
            }
        ]
    }
    
    analysis = processor.analyze_query(query, user_context)
    print(f"意图: {analysis.intent}")
    print(f"实体: {analysis.entities}")
    print(f"紧急程度: {analysis.urgency}")
    print(f"置信度: {analysis.confidence}")
