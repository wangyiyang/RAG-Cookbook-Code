"""
电商智能客服RAG系统 - 提示模板构建器
实现个性化提示模板生成和答案生成优化
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class UserType(Enum):
    """用户类型枚举"""
    NEW_USER = "new_user"
    REGULAR_USER = "regular_user"
    VIP_USER = "vip_user"
    COMPLAINT_USER = "complaint_user"


@dataclass
class PromptTemplate:
    """提示模板结构"""
    template_id: str
    user_type: UserType
    query_type: str
    base_template: str
    special_instructions: List[str]


class PersonalizedPromptBuilder:
    """个性化提示模板构建器"""
    
    def __init__(self):
        self.template_library = self._load_template_library()
        self.user_context_builder = UserContextBuilder()
        
    def build_personalized_prompt(self, query: str, context: str, 
                                 user_profile: Dict[str, Any]) -> str:
        """构建个性化提示"""
        # 1. 分析用户类型
        user_type = self._classify_user_type(user_profile)
        
        # 2. 获取模板
        base_template = self._get_template_for_user_type(user_type)
        
        # 3. 构建用户上下文
        user_context = self.user_context_builder.build_user_context(user_profile)
        
        # 4. 获取特殊指令
        special_instructions = self._get_special_instructions(user_profile, user_type)
        
        # 5. 构建完整提示
        prompt = base_template.format(
            user_context=user_context,
            context=context,
            query=query,
            special_instructions=special_instructions
        )
        
        return prompt
    
    def _load_template_library(self) -> Dict[str, PromptTemplate]:
        """加载模板库"""
        templates = {}
        
        # 新用户模板
        templates['new_user_general'] = PromptTemplate(
            template_id='new_user_general',
            user_type=UserType.NEW_USER,
            query_type='general',
            base_template="""
            您好！欢迎来到我们的商城！

            用户信息：{user_context}
            
            针对您的问题：{query}
            
            基于以下信息：
            {context}
            
            {special_instructions}
            
            请提供详细、友好的回答，多介绍相关服务和优惠政策。
            """,
            special_instructions=[
                "新用户需要详细的操作指导",
                "主动介绍平台优势和服务",
                "语言要更加亲切友好"
            ]
        )
        
        # VIP用户模板
        templates['vip_user_general'] = PromptTemplate(
            template_id='vip_user_general',
            user_type=UserType.VIP_USER,
            query_type='general',
            base_template="""
            尊敬的VIP用户，您好！

            用户信息：{user_context}
            
            针对您的问题：{query}
            
            基于以下信息：
            {context}
            
            {special_instructions}
            
            请提供专业、高效的VIP级服务。
            """,
            special_instructions=[
                "强调VIP专属服务",
                "提供优先处理",
                "介绍VIP专属优惠"
            ]
        )
        
        # 投诉用户模板
        templates['complaint_user_general'] = PromptTemplate(
            template_id='complaint_user_general',
            user_type=UserType.COMPLAINT_USER,
            query_type='general',
            base_template="""
            您好，我们非常重视您的反馈。

            用户信息：{user_context}
            
            针对您的问题：{query}
            
            基于以下信息：
            {context}
            
            {special_instructions}
            
            请以解决问题为导向，耐心细致地回答。
            """,
            special_instructions=[
                "特别注意服务态度",
                "优先解决用户问题",
                "必要时提供补偿方案"
            ]
        )
        
        return templates
    
    def _classify_user_type(self, user_profile: Dict[str, Any]) -> UserType:
        """分类用户类型"""
        # VIP用户判断
        if user_profile.get('vip_level', 0) >= 3:
            return UserType.VIP_USER
        
        # 投诉用户判断
        if user_profile.get('complaint_history'):
            return UserType.COMPLAINT_USER
        
        # 新用户判断
        if not user_profile.get('purchase_history'):
            return UserType.NEW_USER
        
        return UserType.REGULAR_USER
    
    def _get_template_for_user_type(self, user_type: UserType) -> str:
        """获取用户类型对应的模板"""
        template_key = f"{user_type.value}_general"
        template = self.template_library.get(template_key)
        
        if template:
            return template.base_template
        
        # 默认模板
        return self.template_library['new_user_general'].base_template
    
    def select_template(self, user_type: str) -> str:
        """选择模板 - 文档中使用的方法名"""
        if isinstance(user_type, str):
            # 将字符串转换为UserType枚举
            user_type_enum = UserType(user_type.lower())
        else:
            user_type_enum = user_type
        return self._get_template_for_user_type(user_type_enum)
    
    def classify_user_type(self, user_profile: Dict[str, Any]) -> str:
        """分类用户类型 - 文档中使用的方法名"""
        user_type = self._classify_user_type(user_profile)
        return user_type.value.upper()
    
    def build_context(self, user_profile: Dict[str, Any]) -> str:
        """构建用户上下文 - 文档中使用的方法名"""
        return self.user_context_builder.build_user_context(user_profile)
    
    def generate_instructions(self, user_type: str) -> str:
        """生成特殊指令 - 文档中使用的方法名"""
        if isinstance(user_type, str):
            # 将字符串转换为UserType枚举
            user_type_enum = UserType(user_type.lower())
        else:
            user_type_enum = user_type
        
        # 创建一个虚拟的用户配置文件来生成指令
        dummy_profile = {}
        if user_type_enum == UserType.VIP_USER:
            dummy_profile['vip_level'] = 5
        elif user_type_enum == UserType.COMPLAINT_USER:
            dummy_profile['complaint_history'] = [{'issue': 'test'}]
        
        return self._get_special_instructions(dummy_profile, user_type_enum)
    
    def _get_special_instructions(self, user_profile: Dict[str, Any], 
                                 user_type: UserType) -> str:
        """获取特殊指令"""
        instructions = []
        
        # 根据用户类型添加指令
        if user_type == UserType.VIP_USER:
            instructions.extend([
                "请提供VIP专属服务信息",
                "主动介绍VIP优惠政策",
                "确保响应速度和服务质量"
            ])
        elif user_type == UserType.NEW_USER:
            instructions.extend([
                "提供详细的操作指导",
                "介绍平台的主要功能",
                "推荐新用户专享优惠"
            ])
        elif user_type == UserType.COMPLAINT_USER:
            instructions.extend([
                "特别注意服务态度，保持耐心",
                "优先解决用户的问题",
                "必要时提供合理的补偿建议"
            ])
        
        # 根据用户历史添加个性化指令
        if user_profile.get('frequent_categories'):
            categories = ', '.join(user_profile['frequent_categories'])
            instructions.append(f"用户常购买类别：{categories}，可重点推荐相关商品")
        
        return '\n'.join(instructions) if instructions else "请提供专业、友好的服务"


class UserContextBuilder:
    """用户上下文构建器"""
    
    def build_user_context(self, user_profile: Dict[str, Any]) -> str:
        """构建用户上下文信息"""
        context_parts = []
        
        # VIP等级信息
        vip_level = user_profile.get('vip_level', 0)
        if vip_level > 0:
            context_parts.append(f"VIP等级：{vip_level}级会员")
        
        # 购买历史
        purchase_history = user_profile.get('purchase_history', [])
        if purchase_history:
            recent_count = len(purchase_history)
            context_parts.append(f"历史订单：{recent_count}笔")
            
            # 最近购买
            if purchase_history:
                recent_order = purchase_history[-1]
                product_name = recent_order.get('items', [{}])[0].get('product_name', '未知商品')
                context_parts.append(f"最近购买：{product_name}")
        
        # 常购买类别
        frequent_categories = user_profile.get('frequent_categories', [])
        if frequent_categories:
            categories_str = ', '.join(frequent_categories[:3])
            context_parts.append(f"偏好类别：{categories_str}")
        
        # 问题历史
        support_history = user_profile.get('support_history', [])
        if support_history:
            last_issue = support_history[-1].get('issue_type', '一般咨询')
            context_parts.append(f"上次咨询：{last_issue}")
        
        # 投诉历史
        complaint_history = user_profile.get('complaint_history', [])
        if complaint_history:
            complaint_count = len(complaint_history)
            context_parts.append(f"投诉记录：{complaint_count}次")
        
        return '\n'.join(context_parts) if context_parts else "新用户"


class AnswerOptimizer:
    """答案优化器"""
    
    def __init__(self):
        self.optimization_rules = self._load_optimization_rules()
    
    def optimize_answer(self, answer: str, user_profile: Dict[str, Any], 
                       query_type: str) -> str:
        """优化答案"""
        optimized = answer
        
        # 1. 语言风格优化
        optimized = self._optimize_language_style(optimized, user_profile)
        
        # 2. 个性化内容添加
        optimized = self._add_personalized_content(optimized, user_profile)
        
        # 3. 格式优化
        optimized = self._optimize_format(optimized)
        
        return optimized
    
    def _optimize_language_style(self, answer: str, user_profile: Dict[str, Any]) -> str:
        """优化语言风格"""
        vip_level = user_profile.get('vip_level', 0)
        
        if vip_level >= 3:
            # VIP用户使用更正式的语言
            answer = answer.replace('你', '您')
            answer = answer.replace('亲', '尊敬的用户')
        
        return answer
    
    def _add_personalized_content(self, answer: str, user_profile: Dict[str, Any]) -> str:
        """添加个性化内容"""
        # 为VIP用户添加专属信息
        if user_profile.get('vip_level', 0) >= 3:
            if '优惠' in answer and 'VIP' not in answer:
                answer += '\n\n作为VIP用户，您还可以享受额外的专属优惠和优先服务。'
        
        # 为新用户添加引导信息
        if not user_profile.get('purchase_history'):
            if '购买' in answer and '新用户' not in answer:
                answer += '\n\n新用户首次购买还有专享优惠哦！'
        
        return answer
    
    def _optimize_format(self, answer: str) -> str:
        """优化格式"""
        # 确保答案结构清晰
        if not answer.endswith(('。', '！', '？')):
            answer += '。'
        
        # 添加友好的结尾
        if not any(ending in answer for ending in ['如有疑问', '还有问题', '需要帮助']):
            answer += '\n\n如果您还有其他问题，随时可以咨询我！'
        
        return answer
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """加载优化规则"""
        return {
            'vip_keywords': ['专属', '优先', '特权', '尊享'],
            'friendly_endings': [
                '如有疑问请随时联系我们',
                '希望这个回答对您有帮助',
                '还有什么可以为您服务的吗'
            ],
            'format_rules': {
                'max_paragraph_length': 200,
                'use_bullet_points': True,
                'add_greeting': True
            }
        }


# 使用示例
if __name__ == "__main__":
    # 创建提示构建器
    prompt_builder = PersonalizedPromptBuilder()
    
    # 示例用户配置
    user_profile = {
        'vip_level': 3,
        'purchase_history': [
            {'items': [{'product_name': 'iPhone 15 Pro'}]}
        ],
        'frequent_categories': ['电子产品', '数码配件'],
        'support_history': [
            {'issue_type': '配送查询'}
        ]
    }
    
    # 构建个性化提示
    query = "iPhone 15 Pro什么时候有货？"
    context = "iPhone 15 Pro目前缺货，预计下周到货"
    
    prompt = prompt_builder.build_personalized_prompt(query, context, user_profile)
    print("个性化提示:")
    print(prompt)
    
    # 答案优化示例
    optimizer = AnswerOptimizer()
    raw_answer = "iPhone 15 Pro目前缺货，预计下周到货。"
    optimized_answer = optimizer.optimize_answer(raw_answer, user_profile, 'product_inquiry')
    print("\n优化后的答案:")
    print(optimized_answer)
