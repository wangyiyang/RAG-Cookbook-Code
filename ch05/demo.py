#!/usr/bin/env python3
"""
电商智能客服RAG系统 - 完整演示系统
展示整个系统的核心功能和工作流程
"""

import json
import yaml
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# 导入自定义模块
from data_processor import EcommerceDataProcessor, FAQProcessor, SmartCacheSystem
from query_processor import QueryProcessor, PersonalizedRetriever
from prompt_builder import PersonalizedPromptBuilder, AnswerOptimizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockVectorStore:
    """模拟向量数据库"""
    
    def __init__(self):
        self.documents = []
        self.product_docs = {}
        
    def add_documents(self, chunks):
        """添加文档"""
        for chunk in chunks:
            doc = {
                'id': f"{chunk.product_id}_{chunk.chunk_type}",
                'content': chunk.content,
                'metadata': chunk.metadata,
                'product_id': chunk.product_id,
                'chunk_type': chunk.chunk_type
            }
            self.documents.append(doc)
            
            if chunk.product_id not in self.product_docs:
                self.product_docs[chunk.product_id] = []
            self.product_docs[chunk.product_id].append(doc['id'])
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """模拟相似性搜索"""
        results = []
        query_lower = query.lower()
        
        for doc in self.documents:
            score = 0
            content_lower = doc['content'].lower()
            
            # 简单的关键词匹配评分
            for word in query_lower.split():
                if word in content_lower:
                    score += 1
            
            if score > 0:
                results.append({
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'score': score / len(query_lower.split())
                })
        
        # 按分数排序
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]
    
    def get_doc_ids_by_product(self, product_id: str) -> List[str]:
        """根据商品ID获取文档ID"""
        return self.product_docs.get(product_id, [])
    
    def delete(self, doc_id: str):
        """删除文档"""
        self.documents = [doc for doc in self.documents if doc['id'] != doc_id]
    
    def update_product_price(self, product_id: str, new_price: float):
        """更新商品价格"""
        for doc in self.documents:
            if doc['product_id'] == product_id:
                doc['content'] = doc['content'].replace(
                    f"价格：¥{doc['metadata'].get('price', 0)}", 
                    f"价格：¥{new_price}"
                )
                doc['metadata']['price'] = new_price
    
    def update_product_stock(self, product_id: str, new_stock: int):
        """更新商品库存"""
        for doc in self.documents:
            if doc['product_id'] == product_id:
                stock_status = '有货' if new_stock > 0 else '缺货'
                doc['content'] = doc['content'].replace(
                    '库存状态：有货', f'库存状态：{stock_status}'
                ).replace(
                    '库存状态：缺货', f'库存状态：{stock_status}'
                )


class MockProductDB:
    """模拟商品数据库"""
    
    def __init__(self):
        self.products = {}
    
    def add_product(self, product_data: Dict[str, Any]):
        """添加商品"""
        self.products[product_data['id']] = product_data
    
    def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """获取商品信息"""
        return self.products.get(product_id)


class EcommerceRAGSystem:
    """电商RAG系统主类"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self._initialize_components()
        self._setup_sample_data()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
            return {
                'cache': {'l1_cache_size': 1000},
                'system': {'max_results': 10}
            }
    
    def _initialize_components(self):
        """初始化系统组件"""
        # 数据存储
        self.vector_store = MockVectorStore()
        self.product_db = MockProductDB()
        
        # 数据处理器
        self.data_processor = EcommerceDataProcessor()
        self.faq_processor = FAQProcessor()
        
        # 查询处理器
        self.query_processor = QueryProcessor()
        self.retriever = PersonalizedRetriever(self.vector_store, self.product_db)
        
        # 生成组件
        self.prompt_builder = PersonalizedPromptBuilder()
        self.answer_optimizer = AnswerOptimizer()
        
        # 缓存系统
        self.cache_system = SmartCacheSystem()
        
        logger.info("系统组件初始化完成")
    
    def _setup_sample_data(self):
        """设置示例数据"""
        # 示例商品数据
        sample_products = [
            {
                'id': 'prod_001',
                'name': 'iPhone 15 Pro',
                'price': 7999,
                'brand': 'Apple',
                'category': 'smartphone',
                'stock': 50,
                'rating': 4.8,
                'review_count': 1200,
                'description': '搭载A17 Pro芯片的旗舰手机，支持钛合金机身',
                'specifications': {
                    '屏幕尺寸': '6.1英寸',
                    '存储容量': '128GB',
                    '摄像头': '4800万像素三摄',
                    '处理器': 'A17 Pro芯片'
                },
                'return_policy': '支持14天无理由退货',
                'delivery_time': '次日达',
                'promotions': ['VIP用户享受9折优惠', '购买送无线充电器']
            },
            {
                'id': 'prod_002',
                'name': 'MacBook Pro 14寸',
                'price': 14999,
                'brand': 'Apple',
                'category': 'laptop',
                'stock': 20,
                'rating': 4.9,
                'review_count': 800,
                'description': '专业级笔记本电脑，搭载M3 Pro芯片',
                'specifications': {
                    '屏幕尺寸': '14.2英寸',
                    '内存': '18GB',
                    '存储': '512GB SSD',
                    '处理器': 'M3 Pro芯片'
                },
                'return_policy': '支持14天无理由退货',
                'delivery_time': '2-3个工作日',
                'promotions': ['教育优惠可享受9折', '购买送鼠标和键盘']
            }
        ]
        
        # 处理并存储商品数据
        for product in sample_products:
            self.product_db.add_product(product)
            chunks = self.data_processor.process_product_data(product)
            self.vector_store.add_documents(chunks)
        
        # 示例FAQ数据
        sample_faqs = [
            {
                'question': '如何退货',
                'answer': '您可以在订单页面点击"申请退货"，我们支持14天无理由退货。'
            },
            {
                'question': '配送时间',
                'answer': '大部分商品支持次日达，部分商品需要2-3个工作日。'
            },
            {
                'question': 'VIP会员权益',
                'answer': 'VIP会员享受专属折扣、优先配送、专属客服等多项权益。'
            }
        ]
        
        # 处理FAQ数据
        processed_faqs = self.faq_processor.process_faq_data(sample_faqs)
        # 这里可以添加FAQ到向量数据库的逻辑
        
        logger.info("示例数据设置完成")
    
    def process_user_query(self, query: str, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """处理用户查询"""
        try:
            # 1. 查询分析
            query_analysis = self.query_processor.analyze_query(query, user_profile)
            logger.info(f"查询分析完成: 意图={query_analysis.intent.value}, 置信度={query_analysis.confidence}")
            
            # 2. 个性化检索
            search_results = self.retriever.retrieve_personalized(
                query, user_profile, query_analysis
            )
            logger.info(f"检索到 {len(search_results)} 条相关结果")
            
            # 3. 构建上下文
            context = self._build_context(search_results)
            
            # 4. 生成个性化提示
            prompt = self.prompt_builder.build_personalized_prompt(
                query, context, user_profile
            )
            
            # 5. 生成答案（这里简化为基于上下文的简单回答）
            answer = self._generate_answer(query, context, user_profile)
            
            # 6. 优化答案
            optimized_answer = self.answer_optimizer.optimize_answer(
                answer, user_profile, query_analysis.intent.value
            )
            
            # 7. 缓存结果
            response_data = {
                'answer': optimized_answer,
                'user_type': self.prompt_builder.classify_user_type(user_profile),
                'quality_score': 0.9,
                'sources': len(search_results)
            }
            
            query_hash = str(hash(query))
            self.cache_system.cache_response(
                query_hash, response_data, user_profile.get('user_id', 'anonymous')
            )
            
            return {
                'query': query,
                'answer': optimized_answer,
                'intent': query_analysis.intent.value,
                'confidence': query_analysis.confidence,
                'sources_count': len(search_results),
                'user_type': self.prompt_builder.classify_user_type(user_profile)
            }
            
        except Exception as e:
            logger.error(f"处理查询时发生错误: {e}")
            return {
                'query': query,
                'answer': '抱歉，系统暂时无法处理您的查询，请稍后再试。',
                'error': str(e)
            }
    
    def _build_context(self, search_results: List[Dict[str, Any]]) -> str:
        """构建上下文"""
        context_parts = []
        for result in search_results:
            context_parts.append(result['content'])
        return '\n\n'.join(context_parts)
    
    def _generate_answer(self, query: str, context: str, user_profile: Dict[str, Any]) -> str:
        """生成答案（简化版）"""
        # 这里是简化的答案生成逻辑
        # 实际应用中会调用LLM进行答案生成
        
        if '价格' in query or '多少钱' in query:
            return "根据查询结果，相关商品的价格信息如下：\n" + context
        elif '配送' in query or '物流' in query:
            return "关于配送信息：\n" + context
        elif '退货' in query or '售后' in query:
            return "关于退货和售后服务：\n" + context
        else:
            return "根据您的查询，为您找到以下信息：\n" + context
    
    def run_interactive_demo(self):
        """运行交互式演示"""
        print("=== 电商智能客服RAG系统演示 ===")
        print("输入 'exit' 退出演示")
        print("输入 'user1', 'user2', 'vip' 切换用户类型")
        print()
        
        # 预定义用户配置
        user_profiles = {
            'user1': {
                'user_id': 'user_001',
                'vip_level': 0,
                'purchase_history': [],
                'frequent_categories': []
            },
            'user2': {
                'user_id': 'user_002',
                'vip_level': 1,
                'purchase_history': [
                    {'items': [{'product_name': 'iPhone 14', 'brand': 'Apple', 'category': 'smartphone'}]}
                ],
                'frequent_categories': ['smartphone', 'electronics']
            },
            'vip': {
                'user_id': 'vip_001',
                'vip_level': 5,
                'purchase_history': [
                    {'items': [{'product_name': 'MacBook Pro', 'brand': 'Apple', 'category': 'laptop'}]},
                    {'items': [{'product_name': 'iPhone 15 Pro', 'brand': 'Apple', 'category': 'smartphone'}]}
                ],
                'frequent_categories': ['laptop', 'smartphone', 'electronics']
            }
        }
        
        current_user = 'user1'
        
        while True:
            print(f"\n当前用户: {current_user} (VIP等级: {user_profiles[current_user]['vip_level']})")
            user_input = input("请输入您的问题: ").strip()
            
            if user_input.lower() == 'exit':
                print("感谢使用电商智能客服系统！")
                break
            
            if user_input in user_profiles:
                current_user = user_input
                print(f"已切换到用户: {current_user}")
                continue
            
            if not user_input:
                continue
            
            print("\n处理中...")
            result = self.process_user_query(user_input, user_profiles[current_user])
            
            print(f"\n=== 回答 ===")
            print(result['answer'])
            print(f"\n=== 技术信息 ===")
            print(f"识别意图: {result.get('intent', 'unknown')}")
            print(f"置信度: {result.get('confidence', 0):.2f}")
            print(f"用户类型: {result.get('user_type', 'unknown')}")
            print(f"参考来源: {result.get('sources_count', 0)} 条")
    
    def run_batch_test(self):
        """运行批量测试"""
        print("=== 批量测试模式 ===")
        
        test_cases = [
            {
                'query': 'iPhone 15 Pro多少钱？',
                'user': 'user1',
                'expected_intent': 'price_inquiry'
            },
            {
                'query': 'MacBook Pro什么时候能到货？',
                'user': 'user2',
                'expected_intent': 'shipping_info'
            },
            {
                'query': '如何退货？',
                'user': 'vip',
                'expected_intent': 'return_refund'
            }
        ]
        
        user_profiles = {
            'user1': {'user_id': 'test_001', 'vip_level': 0},
            'user2': {'user_id': 'test_002', 'vip_level': 1},
            'vip': {'user_id': 'test_003', 'vip_level': 5}
        }
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- 测试案例 {i} ---")
            print(f"查询: {test_case['query']}")
            print(f"用户: {test_case['user']}")
            
            result = self.process_user_query(
                test_case['query'], 
                user_profiles[test_case['user']]
            )
            
            print(f"结果: {result.get('intent', 'unknown')}")
            print(f"答案: {result['answer'][:100]}...")
            print(f"置信度: {result.get('confidence', 0):.2f}")
            
            # 验证结果
            if result.get('intent') == test_case['expected_intent']:
                print("✓ 意图识别正确")
            else:
                print("✗ 意图识别错误")


def main():
    """主函数"""
    print("电商智能客服RAG系统演示")
    print("1. 交互式演示")
    print("2. 批量测试")
    print("3. 退出")
    
    try:
        # 初始化系统
        system = EcommerceRAGSystem()
        
        while True:
            choice = input("\n请选择运行模式 (1-3): ").strip()
            
            if choice == '1':
                system.run_interactive_demo()
            elif choice == '2':
                system.run_batch_test()
            elif choice == '3':
                print("再见！")
                break
            else:
                print("无效选择，请重新输入")
                
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n系统错误: {e}")
        logger.error(f"系统错误: {e}")


if __name__ == "__main__":
    main()