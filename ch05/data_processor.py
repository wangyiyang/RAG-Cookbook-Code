"""
电商智能客服RAG系统 - 数据处理模块
实现商品数据和FAQ数据的智能处理
"""

import re
import json
import time
import redis
import queue
import logging
from typing import Dict, List, Any
from dataclasses import dataclass

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class ProductChunk:
    """商品数据块"""
    product_id: str
    chunk_type: str  # basic_info, technical, service, logistics, marketing
    content: str
    metadata: Dict[str, Any]


class EcommerceDataProcessor:
    """电商数据处理器"""
    
    def __init__(self):
        self.chunk_size = 500
        self.overlap = 50
        
    def process_product_data(self, product_info: Dict[str, Any]) -> List[ProductChunk]:
        """处理商品数据，生成多角度描述"""
        # 1. 数据清洗
        cleaned_data = self.clean_product_data(product_info)
        
        # 2. 多角度描述生成
        descriptions = {
            'basic_info': self.generate_basic_info(cleaned_data),
            'technical': self.generate_tech_specs(cleaned_data),
            'service': self.generate_service_info(cleaned_data),
            'logistics': self.generate_shipping_info(cleaned_data),
            'marketing': self.generate_promotion_info(cleaned_data)
        }
        
        # 3. 分块处理
        chunks = self.create_searchable_chunks(descriptions, product_info)
        
        return chunks
    
    def clean_product_data(self, product_info: Dict[str, Any]) -> Dict[str, Any]:
        """清洗商品数据"""
        cleaned = {}
        
        # 去除HTML标签
        html_pattern = re.compile(r'<[^>]+>')
        
        for key, value in product_info.items():
            if isinstance(value, str):
                # 清除HTML标签
                cleaned_value = html_pattern.sub('', value)
                # 规范化文本
                cleaned_value = re.sub(r'\s+', ' ', cleaned_value).strip()
                cleaned[key] = cleaned_value
            else:
                cleaned[key] = value
                
        return cleaned
    
    def generate_basic_info(self, product: Dict[str, Any]) -> str:
        """生成基础信息描述"""
        stock_status = '有货' if product.get('stock', 0) > 0 else '缺货'
        
        return f"""
        商品名称：{product.get('name', '未知')}
        价格：¥{product.get('price', 0)}
        品牌：{product.get('brand', '未知')}
        库存状态：{stock_status}
        用户评分：{product.get('rating', 0)}/5.0
        评价数量：{product.get('review_count', 0)}人评价
        商品描述：{product.get('description', '暂无描述')}
        """
    
    def generate_tech_specs(self, product: Dict[str, Any]) -> str:
        """生成技术规格描述"""
        specs = product.get('specifications', {})
        
        spec_lines = []
        for key, value in specs.items():
            spec_lines.append(f"{key}：{value}")
        
        return f"""
        技术规格：
        {chr(10).join(spec_lines)}
        
        产品型号：{product.get('model', '未知')}
        生产日期：{product.get('manufacture_date', '未知')}
        保修期：{product.get('warranty', '未知')}
        """
    
    def generate_service_info(self, product: Dict[str, Any]) -> str:
        """生成服务政策描述"""
        return f"""
        服务政策：
        退货政策：{product.get('return_policy', '支持7天无理由退货')}
        售后服务：{product.get('after_sales', '提供1年质保服务')}
        配送方式：{product.get('delivery_method', '标准配送')}
        支付方式：{product.get('payment_options', '支持多种支付方式')}
        """
    
    def generate_shipping_info(self, product: Dict[str, Any]) -> str:
        """生成物流配送描述"""
        return f"""
        物流信息：
        配送时间：{product.get('delivery_time', '1-3个工作日')}
        配送范围：{product.get('delivery_area', '全国配送')}
        配送费用：{product.get('shipping_fee', '满99元包邮')}
        发货地点：{product.get('ship_from', '上海仓库')}
        """
    
    def generate_promotion_info(self, product: Dict[str, Any]) -> str:
        """生成营销活动描述"""
        promotions = product.get('promotions', [])
        
        promo_lines = []
        for promo in promotions:
            promo_lines.append(f"- {promo}")
        
        return f"""
        优惠活动：
        {chr(10).join(promo_lines)}
        
        会员价格：{product.get('vip_price', '会员享受专属价格')}
        积分奖励：{product.get('points_reward', '购买可获得积分')}
        """
    
    def create_searchable_chunks(self, descriptions: Dict[str, str], 
                               product_info: Dict[str, Any]) -> List[ProductChunk]:
        """创建可搜索的数据块"""
        chunks = []
        
        for chunk_type, content in descriptions.items():
            if content.strip():
                chunk = ProductChunk(
                    product_id=product_info.get('id', ''),
                    chunk_type=chunk_type,
                    content=content,
                    metadata={
                        'product_name': product_info.get('name', ''),
                        'category': product_info.get('category', ''),
                        'brand': product_info.get('brand', ''),
                        'price': product_info.get('price', 0)
                    }
                )
                chunks.append(chunk)
        
        return chunks


class FAQProcessor:
    """FAQ数据处理器"""
    
    def __init__(self):
        self.question_templates = [
            "{}",
            "怎么{}",
            "如何{}",
            "能不能{}",
            "可以{}吗",
            "关于{}的问题",
            "{}是什么意思",
            "{}怎么办",
            "{}的流程",
            "{}的方法"
        ]
    
    def process_faq_data(self, faq_list: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """处理FAQ数据"""
        processed_faqs = []
        
        for faq in faq_list:
            # 生成问题变体
            variants = self.generate_question_variants(faq['question'])
            
            # 优化答案语调
            optimized_answer = self.optimize_answer_tone(faq['answer'])
            
            # 自动分类
            category_tags = self.auto_categorize(faq['question'])
            
            processed_faq = {
                'original_question': faq['question'],
                'question_variants': variants,
                'answer': optimized_answer,
                'category_tags': category_tags,
                'priority': faq.get('priority', 'normal')
            }
            
            processed_faqs.append(processed_faq)
        
        return processed_faqs
    
    def generate_question_variants(self, question: str) -> List[str]:
        """生成问题变体"""
        variants = []
        
        # 提取关键词
        keywords = self.extract_keywords(question)
        
        # 使用模板生成变体
        for keyword in keywords:
            for template in self.question_templates:
                variant = template.format(keyword)
                if variant != question and variant not in variants:
                    variants.append(variant)
        
        return variants[:10]  # 限制变体数量
    
    def extract_keywords(self, question: str) -> List[str]:
        """提取问题关键词"""
        # 简单的关键词提取逻辑
        keywords = []
        
        # 移除常见的疑问词
        stop_words = ['怎么', '如何', '什么', '哪里', '为什么', '能不能', '可以吗']
        
        words = question.replace('？', '').replace('?', '').split()
        for word in words:
            if word not in stop_words and len(word) > 1:
                keywords.append(word)
        
        return keywords
    
    def optimize_answer_tone(self, answer: str) -> str:
        """优化答案语调"""
        # 添加友好的开头
        friendly_starts = [
            "很高兴为您解答：",
            "我来帮您解决这个问题：",
            "关于这个问题，我的回答是：",
            "让我为您详细说明："
        ]
        
        # 添加贴心的结尾
        friendly_ends = [
            "如果还有其他问题，随时问我哦！",
            "希望这个回答对您有帮助！",
            "如需更多帮助，请随时联系我们！"
        ]
        
        # 随机选择友好的开头和结尾
        import random
        start = random.choice(friendly_starts)
        end = random.choice(friendly_ends)
        
        return f"{start}\n\n{answer}\n\n{end}"
    
    def auto_categorize(self, question: str) -> List[str]:
        """自动分类问题"""
        categories = []
        
        # 定义分类关键词
        category_keywords = {
            'product': ['商品', '产品', '价格', '质量', '规格'],
            'order': ['订单', '购买', '下单', '支付', '结算'],
            'shipping': ['配送', '物流', '运费', '发货', '快递'],
            'return': ['退货', '退款', '换货', '售后', '保修'],
            'account': ['账户', '登录', '注册', '密码', '个人信息'],
            'promotion': ['优惠', '折扣', '活动', '券', '积分']
        }
        
        question_lower = question.lower()
        
        for category, keywords in category_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                categories.append(category)
        
        return categories if categories else ['general']


class SmartCacheSystem:
    """智能缓存系统"""
    
    def __init__(self):
        self.l1_cache = {}  # 内存缓存
        try:
            self.l2_cache = redis.Redis(host='localhost', port=6379)  # Redis缓存
        except (ImportError, redis.ConnectionError):
            self.l2_cache = None
        self.l3_cache = DatabaseCache()  # 数据库缓存
        self.cache_stats = CacheStatistics()
        
    def get_cached_response(self, query_hash: str, user_id: str):
        """多级缓存查询"""
        # L1缓存检查（内存）
        if query_hash in self.l1_cache:
            self.cache_stats.record_hit('L1')
            return self.l1_cache[query_hash]
            
        # L2缓存检查（Redis）
        if self.l2_cache:
            l2_key = f"user_{user_id}_{query_hash}"
            try:
                cached_response = self.l2_cache.get(l2_key)
                if cached_response:
                    self.cache_stats.record_hit('L2')
                    response = json.loads(cached_response)
                    # 提升到L1缓存
                    self._promote_to_l1(query_hash, response)
                    return response
            except Exception as e:
                logger.error(f"L2缓存查询失败: {e}")
            
        # L3缓存检查（数据库）
        l3_response = self.l3_cache.get(query_hash)
        if l3_response:
            self.cache_stats.record_hit('L3')
            # 提升到L2缓存
            self._promote_to_l2(f"user_{user_id}_{query_hash}", l3_response)
            return l3_response
            
        self.cache_stats.record_miss()
        return None
    
    def cache_response(self, query_hash: str, response: Dict[str, Any], user_id: str, ttl: int = 3600):
        """智能缓存存储"""
        # 计算缓存优先级
        priority = self.calculate_cache_priority(response, user_id)
        
        # 根据优先级选择缓存级别
        if priority >= 0.8:
            self._cache_to_all_levels(query_hash, response, user_id, ttl)
        elif priority >= 0.5:
            self._cache_to_l2_l3(query_hash, response, user_id, ttl)
        else:
            self._cache_to_l3_only(query_hash, response, ttl)
    
    def calculate_cache_priority(self, response: Dict[str, Any], user_id: str) -> float:
        """计算缓存优先级"""
        priority = 0.5  # 基础优先级
        
        # VIP用户提升优先级
        if response.get('user_type') == 'VIP':
            priority += 0.2
            
        # 常见问题提升优先级
        if response.get('is_common_query'):
            priority += 0.2
            
        # 高质量回答提升优先级
        if response.get('quality_score', 0) > 0.8:
            priority += 0.1
            
        return min(priority, 1.0)
    
    def _promote_to_l1(self, query_hash: str, response: Dict[str, Any]):
        """提升到L1缓存"""
        # 简单的LRU淘汰策略
        if len(self.l1_cache) >= 1000:
            # 删除最久未使用的项
            oldest_key = next(iter(self.l1_cache))
            del self.l1_cache[oldest_key]
        
        self.l1_cache[query_hash] = response
    
    def _promote_to_l2(self, l2_key: str, response: Dict[str, Any]):
        """提升到L2缓存"""
        if self.l2_cache:
            try:
                self.l2_cache.setex(l2_key, 3600, json.dumps(response))
            except Exception as e:
                logger.error(f"L2缓存存储失败: {e}")
    
    def _cache_to_all_levels(self, query_hash: str, response: Dict[str, Any], user_id: str, ttl: int):
        """缓存到所有级别"""
        self._promote_to_l1(query_hash, response)
        self._promote_to_l2(f"user_{user_id}_{query_hash}", response)
        self.l3_cache.set(query_hash, response, ttl)
    
    def _cache_to_l2_l3(self, query_hash: str, response: Dict[str, Any], user_id: str, ttl: int):
        """缓存到L2和L3"""
        self._promote_to_l2(f"user_{user_id}_{query_hash}", response)
        self.l3_cache.set(query_hash, response, ttl)
    
    def _cache_to_l3_only(self, query_hash: str, response: Dict[str, Any], ttl: int):
        """仅缓存到L3"""
        self.l3_cache.set(query_hash, response, ttl)
    
    def clear_related_cache(self, product_id: str):
        """清理相关缓存"""
        # 清理L1缓存中相关的条目
        keys_to_remove = []
        for key in self.l1_cache:
            if product_id in str(self.l1_cache[key]):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.l1_cache[key]
        
        # 清理L2缓存（Redis）
        if self.l2_cache:
            try:
                pattern = f"*{product_id}*"
                keys = self.l2_cache.keys(pattern)
                if keys:
                    self.l2_cache.delete(*keys)
            except Exception as e:
                logger.error(f"L2缓存清理失败: {e}")
        
        # 清理L3缓存
        self.l3_cache.clear_by_product(product_id)


class RealTimeDataSync:
    """实时数据同步"""
    
    def __init__(self, vector_store, message_queue):
        self.vector_store = vector_store
        self.message_queue = message_queue
        self.sync_stats = SyncStatistics()
        
    def start_sync_worker(self):
        """启动实时同步工作器"""
        logger.info("启动实时数据同步工作器")
        
        while True:
            try:
                # 从消息队列获取更新事件
                message = self.message_queue.get(timeout=1)
                
                # 处理不同类型的更新
                if message['type'] == 'product_update':
                    self.handle_product_update(message['data'])
                elif message['type'] == 'price_change':
                    self.handle_price_change(message['data'])
                elif message['type'] == 'stock_change':
                    self.handle_stock_change(message['data'])
                elif message['type'] == 'faq_update':
                    self.handle_faq_update(message['data'])
                    
                self.sync_stats.record_success(message['type'])
                
            except queue.Empty:
                continue
            except Exception as e:
                self.sync_stats.record_error(str(e))
                logger.error(f"数据同步错误: {e}")
    
    def handle_product_update(self, product_data: Dict[str, Any]):
        """处理商品更新"""
        start_time = time.time()
        
        try:
            logger.info(f"开始处理商品更新: {product_data.get('id')}")
            
            # 1. 删除旧的向量数据
            old_doc_ids = self.vector_store.get_doc_ids_by_product(product_data['id'])
            for doc_id in old_doc_ids:
                self.vector_store.delete(doc_id)
            
            # 2. 生成新的向量数据
            processor = EcommerceDataProcessor()
            new_chunks = processor.process_product_data(product_data)
            
            # 3. 添加到向量数据库
            self.vector_store.add_documents(new_chunks)
            
            # 4. 清理相关缓存
            self.clear_related_cache(product_data['id'])
            
            processing_time = time.time() - start_time
            logger.info(f"商品 {product_data['id']} 更新完成，耗时 {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"商品更新失败 {product_data['id']}: {e}")
            raise
    
    def handle_price_change(self, price_data: Dict[str, Any]):
        """处理价格变更"""
        try:
            product_id = price_data['product_id']
            new_price = price_data['new_price']
            
            logger.info(f"处理价格变更: 商品 {product_id} 新价格 {new_price}")
            
            # 更新向量数据库中的价格信息
            self.vector_store.update_product_price(product_id, new_price)
            
            # 清理价格相关缓存
            self.clear_related_cache(product_id)
            
        except Exception as e:
            logger.error(f"价格更新失败: {e}")
            raise
    
    def handle_stock_change(self, stock_data: Dict[str, Any]):
        """处理库存变更"""
        try:
            product_id = stock_data['product_id']
            new_stock = stock_data['new_stock']
            
            logger.info(f"处理库存变更: 商品 {product_id} 新库存 {new_stock}")
            
            # 更新向量数据库中的库存信息
            self.vector_store.update_product_stock(product_id, new_stock)
            
            # 清理库存相关缓存
            self.clear_related_cache(product_id)
            
        except Exception as e:
            logger.error(f"库存更新失败: {e}")
            raise
    
    def handle_faq_update(self, faq_data: Dict[str, Any]):
        """处理FAQ更新"""
        try:
            logger.info(f"处理FAQ更新: {faq_data.get('id')}")
            
            # 1. 删除旧的FAQ数据
            if faq_data.get('id'):
                self.vector_store.delete_faq(faq_data['id'])
            
            # 2. 处理新的FAQ数据
            processor = FAQProcessor()
            processed_faq = processor.process_faq_data([faq_data])
            
            # 3. 添加到向量数据库
            self.vector_store.add_faq_documents(processed_faq)
            
            # 4. 清理相关缓存
            self.clear_related_cache(f"faq_{faq_data.get('id')}")
            
        except Exception as e:
            logger.error(f"FAQ更新失败: {e}")
            raise
    
    def clear_related_cache(self, identifier: str):
        """清理相关缓存"""
        # 这里应该调用缓存系统的清理方法
        # 实际实现中需要与SmartCacheSystem集成
        pass


class DatabaseCache:
    """数据库缓存模拟类"""
    
    def __init__(self):
        self.cache_data = {}
    
    def get(self, key: str):
        return self.cache_data.get(key)
    
    def set(self, key: str, value: Any, ttl: int):
        self.cache_data[key] = value
    
    def clear_by_product(self, product_id: str):
        keys_to_remove = []
        for key in self.cache_data:
            if product_id in str(self.cache_data[key]):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache_data[key]


class CacheStatistics:
    """缓存统计类"""
    
    def __init__(self):
        self.hits = {'L1': 0, 'L2': 0, 'L3': 0}
        self.misses = 0
    
    def record_hit(self, level: str):
        self.hits[level] += 1
    
    def record_miss(self):
        self.misses += 1
    
    def get_hit_rate(self) -> float:
        total_hits = sum(self.hits.values())
        total_requests = total_hits + self.misses
        return total_hits / total_requests if total_requests > 0 else 0


class SyncStatistics:
    """同步统计类"""
    
    def __init__(self):
        self.successes = {}
        self.errors = []
    
    def record_success(self, sync_type: str):
        self.successes[sync_type] = self.successes.get(sync_type, 0) + 1
    
    def record_error(self, error_msg: str):
        self.errors.append({
            'timestamp': time.time(),
            'error': error_msg
        })


# 使用示例
if __name__ == "__main__":
    # 商品数据处理示例
    processor = EcommerceDataProcessor()
    
    sample_product = {
        'id': 'prod_001',
        'name': 'iPhone 15 Pro',
        'price': 7999,
        'brand': 'Apple',
        'stock': 100,
        'rating': 4.8,
        'review_count': 1200,
        'description': '最新款iPhone，配备A17 Pro芯片',
        'specifications': {
            '屏幕尺寸': '6.1英寸',
            '存储容量': '128GB',
            '摄像头': '4800万像素三摄'
        }
    }
    
    chunks = processor.process_product_data(sample_product)
    print(f"生成了 {len(chunks)} 个数据块")
    
    # FAQ处理示例
    faq_processor = FAQProcessor()
    
    sample_faqs = [
        {
            'question': '怎么退货',
            'answer': '您可以在订单页面申请退货，我们支持7天无理由退货。'
        }
    ]
    
    processed_faqs = faq_processor.process_faq_data(sample_faqs)
    print(f"处理了 {len(processed_faqs)} 个FAQ")
    
    # 缓存系统示例
    cache_system = SmartCacheSystem()
    
    sample_response = {
        'answer': '这是一个测试回答',
        'user_type': 'VIP',
        'quality_score': 0.9
    }
    
    cache_system.cache_response('test_query_hash', sample_response, 'user_123')
    cached_result = cache_system.get_cached_response('test_query_hash', 'user_123')
    print(f"缓存测试结果: {cached_result is not None}")
