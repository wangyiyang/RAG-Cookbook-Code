"""
多级缓存系统实现
L1内存缓存 + L2Redis缓存 + L3持久化缓存
"""

import redis
import json
import time
from typing import Any, Optional, Dict, List
from threading import Lock
import hashlib

class MultiLevelCacheSystem:
    def __init__(self, config: Dict):
        # L1: 内存缓存（最热数据）
        self.l1_cache = {}
        self.l1_max_size = config.get('l1_max_size', 1000)
        self.l1_lock = Lock()
        
        # L2: Redis缓存（热数据）
        self.l2_cache = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            db=config.get('redis_db', 0),
            decode_responses=True
        )
        
        # L3: 持久化缓存（温数据）
        self.l3_cache = PersistentCache(config.get('l3_config', {}))
        
        # 缓存统计
        self.cache_stats = {
            'l1_hits': 0, 'l1_misses': 0,
            'l2_hits': 0, 'l2_misses': 0,
            'l3_hits': 0, 'l3_misses': 0
        }
        
        # 热度追踪
        self.access_frequency = {}
        self.access_recency = {}
    
    def get(self, key: str) -> Optional[Any]:
        """多级缓存获取"""
        # 生成缓存键的哈希
        cache_key = self._generate_cache_key(key)
        
        # L1缓存检查
        l1_result = self._get_from_l1(cache_key)
        if l1_result is not None:
            self.cache_stats['l1_hits'] += 1
            self._update_access_stats(cache_key)
            return l1_result
        self.cache_stats['l1_misses'] += 1
        
        # L2缓存检查
        l2_result = self._get_from_l2(cache_key)
        if l2_result is not None:
            self.cache_stats['l2_hits'] += 1
            # 提升到L1缓存
            self._promote_to_l1(cache_key, l2_result)
            self._update_access_stats(cache_key)
            return l2_result
        self.cache_stats['l2_misses'] += 1
        
        # L3缓存检查
        l3_result = self._get_from_l3(cache_key)
        if l3_result is not None:
            self.cache_stats['l3_hits'] += 1
            # 提升到L2缓存
            self._promote_to_l2(cache_key, l3_result)
            self._update_access_stats(cache_key)
            return l3_result
        self.cache_stats['l3_misses'] += 1
        
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600, priority: str = 'normal'):
        """多级缓存设置"""
        cache_key = self._generate_cache_key(key)
        
        # 根据优先级和热度决定缓存级别
        cache_level = self._determine_cache_level(cache_key, priority)
        
        if cache_level >= 1:
            self._set_to_l1(cache_key, value)
        if cache_level >= 2:
            self._set_to_l2(cache_key, value, ttl)
        if cache_level >= 3:
            self._set_to_l3(cache_key, value, ttl)
        
        self._update_access_stats(cache_key)
    
    def _determine_cache_level(self, cache_key: str, priority: str) -> int:
        """确定缓存级别"""
        # 获取访问频率和最近访问时间
        frequency = self.access_frequency.get(cache_key, 0)
        recency = self.access_recency.get(cache_key, 0)
        current_time = time.time()
        
        # 计算热度分数
        frequency_score = min(frequency / 10, 1.0)  # 频率分数
        recency_score = max(0, 1 - (current_time - recency) / 3600)  # 最近访问分数
        priority_score = {'high': 1.0, 'normal': 0.7, 'low': 0.3}.get(priority, 0.7)
        
        hotness_score = (frequency_score * 0.4 + recency_score * 0.4 + priority_score * 0.2)
        
        # 根据热度决定缓存级别
        if hotness_score >= 0.8:
            return 3  # 存储到所有级别
        elif hotness_score >= 0.5:
            return 2  # 存储到L2和L3
        else:
            return 1  # 只存储到L3
    
    def _get_from_l1(self, cache_key: str) -> Optional[Any]:
        """从L1缓存获取"""
        with self.l1_lock:
            return self.l1_cache.get(cache_key)
    
    def _set_to_l1(self, cache_key: str, value: Any):
        """设置到L1缓存"""
        with self.l1_lock:
            # LRU淘汰策略
            if len(self.l1_cache) >= self.l1_max_size:
                # 找到最少使用的键
                lru_key = min(self.access_recency, key=self.access_recency.get)
                if lru_key in self.l1_cache:
                    del self.l1_cache[lru_key]
            
            self.l1_cache[cache_key] = value
    
    def _get_from_l2(self, cache_key: str) -> Optional[Any]:
        """从L2缓存获取"""
        try:
            cached_value = self.l2_cache.get(cache_key)
            if cached_value:
                return json.loads(cached_value)
        except Exception as e:
            print(f"L2缓存获取错误: {e}")
        return None
    
    def _set_to_l2(self, cache_key: str, value: Any, ttl: int):
        """设置到L2缓存"""
        try:
            serialized_value = json.dumps(value, default=str)
            self.l2_cache.setex(cache_key, ttl, serialized_value)
        except Exception as e:
            print(f"L2缓存设置错误: {e}")
    
    def _promote_to_l1(self, cache_key: str, value: Any):
        """提升到L1缓存"""
        self._set_to_l1(cache_key, value)
    
    def _promote_to_l2(self, cache_key: str, value: Any):
        """提升到L2缓存"""
        self._set_to_l2(cache_key, value, 3600)
    
    def _update_access_stats(self, cache_key: str):
        """更新访问统计"""
        current_time = time.time()
        self.access_frequency[cache_key] = self.access_frequency.get(cache_key, 0) + 1
        self.access_recency[cache_key] = current_time
    
    def _generate_cache_key(self, key: str) -> str:
        """生成缓存键"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def get_cache_statistics(self) -> Dict:
        """获取缓存统计信息"""
        total_requests = sum(self.cache_stats.values())
        if total_requests == 0:
            return self.cache_stats
        
        hit_rate = {
            'l1_hit_rate': self.cache_stats['l1_hits'] / total_requests,
            'l2_hit_rate': self.cache_stats['l2_hits'] / total_requests,
            'l3_hit_rate': self.cache_stats['l3_hits'] / total_requests,
            'overall_hit_rate': (
                self.cache_stats['l1_hits'] + 
                self.cache_stats['l2_hits'] + 
                self.cache_stats['l3_hits']
            ) / total_requests
        }
        
        return {**self.cache_stats, **hit_rate}

class PersistentCache:
    """持久化缓存实现"""
    def __init__(self, config: Dict):
        self.config = config
        # 实际实现中可以使用文件系统或数据库
    
    def get(self, key: str) -> Optional[Any]:
        # 实现持久化缓存获取逻辑
        pass
    
    def set(self, key: str, value: Any, ttl: int):
        # 实现持久化缓存设置逻辑
        pass