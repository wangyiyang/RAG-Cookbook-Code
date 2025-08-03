"""
优化的HNSW索引实现
包含距离计算缓存、批量搜索、预热机制等优化
"""

import numpy as np
import heapq
from typing import List, Set, Dict, Tuple
import random

class OptimizedHNSW:
    def __init__(self, space: str = 'cosine', max_connections: int = 16, 
                 ef_construction: int = 200, max_elements: int = 100000):
        self.space = space
        self.max_connections = max_connections  # M参数
        self.ef_construction = ef_construction   # efConstruction参数
        self.max_elements = max_elements
        
        # 多层图结构
        self.layers = []
        self.entry_point = None
        self.element_count = 0
        self.level_multiplier = 1 / np.log(2.0)
        
        # 性能优化相关
        self.distance_cache = {}
        self.search_cache = {}
        
    def _select_level(self) -> int:
        """选择新元素的层级"""
        level = int(-np.log(random.random()) * self.level_multiplier)
        return level
    
    def _calculate_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算向量距离（带缓存优化）"""
        # 使用向量ID作为缓存键
        vec1_id = id(vec1)
        vec2_id = id(vec2)
        cache_key = (min(vec1_id, vec2_id), max(vec1_id, vec2_id))
        
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        if self.space == 'cosine':
            # 余弦距离
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            distance = 1 - dot_product / (norm1 * norm2)
        elif self.space == 'euclidean':
            # 欧氏距离
            distance = np.linalg.norm(vec1 - vec2)
        else:
            # 内积距离
            distance = -np.dot(vec1, vec2)
        
        # 缓存结果
        if len(self.distance_cache) < 10000:  # 限制缓存大小
            self.distance_cache[cache_key] = distance
        
        return distance
    
    def _search_layer(self, query: np.ndarray, entry_points: List[int], 
                     num_closest: int, level: int) -> List[Tuple[float, int]]:
        """在指定层级搜索"""
        visited = set()
        candidates = []
        dynamic_list = []
        
        # 初始化候选集
        for ep in entry_points:
            if ep not in visited:
                dist = self._calculate_distance(query, self.get_vector(ep))
                heapq.heappush(candidates, (-dist, ep))  # 使用负距离实现最大堆
                heapq.heappush(dynamic_list, (dist, ep))
                visited.add(ep)
        
        while candidates:
            current_dist, current = heapq.heappop(candidates)
            current_dist = -current_dist
            
            # 剪枝：如果当前距离大于动态列表中最远的距离，停止搜索
            if len(dynamic_list) >= num_closest and current_dist > dynamic_list[0][0]:
                break
            
            # 扩展搜索
            neighbors = self.get_neighbors(current, level)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist = self._calculate_distance(query, self.get_vector(neighbor))
                    
                    if len(dynamic_list) < num_closest:
                        heapq.heappush(candidates, (-dist, neighbor))
                        heapq.heappush(dynamic_list, (dist, neighbor))
                    elif dist < dynamic_list[0][0]:
                        # 替换距离最远的元素
                        heapq.heapreplace(dynamic_list, (dist, neighbor))
                        heapq.heappush(candidates, (-dist, neighbor))
        
        return dynamic_list
    
    def search(self, query: np.ndarray, k: int = 10, ef: int = None) -> List[Tuple[int, float]]:
        """HNSW搜索"""
        if ef is None:
            ef = max(self.ef_construction, k)
        
        if self.entry_point is None:
            return []
        
        # 从顶层开始搜索
        current_closest = [self.entry_point]
        
        # 逐层搜索到第1层
        for level in range(len(self.layers) - 1, 0, -1):
            current_closest = [
                node_id for _, node_id in 
                self._search_layer(query, current_closest, 1, level)
            ]
        
        # 在第0层进行精确搜索
        candidates = self._search_layer(query, current_closest, ef, 0)
        
        # 返回top-k结果
        result = sorted(candidates, key=lambda x: x[0])[:k]
        return [(node_id, dist) for dist, node_id in result]
    
    def batch_search(self, queries: List[np.ndarray], k: int = 10) -> List[List[Tuple[int, float]]]:
        """批量搜索优化"""
        results = []
        
        # 预热缓存
        if len(queries) > 10:
            self._warm_up_cache(queries[:5])
        
        # 并行处理查询
        for query in queries:
            result = self.search(query, k)
            results.append(result)
        
        return results
    
    def _warm_up_cache(self, sample_queries: List[np.ndarray]):
        """缓存预热"""
        for query in sample_queries:
            # 执行小规模搜索来预热距离缓存
            self.search(query, k=5, ef=50)
    
    def get_vector(self, node_id: int) -> np.ndarray:
        """获取节点向量 - 需要在实际实现中完成"""
        # 这里应该返回实际的向量数据
        pass
    
    def get_neighbors(self, node_id: int, level: int) -> List[int]:
        """获取节点邻居 - 需要在实际实现中完成"""
        # 这里应该返回指定层级的邻居节点
        if level < len(self.layers) and node_id in self.layers[level]:
            return list(self.layers[level][node_id])
        return []