"""
HNSW (Hierarchical Navigable Small World) 向量索引实现
提供高效的近似最近邻搜索功能
"""

import numpy as np
import heapq
import random
from typing import List, Tuple, Dict, Set, Optional
import math
from collections import defaultdict


class HNSWIndex:
    """HNSW索引实现"""
    
    def __init__(self, dimension: int, max_m: int = 16, ef_construction: int = 200, ml: float = 1/math.log(2.0)):
        """
        初始化HNSW索引
        
        Args:
            dimension: 向量维度
            max_m: 每层最大连接数
            ef_construction: 构建时的搜索宽度
            ml: 层数分布参数
        """
        self.dimension = dimension      # 向量维度
        self.max_m = max_m             # 每层最大连接数
        self.ef_construction = ef_construction  # 构建时的搜索宽度
        self.ml = ml                   # 层数分布参数
        
        # 存储结构
        self.levels = []               # 层级结构，每层是一个图
        self.vectors = {}              # 向量存储 {node_id: vector}
        self.entry_point = None        # 顶层入口点
        self.node_count = 0           # 节点计数器
        
        # 每层的连接图 {level: {node_id: [neighbor_ids]}}
        self.graph = defaultdict(lambda: defaultdict(list))
    
    def _get_random_level(self) -> int:
        """
        随机确定节点的最高层数
        使用指数分布确保层次结构
        """
        level = 0
        while random.random() < 0.5 and level < 16:  # 限制最大层数
            level += 1
        return level
    
    def _distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算向量间的欧式距离"""
        return np.linalg.norm(vec1 - vec2)
    
    def _search_layer(self, query: np.ndarray, entry_points: List[int], 
                     num_candidates: int, level: int) -> List[Tuple[int, float]]:
        """
        在指定层搜索最近邻
        
        Args:
            query: 查询向量
            entry_points: 入口点列表
            num_candidates: 候选数量
            level: 搜索层级
            
        Returns:
            (node_id, distance) 的列表，按距离排序
        """
        visited = set()
        candidates = []  # 最小堆：(distance, node_id)
        dynamic_candidates = []  # 最大堆：(-distance, node_id)
        
        # 初始化候选集
        for ep in entry_points:
            if ep in self.vectors:
                dist = self._distance(query, self.vectors[ep])
                heapq.heappush(candidates, (dist, ep))
                heapq.heappush(dynamic_candidates, (-dist, ep))
                visited.add(ep)
        
        while candidates:
            current_dist, current = heapq.heappop(candidates)
            
            # 如果当前距离大于动态候选集中的最远距离，停止搜索
            if len(dynamic_candidates) >= num_candidates:
                farthest_dist = -dynamic_candidates[0][0]
                if current_dist > farthest_dist:
                    break
            
            # 检查当前节点的邻居
            for neighbor in self.graph[level][current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    
                    if neighbor in self.vectors:
                        dist = self._distance(query, self.vectors[neighbor])
                        
                        if len(dynamic_candidates) < num_candidates:
                            heapq.heappush(candidates, (dist, neighbor))
                            heapq.heappush(dynamic_candidates, (-dist, neighbor))
                        else:
                            farthest_dist = -dynamic_candidates[0][0]
                            if dist < farthest_dist:
                                heapq.heappush(candidates, (dist, neighbor))
                                heapq.heapreplace(dynamic_candidates, (-dist, neighbor))
        
        # 返回结果，转换为升序
        result = []
        while dynamic_candidates:
            neg_dist, node_id = heapq.heappop(dynamic_candidates)
            result.append((node_id, -neg_dist))
        
        result.reverse()  # 转为距离从小到大
        return result[:num_candidates]
    
    def _select_neighbors_heuristic(self, candidates: List[Tuple[int, float]], 
                                   m: int, query: np.ndarray) -> List[int]:
        """
        启发式邻居选择，保持图的导航性能
        
        Args:
            candidates: 候选邻居列表 (node_id, distance)
            m: 最大选择数量
            query: 查询向量
            
        Returns:
            选中的邻居ID列表
        """
        if len(candidates) <= m:
            return [node_id for node_id, _ in candidates]
        
        # 按距离排序
        candidates.sort(key=lambda x: x[1])
        
        selected = []
        for node_id, distance in candidates:
            if len(selected) >= m:
                break
                
            # 启发式判断：避免冗余连接
            should_connect = True
            for existing_neighbor in selected:
                if existing_neighbor in self.vectors and node_id in self.vectors:
                    # 如果候选节点到现有邻居的距离比到查询点的距离还近
                    # 说明这个连接可能是冗余的
                    existing_dist = self._distance(
                        self.vectors[node_id],
                        self.vectors[existing_neighbor]
                    )
                    if existing_dist < distance:
                        should_connect = False
                        break
            
            if should_connect:
                selected.append(node_id)
        
        return selected
    
    def insert(self, vector: np.ndarray, node_id: Optional[int] = None) -> int:
        """
        向HNSW图中插入新向量
        
        Args:
            vector: 待插入的向量
            node_id: 指定的节点ID，如果为None则自动分配
            
        Returns:
            插入的节点ID
        """
        if node_id is None:
            node_id = self.node_count
            self.node_count += 1
        
        # 存储向量
        self.vectors[node_id] = vector.copy()
        
        # 步骤1：随机确定这个节点存在于哪些层
        level = self._get_random_level()
        
        # 步骤2：如果需要，扩展图的层数
        while len(self.levels) <= level:
            self.levels.append(set())
        
        # 将节点添加到对应层
        for lev in range(level + 1):
            self.levels[lev].add(node_id)
        
        # 步骤3：如果是第一个节点，设为入口点
        if self.entry_point is None:
            self.entry_point = node_id
            return node_id
        
        # 步骤4：从顶层开始搜索最佳插入位置
        current_closest = [self.entry_point]
        
        # 在高层进行粗粒度搜索，每层只找1个最近邻
        for lev in range(len(self.levels) - 1, level, -1):
            current_closest = [
                node_id for node_id, _ in 
                self._search_layer(vector, current_closest, 1, lev)
            ]
        
        # 步骤5：在目标层及以下建立连接
        for lev in range(min(level, len(self.levels) - 1), -1, -1):
            # 在当前层搜索候选邻居
            candidates = self._search_layer(
                vector, current_closest, self.ef_construction, lev
            )
            
            # 使用启发式算法选择最佳邻居
            max_conn = self.max_m if lev > 0 else self.max_m * 2
            connections = self._select_neighbors_heuristic(
                candidates, max_conn, vector
            )
            
            # 建立双向连接
            self.graph[lev][node_id] = connections
            for neighbor in connections:
                self.graph[lev][neighbor].append(node_id)
                
                # 如果邻居的连接数超过限制，需要修剪
                if len(self.graph[lev][neighbor]) > max_conn:
                    # 重新选择邻居的最佳连接
                    neighbor_candidates = [
                        (conn, self._distance(self.vectors[neighbor], self.vectors[conn]))
                        for conn in self.graph[lev][neighbor]
                        if conn in self.vectors
                    ]
                    
                    new_connections = self._select_neighbors_heuristic(
                        neighbor_candidates, max_conn, self.vectors[neighbor]
                    )
                    
                    # 更新邻居的连接
                    old_connections = set(self.graph[lev][neighbor])
                    new_connections_set = set(new_connections)
                    
                    # 移除不再需要的连接
                    for old_conn in old_connections - new_connections_set:
                        if old_conn in self.graph[lev] and neighbor in self.graph[lev][old_conn]:
                            self.graph[lev][old_conn].remove(neighbor)
                    
                    self.graph[lev][neighbor] = new_connections
            
            # 为下一层搜索准备起始点
            current_closest = connections
        
        # 步骤6：如果是最高层，更新全局入口点
        if level >= len(self.levels) - 1:
            self.entry_point = node_id
        
        return node_id
    
    def search(self, query: np.ndarray, k: int = 10, ef: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        搜索最相似的k个向量
        
        Args:
            query: 查询向量
            k: 返回结果数量
            ef: 搜索时的候选集大小，如果为None则使用ef_construction
            
        Returns:
            (node_id, distance) 的列表，按距离排序
        """
        if self.entry_point is None:
            return []  # 空索引
        
        ef = ef or max(self.ef_construction, k)
        
        # 阶段1：从顶层粗定位到目标区域
        current_closest = [self.entry_point]
        for lev in range(len(self.levels) - 1, 0, -1):
            current_closest = [
                node_id for node_id, _ in 
                self._search_layer(query, current_closest, 1, lev)
            ]
        
        # 阶段2：在底层进行详细搜索
        candidates = self._search_layer(query, current_closest, ef, 0)
        
        # 阶段3：返回Top-K结果
        return candidates[:k]
    
    def get_stats(self) -> Dict:
        """获取索引统计信息"""
        stats = {
            'total_nodes': len(self.vectors),
            'total_levels': len(self.levels),
            'entry_point': self.entry_point,
            'dimension': self.dimension,
            'max_m': self.max_m,
            'ef_construction': self.ef_construction
        }
        
        # 计算每层的节点数
        level_sizes = {}
        for level, nodes in enumerate(self.levels):
            level_sizes[f'level_{level}'] = len(nodes)
        
        stats['level_sizes'] = level_sizes
        
        # 计算平均连接度
        if self.levels:
            total_connections = sum(
                len(connections) 
                for level_graph in self.graph.values() 
                for connections in level_graph.values()
            )
            avg_connections = total_connections / max(len(self.vectors), 1)
            stats['avg_connections'] = avg_connections
        
        return stats


def benchmark_hnsw_performance(vectors: np.ndarray, queries: np.ndarray, 
                              k: int = 10) -> Dict:
    """
    HNSW性能基准测试
    
    Args:
        vectors: 用于构建索引的向量集合
        queries: 查询向量集合
        k: 返回结果数量
        
    Returns:
        性能测试结果
    """
    import time
    
    # 构建索引
    print("构建HNSW索引...")
    start_time = time.time()
    
    index = HNSWIndex(dimension=vectors.shape[1])
    for i, vector in enumerate(vectors):
        index.insert(vector, i)
    
    build_time = time.time() - start_time
    
    # 搜索测试
    print("执行搜索测试...")
    search_times = []
    
    for query in queries:
        start_time = time.time()
        results = index.search(query, k)
        search_time = time.time() - start_time
        search_times.append(search_time)
    
    # 统计结果
    avg_search_time = np.mean(search_times)
    stats = index.get_stats()
    
    return {
        'build_time': build_time,
        'avg_search_time_ms': avg_search_time * 1000,
        'total_vectors': len(vectors),
        'index_stats': stats,
        'search_times': search_times
    }


if __name__ == "__main__":
    # 使用示例
    print("HNSW索引演示")
    
    # 生成测试数据
    dimension = 128
    num_vectors = 1000
    num_queries = 100
    
    np.random.seed(42)
    vectors = np.random.random((num_vectors, dimension)).astype(np.float32)
    queries = np.random.random((num_queries, dimension)).astype(np.float32)
    
    # 创建HNSW索引
    index = HNSWIndex(dimension=dimension, max_m=16, ef_construction=200)
    
    # 插入向量
    print(f"插入 {num_vectors} 个向量...")
    for i, vector in enumerate(vectors):
        index.insert(vector, i)
        if (i + 1) % 100 == 0:
            print(f"已插入 {i + 1} 个向量")
    
    # 搜索测试
    print("\n执行搜索测试...")
    query = queries[0]
    results = index.search(query, k=5)
    
    print(f"查询结果 (Top-5):")
    for i, (node_id, distance) in enumerate(results):
        print(f"  {i+1}. 节点 {node_id}, 距离: {distance:.4f}")
    
    # 获取统计信息
    stats = index.get_stats()
    print(f"\n索引统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 性能基准测试
    print("\n执行性能基准测试...")
    benchmark_results = benchmark_hnsw_performance(vectors[:500], queries[:50], k=10)
    
    print(f"基准测试结果:")
    print(f"  构建时间: {benchmark_results['build_time']:.2f}秒")
    print(f"  平均搜索时间: {benchmark_results['avg_search_time_ms']:.2f}毫秒")
    print(f"  总向量数: {benchmark_results['total_vectors']}")


# HNSW索引演示
# 插入 1000 个向量...
# 已插入 100 个向量
# 已插入 200 个向量
# 已插入 300 个向量
# 已插入 400 个向量
# 已插入 500 个向量
# 已插入 600 个向量
# 已插入 700 个向量
# 已插入 800 个向量
# 已插入 900 个向量
# 已插入 1000 个向量

# 执行搜索测试...
# 查询结果 (Top-5):
#   1. 节点 468, 距离: 3.9181
#   2. 节点 771, 距离: 4.0434
#   3. 节点 12, 距离: 4.0455
#   4. 节点 475, 距离: 4.0498
#   5. 节点 284, 距离: 4.0645

# 索引统计信息:
#   total_nodes: 1000
#   total_levels: 11
#   entry_point: 144
#   dimension: 128
#   max_m: 16
#   ef_construction: 200
#   level_sizes: {'level_0': 1000, 'level_1': 488, 'level_2': 242, 'level_3': 114, 'level_4': 63, 'level_5': 35, 'level_6': 21, 'level_7': 8, 'level_8': 3, 'level_9': 1, 'level_10': 1}
#   avg_connections: 34.835

# 执行性能基准测试...
# 构建HNSW索引...
# 执行搜索测试...
# 基准测试结果:
#   构建时间: 2.56秒
#   平均搜索时间: 4.50毫秒
#   总向量数: 500