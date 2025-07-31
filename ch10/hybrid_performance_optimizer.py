"""
混合架构性能优化模块
针对 LangChain + LlamaIndex 架构的专项优化
"""

import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    query_time: float
    retrieval_time: float
    generation_time: float
    total_time: float
    confidence: float
    retrieved_docs: int
    memory_usage: float
    timestamp: datetime


class HybridRAGOptimizer:
    """混合RAG性能优化器"""
    
    def __init__(self, hybrid_rag_system):
        """初始化优化器"""
        self.hybrid_rag = hybrid_rag_system
        self.metrics_history = []
        self.optimization_history = []
        
        # 缓存配置
        self.query_cache = {}
        self.embedding_cache = {}
        self.cache_ttl = timedelta(hours=1)
        
        # 性能阈值
        self.performance_thresholds = {
            "max_query_time": 3.0,  # 3秒
            "min_confidence": 0.7,   # 70%
            "max_memory_mb": 500     # 500MB
        }
        
        # 优化策略
        self.optimization_strategies = {
            "cache_optimization": True,
            "parallel_processing": True,
            "dynamic_parameters": True,
            "memory_management": True
        }
        
    def measure_performance(self, query: str, method: str = "smart_query") -> PerformanceMetrics:
        """测量性能指标"""
        import psutil
        import os
        
        # 记录开始状态
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        # 执行查询
        retrieval_start = time.time()
        
        if method == "smart_query":
            result = self.hybrid_rag.smart_query(query)
        else:
            raise ValueError(f"不支持的方法: {method}")
            
        retrieval_end = time.time()
        
        # 模拟生成时间（实际中这部分已包含在smart_query中）
        generation_time = result.get("processing_time", 0) * 0.3  # 假设30%用于生成
        retrieval_time = retrieval_end - retrieval_start - generation_time
        
        # 记录结束状态
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        # 构建性能指标
        metrics = PerformanceMetrics(
            query_time=end_time - start_time,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=result.get("processing_time", end_time - start_time),
            confidence=result.get("confidence", 0.0),
            retrieved_docs=result.get("retrieved_docs_count", 0),
            memory_usage=end_memory - start_memory,
            timestamp=datetime.now()
        )
        
        # 保存到历史记录
        self.metrics_history.append(metrics)
        
        return metrics
        
    def cached_query(self, query: str) -> Dict[str, Any]:
        """带缓存的查询"""
        # 检查缓存
        cache_key = self._generate_cache_key(query)
        
        if cache_key in self.query_cache:
            cached_result, cached_time = self.query_cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                cached_result["from_cache"] = True
                cached_result["cache_hit"] = True
                return cached_result
                
        # 执行新查询
        result = self.hybrid_rag.smart_query(query)
        
        # 更新缓存
        self.query_cache[cache_key] = (result, datetime.now())
        result["from_cache"] = False
        result["cache_hit"] = False
        
        return result
        
    def parallel_batch_query(self, queries: List[str], max_workers: int = 3) -> List[Dict[str, Any]]:
        """并行批量查询"""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_query = {
                executor.submit(self.cached_query, query): query 
                for query in queries
            }
            
            # 收集结果
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    result = future.result()
                    result["original_query"] = query
                    results.append(result)
                except Exception as e:
                    results.append({
                        "original_query": query,
                        "error": str(e)
                    })
                    
        return results
        
    def adaptive_parameter_tuning(self, test_queries: List[str]) -> Dict[str, Any]:
        """自适应参数调优"""
        print("开始自适应参数调优...")
        
        # 参数候选值
        parameter_candidates = {
            "similarity_threshold": [0.6, 0.7, 0.8],
            "top_k": [3, 5, 7],
            "temperature": [0.0, 0.1, 0.2]
        }
        
        best_config = None
        best_score = 0
        optimization_results = []
        
        # 测试不同参数组合
        for threshold in parameter_candidates["similarity_threshold"]:
            for top_k in parameter_candidates["top_k"]:
                for temp in parameter_candidates["temperature"]:
                    
                    config = {
                        "similarity_threshold": threshold,
                        "top_k": top_k,
                        "temperature": temp
                    }
                    
                    # 应用参数配置
                    self._apply_config(config)
                    
                    # 测试性能
                    performance_score = self._evaluate_config_performance(test_queries)
                    
                    optimization_results.append({
                        "config": config,
                        "performance_score": performance_score
                    })
                    
                    # 更新最佳配置
                    if performance_score > best_score:
                        best_score = performance_score
                        best_config = config
                        
        # 应用最佳配置
        if best_config:
            self._apply_config(best_config)
            
        optimization_summary = {
            "best_config": best_config,
            "best_score": best_score,
            "total_configs_tested": len(optimization_results),
            "optimization_history": optimization_results,
            "timestamp": datetime.now().isoformat()
        }
        
        self.optimization_history.append(optimization_summary)
        return optimization_summary
        
    def _apply_config(self, config: Dict[str, Any]):
        """应用配置参数"""
        try:
            self.hybrid_rag.optimize_retrieval_params(
                similarity_threshold=config.get("similarity_threshold", 0.7),
                top_k=config.get("top_k", 5)
            )
            
            # 更新LLM温度参数（如果可能）
            if hasattr(self.hybrid_rag, 'langchain_llm'):
                self.hybrid_rag.langchain_llm.temperature = config.get("temperature", 0.1)
                
        except Exception as e:
            print(f"应用配置失败: {e}")
            
    def _evaluate_config_performance(self, test_queries: List[str]) -> float:
        """评估配置性能"""
        total_score = 0
        valid_queries = 0
        
        for query in test_queries[:3]:  # 只测试前3个查询以节省时间
            try:
                metrics = self.measure_performance(query)
                
                # 计算综合性能分数
                time_score = max(0, 1 - metrics.total_time / 5.0)  # 5秒为满分基准
                confidence_score = metrics.confidence
                memory_score = max(0, 1 - metrics.memory_usage / 100)  # 100MB为基准
                
                query_score = (time_score * 0.4 + 
                              confidence_score * 0.5 + 
                              memory_score * 0.1)
                
                total_score += query_score
                valid_queries += 1
                
            except Exception as e:
                print(f"查询评估失败: {e}")
                continue
                
        return total_score / valid_queries if valid_queries > 0 else 0
        
    def _generate_cache_key(self, query: str) -> str:
        """生成缓存键"""
        import hashlib
        return hashlib.md5(query.encode()).hexdigest()
        
    def memory_cleanup(self):
        """内存清理"""
        # 清理过期缓存
        current_time = datetime.now()
        expired_keys = [
            key for key, (_, cached_time) in self.query_cache.items()
            if current_time - cached_time > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.query_cache[key]
            
        # 清理过期指标历史
        cutoff_time = current_time - timedelta(days=7)
        self.metrics_history = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]
        
        cleanup_stats = {
            "expired_cache_entries": len(expired_keys),
            "remaining_cache_entries": len(self.query_cache),
            "metrics_entries_kept": len(self.metrics_history),
            "cleanup_time": current_time.isoformat()
        }
        
        return cleanup_stats
        
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        if not self.metrics_history:
            return {"message": "暂无性能数据"}
            
        # 统计分析
        query_times = [m.query_time for m in self.metrics_history]
        confidences = [m.confidence for m in self.metrics_history]
        memory_usage = [m.memory_usage for m in self.metrics_history]
        
        # 性能趋势分析
        recent_metrics = self.metrics_history[-10:]  # 最近10次查询
        older_metrics = self.metrics_history[-20:-10] if len(self.metrics_history) >= 20 else []
        
        trend_analysis = {}
        if older_metrics and recent_metrics:
            recent_avg_time = statistics.mean([m.query_time for m in recent_metrics])
            older_avg_time = statistics.mean([m.query_time for m in older_metrics])
            
            trend_analysis = {
                "performance_trend": "improving" if recent_avg_time < older_avg_time else "declining",
                "time_change_percent": round(
                    ((recent_avg_time - older_avg_time) / older_avg_time) * 100, 2
                )
            }
            
        # 性能警告
        warnings = []
        if statistics.mean(query_times) > self.performance_thresholds["max_query_time"]:
            warnings.append("查询时间过长")
        if statistics.mean(confidences) < self.performance_thresholds["min_confidence"]:
            warnings.append("置信度偏低")
        if max(memory_usage) > self.performance_thresholds["max_memory_mb"]:
            warnings.append("内存使用过高")
            
        return {
            "total_queries": len(self.metrics_history),
            "performance_stats": {
                "avg_query_time": round(statistics.mean(query_times), 3),
                "avg_confidence": round(statistics.mean(confidences), 3),
                "avg_memory_usage": round(statistics.mean(memory_usage), 2),
                "max_query_time": round(max(query_times), 3),
                "min_confidence": round(min(confidences), 3)
            },
            "trend_analysis": trend_analysis,
            "performance_warnings": warnings,
            "cache_stats": {
                "cache_entries": len(self.query_cache),
                "cache_hit_potential": len([
                    m for m in self.metrics_history[-50:] 
                    if m.query_time < 1.0
                ])
            },
            "optimization_count": len(self.optimization_history),
            "report_generated": datetime.now().isoformat()
        }
        
    def smart_optimization_recommendation(self) -> Dict[str, Any]:
        """智能优化建议"""
        report = self.get_performance_report()
        
        if "message" in report:
            return {"message": "需要更多性能数据才能提供建议"}
            
        recommendations = []
        
        # 基于性能统计的建议
        stats = report["performance_stats"]
        
        if stats["avg_query_time"] > 2.0:
            recommendations.append({
                "issue": "查询时间过长",
                "recommendation": "启用查询缓存和并行处理",
                "expected_improvement": "30-50%性能提升"
            })
            
        if stats["avg_confidence"] < 0.75:
            recommendations.append({
                "issue": "回答置信度偏低",
                "recommendation": "优化检索参数，提高相似度阈值",
                "expected_improvement": "提升回答质量"
            })
            
        if stats["avg_memory_usage"] > 200:
            recommendations.append({
                "issue": "内存使用较高",
                "recommendation": "定期执行内存清理，优化缓存策略",
                "expected_improvement": "减少50%内存占用"
            })
            
        # 基于趋势的建议
        if "trend_analysis" in report:
            trend = report["trend_analysis"]
            if trend.get("performance_trend") == "declining":
                recommendations.append({
                    "issue": "性能呈下降趋势",
                    "recommendation": "重新进行参数调优，检查系统负载",
                    "expected_improvement": "恢复历史最佳性能"
                })
                
        return {
            "recommendations": recommendations,
            "optimization_priority": "high" if len(recommendations) >= 3 else "medium",
            "auto_optimization_available": True,
            "estimated_optimization_time": f"{len(recommendations) * 2}分钟",
            "report_timestamp": datetime.now().isoformat()
        }


# 使用示例
if __name__ == "__main__":
    # 这里需要先初始化HybridRAGSystem
    # from langchain_llamaindex_hybrid import HybridRAGSystem
    
    print("性能优化模块独立测试")
    print("需要结合 HybridRAGSystem 使用")
    
    # 模拟性能指标
    mock_metrics = PerformanceMetrics(
        query_time=1.5,
        retrieval_time=0.8,
        generation_time=0.7,
        total_time=1.5,
        confidence=0.85,
        retrieved_docs=3,
        memory_usage=50.2,
        timestamp=datetime.now()
    )
    
    print(f"模拟性能指标: {mock_metrics}")
    print("完整功能请参考 demo.py 中的集成示例")