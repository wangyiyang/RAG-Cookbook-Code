"""
法律系统监控器
实现系统性能监控、质量追踪和业务指标分析
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import statistics
from datetime import datetime, timedelta


class MetricType(Enum):
    """指标类型"""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    BUSINESS = "business"
    SYSTEM = "system"


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """监控指标"""
    name: str
    value: float
    timestamp: float
    metric_type: MetricType
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class Alert:
    """告警信息"""
    alert_id: str
    level: AlertLevel
    message: str
    timestamp: float
    metric_name: str
    current_value: float
    threshold: float
    resolved: bool = False


@dataclass
class PerformanceStats:
    """性能统计"""
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput: float
    error_rate: float
    cache_hit_rate: float


@dataclass
class QualityStats:
    """质量统计"""
    avg_accuracy: float
    avg_relevance: float
    avg_completeness: float
    validation_pass_rate: float
    risk_distribution: Dict[str, int]


@dataclass
class BusinessStats:
    """业务统计"""
    total_queries: int
    unique_users: int
    case_types_distribution: Dict[str, int]
    user_satisfaction: float
    conversion_rate: float


class LegalSystemMonitor:
    """法律系统监控器"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history = defaultdict(lambda: deque(maxlen=history_size))
        self.alerts = []
        self.alert_rules = self._load_alert_rules()
        self.performance_metrics = {}
        self.quality_metrics = {}
        self.business_metrics = {}
        self.system_metrics = {}
        
        # 监控状态
        self.monitoring_active = False
        self.monitor_thread = None
        self.lock = threading.Lock()
        
        # 配置日志
        self._setup_logging()
    
    def start_monitoring(self) -> None:
        """启动监控"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info("法律系统监控已启动")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("法律系统监控已停止")
    
    def record_metric(self, metric: Metric) -> None:
        """记录指标"""
        with self.lock:
            self.metrics_history[metric.name].append(metric)
            
            # 更新当前指标值
            if metric.metric_type == MetricType.PERFORMANCE:
                self.performance_metrics[metric.name] = metric.value
            elif metric.metric_type == MetricType.QUALITY:
                self.quality_metrics[metric.name] = metric.value
            elif metric.metric_type == MetricType.BUSINESS:
                self.business_metrics[metric.name] = metric.value
            elif metric.metric_type == MetricType.SYSTEM:
                self.system_metrics[metric.name] = metric.value
            
            # 检查告警条件
            self._check_alerts(metric)
    
    def record_query_performance(
        self, 
        query_id: str, 
        response_time: float, 
        success: bool, 
        cache_hit: bool = False
    ) -> None:
        """记录查询性能"""
        timestamp = time.time()
        
        # 响应时间
        self.record_metric(Metric(
            name="response_time",
            value=response_time,
            timestamp=timestamp,
            metric_type=MetricType.PERFORMANCE,
            tags={"query_id": query_id}
        ))
        
        # 成功率
        self.record_metric(Metric(
            name="success_rate",
            value=1.0 if success else 0.0,
            timestamp=timestamp,
            metric_type=MetricType.PERFORMANCE,
            tags={"query_id": query_id}
        ))
        
        # 缓存命中率
        self.record_metric(Metric(
            name="cache_hit_rate",
            value=1.0 if cache_hit else 0.0,
            timestamp=timestamp,
            metric_type=MetricType.PERFORMANCE,
            tags={"query_id": query_id}
        ))
    
    def record_quality_metrics(
        self, 
        query_id: str, 
        accuracy: float, 
        relevance: float, 
        completeness: float,
        validation_passed: bool,
        risk_level: str
    ) -> None:
        """记录质量指标"""
        timestamp = time.time()
        tags = {"query_id": query_id, "risk_level": risk_level}
        
        # 准确性
        self.record_metric(Metric(
            name="accuracy",
            value=accuracy,
            timestamp=timestamp,
            metric_type=MetricType.QUALITY,
            tags=tags
        ))
        
        # 相关性
        self.record_metric(Metric(
            name="relevance",
            value=relevance,
            timestamp=timestamp,
            metric_type=MetricType.QUALITY,
            tags=tags
        ))
        
        # 完整性
        self.record_metric(Metric(
            name="completeness",
            value=completeness,
            timestamp=timestamp,
            metric_type=MetricType.QUALITY,
            tags=tags
        ))
        
        # 验证通过率
        self.record_metric(Metric(
            name="validation_pass_rate",
            value=1.0 if validation_passed else 0.0,
            timestamp=timestamp,
            metric_type=MetricType.QUALITY,
            tags=tags
        ))
    
    def record_business_metrics(
        self, 
        user_id: str, 
        case_type: str, 
        satisfaction_score: float = None,
        converted: bool = False
    ) -> None:
        """记录业务指标"""
        timestamp = time.time()
        tags = {"user_id": user_id, "case_type": case_type}
        
        # 查询数量
        self.record_metric(Metric(
            name="query_count",
            value=1.0,
            timestamp=timestamp,
            metric_type=MetricType.BUSINESS,
            tags=tags
        ))
        
        # 用户满意度
        if satisfaction_score is not None:
            self.record_metric(Metric(
                name="satisfaction_score",
                value=satisfaction_score,
                timestamp=timestamp,
                metric_type=MetricType.BUSINESS,
                tags=tags
            ))
        
        # 转化率
        self.record_metric(Metric(
            name="conversion_rate",
            value=1.0 if converted else 0.0,
            timestamp=timestamp,
            metric_type=MetricType.BUSINESS,
            tags=tags
        ))
    
    def get_performance_dashboard(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """获取性能仪表板"""
        cutoff_time = time.time() - (time_range_hours * 3600)
        
        # 计算性能统计
        performance_stats = self._calculate_performance_stats(cutoff_time)
        
        # 获取性能趋势
        performance_trends = self._get_performance_trends(cutoff_time)
        
        # 获取系统健康状态
        system_health = self._get_system_health()
        
        return {
            "stats": performance_stats,
            "trends": performance_trends,
            "health": system_health,
            "timestamp": time.time()
        }
    
    def get_quality_dashboard(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """获取质量仪表板"""
        cutoff_time = time.time() - (time_range_hours * 3600)
        
        # 计算质量统计
        quality_stats = self._calculate_quality_stats(cutoff_time)
        
        # 获取质量趋势
        quality_trends = self._get_quality_trends(cutoff_time)
        
        # 获取风险分布
        risk_distribution = self._get_risk_distribution(cutoff_time)
        
        return {
            "stats": quality_stats,
            "trends": quality_trends,
            "risk_distribution": risk_distribution,
            "timestamp": time.time()
        }
    
    def get_business_dashboard(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """获取业务仪表板"""
        cutoff_time = time.time() - (time_range_hours * 3600)
        
        # 计算业务统计
        business_stats = self._calculate_business_stats(cutoff_time)
        
        # 获取业务趋势
        business_trends = self._get_business_trends(cutoff_time)
        
        # 获取用户行为分析
        user_behavior = self._analyze_user_behavior(cutoff_time)
        
        return {
            "stats": business_stats,
            "trends": business_trends,
            "user_behavior": user_behavior,
            "timestamp": time.time()
        }
    
    def get_comprehensive_report(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """获取综合报告"""
        return {
            "performance": self.get_performance_dashboard(time_range_hours),
            "quality": self.get_quality_dashboard(time_range_hours),
            "business": self.get_business_dashboard(time_range_hours),
            "alerts": self.get_active_alerts(),
            "recommendations": self._generate_recommendations(),
            "report_timestamp": time.time()
        }
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        with self.lock:
            return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        with self.lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    self.logger.info(f"告警已解决: {alert_id}")
                    return True
            return False
    
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self.monitoring_active:
            try:
                # 定期清理历史数据
                self._cleanup_old_data()
                
                # 计算衍生指标
                self._calculate_derived_metrics()
                
                # 检查系统健康状态
                self._check_system_health()
                
                # 生成周期性报告
                self._generate_periodic_report()
                
                time.sleep(60)  # 每分钟运行一次
                
            except Exception as e:
                self.logger.error(f"监控循环出错: {e}")
                time.sleep(60)
    
    def _calculate_performance_stats(self, cutoff_time: float) -> PerformanceStats:
        """计算性能统计"""
        # 获取时间范围内的响应时间数据
        response_times = self._get_metric_values("response_time", cutoff_time)
        success_rates = self._get_metric_values("success_rate", cutoff_time)
        cache_hits = self._get_metric_values("cache_hit_rate", cutoff_time)
        
        if not response_times:
            return PerformanceStats(0, 0, 0, 0, 0, 0)
        
        # 计算统计值
        avg_response_time = statistics.mean(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times)
        p99_response_time = statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else max(response_times)
        
        throughput = len(response_times) / (time.time() - cutoff_time) * 3600  # 每小时查询数
        error_rate = 1.0 - statistics.mean(success_rates) if success_rates else 0.0
        cache_hit_rate = statistics.mean(cache_hits) if cache_hits else 0.0
        
        return PerformanceStats(
            avg_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            throughput=throughput,
            error_rate=error_rate,
            cache_hit_rate=cache_hit_rate
        )
    
    def _calculate_quality_stats(self, cutoff_time: float) -> QualityStats:
        """计算质量统计"""
        accuracy_values = self._get_metric_values("accuracy", cutoff_time)
        relevance_values = self._get_metric_values("relevance", cutoff_time)
        completeness_values = self._get_metric_values("completeness", cutoff_time)
        validation_values = self._get_metric_values("validation_pass_rate", cutoff_time)
        
        # 获取风险分布
        risk_distribution = self._get_risk_distribution(cutoff_time)
        
        return QualityStats(
            avg_accuracy=statistics.mean(accuracy_values) if accuracy_values else 0.0,
            avg_relevance=statistics.mean(relevance_values) if relevance_values else 0.0,
            avg_completeness=statistics.mean(completeness_values) if completeness_values else 0.0,
            validation_pass_rate=statistics.mean(validation_values) if validation_values else 0.0,
            risk_distribution=risk_distribution
        )
    
    def _calculate_business_stats(self, cutoff_time: float) -> BusinessStats:
        """计算业务统计"""
        query_counts = self._get_metric_values("query_count", cutoff_time)
        satisfaction_scores = self._get_metric_values("satisfaction_score", cutoff_time)
        conversion_rates = self._get_metric_values("conversion_rate", cutoff_time)
        
        # 获取用户和案例类型分布
        unique_users = len(set(self._get_metric_tags("query_count", "user_id", cutoff_time)))
        case_types = self._get_case_types_distribution(cutoff_time)
        
        return BusinessStats(
            total_queries=int(sum(query_counts)) if query_counts else 0,
            unique_users=unique_users,
            case_types_distribution=case_types,
            user_satisfaction=statistics.mean(satisfaction_scores) if satisfaction_scores else 0.0,
            conversion_rate=statistics.mean(conversion_rates) if conversion_rates else 0.0
        )
    
    def _get_metric_values(self, metric_name: str, cutoff_time: float) -> List[float]:
        """获取指标值"""
        with self.lock:
            metrics = self.metrics_history.get(metric_name, [])
            return [m.value for m in metrics if m.timestamp >= cutoff_time]
    
    def _get_metric_tags(self, metric_name: str, tag_key: str, cutoff_time: float) -> List[str]:
        """获取指标标签"""
        with self.lock:
            metrics = self.metrics_history.get(metric_name, [])
            return [m.tags.get(tag_key, '') for m in metrics 
                   if m.timestamp >= cutoff_time and tag_key in m.tags]
    
    def _get_performance_trends(self, cutoff_time: float) -> Dict[str, List[Tuple[float, float]]]:
        """获取性能趋势"""
        trends = {}
        
        # 计算每小时的平均响应时间
        response_time_trend = self._calculate_hourly_trend("response_time", cutoff_time)
        trends["response_time"] = response_time_trend
        
        # 计算每小时的错误率
        error_rate_trend = self._calculate_hourly_trend("success_rate", cutoff_time, inverse=True)
        trends["error_rate"] = error_rate_trend
        
        # 计算每小时的吞吐量
        throughput_trend = self._calculate_hourly_count("response_time", cutoff_time)
        trends["throughput"] = throughput_trend
        
        return trends
    
    def _get_quality_trends(self, cutoff_time: float) -> Dict[str, List[Tuple[float, float]]]:
        """获取质量趋势"""
        trends = {}
        
        trends["accuracy"] = self._calculate_hourly_trend("accuracy", cutoff_time)
        trends["relevance"] = self._calculate_hourly_trend("relevance", cutoff_time)
        trends["completeness"] = self._calculate_hourly_trend("completeness", cutoff_time)
        trends["validation_pass_rate"] = self._calculate_hourly_trend("validation_pass_rate", cutoff_time)
        
        return trends
    
    def _get_business_trends(self, cutoff_time: float) -> Dict[str, List[Tuple[float, float]]]:
        """获取业务趋势"""
        trends = {}
        
        trends["query_count"] = self._calculate_hourly_count("query_count", cutoff_time)
        trends["satisfaction_score"] = self._calculate_hourly_trend("satisfaction_score", cutoff_time)
        trends["conversion_rate"] = self._calculate_hourly_trend("conversion_rate", cutoff_time)
        
        return trends
    
    def _calculate_hourly_trend(
        self, 
        metric_name: str, 
        cutoff_time: float, 
        inverse: bool = False
    ) -> List[Tuple[float, float]]:
        """计算每小时趋势"""
        with self.lock:
            metrics = self.metrics_history.get(metric_name, [])
            
        # 按小时分组
        hourly_data = defaultdict(list)
        for metric in metrics:
            if metric.timestamp >= cutoff_time:
                hour = int(metric.timestamp // 3600) * 3600
                hourly_data[hour].append(metric.value)
        
        # 计算每小时平均值
        trend = []
        for hour, values in sorted(hourly_data.items()):
            avg_value = statistics.mean(values)
            if inverse:
                avg_value = 1.0 - avg_value
            trend.append((hour, avg_value))
        
        return trend
    
    def _calculate_hourly_count(self, metric_name: str, cutoff_time: float) -> List[Tuple[float, float]]:
        """计算每小时计数"""
        with self.lock:
            metrics = self.metrics_history.get(metric_name, [])
        
        # 按小时分组计数
        hourly_counts = defaultdict(int)
        for metric in metrics:
            if metric.timestamp >= cutoff_time:
                hour = int(metric.timestamp // 3600) * 3600
                hourly_counts[hour] += 1
        
        # 转换为列表
        trend = []
        for hour, count in sorted(hourly_counts.items()):
            trend.append((hour, count))
        
        return trend
    
    def _get_risk_distribution(self, cutoff_time: float) -> Dict[str, int]:
        """获取风险分布"""
        risk_levels = self._get_metric_tags("accuracy", "risk_level", cutoff_time)
        risk_distribution = defaultdict(int)
        
        for risk_level in risk_levels:
            if risk_level:
                risk_distribution[risk_level] += 1
        
        return dict(risk_distribution)
    
    def _get_case_types_distribution(self, cutoff_time: float) -> Dict[str, int]:
        """获取案例类型分布"""
        case_types = self._get_metric_tags("query_count", "case_type", cutoff_time)
        case_distribution = defaultdict(int)
        
        for case_type in case_types:
            if case_type:
                case_distribution[case_type] += 1
        
        return dict(case_distribution)
    
    def _analyze_user_behavior(self, cutoff_time: float) -> Dict[str, Any]:
        """分析用户行为"""
        user_ids = self._get_metric_tags("query_count", "user_id", cutoff_time)
        
        # 用户活跃度分析
        user_activity = defaultdict(int)
        for user_id in user_ids:
            if user_id:
                user_activity[user_id] += 1
        
        # 计算用户行为统计
        if user_activity:
            avg_queries_per_user = statistics.mean(user_activity.values())
            max_queries_per_user = max(user_activity.values())
            active_users = len([count for count in user_activity.values() if count > 1])
        else:
            avg_queries_per_user = 0
            max_queries_per_user = 0
            active_users = 0
        
        return {
            "total_users": len(user_activity),
            "active_users": active_users,
            "avg_queries_per_user": avg_queries_per_user,
            "max_queries_per_user": max_queries_per_user,
            "user_retention_rate": active_users / len(user_activity) if user_activity else 0
        }
    
    def _get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        health_status = {
            "overall_status": "healthy",
            "components": {},
            "warnings": [],
            "errors": []
        }
        
        # 检查性能组件
        if self.performance_metrics:
            if self.performance_metrics.get("response_time", 0) > 5.0:
                health_status["components"]["performance"] = "degraded"
                health_status["warnings"].append("响应时间过长")
            else:
                health_status["components"]["performance"] = "healthy"
        
        # 检查质量组件
        if self.quality_metrics:
            if self.quality_metrics.get("accuracy", 1.0) < 0.8:
                health_status["components"]["quality"] = "degraded"
                health_status["warnings"].append("准确率偏低")
            else:
                health_status["components"]["quality"] = "healthy"
        
        # 检查活跃告警
        active_alerts = self.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.level == AlertLevel.CRITICAL]
        
        if critical_alerts:
            health_status["overall_status"] = "critical"
            health_status["errors"].extend([a.message for a in critical_alerts])
        elif health_status["warnings"]:
            health_status["overall_status"] = "warning"
        
        return health_status
    
    def _check_alerts(self, metric: Metric) -> None:
        """检查告警条件"""
        metric_rules = self.alert_rules.get(metric.name, [])
        
        for rule in metric_rules:
            if self._evaluate_alert_condition(metric, rule):
                alert = Alert(
                    alert_id=f"{metric.name}_{int(metric.timestamp)}",
                    level=AlertLevel(rule["level"]),
                    message=rule["message"].format(
                        metric_name=metric.name,
                        value=metric.value,
                        threshold=rule["threshold"]
                    ),
                    timestamp=metric.timestamp,
                    metric_name=metric.name,
                    current_value=metric.value,
                    threshold=rule["threshold"]
                )
                
                with self.lock:
                    self.alerts.append(alert)
                
                self.logger.warning(f"告警触发: {alert.message}")
    
    def _evaluate_alert_condition(self, metric: Metric, rule: Dict[str, Any]) -> bool:
        """评估告警条件"""
        condition = rule["condition"]
        threshold = rule["threshold"]
        
        if condition == "greater_than":
            return metric.value > threshold
        elif condition == "less_than":
            return metric.value < threshold
        elif condition == "equals":
            return metric.value == threshold
        else:
            return False
    
    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 基于性能指标的建议
        if self.performance_metrics.get("response_time", 0) > 3.0:
            recommendations.append("响应时间较长，建议优化检索算法或增加缓存")
        
        if self.performance_metrics.get("cache_hit_rate", 1.0) < 0.6:
            recommendations.append("缓存命中率偏低，建议优化缓存策略")
        
        # 基于质量指标的建议
        if self.quality_metrics.get("accuracy", 1.0) < 0.8:
            recommendations.append("准确率偏低，建议加强质量验证机制")
        
        if self.quality_metrics.get("validation_pass_rate", 1.0) < 0.7:
            recommendations.append("验证通过率偏低，建议优化内容生成质量")
        
        # 基于业务指标的建议
        if self.business_metrics.get("satisfaction_score", 5.0) < 4.0:
            recommendations.append("用户满意度偏低，建议改进用户体验")
        
        if self.business_metrics.get("conversion_rate", 1.0) < 0.3:
            recommendations.append("转化率偏低，建议优化服务流程")
        
        return recommendations
    
    def _cleanup_old_data(self) -> None:
        """清理旧数据"""
        cutoff_time = time.time() - (7 * 24 * 3600)  # 保留7天数据
        
        with self.lock:
            # 清理旧的告警
            self.alerts = [alert for alert in self.alerts 
                          if alert.timestamp > cutoff_time or not alert.resolved]
    
    def _calculate_derived_metrics(self) -> None:
        """计算衍生指标"""
        # 计算系统负载
        current_time = time.time()
        last_hour = current_time - 3600
        
        recent_queries = len(self._get_metric_values("query_count", last_hour))
        system_load = min(recent_queries / 1000.0, 1.0)  # 假设1000查询/小时为满负载
        
        self.record_metric(Metric(
            name="system_load",
            value=system_load,
            timestamp=current_time,
            metric_type=MetricType.SYSTEM
        ))
    
    def _check_system_health(self) -> None:
        """检查系统健康状态"""
        current_time = time.time()
        
        # 检查系统负载
        system_load = self.system_metrics.get("system_load", 0.0)
        if system_load > 0.8:
            self.record_metric(Metric(
                name="system_health",
                value=0.3,  # 不健康
                timestamp=current_time,
                metric_type=MetricType.SYSTEM
            ))
        else:
            self.record_metric(Metric(
                name="system_health",
                value=1.0,  # 健康
                timestamp=current_time,
                metric_type=MetricType.SYSTEM
            ))
    
    def _generate_periodic_report(self) -> None:
        """生成周期性报告"""
        # 每小时生成一次报告
        current_time = time.time()
        if hasattr(self, '_last_report_time'):
            if current_time - self._last_report_time < 3600:
                return
        
        self._last_report_time = current_time
        
        # 生成报告
        report = self.get_comprehensive_report(1)  # 最近1小时
        
        # 记录到日志
        self.logger.info(f"周期性报告: {json.dumps(report, indent=2, ensure_ascii=False)}")
    
    def _load_alert_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载告警规则"""
        return {
            "response_time": [
                {
                    "condition": "greater_than",
                    "threshold": 5.0,
                    "level": "warning",
                    "message": "响应时间过长: {value:.2f}秒 (阈值: {threshold}秒)"
                },
                {
                    "condition": "greater_than",
                    "threshold": 10.0,
                    "level": "critical",
                    "message": "响应时间严重过长: {value:.2f}秒 (阈值: {threshold}秒)"
                }
            ],
            "accuracy": [
                {
                    "condition": "less_than",
                    "threshold": 0.8,
                    "level": "warning",
                    "message": "准确率偏低: {value:.2f} (阈值: {threshold})"
                },
                {
                    "condition": "less_than",
                    "threshold": 0.6,
                    "level": "critical",
                    "message": "准确率严重偏低: {value:.2f} (阈值: {threshold})"
                }
            ],
            "system_load": [
                {
                    "condition": "greater_than",
                    "threshold": 0.8,
                    "level": "warning",
                    "message": "系统负载过高: {value:.2f} (阈值: {threshold})"
                },
                {
                    "condition": "greater_than",
                    "threshold": 0.9,
                    "level": "critical",
                    "message": "系统负载严重过高: {value:.2f} (阈值: {threshold})"
                }
            ]
        }
    
    def _setup_logging(self) -> None:
        """设置日志"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)


# 使用示例
if __name__ == "__main__":
    monitor = LegalSystemMonitor()
    
    # 启动监控
    monitor.start_monitoring()
    
    # 模拟记录一些指标
    import random
    
    for i in range(100):
        # 模拟查询性能
        response_time = random.uniform(0.5, 3.0)
        success = random.choice([True, True, True, False])  # 75%成功率
        cache_hit = random.choice([True, False])
        
        monitor.record_query_performance(
            query_id=f"query_{i}",
            response_time=response_time,
            success=success,
            cache_hit=cache_hit
        )
        
        # 模拟质量指标
        monitor.record_quality_metrics(
            query_id=f"query_{i}",
            accuracy=random.uniform(0.7, 0.95),
            relevance=random.uniform(0.8, 0.95),
            completeness=random.uniform(0.6, 0.9),
            validation_passed=random.choice([True, True, False]),
            risk_level=random.choice(["low_risk", "medium_risk", "high_risk"])
        )
        
        # 模拟业务指标
        monitor.record_business_metrics(
            user_id=f"user_{random.randint(1, 50)}",
            case_type=random.choice(["合同纠纷", "侵权纠纷", "婚姻家庭", "劳动争议"]),
            satisfaction_score=random.uniform(3.0, 5.0),
            converted=random.choice([True, False])
        )
        
        time.sleep(0.1)  # 模拟时间间隔
    
    # 获取监控报告
    print("=== 性能仪表板 ===")
    perf_dashboard = monitor.get_performance_dashboard()
    print(f"平均响应时间: {perf_dashboard['stats'].avg_response_time:.2f}秒")
    print(f"P95响应时间: {perf_dashboard['stats'].p95_response_time:.2f}秒")
    print(f"错误率: {perf_dashboard['stats'].error_rate:.2%}")
    print(f"缓存命中率: {perf_dashboard['stats'].cache_hit_rate:.2%}")
    
    print("\n=== 质量仪表板 ===")
    quality_dashboard = monitor.get_quality_dashboard()
    print(f"平均准确率: {quality_dashboard['stats'].avg_accuracy:.2%}")
    print(f"平均相关性: {quality_dashboard['stats'].avg_relevance:.2%}")
    print(f"验证通过率: {quality_dashboard['stats'].validation_pass_rate:.2%}")
    print(f"风险分布: {quality_dashboard['risk_distribution']}")
    
    print("\n=== 业务仪表板 ===")
    business_dashboard = monitor.get_business_dashboard()
    print(f"总查询数: {business_dashboard['stats'].total_queries}")
    print(f"独立用户数: {business_dashboard['stats'].unique_users}")
    print(f"平均满意度: {business_dashboard['stats'].user_satisfaction:.2f}/5.0")
    print(f"转化率: {business_dashboard['stats'].conversion_rate:.2%}")
    
    print("\n=== 活跃告警 ===")
    active_alerts = monitor.get_active_alerts()
    for alert in active_alerts:
        print(f"[{alert.level.value.upper()}] {alert.message}")
    
    print("\n=== 系统健康状态 ===")
    health = monitor.get_performance_dashboard()['health']
    print(f"总体状态: {health['overall_status']}")
    print(f"组件状态: {health['components']}")
    if health['warnings']:
        print(f"警告: {health['warnings']}")
    
    # 停止监控
    monitor.stop_monitoring()