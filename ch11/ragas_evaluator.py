"""
RAGAS RAG系统评估器
基于RAGAS框架的快速RAG系统质量评估工具
"""

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy, context_precision
from datasets import Dataset
from typing import List, Dict, Any
import time
import logging


class RAGASEvaluator:
    """RAGAS评估器 - RAG系统快速体检工具"""
    
    def __init__(self):
        """初始化评估器"""
        # 设置核心评估指标
        self.core_metrics = {
            'faithfulness': faithfulness,           # 忠实度 - 最重要
            'answer_relevancy': answer_relevancy,   # 答案相关性 
            'context_relevancy': context_relevancy, # 上下文相关性
            'context_precision': context_precision, # 上下文精确度
        }
        
        # 设置质量阈值（基于实践经验）
        self.quality_thresholds = {
            'faithfulness': 0.8,        # 忠实度最关键
            'answer_relevancy': 0.7,    # 答案相关性
            'context_relevancy': 0.6,   # 上下文相关性  
            'context_precision': 0.5,   # 上下文精确度
        }
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def evaluate(self, test_data: List[Dict]) -> Dict[str, Any]:
        """
        执行RAGAS评估
        
        Args:
            test_data: 测试数据列表，每个元素包含：
                - question: 用户问题
                - answer: RAG系统生成的答案  
                - contexts: 检索到的上下文列表
                - ground_truth: 标准答案（可选）
        
        Returns:
            评估结果字典
        """
        self.logger.info(f"开始RAGAS评估，共{len(test_data)}个测试样本...")
        
        start_time = time.time()
        
        try:
            # 转换数据格式
            dataset = self._prepare_dataset(test_data)
            
            # 执行评估
            result = evaluate(
                dataset=dataset,
                metrics=list(self.core_metrics.values())
            )
            
            # 处理和解释结果
            evaluation_report = self._process_results(result)
            evaluation_report['evaluation_time'] = time.time() - start_time
            evaluation_report['sample_count'] = len(test_data)
            
            self.logger.info(f"评估完成，用时{evaluation_report['evaluation_time']:.2f}秒")
            return evaluation_report
            
        except Exception as e:
            self.logger.error(f"评估过程出错: {str(e)}")
            raise
    
    def _prepare_dataset(self, test_data: List[Dict]) -> Dataset:
        """准备RAGAS评估数据集"""
        
        # 验证数据格式
        required_fields = ['question', 'answer', 'contexts']
        for i, item in enumerate(test_data):
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"测试样本{i}缺少必要字段: {field}")
        
        # 转换为RAGAS格式
        dataset_dict = {
            'question': [item['question'] for item in test_data],
            'answer': [item['answer'] for item in test_data],
            'contexts': [item['contexts'] for item in test_data],
        }
        
        # 如果有ground_truth，添加进去
        if all('ground_truth' in item for item in test_data):
            dataset_dict['ground_truth'] = [item['ground_truth'] for item in test_data]
        
        return Dataset.from_dict(dataset_dict)
    
    def _process_results(self, ragas_result: Dict) -> Dict[str, Any]:
        """处理RAGAS评估结果"""
        
        # 提取各指标分数
        scores = {}
        for metric_name, metric_obj in self.core_metrics.items():
            metric_key = metric_obj.name if hasattr(metric_obj, 'name') else metric_name
            if metric_key in ragas_result:
                scores[metric_name] = ragas_result[metric_key]
        
        # 计算综合评分（加权平均）
        weights = {
            'faithfulness': 0.4,        # 忠实度权重最高
            'answer_relevancy': 0.3,    # 答案相关性次之
            'context_relevancy': 0.2,   # 上下文相关性
            'context_precision': 0.1,   # 上下文精确度
        }
        
        overall_score = sum(
            scores.get(metric, 0) * weight 
            for metric, weight in weights.items()
        )
        
        # 判断质量等级
        quality_grade = self._get_quality_grade(scores)
        
        # 生成优化建议
        recommendations = self._generate_recommendations(scores)
        
        return {
            'scores': scores,
            'overall_score': overall_score,
            'quality_grade': quality_grade,
            'recommendations': recommendations,
            'detailed_analysis': self._analyze_performance(scores)
        }
    
    def _get_quality_grade(self, scores: Dict[str, float]) -> str:
        """根据分数判断质量等级"""
        
        # 检查是否有严重问题
        critical_issues = []
        if scores.get('faithfulness', 0) < 0.6:
            critical_issues.append('忠实度严重不足')
        if scores.get('answer_relevancy', 0) < 0.5:
            critical_issues.append('答案相关性极差')
        
        if critical_issues:
            return f"不合格 - {', '.join(critical_issues)}"
        
        # 根据整体表现评级
        good_metrics = sum(1 for metric, score in scores.items() 
                          if score >= self.quality_thresholds.get(metric, 0.7))
        
        total_metrics = len(scores)
        
        if good_metrics == total_metrics:
            return "优秀 - 可以放心上线"
        elif good_metrics >= total_metrics * 0.75:
            return "良好 - 可以部署使用"
        elif good_metrics >= total_metrics * 0.5:
            return "需改进 - 建议优化后使用"
        else:
            return "不合格 - 需要重新设计"
    
    def _generate_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """生成针对性优化建议"""
        recommendations = []
        
        # 按优先级检查问题
        if scores.get('faithfulness', 1.0) < self.quality_thresholds['faithfulness']:
            recommendations.append({
                'priority': 'high',
                'metric': 'faithfulness',
                'issue': f"忠实度偏低 ({scores['faithfulness']:.3f})",
                'solution': "优化prompt设计，加强'仅基于上下文回答'的约束",
                'example': "在prompt中明确要求：'请仅基于以下上下文信息回答，不要添加额外信息。'"
            })
        
        if scores.get('answer_relevancy', 1.0) < self.quality_thresholds['answer_relevancy']:
            recommendations.append({
                'priority': 'medium',
                'metric': 'answer_relevancy', 
                'issue': f"答案相关性不足 ({scores['answer_relevancy']:.3f})",
                'solution': "改进生成策略，确保回答直接针对用户问题",
                'example': "调整prompt让模型先理解问题重点，再基于上下文组织答案。"
            })
        
        if scores.get('context_relevancy', 1.0) < self.quality_thresholds['context_relevancy']:
            recommendations.append({
                'priority': 'medium',
                'metric': 'context_relevancy',
                'issue': f"检索质量有待提升 ({scores['context_relevancy']:.3f})", 
                'solution': "优化检索算法，改进embedding模型或查询预处理",
                'example': "考虑使用更好的embedding模型，或对查询进行扩展和重写。"
            })
        
        if scores.get('context_precision', 1.0) < self.quality_thresholds['context_precision']:
            recommendations.append({
                'priority': 'low',
                'metric': 'context_precision',
                'issue': f"检索排序需要优化 ({scores['context_precision']:.3f})",
                'solution': "实施重排序机制，确保最相关内容排在前面", 
                'example': "可以使用Cross-Encoder进行重排序，或调整检索算法参数。"
            })
        
        if not recommendations:
            recommendations.append({
                'priority': 'info',
                'metric': 'overall',
                'issue': '系统表现良好',
                'solution': '各项指标均达标，可以放心使用',
                'example': '建议定期用真实用户问题进行评估，持续监控质量。'
            })
        
        return recommendations
    
    def _analyze_performance(self, scores: Dict[str, float]) -> Dict[str, str]:
        """详细性能分析"""
        analysis = {}
        
        for metric, score in scores.items():
            if score >= 0.9:
                analysis[metric] = "表现优秀"
            elif score >= self.quality_thresholds.get(metric, 0.7):
                analysis[metric] = "表现良好"
            elif score >= 0.5:
                analysis[metric] = "需要改进"
            else:
                analysis[metric] = "表现较差"
        
        return analysis
    
    def compare_versions(self, version_a_data: List[Dict], version_b_data: List[Dict], 
                        names: tuple = ("版本A", "版本B")) -> Dict[str, Any]:
        """对比两个RAG系统版本"""
        
        self.logger.info(f"开始版本对比评估...")
        
        # 分别评估两个版本
        result_a = self.evaluate(version_a_data)
        result_b = self.evaluate(version_b_data)
        
        # 对比分析
        comparison = {
            'version_names': names,
            'scores_comparison': {},
            'overall_comparison': {
                names[0]: result_a['overall_score'],
                names[1]: result_b['overall_score']
            },
            'improvements': [],
            'regressions': []
        }
        
        # 对比各项指标
        for metric in self.core_metrics.keys():
            score_a = result_a['scores'].get(metric, 0)
            score_b = result_b['scores'].get(metric, 0)
            diff = score_b - score_a
            
            comparison['scores_comparison'][metric] = {
                names[0]: score_a,
                names[1]: score_b,
                'difference': diff,
                'change': 'improved' if diff > 0.02 else 'regressed' if diff < -0.02 else 'stable'
            }
            
            if diff > 0.02:
                comparison['improvements'].append(f"{metric}: +{diff:.3f}")
            elif diff < -0.02:
                comparison['regressions'].append(f"{metric}: {diff:.3f}")
        
        return comparison
    
    def generate_report(self, evaluation_result: Dict[str, Any]) -> str:
        """生成详细评估报告"""
        
        report = f"""
=== RAG系统RAGAS评估报告 ===
评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
样本数量: {evaluation_result.get('sample_count', 'N/A')}
评估耗时: {evaluation_result.get('evaluation_time', 0):.2f}秒

== 综合评估 ==
整体评分: {evaluation_result['overall_score']:.3f}
质量等级: {evaluation_result['quality_grade']}

== 详细指标 ==
"""
        
        for metric, score in evaluation_result['scores'].items():
            threshold = self.quality_thresholds.get(metric, 0.7)
            status = "✅ 达标" if score >= threshold else "❌ 不达标"
            analysis = evaluation_result['detailed_analysis'].get(metric, "")
            
            report += f"- {metric}: {score:.3f} (阈值: {threshold}) {status} - {analysis}\n"
        
        report += "\n== 优化建议 ==\n"
        for i, rec in enumerate(evaluation_result['recommendations'], 1):
            priority_icon = {"high": "🔥", "medium": "⚠️", "low": "💡", "info": "ℹ️"}.get(rec['priority'], "")
            report += f"{i}. {priority_icon} {rec['issue']}\n"
            report += f"   解决方案: {rec['solution']}\n"
            report += f"   示例: {rec['example']}\n\n"
        
        return report
    
    def quick_check(self, question: str, answer: str, contexts: List[str], 
                   ground_truth: str = None) -> Dict[str, Any]:
        """快速单次评估"""
        
        test_data = [{
            'question': question,
            'answer': answer, 
            'contexts': contexts
        }]
        
        if ground_truth:
            test_data[0]['ground_truth'] = ground_truth
        
        return self.evaluate(test_data)


# 便捷函数
def quick_evaluate(question: str, answer: str, contexts: List[str], 
                  ground_truth: str = None) -> Dict[str, Any]:
    """快速评估函数 - 适合单次评估"""
    evaluator = RAGASEvaluator()
    return evaluator.quick_check(question, answer, contexts, ground_truth)


def batch_evaluate(test_data: List[Dict]) -> Dict[str, Any]:
    """批量评估函数 - 适合批量评估"""
    evaluator = RAGASEvaluator() 
    return evaluator.evaluate(test_data)


if __name__ == "__main__":
    # 使用示例
    
    # 准备测试数据
    test_data = [
        {
            'question': 'Python中如何创建虚拟环境？',
            'answer': '在Python中创建虚拟环境可以使用venv模块。命令是：python -m venv myenv，然后用source myenv/bin/activate激活。',
            'contexts': [
                'Python虚拟环境可以使用venv模块创建，命令为python -m venv 环境名称。',
                '激活虚拟环境的命令在Linux/Mac下是source venv/bin/activate，Windows下是venv\\Scripts\\activate。'
            ],
            'ground_truth': 'Python创建虚拟环境使用python -m venv命令，激活用source venv/bin/activate。'
        },
        {
            'question': '什么是Docker容器？',
            'answer': 'Docker容器是一种轻量级的虚拟化技术，它将应用程序及其依赖打包在一个可移植的容器中。',
            'contexts': [
                'Docker是一个开源的容器化平台，用于开发、发布和运行应用程序。',
                '容器是一种轻量级的虚拟化技术，相比传统虚拟机消耗更少资源。'
            ],
            'ground_truth': 'Docker容器是轻量级虚拟化技术，用于打包和运行应用程序。'
        }
    ]
    
    # 创建评估器
    evaluator = RAGASEvaluator()
    
    # 执行评估
    print("开始RAGAS评估...")
    result = evaluator.evaluate(test_data)
    
    # 生成详细报告
    report = evaluator.generate_report(result)
    print(report)
    
    # 快速单次评估示例
    print("\n=== 快速评估示例 ===")
    quick_result = quick_evaluate(
        question="什么是机器学习？",
        answer="机器学习是人工智能的一个分支，让计算机从数据中学习模式。",
        contexts=["机器学习是AI的重要分支，通过算法让计算机学习数据模式。"]
    )
    
    print(f"快速评估结果: {quick_result['overall_score']:.3f}")
    print(f"质量等级: {quick_result['quality_grade']}")