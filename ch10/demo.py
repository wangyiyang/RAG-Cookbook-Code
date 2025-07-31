"""
混合架构RAG系统完整演示
整合所有模块的综合使用示例
"""

import os
import sys
from typing import List, Dict, Any
from datetime import datetime

# 导入自定义模块
from langchain_llamaindex_hybrid import HybridRAGSystem
from llama_packs_integration import LlamaPacksEnhancedRAG
from hybrid_performance_optimizer import HybridRAGOptimizer


class HybridRAGDemo:
    """混合RAG系统演示类"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """初始化演示系统"""
        print("=== 初始化混合RAG演示系统 ===")
        
        # 核心系统组件
        self.hybrid_rag = HybridRAGSystem(model_name=model_name)
        self.packs_rag = LlamaPacksEnhancedRAG(model_name=model_name)
        self.optimizer = None  # 在构建知识库后初始化
        
        # 演示配置
        self.demo_config = {
            "test_queries": [
                "什么是机器学习？",
                "深度学习与传统机器学习有什么区别？",
                "卷积神经网络的工作原理是什么？",
                "如何评估机器学习模型的性能？"
            ],
            "documents_path": "./documents/",
            "enable_optimization": True,
            "show_detailed_output": True
        }
        
        print("系统初始化完成")
        
    def setup_demo_environment(self):
        """设置演示环境"""
        print("\n=== 设置演示环境 ===")
        
        # 检查API密钥
        if not os.getenv("OPENAI_API_KEY"):
            print("警告: 未设置 OPENAI_API_KEY 环境变量")
            print("请设置API密钥: export OPENAI_API_KEY='your-api-key'")
            return False
            
        # 检查文档目录
        docs_path = self.demo_config["documents_path"]
        if not os.path.exists(docs_path):
            print(f"创建文档目录: {docs_path}")
            os.makedirs(docs_path, exist_ok=True)
            
            # 创建示例文档
            self._create_sample_documents(docs_path)
            
        print("演示环境设置完成")
        return True
        
    def _create_sample_documents(self, docs_path: str):
        """创建示例文档"""
        sample_docs = {
            "machine_learning_basics.txt": """
机器学习基础知识

机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。

主要类型：
1. 监督学习：使用标记数据进行训练
2. 无监督学习：从未标记数据中发现模式
3. 强化学习：通过奖励和惩罚机制学习

常见算法：
- 线性回归
- 决策树
- 支持向量机
- 神经网络

应用领域：
- 图像识别
- 自然语言处理
- 推荐系统
- 医疗诊断
            """,
            
            "deep_learning_guide.txt": """
深度学习指南

深度学习是机器学习的一个子集，基于人工神经网络。

核心概念：
- 多层神经网络
- 反向传播算法
- 激活函数
- 梯度下降优化

主要架构：
1. 卷积神经网络（CNN）：擅长图像处理
2. 循环神经网络（RNN）：适合序列数据
3. 长短期记忆网络（LSTM）：解决长序列问题
4. Transformer：现代NLP的基础

优势：
- 自动特征提取
- 处理大规模数据
- 在复杂任务上表现优异

挑战：
- 需要大量数据
- 计算资源要求高
- 模型可解释性差
            """,
            
            "model_evaluation.txt": """
机器学习模型评估

模型评估是机器学习项目的关键环节。

评估指标：

分类任务：
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数
- AUC-ROC曲线

回归任务：
- 均方误差（MSE）
- 平均绝对误差（MAE）
- R²决定系数

交叉验证：
- K折交叉验证
- 留一法
- 时间序列分割

过拟合检测：
- 训练集与测试集性能对比
- 学习曲线分析
- 正则化技术

最佳实践：
1. 数据预处理的重要性
2. 特征选择和工程
3. 超参数调优
4. 模型解释和可视化
            """
        }
        
        for filename, content in sample_docs.items():
            file_path = os.path.join(docs_path, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
                
        print(f"创建了 {len(sample_docs)} 个示例文档")
        
    def build_knowledge_base(self, documents_path: str = None) -> Dict[str, Any]:
        """构建知识库"""
        if documents_path is None:
            documents_path = self.demo_config["documents_path"]
            
        print(f"\n=== 构建知识库: {documents_path} ===")
        
        try:
            # 构建混合架构知识库
            hybrid_build_info = self.hybrid_rag.build_knowledge_base(documents_path)
            print(f"混合架构知识库: {hybrid_build_info}")
            
            # 构建Llama Packs知识库
            packs_build_info = self.packs_rag.build_knowledge_base(documents_path)
            print(f"Llama Packs知识库: {packs_build_info}")
            
            # 初始化性能优化器
            if self.demo_config["enable_optimization"]:
                self.optimizer = HybridRAGOptimizer(self.hybrid_rag)
                print("性能优化器已初始化")
            
            return {
                "hybrid_rag": hybrid_build_info,
                "packs_rag": packs_build_info,
                "optimizer_ready": self.optimizer is not None,
                "build_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"知识库构建失败: {e}")
            return {"error": str(e)}
            
    def demo_basic_queries(self):
        """演示基础查询功能"""
        print(f"\n=== 基础查询演示 ===")
        
        test_queries = self.demo_config["test_queries"][:2]  # 只测试前2个
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- 查询 {i}: {query} ---")
            
            try:
                # 混合架构查询
                result = self.hybrid_rag.smart_query(query)
                
                print(f"回答: {result['answer'][:200]}...")
                print(f"置信度: {result['confidence']}")
                print(f"处理时间: {result['processing_time']}秒")
                print(f"检索文档数: {result['retrieved_docs_count']}")
                
                if self.demo_config["show_detailed_output"]:
                    print(f"来源信息: {len(result['sources'])} 个文档片段")
                    
            except Exception as e:
                print(f"查询失败: {e}")
                
    def demo_llama_packs_features(self):
        """演示Llama Packs功能"""
        print(f"\n=== Llama Packs 功能演示 ===")
        
        test_query = self.demo_config["test_queries"][0]
        print(f"测试查询: {test_query}")
        
        try:
            # Agent Search演示
            print(f"\n--- Agent Search方法 ---")
            agent_result = self.packs_rag.agent_search_query(test_query)
            if "error" not in agent_result:
                print(f"Agent搜索策略: {agent_result.get('search_strategy', 'N/A')}")
                print(f"置信度: {agent_result.get('confidence', 'N/A')}")
                print(f"处理时间: {agent_result.get('processing_time', 'N/A')}秒")
            
            # Fusion Retrieval演示
            print(f"\n--- Fusion Retrieval方法 ---")
            fusion_result = self.packs_rag.fusion_retrieval_query(test_query)
            if "error" not in fusion_result:
                print(f"融合查询数: {len(fusion_result.get('fusion_queries', []))}")
                print(f"融合分数: {fusion_result.get('fusion_score', 'N/A')}")
                print(f"置信度: {fusion_result.get('confidence', 'N/A')}")
                
            # 多Pack对比
            print(f"\n--- 多Pack方法对比 ---")
            comparison = self.packs_rag.multi_pack_comparison(test_query)
            
            if "error" not in comparison:
                perf = comparison["performance_analysis"]
                print(f"最快方法: {perf['fastest_method']}")
                print(f"最高置信度方法: {perf['most_confident']}")
                print(f"平均处理时间: {perf['avg_processing_time']}秒")
                print(f"总处理时间: {comparison['total_processing_time']}秒")
                
        except Exception as e:
            print(f"Llama Packs演示失败: {e}")
            
    def demo_performance_optimization(self):
        """演示性能优化功能"""
        if not self.optimizer:
            print("性能优化器未启用")
            return
            
        print(f"\n=== 性能优化演示 ===")
        
        # 性能测量
        test_query = self.demo_config["test_queries"][0]
        print(f"性能测量查询: {test_query}")
        
        try:
            # 测量基准性能
            print(f"\n--- 基准性能测量 ---")
            baseline_metrics = self.optimizer.measure_performance(test_query)
            print(f"查询时间: {baseline_metrics.query_time:.3f}秒")
            print(f"置信度: {baseline_metrics.confidence}")
            print(f"内存使用: {baseline_metrics.memory_usage:.2f}MB")
            
            # 缓存查询演示
            print(f"\n--- 缓存查询演示 ---")
            cached_result = self.optimizer.cached_query(test_query)
            print(f"缓存命中: {cached_result.get('cache_hit', False)}")
            
            # 再次查询同样问题（应该命中缓存）
            cached_result2 = self.optimizer.cached_query(test_query)
            print(f"第二次查询缓存命中: {cached_result2.get('cache_hit', False)}")
            
            # 并行查询演示
            print(f"\n--- 并行查询演示 ---")
            parallel_queries = self.demo_config["test_queries"][:3]
            parallel_results = self.optimizer.parallel_batch_query(parallel_queries)
            
            successful_queries = [r for r in parallel_results if "error" not in r]
            print(f"并行查询完成: {len(successful_queries)}/{len(parallel_queries)} 成功")
            
            # 性能报告
            print(f"\n--- 性能报告 ---")
            performance_report = self.optimizer.get_performance_report()
            
            if "message" not in performance_report:
                stats = performance_report["performance_stats"]
                print(f"总查询次数: {performance_report['total_queries']}")
                print(f"平均查询时间: {stats['avg_query_time']}秒")
                print(f"平均置信度: {stats['avg_confidence']}")
                print(f"缓存条目数: {performance_report['cache_stats']['cache_entries']}")
                
                # 优化建议
                recommendations = self.optimizer.smart_optimization_recommendation()
                if recommendations.get("recommendations"):
                    print(f"\n优化建议数量: {len(recommendations['recommendations'])}")
                    print(f"优化优先级: {recommendations['optimization_priority']}")
                    
        except Exception as e:
            print(f"性能优化演示失败: {e}")
            
    def run_comprehensive_demo(self):
        """运行完整演示"""
        print("🚀 混合架构RAG系统综合演示开始")
        print("=" * 50)
        
        # 1. 环境设置
        if not self.setup_demo_environment():
            print("❌ 环境设置失败，演示终止")
            return
            
        # 2. 知识库构建
        build_result = self.build_knowledge_base()
        if "error" in build_result:
            print(f"❌ 知识库构建失败: {build_result['error']}")
            return
            
        print("✅ 知识库构建成功")
        
        # 3. 基础查询演示
        self.demo_basic_queries()
        
        # 4. Llama Packs功能演示
        self.demo_llama_packs_features()
        
        # 5. 性能优化演示
        if self.demo_config["enable_optimization"]:
            self.demo_performance_optimization()
            
        # 6. 演示总结
        self._demo_summary()
        
        print("\n🎉 混合架构RAG系统演示完成")
        print("=" * 50)
        
    def _demo_summary(self):
        """演示总结"""
        print(f"\n=== 演示总结 ===")
        
        summary_stats = {
            "系统组件": "LangChain + LlamaIndex 混合架构",
            "Llama Packs": "4种增强方法（Agent Search、Fusion Retrieval等）",
            "性能优化": "缓存、并行处理、自适应参数调优",
            "测试查询": len(self.demo_config["test_queries"]),
            "演示时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        for key, value in summary_stats.items():
            print(f"{key}: {value}")
            
        print(f"\n💡 核心优势:")
        print("- 结合LangChain工作流控制和LlamaIndex检索优化")
        print("- 丰富的Llama Packs生态工具")
        print("- 智能性能监控和优化")
        print("- 企业级缓存和并行处理")


# 主程序入口
if __name__ == "__main__":
    print("混合架构RAG系统演示程序")
    print("请确保已安装所需依赖并设置OpenAI API密钥")
    
    try:
        # 创建演示实例
        demo = HybridRAGDemo(model_name="gpt-3.5-turbo")
        
        # 运行完整演示
        demo.run_comprehensive_demo()
        
    except KeyboardInterrupt:
        print("\n\n用户中断演示")
    except Exception as e:
        print(f"\n\n演示过程中发生错误: {e}")
        print("请检查依赖安装和API配置")
    finally:
        print("\n感谢使用混合架构RAG系统演示！")
        
    # 提供交互式查询选项
    print("\n" + "="*50)
    print("💡 想要交互式体验？")
    print("可以导入 HybridRAGDemo 类进行自定义查询：")
    print("""
from demo import HybridRAGDemo

demo = HybridRAGDemo()
demo.setup_demo_environment()
demo.build_knowledge_base()

# 自定义查询
result = demo.hybrid_rag.smart_query("你的问题")
print(result['answer'])
    """)