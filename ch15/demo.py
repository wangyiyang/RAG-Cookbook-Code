"""
RAG+Agent融合系统演示
完整展示GraphRAG、RAPTOR和Agent增强RAG的集成应用
"""

import sys
import traceback
from typing import Dict, List, Any
import time

# 导入各个模块
from graph_rag import GraphRAGSystem, MockLLM as GraphMockLLM, MockEmbeddingModel as GraphMockEmbedding
from raptor_tree import RAPTORTree, MockLLM as RaptorMockLLM, MockEmbeddingModel as RaptorMockEmbedding
from agent_enhanced_rag import AgentEnhancedRAG, MockLLM as AgentMockLLM, MockRetriever, MockEmbeddingModel as AgentMockEmbedding

class IntegratedRAGSystem:
    """集成的RAG+Agent系统"""
    
    def __init__(self):
        """初始化集成系统"""
        print("正在初始化集成RAG+Agent系统...")
        
        try:
            # 初始化各个组件
            self.graph_llm = GraphMockLLM()
            self.graph_embedding = GraphMockEmbedding()
            self.graph_rag = GraphRAGSystem(self.graph_llm, self.graph_embedding)
            
            self.raptor_llm = RaptorMockLLM()
            self.raptor_embedding = RaptorMockEmbedding()
            self.raptor_tree = RAPTORTree(self.raptor_llm, self.raptor_embedding, max_cluster_size=5)
            
            self.agent_llm = AgentMockLLM()
            self.agent_retriever = MockRetriever()
            self.agent_embedding = AgentMockEmbedding()
            self.agent_rag = AgentEnhancedRAG(self.agent_llm, self.agent_retriever, self.agent_embedding)
            
            # 系统状态
            self.is_initialized = False
            self.knowledge_base_size = 0
            
            print("✅ 集成系统初始化完成")
            
        except Exception as e:
            print(f"❌ 系统初始化失败: {e}")
            raise
    
    def setup_knowledge_base(self, documents: List[Dict]) -> Dict:
        """设置知识库"""
        
        if not documents:
            raise ValueError("文档列表不能为空")
        
        setup_results = {
            'graph_rag_status': 'pending',
            'raptor_status': 'pending',
            'total_documents': len(documents),
            'setup_time': 0,
            'errors': []
        }
        
        start_time = time.time()
        
        try:
            print(f"\n🔄 开始构建知识库，文档数量: {len(documents)}")
            
            # 1. 构建GraphRAG知识图谱
            print("\n📊 构建GraphRAG知识图谱...")
            try:
                self.graph_rag.build_knowledge_graph(documents)
                setup_results['graph_rag_status'] = 'success'
                print("✅ GraphRAG知识图谱构建完成")
            except Exception as e:
                setup_results['graph_rag_status'] = 'failed'
                setup_results['errors'].append(f"GraphRAG构建失败: {e}")
                print(f"❌ GraphRAG构建失败: {e}")
            
            # 2. 构建RAPTOR树
            print("\n🌲 构建RAPTOR分层树...")
            try:
                self.raptor_tree.build_raptor_tree(documents)
                setup_results['raptor_status'] = 'success'
                print("✅ RAPTOR树构建完成")
            except Exception as e:
                setup_results['raptor_status'] = 'failed'
                setup_results['errors'].append(f"RAPTOR构建失败: {e}")
                print(f"❌ RAPTOR构建失败: {e}")
            
            self.knowledge_base_size = len(documents)
            self.is_initialized = True
            
            setup_results['setup_time'] = time.time() - start_time
            
            print(f"\n✅ 知识库设置完成，耗时: {setup_results['setup_time']:.2f}秒")
            
            return setup_results
            
        except Exception as e:
            setup_results['errors'].append(f"知识库设置失败: {e}")
            print(f"❌ 知识库设置失败: {e}")
            return setup_results
    
    def multi_modal_query(self, query: str, user_context: Dict = None, 
                         use_graph_rag: bool = True, 
                         use_raptor: bool = True, 
                         use_agent: bool = True) -> Dict:
        """多模式查询处理"""
        
        if not self.is_initialized:
            return {
                'error': '系统未初始化，请先设置知识库',
                'suggestions': ['调用setup_knowledge_base()方法']
            }
        
        if not query or not query.strip():
            return {
                'error': '查询不能为空',
                'suggestions': ['请提供有效的查询内容']
            }
        
        user_context = user_context or {
            'user_id': 'demo_user',
            'session_id': 'demo_session',
            'original_query': query
        }
        
        query_results = {
            'query': query,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'modes_used': [],
            'results': {},
            'final_answer': '',
            'confidence_scores': {},
            'execution_times': {},
            'errors': []
        }
        
        total_start_time = time.time()
        
        # 1. GraphRAG查询
        if use_graph_rag:
            print(f"\n🔍 GraphRAG知识图谱查询...")
            try:
                graph_start = time.time()
                graph_result = self.graph_rag.generate_graph_augmented_answer(query)
                graph_time = time.time() - graph_start
                
                query_results['modes_used'].append('GraphRAG')
                query_results['results']['graph_rag'] = graph_result
                query_results['confidence_scores']['graph_rag'] = graph_result.get('confidence_score', 0.0)
                query_results['execution_times']['graph_rag'] = graph_time
                
                print(f"✅ GraphRAG查询完成，耗时: {graph_time:.2f}秒")
                
            except Exception as e:
                error_msg = f"GraphRAG查询失败: {e}"
                query_results['errors'].append(error_msg)
                print(f"❌ {error_msg}")
        
        # 2. RAPTOR查询
        if use_raptor:
            print(f"\n🌲 RAPTOR分层树查询...")
            try:
                raptor_start = time.time()
                raptor_result = self.raptor_tree.generate_raptor_answer(query)
                raptor_time = time.time() - raptor_start
                
                query_results['modes_used'].append('RAPTOR')
                query_results['results']['raptor'] = raptor_result
                query_results['confidence_scores']['raptor'] = raptor_result.get('confidence_score', 0.0)
                query_results['execution_times']['raptor'] = raptor_time
                
                print(f"✅ RAPTOR查询完成，耗时: {raptor_time:.2f}秒")
                
            except Exception as e:
                error_msg = f"RAPTOR查询失败: {e}"
                query_results['errors'].append(error_msg)
                print(f"❌ {error_msg}")
        
        # 3. Agent增强RAG查询
        if use_agent:
            print(f"\n🤖 Agent增强RAG查询...")
            try:
                agent_start = time.time()
                agent_result = self.agent_rag.enhanced_query_processing(query, user_context)
                agent_time = time.time() - agent_start
                
                query_results['modes_used'].append('Agent-RAG')
                query_results['results']['agent_rag'] = agent_result
                query_results['confidence_scores']['agent_rag'] = agent_result.get('confidence_score', 0.0)
                query_results['execution_times']['agent_rag'] = agent_time
                
                print(f"✅ Agent-RAG查询完成，耗时: {agent_time:.2f}秒")
                
            except Exception as e:
                error_msg = f"Agent-RAG查询失败: {e}"
                query_results['errors'].append(error_msg)
                print(f"❌ {error_msg}")
        
        # 4. 结果融合
        query_results['final_answer'] = self._fuse_results(query_results['results'])
        query_results['total_execution_time'] = time.time() - total_start_time
        
        return query_results
    
    def _fuse_results(self, results: Dict) -> str:
        """融合多模式查询结果"""
        
        if not results:
            return "抱歉，所有查询模式都未能成功执行。"
        
        answers = []
        confidence_weights = []
        
        # 收集各模式的答案和置信度
        for mode, result in results.items():
            if isinstance(result, dict) and 'answer' in result:
                answer = result['answer']
                confidence = result.get('confidence_score', 0.5)
                
                if answer and answer.strip():
                    answers.append(f"[{mode.upper()}] {answer}")
                    confidence_weights.append(confidence)
        
        if not answers:
            return "抱歉，未能从任何模式获取到有效答案。"
        
        # 如果只有一个答案，直接返回
        if len(answers) == 1:
            return answers[0]
        
        # 多答案融合策略：选择置信度最高的答案，并提及其他观点
        best_index = confidence_weights.index(max(confidence_weights))
        best_answer = answers[best_index]
        
        # 构建融合答案
        fused_answer = f"综合分析结果：\n\n{best_answer}\n\n"
        
        if len(answers) > 1:
            fused_answer += "其他观点参考：\n"
            for i, answer in enumerate(answers):
                if i != best_index:
                    fused_answer += f"• {answer}\n"
        
        return fused_answer
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        
        status = {
            'is_initialized': self.is_initialized,
            'knowledge_base_size': self.knowledge_base_size,
            'components': {
                'graph_rag': {
                    'status': 'ready' if hasattr(self.graph_rag, 'knowledge_graph') else 'not_ready',
                    'nodes': self.graph_rag.knowledge_graph.number_of_nodes() if hasattr(self.graph_rag, 'knowledge_graph') else 0,
                    'edges': self.graph_rag.knowledge_graph.number_of_edges() if hasattr(self.graph_rag, 'knowledge_graph') else 0
                },
                'raptor_tree': {
                    'status': 'ready' if self.raptor_tree.nodes else 'not_ready',
                    'total_nodes': len(self.raptor_tree.nodes),
                    'root_nodes': len(self.raptor_tree.root_nodes)
                },
                'agent_rag': {
                    'status': 'ready',
                    'memory_records': len(self.agent_rag.memory.memories),
                    'registered_tools': len(self.agent_rag.tool_registry.tools)
                }
            }
        }
        
        return status


def run_comprehensive_demo():
    """运行完整演示"""
    
    print("🚀 启动RAG+Agent融合系统完整演示")
    print("=" * 60)
    
    try:
        # 1. 初始化系统
        system = IntegratedRAGSystem()
        
        # 2. 准备测试文档
        test_documents = [
            {
                "id": "doc1",
                "title": "人工智能与机器学习",
                "content": "人工智能(AI)是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。机器学习是AI的核心子领域，通过算法让计算机从数据中自动学习模式，而无需显式编程。深度学习是机器学习的一个专门分支，使用多层神经网络来模拟人脑的工作方式。",
                "source": "AI基础教程"
            },
            {
                "id": "doc2",
                "title": "Transformer架构详解",
                "content": "Transformer是一种革命性的神经网络架构，于2017年在论文《Attention Is All You Need》中首次提出。它完全基于注意力机制，摒弃了传统的循环和卷积结构。Transformer架构包含编码器和解码器两部分，每部分都由多个相同的层堆叠而成。自注意力机制是Transformer的核心，允许模型在处理序列时关注不同位置的信息。",
                "source": "深度学习论文"
            },
            {
                "id": "doc3",
                "title": "大语言模型的发展",
                "content": "大语言模型(LLM)如GPT、BERT等基于Transformer架构构建，通过在大规模文本数据上进行预训练来学习语言的统计规律。这些模型展现出了强大的语言理解和生成能力，在问答、文本摘要、代码生成等任务上取得了显著成果。RAG(检索增强生成)技术将检索系统与生成模型结合，通过检索相关文档来增强模型的回答质量。",
                "source": "AI前沿技术"
            },
            {
                "id": "doc4",
                "title": "知识图谱与RAG融合",
                "content": "知识图谱是一种结构化的知识表示方法，通过实体、关系和属性来组织信息。GraphRAG将知识图谱技术与检索增强生成相结合，能够处理复杂的关系推理任务。RAPTOR技术通过递归抽象和聚类构建分层的文档表示，支持不同粒度的信息检索。Agent增强的RAG系统具备自主决策、记忆管理和工具调用能力。",
                "source": "RAG技术进展"
            }
        ]
        
        # 3. 设置知识库
        print("\n📚 设置知识库...")
        setup_result = system.setup_knowledge_base(test_documents)
        
        print(f"\n📊 知识库设置结果:")
        print(f"  GraphRAG状态: {setup_result['graph_rag_status']}")
        print(f"  RAPTOR状态: {setup_result['raptor_status']}")
        print(f"  处理文档数: {setup_result['total_documents']}")
        print(f"  设置时间: {setup_result['setup_time']:.2f}秒")
        
        if setup_result['errors']:
            print(f"  错误信息: {setup_result['errors']}")
        
        # 4. 显示系统状态
        print("\n🔍 系统状态检查:")
        status = system.get_system_status()
        
        print(f"  系统初始化: {'✅' if status['is_initialized'] else '❌'}")
        print(f"  知识库大小: {status['knowledge_base_size']} 文档")
        print(f"  GraphRAG: {status['components']['graph_rag']['nodes']} 节点, {status['components']['graph_rag']['edges']} 边")
        print(f"  RAPTOR: {status['components']['raptor_tree']['total_nodes']} 总节点, {status['components']['raptor_tree']['root_nodes']} 根节点")
        print(f"  Agent-RAG: {status['components']['agent_rag']['registered_tools']} 注册工具")
        
        # 5. 测试查询
        test_queries = [
            "什么是Transformer架构，它有什么特点？",
            "深度学习和机器学习的关系是什么？",
            "GraphRAG和RAPTOR技术有什么区别？",
            "大语言模型是如何工作的？"
        ]
        
        print(f"\n🔍 开始测试查询 (共{len(test_queries)}个)...")
        print("=" * 60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 测试查询 {i}: {query}")
            print("-" * 50)
            
            # 执行多模式查询
            result = system.multi_modal_query(
                query, 
                use_graph_rag=True, 
                use_raptor=True, 
                use_agent=True
            )
            
            # 显示结果摘要
            print(f"\n📊 查询结果摘要:")
            print(f"  使用模式: {', '.join(result['modes_used'])}")
            print(f"  总执行时间: {result['total_execution_time']:.2f}秒")
            
            # 显示各模式置信度
            if result['confidence_scores']:
                print(f"  置信度分数:")
                for mode, score in result['confidence_scores'].items():
                    print(f"    {mode}: {score:.3f}")
            
            # 显示错误（如果有）
            if result['errors']:
                print(f"  错误: {result['errors']}")
            
            # 显示最终答案
            print(f"\n💡 融合答案:")
            print(f"  {result['final_answer']}")
            
            if i < len(test_queries):
                print("\n" + "="*30 + " 分隔线 " + "="*30)
        
        # 6. 性能统计
        print(f"\n📈 演示完成统计:")
        print(f"  系统初始化: ✅")
        print(f"  知识库构建: ✅")
        print(f"  测试查询数: {len(test_queries)}")
        print(f"  演示状态: 成功完成")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⚠️  演示被用户中断")
        return False
        
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误:")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误信息: {str(e)}")
        print(f"\n📋 错误追踪:")
        traceback.print_exc()
        return False


def run_simple_demo():
    """运行简化演示"""
    
    print("🎯 快速演示模式")
    print("=" * 40)
    
    try:
        # 初始化系统
        system = IntegratedRAGSystem()
        
        # 简单文档
        simple_docs = [
            {
                "id": "simple1",
                "title": "AI基础",
                "content": "人工智能是让机器模拟人类智能的技术。机器学习是其重要分支。",
                "source": "简化教程"
            }
        ]
        
        # 设置知识库
        system.setup_knowledge_base(simple_docs)
        
        # 简单查询
        query = "什么是人工智能？"
        result = system.multi_modal_query(query)
        
        print(f"\n查询: {query}")
        print(f"答案: {result['final_answer']}")
        print(f"模式: {', '.join(result['modes_used'])}")
        
        return True
        
    except Exception as e:
        print(f"简化演示失败: {e}")
        return False


def main():
    """主函数"""
    
    print("🌟 RAG+Agent融合系统演示程序")
    print("欢迎体验下一代智能问答系统!")
    print("=" * 60)
    
    # 检查命令行参数
    demo_mode = "comprehensive"  # 默认完整演示
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--simple":
            demo_mode = "simple"
        elif sys.argv[1] == "--help":
            print("使用方法:")
            print("  python demo.py          # 完整演示")
            print("  python demo.py --simple # 简化演示")
            print("  python demo.py --help   # 显示帮助")
            return
    
    try:
        if demo_mode == "simple":
            success = run_simple_demo()
        else:
            success = run_comprehensive_demo()
        
        if success:
            print(f"\n🎉 演示成功完成!")
            print("感谢您体验RAG+Agent融合系统!")
        else:
            print(f"\n⚠️  演示未能完全成功，请检查错误信息。")
        
    except Exception as e:
        print(f"\n💥 程序异常终止:")
        print(f"错误: {e}")
        traceback.print_exc()
    
    finally:
        print(f"\n👋 演示程序结束")


if __name__ == "__main__":
    main()