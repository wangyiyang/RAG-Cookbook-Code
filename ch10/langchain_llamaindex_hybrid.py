"""
LangChain + LlamaIndex 混合架构核心实现
结合LangChain的工作流控制和LlamaIndex的检索优化能力
"""

from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

# LangChain imports
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.llms import OpenAI as LangChainOpenAI

# LlamaIndex imports
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.llms import OpenAI as LlamaIndexOpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import VectorIndexRetriever
from llama_index.postprocessor import SimilarityPostprocessor


class LlamaIndexRetriever(BaseRetriever):
    """将LlamaIndex检索器包装为LangChain检索器"""
    
    def __init__(self, llamaindex_retriever, service_context):
        self.llamaindex_retriever = llamaindex_retriever
        self.service_context = service_context
        
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """检索相关文档"""
        # 使用LlamaIndex进行检索
        nodes = self.llamaindex_retriever.retrieve(query)
        
        # 转换为LangChain Document格式
        documents = []
        for node in nodes:
            doc = Document(
                page_content=node.node.text,
                metadata={
                    "score": node.score,
                    "node_id": node.node.node_id,
                    "source": getattr(node.node, 'source', 'unknown')
                }
            )
            documents.append(doc)
            
        return documents


class HybridRAGSystem:
    """LangChain + LlamaIndex 混合RAG系统"""
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.1,
                 chunk_size: int = 512):
        """初始化混合RAG系统"""
        
        # LangChain组件配置
        self.langchain_llm = LangChainOpenAI(
            model_name=model_name,
            temperature=temperature
        )
        
        # LlamaIndex组件配置
        llamaindex_llm = LlamaIndexOpenAI(
            model=model_name,
            temperature=temperature
        )
        embed_model = OpenAIEmbedding()
        
        self.service_context = ServiceContext.from_defaults(
            llm=llamaindex_llm,
            embed_model=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=50
        )
        
        # 系统组件
        self.index = None
        self.hybrid_retriever = None
        self.qa_chain = None
        
        # 性能监控
        self.query_history = []
        
    def build_knowledge_base(self, documents_path: str) -> Dict[str, Any]:
        """构建知识库"""
        print("开始构建混合架构知识库...")
        
        # 使用LlamaIndex加载和索引文档
        documents = SimpleDirectoryReader(documents_path).load_data()
        
        # 构建向量索引
        self.index = VectorStoreIndex.from_documents(
            documents,
            service_context=self.service_context
        )
        
        # 创建混合检索器
        self._setup_hybrid_retriever()
        
        # 创建LangChain问答链
        self._setup_qa_chain()
        
        build_info = {
            "documents_count": len(documents),
            "index_built": True,
            "retriever_ready": True,
            "qa_chain_ready": True,
            "build_time": datetime.now().isoformat()
        }
        
        print(f"知识库构建完成: {build_info}")
        return build_info
        
    def _setup_hybrid_retriever(self):
        """设置混合检索器"""
        # LlamaIndex检索器配置
        llamaindex_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=5,
            similarity_cutoff=0.6
        )
        
        # 包装为LangChain检索器
        self.hybrid_retriever = LlamaIndexRetriever(
            llamaindex_retriever=llamaindex_retriever,
            service_context=self.service_context
        )
        
    def _setup_qa_chain(self):
        """设置问答链"""
        # 定义提示模板
        qa_template = """基于以下上下文信息回答问题。如果无法从上下文中找到答案，请说明无法确定。

上下文信息:
{context}

问题: {question}

请提供准确、简洁的答案:"""
        
        qa_prompt = PromptTemplate(
            template=qa_template,
            input_variables=["context", "question"]
        )
        
        # 创建问答链
        self.qa_chain = LLMChain(
            llm=self.langchain_llm,
            prompt=qa_prompt
        )
        
    def smart_query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """智能查询 - 核心混合架构方法"""
        if not self.hybrid_retriever or not self.qa_chain:
            raise ValueError("请先构建知识库")
            
        start_time = datetime.now()
        
        # 第一步：LlamaIndex智能检索
        relevant_docs = self.hybrid_retriever.get_relevant_documents(question)[:top_k]
        
        # 第二步：构建上下文
        context = self._build_context(relevant_docs)
        
        # 第三步：LangChain推理生成
        answer = self.qa_chain.run(context=context, question=question)
        
        # 第四步：结果后处理和评估
        result = self._post_process_result(
            question=question,
            answer=answer,
            docs=relevant_docs,
            processing_time=(datetime.now() - start_time).total_seconds()
        )
        
        # 记录查询历史
        self.query_history.append(result)
        
        return result
        
    def _build_context(self, docs: List[Document]) -> str:
        """构建上下文"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            score = doc.metadata.get('score', 0)
            context_parts.append(
                f"[文档{i}] (相关度: {score:.3f})\n{doc.page_content}\n"
            )
        return "\n".join(context_parts)
        
    def _post_process_result(self, question: str, answer: str, 
                           docs: List[Document], processing_time: float) -> Dict[str, Any]:
        """后处理结果"""
        # 计算置信度
        confidence = self._calculate_confidence(docs, answer)
        
        # 提取来源信息
        sources = [
            {
                "content": doc.page_content[:200] + "...",
                "score": doc.metadata.get('score', 0),
                "source": doc.metadata.get('source', 'unknown')
            }
            for doc in docs
        ]
        
        return {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
            "processing_time": round(processing_time, 3),
            "retrieved_docs_count": len(docs),
            "timestamp": datetime.now().isoformat()
        }
        
    def _calculate_confidence(self, docs: List[Document], answer: str) -> float:
        """计算回答置信度"""
        if not docs:
            return 0.0
            
        # 基于检索分数和回答长度计算置信度
        avg_score = sum(doc.metadata.get('score', 0) for doc in docs) / len(docs)
        answer_length_factor = min(len(answer) / 200, 1.0)
        docs_count_factor = min(len(docs) / 3, 1.0)
        
        confidence = (avg_score * 0.5 + 
                     answer_length_factor * 0.3 + 
                     docs_count_factor * 0.2)
        
        return round(min(confidence, 0.95), 3)
        
    async def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """批量查询"""
        tasks = [
            asyncio.create_task(self._async_query(q)) 
            for q in questions
        ]
        
        results = await asyncio.gather(*tasks)
        return results
        
    async def _async_query(self, question: str) -> Dict[str, Any]:
        """异步查询包装器"""
        return self.smart_query(question)
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.query_history:
            return {"message": "暂无查询历史"}
            
        processing_times = [q["processing_time"] for q in self.query_history]
        confidences = [q["confidence"] for q in self.query_history]
        
        return {
            "total_queries": len(self.query_history),
            "avg_processing_time": round(sum(processing_times) / len(processing_times), 3),
            "avg_confidence": round(sum(confidences) / len(confidences), 3),
            "max_processing_time": max(processing_times),
            "min_processing_time": min(processing_times),
            "recent_queries": self.query_history[-5:]  # 最近5次查询
        }
        
    def optimize_retrieval_params(self, similarity_threshold: float = 0.7, 
                                top_k: int = 5) -> bool:
        """动态优化检索参数"""
        try:
            # 重新配置LlamaIndex检索器
            llamaindex_retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k,
                similarity_cutoff=similarity_threshold
            )
            
            # 更新混合检索器
            self.hybrid_retriever = LlamaIndexRetriever(
                llamaindex_retriever=llamaindex_retriever,
                service_context=self.service_context
            )
            
            print(f"检索参数已优化: threshold={similarity_threshold}, top_k={top_k}")
            return True
            
        except Exception as e:
            print(f"参数优化失败: {e}")
            return False


# 使用示例
if __name__ == "__main__":
    # 初始化混合RAG系统
    hybrid_rag = HybridRAGSystem(
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        chunk_size=512
    )
    
    # 构建知识库（假设有documents目录）
    try:
        build_info = hybrid_rag.build_knowledge_base("./documents/")
        print(f"知识库构建成功: {build_info}")
        
        # 智能查询
        result = hybrid_rag.smart_query("什么是机器学习的监督学习？")
        
        print(f"\n问题: {result['question']}")
        print(f"答案: {result['answer']}")
        print(f"置信度: {result['confidence']}")
        print(f"处理时间: {result['processing_time']}秒")
        print(f"检索文档数: {result['retrieved_docs_count']}")
        
        # 性能统计
        stats = hybrid_rag.get_performance_stats()
        print(f"\n系统性能统计:")
        print(f"平均处理时间: {stats['avg_processing_time']}秒")
        print(f"平均置信度: {stats['avg_confidence']}")
        
    except Exception as e:
        print(f"运行出错: {e}")
        print("请确保有documents目录和正确的API配置")