"""
RAPTOR：递归抽象处理技术
构建分层文档树结构，实现多层次信息检索和抽象
"""

import numpy as np
import uuid
import time
from typing import Dict, List, Optional, Tuple, Union
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
from datetime import datetime

@dataclass
class RAPTORNode:
    """RAPTOR树节点"""
    id: str
    content: str
    summary: str
    embedding: np.ndarray
    level: int
    parent: Optional[str]
    children: List[str]
    metadata: Dict

class RAPTORTree:
    """RAPTOR树构建和检索系统"""
    
    def __init__(self, llm, embedding_model, max_cluster_size: int = 10):
        if llm is None:
            raise ValueError("LLM实例不能为空")
        if embedding_model is None:
            raise ValueError("嵌入模型不能为空")
        if max_cluster_size <= 0:
            raise ValueError("最大聚类大小必须大于0")
            
        self.llm = llm
        self.embedding_model = embedding_model
        self.max_cluster_size = max_cluster_size
        
        # 树结构存储
        self.nodes = {}  # node_id -> RAPTORNode
        self.root_nodes = []  # 根节点列表
        
        # 构建参数
        self.similarity_threshold = 0.7
        self.max_levels = 5
        
    def build_raptor_tree(self, documents: List[Dict]) -> None:
        """构建RAPTOR树"""
        
        if not documents:
            raise ValueError("文档列表不能为空")
            
        print(f"开始构建RAPTOR树，文档数量: {len(documents)}")
        
        try:
            # 1. 文档分块并创建叶子节点
            leaf_nodes = self._create_leaf_nodes(documents)
            
            if not leaf_nodes:
                print("警告: 没有创建任何叶子节点")
                return
            
            # 2. 递归构建树结构
            current_level_nodes = leaf_nodes
            level = 0
            
            while len(current_level_nodes) > 1 and level < self.max_levels:
                print(f"构建第 {level + 1} 层，节点数量: {len(current_level_nodes)}")
                
                # 聚类当前层的节点
                clusters = self._cluster_nodes(current_level_nodes)
                
                if not clusters:
                    print("聚类失败，停止构建")
                    break
                
                # 为每个聚类创建父节点
                next_level_nodes = []
                for cluster in clusters:
                    if len(cluster) > 1:  # 只有多个子节点才创建父节点
                        parent_node = self._create_parent_node(cluster, level + 1)
                        next_level_nodes.append(parent_node)
                        
                        # 更新子节点的父节点引用
                        for child_node in cluster:
                            child_node.parent = parent_node.id
                            self.nodes[child_node.id] = child_node
                    else:
                        # 单个节点直接提升到下一层
                        cluster[0].level = level + 1
                        next_level_nodes.append(cluster[0])
                
                current_level_nodes = next_level_nodes
                level += 1
            
            # 3. 设置根节点
            self.root_nodes = [node.id for node in current_level_nodes]
            
            # 4. 保存所有节点
            for node in current_level_nodes:
                self.nodes[node.id] = node
            
            print(f"RAPTOR树构建完成，共 {level + 1} 层，根节点数量: {len(self.root_nodes)}")
            
        except Exception as e:
            print(f"构建RAPTOR树时发生错误: {e}")
            raise
    
    def _create_leaf_nodes(self, documents: List[Dict]) -> List[RAPTORNode]:
        """创建叶子节点"""
        
        leaf_nodes = []
        
        for doc in documents:
            if 'content' not in doc:
                print(f"警告: 文档缺少content字段，跳过")
                continue
                
            try:
                # 分块处理
                chunks = self._chunk_document(doc['content'])
                
                for i, chunk in enumerate(chunks):
                    node_id = str(uuid.uuid4())
                    
                    # 生成嵌入
                    embedding = self.embedding_model.encode(chunk)
                    
                    # 生成简要摘要
                    summary = self._generate_summary(chunk, max_length=100)
                    
                    node = RAPTORNode(
                        id=node_id,
                        content=chunk,
                        summary=summary,
                        embedding=embedding,
                        level=0,
                        parent=None,
                        children=[],
                        metadata={
                            'doc_id': doc.get('id', f"doc_{hash(doc['content'])}"),
                            'chunk_index': i,
                            'original_doc_title': doc.get('title', ''),
                            'source': doc.get('source', ''),
                            'created_at': datetime.now().isoformat()
                        }
                    )
                    
                    leaf_nodes.append(node)
                    self.nodes[node_id] = node
                    
            except Exception as e:
                print(f"处理文档时发生错误: {e}")
                continue
        
        return leaf_nodes
    
    def _chunk_document(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """文档分块"""
        
        if not text or not text.strip():
            return []
        
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
        
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def _cluster_nodes(self, nodes: List[RAPTORNode]) -> List[List[RAPTORNode]]:
        """聚类节点"""
        
        if len(nodes) <= self.max_cluster_size:
            return [nodes]
        
        try:
            # 提取嵌入向量
            embeddings = np.array([node.embedding for node in nodes])
            
            # 计算聚类数量
            n_clusters = min(
                max(2, len(nodes) // self.max_cluster_size),
                len(nodes)
            )
            
            # K-means聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # 组织聚类结果
            clusters = [[] for _ in range(n_clusters)]
            for node, label in zip(nodes, cluster_labels):
                clusters[label].append(node)
            
            # 过滤空聚类
            return [cluster for cluster in clusters if cluster]
            
        except Exception as e:
            print(f"聚类节点时发生错误: {e}")
            # 备用方案：简单分组
            return [nodes[i:i+self.max_cluster_size] for i in range(0, len(nodes), self.max_cluster_size)]
    
    def _create_parent_node(self, child_nodes: List[RAPTORNode], level: int) -> RAPTORNode:
        """创建父节点"""
        
        node_id = str(uuid.uuid4())
        
        try:
            # 合并子节点内容
            combined_content = self._combine_child_contents(child_nodes)
            
            # 生成父节点摘要
            parent_summary = self._generate_parent_summary(child_nodes)
            
            # 计算父节点嵌入（子节点嵌入的均值）
            child_embeddings = np.array([node.embedding for node in child_nodes])
            parent_embedding = np.mean(child_embeddings, axis=0)
            
            # 合并元数据
            merged_metadata = self._merge_metadata(child_nodes)
            
            parent_node = RAPTORNode(
                id=node_id,
                content=combined_content,
                summary=parent_summary,
                embedding=parent_embedding,
                level=level,
                parent=None,
                children=[node.id for node in child_nodes],
                metadata=merged_metadata
            )
            
            return parent_node
            
        except Exception as e:
            print(f"创建父节点时发生错误: {e}")
            # 创建一个简化的父节点
            return RAPTORNode(
                id=node_id,
                content="合并内容",
                summary="合并摘要",
                embedding=np.random.randn(768),  # 使用随机嵌入
                level=level,
                parent=None,
                children=[node.id for node in child_nodes],
                metadata={"error": "父节点创建失败"}
            )
    
    def _combine_child_contents(self, child_nodes: List[RAPTORNode]) -> str:
        """合并子节点内容"""
        
        # 使用摘要而不是完整内容，以控制长度
        summaries = [node.summary for node in child_nodes if node.summary]
        return "\n\n".join(summaries)
    
    def _generate_parent_summary(self, child_nodes: List[RAPTORNode]) -> str:
        """生成父节点摘要"""
        
        try:
            child_summaries = [node.summary for node in child_nodes if node.summary]
            if not child_summaries:
                return "无子节点摘要信息"
            
            combined_summaries = "\n".join(child_summaries)
            
            summary_prompt = f"""
基于以下子节点摘要，生成一个更高层次的综合摘要：

子节点摘要：
{combined_summaries}

请生成一个200字左右的综合摘要，突出主要主题和关键信息：
"""
            
            summary = self.llm.generate(summary_prompt)
            return summary.strip() if summary else "摘要生成失败"
            
        except Exception as e:
            return f"摘要生成错误: {str(e)}"
    
    def _generate_summary(self, content: str, max_length: int = 150) -> str:
        """生成内容摘要"""
        
        if not content:
            return ""
        
        if len(content) <= max_length:
            return content
        
        try:
            summary_prompt = f"""
请为以下内容生成一个简洁的摘要（约{max_length}字）：

内容：{content}

摘要：
"""
            
            summary = self.llm.generate(summary_prompt)
            return summary.strip() if summary else content[:max_length]
            
        except Exception as e:
            return content[:max_length]
    
    def _merge_metadata(self, child_nodes: List[RAPTORNode]) -> Dict:
        """合并元数据"""
        
        merged = {
            'child_count': len(child_nodes),
            'doc_ids': list(set(node.metadata.get('doc_id', '') for node in child_nodes if node.metadata.get('doc_id'))),
            'sources': list(set(node.metadata.get('source', '') for node in child_nodes if node.metadata.get('source'))),
            'level_summary': f"Merged from {len(child_nodes)} child nodes",
            'created_at': datetime.now().isoformat()
        }
        
        return merged
    
    def raptor_retrieval(self, query: str, top_k: int = 5, 
                        traverse_strategy: str = 'tree_traversal') -> List[Dict]:
        """RAPTOR检索"""
        
        if not query or not query.strip():
            raise ValueError("查询不能为空")
        
        if not self.nodes:
            print("警告: RAPTOR树为空，无法进行检索")
            return []
        
        try:
            query_embedding = self.embedding_model.encode(query)
            
            if traverse_strategy == 'tree_traversal':
                return self._tree_traversal_retrieval(query_embedding, query, top_k)
            elif traverse_strategy == 'layer_wise':
                return self._layer_wise_retrieval(query_embedding, query, top_k)
            else:
                return self._collapsed_tree_retrieval(query_embedding, query, top_k)
                
        except Exception as e:
            print(f"RAPTOR检索时发生错误: {e}")
            return []
    
    def _tree_traversal_retrieval(self, query_embedding: np.ndarray, 
                                 query: str, top_k: int) -> List[Dict]:
        """树遍历检索"""
        
        retrieved_nodes = []
        
        # 从根节点开始遍历
        for root_id in self.root_nodes:
            if root_id in self.nodes:
                path_nodes = self._traverse_from_root(
                    root_id, query_embedding, max_depth=3
                )
                retrieved_nodes.extend(path_nodes)
        
        # 按相似度排序
        retrieved_nodes.sort(key=lambda x: x['similarity'], reverse=True)
        
        return retrieved_nodes[:top_k]
    
    def _traverse_from_root(self, node_id: str, query_embedding: np.ndarray, 
                           max_depth: int) -> List[Dict]:
        """从根节点遍历"""
        
        if max_depth <= 0 or node_id not in self.nodes:
            return []
        
        node = self.nodes[node_id]
        
        # 计算相似度
        similarity = self._calculate_similarity(query_embedding, node.embedding)
        
        result = [{
            'node_id': node_id,
            'content': node.content,
            'summary': node.summary,
            'level': node.level,
            'similarity': similarity,
            'metadata': node.metadata
        }]
        
        # 递归遍历子节点
        for child_id in node.children:
            if child_id in self.nodes:
                child_results = self._traverse_from_root(
                    child_id, query_embedding, max_depth - 1
                )
                result.extend(child_results)
        
        return result
    
    def _layer_wise_retrieval(self, query_embedding: np.ndarray, 
                             query: str, top_k: int) -> List[Dict]:
        """分层检索"""
        
        all_similarities = []
        
        # 计算所有节点的相似度
        for node_id, node in self.nodes.items():
            similarity = self._calculate_similarity(query_embedding, node.embedding)
            all_similarities.append({
                'node_id': node_id,
                'node': node,
                'similarity': similarity
            })
        
        # 按相似度排序
        all_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 选择不同层级的节点
        selected_nodes = []
        used_content = set()
        
        for item in all_similarities:
            node = item['node']
            
            # 避免重复内容
            content_hash = hash(node.content)
            if content_hash not in used_content:
                selected_nodes.append({
                    'node_id': item['node_id'],
                    'content': node.content,
                    'summary': node.summary,
                    'level': node.level,
                    'similarity': item['similarity'],
                    'metadata': node.metadata
                })
                used_content.add(content_hash)
                
                if len(selected_nodes) >= top_k:
                    break
        
        return selected_nodes
    
    def _collapsed_tree_retrieval(self, query_embedding: np.ndarray, 
                                 query: str, top_k: int) -> List[Dict]:
        """扁平化树检索"""
        return self._layer_wise_retrieval(query_embedding, query, top_k)
    
    def _calculate_similarity(self, embedding1: np.ndarray, 
                             embedding2: np.ndarray) -> float:
        """计算嵌入相似度"""
        
        try:
            # 余弦相似度
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
            
        except Exception as e:
            print(f"计算相似度时发生错误: {e}")
            return 0.0
    
    def generate_raptor_answer(self, query: str) -> Dict:
        """生成RAPTOR增强答案"""
        
        try:
            # 1. 多策略检索
            tree_results = self.raptor_retrieval(query, top_k=3, traverse_strategy='tree_traversal')
            layer_results = self.raptor_retrieval(query, top_k=3, traverse_strategy='layer_wise')
            
            # 2. 合并和去重
            all_results = tree_results + layer_results
            unique_results = self._deduplicate_results(all_results)
            
            # 3. 构建分层上下文
            context_prompt = self._build_raptor_context_prompt(query, unique_results)
            
            # 4. 生成答案
            answer = self.llm.generate(context_prompt)
            
            return {
                'answer': answer,
                'retrieved_nodes': unique_results,
                'tree_structure_used': True,
                'retrieval_strategy': 'multi_level_raptor',
                'confidence_score': self._calculate_raptor_confidence(unique_results)
            }
            
        except Exception as e:
            return {
                'answer': f"抱歉，处理您的问题时遇到错误：{str(e)}",
                'retrieved_nodes': [],
                'tree_structure_used': False,
                'retrieval_strategy': 'error',
                'confidence_score': 0.0
            }
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """去重结果"""
        
        seen_content = set()
        unique_results = []
        
        # 按相似度排序
        results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        for result in results:
            content = result.get('content', '')
            content_hash = hash(content)
            if content_hash not in seen_content and content:
                unique_results.append(result)
                seen_content.add(content_hash)
        
        return unique_results[:5]  # 限制为前5个
    
    def _build_raptor_context_prompt(self, query: str, results: List[Dict]) -> str:
        """构建RAPTOR上下文提示"""
        
        if not results:
            return f"问题：{query}\n\n没有找到相关信息，请基于一般知识回答。"
        
        # 按层级组织上下文
        context_by_level = {}
        for result in results:
            level = result.get('level', 0)
            if level not in context_by_level:
                context_by_level[level] = []
            context_by_level[level].append(result)
        
        prompt = f"""
基于以下分层信息结构回答问题：

问题：{query}

"""
        
        # 从高层到低层添加上下文
        for level in sorted(context_by_level.keys(), reverse=True):
            level_results = context_by_level[level]
            prompt += f"\n第{level}层信息（概括性）：\n"
            
            for i, result in enumerate(level_results):
                summary = result.get('summary', '无摘要')
                prompt += f"  {i+1}. {summary}\n"
        
        prompt += "\n详细信息：\n"
        for i, result in enumerate(results[:3]):  # 只显示前3个详细内容
            content = result.get('content', '无内容')
            prompt += f"{i+1}. {content}\n\n"
        
        prompt += "请基于以上分层信息提供准确、全面的答案：\n"
        
        return prompt
    
    def _calculate_raptor_confidence(self, results: List[Dict]) -> float:
        """计算RAPTOR置信度"""
        
        if not results:
            return 0.0
        
        try:
            # 基于相似度和层级多样性计算置信度
            similarities = [r.get('similarity', 0) for r in results]
            avg_similarity = np.mean(similarities) if similarities else 0
            
            levels = [r.get('level', 0) for r in results]
            max_level = max(levels) if levels else 0
            level_diversity = len(set(levels)) / (max_level + 1) if max_level > 0 else 0
            
            confidence = (avg_similarity * 0.7 + level_diversity * 0.3)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            print(f"计算置信度时发生错误: {e}")
            return 0.5


# 模拟LLM类
class MockLLM:
    def generate(self, prompt: str) -> str:
        """模拟LLM生成"""
        if "生成一个简洁的摘要" in prompt:
            content = prompt.split("内容：")[1].split("摘要：")[0].strip()
            return f"摘要: {content[:100]}..." if len(content) > 100 else content
        elif "生成一个更高层次的综合摘要" in prompt:
            return "这是一个综合性的摘要，整合了多个子节点的关键信息和主要主题。"
        else:
            return f"基于RAPTOR分层信息的智能回答: {prompt.split('问题：')[1].split('第')[0].strip() if '问题：' in prompt else '通用回答'}"


# 模拟嵌入模型
class MockEmbeddingModel:
    def encode(self, text: str) -> np.ndarray:
        """模拟嵌入编码"""
        # 基于文本内容生成相对稳定的嵌入向量
        hash_value = hash(text) % (2**31)
        np.random.seed(hash_value)
        return np.random.randn(768)


def main():
    """测试RAPTOR树系统"""
    print("=== RAPTOR 树系统测试 ===")
    
    # 初始化组件
    llm = MockLLM()
    embedding_model = MockEmbeddingModel()
    
    # 创建RAPTOR树
    raptor_tree = RAPTORTree(llm, embedding_model, max_cluster_size=5)
    
    # 测试文档
    documents = [
        {
            "id": "doc1",
            "title": "人工智能基础",
            "content": "人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。机器学习是人工智能的核心技术，通过算法让计算机从数据中学习模式。深度学习则是机器学习的一个子领域，使用多层神经网络进行复杂的模式识别。",
            "source": "AI教程"
        },
        {
            "id": "doc2",
            "title": "机器学习算法",
            "content": "机器学习包含多种算法类型。监督学习使用标记数据训练模型，如分类和回归任务。无监督学习从无标记数据中发现隐藏模式，如聚类和降维。强化学习通过与环境交互来学习最优策略，广泛应用于游戏和控制系统。",
            "source": "技术文档"
        },
        {
            "id": "doc3",
            "title": "深度学习应用",
            "content": "深度学习在多个领域取得了突破性进展。在计算机视觉中，卷积神经网络能够识别图像中的对象和场景。在自然语言处理中，Transformer架构革命了文本理解和生成。在语音识别中，深度神经网络实现了接近人类水平的准确率。",
            "source": "应用案例"
        }
    ]
    
    # 构建RAPTOR树
    print("\n1. 构建RAPTOR树")
    raptor_tree.build_raptor_tree(documents)
    
    print(f"树节点总数: {len(raptor_tree.nodes)}")
    print(f"根节点数: {len(raptor_tree.root_nodes)}")
    
    # 显示树结构
    print("\n2. 树结构概览")
    level_counts = {}
    for node in raptor_tree.nodes.values():
        level = node.level
        level_counts[level] = level_counts.get(level, 0) + 1
    
    for level in sorted(level_counts.keys()):
        print(f"第{level}层: {level_counts[level]}个节点")
    
    # 测试检索
    print("\n3. 测试RAPTOR检索")
    query = "深度学习在计算机视觉中的应用"
    
    # 测试不同检索策略
    strategies = ['tree_traversal', 'layer_wise']
    
    for strategy in strategies:
        print(f"\n{strategy} 检索结果:")
        results = raptor_tree.raptor_retrieval(query, top_k=3, traverse_strategy=strategy)
        
        for i, result in enumerate(results):
            print(f"  {i+1}. 层级{result['level']}, 相似度{result['similarity']:.3f}")
            print(f"     摘要: {result['summary'][:100]}...")
    
    # 生成最终答案
    print("\n4. 生成RAPTOR增强答案")
    answer_result = raptor_tree.generate_raptor_answer(query)
    
    print(f"查询: {query}")
    print(f"答案: {answer_result['answer']}")
    print(f"使用策略: {answer_result['retrieval_strategy']}")
    print(f"置信度: {answer_result['confidence_score']:.3f}")
    print(f"检索节点数: {len(answer_result['retrieved_nodes'])}")


if __name__ == "__main__":
    main()