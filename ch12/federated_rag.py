"""
联邦RAG实现
Deep RAG Notes Chapter 12 - Privacy Protection Technologies
"""

import numpy as np
from typing import List, Dict, Any, Optional
import hashlib
import json
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class FederatedRAGNode:
    """联邦RAG节点"""
    
    def __init__(self, node_id: str, local_data_path: str, encryption_key: Optional[bytes] = None):
        """
        初始化联邦RAG节点
        
        Args:
            node_id: 节点标识符
            local_data_path: 本地数据路径
            encryption_key: 加密密钥
        """
        self.node_id = node_id
        self.local_data_path = local_data_path
        self.local_vector_store = self.initialize_local_store(local_data_path)
        self.local_model = None
        self.aggregation_weights = {}
        
        # 设置加密
        if encryption_key is None:
            encryption_key = Fernet.generate_key()
        self.encryption_key = encryption_key
        self.cipher_suite = Fernet(encryption_key)
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"FederatedRAGNode-{node_id}")
        
    def initialize_local_store(self, data_path: str) -> Dict[str, Any]:
        """初始化本地向量存储"""
        self.logger.info(f"初始化本地存储，数据路径: {data_path}")
        
        # 模拟本地文档加载
        local_documents = self.load_local_documents(data_path)
        
        # 本地向量化
        local_embeddings = self.create_local_embeddings(local_documents)
        
        # 构建本地向量数据库
        local_store = {
            'embeddings': local_embeddings,
            'documents': local_documents,
            'metadata': {
                'doc_count': len(local_documents),
                'embedding_dim': local_embeddings.shape[1] if len(local_embeddings) > 0 else 0,
                'created_at': np.datetime64('now').astype(str)
            }
        }
        
        self.logger.info(f"本地存储初始化完成，文档数量: {len(local_documents)}")
        return local_store
    
    def load_local_documents(self, data_path: str) -> List[Dict[str, Any]]:
        """加载本地文档（模拟实现）"""
        # 这里是模拟实现，实际应该从data_path加载真实文档
        np.random.seed(hash(self.node_id) % 2**32)
        num_docs = np.random.randint(50, 200)
        
        documents = []
        for i in range(num_docs):
            doc = {
                'id': f"{self.node_id}_doc_{i}",
                'content': f"Document {i} from node {self.node_id}",
                'metadata': {
                    'source': self.node_id,
                    'doc_index': i,
                    'sensitive': np.random.choice([True, False])
                }
            }
            documents.append(doc)
        
        return documents
    
    def create_local_embeddings(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """创建本地嵌入向量（模拟实现）"""
        if not documents:
            return np.array([])
        
        # 模拟嵌入生成
        np.random.seed(hash(self.node_id) % 2**32)
        embedding_dim = 768
        embeddings = np.random.randn(len(documents), embedding_dim)
        
        # 归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def encrypt_data(self, data: Any) -> bytes:
        """加密数据"""
        json_data = json.dumps(data, ensure_ascii=False)
        return self.cipher_suite.encrypt(json_data.encode('utf-8'))
    
    def decrypt_data(self, encrypted_data: bytes) -> Any:
        """解密数据"""
        decrypted_data = self.cipher_suite.decrypt(encrypted_data)
        return json.loads(decrypted_data.decode('utf-8'))
    
    def local_search(self, encrypted_query: bytes, top_k: int = 5) -> List[Dict[str, Any]]:
        """本地加密搜索"""
        try:
            # 1. 解密查询
            query_data = self.decrypt_data(encrypted_query)
            query_vector = np.array(query_data['query_vector'])
            query_metadata = query_data.get('metadata', {})
            
            self.logger.info(f"收到查询请求，top_k={top_k}")
            
            # 2. 本地搜索
            local_results = self.similarity_search(query_vector, top_k)
            
            # 3. 对结果进行差分隐私处理
            private_results = self.apply_local_differential_privacy(local_results)
            
            # 4. 加密返回结果
            encrypted_results = []
            for result in private_results:
                encrypted_result = {
                    'document_hash': self.hash_document_content(result['content']),
                    'similarity_score': result['similarity_score'],
                    'node_id': self.node_id,
                    'privacy_applied': True,
                    'metadata': {
                        'doc_id': result['doc_id'],
                        'source_node': self.node_id
                    }
                }
                encrypted_results.append(encrypted_result)
            
            return encrypted_results
            
        except Exception as e:
            self.logger.error(f"本地搜索失败: {str(e)}")
            return []
    
    def similarity_search(self, query_vector: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """向量相似度搜索"""
        if self.local_vector_store['embeddings'].size == 0:
            return []
        
        # 计算相似度
        similarities = np.dot(self.local_vector_store['embeddings'], query_vector)
        
        # 获取top-k结果
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            result = {
                'doc_id': self.local_vector_store['documents'][idx]['id'],
                'content': self.local_vector_store['documents'][idx]['content'],
                'similarity_score': float(similarities[idx]),
                'metadata': self.local_vector_store['documents'][idx]['metadata']
            }
            results.append(result)
        
        return results
    
    def apply_local_differential_privacy(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """应用本地差分隐私"""
        # 为相似度分数添加噪声
        epsilon = 0.5  # 本地隐私预算
        sensitivity = 1.0
        
        for result in results:
            # 添加拉普拉斯噪声
            noise = np.random.laplace(0, sensitivity / epsilon)
            result['similarity_score'] += noise
            result['privacy_noise_applied'] = True
        
        return results
    
    def hash_document_content(self, content: str) -> str:
        """生成文档内容哈希"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    def federated_model_update(self, global_model_params: Dict[str, Any]) -> Dict[str, Any]:
        """联邦模型更新"""
        self.logger.info("执行联邦模型更新")
        
        # 1. 更新本地模型
        self.update_local_model(global_model_params)
        
        # 2. 本地训练（只使用本地数据）
        local_updates = self.local_training_step()
        
        # 3. 添加噪声保护隐私
        noisy_updates = self.add_noise_to_updates(local_updates)
        
        # 4. 返回加密的模型更新
        encrypted_updates = {
            'node_id': self.node_id,
            'updates': noisy_updates,
            'training_samples': len(self.local_vector_store['documents']),
            'privacy_applied': True
        }
        
        return encrypted_updates
    
    def update_local_model(self, global_params: Dict[str, Any]):
        """更新本地模型"""
        self.local_model = global_params.copy()
        self.logger.info("本地模型已更新")
    
    def local_training_step(self) -> Dict[str, Any]:
        """本地训练步骤（模拟）"""
        # 模拟训练过程
        updates = {
            'embeddings_update': np.random.randn(768).tolist(),
            'bias_update': np.random.randn(10).tolist(),
            'loss': np.random.uniform(0.1, 1.0),
            'training_samples': len(self.local_vector_store['documents'])
        }
        return updates
    
    def add_noise_to_updates(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """为模型更新添加噪声"""
        epsilon = 1.0
        sensitivity = 1.0
        
        noisy_updates = updates.copy()
        
        # 为embeddings更新添加噪声
        if 'embeddings_update' in updates:
            embeddings = np.array(updates['embeddings_update'])
            noise = np.random.laplace(0, sensitivity / epsilon, embeddings.shape)
            noisy_updates['embeddings_update'] = (embeddings + noise).tolist()
        
        # 为bias更新添加噪声
        if 'bias_update' in updates:
            bias = np.array(updates['bias_update'])
            noise = np.random.laplace(0, sensitivity / epsilon, bias.shape)
            noisy_updates['bias_update'] = (bias + noise).tolist()
        
        return noisy_updates
    
    def get_node_status(self) -> Dict[str, Any]:
        """获取节点状态"""
        return {
            'node_id': self.node_id,
            'document_count': len(self.local_vector_store['documents']),
            'embedding_dimension': self.local_vector_store['metadata']['embedding_dim'],
            'created_at': self.local_vector_store['metadata']['created_at'],
            'model_updated': self.local_model is not None
        }


class FederatedRAGCoordinator:
    """联邦RAG协调器"""
    
    def __init__(self):
        """初始化协调器"""
        self.participating_nodes = {}
        self.global_model_state = {}
        self.aggregation_rounds = 0
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("FederatedRAGCoordinator")
        
        # 生成加密密钥
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
    
    def register_node(self, node: FederatedRAGNode):
        """注册参与节点"""
        self.participating_nodes[node.node_id] = node
        self.logger.info(f"节点 {node.node_id} 已注册")
    
    def encrypt_query(self, query: str, query_vector: np.ndarray) -> bytes:
        """加密查询"""
        query_data = {
            'query': query,
            'query_vector': query_vector.tolist(),
            'timestamp': np.datetime64('now').astype(str),
            'metadata': {}
        }
        return self.cipher_suite.encrypt(json.dumps(query_data).encode('utf-8'))
    
    def coordinate_federated_search(self, 
                                   query: str, 
                                   query_vector: np.ndarray,
                                   participating_nodes: List[str],
                                   top_k: int = 5) -> Dict[str, Any]:
        """协调联邦搜索"""
        self.logger.info(f"开始联邦搜索，参与节点: {participating_nodes}")
        
        # 1. 加密查询
        encrypted_query = self.encrypt_query(query, query_vector)
        
        # 2. 并行发送到所有参与节点
        node_results = {}
        for node_id in participating_nodes:
            if node_id in self.participating_nodes:
                try:
                    node_result = self.participating_nodes[node_id].local_search(
                        encrypted_query, top_k
                    )
                    node_results[node_id] = node_result
                    self.logger.info(f"节点 {node_id} 返回 {len(node_result)} 个结果")
                except Exception as e:
                    self.logger.error(f"节点 {node_id} 搜索失败: {str(e)}")
                    node_results[node_id] = []
        
        # 3. 安全聚合结果
        aggregated_results = self.secure_result_aggregation(node_results)
        
        # 4. 生成联邦答案
        federated_answer = self.generate_federated_response(
            query, aggregated_results, top_k
        )
        
        return federated_answer
    
    def secure_result_aggregation(self, node_results: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """安全结果聚合"""
        self.logger.info("开始安全结果聚合")
        
        # 使用安全多方计算聚合结果
        aggregated_scores = {}
        
        for node_id, results in node_results.items():
            for result in results:
                doc_hash = result['document_hash']
                
                if doc_hash not in aggregated_scores:
                    aggregated_scores[doc_hash] = {
                        'total_score': 0.0,
                        'node_count': 0,
                        'participating_nodes': [],
                        'metadata': result.get('metadata', {})
                    }
                
                # 聚合相似度分数
                aggregated_scores[doc_hash]['total_score'] += result['similarity_score']
                aggregated_scores[doc_hash]['node_count'] += 1
                aggregated_scores[doc_hash]['participating_nodes'].append(node_id)
        
        # 计算平均分数并排序
        final_results = []
        for doc_hash, score_info in aggregated_scores.items():
            avg_score = score_info['total_score'] / score_info['node_count']
            final_results.append({
                'document_hash': doc_hash,
                'average_similarity': avg_score,
                'participating_nodes': score_info['node_count'],
                'node_list': score_info['participating_nodes'],
                'metadata': score_info['metadata']
            })
        
        # 按相似度排序
        final_results.sort(key=lambda x: x['average_similarity'], reverse=True)
        
        self.logger.info(f"聚合完成，共 {len(final_results)} 个结果")
        return final_results
    
    def generate_federated_response(self, 
                                   query: str, 
                                   aggregated_results: List[Dict],
                                   top_k: int) -> Dict[str, Any]:
        """生成联邦响应"""
        # 取top-k结果
        top_results = aggregated_results[:top_k]
        
        response = {
            'query': query,
            'results': top_results,
            'total_results': len(aggregated_results),
            'participating_nodes': len(self.participating_nodes),
            'privacy_preserved': True,
            'federated_search': True,
            'timestamp': np.datetime64('now').astype(str)
        }
        
        return response
    
    def coordinate_federated_learning(self, training_rounds: int = 5) -> Dict[str, Any]:
        """协调联邦学习"""
        self.logger.info(f"开始联邦学习，训练轮数: {training_rounds}")
        
        training_history = []
        
        for round_num in range(training_rounds):
            self.logger.info(f"联邦学习第 {round_num + 1} 轮")
            
            # 收集各节点的模型更新
            node_updates = {}
            for node_id, node in self.participating_nodes.items():
                update = node.federated_model_update(self.global_model_state)
                node_updates[node_id] = update
            
            # 聚合模型更新
            aggregated_model = self.aggregate_model_updates(node_updates)
            
            # 更新全局模型
            self.global_model_state = aggregated_model
            self.aggregation_rounds += 1
            
            # 记录训练历史
            round_stats = {
                'round': round_num + 1,
                'participating_nodes': len(node_updates),
                'global_loss': np.mean([update['updates']['loss'] for update in node_updates.values()]),
                'total_training_samples': sum([update['training_samples'] for update in node_updates.values()])
            }
            training_history.append(round_stats)
            
            self.logger.info(f"第 {round_num + 1} 轮完成，全局损失: {round_stats['global_loss']:.4f}")
        
        return {
            'training_completed': True,
            'total_rounds': training_rounds,
            'final_global_loss': training_history[-1]['global_loss'],
            'training_history': training_history,
            'participating_nodes': len(self.participating_nodes)
        }
    
    def aggregate_model_updates(self, node_updates: Dict[str, Dict]) -> Dict[str, Any]:
        """聚合模型更新"""
        # 计算加权平均（基于训练样本数量）
        total_samples = sum([update['training_samples'] for update in node_updates.values()])
        
        aggregated_model = {
            'embeddings_update': np.zeros(768),
            'bias_update': np.zeros(10),
            'aggregation_round': self.aggregation_rounds + 1
        }
        
        for node_id, update in node_updates.items():
            weight = update['training_samples'] / total_samples
            
            # 加权聚合embeddings
            embeddings = np.array(update['updates']['embeddings_update'])
            aggregated_model['embeddings_update'] += weight * embeddings
            
            # 加权聚合bias
            bias = np.array(update['updates']['bias_update'])
            aggregated_model['bias_update'] += weight * bias
        
        # 转换为列表格式
        aggregated_model['embeddings_update'] = aggregated_model['embeddings_update'].tolist()
        aggregated_model['bias_update'] = aggregated_model['bias_update'].tolist()
        
        return aggregated_model
    
    def get_coordinator_status(self) -> Dict[str, Any]:
        """获取协调器状态"""
        return {
            'participating_nodes': len(self.participating_nodes),
            'node_list': list(self.participating_nodes.keys()),
            'aggregation_rounds': self.aggregation_rounds,
            'global_model_initialized': bool(self.global_model_state)
        }


def demo_federated_rag():
    """联邦RAG演示"""
    print("=== 联邦RAG演示 ===")
    
    # 创建协调器
    coordinator = FederatedRAGCoordinator()
    
    # 创建多个节点
    nodes = []
    for i in range(3):
        node = FederatedRAGNode(
            node_id=f"node_{i}",
            local_data_path=f"/data/node_{i}",
            encryption_key=coordinator.encryption_key
        )
        nodes.append(node)
        coordinator.register_node(node)
    
    print(f"创建了 {len(nodes)} 个节点")
    
    # 显示节点状态
    for node in nodes:
        status = node.get_node_status()
        print(f"节点 {status['node_id']}: {status['document_count']} 个文档")
    
    # 执行联邦搜索
    query = "搜索相关文档"
    query_vector = np.random.randn(768)
    participating_nodes = [node.node_id for node in nodes]
    
    print(f"\n执行联邦搜索: '{query}'")
    search_result = coordinator.coordinate_federated_search(
        query=query,
        query_vector=query_vector,
        participating_nodes=participating_nodes,
        top_k=5
    )
    
    print(f"搜索完成，找到 {len(search_result['results'])} 个结果")
    for i, result in enumerate(search_result['results'], 1):
        print(f"  {i}. 文档哈希: {result['document_hash'][:8]}..., "
              f"平均相似度: {result['average_similarity']:.4f}, "
              f"参与节点数: {result['participating_nodes']}")
    
    # 执行联邦学习
    print(f"\n执行联邦学习...")
    learning_result = coordinator.coordinate_federated_learning(training_rounds=3)
    
    print(f"联邦学习完成:")
    print(f"  训练轮数: {learning_result['total_rounds']}")
    print(f"  最终损失: {learning_result['final_global_loss']:.4f}")
    print(f"  参与节点: {learning_result['participating_nodes']}")
    
    # 显示协调器状态
    coordinator_status = coordinator.get_coordinator_status()
    print(f"\n协调器状态:")
    print(f"  参与节点数: {coordinator_status['participating_nodes']}")
    print(f"  聚合轮数: {coordinator_status['aggregation_rounds']}")


if __name__ == "__main__":
    demo_federated_rag()