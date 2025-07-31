"""
同态加密RAG实现
Deep RAG Notes Chapter 12 - Privacy Protection Technologies

注意：这是一个简化的同态加密实现，主要用于演示概念。
实际生产环境中建议使用专业的同态加密库如Microsoft SEAL、HElib等。
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import time
from dataclasses import dataclass

@dataclass
class EncryptionContext:
    """加密上下文"""
    polynomial_degree: int
    coefficient_modulus: List[int]  
    plain_modulus: int
    scale: float
    security_level: int

class SimpleHomomorphicEncryption:
    """简化的同态加密实现（仅用于演示）"""
    
    def __init__(self, context: EncryptionContext):
        """
        初始化同态加密系统
        
        Args:
            context: 加密上下文参数
        """
        self.context = context
        self.public_key = None
        self.private_key = None
        self.relin_keys = None
        self.galois_keys = None
        
        # 生成密钥
        self.generate_keys()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def generate_keys(self):
        """生成加密密钥（简化实现）"""
        # 这是一个极度简化的密钥生成过程
        # 实际的同态加密需要复杂的数学运算
        np.random.seed(42)  # 为了演示的可重复性
        
        # 生成私钥（简化）
        self.private_key = np.random.randint(
            0, self.context.plain_modulus, 
            self.context.polynomial_degree
        )
        
        # 生成公钥（简化）
        self.public_key = {
            'key0': np.random.randint(0, self.context.plain_modulus, self.context.polynomial_degree),
            'key1': np.random.randint(0, self.context.plain_modulus, self.context.polynomial_degree)
        }
        
        # 生成重线性化密钥（简化）
        self.relin_keys = np.random.randint(0, self.context.plain_modulus, (2, self.context.polynomial_degree))
        
        # 生成伽罗瓦密钥（用于旋转操作，简化）
        self.galois_keys = {}
        for i in [1, 2, 4, 8, 16]:  # 常用的旋转步长
            self.galois_keys[i] = np.random.randint(0, self.context.plain_modulus, self.context.polynomial_degree)
        
        self.logger.info("同态加密密钥生成完成")
    
    def encrypt_vector(self, plaintext_vector: np.ndarray) -> Dict[str, Any]:
        """
        加密向量
        
        Args:
            plaintext_vector: 明文向量
            
        Returns:
            加密后的向量
        """
        if len(plaintext_vector) > self.context.polynomial_degree:
            raise ValueError(f"向量长度 {len(plaintext_vector)} 超过多项式度数 {self.context.polynomial_degree}")
        
        # 填充向量到多项式度数
        padded_vector = np.zeros(self.context.polynomial_degree)
        padded_vector[:len(plaintext_vector)] = plaintext_vector
        
        # 简化的加密过程：添加噪声并使用公钥
        noise = np.random.normal(0, 0.1, self.context.polynomial_degree)
        
        # 加密（简化实现）
        encrypted_data = {
            'data0': (padded_vector * self.context.scale + noise + self.public_key['key0']) % self.context.plain_modulus,
            'data1': (self.public_key['key1'] + noise) % self.context.plain_modulus,
            'scale': self.context.scale,
            'size': len(plaintext_vector),
            'is_encrypted': True
        }
        
        return encrypted_data
    
    def decrypt_vector(self, encrypted_vector: Dict[str, Any]) -> np.ndarray:
        """
        解密向量
        
        Args:
            encrypted_vector: 加密的向量
            
        Returns:
            解密后的向量
        """
        if not encrypted_vector.get('is_encrypted', False):
            raise ValueError("输入不是有效的加密向量")
        
        # 简化的解密过程
        decrypted_data = (encrypted_vector['data0'] - encrypted_vector['data1']) % self.context.plain_modulus
        
        # 还原比例
        decrypted_data = decrypted_data / encrypted_vector['scale']
        
        # 取出原始长度
        original_size = encrypted_vector['size']
        return decrypted_data[:original_size]
    
    def add_encrypted_vectors(self, vec1: Dict[str, Any], vec2: Dict[str, Any]) -> Dict[str, Any]:
        """
        加密状态下的向量加法
        
        Args:
            vec1: 加密向量1
            vec2: 加密向量2
            
        Returns:
            加法结果（加密状态）
        """
        if not (vec1.get('is_encrypted') and vec2.get('is_encrypted')):
            raise ValueError("输入向量必须都是加密状态")
        
        # 同态加法：密文直接相加
        result = {
            'data0': (vec1['data0'] + vec2['data0']) % self.context.plain_modulus,
            'data1': (vec1['data1'] + vec2['data1']) % self.context.plain_modulus,
            'scale': vec1['scale'],  # 假设比例相同
            'size': min(vec1['size'], vec2['size']),
            'is_encrypted': True
        }
        
        return result
    
    def multiply_encrypted_vectors(self, vec1: Dict[str, Any], vec2: Dict[str, Any]) -> Dict[str, Any]:
        """
        加密状态下的向量乘法（简化实现）
        
        Args:
            vec1: 加密向量1  
            vec2: 加密向量2
            
        Returns:
            乘法结果（加密状态）
        """
        if not (vec1.get('is_encrypted') and vec2.get('is_encrypted')):
            raise ValueError("输入向量必须都是加密状态")
        
        # 同态乘法（极度简化，实际需要重线性化等复杂操作）
        result = {
            'data0': (vec1['data0'] * vec2['data0']) % self.context.plain_modulus,
            'data1': (vec1['data1'] * vec2['data1']) % self.context.plain_modulus,
            'scale': vec1['scale'] * vec2['scale'],
            'size': min(vec1['size'], vec2['size']),
            'is_encrypted': True
        }
        
        # 应用重线性化（简化）
        result = self.relinearize(result)
        
        return result
    
    def relinearize(self, encrypted_vector: Dict[str, Any]) -> Dict[str, Any]:
        """重线性化操作（简化实现）"""
        # 使用重线性化密钥来降低密文大小
        result = encrypted_vector.copy()
        
        # 简化的重线性化：使用重线性化密钥调整密文
        adjustment = np.mean(self.relin_keys[0]) * 0.01
        result['data0'] = (result['data0'] + adjustment) % self.context.plain_modulus
        
        return result
    
    def dot_product_encrypted(self, vec1: Dict[str, Any], vec2: Dict[str, Any]) -> Dict[str, Any]:
        """
        加密状态下的向量内积计算
        
        Args:
            vec1: 加密向量1
            vec2: 加密向量2
            
        Returns:
            内积结果（加密状态）
        """
        # 逐元素乘法
        element_products = self.multiply_encrypted_vectors(vec1, vec2)
        
        # 求和（简化实现）
        # 实际同态加密中这需要复杂的旋转和求和操作
        sum_result = {
            'data0': np.sum(element_products['data0']),
            'data1': np.sum(element_products['data1']),
            'scale': element_products['scale'],
            'size': 1,  # 标量结果
            'is_encrypted': True
        }
        
        return sum_result


class HomomorphicEncryptionRAG:
    """基于同态加密的RAG系统"""
    
    def __init__(self, polynomial_degree: int = 4096):
        """
        初始化同态加密RAG系统
        
        Args:
            polynomial_degree: 多项式度数，影响安全性和性能
        """
        # 设置加密上下文
        self.context = EncryptionContext(
            polynomial_degree=polynomial_degree,
            coefficient_modulus=[60, 40, 40, 60],
            plain_modulus=1024,
            scale=1024.0,
            security_level=128
        )
        
        # 初始化同态加密系统
        self.he_system = SimpleHomomorphicEncryption(self.context)
        
        # 存储加密的嵌入向量
        self.encrypted_embeddings = {}
        self.document_metadata = {}
        
        # 性能统计
        self.performance_stats = {
            'encryption_time': [],
            'search_time': [],
            'decryption_time': []
        }
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        添加文档及其嵌入向量到加密存储
        
        Args:
            documents: 文档列表
            embeddings: 对应的嵌入向量
        """
        if len(documents) != len(embeddings):
            raise ValueError("文档数量与嵌入向量数量不匹配")
        
        self.logger.info(f"开始加密 {len(documents)} 个文档的嵌入向量")
        
        start_time = time.time()
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = doc.get('id', f'doc_{i}')
            
            # 加密嵌入向量
            encrypted_embedding = self.he_system.encrypt_vector(embedding)
            
            # 存储加密嵌入和元数据
            self.encrypted_embeddings[doc_id] = encrypted_embedding
            self.document_metadata[doc_id] = {
                'title': doc.get('title', ''),
                'content_hash': self._hash_content(doc.get('content', '')),
                'timestamp': time.time(),
                'encrypted': True
            }
        
        encryption_time = time.time() - start_time
        self.performance_stats['encryption_time'].append(encryption_time)
        
        self.logger.info(f"文档加密完成，耗时 {encryption_time:.2f} 秒")
    
    def encrypted_similarity_search(self, 
                                   query_embedding: np.ndarray, 
                                   top_k: int = 5) -> List[Dict[str, Any]]:
        """
        在加密状态下进行相似度搜索
        
        Args:
            query_embedding: 查询向量
            top_k: 返回的文档数量
            
        Returns:
            搜索结果列表
        """
        if not self.encrypted_embeddings:
            return []
        
        self.logger.info(f"开始加密相似度搜索，查询 top-{top_k} 结果")
        
        start_time = time.time()
        
        # 1. 加密查询向量
        encrypted_query = self.he_system.encrypt_vector(query_embedding)
        
        # 2. 在加密状态下计算相似度
        similarity_scores = {}
        
        for doc_id, encrypted_doc in self.encrypted_embeddings.items():
            # 计算加密状态下的内积（相似度）
            encrypted_similarity = self.he_system.dot_product_encrypted(
                encrypted_query, encrypted_doc
            )
            
            # 解密相似度分数（只解密分数，不解密原始向量）
            similarity_score = self.he_system.decrypt_vector(encrypted_similarity)[0]
            similarity_scores[doc_id] = float(similarity_score)
        
        # 3. 排序并返回top-k结果
        sorted_results = sorted(
            similarity_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        search_time = time.time() - start_time
        self.performance_stats['search_time'].append(search_time)
        
        # 4. 构建结果
        results = []
        for doc_id, score in sorted_results:
            result = {
                'document_id': doc_id,
                'similarity_score': score,
                'metadata': self.document_metadata[doc_id],
                'encryption_preserved': True,
                'search_time': search_time
            }
            results.append(result)
        
        self.logger.info(f"加密搜索完成，耗时 {search_time:.2f} 秒，找到 {len(results)} 个结果")
        
        return results
    
    def secure_query_processing(self, query_vector: np.ndarray) -> Dict[str, Any]:
        """
        安全查询处理流程
        
        Args:
            query_vector: 查询向量
            
        Returns:
            处理结果
        """
        start_time = time.time()
        
        # 1. 执行加密搜索
        search_results = self.encrypted_similarity_search(query_vector)
        
        # 2. 计算性能开销
        computational_overhead = self.measure_computational_overhead()
        
        # 3. 构建响应
        response = {
            'query_processed': True,
            'results_count': len(search_results),
            'results': search_results,
            'privacy_level': 'fully_encrypted',
            'computational_overhead': computational_overhead,
            'processing_time': time.time() - start_time,
            'security_level': self.context.security_level
        }
        
        return response
    
    def measure_computational_overhead(self) -> Dict[str, Any]:
        """测量计算开销"""
        overhead = {
            'average_encryption_time': np.mean(self.performance_stats['encryption_time']) if self.performance_stats['encryption_time'] else 0,
            'average_search_time': np.mean(self.performance_stats['search_time']) if self.performance_stats['search_time'] else 0,
            'total_encrypted_documents': len(self.encrypted_embeddings),
            'memory_usage_mb': self._estimate_memory_usage(),
            'security_parameters': {
                'polynomial_degree': self.context.polynomial_degree,
                'security_level': self.context.security_level,
                'coefficient_modulus_size': len(self.context.coefficient_modulus)
            }
        }
        
        return overhead
    
    def _estimate_memory_usage(self) -> float:
        """估算内存使用量（MB）"""
        # 简化的内存估算
        bytes_per_encrypted_vector = self.context.polynomial_degree * 8 * 2  # 两个数据组件
        total_bytes = len(self.encrypted_embeddings) * bytes_per_encrypted_vector
        return total_bytes / (1024 * 1024)  # 转换为MB
    
    def _hash_content(self, content: str) -> str:
        """生成内容哈希"""
        import hashlib
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    def get_encryption_status(self) -> Dict[str, Any]:
        """获取加密系统状态"""
        return {
            'encrypted_documents': len(self.encrypted_embeddings),
            'encryption_context': {
                'polynomial_degree': self.context.polynomial_degree,
                'security_level': self.context.security_level,
                'scale': self.context.scale
            },
            'performance_stats': {
                'total_encryptions': len(self.performance_stats['encryption_time']),
                'total_searches': len(self.performance_stats['search_time']),
                'average_encryption_time': np.mean(self.performance_stats['encryption_time']) if self.performance_stats['encryption_time'] else 0,
                'average_search_time': np.mean(self.performance_stats['search_time']) if self.performance_stats['search_time'] else 0
            },
            'memory_usage_mb': self._estimate_memory_usage()
        }
    
    def batch_encrypted_operations(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量加密操作（提高效率）"""
        results = []
        
        self.logger.info(f"执行 {len(operations)} 个批量加密操作")
        
        for operation in operations:
            op_type = operation.get('type')
            
            if op_type == 'search':
                result = self.encrypted_similarity_search(
                    operation['query_vector'],
                    operation.get('top_k', 5)
                )
            elif op_type == 'add_document':
                self.add_documents([operation['document']], [operation['embedding']])
                result = {'status': 'document_added', 'doc_id': operation['document'].get('id')}
            else:
                result = {'error': f'未知操作类型: {op_type}'}
            
            results.append(result)
        
        return results


def demo_homomorphic_encryption_rag():
    """同态加密RAG演示"""
    print("=== 同态加密RAG演示 ===")
    
    # 创建同态加密RAG系统
    he_rag = HomomorphicEncryptionRAG(polynomial_degree=1024)  # 较小的度数用于演示
    
    # 准备测试数据
    np.random.seed(42)
    
    # 模拟文档和嵌入向量
    documents = [
        {'id': 'doc_1', 'title': '人工智能基础', 'content': '人工智能是计算机科学的一个分支...'},
        {'id': 'doc_2', 'title': '机器学习算法', 'content': '机器学习包括监督学习、无监督学习...'},
        {'id': 'doc_3', 'title': '深度学习应用', 'content': '深度学习在图像识别、自然语言处理中应用广泛...'},
        {'id': 'doc_4', 'title': '数据隐私保护', 'content': '数据隐私保护在AI时代变得越来越重要...'},
        {'id': 'doc_5', 'title': '同态加密技术', 'content': '同态加密允许在加密数据上直接进行计算...'}
    ]
    
    # 生成随机嵌入向量（在实际应用中这些应该是真实的文本嵌入）
    embedding_dim = 512  # 使用较小的维度用于演示
    document_embeddings = np.random.randn(len(documents), embedding_dim)
    
    # 归一化嵌入向量
    norms = np.linalg.norm(document_embeddings, axis=1, keepdims=True)
    document_embeddings = document_embeddings / (norms + 1e-8)
    
    print(f"准备了 {len(documents)} 个文档，嵌入维度: {embedding_dim}")
    
    # 添加文档到加密存储
    print("\n1. 加密文档嵌入向量...")
    he_rag.add_documents(documents, document_embeddings)
    
    # 显示加密状态
    status = he_rag.get_encryption_status()
    print(f"加密状态: {status['encrypted_documents']} 个文档已加密")
    print(f"内存使用: {status['memory_usage_mb']:.2f} MB")
    print(f"平均加密时间: {status['performance_stats']['average_encryption_time']:.4f} 秒")
    
    # 执行加密搜索
    print("\n2. 执行加密相似度搜索...")
    
    # 生成查询向量（模拟）
    query_vector = np.random.randn(embedding_dim)
    query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-8)
    
    # 执行搜索
    search_results = he_rag.encrypted_similarity_search(query_vector, top_k=3)
    
    print(f"搜索结果 (top-3):")
    for i, result in enumerate(search_results, 1):
        print(f"  {i}. 文档ID: {result['document_id']}")
        print(f"     相似度: {result['similarity_score']:.6f}")
        print(f"     标题: {result['metadata'].get('title', 'N/A')}")
        print(f"     加密保护: {'是' if result['encryption_preserved'] else '否'}")
        print()
    
    # 安全查询处理演示
    print("3. 安全查询处理流程...")
    secure_result = he_rag.secure_query_processing(query_vector)
    
    print(f"安全查询结果:")
    print(f"  隐私级别: {secure_result['privacy_level']}")
    print(f"  安全级别: {secure_result['security_level']} bits")
    print(f"  处理时间: {secure_result['processing_time']:.4f} 秒")
    print(f"  结果数量: {secure_result['results_count']}")
    
    # 计算开销分析
    overhead = secure_result['computational_overhead']
    print(f"\n计算开销分析:")
    print(f"  平均加密时间: {overhead['average_encryption_time']:.4f} 秒")
    print(f"  平均搜索时间: {overhead['average_search_time']:.4f} 秒")
    print(f"  内存使用: {overhead['memory_usage_mb']:.2f} MB")
    print(f"  多项式度数: {overhead['security_parameters']['polynomial_degree']}")
    
    # 批量操作演示
    print("\n4. 批量加密操作演示...")
    batch_operations = [
        {
            'type': 'search',
            'query_vector': np.random.randn(embedding_dim),
            'top_k': 2
        },
        {
            'type': 'add_document',
            'document': {'id': 'doc_6', 'title': '新文档', 'content': '这是一个新添加的文档'},
            'embedding': np.random.randn(embedding_dim)
        }
    ]
    
    batch_results = he_rag.batch_encrypted_operations(batch_operations)
    print(f"批量操作完成，处理了 {len(batch_results)} 个操作")
    
    # 最终状态
    final_status = he_rag.get_encryption_status()
    print(f"\n最终状态:")
    print(f"  加密文档数: {final_status['encrypted_documents']}")
    print(f"  总加密次数: {final_status['performance_stats']['total_encryptions']}")
    print(f"  总搜索次数: {final_status['performance_stats']['total_searches']}")
    
    # 安全性说明
    print(f"\n=== 安全性说明 ===")
    print("1. 本演示使用简化的同态加密实现，仅用于概念演示")
    print("2. 生产环境建议使用 Microsoft SEAL、HElib 等专业库")
    print("3. 同态加密提供了最强的隐私保护，但计算开销较大")
    print("4. 在实际应用中需要根据安全需求和性能要求选择合适的参数")


if __name__ == "__main__":
    demo_homomorphic_encryption_rag()