"""
RAG隐私保护技术综合演示
Deep RAG Notes Chapter 12 - Privacy Protection Technologies
"""

import numpy as np
import time
import logging
from typing import Dict, List, Any

# 导入各个模块
from differential_privacy import DifferentialPrivacyRAG, PrivacyBudgetManager
from federated_rag import FederatedRAGCoordinator, FederatedRAGNode
from data_masking import IntelligentDataMasking
from homomorphic_encryption import HomomorphicEncryptionRAG
from privacy_assessment import PrivacyImpactAssessment

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('rag_privacy_demo.log', encoding='utf-8')
        ]
    )

def generate_sample_data(num_docs: int = 50, embed_dim: int = 384) -> tuple:
    """生成示例数据"""
    np.random.seed(42)
    
    # 生成文档
    documents = []
    for i in range(num_docs):
        doc = {
            'id': f'doc_{i}',
            'title': f'文档标题 {i}',
            'content': f'这是第{i}个文档的内容，包含了一些重要信息。张三的电话是13812345678，邮箱是zhangsan@example.com。',
            'metadata': {
                'category': np.random.choice(['技术', '业务', '法律', '金融']),
                'sensitivity': np.random.choice(['low', 'medium', 'high']),
                'created_at': f'2024-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}'
            }
        }
        documents.append(doc)
    
    # 生成嵌入向量
    embeddings = np.random.randn(num_docs, embed_dim)
    # 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    # 生成查询
    query_text = "寻找关于隐私保护的技术文档"
    query_embedding = np.random.randn(embed_dim)
    query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    
    return documents, embeddings, query_text, query_embedding

def demo_differential_privacy():
    """差分隐私演示"""
    print("\n" + "="*60)
    print("🔒 差分隐私技术演示")
    print("="*60)
    
    # 生成测试数据
    documents, embeddings, query_text, query_embedding = generate_sample_data(100, 512)
    
    # 创建差分隐私RAG系统
    dp_rag = DifferentialPrivacyRAG(epsilon=1.0)
    budget_manager = PrivacyBudgetManager(total_budget=5.0)
    
    print(f"📊 测试数据: {len(documents)} 个文档，{embeddings.shape[1]} 维嵌入")
    print(f"🎯 查询: {query_text}")
    
    # 分配预算
    query_metadata = {
        'sensitivity_level': 'high',
        'complexity': 'medium',
        'data_sensitivity': 'confidential'
    }
    
    allocated_budget = budget_manager.allocate_budget_for_query(query_metadata)
    print(f"💰 分配隐私预算: {allocated_budget:.4f}")
    
    if allocated_budget > 0:
        # 设置epsilon
        dp_rag.epsilon = allocated_budget
        dp_rag.noise_scale = dp_rag.calculate_noise_scale()
        
        # 执行搜索
        start_time = time.time()
        results = dp_rag.private_similarity_search(query_embedding, embeddings, top_k=5)
        search_time = time.time() - start_time
        
        print(f"⏱️  搜索耗时: {search_time:.4f} 秒")
        print(f"📋 搜索结果:")
        
        for i, result in enumerate(results, 1):
            doc = documents[result['document_id']]
            print(f"  {i}. {doc['title']} (相似度: {result['similarity_score']:.4f})")
        
        # 记录预算使用
        query_result = {'success': True, 'result_count': len(results)}
        budget_manager.record_query_usage(allocated_budget, query_result)
        
        # 显示预算状态
        budget_status = budget_manager.get_budget_status()
        print(f"💳 预算状态: 已用 {budget_status['used_budget']:.4f}/{budget_status['total_budget']:.4f}")
        
        # 隐私指标
        metrics = dp_rag.get_privacy_metrics()
        print(f"🛡️  隐私指标: ε={metrics['epsilon']:.4f}, 噪声水平={metrics['noise_scale']:.4f}")
    else:
        print("❌ 预算不足，拒绝查询")

def demo_data_masking():
    """数据脱敏演示"""
    print("\n" + "="*60)
    print("🎭 数据脱敏技术演示")
    print("="*60)
    
    masking_system = IntelligentDataMasking()
    
    # 测试文本
    test_cases = [
        {
            'text': "客户张三，身份证号：110101199001011234，联系电话：13812345678，邮箱：zhangsan@company.com",
            'context': {'document_type': 'customer_info', 'sensitivity_level': 'high'}
        },
        {
            'text': "服务器192.168.1.100发生故障，管理员联系方式：admin@server.com，银行卡号：6222021234567890",
            'context': {'document_type': 'incident_report', 'sensitivity_level': 'medium'}
        },
        {
            'text': "李四的工作邮箱是lisi@company.com，办公电话是(021)1234-5678",
            'context': {'document_type': 'directory', 'sensitivity_level': 'low', 'is_public': True}
        }
    ]
    
    all_results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📄 测试案例 {i}:")
        print(f"原文: {case['text']}")
        
        # 执行脱敏
        start_time = time.time()
        result = masking_system.intelligent_masking(case['text'], case['context'])
        masking_time = time.time() - start_time
        
        print(f"脱敏后: {result.masked_text}")
        print(f"⏱️  脱敏耗时: {masking_time:.4f} 秒")
        
        if result.masking_operations:
            print(f"🔧 脱敏操作:")
            for op in result.masking_operations:
                print(f"  - {op['info_type']}: {op['original_value']} → {op['masked_value']} ({op['method']})")
        
        all_results.append(result)
    
    # 统计信息
    stats = masking_system.get_masking_statistics(all_results)
    print(f"\n📊 脱敏统计:")
    print(f"总操作数: {stats['total_masking_operations']}")
    print(f"平均每文档: {stats['average_operations_per_document']:.2f}")
    print(f"信息类型分布: {dict(stats['info_type_distribution'])}")

def demo_federated_rag():
    """联邦RAG演示"""
    print("\n" + "="*60)
    print("🌐 联邦RAG架构演示")
    print("="*60)
    
    # 创建协调器
    coordinator = FederatedRAGCoordinator()
    
    # 创建3个节点
    nodes = []
    for i in range(3):
        node = FederatedRAGNode(
            node_id=f"hospital_{i}",
            local_data_path=f"/data/hospital_{i}",
            encryption_key=coordinator.encryption_key
        )
        nodes.append(node)
        coordinator.register_node(node)
    
    print(f"🏥 创建了 {len(nodes)} 个医院节点")
    
    # 显示各节点状态
    for node in nodes:
        status = node.get_node_status()
        print(f"  - {status['node_id']}: {status['document_count']} 个病历文档")
    
    # 执行联邦搜索
    _, _, query_text, query_vector = generate_sample_data(10, 768)
    participating_nodes = [node.node_id for node in nodes]
    
    print(f"\n🔍 执行联邦搜索: '{query_text}'")
    print(f"参与节点: {participating_nodes}")
    
    start_time = time.time()
    search_result = coordinator.coordinate_federated_search(
        query=query_text,
        query_vector=query_vector,
        participating_nodes=participating_nodes,
        top_k=5
    )
    search_time = time.time() - start_time
    
    print(f"⏱️  联邦搜索耗时: {search_time:.4f} 秒")
    print(f"📋 搜索结果:")
    
    for i, result in enumerate(search_result['results'], 1):
        print(f"  {i}. 文档哈希: {result['document_hash'][:12]}...")
        print(f"     平均相似度: {result['average_similarity']:.4f}")
        print(f"     参与节点: {result['participating_nodes']} 个")
        print(f"     节点列表: {result['node_list']}")
    
    print(f"🔐 隐私保护: 数据未离开各节点，仅共享加密的搜索结果")

def demo_homomorphic_encryption():
    """同态加密演示"""
    print("\n" + "="*60)
    print("🔐 同态加密技术演示")
    print("="*60)
    
    # 创建同态加密RAG系统（使用较小参数用于演示）
    he_rag = HomomorphicEncryptionRAG(polynomial_degree=1024)
    
    # 生成测试数据
    documents, embeddings, query_text, query_embedding = generate_sample_data(20, 256)
    
    print(f"📊 测试数据: {len(documents)} 个文档，{embeddings.shape[1]} 维嵌入")
    print(f"🎯 查询: {query_text}")
    
    # 加密文档
    print(f"\n🔒 加密文档嵌入向量...")
    start_time = time.time()
    he_rag.add_documents(documents, embeddings)
    encryption_time = time.time() - start_time
    
    print(f"⏱️  加密耗时: {encryption_time:.4f} 秒")
    
    # 执行加密搜索
    print(f"\n🔍 执行同态加密搜索...")
    start_time = time.time()
    search_results = he_rag.encrypted_similarity_search(query_embedding, top_k=3)
    search_time = time.time() - start_time
    
    print(f"⏱️  搜索耗时: {search_time:.4f} 秒")
    print(f"📋 搜索结果:")
    
    for i, result in enumerate(search_results, 1):
        print(f"  {i}. 文档ID: {result['document_id']}")
        print(f"     相似度: {result['similarity_score']:.6f}")
        print(f"     加密保护: {'✅' if result['encryption_preserved'] else '❌'}")
    
    # 显示性能统计
    status = he_rag.get_encryption_status()
    print(f"\n📈 性能统计:")
    print(f"平均加密时间: {status['performance_stats']['average_encryption_time']:.4f} 秒")
    print(f"平均搜索时间: {status['performance_stats']['average_search_time']:.4f} 秒")
    print(f"内存使用: {status['memory_usage_mb']:.2f} MB")
    print(f"安全级别: {status['encryption_context']['security_level']} bits")

def demo_privacy_assessment():
    """隐私影响评估演示"""
    print("\n" + "="*60)
    print("📋 隐私影响评估演示")
    print("="*60)
    
    pia_system = PrivacyImpactAssessment()
    
    # 模拟RAG系统配置
    rag_config = {
        'data_sources': [
            {'name': 'customer_documents', 'sensitivity': 'high'},
            {'name': 'internal_knowledge', 'sensitivity': 'medium'}
        ],
        'encryption': {'at_rest': True, 'in_transit': True},
        'access_control': {'role_based': True, 'multi_factor_auth': True},
        'consent_mechanism': {'explicit_consent': True},
        'user_rights': {
            'data_access': {'enabled': True},
            'data_deletion': {'enabled': False}
        },
        'third_party_sharing': {'enabled': True, 'data_processing_agreement': False},
        'automated_decision_making': {'enabled': True, 'human_review': False}
    }
    
    business_context = {
        'business_regions': ['EU', 'US'],
        'data_subject_regions': ['EU'],
        'industry': 'financial_services'
    }
    
    print(f"🏢 评估系统: 金融服务RAG系统")
    print(f"🌍 业务范围: {business_context['business_regions']}")
    
    # 执行评估
    print(f"\n📊 执行隐私影响评估...")
    start_time = time.time()
    assessment_report = pia_system.conduct_comprehensive_assessment(
        system_name="金融服务RAG系统",
        rag_system_config=rag_config,
        business_context=business_context
    )
    assessment_time = time.time() - start_time
    
    print(f"⏱️  评估耗时: {assessment_time:.4f} 秒")
    
    # 显示摘要
    print(f"\n📈 评估结果:")
    print(f"整体隐私评分: {assessment_report.overall_privacy_score:.1f}/100")
    
    # 风险统计
    risk_counts = {}
    for risk in assessment_report.privacy_risks:
        severity = risk.severity.value
        risk_counts[severity] = risk_counts.get(severity, 0) + 1
    
    print(f"风险分布: {dict(risk_counts)}")
    
    # 合规状态
    compliance_counts = {}
    for item in assessment_report.compliance_items:
        status = item.status.value
        compliance_counts[status] = compliance_counts.get(status, 0) + 1
    
    print(f"合规状态: {dict(compliance_counts)}")
    
    # 主要建议
    print(f"\n💡 主要建议:")
    for i, rec in enumerate(assessment_report.recommendations[:5], 1):
        print(f"  {i}. {rec}")
    
    print(f"\n📅 下次审查: {assessment_report.next_review_date[:10]}")

def demo_comprehensive_comparison():
    """综合技术对比演示"""
    print("\n" + "="*60)
    print("⚖️  隐私保护技术综合对比")
    print("="*60)
    
    # 生成测试数据
    documents, embeddings, query_text, query_embedding = generate_sample_data(50, 384)
    
    comparison_results = {}
    
    # 1. 差分隐私
    print(f"\n🔒 测试差分隐私...")
    dp_rag = DifferentialPrivacyRAG(epsilon=1.0)
    start_time = time.time()
    dp_results = dp_rag.private_similarity_search(query_embedding, embeddings, top_k=5)
    dp_time = time.time() - start_time
    
    comparison_results['差分隐私'] = {
        '搜索时间': dp_time,
        '结果数量': len(dp_results),
        '隐私保护级别': 'Medium-High',
        '计算开销': 'Low',
        '准确性影响': 'Medium'
    }
    
    # 2. 数据脱敏
    print(f"🎭 测试数据脱敏...")
    masking_system = IntelligentDataMasking()
    start_time = time.time()
    sample_text = documents[0]['content']
    masking_result = masking_system.intelligent_masking(sample_text)
    masking_time = time.time() - start_time
    
    comparison_results['数据脱敏'] = {
        '处理时间': masking_time,
        '脱敏操作数': len(masking_result.masking_operations),
        '隐私保护级别': 'Medium',
        '计算开销': 'Very Low',
        '准确性影响': 'Low'
    }
    
    # 3. 同态加密（简化测试）
    print(f"🔐 测试同态加密...")
    he_rag = HomomorphicEncryptionRAG(polynomial_degree=512)  # 更小的参数
    start_time = time.time()
    he_rag.add_documents(documents[:10], embeddings[:10])  # 只测试10个文档
    he_results = he_rag.encrypted_similarity_search(query_embedding, top_k=3)
    he_time = time.time() - start_time
    
    comparison_results['同态加密'] = {
        '总耗时': he_time,
        '结果数量': len(he_results),
        '隐私保护级别': 'Very High',
        '计算开销': 'Very High',
        '准确性影响': 'Low'
    }
    
    # 显示对比结果
    print(f"\n📊 技术对比结果:")
    print(f"{'技术':<10} {'隐私保护':<12} {'计算开销':<10} {'准确性影响':<12}")
    print("-" * 50)
    
    for tech, metrics in comparison_results.items():
        privacy_level = metrics.get('隐私保护级别', 'N/A')
        compute_cost = metrics.get('计算开销', 'N/A')  
        accuracy_impact = metrics.get('准确性影响', 'N/A')
        print(f"{tech:<10} {privacy_level:<12} {compute_cost:<10} {accuracy_impact:<12}")
    
    # 使用建议
    print(f"\n💡 使用建议:")
    print(f"• 数据脱敏：适合大规模部署，性能影响最小")
    print(f"• 差分隐私：平衡隐私和性能，适合一般场景")
    print(f"• 同态加密：最高隐私保护，适合极敏感数据")
    print(f"• 联邦学习：适合多机构协作，数据不出域")

def main():
    """主演示函数"""
    print("🚀 RAG隐私保护技术综合演示")
    print("Deep RAG Notes Chapter 12 - Privacy Protection Technologies")
    print("="*80)
    
    # 设置日志
    setup_logging()
    
    try:
        # 各技术演示
        demo_differential_privacy()
        demo_data_masking()
        demo_federated_rag()
        demo_homomorphic_encryption()
        demo_privacy_assessment()
        demo_comprehensive_comparison()
        
        print("\n" + "="*80)
        print("✅ 所有演示完成！")
        print("\n📋 演示总结:")
        print("• 差分隐私：通过添加噪声保护个体隐私")
        print("• 数据脱敏：智能识别和隐藏敏感信息")
        print("• 联邦RAG：多方协作但数据不离开本地")
        print("• 同态加密：在加密状态下直接计算")
        print("• 隐私评估：系统性评估隐私风险和合规性")
        print("\n🎯 根据业务需求选择合适的技术组合，实现隐私保护与功能性的最佳平衡")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {str(e)}")
        logging.error(f"演示错误: {str(e)}", exc_info=True)
    
    finally:
        print(f"\n📄 详细日志已保存至: rag_privacy_demo.log")

if __name__ == "__main__":
    main()