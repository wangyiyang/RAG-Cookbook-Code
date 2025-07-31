# RAG隐私保护技术代码实现

本目录包含《深度RAG笔记第12篇》中涉及的所有隐私保护技术的完整实现代码。

## 文件结构

```
code/ch12/
├── README.md                    # 本说明文档
├── differential_privacy.py     # 差分隐私RAG实现
├── federated_rag.py            # 联邦RAG架构实现  
├── data_masking.py             # 数据脱敏与匿名化实现
├── homomorphic_encryption.py   # 同态加密RAG实现
├── privacy_assessment.py       # 隐私影响评估系统
└── demo.py                     # 综合演示程序
```

## 主要技术实现

### 1. 差分隐私 (differential_privacy.py)

**核心功能**：
- `DifferentialPrivacyRAG`: 差分隐私RAG系统主类
- `PrivacyBudgetManager`: 隐私预算管理器
- 高斯机制噪声添加
- 指数机制排序选择
- 自适应预算分配

**使用示例**：
```python
from differential_privacy import DifferentialPrivacyRAG

dp_rag = DifferentialPrivacyRAG(epsilon=1.0)
results = dp_rag.private_similarity_search(query_vector, doc_embeddings)
```

### 2. 联邦RAG (federated_rag.py)

**核心功能**：
- `FederatedRAGCoordinator`: 联邦协调器
- `FederatedRAGNode`: 联邦节点实现
- 分布式私有搜索
- 加密通信协议
- 多节点结果聚合

**使用示例**：
```python
from federated_rag import FederatedRAGCoordinator, FederatedRAGNode

coordinator = FederatedRAGCoordinator()
node = FederatedRAGNode("hospital_1", "/data/hospital_1")
results = coordinator.coordinate_federated_search(query, participating_nodes)
```

### 3. 数据脱敏 (data_masking.py)

**核心功能**：
- `IntelligentDataMasking`: 智能数据脱敏系统
- 多种敏感信息模式识别
- 上下文感知敏感度调整
- 多种脱敏策略（哈希、部分遮蔽、域泛化、令牌化）

**使用示例**：
```python
from data_masking import IntelligentDataMasking

masking_system = IntelligentDataMasking()
result = masking_system.intelligent_masking(text, context)
print(result.masked_text)
```

### 4. 同态加密 (homomorphic_encryption.py)

**核心功能**：
- `HomomorphicEncryptionRAG`: 同态加密RAG系统
- `SimpleHomomorphicEncryption`: 简化同态加密实现
- 加密状态下向量相似度计算
- 加密索引构建和搜索

**使用示例**：
```python
from homomorphic_encryption import HomomorphicEncryptionRAG

he_rag = HomomorphicEncryptionRAG()
he_rag.add_documents(documents, embeddings)
results = he_rag.encrypted_similarity_search(query_vector)
```

### 5. 隐私影响评估 (privacy_assessment.py)

**核心功能**：
- `PrivacyImpactAssessment`: 隐私影响评估系统
- 多维度隐私风险评估
- 法规合规性检查（GDPR、CCPA、PIPL）
- 自动化建议生成

**使用示例**：
```python
from privacy_assessment import PrivacyImpactAssessment

pia = PrivacyImpactAssessment()
report = pia.conduct_comprehensive_assessment(system_name, config, context)
print(f"隐私评分: {report.overall_privacy_score}")
```

## 综合演示

运行 `demo.py` 可以体验所有隐私保护技术的综合演示：

```bash
python demo.py
```

演示内容包括：
- 差分隐私搜索演示
- 数据脱敏效果展示  
- 联邦RAG多节点协作
- 同态加密计算演示
- 隐私影响评估报告
- 技术性能对比分析

## 技术参数配置

### 差分隐私参数
- `epsilon`: 隐私预算 (推荐值: 0.1-2.0)
- `delta`: 失败概率 (推荐值: 1e-5)
- `sensitivity`: 敏感度参数 (默认: 1.0)

### 同态加密参数  
- `polynomial_degree`: 多项式度数 (推荐值: 4096-8192)
- `security_level`: 安全级别 (推荐值: 128 bits)
- `scale`: 编码比例 (推荐值: 1024.0)

### 脱敏策略配置
- `hash_anonymization`: 哈希匿名化（不可逆，高安全性）
- `partial_masking`: 部分遮蔽（保留可读性）
- `domain_generalization`: 域泛化（保持语义）
- `tokenization`: 令牌化（可逆，便于分析）

## 性能基准

在标准测试环境下的性能表现：

| 技术 | 文档数量 | 查询时间 | 内存开销 | 隐私保护级别 |
|------|----------|----------|----------|--------------|
| 差分隐私 | 10,000 | ~0.1s | +20% | Medium-High |
| 数据脱敏 | 无限制 | ~0.01s | +5% | Medium |
| 联邦RAG | 1,000/节点 | ~2.0s | +50% | High |
| 同态加密 | 1,000 | ~10.0s | +500% | Very High |

## 生产部署建议

1. **数据脱敏**：适合大规模部署，性能影响最小
2. **差分隐私**：平衡隐私和性能，适合一般场景
3. **联邦RAG**：适合多机构协作，数据不出域
4. **同态加密**：最高隐私保护，适合极敏感数据

## 法规合规支持

- **GDPR**：数据最小化、被遗忘权、处理透明度
- **CCPA**：数据收集告知、删除权利、访问权利
- **PIPL**：处理同意、跨境传输限制、本地化要求

## 安全注意事项

1. **同态加密**：本实现为简化演示版本，生产环境请使用Microsoft SEAL、HElib等专业库
2. **密钥管理**：所有加密密钥应使用安全的密钥管理系统
3. **审计日志**：建议记录所有隐私保护操作的审计日志
4. **定期评估**：隐私风险评估应定期进行，建议每季度一次

## 联系支持

如有问题或建议，请参考《深度RAG笔记第12篇》或查阅相关技术文档。