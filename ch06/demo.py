"""
法律文档智能检索系统演示
整合所有模块，展示完整的法律RAG系统功能
"""

import json
import time
from typing import Dict, List, Any, Optional

# 导入自定义模块
from document_processor import LegalDocumentProcessor, DocumentType
from legal_ner import LegalEntityRecognizer, EntityType
from legal_retriever import ProfessionalLegalRetriever, LegalQuery
from citation_analyzer import LegalCitationAnalyzer
from knowledge_graph import LegalKnowledgeGraphBuilder
from quality_validator import LegalQualityValidator
from legal_monitor import LegalSystemMonitor, MetricType, Metric


class LegalRAGSystem:
    """法律文档智能检索系统"""
    
    def __init__(self):
        # 初始化所有组件
        self.document_processor = LegalDocumentProcessor()
        self.ner_recognizer = LegalEntityRecognizer()
        self.retriever = ProfessionalLegalRetriever()
        self.citation_analyzer = LegalCitationAnalyzer()
        self.knowledge_graph_builder = LegalKnowledgeGraphBuilder()
        self.quality_validator = LegalQualityValidator()
        self.system_monitor = LegalSystemMonitor()
        
        # 知识库
        self.knowledge_base = {
            'documents': {},
            'processed_docs': {},
            'knowledge_graph': None,
            'citation_network': None
        }
        
        # 启动监控
        self.system_monitor.start_monitoring()
        
        print("🏛️ 法律文档智能检索系统已启动")
        print("=" * 50)
    
    def load_sample_documents(self) -> None:
        """加载示例法律文档"""
        
        sample_docs = [
            {
                'id': 'judgment_001',
                'title': '房屋买卖合同纠纷案',
                'content': '''
                北京市朝阳区人民法院民事判决书
                （2023）京0105民初12345号
                
                原告：张三，男，汉族，1980年1月1日出生，住北京市朝阳区
                被告：李四有限公司，住所地北京市朝阳区，法定代表人王五
                
                经审理查明：2023年1月1日，原告张三与被告李四有限公司签订《房屋买卖合同》，
                约定被告将其位于北京市朝阳区的商业用房以100万元价格出售给原告。
                合同约定2023年3月1日前完成房屋过户手续。
                
                本院认为：根据《中华人民共和国民法典》第464条规定，
                合同是民事主体之间设立、变更、终止民事法律关系的协议。
                依照《中华人民共和国民法典》第577条规定，
                当事人一方不履行合同义务应当承担违约责任。
                
                被告未按合同约定时间办理过户手续，构成违约，应承担违约责任。
                
                判决如下：
                一、被告李四有限公司于本判决生效之日起十日内协助原告张三办理房屋过户手续。
                二、被告李四有限公司赔偿原告张三违约金5万元。
                三、案件受理费由被告承担。
                
                如不服本判决，可在判决书送达之日起十五日内向北京市第二中级人民法院上诉。
                
                审判员：赵六
                书记员：田七
                2023年6月1日
                ''',
                'source': 'court_database',
                'doc_type': 'judgment'
            },
            {
                'id': 'law_civil_code_464',
                'title': '民法典第464条',
                'content': '''
                《中华人民共和国民法典》第464条
                合同是民事主体之间设立、变更、终止民事法律关系的协议。
                婚姻、收养、监护等有关身份关系的协议，适用有关该身份关系的法律规定；
                没有规定的，可以根据其性质参照适用本编规定。
                ''',
                'source': 'law_database',
                'doc_type': 'law'
            },
            {
                'id': 'law_civil_code_577',
                'title': '民法典第577条',
                'content': '''
                《中华人民共和国民法典》第577条
                当事人一方不履行合同义务或者履行合同义务不符合约定的，
                应当承担继续履行、采取补救措施或者赔偿损失等违约责任。
                ''',
                'source': 'law_database',
                'doc_type': 'law'
            },
            {
                'id': 'judgment_002',
                'title': '劳动合同纠纷案',
                'content': '''
                上海市浦东新区人民法院民事判决书
                （2023）沪0115民初56789号
                
                原告：刘八，男，汉族，1985年5月5日出生
                被告：ABC科技有限公司，住所地上海市浦东新区
                
                经审理查明：2022年1月1日，原告与被告签订劳动合同，
                约定试用期3个月，试用期满后正式录用。
                被告于2022年3月25日以不符合录用条件为由解除劳动合同。
                
                本院认为：根据《中华人民共和国劳动法》第21条规定，
                劳动合同可以约定试用期。试用期最长不得超过6个月。
                依照《中华人民共和国劳动合同法》第39条规定，
                在试用期间被证明不符合录用条件的，用人单位可以解除劳动合同。
                
                判决如下：
                一、驳回原告的诉讼请求。
                二、案件受理费由原告承担。
                
                审判员：孙九
                书记员：周十
                2023年7月15日
                ''',
                'source': 'court_database',
                'doc_type': 'judgment'
            }
        ]
        
        print("📚 正在加载示例法律文档...")
        
        # 处理文档
        for doc in sample_docs:
            # 文档处理
            processed_doc = self.document_processor.process_legal_document(doc)
            
            # 存储到知识库
            self.knowledge_base['documents'][doc['id']] = doc
            self.knowledge_base['processed_docs'][doc['id']] = processed_doc
            
            print(f"   ✓ 已处理: {doc['title']}")
        
        print(f"📊 共加载 {len(sample_docs)} 个法律文档")
        print()
    
    def build_knowledge_system(self) -> None:
        """构建知识系统"""
        
        print("🧠 正在构建法律知识系统...")
        
        # 1. 构建引用网络
        print("   📊 分析引用网络...")
        documents = list(self.knowledge_base['documents'].values())
        citation_analysis = self.citation_analyzer.analyze_legal_citations_like_scholar(documents)
        self.knowledge_base['citation_network'] = citation_analysis
        print(f"   ✓ 发现 {len(citation_analysis['citations'])} 个引用关系")
        
        # 2. 构建知识图谱
        print("   🕸️ 构建知识图谱...")
        kg_result = self.knowledge_graph_builder.build_legal_knowledge_graph(documents)
        self.knowledge_base['knowledge_graph'] = kg_result
        print(f"   ✓ 构建完成: {len(kg_result['entities'])} 个实体, {len(kg_result['relations'])} 个关系")
        
        # 3. 更新检索器知识库
        retriever_kb = {
            'documents': self.knowledge_base['documents'],
            'processed_docs': self.knowledge_base['processed_docs'],
            'law_provisions': [
                {
                    'id': 'civil_code_464',
                    'title': '民法典第464条',
                    'content': '合同是民事主体之间设立、变更、终止民事法律关系的协议'
                },
                {
                    'id': 'civil_code_577',
                    'title': '民法典第577条',
                    'content': '当事人一方不履行合同义务应当承担违约责任'
                }
            ],
            'authoritative_sources': [
                {
                    'id': 'supreme_court_guidance',
                    'title': '最高人民法院指导意见',
                    'content': '合同违约责任的认定和处理原则',
                    'authority_score': 0.95,
                    'case_type': '合同纠纷'
                }
            ]
        }
        self.retriever.knowledge_base = retriever_kb
        
        print("🎯 知识系统构建完成")
        print()
    
    def process_legal_query(self, query_text: str, context: str = "") -> Dict[str, Any]:
        """处理法律查询"""
        
        print(f"❓ 法律查询: {query_text}")
        print("-" * 30)
        
        start_time = time.time()
        
        # 1. 构建查询对象
        legal_query = LegalQuery(
            query_text=query_text,
            case_elements={},
            query_type="legal_consultation",
            context=context
        )
        
        # 2. 智能检索
        print("🔍 正在进行专业法律检索...")
        search_results = self.retriever.search_like_senior_lawyer(legal_query, context)
        
        # 3. 实体识别
        print("🏷️ 正在识别法律实体...")
        entities = self.ner_recognizer.extract_legal_entities_like_expert(query_text)
        
        # 4. 生成法律建议
        legal_advice = self._generate_legal_advice(query_text, search_results, entities)
        
        # 5. 质量验证
        print("✅ 正在验证内容质量...")
        quality_report = self.quality_validator.validate_legal_content_like_expert(
            legal_advice, query_text, search_results
        )
        
        # 6. 记录监控指标
        response_time = time.time() - start_time
        self._record_query_metrics(query_text, response_time, quality_report)
        
        # 7. 构建响应
        response = {
            'query': query_text,
            'legal_advice': legal_advice,
            'search_results': search_results[:3],  # 只返回前3个结果
            'entities': [{'text': e.text, 'type': e.entity_type.value, 'importance': e.legal_significance} 
                        for e in entities[:5]],  # 只返回前5个实体
            'quality_report': {
                'overall_score': quality_report.overall_score,
                'risk_level': quality_report.overall_risk_level.value,
                'confidence': quality_report.confidence_score,
                'recommendations': quality_report.recommendations
            },
            'response_time': response_time
        }
        
        return response
    
    def _generate_legal_advice(self, query: str, search_results: List, entities: List) -> str:
        """生成法律建议"""
        
        # 简化的法律建议生成
        advice_parts = []
        
        # 开头
        advice_parts.append("根据您的法律咨询，我为您提供以下专业分析：")
        advice_parts.append("")
        
        # 法律分析
        if search_results:
            advice_parts.append("**法律分析：**")
            top_result = search_results[0]
            advice_parts.append(f"基于相关法律条文和判例，{top_result.content[:100]}...")
            advice_parts.append("")
        
        # 实体分析
        if entities:
            advice_parts.append("**关键法律要素：**")
            for entity in entities[:3]:
                advice_parts.append(f"- {entity.text} ({entity.entity_type.value})")
            advice_parts.append("")
        
        # 建议
        advice_parts.append("**专业建议：**")
        if "合同" in query:
            advice_parts.append("1. 建议仔细审查合同条款，确保权利义务明确")
            advice_parts.append("2. 保留相关证据材料，包括合同原件、履行凭证等")
            advice_parts.append("3. 如需法律救济，建议及时采取法律行动")
        elif "劳动" in query:
            advice_parts.append("1. 建议保留劳动合同、工资条等相关证据")
            advice_parts.append("2. 了解劳动法相关规定，维护自身合法权益")
            advice_parts.append("3. 可考虑通过劳动仲裁途径解决争议")
        else:
            advice_parts.append("1. 建议咨询专业律师，获取具体法律建议")
            advice_parts.append("2. 收集和保存相关证据材料")
            advice_parts.append("3. 及时采取合法途径维护权益")
        
        advice_parts.append("")
        advice_parts.append("**免责声明：**")
        advice_parts.append("本建议仅供参考，不构成正式法律意见。具体法律问题请咨询专业律师。")
        
        return "\n".join(advice_parts)
    
    def _record_query_metrics(self, query: str, response_time: float, quality_report) -> None:
        """记录查询指标"""
        
        # 性能指标
        self.system_monitor.record_query_performance(
            query_id=f"query_{int(time.time())}",
            response_time=response_time,
            success=True,
            cache_hit=False
        )
        
        # 质量指标
        self.system_monitor.record_quality_metrics(
            query_id=f"query_{int(time.time())}",
            accuracy=quality_report.overall_score,
            relevance=0.85,  # 简化值
            completeness=0.80,  # 简化值
            validation_passed=quality_report.overall_risk_level.value in ['low_risk', 'medium_risk'],
            risk_level=quality_report.overall_risk_level.value
        )
        
        # 业务指标
        case_type = "合同纠纷" if "合同" in query else "其他"
        self.system_monitor.record_business_metrics(
            user_id=f"user_{hash(query) % 100}",
            case_type=case_type,
            satisfaction_score=4.2,  # 简化值
            converted=True
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        
        print("📊 系统状态报告")
        print("=" * 30)
        
        # 获取监控报告
        comprehensive_report = self.system_monitor.get_comprehensive_report()
        
        # 简化状态信息
        status = {
            'system_health': comprehensive_report['performance']['health']['overall_status'],
            'total_documents': len(self.knowledge_base['documents']),
            'knowledge_entities': len(self.knowledge_base.get('knowledge_graph', {}).get('entities', {})),
            'citation_relationships': len(self.knowledge_base.get('citation_network', {}).get('citations', [])),
            'performance_stats': {
                'avg_response_time': comprehensive_report['performance']['stats'].avg_response_time,
                'throughput': comprehensive_report['performance']['stats'].throughput,
                'error_rate': comprehensive_report['performance']['stats'].error_rate
            },
            'quality_stats': {
                'avg_accuracy': comprehensive_report['quality']['stats'].avg_accuracy,
                'validation_pass_rate': comprehensive_report['quality']['stats'].validation_pass_rate
            },
            'business_stats': {
                'total_queries': comprehensive_report['business']['stats'].total_queries,
                'unique_users': comprehensive_report['business']['stats'].unique_users,
                'user_satisfaction': comprehensive_report['business']['stats'].user_satisfaction
            }
        }
        
        return status
    
    def shutdown(self) -> None:
        """关闭系统"""
        self.system_monitor.stop_monitoring()
        print("🔚 法律文档智能检索系统已关闭")


def main():
    """主函数 - 系统演示"""
    
    print("🏛️ 法律文档智能检索系统演示")
    print("=" * 50)
    print()
    
    # 初始化系统
    legal_system = LegalRAGSystem()
    
    try:
        # 1. 加载示例文档
        legal_system.load_sample_documents()
        
        # 2. 构建知识系统
        legal_system.build_knowledge_system()
        
        # 3. 演示查询处理
        test_queries = [
            "房屋买卖合同违约了，卖方不配合过户，我应该怎么办？",
            "劳动合同试用期被辞退，这样合理吗？",
            "签订合同时需要注意什么法律风险？"
        ]
        
        print("🎭 开始演示查询处理...")
        print()
        
        for i, query in enumerate(test_queries, 1):
            print(f"📋 演示查询 {i}/{len(test_queries)}")
            
            # 处理查询
            response = legal_system.process_legal_query(query)
            
            # 显示结果
            print(f"📄 法律建议预览:")
            advice_lines = response['legal_advice'].split('\n')
            for line in advice_lines[:8]:  # 只显示前8行
                print(f"   {line}")
            if len(advice_lines) > 8:
                print(f"   ... (还有 {len(advice_lines) - 8} 行)")
            print()
            
            print(f"📊 质量评估:")
            quality = response['quality_report']
            print(f"   总体评分: {quality['overall_score']:.2f}")
            print(f"   风险级别: {quality['risk_level']}")
            print(f"   置信度: {quality['confidence']:.2f}")
            print()
            
            print(f"🏷️ 识别实体: {len(response['entities'])} 个")
            for entity in response['entities'][:3]:
                print(f"   - {entity['text']} ({entity['type']})")
            print()
            
            print(f"⏱️ 响应时间: {response['response_time']:.2f}秒")
            print()
            print("-" * 50)
            print()
            
            # 短暂停顿
            time.sleep(1)
        
        # 4. 显示系统状态
        print("📊 系统状态总览")
        print("=" * 30)
        
        status = legal_system.get_system_status()
        
        print(f"🏥 系统健康状态: {status['system_health']}")
        print(f"📚 文档总数: {status['total_documents']}")
        print(f"🧠 知识实体数: {status['knowledge_entities']}")
        print(f"🔗 引用关系数: {status['citation_relationships']}")
        print()
        
        print("⚡ 性能指标:")
        perf = status['performance_stats']
        print(f"   平均响应时间: {perf['avg_response_time']:.2f}秒")
        print(f"   错误率: {perf['error_rate']:.2%}")
        print()
        
        print("✅ 质量指标:")
        quality = status['quality_stats']
        print(f"   平均准确率: {quality['avg_accuracy']:.2%}")
        print(f"   验证通过率: {quality['validation_pass_rate']:.2%}")
        print()
        
        print("💼 业务指标:")
        business = status['business_stats']
        print(f"   总查询数: {business['total_queries']}")
        print(f"   独立用户数: {business['unique_users']}")
        print(f"   用户满意度: {business['user_satisfaction']:.2f}/5.0")
        print()
        
        print("🎉 演示完成！")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n⚠️ 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
    finally:
        # 关闭系统
        legal_system.shutdown()


if __name__ == "__main__":
    main()