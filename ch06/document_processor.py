"""
法律文档智能处理器
实现专业级法律文档解析、结构化提取和实体识别
"""

import re
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import jieba.posseg as pseg


class DocumentType(Enum):
    """文档类型枚举"""
    JUDGMENT = "judgment"      # 判决书
    LAW = "law"               # 法律法规
    REGULATION = "regulation"  # 司法解释
    CONTRACT = "contract"      # 合同
    GENERAL = "general"        # 一般法律文档


@dataclass
class LegalSection:
    """法律文档段落"""
    section_type: str
    content: str
    entities: List[Dict]
    keywords: List[str]
    importance_score: float


@dataclass 
class ProcessedDocument:
    """处理后的法律文档"""
    doc_id: str
    doc_type: DocumentType
    title: str
    sections: List[LegalSection]
    metadata: Dict[str, Any]
    searchable_chunks: List[Dict]


class LegalDocumentProcessor:
    """法律文档智能处理器"""
    
    def __init__(self):
        self.section_patterns = self._load_section_patterns()
        self.legal_terms = self._load_legal_terms()
        self.court_hierarchy = self._load_court_hierarchy()
        
    def process_legal_document(self, document: Dict[str, Any]) -> ProcessedDocument:
        """处理法律文档的主入口"""
        # 1. 文档类型智能识别
        doc_type = self.classify_document_type(document['content'])
        
        # 2. 根据类型选择专业解析策略
        if doc_type == DocumentType.JUDGMENT:
            return self._process_judgment_document(document)
        elif doc_type == DocumentType.LAW:
            return self._process_law_document(document) 
        elif doc_type == DocumentType.REGULATION:
            return self._process_regulation_document(document)
        else:
            return self._process_general_legal_document(document)
    
    def classify_document_type(self, content: str) -> DocumentType:
        """智能文档类型分类"""
        # 判决书识别模式
        judgment_patterns = [
            r'人民法院.*?判决书',
            r'（\d{4}）.*?\d+号',
            r'原告.*?被告',
            r'本院认为',
            r'判决如下'
        ]
        
        # 法律法规识别模式
        law_patterns = [
            r'第[一二三四五六七八九十百千万\d]+条',
            r'《.*?法》',
            r'总则.*?分则',
            r'附则'
        ]
        
        # 司法解释识别模式
        regulation_patterns = [
            r'最高人民法院.*?解释',
            r'司法解释',
            r'适用.*?问题的规定'
        ]
        
        content_lower = content.lower()
        
        # 计算各类型匹配分数
        judgment_score = sum(1 for pattern in judgment_patterns 
                           if re.search(pattern, content))
        law_score = sum(1 for pattern in law_patterns 
                       if re.search(pattern, content))
        regulation_score = sum(1 for pattern in regulation_patterns 
                             if re.search(pattern, content))
        
        # 选择最高分类型
        max_score = max(judgment_score, law_score, regulation_score)
        
        if max_score == 0:
            return DocumentType.GENERAL
        elif judgment_score == max_score:
            return DocumentType.JUDGMENT
        elif law_score == max_score:
            return DocumentType.LAW
        else:
            return DocumentType.REGULATION
    
    def _process_judgment_document(self, document: Dict) -> ProcessedDocument:
        """判决书专业解析"""
        content = document['content']
        
        # 判决书六大核心段落
        section_extractors = {
            'case_info': self._extract_case_info,
            'case_facts': self._extract_case_facts,
            'court_reasoning': self._extract_court_reasoning,
            'judgment_result': self._extract_judgment_result,
            'legal_basis': self._extract_legal_basis,
            'execution_info': self._extract_execution_info
        }
        
        sections = []
        searchable_chunks = []
        
        for section_type, extractor in section_extractors.items():
            try:
                section_content = extractor(content)
                if section_content:
                    # 提取法律实体
                    entities = self._extract_legal_entities(section_content)
                    
                    # 提取关键词
                    keywords = self._extract_legal_keywords(section_content)
                    
                    # 计算重要性分数
                    importance = self._calculate_section_importance(
                        section_type, section_content, entities
                    )
                    
                    section = LegalSection(
                        section_type=section_type,
                        content=section_content,
                        entities=entities,
                        keywords=keywords,
                        importance_score=importance
                    )
                    sections.append(section)
                    
                    # 创建可搜索的数据块
                    chunk = self._create_searchable_chunk(
                        section, document, DocumentType.JUDGMENT
                    )
                    searchable_chunks.append(chunk)
                    
            except Exception as e:
                print(f"处理段落 {section_type} 时出错: {e}")
                continue
        
        # 提取文档元数据
        metadata = self._extract_judgment_metadata(content)
        
        return ProcessedDocument(
            doc_id=document.get('id', ''),
            doc_type=DocumentType.JUDGMENT,
            title=self._extract_document_title(content),
            sections=sections,
            metadata=metadata,
            searchable_chunks=searchable_chunks
        )
    
    def _extract_case_info(self, content: str) -> str:
        """提取案件基本信息"""
        # 查找案件编号
        case_number_pattern = r'（\d{4}）[^（）]*\d+号'
        case_number_match = re.search(case_number_pattern, content)
        
        # 查找当事人信息
        parties_pattern = r'(原告|被告|申请人|被申请人|上诉人|被上诉人)[:：]([^，。；\n]+)'
        parties_matches = re.findall(parties_pattern, content)
        
        # 查找审理法院
        court_pattern = r'(\w+人民法院)'
        court_match = re.search(court_pattern, content)
        
        # 组织案件信息
        case_info_parts = []
        
        if case_number_match:
            case_info_parts.append(f"案件编号：{case_number_match.group(0)}")
            
        if court_match:
            case_info_parts.append(f"审理法院：{court_match.group(1)}")
            
        if parties_matches:
            case_info_parts.append("当事人：")
            for role, name in parties_matches[:6]:  # 限制显示前6个当事人
                case_info_parts.append(f"  {role}：{name.strip()}")
        
        return "\n".join(case_info_parts) if case_info_parts else ""
    
    def _extract_case_facts(self, content: str) -> str:
        """提取案件事实"""
        # 查找案件事实段落的常见标识
        fact_indicators = [
            r'经审理查明[：，](.+?)(?=本院认为|依照|综上)',
            r'案件事实[：，](.+?)(?=本院认为|依照|综上)',
            r'查明事实[：，](.+?)(?=本院认为|依照|综上)',
            r'事实和理由[：，](.+?)(?=本院认为|依照|综上)'
        ]
        
        for pattern in fact_indicators:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                facts = match.group(1).strip()
                # 清理和格式化
                facts = re.sub(r'\s+', ' ', facts)
                return facts[:2000]  # 限制长度
        
        return ""
    
    def _extract_court_reasoning(self, content: str) -> str:
        """提取法院认定和理由"""
        reasoning_patterns = [
            r'本院认为[：，](.+?)(?=判决如下|综上|依照)',
            r'法院认为[：，](.+?)(?=判决如下|综上|依照)',
            r'审理认为[：，](.+?)(?=判决如下|综上|依照)'
        ]
        
        for pattern in reasoning_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                reasoning = match.group(1).strip()
                reasoning = re.sub(r'\s+', ' ', reasoning)
                return reasoning[:2000]
        
        return ""
    
    def _extract_judgment_result(self, content: str) -> str:
        """提取判决结果"""
        result_patterns = [
            r'判决如下[：，](.+?)(?=如不服|当事人|审判员)',
            r'判决[：，](.+?)(?=如不服|当事人|审判员)',
            r'裁定如下[：，](.+?)(?=如不服|当事人|审判员)'
        ]
        
        for pattern in result_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                result = match.group(1).strip()
                result = re.sub(r'\s+', ' ', result)
                return result[:1000]
        
        return ""
    
    def _extract_legal_basis(self, content: str) -> str:
        """提取法律依据"""
        # 查找依照条款
        basis_patterns = [
            r'依照(.+?)(?=判决如下|特此判决)',
            r'依据(.+?)(?=判决如下|特此判决)',
            r'根据(.+?)(?=判决如下|特此判决)'
        ]
        
        legal_references = []
        
        for pattern in basis_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                # 清理文本
                clean_match = re.sub(r'\s+', ' ', match.strip())
                if len(clean_match) > 10:  # 过滤太短的匹配
                    legal_references.append(clean_match)
        
        # 同时查找明确的法条引用
        law_citation_pattern = r'《[^》]+》第\d+条[^，。]*'
        law_citations = re.findall(law_citation_pattern, content)
        
        all_references = legal_references + law_citations
        return "\n".join(set(all_references))  # 去重
    
    def _extract_execution_info(self, content: str) -> str:
        """提取执行信息"""
        execution_patterns = [
            r'如不服本判决(.+?)(?=审判员|书记员)',
            r'上诉期为(.+?)(?=审判员|书记员)',
            r'执行期限(.+?)(?=审判员|书记员)'
        ]
        
        execution_info = []
        for pattern in execution_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                clean_match = re.sub(r'\s+', ' ', match.strip())
                if clean_match:
                    execution_info.append(clean_match)
        
        return "\n".join(execution_info)
    
    def _extract_legal_entities(self, text: str) -> List[Dict]:
        """提取法律实体"""
        entities = []
        
        # 法律法规实体
        law_pattern = r'《([^》]+)》'
        law_matches = re.findall(law_pattern, text)
        for law in law_matches:
            entities.append({
                'text': law,
                'type': 'LAW',
                'confidence': 0.9
            })
        
        # 法院实体
        court_pattern = r'(\w+人民法院)'
        court_matches = re.findall(court_pattern, text)
        for court in court_matches:
            entities.append({
                'text': court,
                'type': 'COURT',
                'confidence': 0.85
            })
        
        # 金额实体
        money_pattern = r'(\d+(?:,\d{3})*(?:\.\d{2})?[万千百十]?元)'
        money_matches = re.findall(money_pattern, text)
        for money in money_matches:
            entities.append({
                'text': money,
                'type': 'MONEY',
                'confidence': 0.8
            })
        
        # 日期实体
        date_pattern = r'(\d{4}年\d{1,2}月\d{1,2}日)'
        date_matches = re.findall(date_pattern, text)
        for date in date_matches:
            entities.append({
                'text': date,
                'type': 'DATE',
                'confidence': 0.85
            })
        
        return entities
    
    def _extract_legal_keywords(self, text: str) -> List[str]:
        """提取法律关键词"""
        # 使用jieba分词
        words = pseg.cut(text)
        
        keywords = []
        for word, flag in words:
            # 保留名词、动词、形容词
            if flag in ['n', 'v', 'a', 'nr', 'ns', 'nt', 'nz']:
                if len(word) >= 2 and word in self.legal_terms:
                    keywords.append(word)
        
        # 去重并排序
        return sorted(list(set(keywords)))
    
    def _calculate_section_importance(
        self, 
        section_type: str, 
        content: str, 
        entities: List[Dict]
    ) -> float:
        """计算段落重要性分数"""
        base_scores = {
            'case_info': 0.9,
            'legal_basis': 0.95,
            'judgment_result': 1.0,
            'court_reasoning': 0.85,
            'case_facts': 0.8,
            'execution_info': 0.7
        }
        
        base_score = base_scores.get(section_type, 0.5)
        
        # 根据实体数量和内容长度调整
        entity_bonus = min(len(entities) * 0.05, 0.2)
        length_penalty = max(0, (len(content) - 1000) * -0.0001)
        
        final_score = base_score + entity_bonus + length_penalty
        return max(0.1, min(1.0, final_score))
    
    def _create_searchable_chunk(
        self, 
        section: LegalSection, 
        document: Dict, 
        doc_type: DocumentType
    ) -> Dict:
        """创建可搜索的数据块"""
        return {
            'content': section.content,
            'metadata': {
                'doc_id': document.get('id', ''),
                'doc_type': doc_type.value,
                'section_type': section.section_type,
                'title': document.get('title', ''),
                'entities': section.entities,
                'keywords': section.keywords,
                'importance_score': section.importance_score,
                'source': document.get('source', 'unknown')
            }
        }
    
    def _extract_judgment_metadata(self, content: str) -> Dict[str, Any]:
        """提取判决书元数据"""
        metadata = {}
        
        # 提取案件编号
        case_number_pattern = r'（(\d{4}）[^（）]*(\d+)号'
        case_match = re.search(case_number_pattern, content)
        if case_match:
            metadata['year'] = case_match.group(1)
            metadata['case_number'] = case_match.group(0)
        
        # 提取法院层级
        if '最高人民法院' in content:
            metadata['court_level'] = 'supreme'
        elif '高级人民法院' in content:
            metadata['court_level'] = 'high'
        elif '中级人民法院' in content:
            metadata['court_level'] = 'intermediate'
        elif '人民法院' in content:
            metadata['court_level'] = 'basic'
        
        # 提取案件类型
        if '民事' in content:
            metadata['case_type'] = 'civil'
        elif '刑事' in content:
            metadata['case_type'] = 'criminal'
        elif '行政' in content:
            metadata['case_type'] = 'administrative'
        
        return metadata
    
    def _extract_document_title(self, content: str) -> str:
        """提取文档标题"""
        # 尝试从内容开头提取标题
        lines = content.split('\n')[:5]
        for line in lines:
            line = line.strip()
            if line and ('判决书' in line or '裁定书' in line or '决定书' in line):
                return line
        
        return "未知法律文档"
    
    def _process_law_document(self, document: Dict) -> ProcessedDocument:
        """处理法律法规文档"""
        # 法律法规的处理逻辑
        content = document['content']
        
        # 提取条文
        articles = self._extract_law_articles(content)
        
        sections = []
        searchable_chunks = []
        
        for article in articles:
            entities = self._extract_legal_entities(article['content'])
            keywords = self._extract_legal_keywords(article['content'])
            
            section = LegalSection(
                section_type=f"article_{article['number']}",
                content=article['content'],
                entities=entities,
                keywords=keywords,
                importance_score=0.8
            )
            sections.append(section)
            
            chunk = self._create_searchable_chunk(
                section, document, DocumentType.LAW
            )
            searchable_chunks.append(chunk)
        
        return ProcessedDocument(
            doc_id=document.get('id', ''),
            doc_type=DocumentType.LAW,
            title=self._extract_document_title(content),
            sections=sections,
            metadata={'article_count': len(articles)},
            searchable_chunks=searchable_chunks
        )
    
    def _extract_law_articles(self, content: str) -> List[Dict]:
        """提取法条"""
        articles = []
        
        # 匹配法条模式
        article_pattern = r'第([一二三四五六七八九十百千万\d]+)条[^第]*?(?=第[一二三四五六七八九十百千万\d]+条|$)'
        matches = re.findall(article_pattern, content, re.DOTALL)
        
        for i, match in enumerate(matches):
            article_number = match[0] if isinstance(match, tuple) else match
            # 获取完整的条文内容
            full_match = re.search(
                rf'第{re.escape(article_number)}条([^第]*?)(?=第[一二三四五六七八九十百千万\d]+条|$)', 
                content, re.DOTALL
            )
            
            if full_match:
                article_content = f"第{article_number}条{full_match.group(1).strip()}"
                articles.append({
                    'number': article_number,
                    'content': article_content
                })
        
        return articles
    
    def _process_regulation_document(self, document: Dict) -> ProcessedDocument:
        """处理司法解释文档"""
        # 司法解释的处理逻辑
        return self._process_general_legal_document(document)
    
    def _process_general_legal_document(self, document: Dict) -> ProcessedDocument:
        """处理一般法律文档"""
        content = document['content']
        
        # 简单分段处理
        paragraphs = content.split('\n\n')
        
        sections = []
        searchable_chunks = []
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) < 50:  # 跳过太短的段落
                continue
                
            entities = self._extract_legal_entities(paragraph)
            keywords = self._extract_legal_keywords(paragraph)
            
            section = LegalSection(
                section_type=f"paragraph_{i}",
                content=paragraph.strip(),
                entities=entities,
                keywords=keywords,
                importance_score=0.6
            )
            sections.append(section)
            
            chunk = self._create_searchable_chunk(
                section, document, DocumentType.GENERAL
            )
            searchable_chunks.append(chunk)
        
        return ProcessedDocument(
            doc_id=document.get('id', ''),
            doc_type=DocumentType.GENERAL,
            title=self._extract_document_title(content),
            sections=sections,
            metadata={'paragraph_count': len(sections)},
            searchable_chunks=searchable_chunks
        )
    
    def _load_section_patterns(self) -> Dict[str, List[str]]:
        """加载段落识别模式"""
        return {
            'case_info': [
                r'案件编号[：，]',
                r'当事人[：，]',
                r'审理法院[：，]'
            ],
            'case_facts': [
                r'经审理查明',
                r'案件事实',
                r'查明事实'
            ],
            'court_reasoning': [
                r'本院认为',
                r'法院认为',
                r'审理认为'
            ]
        }
    
    def _load_legal_terms(self) -> set:
        """加载法律术语词典"""
        return {
            '合同', '协议', '违约', '侵权', '赔偿', '损害', '责任',
            '权利', '义务', '法律', '法规', '条例', '办法', '规定',
            '判决', '裁定', '调解', '和解', '执行', '强制执行',
            '上诉', '申请', '复议', '仲裁', '诉讼', '起诉', '应诉',
            '原告', '被告', '第三人', '当事人', '代理人', '律师',
            '证据', '证明', '举证', '质证', '认定', '采信',
            '民事', '刑事', '行政', '经济', '劳动', '婚姻', '继承'
        }
    
    def _load_court_hierarchy(self) -> Dict[str, int]:
        """加载法院层级"""
        return {
            '最高人民法院': 4,
            '高级人民法院': 3,
            '中级人民法院': 2,
            '基层人民法院': 1,
            '人民法院': 1
        }


# 使用示例
if __name__ == "__main__":
    processor = LegalDocumentProcessor()
    
    # 测试判决书
    test_judgment = {
        'id': 'judgment_001',
        'title': '民事判决书',
        'content': """
        北京市朝阳区人民法院
        民事判决书
        （2023）京0105民初12345号
        
        原告：张三，男，汉族，1980年1月1日出生
        被告：李四，女，汉族，1985年5月5日出生
        
        经审理查明：2023年1月1日，原告张三与被告李四签订房屋买卖合同，
        约定李四将其位于北京市朝阳区的房屋以100万元价格出售给张三。
        
        本院认为：根据《中华人民共和国民法典》第464条规定，
        合同是民事主体之间设立、变更、终止民事法律关系的协议。
        
        判决如下：
        一、被告李四于本判决生效之日起十日内协助原告张三办理房屋过户手续。
        二、案件受理费由被告承担。
        
        如不服本判决，可在判决书送达之日起十五日内上诉。
        """,
        'source': 'court_database'
    }
    
    # 处理文档
    result = processor.process_legal_document(test_judgment)
    
    print(f"文档类型: {result.doc_type}")
    print(f"标题: {result.title}")
    print(f"段落数量: {len(result.sections)}")
    print(f"可搜索块数量: {len(result.searchable_chunks)}")
    
    for section in result.sections[:3]:  # 显示前3个段落
        print(f"\n段落类型: {section.section_type}")
        print(f"重要性分数: {section.importance_score:.2f}")
        print(f"实体数量: {len(section.entities)}")
        print(f"关键词: {section.keywords[:5]}")
        print(f"内容预览: {section.content[:100]}...")