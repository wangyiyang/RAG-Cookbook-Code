"""
法律实体识别器
实现专业级法律实体识别、标准化和重要性评估
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import jieba
import jieba.posseg as pseg


class EntityType(Enum):
    """法律实体类型"""
    LAW = "LAW"                    # 法律法规
    COURT = "COURT"                # 法院
    PERSON = "PERSON"              # 当事人
    CASE_NUMBER = "CASE_NUMBER"    # 案件编号
    MONEY = "MONEY"                # 金额
    DATE = "DATE"                  # 日期
    LOCATION = "LOCATION"          # 地点
    LEGAL_TERM = "LEGAL_TERM"      # 法律术语
    ORGANIZATION = "ORGANIZATION"   # 机构组织


@dataclass
class LegalEntity:
    """法律实体"""
    text: str
    entity_type: EntityType
    start_pos: int
    end_pos: int
    confidence: float
    normalized_text: str = ""
    legal_significance: float = 0.0
    context: str = ""


class LegalEntityRecognizer:
    """法律实体识别器"""
    
    def __init__(self):
        self.entity_patterns = self._load_entity_patterns()
        self.legal_terms = self._load_legal_terms()
        self.court_hierarchy = self._load_court_hierarchy()
        self.law_abbreviations = self._load_law_abbreviations()
        
        # 初始化jieba分词
        jieba.initialize()
        
    def extract_legal_entities_like_expert(
        self, 
        text: str, 
        context: str = ""
    ) -> List[LegalEntity]:
        """像法律专家一样提取实体"""
        entities = []
        
        # 1. 基于规则的实体识别
        rule_based_entities = self._extract_with_rules(text)
        entities.extend(rule_based_entities)
        
        # 2. 基于词典的实体识别
        dict_based_entities = self._extract_with_dictionary(text)
        entities.extend(dict_based_entities)
        
        # 3. 实体去重和合并
        merged_entities = self._merge_overlapping_entities(entities)
        
        # 4. 实体标准化
        standardized_entities = self._standardize_entities(merged_entities)
        
        # 5. 评估法律重要性
        for entity in standardized_entities:
            entity.legal_significance = self._assess_legal_importance(entity, text)
            entity.context = context
        
        # 6. 按重要性排序
        standardized_entities.sort(key=lambda x: x.legal_significance, reverse=True)
        
        return standardized_entities
    
    def _extract_with_rules(self, text: str) -> List[LegalEntity]:
        """基于规则的实体提取"""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entity = LegalEntity(
                        text=match.group(0),
                        entity_type=EntityType(entity_type),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.9  # 规则匹配高置信度
                    )
                    entities.append(entity)
        
        return entities
    
    def _extract_with_dictionary(self, text: str) -> List[LegalEntity]:
        """基于词典的实体提取"""
        entities = []
        words = pseg.cut(text)
        
        current_pos = 0
        for word, flag in words:
            word_start = text.find(word, current_pos)
            word_end = word_start + len(word)
            
            # 法律术语识别
            if word in self.legal_terms:
                entity = LegalEntity(
                    text=word,
                    entity_type=EntityType.LEGAL_TERM,
                    start_pos=word_start,
                    end_pos=word_end,
                    confidence=0.8
                )
                entities.append(entity)
            
            # 人名识别
            elif flag == 'nr' and len(word) >= 2:
                entity = LegalEntity(
                    text=word,
                    entity_type=EntityType.PERSON,
                    start_pos=word_start,
                    end_pos=word_end,
                    confidence=0.7
                )
                entities.append(entity)
            
            current_pos = word_end
        
        return entities
    
    def _merge_overlapping_entities(self, entities: List[LegalEntity]) -> List[LegalEntity]:
        """合并重叠实体"""
        if not entities:
            return []
        
        # 按起始位置排序
        entities.sort(key=lambda x: x.start_pos)
        
        merged = []
        current = entities[0]
        
        for next_entity in entities[1:]:
            # 检查是否重叠
            if (next_entity.start_pos <= current.end_pos and 
                next_entity.end_pos > current.start_pos):
                
                # 选择更重要的实体类型或更长的文本
                if (self._get_entity_priority(next_entity.entity_type) >
                    self._get_entity_priority(current.entity_type) or
                    len(next_entity.text) > len(current.text)):
                    current = next_entity
            else:
                merged.append(current)
                current = next_entity
        
        merged.append(current)
        return merged
    
    def _get_entity_priority(self, entity_type: EntityType) -> int:
        """获取实体类型优先级"""
        priority_map = {
            EntityType.CASE_NUMBER: 10,
            EntityType.LAW: 9,
            EntityType.COURT: 8,
            EntityType.MONEY: 7,
            EntityType.DATE: 6,
            EntityType.LEGAL_TERM: 5,
            EntityType.PERSON: 4,
            EntityType.ORGANIZATION: 3,
            EntityType.LOCATION: 2
        }
        return priority_map.get(entity_type, 1)
    
    def _standardize_entities(self, entities: List[LegalEntity]) -> List[LegalEntity]:
        """实体标准化处理"""
        standardized = []
        
        for entity in entities:
            normalized_text = entity.text
            
            if entity.entity_type == EntityType.LAW:
                normalized_text = self._normalize_law_reference(entity.text)
            elif entity.entity_type == EntityType.COURT:
                normalized_text = self._normalize_court_name(entity.text)
            elif entity.entity_type == EntityType.CASE_NUMBER:
                normalized_text = self._normalize_case_number(entity.text)
            elif entity.entity_type == EntityType.MONEY:
                normalized_text = self._normalize_money_amount(entity.text)
            elif entity.entity_type == EntityType.DATE:
                normalized_text = self._normalize_date(entity.text)
            
            entity.normalized_text = normalized_text
            standardized.append(entity)
        
        return standardized
    
    def _normalize_law_reference(self, law_text: str) -> str:
        """标准化法律引用"""
        # 移除书名号
        clean_text = re.sub(r'[《》]', '', law_text)
        
        # 处理常见的法律简称
        if clean_text in self.law_abbreviations:
            return self.law_abbreviations[clean_text]
        
        # 标准化常见法律名称
        if '中华人民共和国' in clean_text:
            clean_text = clean_text.replace('中华人民共和国', '')
        
        return clean_text.strip()
    
    def _normalize_court_name(self, court_text: str) -> str:
        """标准化法院名称"""
        # 提取法院层级
        if '最高人民法院' in court_text:
            level = '最高法院'
        elif '高级人民法院' in court_text:
            level = '高级法院'
        elif '中级人民法院' in court_text:
            level = '中级法院'
        elif '基层人民法院' in court_text or '人民法院' in court_text:
            level = '基层法院'
        else:
            level = '法院'
        
        # 提取地区
        region_match = re.search(r'(\w+(?:省|市|区|县))', court_text)
        region = region_match.group(1) if region_match else ''
        
        if region:
            return f"{region}{level}"
        else:
            return level
    
    def _normalize_case_number(self, case_num: str) -> str:
        """标准化案件编号"""
        # 提取案件编号的关键信息
        pattern = r'（(\d{4}）([^）]*)(\d+)号'
        match = re.search(pattern, case_num)
        
        if match:
            year = match.group(1)
            court_code = match.group(2)
            case_id = match.group(3)
            
            return {
                'year': year,
                'court_code': court_code,
                'case_id': case_id,
                'full_number': case_num
            }
        
        return case_num
    
    def _normalize_money_amount(self, money_text: str) -> str:
        """标准化金额"""
        # 提取数字
        number_pattern = r'(\d+(?:,\d{3})*(?:\.\d{2})?)'
        number_match = re.search(number_pattern, money_text)
        
        if number_match:
            amount = number_match.group(1).replace(',', '')
            
            # 处理单位
            if '万' in money_text:
                amount = str(float(amount) * 10000)
            elif '千' in money_text:
                amount = str(float(amount) * 1000)
            elif '百' in money_text:
                amount = str(float(amount) * 100)
            
            return f"{amount}元"
        
        return money_text
    
    def _normalize_date(self, date_text: str) -> str:
        """标准化日期"""
        # 标准化为 YYYY-MM-DD 格式
        pattern = r'(\d{4})年(\d{1,2})月(\d{1,2})日'
        match = re.search(pattern, date_text)
        
        if match:
            year = match.group(1)
            month = match.group(2).zfill(2)
            day = match.group(3).zfill(2)
            return f"{year}-{month}-{day}"
        
        return date_text
    
    def _assess_legal_importance(self, entity: LegalEntity, full_text: str) -> float:
        """评估法律重要性"""
        base_importance = {
            EntityType.CASE_NUMBER: 1.0,
            EntityType.LAW: 0.95,
            EntityType.COURT: 0.9,
            EntityType.MONEY: 0.8,
            EntityType.LEGAL_TERM: 0.7,
            EntityType.DATE: 0.6,
            EntityType.PERSON: 0.5,
            EntityType.ORGANIZATION: 0.4,
            EntityType.LOCATION: 0.3
        }
        
        importance = base_importance.get(entity.entity_type, 0.2)
        
        # 根据上下文调整重要性
        context_window = 100
        start = max(0, entity.start_pos - context_window)
        end = min(len(full_text), entity.end_pos + context_window)
        context = full_text[start:end]
        
        # 关键短语加权
        importance_keywords = [
            '依照', '根据', '违反', '适用', '判决', '裁定',
            '认定', '确认', '支持', '驳回', '赔偿', '承担'
        ]
        
        keyword_count = sum(1 for keyword in importance_keywords if keyword in context)
        importance += keyword_count * 0.05
        
        # 实体频次加权
        entity_count = full_text.count(entity.text)
        if entity_count > 1:
            importance += min(entity_count * 0.02, 0.1)
        
        return min(importance, 1.0)
    
    def _load_entity_patterns(self) -> Dict[str, List[str]]:
        """加载实体识别模式"""
        return {
            'LAW': [
                r'《[^》]+法》',
                r'《[^》]*条例》',
                r'《[^》]*规定》',
                r'《[^》]*办法》',
                r'《中华人民共和国[^》]+》'
            ],
            'COURT': [
                r'\w+人民法院',
                r'最高人民法院',
                r'\w+高级人民法院',
                r'\w+中级人民法院',
                r'\w+基层人民法院'
            ],
            'CASE_NUMBER': [
                r'（\d{4}）[^（）]*\d+号',
                r'\d{4}年[^第]*第\d+号'
            ],
            'MONEY': [
                r'\d+(?:,\d{3})*(?:\.\d{2})?[万千百十]?元',
                r'人民币\d+(?:,\d{3})*(?:\.\d{2})?[万千百十]?元',
                r'\d+[万千百十]元'
            ],
            'DATE': [
                r'\d{4}年\d{1,2}月\d{1,2}日',
                r'\d{4}\.\d{1,2}\.\d{1,2}',
                r'\d{4}-\d{1,2}-\d{1,2}'
            ],
            'PERSON': [
                r'(?:原告|被告|申请人|被申请人|上诉人|被上诉人)[：，]([^，。；\n]+)',
                r'(?:法定代表人|委托代理人)[：，]([^，。；\n]+)'
            ],
            'ORGANIZATION': [
                r'\w+有限公司',
                r'\w+股份有限公司',
                r'\w+企业',
                r'\w+机构',
                r'\w+组织'
            ]
        }
    
    def _load_legal_terms(self) -> set:
        """加载法律术语词典"""
        return {
            # 民事法律术语
            '合同', '协议', '违约', '违约责任', '合同履行', '合同解除',
            '侵权', '侵权责任', '损害赔偿', '精神损害', '财产损失',
            '债权', '债务', '债权人', '债务人', '担保', '抵押', '质押',
            '物权', '所有权', '用益物权', '担保物权', '占有',
            '婚姻', '离婚', '夫妻财产', '子女抚养', '赡养',
            '继承', '遗产', '遗嘱', '法定继承', '遗赠',
            
            # 刑事法律术语
            '犯罪', '犯罪构成', '犯罪主体', '犯罪客体',
            '故意', '过失', '故意犯罪', '过失犯罪',
            '正当防卫', '紧急避险', '犯罪预备', '犯罪未遂',
            '主犯', '从犯', '胁从犯', '教唆犯',
            '刑罚', '主刑', '附加刑', '有期徒刑', '无期徒刑', '死刑',
            '缓刑', '假释', '数罪并罚',
            
            # 行政法律术语
            '行政行为', '行政处罚', '行政许可', '行政强制',
            '行政复议', '行政诉讼', '行政赔偿',
            '公务员', '国家机关', '行政机关',
            
            # 程序法术语
            '起诉', '应诉', '反诉', '第三人', '共同诉讼',
            '管辖', '级别管辖', '地域管辖', '专属管辖',
            '举证', '质证', '认证', '证据', '证明',
            '调解', '和解', '判决', '裁定', '决定',
            '上诉', '抗诉', '申请再审', '执行',
            
            # 其他重要术语
            '法律关系', '法律事实', '法律后果',
            '权利', '义务', '责任', '法律责任',
            '时效', '诉讼时效', '除斥期间',
            '善意', '恶意', '善意取得', '诚实信用',
            '公序良俗', '公共利益', '国家利益'
        }
    
    def _load_court_hierarchy(self) -> Dict[str, int]:
        """加载法院层级信息"""
        return {
            '最高人民法院': 4,
            '高级人民法院': 3,
            '中级人民法院': 2,
            '基层人民法院': 1,
            '人民法院': 1
        }
    
    def _load_law_abbreviations(self) -> Dict[str, str]:
        """加载法律简称映射"""
        return {
            '民法典': '中华人民共和国民法典',
            '刑法': '中华人民共和国刑法',
            '行政法': '中华人民共和国行政法',
            '合同法': '中华人民共和国合同法',
            '侵权责任法': '中华人民共和国侵权责任法',
            '物权法': '中华人民共和国物权法',
            '公司法': '中华人民共和国公司法',
            '劳动法': '中华人民共和国劳动法',
            '劳动合同法': '中华人民共和国劳动合同法',
            '婚姻法': '中华人民共和国婚姻法',
            '继承法': '中华人民共和国继承法',
            '民事诉讼法': '中华人民共和国民事诉讼法',
            '刑事诉讼法': '中华人民共和国刑事诉讼法',
            '行政诉讼法': '中华人民共和国行政诉讼法'
        }
    
    def extract_entities_summary(self, text: str) -> Dict[str, Any]:
        """提取实体摘要信息"""
        entities = self.extract_legal_entities_like_expert(text)
        
        # 按类型分组
        by_type = {}
        for entity in entities:
            entity_type = entity.entity_type.value
            if entity_type not in by_type:
                by_type[entity_type] = []
            by_type[entity_type].append({
                'text': entity.text,
                'normalized': entity.normalized_text,
                'confidence': entity.confidence,
                'importance': entity.legal_significance
            })
        
        # 统计信息
        stats = {
            'total_entities': len(entities),
            'high_importance_count': len([e for e in entities if e.legal_significance > 0.8]),
            'law_references': len(by_type.get('LAW', [])),
            'case_numbers': len(by_type.get('CASE_NUMBER', [])),
            'courts': len(by_type.get('COURT', [])),
            'parties': len(by_type.get('PERSON', []))
        }
        
        return {
            'entities_by_type': by_type,
            'statistics': stats,
            'top_entities': [
                {
                    'text': e.text,
                    'type': e.entity_type.value,
                    'importance': e.legal_significance
                }
                for e in entities[:10]  # 前10个最重要的实体
            ]
        }


# 使用示例
if __name__ == "__main__":
    recognizer = LegalEntityRecognizer()
    
    # 测试文本
    test_text = """
    北京市朝阳区人民法院民事判决书（2023）京0105民初12345号
    原告：张三，男，汉族，1980年1月1日出生，住北京市朝阳区
    被告：李四有限公司，法定代表人王五
    
    经审理查明：2023年1月1日，原告与被告签订《房屋买卖合同》，
    约定被告将其房屋以100万元价格出售给原告。
    根据《中华人民共和国民法典》第464条规定，合同是民事主体之间设立、
    变更、终止民事法律关系的协议。
    
    本院认为，被告违约，应承担违约责任，赔偿原告损失50万元。
    """
    
    # 提取实体
    entities = recognizer.extract_legal_entities_like_expert(test_text)
    
    print("=== 法律实体识别结果 ===")
    for entity in entities:
        print(f"实体: {entity.text}")
        print(f"类型: {entity.entity_type.value}")
        print(f"标准化: {entity.normalized_text}")
        print(f"置信度: {entity.confidence:.2f}")
        print(f"重要性: {entity.legal_significance:.2f}")
        print("-" * 30)
    
    # 获取摘要
    summary = recognizer.extract_entities_summary(test_text)
    print("\n=== 实体摘要统计 ===")
    print(json.dumps(summary['statistics'], ensure_ascii=False, indent=2))
    
    print("\n=== 最重要实体 ===")
    for entity in summary['top_entities'][:5]:
        print(f"{entity['text']} ({entity['type']}) - 重要性: {entity['importance']:.2f}")