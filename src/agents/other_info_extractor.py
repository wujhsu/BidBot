"""
其他重要信息提取节点
Other important information extraction node for the Langgraph workflow
"""

from typing import Dict, Any, List
from loguru import logger
from langchain_core.prompts import PromptTemplate
from src.models.data_models import GraphStateModel, ExtractedField, DocumentSource
from src.utils.llm_factory import LLMFactory
import json
import re

class OtherInfoExtractor:
    """其他重要信息提取器"""
    
    def __init__(self):
        """初始化其他信息提取器"""
        self.llm = LLMFactory.create_llm()
        self.contract_prompt = self._create_contract_prompt()
        self.risk_prompt = self._create_risk_prompt()
    
    def _create_contract_prompt(self) -> PromptTemplate:
        """创建合同条款提取提示模板"""
        template = """
你是一个专业的招投标文件分析专家。请从以下文档片段中提取合同相关的重要信息。

文档片段：
{document_chunks}

请严格按照以下JSON格式提取信息：

{{
    "contract_terms": [
        {{
            "value": "合同主要条款或特殊约定",
            "source_text": "原文片段",
            "confidence": 0.9
        }}
    ],
    "payment_terms": {{
        "value": "付款方式与周期（如预付款、进度款、质保金等）",
        "source_text": "原文片段",
        "confidence": 0.9
    }},
    "delivery_requirements": {{
        "value": "项目完成期限和交付要求",
        "source_text": "原文片段",
        "confidence": 0.9
    }},
    "bid_validity": {{
        "value": "投标有效期",
        "source_text": "原文片段",
        "confidence": 0.9
    }},
    "intellectual_property": {{
        "value": "知识产权归属规定",
        "source_text": "原文片段",
        "confidence": 0.9
    }},
    "confidentiality": {{
        "value": "保密协议要求",
        "source_text": "原文片段",
        "confidence": 0.9
    }}
}}

注意事项：
1. 重点关注对投标人有重大影响的条款
2. 准确提取时间期限和金额比例
3. 严格忠于原文，不要添加主观判断
4. 如果某项信息未找到，填写"招标文件中未提及"
"""
        return PromptTemplate(
            input_variables=["document_chunks"],
            template=template
        )
    
    def _create_risk_prompt(self) -> PromptTemplate:
        """创建风险识别提示模板"""
        template = """
你是一个专业的招投标风险分析专家。请从以下文档片段中识别潜在的风险点。

文档片段：
{document_chunks}

请严格按照以下JSON格式提取风险信息：

{{
    "risk_warnings": [
        {{
            "value": "具体的风险点描述（如不明确的条款、苛刻的条件、可能导致废标的隐藏条款等）",
            "source_text": "原文片段",
            "confidence": 0.9,
            "notes": "风险分析说明"
        }}
    ]
}}

重点关注以下类型的风险：
1. 模糊不清或有歧义的条款
2. 过于苛刻的技术要求或商务条件
3. 可能导致废标的隐藏条款
4. 不合理的时间要求或付款条件
5. 过高的保证金或违约金
6. 不公平的评分标准
7. 限制性的资格条件

注意事项：
1. 客观分析，基于文档内容
2. 重点关注对投标人不利的条款
3. 提供具体的风险分析说明
"""
        return PromptTemplate(
            input_variables=["document_chunks"],
            template=template
        )
    
    def extract_contract_info(self, state: GraphStateModel) -> GraphStateModel:
        """
        提取合同相关信息

        Args:
            state: 图状态

        Returns:
            GraphStateModel: 更新后的状态
        """
        try:
            logger.info("开始提取合同相关信息")
            
            if not state.vector_store:
                raise ValueError("向量存储未初始化")
            
            # 搜索相关文档片段
            contract_queries = [
                "合同条款 合同约定",
                "付款方式 付款条件 预付款 进度款",
                "交付期限 完成时间 工期",
                "投标有效期 有效期",
                "知识产权 专利权 著作权",
                "保密协议 保密条款 商业秘密"
            ]
            
            relevant_chunks = []
            for query in contract_queries:
                docs = state.vector_store.similarity_search(query, k=3)
                relevant_chunks.extend([doc.page_content for doc in docs])
            
            # 去重并限制长度
            unique_chunks = list(set(relevant_chunks))[:12]
            chunks_text = "\n\n---\n\n".join(unique_chunks)
            
            # 调用LLM提取信息
            prompt = self.contract_prompt.format(document_chunks=chunks_text)
            response = self.llm.invoke(prompt)
            
            # 解析响应
            contract_data = self._parse_llm_response(response.content if hasattr(response, 'content') else str(response))
            
            # 更新状态
            if contract_data:
                self._update_contract_info(state, contract_data)
            
            logger.info("合同相关信息提取完成")
            
        except Exception as e:
            error_msg = f"合同信息提取失败: {str(e)}"
            logger.error(error_msg)
            state.error_messages.append(error_msg)
        
        return state
    
    def identify_risks(self, state: GraphStateModel) -> GraphStateModel:
        """
        识别潜在风险

        Args:
            state: 图状态

        Returns:
            GraphStateModel: 更新后的状态
        """
        try:
            logger.info("开始识别潜在风险")
            
            if not state.vector_store:
                raise ValueError("向量存储未初始化")
            
            # 搜索可能包含风险的文档片段
            risk_queries = [
                "违约 责任 赔偿 罚款",
                "保证金 履约保证金",
                "废标 否决 取消资格",
                "特殊要求 特别约定",
                "技术要求 性能指标",
                "时间要求 期限 工期",
                "★ * 重要 关键"
            ]
            
            relevant_chunks = []
            for query in risk_queries:
                docs = state.vector_store.similarity_search(query, k=4)
                relevant_chunks.extend([doc.page_content for doc in docs])
            
            # 去重并限制长度
            unique_chunks = list(set(relevant_chunks))[:15]
            chunks_text = "\n\n---\n\n".join(unique_chunks)
            
            # 调用LLM识别风险
            prompt = self.risk_prompt.format(document_chunks=chunks_text)
            response = self.llm.invoke(prompt)
            
            # 解析响应
            risk_data = self._parse_llm_response(response.content if hasattr(response, 'content') else str(response))
            
            # 更新状态
            if risk_data and 'risk_warnings' in risk_data:
                self._update_risk_warnings(state, risk_data['risk_warnings'])
            
            state.current_step = "other_info_extracted"
            logger.info("风险识别完成")
            
        except Exception as e:
            error_msg = f"风险识别失败: {str(e)}"
            logger.error(error_msg)
            state.error_messages.append(error_msg)
        
        return state
    
    def _parse_llm_response(self, response: str) -> dict:
        """解析LLM响应"""
        try:
            # 尝试提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                logger.warning("未找到有效的JSON响应")
                return {}
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return {}
    
    def _update_contract_info(self, state: GraphStateModel, data: dict) -> None:
        """更新合同信息"""
        other_info = state.analysis_result.other_information
        
        # 更新合同条款
        if 'contract_terms' in data and isinstance(data['contract_terms'], list):
            other_info.contract_terms = [
                ExtractedField(
                    value=item.get('value'),
                    source=DocumentSource(source_text=item.get('source_text')),
                    confidence=item.get('confidence', 0.5)
                )
                for item in data['contract_terms']
                if isinstance(item, dict) and 'value' in item
            ]
        
        # 更新其他单项信息
        single_fields = [
            'payment_terms', 'delivery_requirements', 'bid_validity',
            'intellectual_property', 'confidentiality'
        ]
        
        for field_name in single_fields:
            if field_name in data and isinstance(data[field_name], dict):
                field_data = data[field_name]
                setattr(other_info, field_name, ExtractedField(
                    value=field_data.get('value'),
                    source=DocumentSource(source_text=field_data.get('source_text')),
                    confidence=field_data.get('confidence', 0.5)
                ))
    
    def _update_risk_warnings(self, state: GraphStateModel, risk_warnings: List[dict]) -> None:
        """更新风险警告"""
        other_info = state.analysis_result.other_information
        
        risk_fields = []
        for item in risk_warnings:
            if isinstance(item, dict) and 'value' in item:
                risk_field = ExtractedField(
                    value=item.get('value'),
                    source=DocumentSource(source_text=item.get('source_text')),
                    confidence=item.get('confidence', 0.5),
                    notes=item.get('notes')
                )
                risk_fields.append(risk_field)
        
        other_info.risk_warnings = risk_fields

def create_other_info_extractor_node():
    """创建其他信息提取节点函数"""
    extractor = OtherInfoExtractor()
    
    def other_info_extractor_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """其他信息提取节点函数"""
        # 转换为GraphStateModel对象
        graph_state = GraphStateModel(**state)
        
        # 执行其他信息提取
        graph_state = extractor.extract_contract_info(graph_state)
        graph_state = extractor.identify_risks(graph_state)
        
        # 转换回字典格式
        return graph_state.model_dump()
    
    return other_info_extractor_node
