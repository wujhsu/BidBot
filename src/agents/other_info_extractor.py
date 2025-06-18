"""
合同信息提取节点
Contract information extraction node for the Langgraph workflow
"""

from typing import Dict, Any, List, Optional
from loguru import logger
from langchain_core.prompts import PromptTemplate
from src.models.data_models import GraphStateModel, ExtractedField, DocumentSource
from src.utils.llm_factory import LLMFactory
from src.utils.enhanced_retrieval import ContextualRetriever, SmartQueryRouter
from src.utils.improved_retrieval import ImprovedRetriever
import json
import re

class ContractInfoExtractor:
    """合同信息提取器"""

    def __init__(self):
        """初始化合同信息提取器"""
        self.llm = LLMFactory.create_llm()
        self.breach_liability_prompt = self._create_breach_liability_prompt()
        self.contract_prompt = self._create_contract_prompt()
        self.risk_prompt = self._create_risk_prompt()
        self.query_router = SmartQueryRouter()

    def _create_breach_liability_prompt(self) -> PromptTemplate:
        """创建违约责任提取提示模板"""
        template = """
你是一个专业的招投标文件分析专家。请从以下文档片段中提取违约责任相关信息。

文档片段：
{document_chunks}

请严格按照以下JSON格式提取信息：

{{
    "breach_liability": [
        {{
            "value": "违约责任的具体内容（按照原文提取，不要修改或总结）",
            "source_text": "原文片段",
            "confidence": 0.9
        }}
    ]
}}

提取要求：
1. 重点关注以下关键词及其同义词：
   - 违约责任、违约金、赔偿责任、损失赔偿
   - 合同违约、履约违约、延期违约
   - 责任承担、赔偿标准、违约处理
   - 法律责任、经济责任

2. 按照原文提取，保持原文的完整性和准确性
3. 不要对原文进行总结、修改或重新表述
4. 如果有多条违约责任条款，分别提取
5. 如果文档中没有明确的违约责任条款，返回空数组[]
6. 确保提取的内容与违约责任直接相关
"""
        return PromptTemplate(
            input_variables=["document_chunks"],
            template=template
        )

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

    def extract_breach_liability(self, state: GraphStateModel) -> GraphStateModel:
        """
        提取违约责任信息

        Args:
            state: 图状态

        Returns:
            GraphStateModel: 更新后的状态
        """
        try:
            logger.info("开始提取违约责任信息")

            if not state.vector_store:
                raise ValueError("向量存储未初始化")

            # 使用增强的检索策略
            contextual_retriever = ContextualRetriever(state.vector_store)

            # 构建违约责任查询
            breach_liability_queries = [
                "违约责任 违约金",
                "赔偿责任 损失赔偿",
                "合同违约 履约违约",
                "责任承担 赔偿标准",
                "法律责任 经济责任",
                "违约处理 违约条款"
            ]

            # 多轮增强检索
            enhanced_results = contextual_retriever.multi_round_retrieve(breach_liability_queries)

            # 提取文档内容和元数据
            relevant_chunks = []
            breach_metadata_map = {}
            breach_rag_docs = []
            for doc, vec_score, rerank_score in enhanced_results:
                chunk_content = doc.page_content
                relevant_chunks.append(chunk_content)
                # 保存文档内容与元数据的映射关系
                breach_metadata_map[chunk_content] = doc.metadata
                # 保存原始文档对象
                breach_rag_docs.append(doc)
                logger.debug(f"违约责任检索到文档片段，向量分数: {vec_score:.3f}, 重排序分数: {rerank_score:.3f}")

            # 限制长度
            unique_chunks = list(set(relevant_chunks))[:10]
            chunks_text = "\n\n---\n\n".join(unique_chunks)

            # 保存文档元数据映射和RAG文档，供后续使用
            self._breach_metadata_map = breach_metadata_map
            # 更新最新的RAG文档列表
            self._last_rag_docs = breach_rag_docs

            logger.info(f"违约责任检索完成，使用 {len(unique_chunks)} 个文档片段")

            # 调用LLM提取信息
            prompt = self.breach_liability_prompt.format(document_chunks=chunks_text)
            response = self.llm.invoke(prompt)

            # 解析响应
            breach_data = self._parse_llm_response(response.content if hasattr(response, 'content') else str(response))

            # 更新状态
            if breach_data:
                self._update_breach_liability(state, breach_data)

            logger.info("违约责任信息提取完成")

        except Exception as e:
            error_msg = f"违约责任信息提取失败: {str(e)}"
            logger.error(error_msg)
            state.error_messages.append(error_msg)

        return state

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
            
            # 使用改进的检索策略
            improved_retriever = ImprovedRetriever(state.vector_store)

            # 专门针对合同信息的检索
            enhanced_results = improved_retriever.retrieve_contract_info("合同条款 合同约定")

            # 提取文档内容
            relevant_chunks = []
            contract_rag_docs = []
            for doc, vec_score, rerank_score in enhanced_results:
                relevant_chunks.append(doc.page_content)
                # 保存原始文档对象
                contract_rag_docs.append(doc)
                logger.debug(f"合同信息检索到文档片段，向量分数: {vec_score:.3f}, 重排序分数: {rerank_score:.3f}")

            # 去重并限制长度
            unique_chunks = list(set(relevant_chunks))[:12]
            chunks_text = "\n\n---\n\n".join(unique_chunks)

            # 保存RAG文档，供后续使用
            self._last_rag_docs = contract_rag_docs
            
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
            
            # 使用改进的检索策略识别风险
            improved_retriever = ImprovedRetriever(state.vector_store)

            # 专门针对风险识别的检索
            enhanced_results = improved_retriever.retrieve_risk_info("违约 责任 赔偿 风险")

            # 提取文档内容
            relevant_chunks = []
            risk_rag_docs = []
            for doc, vec_score, rerank_score in enhanced_results:
                relevant_chunks.append(doc.page_content)
                # 保存原始文档对象
                risk_rag_docs.append(doc)
                logger.debug(f"风险识别检索到文档片段，向量分数: {vec_score:.3f}, 重排序分数: {rerank_score:.3f}")

            # 去重并限制长度
            unique_chunks = list(set(relevant_chunks))[:15]
            chunks_text = "\n\n---\n\n".join(unique_chunks)

            # 保存RAG文档，供后续使用
            self._last_rag_docs = risk_rag_docs
            
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

    def _extract_page_number(self, source_text: str) -> Optional[int]:
        """从来源文本中提取页码信息"""
        if not source_text:
            return None

        # 查找页码标记模式：--- 第X页 ---（PDF文件和改进后的DOCX文件）
        page_pattern = r'--- 第(\d+)页 ---'
        match = re.search(page_pattern, source_text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass

        # 查找段落标记模式：--- 第X段 ---（旧版DOCX文件处理方式，作为回退）
        para_pattern = r'--- 第(\d+)段 ---'
        match = re.search(para_pattern, source_text)
        if match:
            try:
                para_num = int(match.group(1))
                # 假设每页大约有20-30段，这里使用25段作为估算
                estimated_page = max(1, (para_num - 1) // 25 + 1)
                logger.warning(f"使用段落号估算页码：段落{para_num} -> 页码{estimated_page}")
                return estimated_page
            except ValueError:
                pass

        # 查找行号标记模式：--- 第X行 ---（TXT文件）
        line_pattern = r'--- 第(\d+)行 ---'
        match = re.search(line_pattern, source_text)
        if match:
            try:
                line_num = int(match.group(1))
                # 假设每页大约有50行
                estimated_page = max(1, (line_num - 1) // 50 + 1)
                logger.warning(f"使用行号估算页码：行{line_num} -> 页码{estimated_page}")
                return estimated_page
            except ValueError:
                pass

        return None

    def _extract_page_from_rag_docs(self, rag_docs: List) -> Optional[int]:
        """从RAG检索的文档中提取页码信息"""
        if not rag_docs:
            return None

        # 遍历所有检索到的文档，寻找页码信息
        for doc in rag_docs:
            # 检查文档元数据中的页码信息
            if hasattr(doc, 'metadata') and doc.metadata:
                page_number = doc.metadata.get('page_number')
                if page_number:
                    return page_number

            # 检查文档内容中的页码标记
            if hasattr(doc, 'page_content'):
                page_number = self._extract_page_number(doc.page_content)
                if page_number:
                    return page_number

        return None

    def _update_breach_liability(self, state: GraphStateModel, data: dict) -> None:
        """更新违约责任信息"""
        contract_info = state.analysis_result.contract_information

        if 'breach_liability' in data and isinstance(data['breach_liability'], list):
            breach_liability_fields = []
            for item in data['breach_liability']:
                if isinstance(item, dict) and 'value' in item:
                    source_text = item.get('source_text', '')

                    # 多层次页码提取策略
                    page_number = None

                    # 1. 优先从item中获取页码
                    page_number = item.get('page_number')

                    # 2. 从文档元数据映射中获取页码信息
                    if not page_number and hasattr(self, '_breach_metadata_map') and source_text:
                        for doc_content, metadata in self._breach_metadata_map.items():
                            if source_text in doc_content:
                                page_number = metadata.get('page_number')
                                break

                    # 3. 从来源文本中提取页码标记
                    if not page_number:
                        page_number = self._extract_page_number(source_text)

                    # 4. 从RAG检索的文档中提取页码（如果有的话）
                    if not page_number and hasattr(self, '_last_rag_docs'):
                        page_number = self._extract_page_from_rag_docs(self._last_rag_docs)

                    # 5. 如果仍然没有页码，记录警告并设置默认值
                    if not page_number:
                        logger.warning(f"无法为违约责任提取页码信息，来源文本: {source_text[:50]}...")
                        page_number = -1  # 设置默认页码为-1

                    breach_liability_field = ExtractedField(
                        value=item.get('value'),
                        source=DocumentSource(
                            source_text=source_text,
                            page_number=page_number
                        ),
                        confidence=item.get('confidence', 0.5)
                    )
                    breach_liability_fields.append(breach_liability_field)
            contract_info.breach_liability = breach_liability_fields

    def _update_contract_info(self, state: GraphStateModel, data: dict) -> None:
        """更新合同信息"""
        contract_info = state.analysis_result.contract_information

        # 更新合同条款
        if 'contract_terms' in data and isinstance(data['contract_terms'], list):
            contract_terms = []
            for item in data['contract_terms']:
                if isinstance(item, dict) and 'value' in item:
                    source_text = item.get('source_text', '')

                    # 多层次页码提取策略
                    page_number = None

                    # 1. 优先从item中获取页码
                    page_number = item.get('page_number')

                    # 2. 从来源文本中提取页码标记
                    if not page_number:
                        page_number = self._extract_page_number(source_text)

                    # 3. 从RAG检索的文档中提取页码（如果有的话）
                    if not page_number and hasattr(self, '_last_rag_docs'):
                        page_number = self._extract_page_from_rag_docs(self._last_rag_docs)

                    # 4. 如果仍然没有页码，记录警告并设置默认值
                    if not page_number:
                        logger.warning(f"无法为合同条款提取页码信息，来源文本: {source_text[:50]}...")
                        page_number = -1  # 设置默认页码为-1

                    contract_terms.append(ExtractedField(
                        value=item.get('value'),
                        source=DocumentSource(
                            source_text=source_text,
                            page_number=page_number
                        ),
                        confidence=item.get('confidence', 0.5)
                    ))
            contract_info.contract_terms = contract_terms

        # 更新其他单项信息
        single_fields = [
            'payment_terms', 'delivery_requirements', 'bid_validity',
            'intellectual_property', 'confidentiality'
        ]

        for field_name in single_fields:
            if field_name in data and isinstance(data[field_name], dict):
                field_data = data[field_name]
                source_text = field_data.get('source_text', '')

                # 多层次页码提取策略
                page_number = None

                # 1. 优先从field_data中获取页码
                page_number = field_data.get('page_number')

                # 2. 从来源文本中提取页码标记
                if not page_number:
                    page_number = self._extract_page_number(source_text)

                # 3. 从RAG检索的文档中提取页码（如果有的话）
                if not page_number and hasattr(self, '_last_rag_docs'):
                    page_number = self._extract_page_from_rag_docs(self._last_rag_docs)

                # 4. 如果仍然没有页码，记录警告并设置默认值
                if not page_number:
                    logger.warning(f"无法为字段 {field_name} 提取页码信息，来源文本: {source_text[:50]}...")
                    page_number = -1  # 设置默认页码为-1

                setattr(contract_info, field_name, ExtractedField(
                    value=field_data.get('value'),
                    source=DocumentSource(
                        source_text=source_text,
                        page_number=page_number
                    ),
                    confidence=field_data.get('confidence', 0.5)
                ))
    
    def _update_risk_warnings(self, state: GraphStateModel, risk_warnings: List[dict]) -> None:
        """更新风险警告"""
        contract_info = state.analysis_result.contract_information

        risk_fields = []
        for item in risk_warnings:
            if isinstance(item, dict) and 'value' in item:
                source_text = item.get('source_text', '')

                # 多层次页码提取策略
                page_number = None

                # 1. 优先从item中获取页码
                page_number = item.get('page_number')

                # 2. 从来源文本中提取页码标记
                if not page_number:
                    page_number = self._extract_page_number(source_text)

                # 3. 从RAG检索的文档中提取页码（如果有的话）
                if not page_number and hasattr(self, '_last_rag_docs'):
                    page_number = self._extract_page_from_rag_docs(self._last_rag_docs)

                # 4. 如果仍然没有页码，记录警告并设置默认值
                if not page_number:
                    logger.warning(f"无法为风险警告提取页码信息，来源文本: {source_text[:50]}...")
                    page_number = -1  # 设置默认页码为-1

                risk_field = ExtractedField(
                    value=item.get('value'),
                    source=DocumentSource(
                        source_text=source_text,
                        page_number=page_number
                    ),
                    confidence=item.get('confidence', 0.5),
                    notes=item.get('notes')
                )
                risk_fields.append(risk_field)

        contract_info.risk_warnings = risk_fields

def create_contract_info_extractor_node():
    """创建合同信息提取节点函数"""
    extractor = ContractInfoExtractor()
    
    def contract_info_extractor_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """合同信息提取节点函数"""
        # 转换为GraphStateModel对象
        graph_state = GraphStateModel(**state)

        # 执行合同信息提取
        graph_state = extractor.extract_breach_liability(graph_state)
        graph_state = extractor.extract_contract_info(graph_state)
        graph_state = extractor.identify_risks(graph_state)

        # 转换回字典格式
        return graph_state.model_dump()

    return contract_info_extractor_node
