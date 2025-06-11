"""
基础信息提取节点
Basic information extraction node for the Langgraph workflow
"""

from typing import Dict, Any, List, Optional
from loguru import logger
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from src.models.data_models import GraphState, GraphStateModel, ExtractedField, DocumentSource, QualificationCriteria
from src.utils.llm_factory import LLMFactory
from src.utils.enhanced_retrieval import ContextualRetriever, SmartQueryRouter
import json
import re

class BasicInfoExtractor:
    """基础信息提取器"""
    
    def __init__(self):
        """初始化基础信息提取器"""
        self.llm = LLMFactory.create_llm()
        self.extraction_prompt = self._create_extraction_prompt()
        self.query_router = SmartQueryRouter()

    def _create_extraction_prompt(self) -> PromptTemplate:
        """创建信息提取提示模板"""
        template = """
你是一个专业的招投标文件分析专家。请从以下文档片段中提取基础信息。

文档片段：
{document_chunks}

请严格按照以下JSON格式提取信息，如果某项信息在文档中未找到，请在value字段填写"招标文件中未提及"：

{{
    "project_name": {{
        "value": "项目名称",
        "source_text": "原文片段",
        "confidence": 0.9
    }},
    "tender_number": {{
        "value": "招标编号",
        "source_text": "原文片段",
        "confidence": 0.9
    }},
    "budget_amount": {{
        "value": "预算金额（包含币种）",
        "source_text": "原文片段",
        "confidence": 0.9
    }},
    "bid_deadline": {{
        "value": "投标截止时间（具体日期和时间）",
        "source_text": "原文片段",
        "confidence": 0.9
    }},
    "bid_opening_time": {{
        "value": "开标时间（具体日期和时间）",
        "source_text": "原文片段",
        "confidence": 0.9
    }},
    "bid_bond_amount": {{
        "value": "投标保证金金额（包含币种）",
        "source_text": "原文片段",
        "confidence": 0.9
    }},
    "bid_bond_account": {{
        "value": "投标保证金缴纳账户信息（户名、账号、开户行）",
        "source_text": "原文片段",
        "confidence": 0.9
    }},
    "purchaser_name": {{
        "value": "采购人名称",
        "source_text": "原文片段",
        "confidence": 0.9
    }},
    "purchaser_contact": {{
        "value": "采购人联系方式",
        "source_text": "原文片段",
        "confidence": 0.9
    }},
    "agent_name": {{
        "value": "采购代理机构名称",
        "source_text": "原文片段",
        "confidence": 0.9
    }},
    "agent_contact": {{
        "value": "采购代理机构联系人及联系方式",
        "source_text": "原文片段",
        "confidence": 0.9
    }}
}}

注意事项：
1. 严格忠于原文，不要添加任何主观判断
2. confidence字段表示提取信息的置信度（0-1之间）
3. 如果信息不完整或模糊，请在confidence中体现
4. 必须返回有效的JSON格式
"""
        return PromptTemplate(
            input_variables=["document_chunks"],
            template=template
        )
    
    def _create_qualification_prompt(self) -> PromptTemplate:
        """创建资格审查条件提取提示模板"""
        template = """
你是一个专业的招投标文件分析专家。请从以下文档片段中提取资格审查硬性条件。

文档片段：
{document_chunks}

请严格按照以下JSON格式提取信息：

{{
    "company_certifications": [
        {{
            "value": "具体的企业资质要求",
            "source_text": "原文片段",
            "confidence": 0.9
        }}
    ],
    "project_experience": [
        {{
            "value": "类似项目业绩要求（包括项目数量、金额、时间范围等）",
            "source_text": "原文片段",
            "confidence": 0.9
        }}
    ],
    "team_requirements": [
        {{
            "value": "项目团队人员要求（岗位、资格证书、经验年限等）",
            "source_text": "原文片段",
            "confidence": 0.9
        }}
    ],
    "other_requirements": [
        {{
            "value": "其他硬性要求（财务状况、体系认证等）",
            "source_text": "原文片段",
            "confidence": 0.9
        }}
    ]
}}

注意事项：
1. 重点关注带有"必须"、"应当"、"要求"等强制性词汇的条款
2. 逐条列出所有硬性条件，不要遗漏
3. 严格忠于原文，不要添加主观判断
4. 如果某类要求未找到，返回空数组[]
"""
        return PromptTemplate(
            input_variables=["document_chunks"],
            template=template
        )
    
    def extract_basic_info(self, state: GraphStateModel) -> GraphStateModel:
        """
        提取基础信息
        
        Args:
            state: 图状态
            
        Returns:
            GraphState: 更新后的状态
        """
        try:
            logger.info("开始提取基础信息")
            
            if not state.vector_store:
                raise ValueError("向量存储未初始化")
            
            # 使用增强的检索策略
            contextual_retriever = ContextualRetriever(state.vector_store)

            # 智能查询路由
            main_query = "项目名称 招标编号 基础信息"
            query_category = self.query_router.route_query(main_query)

            # 获取基础信息相关的查询
            basic_info_queries = self.query_router.get_category_specific_queries(query_category, main_query)

            # 多轮增强检索
            enhanced_results = contextual_retriever.multi_round_retrieve(basic_info_queries)

            # 提取文档内容和元数据
            relevant_chunks = []
            doc_metadata_map = {}
            rag_docs = []
            for doc, vec_score, rerank_score in enhanced_results:
                chunk_content = doc.page_content
                relevant_chunks.append(chunk_content)
                # 保存文档内容与元数据的映射关系
                doc_metadata_map[chunk_content] = doc.metadata
                # 保存原始文档对象
                rag_docs.append(doc)
                logger.debug(f"基础信息检索到文档片段，向量分数: {vec_score:.3f}, 重排序分数: {rerank_score:.3f}")

            # 限制长度
            unique_chunks = list(set(relevant_chunks))[:10]
            chunks_text = "\n\n---\n\n".join(unique_chunks)

            # 保存文档元数据映射和RAG文档，供后续使用
            self._doc_metadata_map = doc_metadata_map
            self._last_rag_docs = rag_docs

            logger.info(f"基础信息检索完成，使用 {len(unique_chunks)} 个文档片段")
            
            # 调用LLM提取信息
            prompt = self.extraction_prompt.format(document_chunks=chunks_text)
            response = self.llm.invoke(prompt)
            
            # 解析响应
            extracted_data = self._parse_llm_response(response.content if hasattr(response, 'content') else str(response))
            
            # 更新状态
            if extracted_data:
                self._update_basic_info(state, extracted_data)
            
            state.current_step = "basic_info_extracted"
            logger.info("基础信息提取完成")
            
        except Exception as e:
            error_msg = f"基础信息提取失败: {str(e)}"
            logger.error(error_msg)
            state.error_messages.append(error_msg)
        
        return state
    
    def extract_qualification_criteria(self, state: GraphStateModel) -> GraphStateModel:
        """
        提取资格审查硬性条件
        
        Args:
            state: 图状态
            
        Returns:
            GraphState: 更新后的状态
        """
        try:
            logger.info("开始提取资格审查条件")
            
            if not state.vector_store:
                raise ValueError("向量存储未初始化")
            
            # 使用增强的检索策略
            contextual_retriever = ContextualRetriever(state.vector_store)

            # 构建资格审查查询
            qualification_queries = [
                "资格审查 资质要求",
                "企业资质 营业执照",
                "项目业绩 类似项目",
                "项目经理 技术负责人",
                "财务状况 注册资金",
                "体系认证 ISO认证"
            ]

            # 多轮增强检索
            enhanced_results = contextual_retriever.multi_round_retrieve(qualification_queries)

            # 提取文档内容和元数据
            relevant_chunks = []
            qualification_metadata_map = {}
            qualification_rag_docs = []
            for doc, vec_score, rerank_score in enhanced_results:
                chunk_content = doc.page_content
                relevant_chunks.append(chunk_content)
                # 保存文档内容与元数据的映射关系
                qualification_metadata_map[chunk_content] = doc.metadata
                # 保存原始文档对象
                qualification_rag_docs.append(doc)
                logger.debug(f"资格审查检索到文档片段，向量分数: {vec_score:.3f}, 重排序分数: {rerank_score:.3f}")

            # 限制长度
            unique_chunks = list(set(relevant_chunks))[:10]
            chunks_text = "\n\n---\n\n".join(unique_chunks)

            # 保存文档元数据映射和RAG文档，供后续使用
            self._qualification_metadata_map = qualification_metadata_map
            # 更新最新的RAG文档列表
            self._last_rag_docs = qualification_rag_docs

            logger.info(f"资格审查检索完成，使用 {len(unique_chunks)} 个文档片段")
            
            # 调用LLM提取信息
            qualification_prompt = self._create_qualification_prompt()
            prompt = qualification_prompt.format(document_chunks=chunks_text)
            response = self.llm.invoke(prompt)
            
            # 解析响应
            qualification_data = self._parse_llm_response(response.content if hasattr(response, 'content') else str(response))
            
            # 更新状态
            if qualification_data:
                self._update_qualification_criteria(state, qualification_data)
            
            logger.info("资格审查条件提取完成")
            
        except Exception as e:
            error_msg = f"资格审查条件提取失败: {str(e)}"
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
    
    def _update_basic_info(self, state: GraphStateModel, data: dict) -> None:
        """更新基础信息"""
        for field_name, field_data in data.items():
            if isinstance(field_data, dict) and 'value' in field_data:
                source_text = field_data.get('source_text', '')

                # 多层次页码提取策略
                page_number = None

                # 1. 优先从field_data中获取页码
                page_number = field_data.get('page_number')

                # 2. 从文档元数据映射中获取页码信息
                if not page_number and hasattr(self, '_doc_metadata_map') and source_text:
                    for doc_content, metadata in self._doc_metadata_map.items():
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
                    logger.warning(f"无法为字段 {field_name} 提取页码信息，来源文本: {source_text[:50]}...")
                    page_number = 1  # 设置默认页码为1

                extracted_field = ExtractedField(
                    value=field_data.get('value'),
                    source=DocumentSource(
                        source_text=source_text,
                        page_number=page_number
                    ),
                    confidence=field_data.get('confidence', 0.5)
                )
                setattr(state.analysis_result.basic_information, field_name, extracted_field)
    
    def _update_qualification_criteria(self, state: GraphStateModel, data: dict) -> None:
        """更新资格审查条件"""
        qualification = state.analysis_result.basic_information.qualification_criteria

        for category, items in data.items():
            if isinstance(items, list):
                extracted_fields = []
                for item in items:
                    if isinstance(item, dict) and 'value' in item:
                        source_text = item.get('source_text', '')

                        # 多层次页码提取策略
                        page_number = None

                        # 1. 优先从item中获取页码
                        page_number = item.get('page_number')

                        # 2. 从文档元数据映射中获取页码信息
                        if not page_number and hasattr(self, '_qualification_metadata_map') and source_text:
                            for doc_content, metadata in self._qualification_metadata_map.items():
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
                            logger.warning(f"无法为资格审查项 {category} 提取页码信息，来源文本: {source_text[:50]}...")
                            page_number = 1  # 设置默认页码为1

                        extracted_field = ExtractedField(
                            value=item.get('value'),
                            source=DocumentSource(
                                source_text=source_text,
                                page_number=page_number
                            ),
                            confidence=item.get('confidence', 0.5)
                        )
                        extracted_fields.append(extracted_field)
                setattr(qualification, category, extracted_fields)

def create_basic_info_extractor_node():
    """创建基础信息提取节点函数"""
    extractor = BasicInfoExtractor()
    
    def basic_info_extractor_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """基础信息提取节点函数"""
        # 转换为GraphStateModel对象
        graph_state = GraphStateModel(**state)

        # 执行基础信息提取
        graph_state = extractor.extract_basic_info(graph_state)
        graph_state = extractor.extract_qualification_criteria(graph_state)

        # 转换回字典格式
        return graph_state.model_dump()
    
    return basic_info_extractor_node
