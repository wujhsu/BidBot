"""
基础信息提取节点
Basic information extraction node for the Langgraph workflow
"""

from typing import Dict, Any, List, Optional
from loguru import logger
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from src.models.data_models import GraphState, GraphStateModel, ExtractedField, DocumentSource, QualificationCriteria, BidDocumentRequirements, BidEvaluationProcess
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
        self.qualification_prompt = self._create_qualification_prompt()
        self.bid_document_requirements_prompt = self._create_bid_document_requirements_prompt()
        self.bid_evaluation_process_prompt = self._create_bid_evaluation_process_prompt()
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
        """创建投标人资格要求提取提示模板"""
        template = """
你是一个专业的招投标文件分析专家。请从以下文档片段中提取投标人资格要求。

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

提取要求：
1. 重点关注以下关键词及其同义词：
   - 投标人资格要求、投标人（企业）资格要求、资格审查条件、资格要求
   - 资质要求、业绩要求、信誉要求、人员要求、其他要求
   - 企业资质、营业执照、资质证书、认证证书
   - 项目业绩、类似项目、合同业绩、项目经验
   - 项目经理、技术负责人、项目团队、人员配置
   - 财务状况、注册资金、信用记录、体系认证

2. 理解每个提取项的核心意图，即使招标文件中使用了不同的术语，只要表达的是相同含义，就必须准确识别并提取
3. 重点关注带有"必须"、"应当"、"要求"等强制性词汇的条款
4. 逐条列出所有硬性条件，不要遗漏
5. 严格忠于原文，完整保留原文内容，不要添加主观判断或进行总结
6. 如果某类要求未找到，返回空数组[]
7. 如果找到"投标人资格要求"或类似标题的完整章节，请完整提取该章节的所有内容
"""
        return PromptTemplate(
            input_variables=["document_chunks"],
            template=template
        )

    def _create_bid_document_requirements_prompt(self) -> PromptTemplate:
        """创建投标文件要求提取提示模板"""
        template = """
你是一个专业的招投标文件分析专家。请从以下文档片段中提取投标文件要求相关信息。

文档片段：
{document_chunks}

请严格按照以下JSON格式提取信息：

{{
    "composition_and_format": [
        {{
            "value": "投标文件组成部分和编制格式要求的具体内容",
            "source_text": "原文片段",
            "confidence": 0.9
        }}
    ],
    "binding_and_sealing": [
        {{
            "value": "装订与密封要求的具体内容（包括正本副本份数、装订方式、密封要求等）",
            "source_text": "原文片段",
            "confidence": 0.9
        }}
    ],
    "signature_and_seal": [
        {{
            "value": "签字盖章要求的具体内容（包括签字人身份、盖章位置等）",
            "source_text": "原文片段",
            "confidence": 0.9
        }}
    ],
    "document_structure": [
        {{
            "value": "投标文件章节框架或目录结构的具体内容",
            "source_text": "原文片段",
            "confidence": 0.9
        }}
    ]
}}

提取要求：
1. 重点关注以下关键词及其同义词：
   - 组成与编制规范：组成部分、构成、内容应包括、编制要求、格式规范、文件构成、材料清单
   - 装订与密封要求：装订、成册、密封、包装、标记、正本、副本、份数、单独密封、分别包装、外包、封套
   - 签字盖章要求：签字、盖章、法定代表人、授权代表、签署、签章、公章、法人章
   - 投标文件章节框架：目录、章节、顺序、编排、内容索引、框架、第一个信封、第二个信封、商务及技术文件、报价文件

2. **特别注意投标文件目录结构的完整性**：
   - 投标文件通常包含多个部分（如商务及技术文件、报价文件等）
   - 每个部分都有自己的目录结构
   - 必须识别并提取所有相关的目录，包括：
     * 第一个信封（商务及技术文件）的目录
     * 第二个信封（报价文件）的目录
     * 或者其他分类方式的目录结构
   - 如果发现多个目录，请分别提取，不要合并

3. 理解每个提取项的核心意图，即使招标文件中使用了不同的术语，只要表达的是相同含义，就必须准确识别并提取
4. 确保在每个子项下提取所有相关信息
5. **严格忠于原文，完整保留原文内容，不要进行总结或简化**
6. **特别是对于投标文件的组成部分和目录结构，必须完整提取，保持原有的层次结构和格式**
7. 如果某类要求未找到，返回空数组[]
8. 如果文件中没有明确提及某一具体要求，请在该项后标注"未提及"
9. 对于目录结构，请保持原文的层次关系和编号格式
"""
        return PromptTemplate(
            input_variables=["document_chunks"],
            template=template
        )

    def _create_bid_evaluation_process_prompt(self) -> PromptTemplate:
        """创建开评定标流程提取提示模板"""
        template = """
你是一个专业的招投标文件分析专家。请从以下文档片段中提取开评定标流程相关信息。

文档片段：
{document_chunks}

请严格按照以下JSON格式提取信息：

{{
    "bid_opening": [
        {{
            "value": "开标环节的具体信息（时间、地点、程序等）",
            "source_text": "原文片段",
            "confidence": 0.9
        }}
    ],
    "evaluation": [
        {{
            "value": "评标环节的具体信息（评委会、评审方法/标准、主要流程等）",
            "source_text": "原文片段",
            "confidence": 0.9
        }}
    ],
    "award_decision": [
        {{
            "value": "定标环节的具体信息（定标原则、中标通知等）",
            "source_text": "原文片段",
            "confidence": 0.9
        }}
    ]
}}

提取要求：
1. 重点关注以下关键词及其同义词：
   - 开标环节：开标时间、开标地点、开标仪式、开标程序、唱标
   - 评标环节：评标委员会、评审小组、评审方法、评审标准、初步评审、详细评审、技术评分、商务评分、打分表
   - 定标环节：定标原则、中标候选人、推荐、合同授予、中标通知书、结果公示

2. 理解每个提取项的核心意图，即使招标文件中使用了不同的术语，只要表达的是相同含义，就必须准确识别并提取
3. 梳理从开标到最终确定中标人的整个流程和关键节点
4. 确保在每个子项下提取所有相关信息
5. 如果某类要求未找到，返回空数组[]
6. 如果文件中没有明确提及某一具体要求，请在该项后标注"未提及"
7. 直接提炼核心要点，避免大段复制原文
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

            # 构建投标人资格要求查询
            qualification_queries = [
                "投标人资格要求 投标人（企业）资格要求",
                "资格审查 资质要求 资格要求",
                "企业资质 营业执照 资质证书",
                "项目业绩 类似项目 合同业绩",
                "项目经理 技术负责人 项目团队",
                "财务状况 注册资金 信用记录",
                "体系认证 ISO认证 认证证书",
                "资质要求 业绩要求 信誉要求 人员要求"
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

    def extract_bid_document_requirements(self, state: GraphStateModel) -> GraphStateModel:
        """
        提取投标文件要求

        Args:
            state: 图状态

        Returns:
            GraphState: 更新后的状态
        """
        try:
            logger.info("开始提取投标文件要求")

            if not state.vector_store:
                raise ValueError("向量存储未初始化")

            # 使用增强的检索策略
            contextual_retriever = ContextualRetriever(state.vector_store)

            # 构建投标文件要求查询
            bid_doc_queries = [
                "投标文件 组成 编制",
                "装订 密封 正本 副本",
                "签字 盖章 法定代表人",
                "目录 章节 框架 顺序",
                "第一个信封 商务及技术文件 目录",
                "第二个信封 报价文件 目录",
                "投标文件要求 制作规定",
                "文件构成 材料清单",
                "商务文件 技术文件 目录结构",
                "报价清单 报价表 目录"
            ]

            # 多轮增强检索
            enhanced_results = contextual_retriever.multi_round_retrieve(bid_doc_queries)

            # 提取文档内容和元数据
            relevant_chunks = []
            bid_doc_metadata_map = {}
            bid_doc_rag_docs = []
            for doc, vec_score, rerank_score in enhanced_results:
                chunk_content = doc.page_content
                relevant_chunks.append(chunk_content)
                # 保存文档内容与元数据的映射关系
                bid_doc_metadata_map[chunk_content] = doc.metadata
                # 保存原始文档对象
                bid_doc_rag_docs.append(doc)
                logger.debug(f"投标文件要求检索到文档片段，向量分数: {vec_score:.3f}, 重排序分数: {rerank_score:.3f}")

            # 限制长度
            unique_chunks = list(set(relevant_chunks))[:10]
            chunks_text = "\n\n---\n\n".join(unique_chunks)

            # 保存文档元数据映射和RAG文档，供后续使用
            self._bid_doc_metadata_map = bid_doc_metadata_map
            # 更新最新的RAG文档列表
            self._last_rag_docs = bid_doc_rag_docs

            logger.info(f"投标文件要求检索完成，使用 {len(unique_chunks)} 个文档片段")

            # 调用LLM提取信息
            prompt = self.bid_document_requirements_prompt.format(document_chunks=chunks_text)
            response = self.llm.invoke(prompt)

            # 解析响应
            bid_doc_data = self._parse_llm_response(response.content if hasattr(response, 'content') else str(response))

            # 更新状态
            if bid_doc_data:
                self._update_bid_document_requirements(state, bid_doc_data)

            logger.info("投标文件要求提取完成")

        except Exception as e:
            error_msg = f"投标文件要求提取失败: {str(e)}"
            logger.error(error_msg)
            state.error_messages.append(error_msg)

        return state

    def extract_bid_evaluation_process(self, state: GraphStateModel) -> GraphStateModel:
        """
        提取开评定标流程

        Args:
            state: 图状态

        Returns:
            GraphState: 更新后的状态
        """
        try:
            logger.info("开始提取开评定标流程")

            if not state.vector_store:
                raise ValueError("向量存储未初始化")

            # 使用增强的检索策略
            contextual_retriever = ContextualRetriever(state.vector_store)

            # 构建开评定标流程查询
            evaluation_process_queries = [
                "开标 时间 地点 程序",
                "评标 委员会 评审方法",
                "定标 中标 原则 通知",
                "开标仪式 唱标",
                "评审标准 技术评分 商务评分",
                "中标候选人 结果公示"
            ]

            # 多轮增强检索
            enhanced_results = contextual_retriever.multi_round_retrieve(evaluation_process_queries)

            # 提取文档内容和元数据
            relevant_chunks = []
            evaluation_metadata_map = {}
            evaluation_rag_docs = []
            for doc, vec_score, rerank_score in enhanced_results:
                chunk_content = doc.page_content
                relevant_chunks.append(chunk_content)
                # 保存文档内容与元数据的映射关系
                evaluation_metadata_map[chunk_content] = doc.metadata
                # 保存原始文档对象
                evaluation_rag_docs.append(doc)
                logger.debug(f"开评定标流程检索到文档片段，向量分数: {vec_score:.3f}, 重排序分数: {rerank_score:.3f}")

            # 限制长度
            unique_chunks = list(set(relevant_chunks))[:10]
            chunks_text = "\n\n---\n\n".join(unique_chunks)

            # 保存文档元数据映射和RAG文档，供后续使用
            self._evaluation_metadata_map = evaluation_metadata_map
            # 更新最新的RAG文档列表
            self._last_rag_docs = evaluation_rag_docs

            logger.info(f"开评定标流程检索完成，使用 {len(unique_chunks)} 个文档片段")

            # 调用LLM提取信息
            prompt = self.bid_evaluation_process_prompt.format(document_chunks=chunks_text)
            response = self.llm.invoke(prompt)

            # 解析响应
            evaluation_data = self._parse_llm_response(response.content if hasattr(response, 'content') else str(response))

            # 更新状态
            if evaluation_data:
                self._update_bid_evaluation_process(state, evaluation_data)

            logger.info("开评定标流程提取完成")

        except Exception as e:
            error_msg = f"开评定标流程提取失败: {str(e)}"
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
                    page_number = -1  # 设置默认页码为-1

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
                            page_number = -1  # 设置默认页码为-1

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

    def _update_bid_document_requirements(self, state: GraphStateModel, data: dict) -> None:
        """更新投标文件要求"""
        bid_doc_requirements = state.analysis_result.basic_information.bid_document_requirements

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
                        if not page_number and hasattr(self, '_bid_doc_metadata_map') and source_text:
                            for doc_content, metadata in self._bid_doc_metadata_map.items():
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
                            logger.warning(f"无法为投标文件要求项 {category} 提取页码信息，来源文本: {source_text[:50]}...")
                            page_number = -1  # 设置默认页码为-1

                        extracted_field = ExtractedField(
                            value=item.get('value'),
                            source=DocumentSource(
                                source_text=source_text,
                                page_number=page_number
                            ),
                            confidence=item.get('confidence', 0.5)
                        )
                        extracted_fields.append(extracted_field)
                setattr(bid_doc_requirements, category, extracted_fields)

    def _update_bid_evaluation_process(self, state: GraphStateModel, data: dict) -> None:
        """更新开评定标流程"""
        bid_evaluation_process = state.analysis_result.basic_information.bid_evaluation_process

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
                        if not page_number and hasattr(self, '_evaluation_metadata_map') and source_text:
                            for doc_content, metadata in self._evaluation_metadata_map.items():
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
                            logger.warning(f"无法为开评定标流程项 {category} 提取页码信息，来源文本: {source_text[:50]}...")
                            page_number = -1  # 设置默认页码为-1

                        extracted_field = ExtractedField(
                            value=item.get('value'),
                            source=DocumentSource(
                                source_text=source_text,
                                page_number=page_number
                            ),
                            confidence=item.get('confidence', 0.5)
                        )
                        extracted_fields.append(extracted_field)
                setattr(bid_evaluation_process, category, extracted_fields)

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
        graph_state = extractor.extract_bid_document_requirements(graph_state)
        graph_state = extractor.extract_bid_evaluation_process(graph_state)

        # 转换回字典格式
        return graph_state.model_dump()
    
    return basic_info_extractor_node
