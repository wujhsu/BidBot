"""
评分标准分析节点
Scoring criteria analysis node for the Langgraph workflow
"""

from typing import Dict, Any, List
from loguru import logger
from langchain_core.prompts import PromptTemplate
from src.models.data_models import GraphStateModel, ExtractedField, DocumentSource, ScoringItem, ScoreComposition
from src.utils.llm_factory import LLMFactory
from src.utils.enhanced_retrieval import ContextualRetriever, SmartQueryRouter
from src.utils.improved_retrieval import ImprovedRetriever
import json
import re

class ScoringAnalyzer:
    """评分标准分析器"""
    
    def __init__(self):
        """初始化评分标准分析器"""
        self.llm = LLMFactory.create_llm()
        self.scoring_prompt = self._create_scoring_prompt()
        self.detailed_scoring_prompt = self._create_detailed_scoring_prompt()
        self.query_router = SmartQueryRouter()
    
    def _create_scoring_prompt(self) -> PromptTemplate:
        """创建评分标准提取提示模板"""
        template = """
你是一个专业的招投标文件分析专家。请从以下文档片段中提取评分标准相关信息。

文档片段：
{document_chunks}

请严格按照以下JSON格式提取信息：

{{
    "preliminary_review": [
        {{
            "value": "初步评审的具体通过性条款",
            "source_text": "原文片段",
            "confidence": 0.9
        }}
    ],
    "evaluation_method": {{
        "value": "详细评审方法（如综合评估法、最低价评标法等）",
        "source_text": "原文片段",
        "confidence": 0.9
    }},
    "score_composition": {{
        "technical_score": {{
            "value": "技术分占比（如30%或30分）",
            "source_text": "原文片段",
            "confidence": 0.9
        }},
        "commercial_score": {{
            "value": "商务分占比",
            "source_text": "原文片段",
            "confidence": 0.9
        }},
        "price_score": {{
            "value": "价格分占比",
            "source_text": "原文片段",
            "confidence": 0.9
        }},
        "other_scores": [
            {{
                "value": "其他评分项占比",
                "source_text": "原文片段",
                "confidence": 0.9
            }}
        ]
    }},
    "bonus_points": [
        {{
            "value": "加分项条件和分值（如专利、本地化服务、荣誉资质等）",
            "source_text": "原文片段",
            "confidence": 0.9
        }}
    ],
    "disqualification_clauses": [
        {{
            "value": "否决项条款（重点关注★号、*号条款）",
            "source_text": "原文片段",
            "confidence": 0.9
        }}
    ]
}}

注意事项：
1. 重点关注带有"★"、"*"、"否决"、"废标"等关键词的条款
2. 准确提取分值和占比信息
3. 严格忠于原文，不要添加主观判断
4. 如果某项信息未找到，对应字段填写"招标文件中未提及"或返回空数组
"""
        return PromptTemplate(
            input_variables=["document_chunks"],
            template=template
        )
    
    def _create_detailed_scoring_prompt(self) -> PromptTemplate:
        """创建详细评分细则提取提示模板"""
        template = """
你是一个专业的招投标文件分析专家。请从以下文档片段中提取详细的评分细则表。

文档片段：
{document_chunks}

请严格按照以下JSON格式提取信息：

{{
    "scoring_items": [
        {{
            "category": "评分类别（如技术、商务、价格）",
            "item_name": "具体评分项名称",
            "max_score": 10.0,
            "criteria": "评分标准描述",
            "source_text": "原文片段"
        }}
    ]
}}

重要提取规则：
1. **优先查找评分表格、附表、评分细则表**：
   - 寻找"评标办法前附表"、"评分标准表"、"技术评分细则表"等
   - 从表格中提取具体的分值数字

2. **分值提取要求**：
   - 如果找到具体数字分值，使用数字（如30.0、20.0）
   - 如果只有"见评标办法前附表"等引用，但文档中包含附表内容，从附表中提取具体分值
   - 如果确实无法找到具体分值，使用字符串说明（如"见评标办法前附表"）

3. **全面搜索策略**：
   - 仔细查看所有文档片段，寻找评分表格
   - 关注"技术分"、"商务分"、"价格分"的具体分值
   - 查找"30分"、"20分"、"15分"等具体数字

4. **表格信息优先**：
   - 如果文档中同时有概述和详细表格，优先使用表格中的具体分值
   - 将表格中的每一行都作为一个评分项提取

5. **完整性要求**：
   - 逐项列出所有评分项、分值、评分标准
   - 按技术、商务、价格等类别分类
   - 严格忠于原文内容，不要遗漏任何评分项
"""
        return PromptTemplate(
            input_variables=["document_chunks"],
            template=template
        )
    
    def extract_scoring_criteria(self, state: GraphStateModel) -> GraphStateModel:
        """
        提取评分标准
        
        Args:
            state: 图状态
            
        Returns:
            GraphState: 更新后的状态
        """
        try:
            logger.info("开始提取评分标准")
            
            if not state.vector_store:
                raise ValueError("向量存储未初始化")
            
            # 使用改进的检索策略
            improved_retriever = ImprovedRetriever(state.vector_store)

            # 专门针对评分标准的检索
            enhanced_results = improved_retriever.retrieve_scoring_criteria("评分标准 评分方法")

            # 提取文档内容
            relevant_chunks = []
            for doc, vec_score, rerank_score in enhanced_results:
                relevant_chunks.append(doc.page_content)
                logger.debug(f"检索到文档片段，向量分数: {vec_score:.3f}, 重排序分数: {rerank_score:.3f}")

            # 不过度限制文档数量，保留更多信息
            chunks_text = "\n\n---\n\n".join(relevant_chunks)

            logger.info(f"评分标准检索完成，使用 {len(relevant_chunks)} 个文档片段")
            
            # 调用LLM提取信息
            prompt = self.scoring_prompt.format(document_chunks=chunks_text)
            response = self.llm.invoke(prompt)
            
            # 解析响应
            scoring_data = self._parse_llm_response(response.content if hasattr(response, 'content') else str(response))
            
            # 更新状态
            if scoring_data:
                self._update_scoring_criteria(state, scoring_data)
            
            logger.info("评分标准提取完成")
            
        except Exception as e:
            error_msg = f"评分标准提取失败: {str(e)}"
            logger.error(error_msg)
            state.error_messages.append(error_msg)
        
        return state
    
    def extract_detailed_scoring(self, state: GraphStateModel) -> GraphStateModel:
        """
        提取详细评分细则
        
        Args:
            state: 图状态
            
        Returns:
            GraphState: 更新后的状态
        """
        try:
            logger.info("开始提取详细评分细则")
            
            if not state.vector_store:
                raise ValueError("向量存储未初始化")
            
            # 使用改进的检索策略获取详细评分信息
            improved_retriever = ImprovedRetriever(state.vector_store)

            # 专门针对详细评分的检索
            enhanced_results = improved_retriever.retrieve_detailed_scoring("评分细则 评分表")

            # 提取文档内容
            relevant_chunks = []
            for doc, vec_score, rerank_score in enhanced_results:
                relevant_chunks.append(doc.page_content)
                logger.debug(f"详细评分检索到文档片段，向量分数: {vec_score:.3f}, 重排序分数: {rerank_score:.3f}")

            # 保留更多文档信息
            chunks_text = "\n\n---\n\n".join(relevant_chunks)

            logger.info(f"详细评分检索完成，使用 {len(relevant_chunks)} 个文档片段")
            
            # 调用LLM提取信息
            prompt = self.detailed_scoring_prompt.format(document_chunks=chunks_text)
            response = self.llm.invoke(prompt)
            
            # 解析响应
            detailed_data = self._parse_llm_response(response.content if hasattr(response, 'content') else str(response))
            
            # 更新状态
            if detailed_data and 'scoring_items' in detailed_data:
                self._update_detailed_scoring(state, detailed_data['scoring_items'])
            
            state.current_step = "scoring_analyzed"
            logger.info("详细评分细则提取完成")
            
        except Exception as e:
            error_msg = f"详细评分细则提取失败: {str(e)}"
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
    
    def _update_scoring_criteria(self, state: GraphStateModel, data: dict) -> None:
        """更新评分标准"""
        scoring_criteria = state.analysis_result.scoring_criteria
        
        # 更新初步评审标准
        if 'preliminary_review' in data and isinstance(data['preliminary_review'], list):
            scoring_criteria.preliminary_review = [
                ExtractedField(
                    value=item.get('value'),
                    source=DocumentSource(source_text=item.get('source_text')),
                    confidence=item.get('confidence', 0.5)
                )
                for item in data['preliminary_review']
                if isinstance(item, dict) and 'value' in item
            ]
        
        # 更新评审方法
        if 'evaluation_method' in data and isinstance(data['evaluation_method'], dict):
            method_data = data['evaluation_method']
            scoring_criteria.evaluation_method = ExtractedField(
                value=method_data.get('value'),
                source=DocumentSource(source_text=method_data.get('source_text')),
                confidence=method_data.get('confidence', 0.5)
            )
        
        # 更新分值构成
        if 'score_composition' in data:
            comp_data = data['score_composition']
            score_comp = ScoreComposition()
            
            for field_name in ['technical_score', 'commercial_score', 'price_score']:
                if field_name in comp_data and isinstance(comp_data[field_name], dict):
                    field_data = comp_data[field_name]
                    setattr(score_comp, field_name, ExtractedField(
                        value=field_data.get('value'),
                        source=DocumentSource(source_text=field_data.get('source_text')),
                        confidence=field_data.get('confidence', 0.5)
                    ))
            
            if 'other_scores' in comp_data and isinstance(comp_data['other_scores'], list):
                score_comp.other_scores = [
                    ExtractedField(
                        value=item.get('value'),
                        source=DocumentSource(source_text=item.get('source_text')),
                        confidence=item.get('confidence', 0.5)
                    )
                    for item in comp_data['other_scores']
                    if isinstance(item, dict) and 'value' in item
                ]
            
            scoring_criteria.score_composition = score_comp
        
        # 更新加分项和否决项
        for field_name in ['bonus_points', 'disqualification_clauses']:
            if field_name in data and isinstance(data[field_name], list):
                setattr(scoring_criteria, field_name, [
                    ExtractedField(
                        value=item.get('value'),
                        source=DocumentSource(source_text=item.get('source_text')),
                        confidence=item.get('confidence', 0.5)
                    )
                    for item in data[field_name]
                    if isinstance(item, dict) and 'value' in item
                ])
    
    def _update_detailed_scoring(self, state: GraphStateModel, scoring_items: List[dict]) -> None:
        """更新详细评分细则"""
        detailed_scoring = []

        for item in scoring_items:
            if isinstance(item, dict):
                # 处理max_score字段，支持数字和字符串
                max_score = item.get('max_score')
                if isinstance(max_score, str):
                    # 尝试从字符串中提取数字
                    import re
                    number_match = re.search(r'(\d+(?:\.\d+)?)', max_score)
                    if number_match:
                        try:
                            max_score = float(number_match.group(1))
                        except ValueError:
                            # 如果转换失败，保持原字符串
                            pass

                scoring_item = ScoringItem(
                    category=item.get('category', ''),
                    item_name=item.get('item_name', ''),
                    max_score=max_score,
                    criteria=item.get('criteria'),
                    source=DocumentSource(source_text=item.get('source_text'))
                )
                detailed_scoring.append(scoring_item)

        state.analysis_result.scoring_criteria.detailed_scoring = detailed_scoring

def create_scoring_analyzer_node():
    """创建评分标准分析节点函数"""
    analyzer = ScoringAnalyzer()
    
    def scoring_analyzer_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """评分标准分析节点函数"""
        # 转换为GraphStateModel对象
        graph_state = GraphStateModel(**state)
        
        # 执行评分标准分析
        graph_state = analyzer.extract_scoring_criteria(graph_state)
        graph_state = analyzer.extract_detailed_scoring(graph_state)
        
        # 转换回字典格式
        return graph_state.model_dump()
    
    return scoring_analyzer_node
