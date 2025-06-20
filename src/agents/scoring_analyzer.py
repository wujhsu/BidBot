"""
评分标准分析节点
Scoring criteria analysis node for the Langgraph workflow
"""

from typing import Dict, Any, List, Optional
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

请严格按照以下JSON格式提取信息，注意JSON格式的正确性：

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

**JSON格式要求**：
- 所有字符串值必须用双引号包围
- 字符串内的双引号必须转义为 \\"
- 不要在JSON中使用单引号
- 确保所有括号和逗号正确匹配
- criteria字段如果包含复杂文本，请确保正确转义特殊字符

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

6. **文本处理要求**：
   - criteria字段中的复杂评分标准描述要完整保留
   - 如果评分标准包含分号、引号等特殊字符，请正确转义
   - 保持原文的完整性和准确性
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
            scoring_rag_docs = []
            for doc, vec_score, rerank_score in enhanced_results:
                relevant_chunks.append(doc.page_content)
                # 保存原始文档对象
                scoring_rag_docs.append(doc)
                logger.debug(f"检索到文档片段，向量分数: {vec_score:.3f}, 重排序分数: {rerank_score:.3f}")

            # 不过度限制文档数量，保留更多信息
            chunks_text = "\n\n---\n\n".join(relevant_chunks)

            # 保存RAG文档，供后续使用
            self._last_rag_docs = scoring_rag_docs

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
            detailed_rag_docs = []
            for doc, vec_score, rerank_score in enhanced_results:
                relevant_chunks.append(doc.page_content)
                # 保存原始文档对象
                detailed_rag_docs.append(doc)
                logger.debug(f"详细评分检索到文档片段，向量分数: {vec_score:.3f}, 重排序分数: {rerank_score:.3f}")

            # 保留更多文档信息
            chunks_text = "\n\n---\n\n".join(relevant_chunks)

            # 保存RAG文档，供后续使用
            self._last_rag_docs = detailed_rag_docs

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
        """解析LLM响应，增强容错性"""
        try:
            # 尝试提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()

                # 尝试直接解析
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"直接JSON解析失败: {e}")
                    logger.debug(f"原始JSON字符串长度: {len(json_str)}")

                    # 尝试多层次修复策略
                    for attempt in range(3):
                        try:
                            if attempt == 0:
                                # 第一次尝试：基础清理
                                cleaned_json = self._clean_json_string(json_str)
                            elif attempt == 1:
                                # 第二次尝试：更激进的修复
                                cleaned_json = self._aggressive_json_fix(json_str)
                            else:
                                # 第三次尝试：重构JSON
                                cleaned_json = self._reconstruct_json(json_str)

                            if cleaned_json:
                                result = json.loads(cleaned_json)
                                logger.info(f"JSON修复成功（尝试 {attempt + 1}）")
                                return result
                        except json.JSONDecodeError as e2:
                            logger.debug(f"修复尝试 {attempt + 1} 失败: {e2}")
                            continue

                    # 记录详细调试信息
                    self._log_json_debug_info(json_str, e)

                    # 尝试备用解析策略
                    backup_result = self._backup_parse_strategy(response)
                    if backup_result:
                        logger.info("备用解析策略成功")
                        return backup_result

                    return {}
            else:
                logger.warning("未找到有效的JSON响应")
                logger.debug(f"响应内容前500字符: {response[:500]}")
                return {}
        except Exception as e:
            logger.error(f"解析LLM响应时发生未预期错误: {e}")
            return {}

    def _clean_json_string(self, json_str: str) -> str:
        """清理JSON字符串，修复常见格式错误"""
        try:
            # 移除可能的BOM和其他不可见字符
            json_str = json_str.strip().strip('\ufeff')

            # 修复常见的JSON格式问题
            # 1. 移除对象或数组末尾的多余逗号
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

            # 2. 修复字符串中的未转义引号
            # 这是一个更安全的方法，专门处理字符串值中的引号
            def fix_quotes_in_strings(match):
                """修复字符串值中的引号"""
                field_name = match.group(1)
                content = match.group(2)
                # 转义内容中的双引号，但保留原有的转义
                content = content.replace('\\"', '___ESCAPED_QUOTE___')  # 临时标记已转义的引号
                content = content.replace('"', '\\"')  # 转义未转义的引号
                content = content.replace('___ESCAPED_QUOTE___', '\\"')  # 恢复已转义的引号
                return f'"{field_name}": "{content}"'

            # 匹配字符串字段并修复其中的引号
            json_str = re.sub(r'"(category|item_name|criteria|source_text|value)"\s*:\s*"([^"]*(?:"[^"]*)*)"',
                            fix_quotes_in_strings, json_str)

            # 3. 修复可能的换行符问题
            # 将字符串值中的换行符转义
            def fix_newlines_in_strings(match):
                """修复字符串值中的换行符"""
                field_name = match.group(1)
                content = match.group(2)
                # 转义换行符
                content = content.replace('\n', '\\n').replace('\r', '\\r')
                return f'"{field_name}": "{content}"'

            # 再次处理可能的换行符
            json_str = re.sub(r'"(category|item_name|criteria|source_text|value)"\s*:\s*"([^"]*)"',
                            fix_newlines_in_strings, json_str)

            # 4. 确保数字格式正确
            # 修复数字后面意外的字符
            json_str = re.sub(r'"max_score"\s*:\s*(\d+(?:\.\d+)?)([a-zA-Z]+)', r'"max_score": "\1\2"', json_str)

            # 5. 修复可能的尾随逗号
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

            return json_str
        except Exception as e:
            logger.error(f"JSON清理过程中出错: {e}")
            return ""

    def _aggressive_json_fix(self, json_str: str) -> str:
        """更激进的JSON修复策略"""
        try:
            # 移除BOM和不可见字符
            json_str = json_str.strip().strip('\ufeff')

            # 1. 修复缺少逗号的问题
            # 在 } 后面如果直接跟 { 则添加逗号
            json_str = re.sub(r'}\s*\n\s*{', '},\n            {', json_str)

            # 2. 修复字符串中的未转义引号
            def fix_string_field(match):
                field_name = match.group(1)
                field_value = match.group(2)

                # 转义字符串值中的引号
                # 先保护已经转义的引号
                field_value = field_value.replace('\\"', '___ESCAPED_QUOTE___')
                # 转义未转义的引号
                field_value = field_value.replace('"', '\\"')
                # 恢复已转义的引号
                field_value = field_value.replace('___ESCAPED_QUOTE___', '\\"')

                return f'"{field_name}": "{field_value}"'

            # 应用字符串字段修复
            json_str = re.sub(
                r'"(category|item_name|criteria|source_text|value)"\s*:\s*"([^"]*(?:"[^"]*)*)"',
                fix_string_field,
                json_str,
                flags=re.DOTALL
            )

            # 3. 修复尾随逗号
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

            # 4. 确保数组和对象正确闭合
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            if open_braces > close_braces:
                json_str += '}' * (open_braces - close_braces)

            open_brackets = json_str.count('[')
            close_brackets = json_str.count(']')
            if open_brackets > close_brackets:
                json_str += ']' * (open_brackets - close_brackets)

            # 5. 清理多余的转义和格式问题
            json_str = json_str.replace('\\n', ' ').replace('\\r', ' ')

            return json_str
        except Exception as e:
            logger.error(f"激进JSON修复过程中出错: {e}")
            return ""

    def _reconstruct_json(self, json_str: str) -> str:
        """重构JSON - 从原始文本中提取关键信息并重新构建JSON"""
        try:
            # 提取所有可能的字段值
            categories = re.findall(r'"category"\s*:\s*"([^"]*)"', json_str)
            item_names = re.findall(r'"item_name"\s*:\s*"([^"]*)"', json_str)
            max_scores = re.findall(r'"max_score"\s*:\s*([^,}\]]+)', json_str)

            # 提取criteria字段（更复杂的处理）
            criteria_list = []
            criteria_matches = re.finditer(r'"criteria"\s*:\s*"([^"]*(?:"[^"]*)*)"', json_str, re.DOTALL)
            for match in criteria_matches:
                criteria_content = match.group(1)
                # 清理和转义
                criteria_content = criteria_content.replace('\\"', '"').replace('"', '\\"')
                criteria_list.append(criteria_content)

            # 提取source_text
            source_texts = re.findall(r'"source_text"\s*:\s*"([^"]*)"', json_str)

            # 重构JSON
            scoring_items = []
            max_items = max(len(item_names), len(max_scores), len(categories))

            for i in range(max_items):
                item = {
                    "category": categories[i] if i < len(categories) else "技术分",
                    "item_name": item_names[i] if i < len(item_names) else f"评分项{i+1}",
                    "max_score": self._parse_score(max_scores[i]) if i < len(max_scores) else 0,
                    "criteria": criteria_list[i] if i < len(criteria_list) else "从文档中提取的评分信息",
                    "source_text": source_texts[i] if i < len(source_texts) else ""
                }
                scoring_items.append(item)

            # 构建完整的JSON
            result = {"scoring_items": scoring_items}
            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"JSON重构过程中出错: {e}")
            return ""

    def _parse_score(self, score_str: str) -> float:
        """解析分数字符串"""
        try:
            # 清理分数字符串
            score_clean = re.sub(r'[^\d.]', '', score_str.strip())
            if score_clean:
                return float(score_clean)
            else:
                return 0.0
        except:
            return 0.0

    def _log_json_debug_info(self, json_str: str, error: json.JSONDecodeError):
        """记录JSON调试信息"""
        try:
            # 记录错误位置附近的内容
            error_pos = getattr(error, 'pos', 0)
            start_pos = max(0, error_pos - 100)
            end_pos = min(len(json_str), error_pos + 100)
            context = json_str[start_pos:end_pos]

            logger.error(f"JSON解析错误详情:")
            logger.error(f"  错误位置: {error_pos}")
            logger.error(f"  错误消息: {error.msg}")
            logger.error(f"  错误上下文: {repr(context)}")

            # 记录JSON的行数和大致结构
            lines = json_str.split('\n')
            logger.error(f"  JSON总行数: {len(lines)}")
            if len(lines) > 10:
                logger.error(f"  前10行: {lines[:10]}")
                logger.error(f"  后10行: {lines[-10:]}")
        except Exception as e:
            logger.error(f"记录调试信息时出错: {e}")

    def _backup_parse_strategy(self, response: str) -> dict:
        """备用解析策略，当JSON解析失败时使用"""
        try:
            # 尝试从响应中提取关键信息，即使JSON格式不完整
            result = {"scoring_items": []}

            # 策略1: 更健壮的评分项模式匹配
            # 使用更宽松的模式来匹配评分项

            # 首先尝试匹配完整的评分项块
            item_blocks = re.findall(
                r'\{\s*"category"\s*:\s*"([^"]*)"[^}]*?"item_name"\s*:\s*"([^"]*)"[^}]*?"max_score"\s*:\s*([^,}\]]+)[^}]*?\}',
                response, re.DOTALL | re.IGNORECASE
            )

            for category, item_name, max_score in item_blocks:
                # 查找对应的criteria和source_text
                item_start = response.find(f'"item_name": "{item_name}"')
                if item_start != -1:
                    # 查找这个项目的完整上下文（更宽松的边界检测）
                    item_end = item_start + 2000  # 限制搜索范围
                    next_item = response.find('"item_name":', item_start + 1)
                    if next_item != -1 and next_item < item_end:
                        item_end = next_item

                    item_context = response[item_start:item_end]

                    # 提取criteria（使用更宽松的模式）
                    criteria_match = re.search(r'"criteria"\s*:\s*"([^"]*(?:"[^"]*)*)"', item_context, re.DOTALL)
                    if not criteria_match:
                        # 尝试更宽松的匹配
                        criteria_match = re.search(r'"criteria"\s*:\s*"([^"]+)', item_context)

                    # 提取source_text
                    source_match = re.search(r'"source_text"\s*:\s*"([^"]*)"', item_context)

                    # 处理max_score
                    try:
                        max_score_clean = re.sub(r'[^\d.]', '', max_score.strip())
                        if max_score_clean:
                            max_score_val = float(max_score_clean)
                        else:
                            max_score_val = max_score.strip().strip('"')
                    except:
                        max_score_val = max_score.strip().strip('"')

                    scoring_item = {
                        "category": category.strip(),
                        "item_name": item_name.strip(),
                        "max_score": max_score_val,
                        "criteria": criteria_match.group(1).strip() if criteria_match else "从文档中提取的评分信息",
                        "source_text": source_match.group(1).strip() if source_match else ""
                    }
                    result["scoring_items"].append(scoring_item)

            # 策略2: 如果策略1没有找到足够结果，尝试简单的字段匹配
            if len(result["scoring_items"]) < 3:  # 如果找到的项目太少
                # 查找所有item_name
                item_names = re.findall(r'"item_name"\s*:\s*"([^"]+)"', response)
                max_scores = re.findall(r'"max_score"\s*:\s*([^,}\]]+)', response)
                categories = re.findall(r'"category"\s*:\s*"([^"]*)"', response)

                # 尝试配对这些信息
                for i, item_name in enumerate(item_names):
                    if i < len(max_scores):
                        max_score = max_scores[i]
                        category = categories[i] if i < len(categories) else "技术分"

                        # 处理max_score
                        try:
                            max_score_clean = re.sub(r'[^\d.]', '', max_score.strip())
                            if max_score_clean:
                                max_score_val = float(max_score_clean)
                            else:
                                max_score_val = max_score.strip().strip('"')
                        except:
                            max_score_val = max_score.strip().strip('"')

                        scoring_item = {
                            "category": category.strip(),
                            "item_name": item_name.strip(),
                            "max_score": max_score_val,
                            "criteria": "从文档中提取的评分信息",
                            "source_text": ""
                        }

                        # 避免重复添加
                        if not any(item["item_name"] == item_name for item in result["scoring_items"]):
                            result["scoring_items"].append(scoring_item)

            # 策略3: 如果前面的策略都没有找到足够结果，尝试更宽松的模式匹配
            if len(result["scoring_items"]) < 2:
                # 查找分散的评分信息
                loose_patterns = [
                    r'技术.*?(\d+(?:\.\d+)?).*?分',
                    r'商务.*?(\d+(?:\.\d+)?).*?分',
                    r'价格.*?(\d+(?:\.\d+)?).*?分',
                    r'(\d+(?:\.\d+)?).*?分.*?技术',
                    r'(\d+(?:\.\d+)?).*?分.*?商务',
                    r'(\d+(?:\.\d+)?).*?分.*?价格'
                ]

                for pattern in loose_patterns:
                    matches = re.findall(pattern, response, re.IGNORECASE)
                    for match in matches:
                        try:
                            score = float(match)
                            if 'tech' in pattern.lower() or '技术' in pattern:
                                category = "技术分"
                                item_name = "技术评分"
                            elif 'business' in pattern.lower() or '商务' in pattern:
                                category = "商务分"
                                item_name = "商务评分"
                            elif 'price' in pattern.lower() or '价格' in pattern:
                                category = "价格分"
                                item_name = "价格评分"
                            else:
                                category = "其他"
                                item_name = "评分项"

                            scoring_item = {
                                "category": category,
                                "item_name": item_name,
                                "max_score": score,
                                "criteria": "从文档中提取的评分信息",
                                "source_text": f"匹配模式: {pattern}"
                            }

                            # 避免重复添加相同的评分项
                            if not any(item["item_name"] == item_name and item["max_score"] == score
                                     for item in result["scoring_items"]):
                                result["scoring_items"].append(scoring_item)
                        except ValueError:
                            continue

            if result["scoring_items"]:
                logger.info(f"备用解析策略提取到 {len(result['scoring_items'])} 个评分项")
                return result

            return {}
        except Exception as e:
            logger.error(f"备用解析策略失败: {e}")
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
    
    def _update_scoring_criteria(self, state: GraphStateModel, data: dict) -> None:
        """更新评分标准"""
        scoring_criteria = state.analysis_result.scoring_criteria
        
        # 更新初步评审标准
        if 'preliminary_review' in data and isinstance(data['preliminary_review'], list):
            preliminary_review = []
            for item in data['preliminary_review']:
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
                        logger.warning(f"无法为初步评审标准提取页码信息，来源文本: {source_text[:50]}...")
                        page_number = -1  # 设置默认页码为-1

                    preliminary_review.append(ExtractedField(
                        value=item.get('value'),
                        source=DocumentSource(
                            source_text=source_text,
                            page_number=page_number
                        ),
                        confidence=item.get('confidence', 0.5)
                    ))
            scoring_criteria.preliminary_review = preliminary_review

        # 更新评审方法
        if 'evaluation_method' in data and isinstance(data['evaluation_method'], dict):
            method_data = data['evaluation_method']
            source_text = method_data.get('source_text', '')

            # 多层次页码提取策略
            page_number = None

            # 1. 优先从method_data中获取页码
            page_number = method_data.get('page_number')

            # 2. 从来源文本中提取页码标记
            if not page_number:
                page_number = self._extract_page_number(source_text)

            # 3. 从RAG检索的文档中提取页码（如果有的话）
            if not page_number and hasattr(self, '_last_rag_docs'):
                page_number = self._extract_page_from_rag_docs(self._last_rag_docs)

            # 4. 如果仍然没有页码，记录警告并设置默认值
            if not page_number:
                logger.warning(f"无法为评审方法提取页码信息，来源文本: {source_text[:50]}...")
                page_number = -1  # 设置默认页码为-1

            scoring_criteria.evaluation_method = ExtractedField(
                value=method_data.get('value'),
                source=DocumentSource(
                    source_text=source_text,
                    page_number=page_number
                ),
                confidence=method_data.get('confidence', 0.5)
            )
        
        # 更新分值构成
        if 'score_composition' in data:
            comp_data = data['score_composition']
            score_comp = ScoreComposition()
            
            for field_name in ['technical_score', 'commercial_score', 'price_score']:
                if field_name in comp_data and isinstance(comp_data[field_name], dict):
                    field_data = comp_data[field_name]
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
                        logger.warning(f"无法为分值构成 {field_name} 提取页码信息，来源文本: {source_text[:50]}...")
                        page_number = -1  # 设置默认页码为-1

                    setattr(score_comp, field_name, ExtractedField(
                        value=field_data.get('value'),
                        source=DocumentSource(
                            source_text=source_text,
                            page_number=page_number
                        ),
                        confidence=field_data.get('confidence', 0.5)
                    ))
            
            if 'other_scores' in comp_data and isinstance(comp_data['other_scores'], list):
                other_scores = []
                for item in comp_data['other_scores']:
                    if isinstance(item, dict) and 'value' in item:
                        source_text = item.get('source_text', '')
                        page_number = self._extract_page_number(source_text)

                        other_scores.append(ExtractedField(
                            value=item.get('value'),
                            source=DocumentSource(
                                source_text=source_text,
                                page_number=page_number
                            ),
                            confidence=item.get('confidence', 0.5)
                        ))
                score_comp.other_scores = other_scores
            
            scoring_criteria.score_composition = score_comp
        
        # 更新加分项和否决项
        for field_name in ['bonus_points', 'disqualification_clauses']:
            if field_name in data and isinstance(data[field_name], list):
                field_items = []
                for item in data[field_name]:
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
                            logger.warning(f"无法为 {field_name} 提取页码信息，来源文本: {source_text[:50]}...")
                            page_number = -1  # 设置默认页码为-1

                        field_items.append(ExtractedField(
                            value=item.get('value'),
                            source=DocumentSource(
                                source_text=source_text,
                                page_number=page_number
                            ),
                            confidence=item.get('confidence', 0.5)
                        ))
                setattr(scoring_criteria, field_name, field_items)
    
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
                    logger.warning(f"无法为详细评分项 {item.get('item_name', '')} 提取页码信息，来源文本: {source_text[:50]}...")
                    page_number = -1  # 设置默认页码为-1

                scoring_item = ScoringItem(
                    category=item.get('category', ''),
                    item_name=item.get('item_name', ''),
                    max_score=max_score,
                    criteria=item.get('criteria'),
                    source=DocumentSource(
                        source_text=source_text,
                        page_number=page_number
                    )
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
