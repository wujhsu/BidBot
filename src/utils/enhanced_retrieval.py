"""
增强的RAG检索策略
Enhanced RAG retrieval strategies for better information extraction
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from loguru import logger
from src.utils.reranker import HybridRetriever, RerankerManager
from src.utils.llm_factory import LLMFactory
from config.settings import settings
import re
import json


class QueryExpander:
    """查询扩展器"""
    
    def __init__(self):
        """初始化查询扩展器"""
        self.llm = LLMFactory.create_llm()
    
    def expand_query(self, original_query: str, context: str = "") -> List[str]:
        """
        扩展查询，生成相关的查询变体
        
        Args:
            original_query: 原始查询
            context: 上下文信息
            
        Returns:
            List[str]: 扩展后的查询列表
        """
        try:
            prompt = f"""
你是一个专业的招投标文件分析专家。请根据原始查询生成3-5个相关的查询变体，以便更全面地检索相关信息。

原始查询: {original_query}
上下文: {context}

要求：
1. 生成的查询应该涵盖原始查询的不同表达方式
2. 包含可能的同义词和相关术语
3. 考虑招投标文件的专业术语
4. 每个查询应该简洁明确

请以JSON格式返回查询列表：
{{"expanded_queries": ["查询1", "查询2", "查询3", ...]}}
"""
            
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 解析JSON响应
            try:
                result = json.loads(content)
                expanded_queries = result.get("expanded_queries", [])
                
                # 确保包含原始查询
                if original_query not in expanded_queries:
                    expanded_queries.insert(0, original_query)
                
                logger.debug(f"查询扩展完成: {original_query} -> {len(expanded_queries)}个查询")
                return expanded_queries[:5]  # 限制最多5个查询
                
            except json.JSONDecodeError:
                logger.warning("查询扩展响应解析失败，使用原始查询")
                return [original_query]
                
        except Exception as e:
            logger.error(f"查询扩展失败: {e}")
            return [original_query]
    
    def generate_contextual_queries(self, main_query: str, document_type: str = "招标文件") -> List[str]:
        """
        根据文档类型生成上下文相关的查询
        
        Args:
            main_query: 主查询
            document_type: 文档类型
            
        Returns:
            List[str]: 上下文相关的查询列表
        """
        # 预定义的查询模板
        query_templates = {
            "评分标准": [
                "评分标准 评分方法",
                "技术分 商务分 价格分",
                "评分细则 评分表",
                "评标办法 评标标准",
                "附表 评分附表",
                "分值 得分 评分项"
            ],
            "基础信息": [
                "项目名称 招标编号",
                "预算金额 采购金额",
                "投标截止时间 开标时间",
                "投标保证金",
                "采购人 采购代理机构"
            ],
            "合同条款": [
                "合同条款 合同约定",
                "付款方式 付款条件",
                "交付期限 完成时间",
                "知识产权 专利权",
                "保密协议 保密条款"
            ]
        }
        
        # 根据主查询匹配相关模板
        relevant_queries = []
        for category, templates in query_templates.items():
            if any(keyword in main_query for keyword in category.split()):
                relevant_queries.extend(templates)
        
        # 如果没有匹配到模板，使用通用扩展
        if not relevant_queries:
            relevant_queries = [main_query]
        
        # 添加原始查询
        if main_query not in relevant_queries:
            relevant_queries.insert(0, main_query)
        
        return relevant_queries[:6]  # 限制查询数量


class ContextualRetriever:
    """上下文感知检索器"""
    
    def __init__(self, vector_store):
        """
        初始化上下文感知检索器
        
        Args:
            vector_store: 向量存储
        """
        self.vector_store = vector_store
        self.reranker = RerankerManager()
        self.hybrid_retriever = HybridRetriever(vector_store, self.reranker)
        self.query_expander = QueryExpander()
    
    def retrieve_with_context(
        self, 
        query: str, 
        context: str = "", 
        retrieval_strategy: str = "hybrid"
    ) -> List[Tuple[Document, float, float]]:
        """
        基于上下文的检索
        
        Args:
            query: 查询文本
            context: 上下文信息
            retrieval_strategy: 检索策略 ("simple", "expanded", "hybrid")
            
        Returns:
            List[Tuple[Document, float, float]]: (文档, 向量分数, 重排序分数)列表
        """
        try:
            if retrieval_strategy == "simple":
                return self._simple_retrieve(query)
            elif retrieval_strategy == "expanded":
                return self._expanded_retrieve(query, context)
            elif retrieval_strategy == "hybrid":
                return self._hybrid_retrieve(query, context)
            else:
                logger.warning(f"未知的检索策略: {retrieval_strategy}，使用默认策略")
                return self._hybrid_retrieve(query, context)
                
        except Exception as e:
            logger.error(f"上下文检索失败: {e}")
            return []
    
    def _simple_retrieve(self, query: str) -> List[Tuple[Document, float, float]]:
        """简单检索"""
        return self.hybrid_retriever.retrieve_and_rerank(query)
    
    def _expanded_retrieve(self, query: str, context: str) -> List[Tuple[Document, float, float]]:
        """扩展查询检索"""
        if settings.enable_query_expansion:
            expanded_queries = self.query_expander.expand_query(query, context)
            return self.hybrid_retriever.multi_query_retrieve(expanded_queries)
        else:
            return self._simple_retrieve(query)
    
    def _hybrid_retrieve(self, query: str, context: str) -> List[Tuple[Document, float, float]]:
        """混合检索策略"""
        # 第一轮：基础检索
        basic_results = self._simple_retrieve(query)
        
        # 第二轮：扩展查询检索（如果启用）
        if settings.enable_query_expansion:
            expanded_queries = self.query_expander.generate_contextual_queries(query)
            expanded_results = self.hybrid_retriever.multi_query_retrieve(expanded_queries)
            
            # 合并结果并去重
            combined_results = self._merge_and_deduplicate(basic_results, expanded_results)
            return combined_results
        
        return basic_results
    
    def _merge_and_deduplicate(
        self, 
        results1: List[Tuple[Document, float, float]], 
        results2: List[Tuple[Document, float, float]]
    ) -> List[Tuple[Document, float, float]]:
        """合并并去重检索结果"""
        seen_contents = set()
        merged_results = []
        
        # 合并两个结果列表
        all_results = results1 + results2
        
        for doc, vec_score, rerank_score in all_results:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                merged_results.append((doc, vec_score, rerank_score))
        
        # 按重排序分数排序
        merged_results.sort(key=lambda x: x[2], reverse=True)
        
        # 返回top_k结果
        return merged_results[:settings.rerank_final_k]
    
    def multi_round_retrieve(
        self, 
        queries: List[str], 
        max_rounds: Optional[int] = None
    ) -> List[Tuple[Document, float, float]]:
        """
        多轮检索
        
        Args:
            queries: 查询列表
            max_rounds: 最大轮数
            
        Returns:
            List[Tuple[Document, float, float]]: 多轮检索的合并结果
        """
        if max_rounds is None:
            max_rounds = settings.max_retrieval_rounds
        
        all_results = []
        seen_contents = set()
        
        for round_idx, query in enumerate(queries[:max_rounds]):
            logger.debug(f"第 {round_idx + 1} 轮检索: {query[:50]}...")
            
            round_results = self.retrieve_with_context(query, retrieval_strategy="hybrid")
            
            # 添加新的结果
            for doc, vec_score, rerank_score in round_results:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_results.append((doc, vec_score, rerank_score))
        
        # 最终排序
        all_results.sort(key=lambda x: x[2], reverse=True)
        return all_results[:settings.rerank_final_k * 2]  # 返回更多结果供后续处理


class SmartQueryRouter:
    """智能查询路由器"""
    
    def __init__(self):
        """初始化智能查询路由器"""
        self.query_patterns = {
            "评分标准": [
                r"评分.*?标准", r"评分.*?方法", r"技术.*?分", r"商务.*?分", 
                r"价格.*?分", r"评分.*?细则", r"评标.*?办法", r"分值", r"得分"
            ],
            "基础信息": [
                r"项目.*?名称", r"招标.*?编号", r"预算.*?金额", r"采购.*?金额",
                r"投标.*?截止", r"开标.*?时间", r"保证金", r"采购人", r"代理.*?机构"
            ],
            "合同条款": [
                r"合同.*?条款", r"付款.*?方式", r"交付.*?期限", r"知识.*?产权",
                r"保密.*?协议", r"违约.*?责任", r"履约.*?保证"
            ],
            "风险识别": [
                r"否决.*?项", r"废标.*?条件", r"违约", r"责任", r"赔偿", 
                r"罚款", r"特殊.*?要求", r"★", r"\*"
            ]
        }
    
    def route_query(self, query: str) -> str:
        """
        根据查询内容路由到合适的类别
        
        Args:
            query: 查询文本
            
        Returns:
            str: 查询类别
        """
        for category, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    logger.debug(f"查询路由: {query[:30]}... -> {category}")
                    return category
        
        logger.debug(f"查询路由: {query[:30]}... -> 通用")
        return "通用"
    
    def get_category_specific_queries(self, category: str, base_query: str) -> List[str]:
        """
        获取类别特定的查询扩展
        
        Args:
            category: 查询类别
            base_query: 基础查询
            
        Returns:
            List[str]: 类别特定的查询列表
        """
        category_queries = {
            "评分标准": [
                f"{base_query}",
                "评标办法前附表",
                "评分标准附表",
                "技术评分标准",
                "商务评分标准",
                "价格评分方法"
            ],
            "基础信息": [
                f"{base_query}",
                "项目概况",
                "招标公告",
                "投标须知",
                "采购需求"
            ],
            "合同条款": [
                f"{base_query}",
                "合同主要条款",
                "合同格式",
                "技术服务要求",
                "验收标准"
            ],
            "风险识别": [
                f"{base_query}",
                "投标无效条款",
                "否决投标情形",
                "特别提醒",
                "重要说明"
            ]
        }
        
        return category_queries.get(category, [base_query])
