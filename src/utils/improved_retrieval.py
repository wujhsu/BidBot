"""
改进的检索策略
Improved retrieval strategies specifically for bidding document analysis
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from loguru import logger
from src.utils.reranker import HybridRetriever, RerankerManager
from config.settings import settings


class ImprovedRetriever:
    """改进的检索器，专门针对招投标文件分析优化"""
    
    def __init__(self, vector_store):
        """
        初始化改进的检索器
        
        Args:
            vector_store: 向量存储
        """
        self.vector_store = vector_store
        self.reranker = RerankerManager()
        self.hybrid_retriever = HybridRetriever(vector_store, self.reranker)
    
    def retrieve_scoring_criteria(self, main_query: str = "评分标准") -> List[Tuple[Document, float, float]]:
        """
        专门用于评分标准检索的优化策略
        
        Args:
            main_query: 主查询
            
        Returns:
            List[Tuple[Document, float, float]]: (文档, 向量分数, 重排序分数)列表
        """
        try:
            # 定义评分标准相关的核心查询
            scoring_queries = [
                "评分标准 评分方法 评标办法",
                "技术分 商务分 价格分 分值",
                "评分细则 评分表 附表",
                "评标办法前附表 评分附表",
                "综合评估法 最低价评标法",
                "初步评审 详细评审",
                "否决项 废标条件 ★ *",
                "加分项 优惠条件"
            ]
            
            # 使用传统的多查询检索，避免复杂的路由逻辑
            all_results = []
            seen_contents = set()
            
            for query in scoring_queries:
                try:
                    # 直接使用向量检索，获取更多候选
                    vector_results = self.vector_store.similarity_search_with_score(
                        query, k=settings.rerank_top_k
                    )
                    
                    # 如果启用重排序，进行重排序
                    if settings.enable_reranking and self.reranker.enabled:
                        try:
                            reranked_results = self.reranker.rerank_with_scores(
                                query, vector_results, settings.rerank_final_k
                            )
                            query_results = reranked_results
                        except Exception as e:
                            logger.warning(f"重排序失败，使用向量检索结果: {e}")
                            query_results = [(doc, score, score) for doc, score in vector_results]
                    else:
                        query_results = [(doc, score, score) for doc, score in vector_results]
                    
                    # 添加到总结果中，去重
                    for doc, vec_score, rerank_score in query_results:
                        content_hash = hash(doc.page_content)
                        if content_hash not in seen_contents:
                            seen_contents.add(content_hash)
                            all_results.append((doc, vec_score, rerank_score))
                            
                except Exception as e:
                    logger.error(f"查询 '{query}' 检索失败: {e}")
                    continue
            
            # 按重排序分数排序，返回更多结果
            all_results.sort(key=lambda x: x[2], reverse=True)
            final_results = all_results[:25]  # 增加返回的文档数量
            
            logger.info(f"评分标准检索完成，返回 {len(final_results)} 个文档片段")
            return final_results
            
        except Exception as e:
            logger.error(f"评分标准检索失败: {e}")
            return []
    
    def retrieve_detailed_scoring(self, main_query: str = "评分细则") -> List[Tuple[Document, float, float]]:
        """
        专门用于详细评分细则检索的优化策略
        
        Args:
            main_query: 主查询
            
        Returns:
            List[Tuple[Document, float, float]]: (文档, 向量分数, 重排序分数)列表
        """
        try:
            # 定义详细评分相关的核心查询，特别关注附表信息
            detailed_queries = [
                "评分细则表 评分表 评分标准表",
                "技术评分标准 技术分值 技术方案 30分 20分",
                "商务评分标准 商务分值 商务条件 15分 10分",
                "价格评分方法 价格计算 价格分 15分",
                "评标办法前附表 评分附表 前附表",
                "分值构成 得分标准 评分项 分值",
                "评分细则 评分标准 评分方法",
                "附表 前附表 评分表格 表格",
                "30分 20分 15分 10分 分值 得分",
                "技术方案 技术团队 实施方案 商务条件 服务承诺"
            ]
            
            # 使用相同的策略但针对详细评分优化
            all_results = []
            seen_contents = set()
            
            for query in detailed_queries:
                try:
                    # 直接使用向量检索
                    vector_results = self.vector_store.similarity_search_with_score(
                        query, k=settings.rerank_top_k
                    )
                    
                    # 重排序
                    if settings.enable_reranking and self.reranker.enabled:
                        try:
                            reranked_results = self.reranker.rerank_with_scores(
                                query, vector_results, settings.rerank_final_k
                            )
                            query_results = reranked_results
                        except Exception as e:
                            logger.warning(f"重排序失败，使用向量检索结果: {e}")
                            query_results = [(doc, score, score) for doc, score in vector_results]
                    else:
                        query_results = [(doc, score, score) for doc, score in vector_results]
                    
                    # 去重添加
                    for doc, vec_score, rerank_score in query_results:
                        content_hash = hash(doc.page_content)
                        if content_hash not in seen_contents:
                            seen_contents.add(content_hash)
                            all_results.append((doc, vec_score, rerank_score))
                            
                except Exception as e:
                    logger.error(f"查询 '{query}' 检索失败: {e}")
                    continue
            
            # 排序并返回更多结果
            all_results.sort(key=lambda x: x[2], reverse=True)
            final_results = all_results[:30]  # 详细评分需要更多文档
            
            logger.info(f"详细评分检索完成，返回 {len(final_results)} 个文档片段")
            return final_results
            
        except Exception as e:
            logger.error(f"详细评分检索失败: {e}")
            return []
    
    def retrieve_contract_info(self, main_query: str = "合同条款") -> List[Tuple[Document, float, float]]:
        """
        专门用于合同信息检索的优化策略
        
        Args:
            main_query: 主查询
            
        Returns:
            List[Tuple[Document, float, float]]: (文档, 向量分数, 重排序分数)列表
        """
        try:
            # 定义合同相关的核心查询
            contract_queries = [
                "合同条款 合同约定 合同主要条款",
                "付款方式 付款条件 预付款 进度款",
                "交付期限 完成时间 工期 交付要求",
                "投标有效期 有效期",
                "知识产权 专利权 著作权",
                "保密协议 保密条款 商业秘密",
                "违约责任 赔偿 罚款",
                "履约保证金 保证金"
            ]
            
            return self._execute_multi_query_retrieval(contract_queries, max_results=20)
            
        except Exception as e:
            logger.error(f"合同信息检索失败: {e}")
            return []
    
    def retrieve_risk_info(self, main_query: str = "风险识别") -> List[Tuple[Document, float, float]]:
        """
        专门用于风险识别检索的优化策略
        
        Args:
            main_query: 主查询
            
        Returns:
            List[Tuple[Document, float, float]]: (文档, 向量分数, 重排序分数)列表
        """
        try:
            # 定义风险相关的核心查询
            risk_queries = [
                "违约 责任 赔偿 罚款",
                "保证金 履约保证金",
                "废标 否决 取消资格",
                "特殊要求 特别约定",
                "技术要求 性能指标",
                "时间要求 期限 工期",
                "★ * 重要 关键 注意",
                "否决项 废标条件"
            ]
            
            return self._execute_multi_query_retrieval(risk_queries, max_results=20)
            
        except Exception as e:
            logger.error(f"风险识别检索失败: {e}")
            return []
    
    def _execute_multi_query_retrieval(
        self, 
        queries: List[str], 
        max_results: int = 20
    ) -> List[Tuple[Document, float, float]]:
        """
        执行多查询检索的通用方法
        
        Args:
            queries: 查询列表
            max_results: 最大返回结果数
            
        Returns:
            List[Tuple[Document, float, float]]: 检索结果
        """
        all_results = []
        seen_contents = set()
        
        for query in queries:
            try:
                # 向量检索
                vector_results = self.vector_store.similarity_search_with_score(
                    query, k=settings.rerank_top_k
                )
                
                # 重排序
                if settings.enable_reranking and self.reranker.enabled:
                    try:
                        reranked_results = self.reranker.rerank_with_scores(
                            query, vector_results, settings.rerank_final_k
                        )
                        query_results = reranked_results
                    except Exception as e:
                        logger.warning(f"重排序失败，使用向量检索结果: {e}")
                        query_results = [(doc, score, score) for doc, score in vector_results]
                else:
                    query_results = [(doc, score, score) for doc, score in vector_results]
                
                # 去重添加
                for doc, vec_score, rerank_score in query_results:
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_contents:
                        seen_contents.add(content_hash)
                        all_results.append((doc, vec_score, rerank_score))
                        
            except Exception as e:
                logger.error(f"查询 '{query}' 检索失败: {e}")
                continue
        
        # 排序并返回结果
        all_results.sort(key=lambda x: x[2], reverse=True)
        return all_results[:max_results]
