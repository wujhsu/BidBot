"""
文档预处理节点
Document preprocessing node for the Langgraph workflow
"""

from typing import Dict, Any
from loguru import logger
from src.models.data_models import GraphStateModel, BiddingAnalysisResult
from src.utils.document_loader import DocumentLoader
from src.utils.vector_store import VectorStoreManager
from src.utils.llm_factory import LLMFactory
from config.settings import settings
import os

class DocumentProcessor:
    """文档预处理节点"""
    
    def __init__(self, session_id: str = None):
        """
        初始化文档预处理器

        Args:
            session_id: 会话ID，用于创建会话级向量存储
        """
        self.document_loader = DocumentLoader(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )

        # 尝试创建嵌入模型，如果失败则记录错误
        try:
            self.embeddings = LLMFactory.create_embeddings()
            logger.info(f"嵌入模型初始化成功，提供商: {settings.llm_provider}")
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {e}")
            # 检查API密钥配置
            if settings.llm_provider == 'dashscope' and not settings.dashscope_api_key:
                logger.error("阿里云百炼API密钥未设置，请检查DASHSCOPE_API_KEY环境变量")
            elif settings.llm_provider == 'openai' and not settings.openai_api_key:
                logger.error("OpenAI API密钥未设置，请检查OPENAI_API_KEY环境变量")
            raise

        self.session_id = session_id
        # 创建会话级向量存储管理器
        self.vector_store_manager = VectorStoreManager(
            self.embeddings,
            session_id=session_id
        )

    def process_document(self, state: GraphStateModel) -> GraphStateModel:
        """
        处理文档：加载、分割、向量化

        Args:
            state: 图状态

        Returns:
            GraphStateModel: 更新后的状态
        """
        try:
            logger.info(f"开始处理文档: {state.document_path}")

            # 检查文件是否存在
            if not os.path.exists(state.document_path):
                error_msg = f"文件不存在: {state.document_path}"
                logger.error(error_msg)
                state.error_messages.append(error_msg)
                state.current_step = "error"
                return state
            
            # 加载和分割文档
            full_text, documents = self.document_loader.process_document(state.document_path)
            
            # 更新状态
            state.document_content = full_text
            state.chunks = [doc.page_content for doc in documents]
            
            # 创建会话级向量存储（每个会话完全隔离）
            if self.session_id:
                # 会话级隔离：为每个文档创建独立的向量存储
                logger.info(f"会话 {self.session_id}: 开始创建会话级向量存储")

                # 生成唯一的集合名称，包含文档信息
                import time
                doc_hash = abs(hash(state.document_path)) % 10000
                timestamp = int(time.time())
                collection_name = f"session_{self.session_id}_doc_{doc_hash}_{timestamp}"

                try:
                    # 先尝试清空会话的向量存储目录
                    logger.info(f"会话 {self.session_id}: 开始清理向量存储...")
                    self.vector_store_manager.clear_vector_store()

                    # 强制垃圾回收，确保资源释放
                    import gc
                    gc.collect()

                    # 创建新的向量存储
                    logger.info(f"会话 {self.session_id}: 开始创建新的向量存储...")
                    vector_store = self.vector_store_manager.create_vector_store(
                        documents,
                        collection_name=collection_name
                    )
                    logger.info(f"会话 {self.session_id}: 创建会话级向量存储成功，文档数量: {len(documents)}")

                except Exception as e:
                    logger.warning(f"会话 {self.session_id}: 清理向量存储失败，尝试强制重新初始化: {e}")
                    # 如果清理失败，强制重新初始化向量存储管理器
                    try:
                        # 重新创建嵌入模型和向量存储管理器
                        logger.info(f"会话 {self.session_id}: 重新初始化向量存储管理器...")
                        self.embeddings = LLMFactory.create_embeddings()
                        self.vector_store_manager = VectorStoreManager(
                            self.embeddings,
                            session_id=self.session_id
                        )

                        # 再次尝试创建向量存储
                        vector_store = self.vector_store_manager.create_vector_store(
                            documents,
                            collection_name=collection_name
                        )
                        logger.info(f"会话 {self.session_id}: 重新初始化后创建向量存储成功")
                    except Exception as e2:
                        logger.error(f"会话 {self.session_id}: 重新初始化也失败: {e2}")
                        raise e2

            else:
                # 兼容模式：如果没有会话ID，使用传统方式
                logger.warning("未提供会话ID，使用传统向量存储模式（可能存在多用户冲突）")
                if settings.clear_vector_store_on_new_document:
                    vector_store = self.vector_store_manager.create_isolated_vector_store(
                        documents,
                        state.document_path
                    )
                else:
                    collection_name = f"doc_{hash(state.document_path) % 10000}"
                    vector_store = self.vector_store_manager.create_vector_store(
                        documents,
                        collection_name=collection_name
                    )

            state.vector_store = vector_store
            
            # 保持现有的分析结果，不要重新创建（避免覆盖已设置的document_name）
            # 如果分析结果不存在，才创建新的
            if not hasattr(state, 'analysis_result') or state.analysis_result is None:
                file_name = os.path.basename(state.document_path)
                state.analysis_result = BiddingAnalysisResult(document_name=file_name)
                logger.info(f"创建新的分析结果，文档名称: {file_name}")
            else:
                logger.info(f"保持现有分析结果，文档名称: {state.analysis_result.document_name}")
            
            # 更新处理步骤
            state.current_step = "document_processed"
            
            logger.info(f"文档预处理完成: {len(documents)}个文档块")
            
        except Exception as e:
            error_msg = f"文档预处理失败: {str(e)}"
            logger.error(error_msg)
            state.error_messages.append(error_msg)
            state.current_step = "error"
        
        return state
    
    def validate_document(self, state: GraphStateModel) -> GraphStateModel:
        """
        验证文档内容

        Args:
            state: 图状态

        Returns:
            GraphStateModel: 更新后的状态
        """
        try:
            if not state.document_content:
                error_msg = "文档内容为空"
                logger.error(error_msg)
                state.error_messages.append(error_msg)
                state.current_step = "error"
                return state
            
            # 检查文档长度
            if len(state.document_content) < 100:
                warning_msg = "文档内容过短，可能影响分析质量"
                logger.warning(warning_msg)
                state.analysis_result.processing_notes.append(warning_msg)
            
            # 检查是否包含招标相关关键词
            bidding_keywords = [
                "招标", "投标", "采购", "评标", "开标", "中标",
                "招标公告", "招标文件", "投标文件", "评分标准"
            ]
            
            found_keywords = [kw for kw in bidding_keywords if kw in state.document_content]
            
            if not found_keywords:
                warning_msg = "文档中未发现招标相关关键词，请确认文档类型"
                logger.warning(warning_msg)
                state.analysis_result.processing_notes.append(warning_msg)
            else:
                logger.info(f"发现招标关键词: {found_keywords}")
            
            state.current_step = "document_validated"
            
        except Exception as e:
            error_msg = f"文档验证失败: {str(e)}"
            logger.error(error_msg)
            state.error_messages.append(error_msg)
            state.current_step = "error"
        
        return state
    
    def extract_document_structure(self, state: GraphStateModel) -> GraphStateModel:
        """
        提取文档结构信息

        Args:
            state: 图状态

        Returns:
            GraphStateModel: 更新后的状态
        """
        try:
            logger.info("开始提取文档结构")
            
            # 查找可能的章节标题
            structure_patterns = [
                "第一章", "第二章", "第三章", "第四章", "第五章",
                "一、", "二、", "三、", "四、", "五、", "六、", "七、", "八、", "九、", "十、",
                "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.",
                "（一）", "（二）", "（三）", "（四）", "（五）",
                "招标公告", "投标人须知", "评标办法", "合同条款", "技术规格",
                "商务要求", "资格审查", "评分标准"
            ]
            
            found_structures = []
            lines = state.document_content.split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                if line:
                    for pattern in structure_patterns:
                        if line.startswith(pattern) or pattern in line:
                            found_structures.append({
                                "line_number": i + 1,
                                "content": line[:100],  # 只保留前100个字符
                                "pattern": pattern
                            })
                            break
            
            if found_structures:
                structure_info = f"发现{len(found_structures)}个可能的章节结构"
                logger.info(structure_info)
                state.analysis_result.processing_notes.append(structure_info)
            
            state.current_step = "structure_extracted"
            
        except Exception as e:
            error_msg = f"提取文档结构失败: {str(e)}"
            logger.error(error_msg)
            state.error_messages.append(error_msg)
            state.current_step = "error"
        
        return state

def create_document_processor_node(session_id: str = None):
    """
    创建文档预处理节点函数

    Args:
        session_id: 会话ID，用于创建会话级处理器
    """
    processor = DocumentProcessor(session_id=session_id)

    def document_processor_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """文档预处理节点函数"""
        # 转换为GraphStateModel对象
        graph_state = GraphStateModel(**state)

        # 执行文档处理流程
        graph_state = processor.process_document(graph_state)
        if graph_state.current_step != "error":
            graph_state = processor.validate_document(graph_state)
        if graph_state.current_step != "error":
            graph_state = processor.extract_document_structure(graph_state)

        # 转换回字典格式
        return graph_state.model_dump()

    return document_processor_node
