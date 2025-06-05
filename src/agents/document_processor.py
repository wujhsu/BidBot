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
    
    def __init__(self):
        """初始化文档预处理器"""
        self.document_loader = DocumentLoader(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        self.embeddings = LLMFactory.create_embeddings()
        self.vector_store_manager = VectorStoreManager(self.embeddings)

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
            
            # 创建向量存储（根据配置决定是否隔离）
            if settings.clear_vector_store_on_new_document:
                # 使用隔离模式，确保新文档不受历史数据影响
                vector_store = self.vector_store_manager.create_isolated_vector_store(
                    documents,
                    state.document_path
                )
                logger.info("使用隔离模式创建向量存储，历史数据已清空")
            else:
                # 使用传统模式，可能存在历史数据交叉污染
                collection_name = f"doc_{hash(state.document_path) % 10000}"
                vector_store = self.vector_store_manager.create_vector_store(
                    documents,
                    collection_name=collection_name
                )
                logger.warning("使用传统模式创建向量存储，可能存在历史数据交叉污染风险")
            state.vector_store = vector_store
            
            # 初始化分析结果
            file_name = os.path.basename(state.document_path)
            state.analysis_result = BiddingAnalysisResult(document_name=file_name)
            
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

def create_document_processor_node():
    """创建文档预处理节点函数"""
    processor = DocumentProcessor()
    
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
