"""
智能投标助手Langgraph图定义
Langgraph workflow definition for the Intelligent Bidding Assistant
"""

from typing import Dict, Any, Literal, Optional, Callable
from langgraph.graph import StateGraph, END
from loguru import logger
from src.models.data_models import GraphState, GraphStateModel

from src.agents.document_processor import create_document_processor_node
from src.agents.basic_info_extractor import create_basic_info_extractor_node
from src.agents.scoring_analyzer import create_scoring_analyzer_node
from src.agents.other_info_extractor import create_contract_info_extractor_node
from src.agents.output_formatter import create_output_formatter_node
from src.agents.parallel_aggregator import create_parallel_aggregator_node, parallel_progress_manager

class BiddingAnalysisGraph:
    """智能投标助手分析图"""
    
    def __init__(self, session_id: str = None):
        """
        初始化分析图

        Args:
            session_id: 会话ID，用于创建会话级节点
        """
        self.session_id = session_id
        self.graph = self._create_graph()
        self.progress_callback: Optional[Callable[[str], None]] = None
    
    def _create_graph(self) -> StateGraph:
        """创建Langgraph工作流图"""

        # 创建状态图
        workflow = StateGraph(GraphState)

        # 添加节点（传递会话ID给需要的节点）
        workflow.add_node("document_processor", create_document_processor_node(self.session_id))
        workflow.add_node("basic_info_extractor", create_basic_info_extractor_node())
        workflow.add_node("scoring_analyzer", create_scoring_analyzer_node())
        workflow.add_node("contract_info_extractor", create_contract_info_extractor_node())
        workflow.add_node("parallel_aggregator", create_parallel_aggregator_node())
        workflow.add_node("output_formatter", create_output_formatter_node())
        workflow.add_node("error_handler", self._create_error_handler_node())
        
        # 设置入口点
        workflow.set_entry_point("document_processor")

        # 并行执行模式
        workflow.add_edge("document_processor", "basic_info_extractor")
        workflow.add_edge("document_processor", "scoring_analyzer")
        workflow.add_edge("document_processor", "contract_info_extractor")

        # 所有并行节点完成后，进入聚合节点
        workflow.add_edge("basic_info_extractor", "parallel_aggregator")
        workflow.add_edge("scoring_analyzer", "parallel_aggregator")
        workflow.add_edge("contract_info_extractor", "parallel_aggregator")

        # 聚合完成后进入输出格式化
        workflow.add_conditional_edges(
            "parallel_aggregator",
            self._route_after_parallel_aggregation,
            {
                "continue": "output_formatter",
                "error": "error_handler"
            }
        )
        
        # 添加结束边
        workflow.add_edge("output_formatter", END)
        workflow.add_edge("error_handler", END)
        
        return workflow.compile()

    def _update_progress(self, step: str) -> None:
        """更新进度"""
        if self.progress_callback:
            self.progress_callback(step)


    
    def _route_after_parallel_aggregation(self, state: Dict[str, Any]) -> Literal["continue", "error"]:
        """并行聚合后的路由决策"""
        current_step = state.get("current_step", "")
        if current_step == "error" or current_step == "aggregation_failed":
            return "error"
        elif current_step in ["parallel_extraction_completed", "partial_extraction_completed"]:
            # 更新进度到结果格式化
            self._update_progress("output_formatter")
            return "continue"
        elif current_step == "extraction_failed":
            logger.warning("所有并行提取都失败，进入错误处理")
            return "error"
        else:
            logger.warning(f"未知的并行聚合状态: {current_step}")
            return "continue"
    
    def _create_error_handler_node(self):
        """创建错误处理节点"""
        def error_handler_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """错误处理节点函数"""
            logger.error("进入错误处理节点")
            
            error_messages = state.get("error_messages", [])
            if error_messages:
                logger.error(f"错误信息: {error_messages}")
            
            # 尝试生成部分结果
            try:
                # 如果有部分分析结果，仍然尝试格式化输出
                if state.get("analysis_result"):
                    logger.info("尝试输出部分分析结果")
                    formatter_node = create_output_formatter_node()
                    state = formatter_node(state)
                else:
                    state["current_step"] = "failed"
                    logger.error("无法生成任何分析结果")
            except Exception as e:
                logger.error(f"错误处理失败: {e}")
                state["current_step"] = "failed"
            
            return state
        
        return error_handler_node
    
    def run(self, document_path: str, progress_callback: Optional[Callable[[str], None]] = None, original_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        运行分析流程

        Args:
            document_path: 文档路径
            progress_callback: 进度回调函数
            original_filename: 原始文件名（用于显示）

        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            logger.info(f"开始分析文档: {document_path}")

            # 设置进度回调
            self.progress_callback = progress_callback

            # 初始化状态
            from src.models.data_models import BiddingAnalysisResult
            import os
            from pathlib import Path

            # 确定显示的文档名称
            logger.info(f"分析图接收到的原始文件名: {original_filename}")
            if original_filename:
                # 使用原始文件名，去掉扩展名
                display_name = Path(original_filename).stem
                logger.info(f"使用原始文件名，显示名称: {display_name}")
            else:
                # 回退到使用文件路径的basename
                display_name = os.path.basename(document_path)
                logger.info(f"使用文件路径basename，显示名称: {display_name}")

            initial_state = GraphStateModel(
                document_path=document_path,
                document_content=None,
                chunks=[],
                vector_store=None,
                analysis_result=BiddingAnalysisResult(document_name=display_name),
                current_step="start",
                error_messages=[],
                retry_count=0
            )

            # 调用进度回调 - 开始文档处理
            if self.progress_callback:
                self.progress_callback("document_processor")

            # 运行图
            final_state = self.graph.invoke(initial_state.model_dump())

            # 如果分析成功完成，调用最终进度更新
            if final_state.get('current_step') == 'completed' and self.progress_callback:
                self.progress_callback("completed")

            logger.info(f"分析完成，最终状态: {final_state.get('current_step', 'unknown')}")
            return final_state
            
        except Exception as e:
            logger.error(f"运行分析流程失败: {e}")
            return {
                "current_step": "failed",
                "error_messages": [str(e)],
                "document_path": document_path
            }
    
    def get_graph_visualization(self) -> str:
        """
        获取图的可视化表示
        
        Returns:
            str: 图的Mermaid格式描述
        """
        mermaid_graph = """
graph TD
    A[开始] --> B[文档预处理]
    B --> C{处理成功?}
    C -->|是| D[基础信息提取]
    C -->|是| E[评分标准分析]
    C -->|是| F[合同信息提取]
    C -->|否| H[错误处理]

    D --> G[并行结果聚合]
    E --> G
    F --> G

    G --> I{聚合成功?}
    I -->|是| J[结果格式化输出]
    I -->|否| H
    J --> K[结束]
    H --> L[尝试输出部分结果]
    L --> K

    style A fill:#e1f5fe
    style K fill:#e8f5e8
    style H fill:#ffebee
    style B fill:#f3e5f5
    style D fill:#e3f2fd
    style E fill:#e3f2fd
    style F fill:#e3f2fd
    style G fill:#fff3e0
    style J fill:#fff3e0
"""
        return mermaid_graph

# 创建全局图实例
bidding_graph = BiddingAnalysisGraph()
