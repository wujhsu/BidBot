"""
智能投标助手Langgraph图定义
Langgraph workflow definition for the Intelligent Bidding Assistant
"""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from loguru import logger
from src.models.data_models import GraphState

from src.agents.document_processor import create_document_processor_node
from src.agents.basic_info_extractor import create_basic_info_extractor_node
from src.agents.scoring_analyzer import create_scoring_analyzer_node
from src.agents.other_info_extractor import create_other_info_extractor_node
from src.agents.output_formatter import create_output_formatter_node

class BiddingAnalysisGraph:
    """智能投标助手分析图"""
    
    def __init__(self):
        """初始化分析图"""
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """创建Langgraph工作流图"""
        
        # 创建状态图
        workflow = StateGraph(GraphState)
        
        # 添加节点
        workflow.add_node("document_processor", create_document_processor_node())
        workflow.add_node("basic_info_extractor", create_basic_info_extractor_node())
        workflow.add_node("scoring_analyzer", create_scoring_analyzer_node())
        workflow.add_node("other_info_extractor", create_other_info_extractor_node())
        workflow.add_node("output_formatter", create_output_formatter_node())
        workflow.add_node("error_handler", self._create_error_handler_node())
        
        # 设置入口点
        workflow.set_entry_point("document_processor")
        
        # 添加条件边
        workflow.add_conditional_edges(
            "document_processor",
            self._route_after_document_processing,
            {
                "continue": "basic_info_extractor",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "basic_info_extractor",
            self._route_after_basic_info,
            {
                "continue": "scoring_analyzer",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "scoring_analyzer",
            self._route_after_scoring,
            {
                "continue": "other_info_extractor",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "other_info_extractor",
            self._route_after_other_info,
            {
                "continue": "output_formatter",
                "error": "error_handler"
            }
        )
        
        # 添加结束边
        workflow.add_edge("output_formatter", END)
        workflow.add_edge("error_handler", END)
        
        return workflow.compile()
    
    def _route_after_document_processing(self, state: Dict[str, Any]) -> Literal["continue", "error"]:
        """文档处理后的路由决策"""
        current_step = state.get("current_step", "")
        if current_step == "error":
            return "error"
        elif current_step in ["document_processed", "document_validated", "structure_extracted"]:
            return "continue"
        else:
            logger.warning(f"未知的文档处理状态: {current_step}")
            return "continue"
    
    def _route_after_basic_info(self, state: Dict[str, Any]) -> Literal["continue", "error"]:
        """基础信息提取后的路由决策"""
        current_step = state.get("current_step", "")
        if current_step == "error":
            return "error"
        elif current_step == "basic_info_extracted":
            return "continue"
        else:
            logger.warning(f"未知的基础信息提取状态: {current_step}")
            return "continue"
    
    def _route_after_scoring(self, state: Dict[str, Any]) -> Literal["continue", "error"]:
        """评分分析后的路由决策"""
        current_step = state.get("current_step", "")
        if current_step == "error":
            return "error"
        elif current_step == "scoring_analyzed":
            return "continue"
        else:
            logger.warning(f"未知的评分分析状态: {current_step}")
            return "continue"
    
    def _route_after_other_info(self, state: Dict[str, Any]) -> Literal["continue", "error"]:
        """其他信息提取后的路由决策"""
        current_step = state.get("current_step", "")
        if current_step == "error":
            return "error"
        elif current_step == "other_info_extracted":
            return "continue"
        else:
            logger.warning(f"未知的其他信息提取状态: {current_step}")
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
    
    def run(self, document_path: str) -> Dict[str, Any]:
        """
        运行分析流程
        
        Args:
            document_path: 文档路径
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            logger.info(f"开始分析文档: {document_path}")
            
            # 初始化状态
            from src.models.data_models import BiddingAnalysisResult
            import os

            initial_state = {
                "document_path": document_path,
                "document_content": None,
                "chunks": [],
                "vector_store": None,
                "analysis_result": BiddingAnalysisResult(document_name=os.path.basename(document_path)),
                "current_step": "start",
                "error_messages": [],
                "retry_count": 0
            }
            
            # 运行图
            final_state = self.graph.invoke(initial_state)
            
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
    C -->|否| H[错误处理]
    D --> E{提取成功?}
    E -->|是| F[评分标准分析]
    E -->|否| H
    F --> G{分析成功?}
    G -->|是| I[其他信息提取]
    G -->|否| H
    I --> J{提取成功?}
    J -->|是| K[结果格式化输出]
    J -->|否| H
    K --> L[结束]
    H --> M[尝试输出部分结果]
    M --> L
    
    style A fill:#e1f5fe
    style L fill:#e8f5e8
    style H fill:#ffebee
    style B fill:#f3e5f5
    style D fill:#f3e5f5
    style F fill:#f3e5f5
    style I fill:#f3e5f5
    style K fill:#fff3e0
"""
        return mermaid_graph

# 创建全局图实例
bidding_graph = BiddingAnalysisGraph()
