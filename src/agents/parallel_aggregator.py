"""
并行状态聚合节点
Parallel State Aggregator Node for the Intelligent Bidding Assistant
"""

import threading
from typing import Dict, Any, List, Optional
from loguru import logger
from src.models.data_models import GraphState, GraphStateModel


class ParallelAggregator:
    """并行状态聚合器"""
    
    def __init__(self):
        """初始化并行状态聚合器"""
        self.error_lock = threading.Lock()
        self.state_lock = threading.Lock()
    
    def aggregate_parallel_results(self, state: GraphStateModel) -> GraphStateModel:
        """
        聚合并行执行结果
        
        Args:
            state: 图状态
            
        Returns:
            GraphStateModel: 聚合后的状态
        """
        try:
            logger.info("开始聚合并行执行结果")
            
            # 验证analysis_result的完整性
            completeness_status = self._validate_analysis_completeness(state)
            
            # 设置最终的current_step状态
            final_step = self._determine_final_step(state, completeness_status)
            
            # 线程安全地更新状态
            with self.state_lock:
                state.current_step = final_step
            
            # 生成聚合日志
            self._log_aggregation_summary(state, completeness_status)
            
            logger.info(f"并行结果聚合完成，最终状态: {final_step}")
            
        except Exception as e:
            error_msg = f"并行结果聚合失败: {str(e)}"
            logger.error(error_msg)
            self._safe_append_error(state, error_msg)
            state.current_step = "aggregation_failed"
        
        return state
    
    def _validate_analysis_completeness(self, state: GraphStateModel) -> Dict[str, bool]:
        """
        验证分析结果的完整性
        
        Args:
            state: 图状态
            
        Returns:
            Dict[str, bool]: 各部分的完整性状态
        """
        completeness = {
            "basic_information": False,
            "scoring_criteria": False,
            "contract_information": False
        }
        
        try:
            # 检查基础信息完整性
            basic_info = state.analysis_result.basic_information
            if (basic_info.project_name.value or 
                basic_info.tender_number.value or 
                basic_info.qualification_criteria.company_certifications):
                completeness["basic_information"] = True
            
            # 检查评分标准完整性
            scoring = state.analysis_result.scoring_criteria
            if (scoring.evaluation_method.value or 
                scoring.detailed_scoring or 
                scoring.score_composition.technical_score.value):
                completeness["scoring_criteria"] = True
            
            # 检查合同信息完整性
            contract = state.analysis_result.contract_information
            if (contract.contract_terms or 
                contract.breach_liability or 
                contract.payment_terms.value):
                completeness["contract_information"] = True
                
        except Exception as e:
            logger.warning(f"完整性验证过程中出现异常: {e}")
        
        logger.info(f"分析完整性检查: {completeness}")
        return completeness
    
    def _determine_final_step(self, state: GraphStateModel, completeness: Dict[str, bool]) -> str:
        """
        根据完整性状态确定最终步骤
        
        Args:
            state: 图状态
            completeness: 完整性状态
            
        Returns:
            str: 最终步骤状态
        """
        # 检查是否有错误
        if state.error_messages:
            # 有错误但有部分结果
            completed_count = sum(completeness.values())
            if completed_count > 0:
                return "partial_extraction_completed"
            else:
                return "extraction_failed"
        
        # 无错误情况
        completed_count = sum(completeness.values())
        if completed_count == 3:
            return "parallel_extraction_completed"
        elif completed_count > 0:
            return "partial_extraction_completed"
        else:
            return "extraction_failed"
    
    def _safe_append_error(self, state: GraphStateModel, error_msg: str) -> None:
        """
        追加错误信息（LangGraph自动处理并发）

        Args:
            state: 图状态
            error_msg: 错误信息
        """
        # LangGraph的Annotated类型会自动处理并发追加
        state.error_messages.append(error_msg)
    
    def _log_aggregation_summary(self, state: GraphStateModel, completeness: Dict[str, bool]) -> None:
        """
        记录聚合摘要日志
        
        Args:
            state: 图状态
            completeness: 完整性状态
        """
        completed_modules = [module for module, status in completeness.items() if status]
        failed_modules = [module for module, status in completeness.items() if not status]
        
        logger.info("=== 并行执行聚合摘要 ===")
        logger.info(f"成功完成的模块: {completed_modules}")
        if failed_modules:
            logger.warning(f"未完成的模块: {failed_modules}")
        
        if state.error_messages:
            logger.warning(f"执行过程中的错误数量: {len(state.error_messages)}")
            for i, error in enumerate(state.error_messages[-3:], 1):  # 只显示最后3个错误
                logger.warning(f"错误 {i}: {error}")
        
        logger.info("========================")


class ParallelProgressManager:
    """并行进度管理器"""
    
    def __init__(self):
        """初始化并行进度管理器"""
        self.progress_lock = threading.Lock()
        self.agent_progress = {
            "basic_info_extractor": 0,
            "scoring_analyzer": 0,
            "contract_info_extractor": 0
        }
    
    def update_agent_progress(self, agent_name: str, progress: int) -> None:
        """
        更新智能体进度
        
        Args:
            agent_name: 智能体名称
            progress: 进度百分比 (0-100)
        """
        with self.progress_lock:
            if agent_name in self.agent_progress:
                self.agent_progress[agent_name] = progress
                logger.debug(f"智能体 {agent_name} 进度更新: {progress}%")
    
    def get_overall_progress(self) -> int:
        """
        获取总体进度
        
        Returns:
            int: 总体进度百分比 (0-100)
        """
        with self.progress_lock:
            total_progress = sum(self.agent_progress.values())
            overall_progress = total_progress // 3  # 平均进度
            return min(100, overall_progress)
    
    def get_progress_description(self) -> str:
        """
        获取进度描述
        
        Returns:
            str: 进度描述
        """
        with self.progress_lock:
            completed_agents = [name for name, progress in self.agent_progress.items() if progress >= 100]
            in_progress_agents = [name for name, progress in self.agent_progress.items() if 0 < progress < 100]
            
            if len(completed_agents) == 3:
                return "并行提取完成，正在聚合结果..."
            elif completed_agents:
                return f"并行提取中... ({len(completed_agents)}/3 已完成)"
            elif in_progress_agents:
                return "并行提取中..."
            else:
                return "准备开始并行提取..."


def create_parallel_aggregator_node():
    """创建并行状态聚合节点函数"""
    aggregator = ParallelAggregator()
    
    def parallel_aggregator_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """并行状态聚合节点函数"""
        # 转换为GraphStateModel对象
        graph_state = GraphStateModel(**state)
        
        # 执行并行结果聚合
        graph_state = aggregator.aggregate_parallel_results(graph_state)
        
        # 转换回字典格式
        return graph_state.model_dump()
    
    return parallel_aggregator_node


# 全局进度管理器实例
parallel_progress_manager = ParallelProgressManager()
