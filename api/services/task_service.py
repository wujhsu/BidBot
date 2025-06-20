"""
异步任务处理服务
Asynchronous Task Processing Service
"""

import asyncio
import uuid
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from src.graph.bidding_graph import BiddingAnalysisGraph
from src.agents.parallel_aggregator import parallel_progress_manager
from api.models.api_models import TaskStatus, AnalysisProgress, AnalysisStatusResponse
from api.services.file_service import file_service


class TaskService:
    """异步任务处理服务"""
    
    def __init__(self, max_workers: int = 2):
        """
        初始化任务服务
        
        Args:
            max_workers: 最大并发工作线程数
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 任务存储（实际项目中应该使用数据库）
        self.tasks: Dict[str, Dict[str, Any]] = {}
        
        # 分析图实例（会在运行时根据会话ID创建）
        self.analysis_graph = None
        
        # 步骤映射（用于进度计算）
        self.step_mapping = {
            "start": {"progress": 0, "description": "开始分析"},
            "document_processor": {"progress": 20, "description": "文档预处理中..."},
            "basic_info_extractor": {"progress": 40, "description": "提取基础信息中..."},
            "scoring_analyzer": {"progress": 60, "description": "分析评分标准中..."},
            "other_info_extractor": {"progress": 80, "description": "提取其他信息中..."},
            "contract_info_extractor": {"progress": 80, "description": "提取合同信息中..."},
            "parallel_aggregator": {"progress": 85, "description": "聚合并行结果中..."},
            "parallel_extraction_completed": {"progress": 85, "description": "并行提取完成"},
            "partial_extraction_completed": {"progress": 75, "description": "部分提取完成"},
            "extraction_failed": {"progress": 0, "description": "提取失败"},
            "aggregation_failed": {"progress": 0, "description": "聚合失败"},
            "output_formatter": {"progress": 90, "description": "格式化结果中..."},
            "completed": {"progress": 100, "description": "分析完成"},
            "failed": {"progress": 0, "description": "分析失败"}
        }

        # 并行执行阶段的步骤
        self.parallel_steps = {
            "basic_info_extractor",
            "scoring_analyzer",
            "contract_info_extractor",
            "parallel_aggregator",
            "parallel_extraction_completed",
            "partial_extraction_completed"
        }
    
    async def create_analysis_task(self, file_id: str, session_id: str = None, options: Optional[Dict[str, Any]] = None) -> str:
        """
        创建文档分析任务

        Args:
            file_id: 文件ID
            session_id: 会话ID，用于隔离不同用户的分析任务
            options: 分析选项

        Returns:
            任务ID

        Raises:
            ValueError: 文件不存在或无效
        """
        # 验证文件存在
        file_info = file_service.get_file_info(file_id)
        if not file_info:
            raise ValueError(f"文件不存在: {file_id}")
        
        pdf_path = file_service.get_pdf_path(file_id)
        if not pdf_path:
            raise ValueError(f"PDF文件不可用: {file_id}")
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 创建任务记录
        task_info = {
            "task_id": task_id,
            "file_id": file_id,
            "session_id": session_id,  # 添加会话ID
            "pdf_path": pdf_path,
            "status": TaskStatus.PENDING,
            "progress": AnalysisProgress(
                current_step="start",
                progress_percentage=0,
                step_description="等待开始分析",
                agent_progress=None
            ),
            "result": None,
            "error_message": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "options": options or {}
        }
        
        self.tasks[task_id] = task_info
        
        # 启动异步任务
        asyncio.create_task(self._run_analysis_task(task_id))
        
        logger.info(f"创建分析任务: {task_id} (文件: {file_id})")
        return task_id
    
    async def _run_analysis_task(self, task_id: str) -> None:
        """
        运行分析任务
        
        Args:
            task_id: 任务ID
        """
        task_info = self.tasks.get(task_id)
        if not task_info:
            logger.error(f"任务不存在: {task_id}")
            return
        
        try:
            # 更新任务状态为处理中
            self._update_task_status(task_id, TaskStatus.PROCESSING, "start")
            
            # 在线程池中运行分析
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._run_analysis_sync,
                task_id,
                task_info["pdf_path"],
                task_info.get("session_id")  # 传递会话ID
            )
            
            # 检查结果
            if result and result.get("current_step") == "completed":
                # 分析成功
                self._update_task_result(task_id, TaskStatus.COMPLETED, result)
                logger.info(f"分析任务完成: {task_id}")
            else:
                # 分析失败
                error_messages = result.get("error_messages", []) if result else ["未知错误"]
                error_message = "; ".join(error_messages)
                self._update_task_error(task_id, TaskStatus.FAILED, error_message)
                logger.error(f"分析任务失败: {task_id}, 错误: {error_message}")
                
        except Exception as e:
            # 处理异常
            error_message = f"任务执行异常: {str(e)}"
            self._update_task_error(task_id, TaskStatus.FAILED, error_message)
            logger.error(f"分析任务异常: {task_id}, 错误: {error_message}")
    
    def _run_analysis_sync(self, task_id: str, pdf_path: str, session_id: str = None) -> Dict[str, Any]:
        """
        同步运行分析（在线程池中执行）

        Args:
            task_id: 任务ID
            pdf_path: PDF文件路径
            session_id: 会话ID，用于创建会话级分析图

        Returns:
            分析结果
        """
        try:
            # 创建会话级分析图
            analysis_graph = BiddingAnalysisGraph(session_id=session_id)

            # 创建一个自定义的回调来更新进度
            def progress_callback(step: str):
                self._update_task_status(task_id, TaskStatus.PROCESSING, step)

            # 获取原始文件名
            task_info = self.tasks.get(task_id)
            original_filename = None
            if task_info:
                file_id = task_info.get("file_id")
                logger.info(f"会话 {session_id}: 获取文件ID: {file_id}")
                if file_id:
                    file_info = file_service.get_file_info(file_id)
                    logger.info(f"会话 {session_id}: 文件信息: {file_info}")
                    if file_info and file_info.get("upload_info"):
                        original_filename = file_info["upload_info"].get("filename")
                        logger.info(f"会话 {session_id}: 获取到原始文件名: {original_filename}")

            # 运行分析，传递进度回调和原始文件名
            logger.info(f"会话 {session_id}: 开始分析，文件: {original_filename}")
            result = analysis_graph.run(pdf_path, progress_callback, original_filename)

            return result

        except Exception as e:
            logger.error(f"同步分析失败: {e}")
            return {"current_step": "failed", "error_messages": [str(e)]}
    
    def _update_task_status(self, task_id: str, status: TaskStatus, current_step: str) -> None:
        """
        更新任务状态

        Args:
            task_id: 任务ID
            status: 任务状态
            current_step: 当前步骤
        """
        task_info = self.tasks.get(task_id)
        if not task_info:
            return

        # 检查状态是否真的发生了变化
        old_step = task_info.get("progress", {}).current_step if task_info.get("progress") else None
        if old_step == current_step:
            return  # 状态没有变化，不需要更新

        step_info = self.step_mapping.get(current_step, {
            "progress": 0,
            "description": current_step
        })

        # 获取并行进度数据
        agent_progress = None
        progress_percentage = step_info["progress"]
        step_description = step_info["description"]

        if current_step in self.parallel_steps:
            try:
                # 获取并行进度管理器的数据
                agent_progress = parallel_progress_manager.agent_progress.copy()
                overall_progress = parallel_progress_manager.get_overall_progress()
                parallel_description = parallel_progress_manager.get_progress_description()

                # 在并行阶段使用并行进度管理器的数据
                if current_step in ["basic_info_extractor", "scoring_analyzer", "contract_info_extractor"]:
                    progress_percentage = max(20, overall_progress)  # 确保不低于文档预处理的20%
                    step_description = parallel_description
                elif current_step in ["parallel_extraction_completed", "partial_extraction_completed"]:
                    step_description = parallel_description

                logger.debug(f"并行进度数据: {agent_progress}, 总体进度: {overall_progress}%")
            except Exception as e:
                logger.warning(f"获取并行进度数据失败: {e}")
                agent_progress = None

        task_info["status"] = status
        task_info["progress"] = AnalysisProgress(
            current_step=current_step,
            progress_percentage=progress_percentage,
            step_description=step_description,
            agent_progress=agent_progress
        )
        task_info["updated_at"] = datetime.now()

        # 只在状态真正改变时记录日志
        if agent_progress:
            logger.info(f"任务 {task_id} 进度更新: {current_step} ({progress_percentage}%), 并行进度: {agent_progress}")
        else:
            logger.info(f"任务 {task_id} 进度更新: {current_step} ({progress_percentage}%)")
    
    def _update_task_result(self, task_id: str, status: TaskStatus, result: Dict[str, Any]) -> None:
        """
        更新任务结果
        
        Args:
            task_id: 任务ID
            status: 任务状态
            result: 分析结果
        """
        task_info = self.tasks.get(task_id)
        if not task_info:
            return
        
        task_info["status"] = status
        task_info["result"] = result.get("analysis_result")
        task_info["progress"] = AnalysisProgress(
            current_step="completed",
            progress_percentage=100,
            step_description="分析完成",
            agent_progress=None  # 完成时清除并行进度数据
        )
        task_info["updated_at"] = datetime.now()
    
    def _update_task_error(self, task_id: str, status: TaskStatus, error_message: str) -> None:
        """
        更新任务错误
        
        Args:
            task_id: 任务ID
            status: 任务状态
            error_message: 错误信息
        """
        task_info = self.tasks.get(task_id)
        if not task_info:
            return
        
        task_info["status"] = status
        task_info["error_message"] = error_message
        task_info["progress"] = AnalysisProgress(
            current_step="failed",
            progress_percentage=0,
            step_description="分析失败",
            agent_progress=None  # 失败时清除并行进度数据
        )
        task_info["updated_at"] = datetime.now()
    
    def get_task_status(self, task_id: str) -> Optional[AnalysisStatusResponse]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务状态响应，如果不存在返回None
        """
        task_info = self.tasks.get(task_id)
        if not task_info:
            return None
        
        return AnalysisStatusResponse(
            task_id=task_info["task_id"],
            file_id=task_info["file_id"],
            status=task_info["status"],
            progress=task_info["progress"],
            result=task_info["result"],
            error_message=task_info["error_message"],
            created_at=task_info["created_at"],
            updated_at=task_info["updated_at"]
        )
    
    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """
        清理过期任务
        
        Args:
            max_age_hours: 任务最大保存时间（小时）
            
        Returns:
            清理的任务数量
        """
        cleaned_count = 0
        current_time = datetime.now()
        
        for task_id, task_info in list(self.tasks.items()):
            created_at = task_info.get("created_at")
            if created_at and (current_time - created_at).total_seconds() > max_age_hours * 3600:
                del self.tasks[task_id]
                cleaned_count += 1
                logger.info(f"清理过期任务: {task_id}")
        
        return cleaned_count


# 全局任务服务实例
task_service = TaskService()
