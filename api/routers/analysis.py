"""
文档分析路由
Document Analysis Router
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from api.models.api_models import (
    AnalysisRequest, 
    AnalysisTaskResponse, 
    AnalysisStatusResponse,
    TaskStatus
)
from api.services.task_service import task_service

router = APIRouter(prefix="/api", tags=["文档分析"])


@router.post("/analyze", response_model=AnalysisTaskResponse)
async def start_analysis(request: AnalysisRequest) -> AnalysisTaskResponse:
    """
    启动文档分析任务
    
    Args:
        request: 分析请求，包含文件ID和分析选项
        
    Returns:
        AnalysisTaskResponse: 任务信息，包含任务ID
        
    Raises:
        HTTPException: 文件不存在或任务创建失败
    """
    try:
        logger.info(f"开始创建分析任务，文件ID: {request.file_id}")
        
        # 创建分析任务
        task_id = await task_service.create_analysis_task(
            file_id=request.file_id,
            options=request.options
        )
        
        # 获取任务状态
        task_status = task_service.get_task_status(task_id)
        if not task_status:
            raise HTTPException(
                status_code=500,
                detail="任务创建失败"
            )
        
        response = AnalysisTaskResponse(
            task_id=task_id,
            file_id=request.file_id,
            status=task_status.status,
            created_at=task_status.created_at,
            message="分析任务已创建，正在处理中..."
        )
        
        logger.info(f"分析任务创建成功: {task_id}")
        return response
        
    except ValueError as e:
        # 文件不存在等业务逻辑错误
        logger.warning(f"分析任务创建失败: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"分析任务创建异常: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"任务创建失败: {str(e)}"
        )


@router.get("/analysis/{task_id}", response_model=AnalysisStatusResponse)
async def get_analysis_status(task_id: str) -> AnalysisStatusResponse:
    """
    获取分析任务状态和结果
    
    Args:
        task_id: 任务ID
        
    Returns:
        AnalysisStatusResponse: 任务状态和结果
        
    Raises:
        HTTPException: 任务不存在
    """
    try:
        # 降低日志级别，避免频繁轮询产生过多日志
    # logger.debug(f"查询分析任务状态: {task_id}")
        
        # 获取任务状态
        task_status = task_service.get_task_status(task_id)
        
        if not task_status:
            raise HTTPException(
                status_code=404,
                detail=f"任务不存在: {task_id}"
            )
        
        return task_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务状态异常: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取任务状态失败: {str(e)}"
        )


@router.delete("/analysis/{task_id}")
async def cancel_analysis(task_id: str):
    """
    取消分析任务（暂不实现，预留接口）
    
    Args:
        task_id: 任务ID
        
    Returns:
        取消结果
    """
    # TODO: 实现任务取消逻辑
    raise HTTPException(
        status_code=501,
        detail="任务取消功能暂未实现"
    )


@router.get("/analysis")
async def list_analysis_tasks(
    page: int = 1,
    page_size: int = 10,
    status: str = None
):
    """
    获取分析任务列表（暂不实现，预留接口）
    
    Args:
        page: 页码
        page_size: 每页大小
        status: 状态过滤
        
    Returns:
        任务列表
    """
    # TODO: 实现任务列表查询
    raise HTTPException(
        status_code=501,
        detail="任务列表功能暂未实现"
    )
