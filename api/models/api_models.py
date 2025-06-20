"""
API 请求和响应数据模型
API Request and Response Data Models
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

# 导入现有的数据模型
from src.models.data_models import BiddingAnalysisResult


class FileUploadResponse(BaseModel):
    """文件上传响应"""
    file_id: str = Field(description="文件唯一标识")
    filename: str = Field(description="原始文件名")
    file_size: int = Field(description="文件大小（字节）")
    file_type: str = Field(description="文件类型")
    upload_time: datetime = Field(description="上传时间")
    is_converted: bool = Field(default=False, description="是否已转换为PDF")
    pdf_path: Optional[str] = Field(None, description="转换后的PDF路径")


class AnalysisRequest(BaseModel):
    """文档分析请求"""
    file_id: str = Field(description="文件ID")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="分析选项")


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"      # 等待中
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"        # 失败


class AnalysisTaskResponse(BaseModel):
    """分析任务响应"""
    task_id: str = Field(description="任务ID")
    file_id: str = Field(description="文件ID")
    status: TaskStatus = Field(description="任务状态")
    created_at: datetime = Field(description="创建时间")
    message: Optional[str] = Field(None, description="状态消息")


class AnalysisProgress(BaseModel):
    """分析进度"""
    current_step: str = Field(description="当前步骤")
    progress_percentage: int = Field(description="进度百分比 (0-100)")
    step_description: str = Field(description="步骤描述")
    estimated_remaining_time: Optional[int] = Field(None, description="预估剩余时间（秒）")
    agent_progress: Optional[Dict[str, int]] = Field(None, description="并行智能体进度 (智能体名称: 进度百分比)")


class AnalysisStatusResponse(BaseModel):
    """分析状态响应"""
    task_id: str = Field(description="任务ID")
    file_id: str = Field(description="文件ID")
    status: TaskStatus = Field(description="任务状态")
    progress: Optional[AnalysisProgress] = Field(None, description="进度信息")
    result: Optional[BiddingAnalysisResult] = Field(None, description="分析结果")
    error_message: Optional[str] = Field(None, description="错误信息")
    report_file_path: Optional[str] = Field(None, description="分析报告文件路径")
    created_at: datetime = Field(description="创建时间")
    updated_at: datetime = Field(description="更新时间")


class ErrorResponse(BaseModel):
    """错误响应"""
    error_code: str = Field(description="错误代码")
    error_message: str = Field(description="错误信息")
    details: Optional[Dict[str, Any]] = Field(None, description="错误详情")
    timestamp: datetime = Field(default_factory=datetime.now, description="错误时间")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthCheckResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(description="服务状态")
    timestamp: datetime = Field(default_factory=datetime.now, description="检查时间")
    version: str = Field(description="API版本")
    dependencies: Dict[str, str] = Field(description="依赖服务状态")


class FileListResponse(BaseModel):
    """文件列表响应"""
    files: List[FileUploadResponse] = Field(description="文件列表")
    total: int = Field(description="文件总数")
    page: int = Field(description="当前页码")
    page_size: int = Field(description="每页大小")


class TaskListResponse(BaseModel):
    """任务列表响应"""
    tasks: List[AnalysisStatusResponse] = Field(description="任务列表")
    total: int = Field(description="任务总数")
    page: int = Field(description="当前页码")
    page_size: int = Field(description="每页大小")
