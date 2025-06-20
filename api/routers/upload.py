"""
文件上传路由
File Upload Router
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger

from api.models.api_models import FileUploadResponse, ErrorResponse
from api.services.file_service import file_service
from api.middleware.session import SessionManager

router = APIRouter(prefix="/api", tags=["文件上传"])


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    request: Request,
    file: UploadFile = File(..., description="上传的文件（支持 .pdf, .doc, .docx 格式）")
) -> FileUploadResponse:
    """
    上传招标文件
    
    支持的文件格式：
    - PDF (.pdf)
    - Word 文档 (.doc, .docx)
    
    文件大小限制：50MB
    
    Args:
        file: 上传的文件
        
    Returns:
        FileUploadResponse: 上传结果，包含文件ID和基本信息
        
    Raises:
        HTTPException: 文件验证失败或上传失败
    """
    try:
        # 获取会话信息
        session_id = SessionManager.get_session_id(request)
        session_upload_dir = SessionManager.get_session_upload_dir(request)
        session_temp_dir = SessionManager.get_session_temp_dir(request)

        logger.info(f"会话 {session_id} 开始上传文件: {file.filename}")

        # 调用文件服务处理上传，传递会话目录
        result = await file_service.upload_file(
            file,
            session_upload_dir=session_upload_dir,
            session_temp_dir=session_temp_dir
        )

        logger.info(f"会话 {session_id} 文件上传成功: {file.filename} (ID: {result.file_id})")
        return result
        
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        logger.error(f"文件上传异常: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"文件上传失败: {str(e)}"
        )


@router.get("/files/{file_id}")
async def get_file_info(file_id: str):
    """
    获取文件信息
    
    Args:
        file_id: 文件ID
        
    Returns:
        文件信息
        
    Raises:
        HTTPException: 文件不存在
    """
    try:
        file_info = file_service.get_file_info(file_id)
        
        if not file_info:
            raise HTTPException(
                status_code=404,
                detail=f"文件不存在: {file_id}"
            )
        
        return file_info["upload_info"]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文件信息失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取文件信息失败: {str(e)}"
        )
