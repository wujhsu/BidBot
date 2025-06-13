"""
文件服务路由
File Service Router
"""

import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from loguru import logger

from api.services.file_service import file_service

router = APIRouter(prefix="/api", tags=["文件服务"])


@router.get("/pdf/{file_id}")
async def get_pdf_file(file_id: str):
    """
    获取PDF文件（用于预览）
    
    Args:
        file_id: 文件ID
        
    Returns:
        PDF文件响应
        
    Raises:
        HTTPException: 文件不存在或无法访问
    """
    try:
        logger.debug(f"请求PDF文件: {file_id}")
        
        # 获取PDF文件路径
        pdf_path = file_service.get_pdf_path(file_id)
        
        if not pdf_path:
            raise HTTPException(
                status_code=404,
                detail=f"PDF文件不存在: {file_id}"
            )
        
        # 检查文件是否存在
        if not os.path.exists(pdf_path):
            logger.error(f"PDF文件路径不存在: {pdf_path}")
            raise HTTPException(
                status_code=404,
                detail=f"PDF文件不存在: {file_id}"
            )
        
        # 获取文件信息
        file_info = file_service.get_file_info(file_id)
        original_filename = "document.pdf"
        if file_info and file_info.get("upload_info"):
            upload_info = file_info["upload_info"]
            original_name = Path(upload_info.get("filename", "document")).stem
            original_filename = f"{original_name}.pdf"
        
        # 返回文件响应
        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename=original_filename,
            headers={
                "Cache-Control": "public, max-age=3600",  # 缓存1小时
                "Content-Disposition": f"inline; filename=\"{original_filename}\""
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取PDF文件异常: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取PDF文件失败: {str(e)}"
        )


@router.get("/download/{file_id}")
async def download_file(file_id: str, file_type: str = "pdf"):
    """
    下载文件
    
    Args:
        file_id: 文件ID
        file_type: 文件类型 ("original" 或 "pdf")
        
    Returns:
        文件下载响应
        
    Raises:
        HTTPException: 文件不存在或无法访问
    """
    try:
        logger.info(f"请求下载文件: {file_id}, 类型: {file_type}")
        
        # 获取文件信息
        file_info = file_service.get_file_info(file_id)
        if not file_info:
            raise HTTPException(
                status_code=404,
                detail=f"文件不存在: {file_id}"
            )
        
        # 根据类型选择文件路径
        if file_type == "original":
            file_path = file_info.get("original_path")
            upload_info = file_info.get("upload_info", {})
            filename = upload_info.get("filename", "document")
        else:  # pdf
            file_path = file_info.get("pdf_path")
            upload_info = file_info.get("upload_info", {})
            original_name = Path(upload_info.get("filename", "document")).stem
            filename = f"{original_name}.pdf"
        
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"请求的文件不存在: {file_type}"
            )
        
        # 确定媒体类型
        file_extension = Path(file_path).suffix.lower()
        media_type_mapping = {
            ".pdf": "application/pdf",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        }
        media_type = media_type_mapping.get(file_extension, "application/octet-stream")
        
        # 返回文件下载响应
        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=filename,
            headers={
                "Content-Disposition": f"attachment; filename=\"{filename}\""
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"下载文件异常: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"下载文件失败: {str(e)}"
        )
