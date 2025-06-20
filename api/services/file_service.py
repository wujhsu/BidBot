"""
文件处理服务
File Processing Service
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
import aiofiles
from fastapi import UploadFile, HTTPException
from loguru import logger

from src.utils.document_loader import UnifiedDocumentConverter
from api.models.api_models import FileUploadResponse


class FileService:
    """文件处理服务"""

    def __init__(self, upload_dir: str = "./uploads", temp_dir: str = "./temp"):
        """
        初始化文件服务

        Args:
            upload_dir: 上传文件存储目录（会被会话中间件覆盖）
            temp_dir: 临时文件目录（会被会话中间件覆盖）
        """
        self.base_upload_dir = Path(upload_dir)
        self.base_temp_dir = Path(temp_dir)
        self.converter = UnifiedDocumentConverter()

        # 创建基础目录
        self.base_upload_dir.mkdir(exist_ok=True)
        self.base_temp_dir.mkdir(exist_ok=True)

        # 支持的文件格式
        self.supported_formats = {'.pdf', '.doc', '.docx'}

        # 最大文件大小 (50MB)
        self.max_file_size = 50 * 1024 * 1024

        # 文件信息存储（实际项目中应该使用数据库）
        # 注意：这里使用全局存储，但文件路径是会话隔离的
        self.file_registry: Dict[str, Dict[str, Any]] = {}
    
    async def upload_file(self, file: UploadFile, session_upload_dir: str = None, session_temp_dir: str = None) -> FileUploadResponse:
        """
        上传文件

        Args:
            file: 上传的文件
            session_upload_dir: 会话级上传目录
            session_temp_dir: 会话级临时目录

        Returns:
            FileUploadResponse: 上传响应

        Raises:
            HTTPException: 文件验证失败或上传失败
        """
        try:
            # 验证文件
            self._validate_file(file)

            # 使用会话级目录，如果没有提供则使用默认目录
            upload_dir = Path(session_upload_dir) if session_upload_dir else self.base_upload_dir
            temp_dir = Path(session_temp_dir) if session_temp_dir else self.base_temp_dir

            # 确保目录存在
            upload_dir.mkdir(parents=True, exist_ok=True)
            temp_dir.mkdir(parents=True, exist_ok=True)

            # 生成文件ID和路径
            file_id = str(uuid.uuid4())
            file_extension = Path(file.filename).suffix.lower()
            original_filename = f"{file_id}_original{file_extension}"
            original_path = upload_dir / original_filename
            
            # 保存原始文件
            await self._save_uploaded_file(file, original_path)
            
            # 获取文件大小
            file_size = original_path.stat().st_size
            
            # 转换为PDF（如果需要）
            pdf_path = None
            is_converted = False
            
            if file_extension != '.pdf':
                try:
                    pdf_filename = f"{file_id}.pdf"
                    pdf_path = upload_dir / pdf_filename

                    # 使用现有的转换器
                    converted_pdf_path = self.converter.convert_to_pdf(str(original_path))

                    # 移动转换后的文件到正确位置
                    if converted_pdf_path != str(original_path):
                        shutil.move(converted_pdf_path, pdf_path)
                        is_converted = True
                    
                    logger.info(f"文件转换成功: {file.filename} -> {pdf_filename}")
                    
                except Exception as e:
                    logger.error(f"文件转换失败: {e}")
                    # 转换失败不影响上传，但标记为未转换
                    pdf_path = None
                    is_converted = False
            else:
                # PDF文件直接复制
                pdf_filename = f"{file_id}.pdf"
                pdf_path = upload_dir / pdf_filename
                shutil.copy2(original_path, pdf_path)
                is_converted = False
            
            # 创建响应对象
            upload_time = datetime.now()
            response = FileUploadResponse(
                file_id=file_id,
                filename=file.filename,
                file_size=file_size,
                file_type=file_extension,
                upload_time=upload_time,
                is_converted=is_converted,
                pdf_path=str(pdf_path) if pdf_path else None
            )
            
            # 存储文件信息
            self.file_registry[file_id] = {
                "original_path": str(original_path),
                "pdf_path": str(pdf_path) if pdf_path else None,
                "upload_info": response.model_dump(),
                "created_at": upload_time
            }
            
            logger.info(f"文件上传成功: {file.filename} (ID: {file_id})")
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"文件上传失败: {e}")
            raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")
    
    def _validate_file(self, file: UploadFile) -> None:
        """
        验证上传文件
        
        Args:
            file: 上传的文件
            
        Raises:
            HTTPException: 验证失败
        """
        # 检查文件名
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        
        # 检查文件格式
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in self.supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的文件格式: {file_extension}，支持的格式: {', '.join(self.supported_formats)}"
            )
        
        # 检查文件大小（如果可获取）
        if hasattr(file, 'size') and file.size and file.size > self.max_file_size:
            raise HTTPException(
                status_code=413, 
                detail=f"文件过大，最大支持 {self.max_file_size // (1024*1024)}MB"
            )
    
    async def _save_uploaded_file(self, file: UploadFile, file_path: Path) -> None:
        """
        保存上传的文件
        
        Args:
            file: 上传的文件
            file_path: 保存路径
        """
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            
            # 检查实际文件大小
            if len(content) > self.max_file_size:
                raise HTTPException(
                    status_code=413, 
                    detail=f"文件过大，最大支持 {self.max_file_size // (1024*1024)}MB"
                )
            
            await f.write(content)
    
    def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        获取文件信息
        
        Args:
            file_id: 文件ID
            
        Returns:
            文件信息字典，如果不存在返回None
        """
        return self.file_registry.get(file_id)
    
    def get_pdf_path(self, file_id: str) -> Optional[str]:
        """
        获取PDF文件路径
        
        Args:
            file_id: 文件ID
            
        Returns:
            PDF文件路径，如果不存在返回None
        """
        file_info = self.get_file_info(file_id)
        if file_info:
            return file_info.get("pdf_path")
        return None
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """
        清理过期文件
        
        Args:
            max_age_hours: 文件最大保存时间（小时）
            
        Returns:
            清理的文件数量
        """
        cleaned_count = 0
        current_time = datetime.now()
        
        for file_id, file_info in list(self.file_registry.items()):
            created_at = file_info.get("created_at")
            if created_at and (current_time - created_at).total_seconds() > max_age_hours * 3600:
                try:
                    # 删除文件
                    original_path = file_info.get("original_path")
                    pdf_path = file_info.get("pdf_path")
                    
                    if original_path and os.path.exists(original_path):
                        os.remove(original_path)
                    
                    if pdf_path and os.path.exists(pdf_path):
                        os.remove(pdf_path)
                    
                    # 从注册表中移除
                    del self.file_registry[file_id]
                    cleaned_count += 1
                    
                    logger.info(f"清理过期文件: {file_id}")
                    
                except Exception as e:
                    logger.error(f"清理文件失败 {file_id}: {e}")
        
        return cleaned_count


# 全局文件服务实例
file_service = FileService()
