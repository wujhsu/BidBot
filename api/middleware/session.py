"""
会话管理中间件
Session management middleware for multi-user isolation
"""

import os
import re
import time
from typing import Optional
from fastapi import Request, HTTPException
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger


class SessionMiddleware(BaseHTTPMiddleware):
    """会话管理中间件，为每个用户会话提供隔离的工作环境"""
    
    def __init__(self, app, base_upload_dir: str = "./uploads", base_vector_dir: str = "./vector_store"):
        super().__init__(app)
        self.base_upload_dir = base_upload_dir
        self.base_vector_dir = base_vector_dir
        
        # 创建基础目录
        os.makedirs(base_upload_dir, exist_ok=True)
        os.makedirs(base_vector_dir, exist_ok=True)
    
    async def dispatch(self, request: Request, call_next):
        """处理请求，添加会话隔离"""

        # 从请求头获取会话ID
        session_id = self._extract_session_id(request)

        # 检查是否需要强制生成新会话ID（用于处理会话重置）
        force_new_session = request.headers.get("X-Force-New-Session", "").lower() == "true"

        if not session_id or force_new_session:
            # 如果没有会话ID或强制要求新会话，生成一个新的
            session_id = self._generate_session_id()
            if force_new_session:
                logger.info(f"强制生成新会话ID: {session_id}")
            else:
                logger.info(f"生成新会话ID: {session_id}")

        # 验证会话ID格式
        if not self._validate_session_id(session_id):
            raise HTTPException(
                status_code=400,
                detail=f"无效的会话ID格式: {session_id}"
            )
        
        # 创建会话级目录
        session_upload_dir = os.path.join(self.base_upload_dir, session_id)
        session_vector_dir = os.path.join(self.base_vector_dir, session_id)
        session_temp_dir = os.path.join("./temp", session_id)
        
        # 确保会话目录存在
        os.makedirs(session_upload_dir, exist_ok=True)
        os.makedirs(session_vector_dir, exist_ok=True)
        os.makedirs(session_temp_dir, exist_ok=True)
        
        # 将会话信息添加到请求状态
        request.state.session_id = session_id
        request.state.session_upload_dir = session_upload_dir
        request.state.session_vector_dir = session_vector_dir
        request.state.session_temp_dir = session_temp_dir
        
        # 只在首次创建会话时记录日志，避免频繁轮询产生过多日志
        if not os.path.exists(session_upload_dir):
            logger.info(f"创建新会话 {session_id} - 上传目录: {session_upload_dir}")
            logger.info(f"创建新会话 {session_id} - 向量目录: {session_vector_dir}")
        
        # 处理请求
        response = await call_next(request)
        
        # 在响应头中返回会话ID
        if isinstance(response, Response):
            response.headers["X-Session-ID"] = session_id
        
        return response
    
    def _extract_session_id(self, request: Request) -> Optional[str]:
        """从请求中提取会话ID"""
        
        # 优先从请求头获取
        session_id = request.headers.get("X-Session-ID")
        if session_id:
            return session_id
        
        # 从查询参数获取
        session_id = request.query_params.get("session_id")
        if session_id:
            return session_id
        
        return None
    
    def _generate_session_id(self) -> str:
        """生成新的会话ID"""
        import uuid
        timestamp = int(time.time())
        unique_id = uuid.uuid4().hex[:8]
        return f"session_{timestamp}_{unique_id}"
    
    def _validate_session_id(self, session_id: str) -> bool:
        """验证会话ID格式"""
        if not session_id:
            return False

        # 检查格式：session_timestamp_uniqueid (支持字母数字组合)
        pattern = r'^session_\d+_[a-zA-Z0-9]{8,}$'
        return bool(re.match(pattern, session_id))


class SessionManager:
    """会话管理器，提供会话相关的工具方法"""
    
    @staticmethod
    def get_session_info(request: Request) -> dict:
        """获取当前请求的会话信息"""
        return {
            "session_id": getattr(request.state, "session_id", None),
            "upload_dir": getattr(request.state, "session_upload_dir", None),
            "vector_dir": getattr(request.state, "session_vector_dir", None),
            "temp_dir": getattr(request.state, "session_temp_dir", None)
        }
    
    @staticmethod
    def get_session_id(request: Request) -> str:
        """获取当前请求的会话ID"""
        session_id = getattr(request.state, "session_id", None)
        if not session_id:
            raise HTTPException(
                status_code=500,
                detail="会话ID未找到，请检查会话中间件配置"
            )
        return session_id
    
    @staticmethod
    def get_session_upload_dir(request: Request) -> str:
        """获取当前会话的上传目录"""
        upload_dir = getattr(request.state, "session_upload_dir", None)
        if not upload_dir:
            raise HTTPException(
                status_code=500,
                detail="会话上传目录未找到"
            )
        return upload_dir
    
    @staticmethod
    def get_session_vector_dir(request: Request) -> str:
        """获取当前会话的向量存储目录"""
        vector_dir = getattr(request.state, "session_vector_dir", None)
        if not vector_dir:
            raise HTTPException(
                status_code=500,
                detail="会话向量存储目录未找到"
            )
        return vector_dir
    
    @staticmethod
    def get_session_temp_dir(request: Request) -> str:
        """获取当前会话的临时目录"""
        temp_dir = getattr(request.state, "session_temp_dir", None)
        if not temp_dir:
            raise HTTPException(
                status_code=500,
                detail="会话临时目录未找到"
            )
        return temp_dir


def cleanup_expired_sessions(base_upload_dir: str = "./uploads",
                           base_vector_dir: str = "./vector_store",
                           base_temp_dir: str = "./temp",
                           max_age_hours: int = 24,
                           cleanup_empty_dirs: bool = True):
    """清理过期的会话数据和空目录"""
    try:
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)

        cleaned_count = 0
        empty_dirs_cleaned = 0

        # 要清理的目录列表
        dirs_to_clean = [
            (base_upload_dir, "上传"),
            (base_vector_dir, "向量存储"),
            (base_temp_dir, "临时文件")
        ]

        for base_dir, dir_type in dirs_to_clean:
            if not os.path.exists(base_dir):
                continue

            for session_dir in os.listdir(base_dir):
                if session_dir.startswith("session_"):
                    session_path = os.path.join(base_dir, session_dir)
                    if os.path.isdir(session_path):
                        # 从会话ID中提取时间戳
                        try:
                            timestamp = int(session_dir.split("_")[1])
                            if timestamp < cutoff_time:
                                import shutil
                                shutil.rmtree(session_path)
                                cleaned_count += 1
                                logger.info(f"清理过期会话{dir_type}目录: {session_dir}")
                        except (IndexError, ValueError):
                            logger.warning(f"无法解析会话时间戳: {session_dir}")

        # 清理空目录（如果启用）
        if cleanup_empty_dirs:
            for base_dir, dir_type in dirs_to_clean:
                empty_count = _cleanup_empty_directories(base_dir, dir_type)
                empty_dirs_cleaned += empty_count

        total_message = f"会话清理完成，清理了 {cleaned_count} 个过期会话"
        if cleanup_empty_dirs and empty_dirs_cleaned > 0:
            total_message += f"，清理了 {empty_dirs_cleaned} 个空目录"
        logger.info(total_message)

        return cleaned_count

    except Exception as e:
        logger.error(f"清理过期会话失败: {e}")
        return 0


def _cleanup_empty_directories(base_dir: str, dir_type: str) -> int:
    """清理指定目录下的空目录"""
    if not os.path.exists(base_dir):
        return 0

    cleaned_count = 0

    try:
        # 递归查找并删除空目录
        for root, _, _ in os.walk(base_dir, topdown=False):
            # 跳过基础目录本身
            if root == base_dir:
                continue

            # 检查目录是否为空
            try:
                if not os.listdir(root):
                    os.rmdir(root)
                    cleaned_count += 1
                    relative_path = os.path.relpath(root, base_dir)
                    logger.info(f"清理空{dir_type}目录: {relative_path}")
            except OSError as e:
                # 目录可能不为空或有权限问题
                logger.debug(f"无法删除目录 {root}: {e}")
                continue

    except Exception as e:
        logger.error(f"清理{dir_type}空目录时出错: {e}")

    return cleaned_count
