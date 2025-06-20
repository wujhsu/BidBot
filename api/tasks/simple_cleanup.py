"""
简化的清理任务（不依赖schedule包）
Simple cleanup tasks without schedule dependency
"""

import asyncio
import time
import os
import shutil
from threading import Thread
from loguru import logger
from api.middleware.session import cleanup_expired_sessions


class SimpleCleanupManager:
    """简化的清理任务管理器"""
    
    def __init__(self):
        """初始化清理管理器"""
        self.cleanup_thread = None
        self.running = False
        self.cleanup_interval = 3600  # 1小时
    
    def start_cleanup_task(self):
        """启动清理任务"""
        if self.running:
            logger.warning("清理任务已在运行")
            return
        
        self.running = True
        self.cleanup_thread = Thread(target=self._run_cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("简化清理任务已启动")
    
    def stop_cleanup_task(self):
        """停止清理任务"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        
        logger.info("简化清理任务已停止")
    
    def _run_cleanup_loop(self):
        """运行清理循环"""
        while self.running:
            try:
                # 执行清理
                self._cleanup_expired_sessions()
                
                # 等待下次清理
                for _ in range(self.cleanup_interval):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"清理任务运行异常: {e}")
                time.sleep(300)  # 出错时等待5分钟再重试
    
    def _cleanup_expired_sessions(self):
        """清理过期会话"""
        try:
            logger.info("开始清理过期会话...")
            cleaned_count = cleanup_expired_sessions(
                base_upload_dir="./uploads",
                base_vector_dir="./vector_store",
                base_temp_dir="./temp",
                max_age_hours=24,  # 24小时后过期
                cleanup_empty_dirs=True  # 同时清理空目录
            )
            if cleaned_count > 0:
                logger.info(f"过期会话清理完成，清理了 {cleaned_count} 个会话")
        except Exception as e:
            logger.error(f"清理过期会话失败: {e}")
    
    def manual_cleanup(self, max_age_hours: int = 1):
        """
        手动执行清理
        
        Args:
            max_age_hours: 最大保留时间（小时）
        """
        try:
            logger.info(f"开始手动清理，保留时间: {max_age_hours}小时")
            
            # 清理会话
            session_count = cleanup_expired_sessions(
                base_upload_dir="./uploads",
                base_vector_dir="./vector_store",
                base_temp_dir="./temp",
                max_age_hours=max_age_hours,
                cleanup_empty_dirs=True
            )
            
            logger.info(f"手动清理完成 - 会话: {session_count}")
            
            return {
                "sessions_cleaned": session_count
            }
            
        except Exception as e:
            logger.error(f"手动清理失败: {e}")
            raise


# 全局清理管理器实例
simple_cleanup_manager = SimpleCleanupManager()


def start_simple_cleanup():
    """启动简化清理任务"""
    simple_cleanup_manager.start_cleanup_task()


def stop_simple_cleanup():
    """停止简化清理任务"""
    simple_cleanup_manager.stop_cleanup_task()


async def manual_cleanup_endpoint(max_age_hours: int = 1):
    """手动清理端点（用于API调用）"""
    return simple_cleanup_manager.manual_cleanup(max_age_hours)
