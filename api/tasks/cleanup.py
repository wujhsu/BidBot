"""
定期清理任务
Periodic cleanup tasks for session data and temporary files
"""

import asyncio
import schedule
import time
from threading import Thread
from loguru import logger
from api.middleware.session import cleanup_expired_sessions
from api.services.file_service import file_service
from api.services.task_service import task_service


class CleanupManager:
    """清理任务管理器"""
    
    def __init__(self):
        """初始化清理管理器"""
        self.cleanup_thread = None
        self.running = False
    
    def start_cleanup_scheduler(self):
        """启动清理调度器"""
        if self.running:
            logger.warning("清理调度器已在运行")
            return
        
        # 配置清理任务
        schedule.every(1).hours.do(self._cleanup_expired_sessions)
        schedule.every(6).hours.do(self._cleanup_old_files)
        schedule.every(12).hours.do(self._cleanup_old_tasks)
        
        # 启动调度线程
        self.running = True
        self.cleanup_thread = Thread(target=self._run_scheduler, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("清理调度器已启动")
    
    def stop_cleanup_scheduler(self):
        """停止清理调度器"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        
        schedule.clear()
        logger.info("清理调度器已停止")
    
    def _run_scheduler(self):
        """运行调度器"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
            except Exception as e:
                logger.error(f"清理调度器运行异常: {e}")
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
            logger.info(f"过期会话清理完成，清理了 {cleaned_count} 个会话")
        except Exception as e:
            logger.error(f"清理过期会话失败: {e}")
    
    def _cleanup_old_files(self):
        """清理过期文件"""
        try:
            logger.info("开始清理过期文件...")
            cleaned_count = file_service.cleanup_old_files(max_age_hours=24)
            logger.info(f"过期文件清理完成，清理了 {cleaned_count} 个文件")
        except Exception as e:
            logger.error(f"清理过期文件失败: {e}")
    
    def _cleanup_old_tasks(self):
        """清理过期任务"""
        try:
            logger.info("开始清理过期任务...")
            cleaned_count = task_service.cleanup_old_tasks(max_age_hours=48)
            logger.info(f"过期任务清理完成，清理了 {cleaned_count} 个任务")
        except Exception as e:
            logger.error(f"清理过期任务失败: {e}")
    
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
            
            # 清理文件
            file_count = file_service.cleanup_old_files(max_age_hours=max_age_hours)
            
            # 清理任务
            task_count = task_service.cleanup_old_tasks(max_age_hours=max_age_hours)
            
            logger.info(f"手动清理完成 - 会话: {session_count}, 文件: {file_count}, 任务: {task_count}")
            
            return {
                "sessions_cleaned": session_count,
                "files_cleaned": file_count,
                "tasks_cleaned": task_count
            }
            
        except Exception as e:
            logger.error(f"手动清理失败: {e}")
            raise


# 全局清理管理器实例
cleanup_manager = CleanupManager()


def start_background_cleanup():
    """启动后台清理任务"""
    cleanup_manager.start_cleanup_scheduler()


def stop_background_cleanup():
    """停止后台清理任务"""
    cleanup_manager.stop_cleanup_scheduler()


async def manual_cleanup_endpoint(max_age_hours: int = 1):
    """手动清理端点（用于API调用）"""
    return cleanup_manager.manual_cleanup(max_age_hours)
