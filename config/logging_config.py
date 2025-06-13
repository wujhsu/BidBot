"""
日志配置模块
Logging Configuration Module
"""

import sys
import logging
from loguru import logger


class InterceptHandler(logging.Handler):
    """
    将标准logging重定向到loguru
    Intercept standard logging and redirect to loguru
    """
    
    def emit(self, record):
        # 获取对应的loguru级别
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 查找调用者
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(log_level: str = "INFO"):
    """
    设置统一的日志格式
    Setup unified logging format
    
    Args:
        log_level: 日志级别
    """
    # 移除默认的loguru处理器
    logger.remove()
    
    # 添加自定义格式的loguru处理器
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level.upper(),
        colorize=True
    )
    
    # 添加文件日志（可选）
    logger.add(
        "./logs/bidding_assistant.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level.upper(),
        rotation="10 MB",
        retention="7 days",
        compression="zip"
    )
    
    # 拦截标准logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # 设置特定logger使用InterceptHandler
    loggers_to_intercept = [
        "uvicorn",
        "uvicorn.access", 
        "uvicorn.error",
        "fastapi",
        "httpx",
        "httpcore"
    ]
    
    for logger_name in loggers_to_intercept:
        logging.getLogger(logger_name).handlers = [InterceptHandler()]
        logging.getLogger(logger_name).propagate = False


def get_uvicorn_log_config():
    """
    获取uvicorn的日志配置
    Get uvicorn log configuration
    
    Returns:
        None: 禁用uvicorn默认日志配置
    """
    return None
