"""
启动 FastAPI 服务器
Start FastAPI Server
"""

import sys
import os
from pathlib import Path
import uvicorn
from loguru import logger

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """启动API服务"""
    try:
        # 设置统一日志格式
        from config.logging_config import setup_logging, get_uvicorn_log_config
        setup_logging("INFO")

        logger.info("正在启动智能投标助手 API 服务...")

        # 检查环境
        try:
            from config.settings import settings
            logger.info(f"配置加载成功，LLM提供商: {settings.llm_provider}")
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            return

        # 创建必要的目录
        os.makedirs("./uploads", exist_ok=True)
        os.makedirs("./temp", exist_ok=True)
        os.makedirs("./logs", exist_ok=True)
        
        # 启动服务器
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True,
            log_config=get_uvicorn_log_config()  # 使用我们的日志配置
        )
        
    except KeyboardInterrupt:
        logger.info("服务器已停止")
    except Exception as e:
        logger.error(f"启动服务器失败: {e}")

if __name__ == "__main__":
    main()
