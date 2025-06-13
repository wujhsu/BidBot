"""
智能投标助手 FastAPI 主应用
Intelligent Bidding Assistant FastAPI Main Application
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入配置和路由
from config.settings import settings
from api.routers import upload, analysis, files
from api.models.api_models import HealthCheckResponse, ErrorResponse

# 创建FastAPI应用
app = FastAPI(
    title="智能投标助手 API",
    description="基于AI的招投标文件分析系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(upload.router)
app.include_router(analysis.router)
app.include_router(files.router)


@app.get("/", response_model=HealthCheckResponse)
async def root():
    """
    根路径 - 健康检查
    
    Returns:
        HealthCheckResponse: 服务状态信息
    """
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        dependencies={
            "langchain": "ok",
            "document_processor": "ok",
            "vector_store": "ok"
        }
    )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    健康检查接口
    
    Returns:
        HealthCheckResponse: 详细的服务状态信息
    """
    try:
        # 检查核心依赖
        dependencies = {}
        
        # 检查LLM连接
        try:
            from src.utils.llm_factory import LLMFactory
            llm_factory = LLMFactory()
            dependencies["llm"] = "ok"
        except Exception as e:
            dependencies["llm"] = f"error: {str(e)}"
        
        # 检查文档处理
        try:
            from src.utils.document_loader import DocumentLoader
            dependencies["document_loader"] = "ok"
        except Exception as e:
            dependencies["document_loader"] = f"error: {str(e)}"
        
        # 检查向量存储
        try:
            from src.utils.vector_store import VectorStoreManager
            dependencies["vector_store"] = "ok"
        except Exception as e:
            dependencies["vector_store"] = f"error: {str(e)}"
        
        # 检查分析图
        try:
            from src.graph.bidding_graph import BiddingAnalysisGraph
            dependencies["analysis_graph"] = "ok"
        except Exception as e:
            dependencies["analysis_graph"] = f"error: {str(e)}"
        
        return HealthCheckResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="1.0.0",
            dependencies=dependencies
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            version="1.0.0",
            dependencies={"error": str(e)}
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """
    HTTP异常处理器
    
    Args:
        request: 请求对象
        exc: HTTP异常
        
    Returns:
        JSON错误响应
    """
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_code=f"HTTP_{exc.status_code}",
            error_message=exc.detail,
            timestamp=datetime.now()
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """
    通用异常处理器
    
    Args:
        request: 请求对象
        exc: 异常
        
    Returns:
        JSON错误响应
    """
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="INTERNAL_SERVER_ERROR",
            error_message="服务器内部错误",
            details={"exception": str(exc)},
            timestamp=datetime.now()
        ).model_dump()
    )


# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("智能投标助手 API 服务启动")
    logger.info(f"API文档地址: http://localhost:8000/docs")
    
    # 创建必要的目录
    os.makedirs("./uploads", exist_ok=True)
    os.makedirs("./temp", exist_ok=True)
    
    # 启动定期清理任务（可选）
    # asyncio.create_task(periodic_cleanup())


# 关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("智能投标助手 API 服务关闭")


# 定期清理任务（可选）
async def periodic_cleanup():
    """定期清理过期文件和任务"""
    import asyncio
    from api.services.file_service import file_service
    from api.services.task_service import task_service
    
    while True:
        try:
            # 每小时清理一次
            await asyncio.sleep(3600)
            
            # 清理过期文件（24小时）
            cleaned_files = file_service.cleanup_old_files(24)
            if cleaned_files > 0:
                logger.info(f"清理了 {cleaned_files} 个过期文件")
            
            # 清理过期任务（24小时）
            cleaned_tasks = task_service.cleanup_old_tasks(24)
            if cleaned_tasks > 0:
                logger.info(f"清理了 {cleaned_tasks} 个过期任务")
                
        except Exception as e:
            logger.error(f"定期清理任务异常: {e}")


if __name__ == "__main__":
    import uvicorn
    
    # 开发模式运行
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
