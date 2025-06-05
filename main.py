"""
智能投标助手主程序
Main entry point for the Intelligent Bidding Assistant
"""

import os
import sys
import argparse
from pathlib import Path
from loguru import logger

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from src.graph.bidding_graph import bidding_graph
from src.utils.llm_factory import LLMFactory

def setup_logging():
    """设置日志配置"""
    # 移除默认的日志处理器
    logger.remove()
    
    # 添加控制台输出
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # 添加文件输出
    logger.add(
        settings.log_file,
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8"
    )

def test_llm_connection():
    """测试LLM连接"""
    logger.info("测试LLM连接...")
    
    # 测试文本生成模型
    text_success = LLMFactory.test_connection(settings.llm_provider)
    if not text_success:
        logger.error(f"文本生成模型连接失败: {settings.llm_provider}")
        return False
    
    # 测试嵌入模型
    embedding_success = LLMFactory.test_embeddings(settings.llm_provider)
    if not embedding_success:
        logger.error(f"嵌入模型连接失败: {settings.llm_provider}")
        return False
    
    logger.info("LLM连接测试成功")
    return True

def analyze_document(document_path: str) -> bool:
    """
    分析单个文档
    
    Args:
        document_path: 文档路径
        
    Returns:
        bool: 分析是否成功
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(document_path):
            logger.error(f"文件不存在: {document_path}")
            return False
        
        # 检查文件格式
        file_extension = Path(document_path).suffix.lower()
        if file_extension not in ['.pdf', '.docx', '.doc', '.txt']:
            logger.error(f"不支持的文件格式: {file_extension}")
            return False
        
        logger.info(f"开始分析文档: {document_path}")
        
        # 运行分析流程
        result = bidding_graph.run(document_path)
        
        # 检查结果
        final_step = result.get("current_step", "unknown")
        error_messages = result.get("error_messages", [])
        
        if final_step == "completed":
            logger.info("文档分析成功完成")
            return True
        elif final_step == "failed":
            logger.error("文档分析失败")
            for error in error_messages:
                logger.error(f"错误: {error}")
            return False
        else:
            logger.warning(f"文档分析状态未知: {final_step}")
            if error_messages:
                for error in error_messages:
                    logger.warning(f"警告: {error}")
            return True  # 可能有部分结果
            
    except Exception as e:
        logger.error(f"分析文档时发生异常: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="智能投标助手 - 招投标文件分析工具")
    parser.add_argument("document", help="要分析的招投标文件路径")
    parser.add_argument("--provider", choices=["openai", "dashscope"], 
                       help="LLM提供商选择 (默认使用配置文件中的设置)")
    parser.add_argument("--test-connection", action="store_true", 
                       help="仅测试LLM连接，不进行文档分析")
    parser.add_argument("--show-graph", action="store_true", 
                       help="显示工作流图结构")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="详细输出模式")
    parser.add_argument("--no-isolation", action="store_true",
                       help="禁用向量库隔离模式（可能导致历史数据交叉污染）")
    parser.add_argument("--clear-vector-store", action="store_true",
                       help="手动清空向量库历史数据")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        settings.log_level = "DEBUG"
    
    # 设置日志
    setup_logging()
    
    # 设置LLM提供商
    if args.provider:
        settings.llm_provider = args.provider
        logger.info(f"使用LLM提供商: {args.provider}")
    else:
        logger.info(f"使用默认LLM提供商: {settings.llm_provider}")

    # 设置向量库隔离模式
    if args.no_isolation:
        settings.clear_vector_store_on_new_document = False
        logger.warning("已禁用向量库隔离模式，可能存在历史数据交叉污染风险")
    else:
        logger.info(f"向量库隔离模式: {'开启' if settings.clear_vector_store_on_new_document else '关闭'}")

    # 手动清空向量库
    if args.clear_vector_store:
        logger.info("手动清空向量库...")
        try:
            from src.utils.vector_store import VectorStoreManager
            from src.utils.llm_factory import LLMFactory
            embeddings = LLMFactory.create_embeddings()
            vector_manager = VectorStoreManager(embeddings)
            vector_manager.clear_vector_store()
            logger.info("向量库已清空")
            return
        except Exception as e:
            logger.error(f"清空向量库失败: {e}")
            sys.exit(1)
    
    # 显示配置信息
    logger.info("智能投标助手启动")
    logger.info(f"配置信息:")
    logger.info(f"  - LLM提供商: {settings.llm_provider}")
    logger.info(f"  - 输出目录: {settings.output_dir}")
    logger.info(f"  - 向量存储路径: {settings.vector_store_path}")
    logger.info(f"  - 文本分块大小: {settings.chunk_size}")
    logger.info(f"  - 向量库隔离模式: {'开启' if settings.clear_vector_store_on_new_document else '关闭'}")
    
    # 显示工作流图
    if args.show_graph:
        print("\n工作流图结构:")
        print(bidding_graph.get_graph_visualization())
        return
    
    # 测试连接
    if args.test_connection:
        success = test_llm_connection()
        sys.exit(0 if success else 1)
    
    # 测试LLM连接
    if not test_llm_connection():
        logger.error("LLM连接测试失败，请检查配置")
        sys.exit(1)
    
    # 分析文档
    success = analyze_document(args.document)
    
    if success:
        logger.info("分析完成！")
        logger.info(f"报告已保存到: {settings.output_dir}")
        sys.exit(0)
    else:
        logger.error("分析失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()
