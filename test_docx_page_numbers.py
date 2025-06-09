#!/usr/bin/env python3
"""
测试DOCX文件页码处理功能
Test script for DOCX page number processing improvements
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.document_loader import DocumentLoader
from loguru import logger

def test_docx_page_numbers():
    """测试DOCX文件页码处理功能"""
    
    # 初始化文档加载器
    loader = DocumentLoader()
    
    # 测试文件路径
    test_file = "test_documents/test_bidding.docx"
    
    if not os.path.exists(test_file):
        logger.error(f"测试文件不存在: {test_file}")
        return
    
    logger.info(f"开始测试DOCX文件页码处理: {test_file}")
    
    try:
        # 加载DOCX文件
        full_text, content_parts = loader.load_docx(test_file)
        
        logger.info(f"成功加载文档，共{len(content_parts)}个内容块")
        
        # 分析页码分布
        page_numbers = [page_num for _, page_num in content_parts]
        unique_pages = set(page_numbers)
        
        logger.info(f"文档包含页码: {sorted(unique_pages)}")
        logger.info(f"总页数: {len(unique_pages)}")
        
        # 显示前几个内容块的页码信息
        logger.info("前10个内容块的页码信息:")
        for i, (content, page_num) in enumerate(content_parts[:10]):
            content_preview = content[:50].replace('\n', ' ')
            logger.info(f"  块{i+1}: 页码{page_num} - {content_preview}...")
        
        # 测试文本分割和元数据提取
        logger.info("\n测试文本分割和元数据提取:")
        documents = loader.split_text(full_text, {"source": test_file})
        
        logger.info(f"分割后共{len(documents)}个文档块")
        
        # 显示前几个文档块的元数据
        for i, doc in enumerate(documents[:5]):
            metadata = doc.metadata
            page_info = metadata.get('page_number', '未知')
            location_type = metadata.get('location_type', '未知')
            content_preview = doc.page_content[:50].replace('\n', ' ')
            
            logger.info(f"  文档块{i+1}: 页码{page_info} (类型:{location_type}) - {content_preview}...")
        
        logger.success("DOCX页码处理测试完成")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def test_fallback_method():
    """测试回退方法"""
    logger.info("\n测试回退方法:")
    
    loader = DocumentLoader()
    test_file = "test_documents/test_bidding.docx"
    
    if not os.path.exists(test_file):
        logger.error(f"测试文件不存在: {test_file}")
        return
    
    try:
        # 直接调用回退方法
        full_text, content_parts = loader._load_docx_fallback(test_file)
        
        logger.info(f"回退方法加载成功，共{len(content_parts)}个内容块")
        
        # 分析页码分布
        page_numbers = [page_num for _, page_num in content_parts]
        unique_pages = set(page_numbers)
        
        logger.info(f"回退方法页码分布: {sorted(unique_pages)}")
        
    except Exception as e:
        logger.error(f"回退方法测试失败: {e}")

if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    
    test_docx_page_numbers()
    test_fallback_method()
