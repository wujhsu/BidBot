#!/usr/bin/env python3
"""
清理空目录工具
Tool for cleaning up empty directories in temp, uploads, and vector_store
"""

import os
import sys
from pathlib import Path
from loguru import logger

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.middleware.session import cleanup_expired_sessions, _cleanup_empty_directories


def cleanup_empty_dirs_only(base_dirs: list = None, dry_run: bool = False):
    """
    仅清理空目录（不删除有效会话）
    
    Args:
        base_dirs: 要清理的基础目录列表，默认为 ["./temp", "./uploads", "./vector_store"]
        dry_run: 是否为试运行模式（只显示会删除什么，不实际删除）
    
    Returns:
        dict: 清理结果统计
    """
    if base_dirs is None:
        base_dirs = ["./temp", "./uploads", "./vector_store"]
    
    results = {}
    total_cleaned = 0
    
    logger.info(f"开始清理空目录{'（试运行模式）' if dry_run else ''}...")
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            logger.info(f"目录不存在，跳过: {base_dir}")
            results[base_dir] = 0
            continue
        
        logger.info(f"检查目录: {base_dir}")
        
        if dry_run:
            # 试运行模式：只统计不删除
            empty_dirs = []
            for root, dirs, files in os.walk(base_dir, topdown=False):
                if root == base_dir:
                    continue
                try:
                    if not os.listdir(root):
                        empty_dirs.append(root)
                except OSError:
                    continue
            
            results[base_dir] = len(empty_dirs)
            total_cleaned += len(empty_dirs)
            
            if empty_dirs:
                logger.info(f"发现 {len(empty_dirs)} 个空目录:")
                for empty_dir in empty_dirs:
                    relative_path = os.path.relpath(empty_dir, base_dir)
                    logger.info(f"  - {relative_path}")
            else:
                logger.info(f"  没有发现空目录")
        else:
            # 实际清理模式
            dir_type = os.path.basename(base_dir)
            cleaned_count = _cleanup_empty_directories(base_dir, dir_type)
            results[base_dir] = cleaned_count
            total_cleaned += cleaned_count
    
    if dry_run:
        logger.info(f"试运行完成，共发现 {total_cleaned} 个空目录")
        if total_cleaned > 0:
            logger.info("运行 'python cleanup_empty_dirs.py --clean' 来实际清理这些空目录")
    else:
        logger.info(f"清理完成，共清理了 {total_cleaned} 个空目录")
    
    return results


def cleanup_all_expired(max_age_hours: int = 24, include_empty_dirs: bool = True):
    """
    清理过期会话和空目录
    
    Args:
        max_age_hours: 会话最大保留时间（小时）
        include_empty_dirs: 是否同时清理空目录
    
    Returns:
        int: 清理的会话数量
    """
    logger.info(f"开始清理过期会话（保留时间: {max_age_hours}小时）...")
    
    cleaned_count = cleanup_expired_sessions(
        base_upload_dir="./uploads",
        base_vector_dir="./vector_store",
        base_temp_dir="./temp",
        max_age_hours=max_age_hours,
        cleanup_empty_dirs=include_empty_dirs
    )
    
    return cleaned_count


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="清理空目录和过期会话工具")
    parser.add_argument("--dry-run", action="store_true", help="试运行模式，只显示会删除什么")
    parser.add_argument("--clean", action="store_true", help="清理空目录")
    parser.add_argument("--clean-expired", type=int, metavar="HOURS", 
                       help="清理指定小时数之前的过期会话（默认24小时）")
    parser.add_argument("--dirs", nargs="+", default=["./temp", "./uploads", "./vector_store"],
                       help="要清理的目录列表")
    
    args = parser.parse_args()
    
    # 配置日志
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    
    try:
        if args.clean_expired is not None:
            # 清理过期会话
            max_age = args.clean_expired if args.clean_expired > 0 else 24
            cleaned_count = cleanup_all_expired(max_age_hours=max_age, include_empty_dirs=True)
            logger.info(f"清理完成，共清理了 {cleaned_count} 个过期会话")
            
        elif args.clean:
            # 清理空目录
            results = cleanup_empty_dirs_only(base_dirs=args.dirs, dry_run=False)
            total = sum(results.values())
            if total > 0:
                logger.success(f"清理完成！共清理了 {total} 个空目录")
            else:
                logger.info("没有发现空目录")
                
        else:
            # 默认为试运行模式
            results = cleanup_empty_dirs_only(base_dirs=args.dirs, dry_run=True)
            total = sum(results.values())
            if total > 0:
                logger.warning(f"发现 {total} 个空目录，使用 --clean 参数来清理它们")
            else:
                logger.success("没有发现空目录")
    
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        sys.exit(1)
    except Exception as e:
        logger.error(f"清理过程中出现错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
