#!/usr/bin/env python3
"""
测试会话隔离修复
Test session isolation fixes
"""

import os
import sys
import time
import threading
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_vector_store_isolation():
    """测试向量存储隔离"""
    print("=== 测试向量存储隔离 ===")
    
    try:
        from src.utils.vector_store import VectorStoreManager
        from src.utils.llm_factory import LLMFactory
        from langchain_core.documents import Document
        
        # 创建测试文档
        test_docs = [
            Document(page_content="测试文档1", metadata={"source": "test1"}),
            Document(page_content="测试文档2", metadata={"source": "test2"})
        ]
        
        embeddings = LLMFactory.create_embeddings()
        
        # 测试两个不同会话
        session1_id = "test_session_001"
        session2_id = "test_session_002"
        
        print(f"创建会话1向量存储管理器: {session1_id}")
        manager1 = VectorStoreManager(embeddings, session_id=session1_id)
        print(f"会话1向量存储路径: {manager1.persist_directory}")
        
        print(f"创建会话2向量存储管理器: {session2_id}")
        manager2 = VectorStoreManager(embeddings, session_id=session2_id)
        print(f"会话2向量存储路径: {manager2.persist_directory}")
        
        # 验证路径不同
        if manager1.persist_directory != manager2.persist_directory:
            print("✅ 向量存储路径隔离成功")
        else:
            print("❌ 向量存储路径隔离失败")
        
        # 测试清理功能
        print("\n测试向量存储清理...")
        try:
            manager1.clear_vector_store()
            print("✅ 会话1向量存储清理成功")
        except Exception as e:
            print(f"⚠️ 会话1向量存储清理失败: {e}")
        
        try:
            manager2.clear_vector_store()
            print("✅ 会话2向量存储清理成功")
        except Exception as e:
            print(f"⚠️ 会话2向量存储清理失败: {e}")
            
    except Exception as e:
        print(f"❌ 向量存储隔离测试失败: {e}")


def test_concurrent_sessions():
    """测试并发会话"""
    print("\n=== 测试并发会话处理 ===")
    
    def simulate_session(session_id: str):
        """模拟会话处理"""
        try:
            print(f"会话 {session_id}: 开始处理")
            
            # 模拟创建会话目录
            session_upload_dir = f"./uploads/{session_id}"
            session_vector_dir = f"./vector_store/{session_id}"
            
            os.makedirs(session_upload_dir, exist_ok=True)
            os.makedirs(session_vector_dir, exist_ok=True)
            
            print(f"会话 {session_id}: 目录创建成功")
            
            # 模拟一些处理时间
            time.sleep(1)
            
            print(f"会话 {session_id}: 处理完成")
            
        except Exception as e:
            print(f"会话 {session_id}: 处理失败 - {e}")
    
    # 创建多个并发会话
    sessions = ["concurrent_001", "concurrent_002", "concurrent_003"]
    threads = []
    
    for session_id in sessions:
        thread = threading.Thread(target=simulate_session, args=(session_id,))
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    print("✅ 并发会话测试完成")


def test_directory_structure():
    """测试目录结构"""
    print("\n=== 检查目录结构 ===")
    
    base_dirs = ["./uploads", "./vector_store", "./temp"]
    
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            print(f"✅ 基础目录存在: {base_dir}")
            
            # 检查会话子目录
            try:
                subdirs = [d for d in os.listdir(base_dir) 
                          if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("session_")]
                if subdirs:
                    print(f"  📁 会话子目录: {len(subdirs)} 个")
                    for subdir in subdirs[:3]:  # 只显示前3个
                        print(f"    - {subdir}")
                    if len(subdirs) > 3:
                        print(f"    - ... 还有 {len(subdirs) - 3} 个")
                else:
                    print(f"  📂 暂无会话子目录")
            except Exception as e:
                print(f"  ❌ 检查子目录失败: {e}")
        else:
            print(f"❌ 基础目录不存在: {base_dir}")


def cleanup_test_data():
    """清理测试数据"""
    print("\n=== 清理测试数据 ===")
    
    test_patterns = ["test_session_", "concurrent_"]
    base_dirs = ["./uploads", "./vector_store", "./temp"]
    
    cleaned_count = 0
    
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            try:
                for item in os.listdir(base_dir):
                    item_path = os.path.join(base_dir, item)
                    if os.path.isdir(item_path):
                        # 检查是否是测试目录
                        for pattern in test_patterns:
                            if item.startswith(pattern):
                                try:
                                    import shutil
                                    shutil.rmtree(item_path)
                                    print(f"清理测试目录: {item_path}")
                                    cleaned_count += 1
                                except Exception as e:
                                    print(f"清理失败: {item_path} - {e}")
                                break
            except Exception as e:
                print(f"清理 {base_dir} 失败: {e}")
    
    print(f"✅ 清理完成，共清理 {cleaned_count} 个测试目录")


def main():
    """主测试函数"""
    print("🚀 开始会话隔离修复测试")
    print("=" * 50)
    
    test_directory_structure()
    test_vector_store_isolation()
    test_concurrent_sessions()
    
    print("\n" + "=" * 50)
    print("🎉 会话隔离修复测试完成")
    
    # 询问是否清理测试数据
    try:
        response = input("\n是否清理测试数据？(y/N): ").strip().lower()
        if response in ['y', 'yes']:
            cleanup_test_data()
    except KeyboardInterrupt:
        print("\n测试结束")


if __name__ == "__main__":
    main()
