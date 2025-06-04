"""
项目设置测试脚本
Test script to verify project setup
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试模块导入"""
    print("测试模块导入...")
    
    try:
        # 测试配置导入
        from config.settings import settings
        print("✓ 配置模块导入成功")
        
        # 测试数据模型导入
        from src.models.data_models import GraphState, BiddingAnalysisResult
        print("✓ 数据模型导入成功")
        
        # 测试工具模块导入
        from src.utils.document_loader import DocumentLoader
        from src.utils.llm_factory import LLMFactory
        from src.utils.vector_store import VectorStoreManager
        print("✓ 工具模块导入成功")
        
        # 测试智能体模块导入
        from src.agents.document_processor import create_document_processor_node
        from src.agents.basic_info_extractor import create_basic_info_extractor_node
        from src.agents.scoring_analyzer import create_scoring_analyzer_node
        from src.agents.other_info_extractor import create_other_info_extractor_node
        from src.agents.output_formatter import create_output_formatter_node
        print("✓ 智能体模块导入成功")
        
        # 测试图模块导入
        from src.graph.bidding_graph import bidding_graph
        print("✓ 图模块导入成功")
        
        return True
        
    except ImportError as e:
        print(f"✗ 模块导入失败: {e}")
        return False

def test_dependencies():
    """测试依赖包"""
    print("\n测试依赖包...")
    
    required_packages = [
        'langchain',
        'langgraph',
        'langchain_community',
        'PyPDF2',
        'docx',
        'chromadb',
        'openai',
        'dashscope',
        'pandas',
        'numpy',
        'pydantic',
        'loguru',
        'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'docx':
                import docx
            elif package == 'PyPDF2':
                import PyPDF2
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def test_configuration():
    """测试配置"""
    print("\n测试配置...")
    
    try:
        from config.settings import settings
        
        print(f"✓ LLM提供商: {settings.llm_provider}")
        print(f"✓ 输出目录: {settings.output_dir}")
        print(f"✓ 向量存储路径: {settings.vector_store_path}")
        print(f"✓ 文本分块大小: {settings.chunk_size}")
        print(f"✓ 日志级别: {settings.log_level}")
        
        # 检查目录是否创建
        if os.path.exists(settings.output_dir):
            print(f"✓ 输出目录已创建: {settings.output_dir}")
        else:
            print(f"✗ 输出目录未创建: {settings.output_dir}")
        
        # 检查API密钥配置
        if settings.llm_provider == "openai":
            if settings.openai_api_key:
                print("✓ OpenAI API密钥已配置")
            else:
                print("⚠ OpenAI API密钥未配置")
        elif settings.llm_provider == "dashscope":
            if settings.dashscope_api_key:
                print("✓ 阿里云百炼API密钥已配置")
            else:
                print("⚠ 阿里云百炼API密钥未配置")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置测试失败: {e}")
        return False

def test_graph_creation():
    """测试图创建"""
    print("\n测试Langgraph图创建...")
    
    try:
        from src.graph.bidding_graph import bidding_graph
        
        # 检查图是否正确创建
        if hasattr(bidding_graph, 'graph'):
            print("✓ Langgraph图创建成功")
            
            # 显示图的可视化
            print("\n工作流图结构:")
            print(bidding_graph.get_graph_visualization())
            
            return True
        else:
            print("✗ Langgraph图创建失败")
            return False
            
    except Exception as e:
        print(f"✗ 图创建测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("智能投标助手项目设置测试")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_dependencies,
        test_configuration,
        test_graph_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！项目设置正确。")
        print("\n下一步:")
        print("1. 配置API密钥 (复制.env.example为.env并填入密钥)")
        print("2. 运行: python main.py your_document.pdf --test-connection")
        print("3. 开始分析招投标文件!")
    else:
        print("❌ 部分测试失败，请检查上述错误信息。")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
