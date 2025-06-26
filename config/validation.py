"""
配置验证模块
Configuration Validation Module
"""

import os
import sys
from typing import List, Tuple, Optional
from loguru import logger

from config.settings import settings


class ConfigValidator:
    """配置验证器"""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_all(self) -> Tuple[bool, List[str], List[str]]:
        """
        验证所有配置
        
        Returns:
            Tuple[bool, List[str], List[str]]: (是否通过验证, 错误列表, 警告列表)
        """
        logger.info("开始配置验证...")
        
        # 验证LLM配置
        self._validate_llm_config()
        
        # 验证目录配置
        self._validate_directories()
        
        # 验证数值配置
        self._validate_numeric_config()
        
        # 输出验证结果
        self._log_validation_results()
        
        return len(self.errors) == 0, self.errors, self.warnings
    
    def _validate_llm_config(self) -> None:
        """验证LLM配置"""
        logger.debug("验证LLM配置...")
        
        # 检查LLM提供商
        if settings.llm_provider not in ['openai', 'dashscope']:
            self.errors.append(f"不支持的LLM提供商: {settings.llm_provider}")
            return
        
        # 检查对应的API密钥
        if settings.llm_provider == 'dashscope':
            if not settings.dashscope_api_key:
                self.errors.append("阿里云百炼API密钥未设置，请设置DASHSCOPE_API_KEY环境变量")
            else:
                logger.info("✓ 阿里云百炼API密钥已设置")
                
        elif settings.llm_provider == 'openai':
            if not settings.openai_api_key:
                self.errors.append("OpenAI API密钥未设置，请设置OPENAI_API_KEY环境变量")
            else:
                logger.info("✓ OpenAI API密钥已设置")
                
            # 检查可选的base_url
            if settings.openai_base_url:
                logger.info(f"✓ OpenAI Base URL已设置: {settings.openai_base_url}")
    
    def _validate_directories(self) -> None:
        """验证目录配置"""
        logger.debug("验证目录配置...")
        
        directories = [
            ("向量存储目录", settings.vector_store_path),
            ("输出目录", settings.output_dir),
            ("日志目录", os.path.dirname(settings.log_file))
        ]
        
        for name, path in directories:
            try:
                os.makedirs(path, exist_ok=True)
                if os.access(path, os.W_OK):
                    logger.info(f"✓ {name}可写: {path}")
                else:
                    self.errors.append(f"{name}不可写: {path}")
            except Exception as e:
                self.errors.append(f"创建{name}失败: {path}, 错误: {e}")
    
    def _validate_numeric_config(self) -> None:
        """验证数值配置"""
        logger.debug("验证数值配置...")
        
        # 检查分块配置
        if settings.chunk_size <= 0:
            self.errors.append(f"文本分块大小必须大于0: {settings.chunk_size}")
        elif settings.chunk_size < 100:
            self.warnings.append(f"文本分块大小可能过小: {settings.chunk_size}")
        elif settings.chunk_size > 2000:
            self.warnings.append(f"文本分块大小可能过大: {settings.chunk_size}")
        
        if settings.chunk_overlap < 0:
            self.errors.append(f"文本分块重叠不能为负数: {settings.chunk_overlap}")
        elif settings.chunk_overlap >= settings.chunk_size:
            self.errors.append(f"文本分块重叠不能大于等于分块大小: {settings.chunk_overlap} >= {settings.chunk_size}")
        
        # 检查检索配置
        if settings.retrieval_k <= 0:
            self.errors.append(f"检索文档数量必须大于0: {settings.retrieval_k}")
        elif settings.retrieval_k > 20:
            self.warnings.append(f"检索文档数量可能过大，影响性能: {settings.retrieval_k}")
        
        if not 0 <= settings.similarity_threshold <= 1:
            self.errors.append(f"相似度阈值必须在0-1之间: {settings.similarity_threshold}")
        
        # 检查重排序配置
        if settings.enable_reranking:
            if settings.rerank_top_k <= 0:
                self.errors.append(f"重排序候选文档数量必须大于0: {settings.rerank_top_k}")
            if settings.rerank_final_k <= 0:
                self.errors.append(f"重排序最终文档数量必须大于0: {settings.rerank_final_k}")
            if settings.rerank_final_k > settings.rerank_top_k:
                self.errors.append(f"重排序最终文档数量不能大于候选数量: {settings.rerank_final_k} > {settings.rerank_top_k}")
    
    def _log_validation_results(self) -> None:
        """输出验证结果"""
        logger.info("=" * 50)
        logger.info("配置验证结果")
        logger.info("=" * 50)
        
        if self.errors:
            logger.error(f"发现 {len(self.errors)} 个配置错误:")
            for i, error in enumerate(self.errors, 1):
                logger.error(f"  {i}. {error}")
        
        if self.warnings:
            logger.warning(f"发现 {len(self.warnings)} 个配置警告:")
            for i, warning in enumerate(self.warnings, 1):
                logger.warning(f"  {i}. {warning}")
        
        if not self.errors and not self.warnings:
            logger.info("✓ 所有配置验证通过")
        elif not self.errors:
            logger.info("✓ 配置验证通过（有警告）")
        else:
            logger.error("✗ 配置验证失败")
        
        logger.info("=" * 50)


def validate_config_on_startup() -> bool:
    """
    启动时验证配置
    
    Returns:
        bool: 验证是否通过
    """
    validator = ConfigValidator()
    is_valid, errors, warnings = validator.validate_all()
    
    if not is_valid:
        logger.error("配置验证失败，应用无法启动")
        logger.error("请检查以下配置错误:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    if warnings:
        logger.warning("配置验证通过，但有以下警告:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
    
    return True


def test_llm_connection() -> bool:
    """
    测试LLM连接
    
    Returns:
        bool: 连接是否成功
    """
    try:
        logger.info("测试LLM连接...")
        
        from src.utils.llm_factory import LLMFactory
        
        # 测试嵌入模型（更轻量级的测试）
        logger.info(f"测试{settings.llm_provider}嵌入模型连接...")
        embeddings = LLMFactory.create_embeddings()
        
        # 简单的嵌入测试
        test_result = embeddings.embed_query("测试连接")
        if test_result and len(test_result) > 0:
            logger.info("✓ LLM连接测试成功")
            return True
        else:
            logger.error("✗ LLM连接测试失败：返回空结果")
            return False
            
    except Exception as e:
        logger.error(f"✗ LLM连接测试失败: {e}")
        
        # 提供具体的错误提示
        if "Access denied" in str(e) or "Arrearage" in str(e):
            logger.error("API访问被拒绝，请检查:")
            logger.error("  1. API密钥是否正确")
            logger.error("  2. 账户余额是否充足")
            logger.error("  3. API密钥是否有相应权限")
        elif "api_key" in str(e).lower():
            logger.error("API密钥相关错误，请检查环境变量设置")
        
        return False


def print_config_summary() -> None:
    """打印配置摘要"""
    logger.info("当前配置摘要:")
    logger.info(f"  LLM提供商: {settings.llm_provider}")
    logger.info(f"  向量存储路径: {settings.vector_store_path}")
    logger.info(f"  输出目录: {settings.output_dir}")
    logger.info(f"  分块大小: {settings.chunk_size}")
    logger.info(f"  检索文档数: {settings.retrieval_k}")
    logger.info(f"  重排序启用: {settings.enable_reranking}")
    logger.info(f"  向量库隔离: {settings.clear_vector_store_on_new_document}")


if __name__ == "__main__":
    """独立运行配置验证"""
    # 设置日志
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    
    print_config_summary()
    
    # 验证配置
    if not validate_config_on_startup():
        sys.exit(1)
    
    # 测试LLM连接
    if not test_llm_connection():
        sys.exit(1)
    
    logger.info("所有验证通过，配置正常")
