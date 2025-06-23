"""
统一文档加载器
Unified document loader that converts all formats to PDF for consistent processing
"""

import os
import tempfile
import subprocess
from typing import List, Optional, Tuple
from pathlib import Path
import PyPDF2
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
from loguru import logger

# 尝试导入pymupdf，如果失败则设置为None
try:
    import fitz  # pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    PYMUPDF_AVAILABLE = False
    logger.warning("pymupdf未安装，将无法使用高级PDF处理功能")


class UnifiedDocumentConverter:
    """统一文档转换器，将所有格式转换为PDF"""

    def __init__(self):
        """初始化转换器"""
        self.temp_files = []  # 跟踪临时文件以便清理

    def convert_to_pdf(self, file_path: str) -> str:
        """
        将文档转换为PDF（支持PDF、DOCX、DOC格式）

        Args:
            file_path: 原始文件路径

        Returns:
            str: 转换后的PDF文件路径

        Raises:
            ValueError: 不支持的文件格式
            Exception: 转换失败
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_extension = Path(file_path).suffix.lower()

        if file_extension == '.pdf':
            # 已经是PDF，直接返回
            logger.info(f"文件已是PDF格式: {file_path}")
            return file_path
        elif file_extension in ['.docx', '.doc']:
            return self._convert_docx_to_pdf(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_extension}，仅支持 PDF、DOCX、DOC 格式")

    def _convert_docx_to_pdf(self, docx_path: str) -> str:
        """
        将DOCX文件转换为PDF

        Args:
            docx_path: DOCX文件路径

        Returns:
            str: 转换后的PDF文件路径
        """
        try:
            # 创建临时PDF文件
            temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            temp_pdf_path = temp_pdf.name
            temp_pdf.close()
            self.temp_files.append(temp_pdf_path)

            # 尝试使用LibreOffice转换（推荐方法）
            if self._try_libreoffice_conversion(docx_path, temp_pdf_path):
                logger.info(f"使用LibreOffice成功转换DOCX为PDF: {temp_pdf_path}")
                return temp_pdf_path

            # 尝试使用python-docx2pdf
            if self._try_docx2pdf_conversion(docx_path, temp_pdf_path):
                logger.info(f"使用docx2pdf成功转换DOCX为PDF: {temp_pdf_path}")
                return temp_pdf_path

            # 如果都失败，抛出异常
            raise Exception("所有DOCX转PDF方法都失败了")

        except Exception as e:
            logger.error(f"DOCX转PDF失败: {e}")
            # 清理失败的临时文件
            if temp_pdf_path in self.temp_files:
                self.temp_files.remove(temp_pdf_path)
            if os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)
            raise

    def _try_libreoffice_conversion(self, docx_path: str, output_path: str) -> bool:
        """尝试使用LibreOffice转换"""
        try:
            # 在Docker环境中，LibreOffice需要特殊的环境设置
            env = os.environ.copy()
            env['HOME'] = '/tmp'  # 设置临时HOME目录

            result = subprocess.run([
                'libreoffice', '--headless', '--convert-to', 'pdf',
                '--outdir', os.path.dirname(output_path),
                docx_path
            ], capture_output=True, text=True, timeout=60, env=env)

            if result.returncode == 0:
                # LibreOffice会生成与原文件同名的PDF
                docx_name = Path(docx_path).stem
                generated_pdf = os.path.join(os.path.dirname(output_path), f"{docx_name}.pdf")

                if os.path.exists(generated_pdf):
                    # 重命名为我们的临时文件名
                    os.rename(generated_pdf, output_path)
                    logger.info(f"LibreOffice转换成功: {docx_path} -> {output_path}")
                    return True
                else:
                    logger.warning(f"LibreOffice转换完成但未找到输出文件: {generated_pdf}")

            else:
                logger.warning(f"LibreOffice转换失败，返回码: {result.returncode}")
                if result.stderr:
                    logger.warning(f"LibreOffice错误输出: {result.stderr}")

            return False

        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            logger.warning(f"LibreOffice转换失败: {e}")
            return False

    def _try_docx2pdf_conversion(self, docx_path: str, output_path: str) -> bool:
        """尝试使用docx2pdf转换"""
        try:
            from docx2pdf import convert
            convert(docx_path, output_path)
            return os.path.exists(output_path) and os.path.getsize(output_path) > 0
        except ImportError:
            logger.warning("docx2pdf未安装，无法使用此方法转换")
            return False
        except Exception as e:
            logger.warning(f"docx2pdf转换失败: {e}")
            return False



    def cleanup(self):
        """清理临时文件"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.debug(f"清理临时文件: {temp_file}")
            except Exception as e:
                logger.warning(f"清理临时文件失败 {temp_file}: {e}")
        self.temp_files.clear()

    def __del__(self):
        """析构函数，确保清理临时文件"""
        self.cleanup()

class DocumentLoader:
    """统一文档加载器类 - 将所有格式转换为PDF后统一处理"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        初始化文档加载器

        Args:
            chunk_size: 文本分块大小
            chunk_overlap: 文本分块重叠大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        self.converter = UnifiedDocumentConverter()
    
    def load_pdf(self, file_path: str) -> Tuple[str, List[Tuple[str, int]]]:
        """
        加载PDF文件，优先使用pymupdf（最强大的PDF处理库）

        Args:
            file_path: PDF文件路径

        Returns:
            Tuple[str, List[Tuple[str, int]]]: (全文内容, [(页面内容, 页码)])
        """
        # 优先使用pymupdf
        if PYMUPDF_AVAILABLE:
            try:
                logger.info(f"使用pymupdf加载PDF: {file_path}")
                full_text, pages_content = self._load_pdf_with_pymupdf(file_path)
                if full_text.strip():
                    logger.info(f"使用pymupdf成功加载PDF文件: {file_path}, 共{len(pages_content)}页")
                    return full_text, pages_content
            except Exception as e:
                logger.warning(f"pymupdf加载失败，尝试回退方法: {e}")

        # 回退到其他方法
        methods = [
            ("pypdf", self._load_pdf_with_pypdf),
            ("PyPDF2", self._load_pdf_with_pypdf2),
        ]

        last_error = None
        for method_name, method_func in methods:
            try:
                logger.info(f"尝试使用{method_name}加载PDF: {file_path}")
                full_text, pages_content = method_func(file_path)

                if full_text.strip():
                    logger.info(f"使用{method_name}成功加载PDF文件: {file_path}, 共{len(pages_content)}页")
                    return full_text, pages_content
                else:
                    logger.warning(f"{method_name}提取的内容为空，尝试下一种方法")

            except Exception as e:
                logger.warning(f"{method_name}加载失败: {e}")
                last_error = e
                continue

        # 所有方法都失败了
        error_msg = f"所有PDF加载方法都失败了，最后一个错误: {last_error}"
        logger.error(error_msg)
        raise Exception(error_msg)

    def _load_pdf_with_pypdf(self, file_path: str) -> Tuple[str, List[Tuple[str, int]]]:
        """使用pypdf库加载PDF"""
        full_text = ""
        pages_content = []

        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)

            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        full_text += f"\n\n--- 第{page_num}页 ---\n\n{page_text}"
                        pages_content.append((page_text, page_num))
                    else:
                        logger.warning(f"第{page_num}页无法提取文本，可能是扫描件")
                        pages_content.append(("", page_num))
                except Exception as e:
                    logger.error(f"处理第{page_num}页时出错: {e}")
                    pages_content.append(("", page_num))

        return full_text, pages_content

    def _load_pdf_with_pypdf2(self, file_path: str) -> Tuple[str, List[Tuple[str, int]]]:
        """使用PyPDF2库加载PDF"""
        full_text = ""
        pages_content = []

        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        full_text += f"\n\n--- 第{page_num}页 ---\n\n{page_text}"
                        pages_content.append((page_text, page_num))
                    else:
                        logger.warning(f"第{page_num}页无法提取文本，可能是扫描件")
                        pages_content.append(("", page_num))
                except Exception as e:
                    logger.error(f"处理第{page_num}页时出错: {e}")
                    pages_content.append(("", page_num))

        return full_text, pages_content



    def _load_pdf_with_pymupdf(self, file_path: str) -> Tuple[str, List[Tuple[str, int]]]:
        """使用pymupdf库加载PDF（最强大的PDF处理库）"""
        if not PYMUPDF_AVAILABLE:
            raise ImportError("pymupdf未安装")

        full_text = ""
        pages_content = []

        try:
            # 打开PDF文档
            doc = fitz.open(file_path)

            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    page_text = page.get_text()

                    if page_text.strip():
                        full_text += f"\n\n--- 第{page_num + 1}页 ---\n\n{page_text}"
                        pages_content.append((page_text, page_num + 1))
                    else:
                        logger.warning(f"第{page_num + 1}页无法提取文本，可能是扫描件")
                        pages_content.append(("", page_num + 1))

                except Exception as e:
                    logger.error(f"处理第{page_num + 1}页时出错: {e}")
                    pages_content.append(("", page_num + 1))

            doc.close()
            return full_text, pages_content

        except Exception as e:
            logger.error(f"pymupdf加载失败: {e}")
            raise
    


    def load_document(self, file_path: str) -> Tuple[str, List[Tuple[str, int]]]:
        """
        统一文档加载方法 - 将所有格式转换为PDF后处理

        Args:
            file_path: 文件路径

        Returns:
            Tuple[str, List[Tuple[str, int]]]: (全文内容, [(页面内容, 页码)])
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        try:
            # 统一转换为PDF格式
            pdf_path = self.converter.convert_to_pdf(file_path)

            # 使用统一的PDF处理方法
            full_text, pages_content = self.load_pdf(pdf_path)

            # 如果是转换生成的临时PDF，记录原始文件信息
            original_extension = Path(file_path).suffix.lower()
            if pdf_path != file_path:
                logger.info(f"成功将{original_extension}文件转换为PDF并处理: {file_path}")

            return full_text, pages_content

        except Exception as e:
            logger.error(f"统一文档加载失败: {file_path}, 错误: {e}")
            raise
    
    def split_text(self, text: str, metadata: Optional[dict] = None) -> List[LangchainDocument]:
        """
        将文本分割成块

        Args:
            text: 要分割的文本
            metadata: 元数据

        Returns:
            List[LangchainDocument]: 分割后的文档块列表
        """
        if metadata is None:
            metadata = {}

        # 使用文本分割器分割文本
        chunks = self.text_splitter.split_text(text)

        # 创建LangchainDocument对象
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata.copy()
            doc_metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk)
            })

            # 提取页码/段落号信息并添加到元数据
            page_info = self._extract_location_info(chunk)
            if page_info:
                doc_metadata.update(page_info)

            documents.append(LangchainDocument(page_content=chunk, metadata=doc_metadata))

        logger.info(f"文本分割完成，共生成{len(documents)}个文档块")
        return documents

    def _extract_location_info(self, text: str) -> dict:
        """
        从文本块中提取页码信息（统一处理后都是页码格式）

        Args:
            text: 文本块

        Returns:
            dict: 包含页码信息的字典
        """
        import re
        location_info = {}

        # 查找页码标记（统一格式）
        page_pattern = r'--- 第(\d+)页 ---'
        page_matches = re.findall(page_pattern, text)
        if page_matches:
            # 如果有多个页码，取第一个
            location_info['page_number'] = int(page_matches[0])
            location_info['location_type'] = 'page'

        return location_info
    
    def process_document(self, file_path: str) -> Tuple[str, List[LangchainDocument]]:
        """
        处理文档：加载并分割

        Args:
            file_path: 文件路径

        Returns:
            Tuple[str, List[LangchainDocument]]: (全文内容, 文档块列表)
        """
        # 加载文档（统一转换为PDF后处理）
        full_text, _ = self.load_document(file_path)

        # 准备元数据
        metadata = {
            "source": file_path,
            "file_name": Path(file_path).name,
            "file_type": Path(file_path).suffix.lower()
        }

        # 分割文本
        documents = self.split_text(full_text, metadata)

        return full_text, documents

    def cleanup(self):
        """清理临时文件"""
        if hasattr(self, 'converter'):
            self.converter.cleanup()

    def __del__(self):
        """析构函数，确保清理临时文件"""
        self.cleanup()
