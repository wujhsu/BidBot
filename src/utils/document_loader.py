"""
文档加载器
Document loader utilities for processing PDF and DOCX files
"""

import os
from typing import List, Optional, Tuple
from pathlib import Path
import PyPDF2
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
from loguru import logger

class DocumentLoader:
    """文档加载器类"""
    
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
    
    def load_pdf(self, file_path: str) -> Tuple[str, List[Tuple[str, int]]]:
        """
        加载PDF文件
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            Tuple[str, List[Tuple[str, int]]]: (全文内容, [(页面内容, 页码)])
        """
        try:
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
            
            logger.info(f"成功加载PDF文件: {file_path}, 共{len(pages_content)}页")
            return full_text, pages_content
            
        except Exception as e:
            logger.error(f"加载PDF文件失败: {file_path}, 错误: {e}")
            raise
    
    def load_docx(self, file_path: str) -> Tuple[str, List[Tuple[str, int]]]:
        """
        加载DOCX文件
        
        Args:
            file_path: DOCX文件路径
            
        Returns:
            Tuple[str, List[Tuple[str, int]]]: (全文内容, [(段落内容, 段落号)])
        """
        try:
            doc = Document(file_path)
            full_text = ""
            paragraphs_content = []
            
            for para_num, paragraph in enumerate(doc.paragraphs, 1):
                para_text = paragraph.text.strip()
                if para_text:
                    full_text += f"\n\n--- 第{para_num}段 ---\n\n{para_text}"
                    paragraphs_content.append((para_text, para_num))
            
            logger.info(f"成功加载DOCX文件: {file_path}, 共{len(paragraphs_content)}段")
            return full_text, paragraphs_content
            
        except Exception as e:
            logger.error(f"加载DOCX文件失败: {file_path}, 错误: {e}")
            raise
    
    def load_txt(self, file_path: str) -> Tuple[str, List[Tuple[str, int]]]:
        """
        加载TXT文件

        Args:
            file_path: TXT文件路径

        Returns:
            Tuple[str, List[Tuple[str, int]]]: (全文内容, [(行内容, 行号)])
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            full_text = ""
            lines_content = []

            for line_num, line in enumerate(lines, 1):
                line_text = line.strip()
                if line_text:
                    full_text += f"\n\n--- 第{line_num}行 ---\n\n{line_text}"
                    lines_content.append((line_text, line_num))

            logger.info(f"成功加载TXT文件: {file_path}, 共{len(lines_content)}行")
            return full_text, lines_content

        except Exception as e:
            logger.error(f"加载TXT文件失败: {file_path}, 错误: {e}")
            raise

    def load_document(self, file_path: str) -> Tuple[str, List[Tuple[str, int]]]:
        """
        根据文件扩展名自动选择加载方法

        Args:
            file_path: 文件路径

        Returns:
            Tuple[str, List[Tuple[str, int]]]: (全文内容, [(内容, 位置号)])
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_extension = Path(file_path).suffix.lower()

        if file_extension == '.pdf':
            return self.load_pdf(file_path)
        elif file_extension in ['.docx', '.doc']:
            return self.load_docx(file_path)
        elif file_extension == '.txt':
            return self.load_txt(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_extension}")
    
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
            documents.append(LangchainDocument(page_content=chunk, metadata=doc_metadata))
        
        logger.info(f"文本分割完成，共生成{len(documents)}个文档块")
        return documents
    
    def process_document(self, file_path: str) -> Tuple[str, List[LangchainDocument]]:
        """
        处理文档：加载并分割
        
        Args:
            file_path: 文件路径
            
        Returns:
            Tuple[str, List[LangchainDocument]]: (全文内容, 文档块列表)
        """
        # 加载文档
        full_text, content_parts = self.load_document(file_path)
        
        # 准备元数据
        metadata = {
            "source": file_path,
            "file_name": Path(file_path).name,
            "file_type": Path(file_path).suffix.lower()
        }
        
        # 分割文本
        documents = self.split_text(full_text, metadata)
        
        return full_text, documents
