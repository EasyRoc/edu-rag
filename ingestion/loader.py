"""文档加载模块：支持 PDF、Markdown、TXT 等多种格式的文档加载"""

import os
from typing import AsyncGenerator

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

from utils.logger import logger

# 支持的文件扩展名及其对应的加载器
SUPPORTED_EXTENSIONS = {
    ".pdf": "pypdf",
    ".md": "markdown",
    ".txt": "text",
}


def load_document(file_path: str) -> list[Document]:
    """
       根据文件扩展名自动选择合适的加载器，返回 Document 列表。
       每个 Document 包含 page_content（文本）和 metadata（元数据）。
       """
    ext = os.path.splitext(file_path)[1].lower()
    logger.info(f"正在加载文档: {file_path} (类型: {ext})")

    if ext == ".pdf":
        return _load_pdf(file_path)
    elif ext == ".md":
        return _load_markdown(file_path)
    elif ext == ".txt":
        return _load_text(file_path)
    else:
        logger.error(f"不支持的文件类型: {ext}")
        raise ValueError(f"不支持的文件类型: {ext}，仅支持 PDF/MD/TXT")


def _load_pdf(file_path: str) -> list[Document]:
    """使用 PyPDF 加载 PDF 文件，每页一个 Document"""
    logger.info("使用 PyPDFLoader 加载 PDF")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    logger.info(f"PDF 加载完成，共 {len(docs)} 页")
    for i, doc in enumerate(docs):
        doc.metadata["page"] = i + 1
        doc.metadata["source_file"] = os.path.basename(file_path)
        doc.metadata["file_type"] = "pdf"
    return docs


def _load_text(file_path: str) -> list[Document]:
    """加载纯文本文件"""
    logger.info("使用 TextLoader 加载文本文件")
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    logger.info(f"文本文件加载完成，共 {len(docs)} 段")
    for doc in docs:
        doc.metadata["source_file"] = os.path.basename(file_path)
        doc.metadata["file_type"] = "txt"
    return docs


def _load_markdown(file_path: str) -> list[Document]:
    """加载 Markdown 文件"""
    logger.info("使用 UnstructuredMarkdownLoader 加载 Markdown")
    loader = UnstructuredMarkdownLoader(file_path, mode="elements")
    docs = loader.load()
    logger.info(f"Markdown 加载完成，共 {len(docs)} 个元素")
    for doc in docs:
        doc.metadata["source_file"] = os.path.basename(file_path)
        doc.metadata["file_type"] = "md"
