"""文档切片模块：提供多种切片策略（语义切分、固定大小切分、结构切分）"""

import uuid
from typing import Any

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

# SemanticChunker 在 langchain_experimental 中，非必需依赖
try:
    from langchain_experimental.text_splitter import SemanticChunker

    _has_semantic = True
except ImportError:
    SemanticChunker = None
    _has_semantic = False
from langchain_core.documents import Document

from config import settings
from core.embeddings import get_embedding_model
from utils.logger import logger


# SemanticChunker 在 langchain_experimental 中，非必需依赖


def split_documents(
        docs: list[Document],
        subject: str = "",
        grade: str = "",
        chapter: str = "",
        strategy: str = "recursive",
) -> list[dict]:
    """
    将加载后的 Document 列表按策略切分为切片。

    参数:
        docs: 原始文档列表
        subject: 学科（如 数学）
        grade: 年级（如 七年级）
        chapter: 章节名称
        strategy: 切分策略（recursive / semantic / markdown）

    返回:
        切片字典列表，每项包含 text, doc_id, subject, grade, chapter 等字段
    """
    logger.info(f"开始切片: strategy={strategy}, docs={len(docs)}个")
    if strategy == "markdown":
        chunks = _split_markdown(docs)
    elif strategy == "semantic":
        chunks = _split_semantic(docs)
    else:
        chunks = _split_recursive(docs)
    # 统一封装为字典格式
    doc_id = str(uuid.uuid4())
    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "text": chunk.page_content,
            "doc_id": doc_id,
            "subject": subject,
            "grade": grade,
            "chapter": chapter,
            "knowledge_point": chunk.metadata.get("knowledge_point", ""),
            "chunk_type": chunk.metadata.get("chunk_type", "text"),
            "chunk_index": i,
        })

    logger.info(f"切片完成，共 {len(result)} 个切片")
    return result


def _split_markdown(docs: list[Document]) -> list[Document]:
    """Markdown 标题层级切分：适合有层级结构的文档"""
    logger.info("使用 Markdown 标题层级切分")
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "header1"),
            ("##", "header2"),
            ("###", "header3"),
        ]
    )
    chunks = []
    for doc in docs:
        if doc.metadata.get("file_type") == "md":
            sub_chunks = splitter.split_text(doc.page_content)
            chunks.extend(sub_chunks)
        else:
            # 非 Markdown 文档回退到递归切分
            logger.debug("非 Markdown 文档，回退到递归切分")
    if not chunks:
        return _split_recursive(docs)
    return chunks


def _split_recursive(docs: list[Document]) -> list[Document]:
    """递归字符切分：适合普通文本和试题"""
    logger.info(f"使用递归字符切分: chunk_size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    pass


def _split_semantic(docs: list[Document]) -> list[Document]:
    """语义切分：适合教材正文、概念讲解"""
    if not _has_semantic:
        logger.warning("SemanticChunker 不可用，回退到递归切分（安装 langchain_experimental 可启用）")
        return _split_recursive(docs)
    embedding_model = get_embedding_model()
    splitter = SemanticChunker(
        embedding=embedding_model,
        breakpoint_threshold_type="percentile",
    )
    # SemanticChunker 只能对单个文本操作
    chunks = []
    for doc in docs:
        sub_chunks = splitter.split_documents([doc])
        chunks.extend(sub_chunks)
    logger.debug(f"语义切分得到 {len(chunks)} 个切片")
    return chunks
