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


# 模块级缓存，避免重复初始化 splitter 和 embedding model
_recursive_splitter = None
_semantic_splitter = None


def _get_recursive_splitter() -> RecursiveCharacterTextSplitter:
    global _recursive_splitter
    if _recursive_splitter is None:
        _recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", ".", " ", ""],
            length_function=len,
        )
    return _recursive_splitter


def _get_semantic_splitter() -> SemanticChunker:
    global _semantic_splitter
    if _semantic_splitter is None:
        embedding_model = get_embedding_model()
        _semantic_splitter = SemanticChunker(
            embedding=embedding_model,
            breakpoint_threshold_type="percentile",
        )
    return _semantic_splitter


def _enforce_max_chunk_size(chunks: list[Document]) -> list[Document]:
    """对超大 chunk 二次切分，解决长段落黑洞和语义距离漂移问题。

    长段落黑洞：相邻句子语义始终相似 → 整段无断点 → 变成巨大 chunk。
    语义漂移：句1~句50逐句相似，但首尾已完全不同 → 局部相似掩盖了全局漂移。
    两者都通过硬性大小上限 + 递归切分兜底来解决。
    """
    result = []
    splitter = _get_recursive_splitter()
    for chunk in chunks:
        if len(chunk.page_content) > settings.CHUNK_SIZE:
            result.extend(splitter.split_documents([chunk]))
        else:
            result.append(chunk)
    return result


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
            "page": chunk.metadata.get("page", 0),
            "source_file": chunk.metadata.get("source_file", ""),
            "file_type": chunk.metadata.get("file_type", ""),
        })

    logger.info(f"切片完成，共 {len(result)} 个切片")
    return result


def _split_markdown(docs: list[Document]) -> list[Document]:
    """Markdown 标题层级切分：适合有层级结构的文档，非 md 文档回退递归切分"""
    logger.info("使用 Markdown 标题层级切分")
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "header1"),
            ("##", "header2"),
            ("###", "header3"),
        ]
    )
    recursive = _get_recursive_splitter()
    chunks = []
    for doc in docs:
        if doc.metadata.get("file_type") == "md":
            sub_chunks = md_splitter.split_text(doc.page_content)
        else:
            sub_chunks = recursive.split_documents([doc])
        for chunk in sub_chunks:
            if len(chunk.page_content) > settings.CHUNK_SIZE:
                chunks.extend(recursive.split_documents([chunk]))
            else:
                chunks.append(chunk)
    return chunks


def _split_recursive(docs: list[Document]) -> list[Document]:
    """递归字符切分：适合普通文本和试题"""
    logger.info(f"使用递归字符切分: chunk_size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP}")
    return _get_recursive_splitter().split_documents(docs)


def _split_semantic(docs: list[Document]) -> list[Document]:
    """语义切分：适合教材正文、概念讲解

    会二次拆分超大 chunk，防止两个典型问题：
    1. 长段落黑洞 — 语义连贯的长段落找不到断点，整段变成一个 chunk
    2. 语义距离漂移 — 相邻句逐对相似但首尾已完全不同，局部相似掩盖全局漂移
    """
    if not _has_semantic:
        logger.warning("SemanticChunker 不可用，回退到递归切分")
        return _get_recursive_splitter().split_documents(docs)
    splitter = _get_semantic_splitter()
    chunks = splitter.split_documents(docs)
    logger.debug(f"语义切分得到 {len(chunks)} 个一级切片")
    return _enforce_max_chunk_size(chunks)
