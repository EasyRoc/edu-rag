"""知识库构建流水线：文档加载 → 数据清洗 → 切片 → 向量化 → 入库"""

import os
import uuid
import time

from langchain_core.documents import Document as LCDocument

from ingestion.loader import load_document
from ingestion.chunker import split_documents
from ingestion.cleaner import (
    CleaningPipeline,
    FileSourceAdapter,
    SQLSourceAdapter,
    CleanStats,
)
from core.vectorestore import K12VectorStore
from utils.logger import logger


class IngestionPipeline:

    def __init__(self, vector_store: K12VectorStore):
        self.vector_store = vector_store

    def process_file(
            self,
            file_path: str,
            subject: str = "",
            grade: str = "",
            chapter: str = "",
            strategy: str = "recursive",
    ) -> dict:
        """
        处理单个文件：加载 → 清洗 → 切片 → 入库

        返回处理结果统计。
        """
        start_time = time.time()
        file_name = os.path.basename(file_path)
        logger.info(f"========== 开始处理文件: {file_name} ==========")

        # 1. 加载文档
        try:
            docs = load_document(file_path)
            logger.info(f"文档加载完成: {len(docs)} 页/段")
        except Exception as e:
            logger.error(f"文档加载失败: {e}")
            return {"status": "error", "message": str(e), "file": file_name}

        # 2. 数据清洗
        try:
            source_type = os.path.splitext(file_path)[1].lstrip(".")
            docs = self._clean_file_docs(docs, file_path, source_type)
            logger.info(f"数据清洗完成: {len(docs)} 条有效记录")
        except Exception as e:
            logger.error(f"数据清洗失败: {e}")
            return {"status": "error", "message": str(e), "file": file_name}

        if not docs:
            logger.warning("清洗后无有效数据")
            return {
                "status": "success",
                "file": file_name,
                "chunk_count": 0,
                "inserted_ids": 0,
                "elapsed_seconds": round(time.time() - start_time, 2),
                "clean_stats": {"input": 0, "output": 0, "message": "所有数据被过滤"},
            }

        # 3. 切片
        try:
            chunks = split_documents(
                docs,
                subject=subject,
                grade=grade,
                chapter=chapter,
                strategy=strategy,
            )
            logger.info(f"文档切片完成: {len(chunks)} 个切片")
        except Exception as e:
            logger.error(f"文档切片失败: {e}")
            return {"status": "error", "message": str(e), "file": file_name}

        # 4. 向量化 + 入库
        try:
            ids = self.vector_store.insert_chunks(chunks)
            elapsed = time.time() - start_time
            logger.info(f"文件处理完成: {file_name}, 耗时: {elapsed:.2f}s, 入库: {len(ids)} 条")
            return {
                "status": "success",
                "file": file_name,
                "chunk_count": len(chunks),
                "inserted_ids": len(ids),
                "elapsed_seconds": round(elapsed, 2),
            }
        except Exception as e:
            logger.error(f"向量入库失败: {e}")
            return {"status": "error", "message": str(e), "file": file_name}

    def process_sql(
            self,
            adapter: SQLSourceAdapter,
            subject: str = "",
            grade: str = "",
            chapter: str = "",
            strategy: str = "recursive",
    ) -> dict:
        """
        处理 SQL 数据导入：连接数据库 → 流式读取 → 行转文本 → 清洗 → 切片 → 入库

        参数:
            adapter: SQLSourceAdapter（已配置好连接参数和字段映射）
            subject: 学科
            grade: 年级
            chapter: 章节
            strategy: 切片策略

        返回处理结果统计。
        """
        start_time = time.time()
        table_name = adapter.table_name
        logger.info(f"========== 开始处理 SQL 数据: {table_name} ==========")

        # 1. 流式读取 + 清洗
        cleaner = CleaningPipeline()
        clean_results, stats = cleaner.clean_batch(
            adapter.stream_rows(),
            source_type="mysql",
            source_id=table_name,
        )
        logger.info(f"SQL 清洗完成: input={stats.input_count}, output={stats.output_count}")

        if not clean_results:
            return {
                "status": "success",
                "table": table_name,
                "chunk_count": 0,
                "inserted_ids": 0,
                "elapsed_seconds": round(time.time() - start_time, 2),
                "clean_stats": {
                    "input_count": stats.input_count,
                    "output_count": 0,
                    "dropped_count": stats.dropped_count,
                    "dedup_count": stats.dedup_count,
                },
            }

        # 2. 转为 langchain Document
        docs = [
            LCDocument(
                page_content=r.content,
                metadata={
                    "source_file": table_name,
                    "file_type": "mysql",
                    "clean_id": r.id,
                    **r.metadata,
                },
            )
            for r in clean_results
        ]

        # 3. 切片
        try:
            chunks = split_documents(
                docs,
                subject=subject,
                grade=grade,
                chapter=chapter,
                strategy=strategy,
            )
            logger.info(f"SQL 切片完成: {len(chunks)} 个切片")
        except Exception as e:
            logger.error(f"SQL 切片失败: {e}")
            return {"status": "error", "message": str(e), "table": table_name}

        # 4. 向量化 + 入库
        try:
            ids = self.vector_store.insert_chunks(chunks)
            elapsed = time.time() - start_time
            logger.info(f"SQL 处理完成: {table_name}, 耗时: {elapsed:.2f}s, 入库: {len(ids)} 条")
            return {
                "status": "success",
                "table": table_name,
                "chunk_count": len(chunks),
                "inserted_ids": len(ids),
                "elapsed_seconds": round(elapsed, 2),
                "clean_stats": {
                    "input_count": stats.input_count,
                    "output_count": stats.output_count,
                    "dropped_count": stats.dropped_count,
                    "dedup_count": stats.dedup_count,
                    "elapsed_ms": stats.elapsed_ms,
                },
            }
        except Exception as e:
            logger.error(f"向量入库失败: {e}")
            return {"status": "error", "message": str(e), "table": table_name}

    def _clean_file_docs(
            self,
            docs: list,
            file_path: str,
            source_type: str,
    ) -> list:
        """对加载后的文档执行清洗，返回清洗后的 langchain Document 列表"""
        cleaner = CleaningPipeline()
        source_id = os.path.basename(file_path)
        records = FileSourceAdapter.doc_to_records(docs)

        clean_results, stats = cleaner.clean_batch(
            records,
            source_type=source_type,
            source_id=source_id,
            file_name=source_id,
        )

        # 转回 langchain Document
        cleaned_docs = []
        for r in clean_results:
            cleaned_docs.append(
                LCDocument(
                    page_content=r.content,
                    metadata={
                        "source_file": source_id,
                        "file_type": source_type,
                        "clean_id": r.id,
                        "content_hash": r.metadata.get("content_hash", ""),
                        "quality_score": r.metadata.get("quality_score", 0.0),
                        "page": r.metadata.get("page", 0),
                        **{k: v for k, v in r.metadata.items()
                           if k not in ("content_hash", "quality_score", "page")},
                    },
                )
            )
        return cleaned_docs
