"""知识库构建流水线：文档加载 → 切片 → 向量化 → 入库"""

import os
import uuid
import time

from ingestion.loader import load_document
from ingestion.chunker import split_documents
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
        处理单个文件：加载 → 切片 → 入库

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

        # 2. 切片
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

        # 3. 向量化 + 入库
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
