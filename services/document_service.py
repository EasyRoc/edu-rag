"""文档管理服务：上传、列表、删除文档，支持文件(PDF/MD/TXT)和SQL数据导入"""

import os
import uuid
import time
from datetime import datetime

from ingestion.pipeline import IngestionPipeline
from core.vectorestore import K12VectorStore
from models.db_models import Document, get_session_maker
from utils.logger import logger


# 文档存储目录
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploaded_docs")

class DocumentService:
    """文档管理服务"""
    def __init__(self, vector_store: K12VectorStore):
        self.vector_store = vector_store
        self.pipeline = IngestionPipeline(vector_store)
        os.makedirs(DOCS_DIR, exist_ok=True)

    async def upload_and_process(
            self,
            file_content: bytes,
            filename: str,
            subject: str,
            grade: str = "",
            chapter: str = "",
            strategy: str = "recursive",
    ) -> dict:
        """
        上传文件并处理入库。

        流程：保存文件 → 创建文档记录 → 执行 IngestPipeline → 更新状态
        """
        logger.info(f"上传文档: {filename}, subject={subject}, grade={grade}")

        # 1. 保存文件
        doc_id = str(uuid.uuid4())
        safe_name = f"{doc_id}_{filename}"
        file_path = os.path.join(DOCS_DIR, safe_name)
        with open(file_path, "wb") as f:
            f.write(file_content)
        # 2. 创建文档记录
        session_maker = get_session_maker()
        async with session_maker() as session:
            doc_record = Document(
                id=doc_id,
                title=filename,
                doc_type=os.path.splitext(filename)[1].lstrip("."),
                subject=subject,
                grade=grade,
                chapter=chapter,
                file_path=file_path,
                status="processing",
            )
            session.add(doc_record)
            await session.commit()
            logger.info(f"文档记录已创建: {doc_id}")

        # 3. 执行 IngestPipeline
        result = self.pipeline.process_file(
            file_path=file_path,
            subject=subject,
            grade=grade,
            chapter=chapter,
            strategy=strategy,
        )
        # 4. 更新文档状态
        async with session_maker() as session:
            doc = await session.get(Document, doc_id)
            if doc:
                if result["status"] == "success":
                    doc.status = "completed"
                    doc.chunk_count = result.get("chunk_count", 0)
                else:
                    doc.status = "failed"
                    doc.error_message = result.get("message", "未知错误")
                await session.commit()

        result["doc_id"] = doc_id
        return result

    async def import_from_sql(
            self,
            db_url: str,
            table_name: str,
            subject: str = "",
            grade: str = "",
            chapter: str = "",
            field_map: dict[str, str] | None = None,
            id_column: str = "id",
            columns: list[str] | None = None,
            where_clause: str = "",
            batch_size: int = 1000,
            strategy: str = "recursive",
    ) -> dict:
        """
        连接数据库 → 流式读取 → 清洗 → 切片 → 入库。

        参数:
            db_url: 数据库连接串，如 mysql+pymysql://user:pass@host:3306/db
            table_name: 源表名
            subject: 学科
            grade: 年级
            chapter: 章节
            field_map: 字段中文映射 {字段名: 中文标签}
            id_column: 主键列名，用于游标分页
            columns: 需要查询的列
            where_clause: 附加 WHERE 条件
            batch_size: 每批读取行数
            strategy: 切片策略
        """
        from ingestion.cleaner import SQLSourceAdapter

        logger.info(f"SQL 导入: db={db_url}, table={table_name}, subject={subject}")

        # 1. 创建文档记录
        doc_id = str(uuid.uuid4())
        session_maker = get_session_maker()
        async with session_maker() as session:
            doc_record = Document(
                id=doc_id,
                title=f"[SQL] {table_name}",
                doc_type="mysql",
                subject=subject,
                grade=grade,
                chapter=chapter,
                file_path=f"mysql://{table_name}",
                status="processing",
            )
            session.add(doc_record)
            await session.commit()
            logger.info(f"SQL 文档记录已创建: {doc_id}")

        # 2. 创建 adapter 并执行 pipeline（stream_rows 内部管理连接生命周期）
        adapter = SQLSourceAdapter(
            db_url=db_url,
            table_name=table_name,
            field_map=field_map,
            id_column=id_column,
            columns=columns,
            where_clause=where_clause,
            batch_size=batch_size,
        )
        result = self.pipeline.process_sql(
            adapter=adapter,
            subject=subject,
            grade=grade,
            chapter=chapter,
            strategy=strategy,
        )

        # 3. 更新文档状态
        async with session_maker() as session:
            doc = await session.get(Document, doc_id)
            if doc:
                if result["status"] == "success":
                    doc.status = "completed"
                    doc.chunk_count = result.get("chunk_count", 0)
                else:
                    doc.status = "failed"
                    doc.error_message = result.get("message", "未知错误")
                await session.commit()

        result["doc_id"] = doc_id
        return result

    async def list_documents(self) -> list[dict]:
        """获取文档列表"""
        session_maker = get_session_maker()
        async with session_maker() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(Document).order_by(Document.created_at.desc())
            )
            docs = result.scalars().all()
            return [
                {
                    "id": d.id,
                    "title": d.title,
                    "doc_type": d.doc_type,
                    "subject": d.subject,
                    "grade": d.grade,
                    "chapter": d.chapter,
                    "chunk_count": d.chunk_count,
                    "status": d.status,
                    "error_message": d.error_message,
                    "created_at": d.created_at.isoformat() if d.created_at else "",
                }
                for d in docs
            ]

    async def delete_document(self, doc_id: str) -> bool:
        """删除文档及其向量"""
        logger.info(f"删除文档: {doc_id}")
        session_maker = get_session_maker()
        async with session_maker() as session:
            doc = await session.get(Document, doc_id)
            if not doc:
                logger.warning(f"文档不存在: {doc_id}")
                return False

            # 删除本地文件
            if os.path.exists(doc.file_path):
                os.remove(doc.file_path)
                logger.info(f"已删除文件: {doc.file_path}")

            # 删除向量
            self.vector_store.delete_by_doc_id(doc_id)

            # 删除数据库记录
            await session.delete(doc)
            await session.commit()
            logger.info(f"文档已删除: {doc_id}")
            return True
