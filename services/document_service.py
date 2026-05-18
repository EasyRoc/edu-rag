"""文档管理服务：上传、列表、删除文档，支持文件(PDF/MD/TXT)和SQL数据导入"""

import os
import uuid
import re

from config import settings
from ingestion.pipeline import IngestionPipeline
from core.vectorestore import K12VectorStore
from models.db_models import Document, get_session_maker
from utils.logger import logger


class DocumentService:
    """文档管理服务"""
    def __init__(self, vector_store: K12VectorStore, upload_dir: str | None = None):
        self.vector_store = vector_store
        self.pipeline = IngestionPipeline(vector_store)
        self.upload_dir = upload_dir or settings.UPLOAD_DIR
        os.makedirs(self.upload_dir, exist_ok=True)

    def _save_uploaded_file(self, doc_id: str, filename: str, file_content: bytes) -> str:
        safe_name = f"{doc_id}_{os.path.basename(filename)}"
        file_path = os.path.join(self.upload_dir, safe_name)
        with open(file_path, "wb") as f:
            f.write(file_content)
        return file_path

    async def _create_document_record(
        self,
        doc_id: str,
        title: str,
        doc_type: str,
        subject: str,
        grade: str,
        chapter: str,
        file_path: str,
    ) -> None:
        session_maker = get_session_maker()
        async with session_maker() as session:
            doc_record = Document(
                id=doc_id,
                title=title,
                doc_type=doc_type,
                subject=subject,
                grade=grade,
                chapter=chapter,
                file_path=file_path,
                status="processing",
            )
            session.add(doc_record)
            await session.commit()
            logger.info(f"文档记录已创建: {doc_id}")

    async def _update_document_record_status(self, doc_id: str, result: dict) -> None:
        session_maker = get_session_maker()
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

    @staticmethod
    def _delete_file_if_exists(file_path: str) -> bool:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"已删除文件: {file_path}")
            return True
        return False

    @staticmethod
    def _validate_sql_import_identifiers(
        table_name: str,
        id_column: str,
        columns: list[str] | None,
    ) -> None:
        pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        candidates = [table_name, id_column, *(columns or [])]
        invalid = [name for name in candidates if not pattern.match(name)]
        if invalid:
            raise ValueError(f"SQL 标识符不合法: {', '.join(invalid)}")

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
        file_path = self._save_uploaded_file(doc_id, filename, file_content)
        # 2. 创建文档记录
        await self._create_document_record(
            doc_id=doc_id,
            title=filename,
            doc_type=os.path.splitext(filename)[1].lstrip("."),
            subject=subject,
            grade=grade,
            chapter=chapter,
            file_path=file_path,
        )

        # 3. 执行 IngestPipeline
        result = self.pipeline.process_file(
            file_path=file_path,
            subject=subject,
            grade=grade,
            chapter=chapter,
            strategy=strategy,
        )
        # 4. 更新文档状态
        await self._update_document_record_status(doc_id, result)

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
        self._validate_sql_import_identifiers(table_name, id_column, columns)

        # 1. 创建文档记录
        doc_id = str(uuid.uuid4())
        await self._create_document_record(
            doc_id=doc_id,
            title=f"[SQL] {table_name}",
            doc_type="mysql",
            subject=subject,
            grade=grade,
            chapter=chapter,
            file_path=f"mysql://{table_name}",
        )

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
        await self._update_document_record_status(doc_id, result)

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
            self._delete_file_if_exists(doc.file_path)

            # 删除向量
            self.vector_store.delete_by_doc_id(doc_id)

            # 删除数据库记录
            await session.delete(doc)
            await session.commit()
            logger.info(f"文档已删除: {doc_id}")
            return True
