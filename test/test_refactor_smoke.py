"""Regression tests for the stable refactor path.

These tests are intentionally small and dependency-light so they can run both
with pytest discovery and as a plain Python script.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class MarkdownLoaderTests(unittest.TestCase):
    def test_load_markdown_returns_documents_with_metadata(self):
        from ingestion.loader import load_document

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lesson.md"
            path.write_text("# 一元一次方程\n\n这是一个有效的 Markdown 教材片段。", encoding="utf-8")

            docs = load_document(str(path))

        self.assertIsInstance(docs, list)
        self.assertGreater(len(docs), 0)
        self.assertTrue(all(doc.metadata.get("source_file") == "lesson.md" for doc in docs))
        self.assertTrue(all(doc.metadata.get("file_type") == "md" for doc in docs))


class AppFactoryTests(unittest.TestCase):
    def test_importing_main_does_not_initialize_vector_store(self):
        sys.modules.pop("main", None)
        original_module = sys.modules.get("core.vectorestore")

        fake_module = types.ModuleType("core.vectorestore")

        class ExplodingVectorStore:
            def __init__(self):
                raise AssertionError("K12VectorStore should not be constructed during import")

        fake_module.K12VectorStore = ExplodingVectorStore
        sys.modules["core.vectorestore"] = fake_module
        try:
            main = importlib.import_module("main")
            self.assertTrue(callable(main.create_app))
            self.assertTrue(callable(main.build_app_state))
        finally:
            sys.modules.pop("main", None)
            if original_module is None:
                sys.modules.pop("core.vectorestore", None)
            else:
                sys.modules["core.vectorestore"] = original_module

    def test_create_app_can_register_routes_with_injected_state(self):
        sys.modules.pop("main", None)
        main = importlib.import_module("main")

        app = main.create_app(
            app_state=main.AppState(
                vector_store=object(),
                rag_graph=object(),
                rag_service=object(),
                document_service=object(),
                knowledge_service=object(),
                analytics_service=object(),
            ),
            initialize_runtime=False,
        )
        paths = {route.path for route in app.routes}

        self.assertIn("/health", paths)
        self.assertIn("/api/v1/rag/ask", paths)
        self.assertIn("/api/v1/documents/list", paths)


class ServiceHelperTests(unittest.TestCase):
    def test_format_references_preserves_compatible_fields(self):
        from services.rag_service import format_references

        refs = format_references(
            [
                {
                    "id": 12,
                    "text": "x" * 250,
                    "source_file": "",
                    "doc_id": "doc-1",
                    "score": 0.87654,
                    "subject": "数学",
                    "grade": "七年级",
                }
            ]
        )

        self.assertEqual(refs[0]["index"], 1)
        self.assertEqual(refs[0]["chunk_id"], 12)
        self.assertEqual(refs[0]["source_file"], "doc-1")
        self.assertEqual(refs[0]["source"], "doc-1")
        self.assertEqual(refs[0]["score"], 0.8765)
        self.assertEqual(len(refs[0]["text"]), 200)

    def test_vector_store_rrf_does_not_mutate_inputs(self):
        from core.vectorestore import K12VectorStore

        store = object.__new__(K12VectorStore)
        dense = [{"id": 1, "text": "a", "score": 0.9}]
        sparse = [{"id": 1, "text": "a", "score": 12.0}]

        fused = store._rrf_fusion(dense, sparse, top_k=1)

        self.assertEqual(fused[0]["score"], 1.0)
        self.assertEqual(dense[0]["score"], 0.9)
        self.assertEqual(sparse[0]["score"], 12.0)


class DocumentServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from config import settings
        from models import db_models

        self._original_database_url = settings.DATABASE_URL
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(self._tmpdir.name) / "business.db"
        settings.DATABASE_URL = f"sqlite+aiosqlite:///{db_path}"
        if db_models._engine is not None:
            await db_models._engine.dispose()
        db_models._engine = None
        db_models._session_maker = None
        await db_models.init_db()

    async def asyncTearDown(self):
        from config import settings
        from models import db_models

        if db_models._engine is not None:
            await db_models._engine.dispose()
        db_models._engine = None
        db_models._session_maker = None
        settings.DATABASE_URL = self._original_database_url
        self._tmpdir.cleanup()

    async def test_delete_document_removes_db_record_when_file_is_missing(self):
        from models.db_models import Document, get_session_maker
        from services.document_service import DocumentService

        class FakeVectorStore:
            def __init__(self):
                self.deleted_doc_ids = []

            def delete_by_doc_id(self, doc_id):
                self.deleted_doc_ids.append(doc_id)

        session_maker = get_session_maker()
        async with session_maker() as session:
            session.add(
                Document(
                    id="doc-1",
                    title="missing.txt",
                    doc_type="txt",
                    subject="数学",
                    file_path=str(Path(self._tmpdir.name) / "not-found.txt"),
                    status="completed",
                )
            )
            await session.commit()

        vector_store = FakeVectorStore()
        service = DocumentService(vector_store, upload_dir=str(Path(self._tmpdir.name) / "uploads"))

        deleted = await service.delete_document("doc-1")

        self.assertTrue(deleted)
        self.assertEqual(vector_store.deleted_doc_ids, ["doc-1"])
        async with session_maker() as session:
            self.assertIsNone(await session.get(Document, "doc-1"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
