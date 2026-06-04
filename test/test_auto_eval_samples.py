"""自动问答评估样本的采集、查询与 CLI 回归测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TempDatabaseMixin:
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


class AutoEvalDatabaseTests(TempDatabaseMixin, unittest.IsolatedAsyncioTestCase):
    async def test_init_db_creates_auto_eval_samples_table(self):
        from models.db_models import AutoEvalSample, get_session_maker

        session_maker = get_session_maker()
        async with session_maker() as session:
            sample = AutoEvalSample(
                question="什么是浮力？",
                answer="浮力是液体或气体对物体向上的力。",
                contexts=["浮力相关教材片段"],
                reference_count=1,
                retrieval_decision={"action": "accept"},
            )
            session.add(sample)
            await session.commit()

        async with session_maker() as session:
            saved = await session.get(AutoEvalSample, sample.id)

        self.assertIsNotNone(saved)
        self.assertEqual(saved.question, "什么是浮力？")

    async def test_dataset_builder_reads_recent_auto_samples_with_filters(self):
        from evaluation.dataset_builder import EvalDatasetBuilder
        from models.db_models import AutoEvalSample, get_session_maker

        session_maker = get_session_maker()
        old_time = datetime(2026, 1, 1, 10, 0, 0)
        new_time = datetime(2026, 1, 1, 10, 1, 0)
        async with session_maker() as session:
            session.add_all(
                [
                    AutoEvalSample(
                        question="旧数学题",
                        answer="旧答案",
                        contexts=["旧上下文"],
                        subject="数学",
                        grade="七年级",
                        reference_count=1,
                        created_at=old_time,
                    ),
                    AutoEvalSample(
                        question="语文题",
                        answer="语文答案",
                        contexts=["语文上下文"],
                        subject="语文",
                        grade="七年级",
                        reference_count=1,
                        created_at=new_time,
                    ),
                    AutoEvalSample(
                        question="新数学题",
                        answer="新答案",
                        contexts=["新上下文"],
                        subject="数学",
                        grade="七年级",
                        reference_count=1,
                        created_at=new_time,
                    ),
                ]
            )
            await session.commit()

        dataset = await EvalDatasetBuilder.from_auto_samples(limit=1, subject="数学", grade="七年级")

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["question"], "新数学题")
        self.assertEqual(dataset[0]["contexts"], ["新上下文"])


class AutoEvalApiTests(TempDatabaseMixin, unittest.IsolatedAsyncioTestCase):
    async def test_from_auto_endpoint_evaluates_recent_auto_samples_with_filters(self):
        from api import evaluation as evaluation_api
        from evaluation.schemas import EvalResult, EvalSample
        from models.db_models import AutoEvalSample, get_session_maker

        session_maker = get_session_maker()
        async with session_maker() as session:
            session.add(
                AutoEvalSample(
                    question="自动样本问题",
                    answer="自动样本答案",
                    contexts=["自动样本上下文"],
                    subject="数学",
                    grade="七年级",
                    reference_count=1,
                )
            )
            await session.commit()

        calls = []

        async def fake_run_evaluation(dataset, name, metrics, save_to_db=True):
            calls.append({
                "dataset": dataset,
                "name": name,
                "metrics": metrics,
                "save_to_db": save_to_db,
            })
            return EvalResult(
                metrics=metrics,
                scores={"faithfulness": 0.9},
                sample_count=len(dataset),
                samples=[
                    EvalSample(
                        question=dataset[0]["question"],
                        answer=dataset[0]["answer"],
                        scores={"faithfulness": 0.9},
                    )
                ],
                extra={"record_id": "record-1", "name": name, "elapsed_seconds": 0.1},
            )

        original_run = evaluation_api.run_evaluation
        try:
            evaluation_api.run_evaluation = fake_run_evaluation
            response = await evaluation_api.evaluate_from_auto(
                limit=5,
                subject="数学",
                grade="七年级",
                metrics="faithfulness, answer_relevancy",
                name="frontend_auto",
            )
        finally:
            evaluation_api.run_evaluation = original_run

        self.assertEqual(response.code, 0)
        self.assertEqual(response.data["record_id"], "record-1")
        self.assertEqual(calls[0]["metrics"], ["faithfulness", "answer_relevancy"])
        self.assertTrue(calls[0]["save_to_db"])
        self.assertEqual(calls[0]["dataset"][0]["question"], "自动样本问题")

    async def test_from_auto_endpoint_returns_business_error_for_incompatible_metrics(self):
        from api import evaluation as evaluation_api
        from models.db_models import AutoEvalSample, get_session_maker

        session_maker = get_session_maker()
        async with session_maker() as session:
            session.add(
                AutoEvalSample(
                    question="自动样本问题",
                    answer="自动样本答案",
                    contexts=["自动样本上下文"],
                    reference_count=1,
                )
            )
            await session.commit()

        async def fake_run_evaluation(*args, **kwargs):
            raise ValueError("当前数据集没有 reference/ground_truth")

        original_run = evaluation_api.run_evaluation
        try:
            evaluation_api.run_evaluation = fake_run_evaluation
            response = await evaluation_api.evaluate_from_auto(
                limit=5,
                metrics="context_precision",
            )
        finally:
            evaluation_api.run_evaluation = original_run

        self.assertEqual(response.code, 1)
        self.assertIn("reference/ground_truth", response.message)


class RagasMetricCompatibilityTests(unittest.TestCase):
    def test_reference_dependent_metrics_are_skipped_without_reference_column(self):
        from datasets import Dataset
        from evaluation.ragas_evaluator import _prepare_dataset_and_metric_names

        dataset = Dataset.from_dict({
            "question": ["问题"],
            "answer": ["回答"],
            "contexts": [["上下文"]],
        })

        prepared, metric_names = _prepare_dataset_and_metric_names(
            dataset,
            ["faithfulness", "context_precision", "context_recall"],
        )

        self.assertEqual(prepared.column_names, ["question", "answer", "contexts"])
        self.assertEqual(metric_names, ["faithfulness"])

    def test_ground_truth_is_mirrored_to_reference_for_current_ragas_metrics(self):
        from datasets import Dataset
        from evaluation.ragas_evaluator import _prepare_dataset_and_metric_names

        dataset = Dataset.from_dict({
            "question": ["问题"],
            "answer": ["回答"],
            "contexts": [["上下文"]],
            "ground_truth": ["标准答案"],
        })

        prepared, metric_names = _prepare_dataset_and_metric_names(
            dataset,
            ["context_precision", "context_recall"],
        )

        self.assertIn("reference", prepared.column_names)
        self.assertEqual(prepared[0]["reference"], "标准答案")
        self.assertEqual(metric_names, ["context_precision", "context_recall"])


class AutoEvalCaptureTests(TempDatabaseMixin, unittest.IsolatedAsyncioTestCase):
    async def test_successful_educational_rag_answer_is_captured_without_user_id(self):
        from models.db_models import AutoEvalSample, get_session_maker
        from services.rag_service import RAGService

        class AcceptingGraph:
            async def ainvoke(self, *args, **kwargs):
                return {
                    "intent": "educational",
                    "complexity": "simple",
                    "answer": "可靠回答",
                    "retrieved_docs": [
                        {
                            "id": 101,
                            "text": "可靠教材片段",
                            "source_file": "lesson.md",
                            "rerank_score": 0.92,
                            "subject": "数学",
                            "grade": "七年级",
                        }
                    ],
                    "retrieval_decision": {"action": "accept", "reason_codes": ["quality_passed"]},
                    "retrieval_metrics": {"top1_score": 0.92},
                    "retrieval_attempts": [{"retry_count": 0}],
                }

        service = RAGService(vector_store=object(), rag_graph=AcceptingGraph())

        result = await service.ask("一元一次方程怎么解？", subject="数学", grade="七年级", session_id="s-1")

        self.assertIsNone(result["record_id"])
        session_maker = get_session_maker()
        async with session_maker() as session:
            saved = (await session.execute(AutoEvalSample.__table__.select())).all()

        self.assertEqual(len(saved), 1)
        sample = saved[0]._mapping
        self.assertEqual(sample["question"], "一元一次方程怎么解？")
        self.assertEqual(sample["answer"], "可靠回答")
        self.assertEqual(sample["contexts"], ["可靠教材片段"])
        self.assertIsNone(sample["user_id"])

    async def test_chitchat_and_abstain_answers_are_not_captured(self):
        from models.db_models import AutoEvalSample, get_session_maker
        from services.rag_service import RAGService

        class Graph:
            def __init__(self, state):
                self.state = state

            async def ainvoke(self, *args, **kwargs):
                return self.state

        chitchat = RAGService(
            vector_store=object(),
            rag_graph=Graph({"intent": "chitchat", "answer": "你好", "retrieved_docs": []}),
        )
        abstain = RAGService(
            vector_store=object(),
            rag_graph=Graph(
                {
                    "intent": "educational",
                    "answer": "抱歉，我暂时没有检索到足够可靠的资料来回答这个问题。",
                    "retrieved_docs": [],
                    "retrieval_decision": {"action": "abstain"},
                }
            ),
        )

        await chitchat.ask("你好", session_id="c")
        await abstain.ask("难题", session_id="a")

        session_maker = get_session_maker()
        async with session_maker() as session:
            rows = (await session.execute(AutoEvalSample.__table__.select())).all()

        self.assertEqual(rows, [])


class EvalFrontendMarkupTests(unittest.TestCase):
    def test_evaluation_tab_contains_manual_and_auto_modes(self):
        html = (PROJECT_ROOT / "static" / "index.html").read_text(encoding="utf-8")

        self.assertIn('data-eval-mode="manual"', html)
        self.assertIn('data-eval-mode="auto"', html)
        self.assertIn("eval-auto-limit", html)
        self.assertIn("data-needs-reference", html)
        self.assertIn("/api/v1/evaluation/from-auto", html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
