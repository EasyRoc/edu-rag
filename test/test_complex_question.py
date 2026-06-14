"""Complex-question optimization regression tests."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _make_docs(*scores: float) -> list[dict]:
    return [{"id": i, "text": f"doc{i}", "rerank_score": score} for i, score in enumerate(scores)]


class TestComplexityGradedGate(unittest.TestCase):
    def test_complex_accepts_lower_top1_than_default(self):
        from core.retrieval_quality import evaluate_retrieval_gate

        decision = evaluate_retrieval_gate(_make_docs(0.50, 0.40, 0.30), complexity="complex")

        self.assertEqual(decision["action"], "accept")

    def test_complex_rejects_below_complex_threshold(self):
        from core.retrieval_quality import evaluate_retrieval_gate

        decision = evaluate_retrieval_gate(
            _make_docs(0.30, 0.25),
            complexity="complex",
            retry_count=2,
            max_retries=2,
        )

        self.assertEqual(decision["action"], "abstain")

    def test_simple_uses_strict_threshold(self):
        from core.retrieval_quality import evaluate_retrieval_gate

        decision = evaluate_retrieval_gate(
            _make_docs(0.50, 0.40),
            complexity="simple",
            retry_count=2,
            max_retries=2,
        )

        self.assertEqual(decision["action"], "abstain")

    def test_medium_unchanged(self):
        from core.retrieval_quality import evaluate_retrieval_gate

        decision = evaluate_retrieval_gate(_make_docs(0.65, 0.55), complexity="medium")

        self.assertEqual(decision["action"], "accept")

    def test_complex_retries_when_sub_query_coverage_is_incomplete(self):
        from core.retrieval_quality import evaluate_retrieval_gate

        docs = [
            {
                "id": 1,
                "text": "A 的强证据",
                "rerank_score": 0.82,
                "source_sub_query": "子问题A",
            },
            {
                "id": 2,
                "text": "A 的补充证据",
                "rerank_score": 0.70,
                "source_sub_query": "子问题A",
            },
        ]

        decision = evaluate_retrieval_gate(
            docs,
            complexity="complex",
            sub_queries=["子问题A", "子问题B"],
            retry_count=0,
            max_retries=1,
        )

        self.assertEqual(decision["action"], "retry")
        self.assertIn("subquery_coverage_low", decision["reason_codes"])
        self.assertEqual(decision["metrics"]["covered_subquery_count"], 1)
        self.assertEqual(decision["metrics"]["total_subquery_count"], 2)
        self.assertAlmostEqual(decision["metrics"]["coverage_ratio"], 0.5)
        self.assertEqual(decision["suggested_plan"]["strategy"], "complex_repair")
        repairs = {item["query"]: item["repair"] for item in decision["suggested_plan"]["subqueries"]}
        self.assertEqual(repairs["子问题B"], "hyde")

    def test_complex_abstains_when_sub_query_coverage_still_incomplete_after_retry_limit(self):
        from core.retrieval_quality import evaluate_retrieval_gate

        decision = evaluate_retrieval_gate(
            [
                {
                    "id": 1,
                    "text": "A 的强证据",
                    "rerank_score": 0.82,
                    "source_sub_query": "子问题A",
                }
            ],
            complexity="complex",
            sub_queries=["子问题A", "子问题B"],
            retry_count=1,
            max_retries=1,
        )

        self.assertEqual(decision["action"], "abstain")
        self.assertIn("subquery_coverage_low", decision["reason_codes"])


class TestSubQueryAnnotation(unittest.TestCase):
    def test_merge_annotates_sub_query_source(self):
        from core.strategies.decomposition import merge_sub_results

        sub1 = [{"id": 1, "text": "a", "score": 0.9, "source_sub_query": "子问题A"}]
        sub2 = [{"id": 2, "text": "b", "score": 0.8, "source_sub_query": "子问题B"}]

        merged = merge_sub_results([sub1, sub2], top_k=10)
        sources = {doc.get("source_sub_query") for doc in merged}

        self.assertIn("子问题A", sources)
        self.assertIn("子问题B", sources)

    def test_merge_combines_sources_for_duplicate_chunk(self):
        from core.strategies.decomposition import merge_sub_results

        sub1 = [{"id": 1, "text": "a", "score": 0.9, "source_sub_query": "子问题A"}]
        sub2 = [{"id": 1, "text": "a", "score": 0.8, "source_sub_query": "子问题B"}]

        merged = merge_sub_results([sub1, sub2], top_k=10)

        self.assertEqual(len(merged), 1)
        self.assertIn("子问题A", merged[0]["source_sub_query"])
        self.assertIn("子问题B", merged[0]["source_sub_query"])

    def test_merge_does_not_collapse_distinct_docs_without_id(self):
        from core.strategies.decomposition import merge_sub_results

        sub1 = [{"text": "不同片段A", "score": 0.9, "source_sub_query": "子问题A"}]
        sub2 = [{"text": "不同片段B", "score": 0.8, "source_sub_query": "子问题B"}]

        merged = merge_sub_results([sub1, sub2], top_k=10)

        self.assertEqual(len(merged), 2)
        self.assertEqual({doc["text"] for doc in merged}, {"不同片段A", "不同片段B"})

    def test_state_has_sub_queries_field(self):
        from core.state import RAGState

        state: RAGState = {
            "query": "test",
            "subject": None,
            "grade": None,
            "session_id": "s1",
            "intent": "educational",
            "complexity": "complex",
            "retrieved_docs": [],
            "answer": "",
            "retry_count": 0,
            "max_retries": 2,
            "conversation_history": [],
            "retrieval_plan": {},
            "retrieval_attempts": [],
            "retrieval_metrics": {},
            "retrieval_decision": {},
            "abstain_reason": "",
            "retrieval_latency_ms": 0.0,
            "rerank_latency_ms": 0.0,
            "reranker_available": True,
            "sub_queries": [],
            "_queue_id": "",
        }

        self.assertEqual(state["sub_queries"], [])


class FakeTwoStageReranker:
    async def rerank(self, query: str, docs: list[dict]) -> list[dict]:
        ranked = []
        for doc in docs:
            item = dict(doc)
            item["rerank_raw_score"] = float(item.get("quality", item.get("score", 0.5)))
            item["rerank_score"] = float(item.get("quality", item.get("score", 0.5)))
            item["rerank_query"] = query
            ranked.append(item)
        return sorted(ranked, key=lambda item: item["rerank_score"], reverse=True)


class TestTwoStageRerank(unittest.TestCase):
    def test_two_stage_rerank_preserves_sub_query_sources(self):
        from core.graph import _two_stage_rerank

        docs = [
            {"id": 1, "text": "勾股定理定义", "score": 0.9, "source_sub_query": "勾股定理是什么"},
            {"id": 2, "text": "相似三角形判定", "score": 0.85, "source_sub_query": "相似三角形判定定理"},
            {"id": 3, "text": "勾股定理应用例题", "score": 0.8, "source_sub_query": "勾股定理的例题"},
        ]
        sub_queries = ["勾股定理是什么", "相似三角形判定定理", "勾股定理的例题"]

        final = asyncio.run(
            _two_stage_rerank(
                "比较勾股定理和相似三角形的异同",
                docs,
                sub_queries,
                FakeTwoStageReranker(),
            )
        )

        sources = {doc.get("source_sub_query", "") for doc in final}
        self.assertTrue(any("勾股定理" in source for source in sources))
        self.assertTrue(any("相似三角形" in source for source in sources))


class TestSubAnswerSynthesis(unittest.TestCase):
    def test_sub_answer_prompt_structure(self):
        from core.nodes.generator import _build_sub_answer_prompt

        prompt = _build_sub_answer_prompt(
            sub_query="勾股定理是什么",
            context_docs=[{"text": "勾股定理是直角三角形斜边平方等于两直角边平方和"}],
        )

        self.assertIn("勾股定理是什么", prompt)
        self.assertIn("直角三角形", prompt)

    def test_synthesis_prompt_includes_sub_answers(self):
        from core.nodes.generator import _build_synthesis_prompt

        prompt = _build_synthesis_prompt(
            original_query="比较勾股定理和相似三角形",
            sub_answers=[
                ("勾股定理是什么", "勾股定理是a²+b²=c²"),
                ("相似三角形判定", "SSS/SAS/AA三种判定方法"),
            ],
            context_docs=[{"text": "勾股定理定义"}, {"text": "相似三角形判定"}],
        )

        self.assertIn("a²+b²=c²", prompt)
        self.assertIn("SSS/SAS/AA", prompt)
        self.assertIn("比较勾股定理和相似三角形", prompt)


class TestComplexGenerateBranch(unittest.TestCase):
    def test_synthesis_path_branch_condition(self):
        from core.graph import _should_use_synthesis

        self.assertTrue(_should_use_synthesis("complex", ["q1", "q2", "q3"], True))
        self.assertFalse(_should_use_synthesis("medium", ["q1", "q2"], True))
        self.assertFalse(_should_use_synthesis("complex", ["q1"], True))
        self.assertFalse(_should_use_synthesis("complex", ["q1", "q2"], False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
