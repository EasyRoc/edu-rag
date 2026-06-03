"""Focused tests for the RAG retrieval quality V1 behavior."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class RetrievalGateTests(unittest.TestCase):
    def test_accepts_when_top1_and_relevant_count_pass_thresholds(self):
        from core.retrieval_quality import evaluate_retrieval_gate

        decision = evaluate_retrieval_gate(
            [{"id": 1, "doc_id": "a", "rerank_score": 0.60}],
            retry_count=0,
            max_retries=2,
            reranker_available=True,
        )

        self.assertEqual(decision["action"], "accept")
        self.assertEqual(decision["metrics"]["relevant_count"], 1)

    def test_retries_empty_candidates_then_abstains_after_limit(self):
        from core.retrieval_quality import evaluate_retrieval_gate

        retry = evaluate_retrieval_gate([], retry_count=0, max_retries=2)
        abstain = evaluate_retrieval_gate([], retry_count=2, max_retries=2)

        self.assertEqual(retry["action"], "retry")
        self.assertEqual(retry["suggested_strategy"], "hyde")
        self.assertEqual(abstain["action"], "abstain")

    def test_enforce_mode_abstains_when_reranker_is_unavailable(self):
        from core.retrieval_quality import evaluate_retrieval_gate

        decision = evaluate_retrieval_gate(
            [{"id": 1, "score": 1.0}],
            reranker_available=False,
            gate_mode="enforce",
        )

        self.assertEqual(decision["action"], "abstain")
        self.assertIn("reranker_unavailable", decision["reason_codes"])

    def test_observe_mode_records_unavailable_reranker_but_allows_generation(self):
        from core.retrieval_quality import evaluate_retrieval_gate

        decision = evaluate_retrieval_gate(
            [{"id": 1, "score": 1.0}],
            reranker_available=False,
            gate_mode="observe",
        )

        self.assertEqual(decision["action"], "accept")
        self.assertIn("would_abstain", decision["reason_codes"])

    def test_low_top1_uses_step_back_after_first_retry(self):
        from core.retrieval_quality import evaluate_retrieval_gate

        decision = evaluate_retrieval_gate(
            [{"id": 1, "rerank_score": 0.59}],
            retry_count=1,
            max_retries=2,
            reranker_available=True,
        )

        self.assertEqual(decision["action"], "retry")
        self.assertEqual(decision["suggested_strategy"], "step_back")


class RerankerTests(unittest.IsolatedAsyncioTestCase):
    async def test_cross_encoder_is_lazy_and_scores_are_sigmoid_normalized(self):
        from core.reranker import CrossEncoderReranker

        calls = []

        class FakeModel:
            def predict(self, pairs, **kwargs):
                calls.append((pairs, kwargs))
                return [0.0, math.log(3)]

        model_builds = []

        def build_model(*args, **kwargs):
            model_builds.append((args, kwargs))
            return FakeModel()

        reranker = CrossEncoderReranker(
            enabled=True,
            model_name="fake-model",
            batch_size=2,
            model_factory=build_model,
        )

        self.assertEqual(model_builds, [])
        ranked = await reranker.rerank(
            "query",
            [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}],
        )

        self.assertEqual(len(model_builds), 1)
        self.assertEqual(calls[0][0], [("query", "a"), ("query", "b")])
        self.assertEqual(calls[0][1]["activation_fn"](2.0), 2.0)
        self.assertAlmostEqual(ranked[0]["rerank_score"], 0.75, places=6)
        self.assertAlmostEqual(ranked[1]["rerank_score"], 0.5, places=6)
        self.assertEqual(ranked[0]["id"], 2)

    async def test_disabled_reranker_reports_unavailable(self):
        from core.reranker import CrossEncoderReranker, RerankerUnavailableError

        reranker = CrossEncoderReranker(enabled=False)

        with self.assertRaises(RerankerUnavailableError):
            await reranker.rerank("query", [{"id": 1, "text": "a"}])


class VectorStoreScoreTests(unittest.TestCase):
    def test_rrf_preserves_component_scores_without_mutating_inputs(self):
        from core.vectorestore import K12VectorStore

        store = object.__new__(K12VectorStore)
        dense = [{"id": 1, "text": "a", "score": 0.9, "dense_raw_score": 0.9}]
        sparse = [{"id": 1, "text": "a", "score": 12.0, "sparse_raw_score": 12.0}]

        fused = store._rrf_fusion(dense, sparse, top_k=1)

        self.assertEqual(fused[0]["fusion_score"], 1.0)
        self.assertEqual(fused[0]["score"], 1.0)
        self.assertEqual(fused[0]["dense_raw_score"], 0.9)
        self.assertEqual(fused[0]["sparse_raw_score"], 12.0)
        self.assertEqual(dense[0]["score"], 0.9)
        self.assertEqual(sparse[0]["score"], 12.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
