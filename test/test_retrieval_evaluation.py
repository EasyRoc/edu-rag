"""Pure retrieval evaluation and threshold calibration tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class RankingMetricsTests(unittest.TestCase):
    def test_compute_ranking_metrics_uses_chunk_ids(self):
        from evaluation.retrieval_evaluator import compute_ranking_metrics

        metrics = compute_ranking_metrics(
            relevant_chunk_ids={2, 4},
            docs=[{"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}],
        )

        self.assertEqual(metrics["recall@5"], 1.0)
        self.assertEqual(metrics["precision@5"], 0.4)
        self.assertEqual(metrics["mrr@10"], 1.0)
        self.assertGreater(metrics["ndcg@10"], 0.9)

    def test_report_tracks_acceptance_and_retry_recovery(self):
        from evaluation.retrieval_evaluator import build_retrieval_report

        report = build_retrieval_report(
            [
                {
                    "case": {"subject": "数学", "grade": "九年级", "complexity": "simple", "answerable": True},
                    "ranking_metrics": {"recall@5": 1.0, "recall@10": 1.0, "recall@20": 1.0, "precision@5": 0.2, "mrr@10": 1.0, "ndcg@10": 1.0},
                    "initial_action": "retry",
                    "action": "accept",
                    "retry_count": 1,
                    "strategy": "query_variants",
                    "retrieval_latency_ms": 10.0,
                    "rerank_latency_ms": 5.0,
                    "total_latency_ms": 20.0,
                    "docs": [{"id": 1, "rerank_score": 0.8}],
                },
                {
                    "case": {"subject": "数学", "grade": "九年级", "complexity": "simple", "answerable": False},
                    "ranking_metrics": None,
                    "initial_action": "abstain",
                    "action": "abstain",
                    "retry_count": 0,
                    "strategy": "direct",
                    "retrieval_latency_ms": 4.0,
                    "rerank_latency_ms": 2.0,
                    "total_latency_ms": 8.0,
                    "docs": [],
                },
            ]
        )

        self.assertEqual(report["false_accept_rate"], 0.0)
        self.assertEqual(report["false_reject_rate"], 0.0)
        self.assertEqual(report["abstention_accuracy"], 1.0)
        self.assertEqual(report["retry_recovery_rate"], 1.0)
        self.assertIn("subject=数学", report["slices"])


class CalibrationTests(unittest.TestCase):
    def test_calibration_respects_false_accept_budget(self):
        from evaluation.retrieval_evaluator import calibrate_thresholds

        recommendation = calibrate_thresholds(
            [
                {"answerable": False, "docs": [{"rerank_score": 0.55}]},
                {"answerable": True, "docs": [{"rerank_score": 0.65}]},
                {"answerable": True, "docs": [{"rerank_score": 0.85}]},
            ],
            max_false_accept_rate=0.0,
        )

        self.assertEqual(recommendation["false_accept_rate"], 0.0)
        self.assertEqual(recommendation["answerable_accept_rate"], 1.0)
        self.assertGreater(recommendation["top1_threshold"], 0.55)

    def test_calibration_requires_both_answerable_classes(self):
        from evaluation.retrieval_evaluator import calibrate_thresholds

        with self.assertRaisesRegex(ValueError, "answerable=true"):
            calibrate_thresholds([{"answerable": False, "docs": []}])

        with self.assertRaisesRegex(ValueError, "answerable=false"):
            calibrate_thresholds([{"answerable": True, "docs": []}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
