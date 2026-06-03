"""在线检索质量门控。

这里有意只读取 `rerank_score`，不再混用 RRF、BM25 或余弦相似度。
RRF 只决定候选顺序，最终能否生成答案由重排分数决定。
"""

from __future__ import annotations

from typing import Literal, TypedDict

from config import settings
from utils.logger import logger


class RetrievalMetrics(TypedDict):
    candidate_count: int
    relevant_count: int
    distinct_doc_count: int
    top1_score: float
    topk_mean_score: float
    top1_margin: float
    coverage_ratio: float | None


class RetrievalDecision(TypedDict):
    action: Literal["accept", "retry", "abstain"]
    reason_codes: list[str]
    metrics: RetrievalMetrics
    suggested_strategy: str | None


def compute_retrieval_metrics(
    docs: list[dict],
    *,
    relevant_threshold: float | None = None,
) -> RetrievalMetrics:
    """基于归一化重排分数计算门控指标。"""
    threshold = (
        settings.RERANKER_RELEVANCE_THRESHOLD
        if relevant_threshold is None
        else relevant_threshold
    )
    scores = sorted(
        [float(doc["rerank_score"]) for doc in docs if doc.get("rerank_score") is not None],
        reverse=True,
    )
    top_scores = scores[:5]
    distinct_docs = {
        str(doc.get("doc_id") or doc.get("id"))
        for doc in docs
        if doc.get("doc_id") is not None or doc.get("id") is not None
    }
    return {
        "candidate_count": len(docs),
        "relevant_count": sum(score >= threshold for score in scores),
        "distinct_doc_count": len(distinct_docs),
        "top1_score": scores[0] if scores else 0.0,
        "topk_mean_score": sum(top_scores) / len(top_scores) if top_scores else 0.0,
        "top1_margin": scores[0] - scores[1] if len(scores) > 1 else (scores[0] if scores else 0.0),
        "coverage_ratio": None,
    }


def evaluate_retrieval_gate(
    docs: list[dict],
    *,
    retry_count: int = 0,
    max_retries: int = 2,
    reranker_available: bool = True,
    gate_mode: str | None = None,
    relevant_threshold: float | None = None,
    accept_top1_threshold: float | None = None,
) -> RetrievalDecision:
    """返回可被 LangGraph 序列化的 accept/retry/abstain 决策。"""
    mode = gate_mode or settings.RETRIEVAL_GATE_MODE
    relevant_min = (
        settings.RERANKER_RELEVANCE_THRESHOLD
        if relevant_threshold is None
        else relevant_threshold
    )
    top1_min = (
        settings.RETRIEVAL_ACCEPT_TOP1_THRESHOLD
        if accept_top1_threshold is None
        else accept_top1_threshold
    )
    metrics = compute_retrieval_metrics(docs, relevant_threshold=relevant_min)

    if not docs:
        action = "retry" if retry_count < max_retries else "abstain"
        logger.info("检索门控: action=%s, reason=no_candidates", action)
        return {
            "action": action,
            "reason_codes": ["no_candidates"],
            "metrics": metrics,
            "suggested_strategy": "hyde",
        }

    if not reranker_available:
        action = "accept" if mode == "observe" else "abstain"
        logger.info(
            "检索门控: action=%s, reason=reranker_unavailable, mode=%s",
            action,
            mode,
        )
        return {
            "action": action,
            "reason_codes": ["reranker_unavailable", "would_abstain"] if mode == "observe" else ["reranker_unavailable"],
            "metrics": metrics,
            "suggested_strategy": None,
        }

    if metrics["top1_score"] >= top1_min and metrics["relevant_count"] >= 1:
        logger.info(
            "检索门控: action=accept, top1=%.4f, relevant_count=%d",
            metrics["top1_score"],
            metrics["relevant_count"],
        )
        return {
            "action": "accept",
            "reason_codes": ["quality_passed"],
            "metrics": metrics,
            "suggested_strategy": None,
        }

    reasons = []
    if metrics["relevant_count"] == 0:
        reasons.append("no_relevant_docs")
    if metrics["top1_score"] < top1_min:
        reasons.append("low_top1_score")
    action = "retry" if retry_count < max_retries else "abstain"
    logger.info(
        "检索门控: action=%s, reasons=%s, top1=%.4f, relevant_count=%d",
        action,
        ",".join(reasons or ["quality_below_threshold"]),
        metrics["top1_score"],
        metrics["relevant_count"],
    )
    return {
        "action": action,
        "reason_codes": reasons or ["quality_below_threshold"],
        "metrics": metrics,
        "suggested_strategy": "hyde" if metrics["relevant_count"] == 0 else "step_back",
    }
