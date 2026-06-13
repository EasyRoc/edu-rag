"""离线检索评估、门控回放与阈值校准。"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from config import settings
from core.nodes.retriever import build_retry_plan, hybrid_retrieve
from core.reranker import CrossEncoderReranker, RerankerUnavailableError
from core.retrieval_quality import evaluate_retrieval_gate
from utils.logger import logger


def load_retrieval_cases(path: str) -> list[dict]:
    """加载并校验检索标注集，支持 JSON 与 JSONL。"""
    source = Path(path)
    raw = source.read_text(encoding="utf-8")
    if source.suffix == ".jsonl":
        cases = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        data = json.loads(raw)
        cases = data if isinstance(data, list) else [data]
    for index, case in enumerate(cases, start=1):
        if not case.get("question"):
            raise ValueError(f"case {index} 缺少 question")
        if not isinstance(case.get("answerable"), bool):
            raise ValueError(f"case {index} 缺少布尔字段 answerable")
        case.setdefault("id", f"case-{index}")
        case.setdefault("relevant_chunk_ids", [])
        case.setdefault("complexity", "medium")
        case.setdefault("tags", [])
    logger.info("检索评估集加载完成: path=%s, cases=%d", path, len(cases))
    return cases


def compute_ranking_metrics(*, relevant_chunk_ids: set[int], docs: list[dict]) -> dict[str, float]:
    """计算单条可回答样本的排序指标。"""
    ranked_ids = [int(doc.get("id", 0)) for doc in docs]

    def recall_at(k: int) -> float:
        return len(set(ranked_ids[:k]) & relevant_chunk_ids) / len(relevant_chunk_ids) if relevant_chunk_ids else 0.0

    def precision_at(k: int) -> float:
        return len(set(ranked_ids[:k]) & relevant_chunk_ids) / k

    first_rank = next((index for index, chunk_id in enumerate(ranked_ids[:10], start=1) if chunk_id in relevant_chunk_ids), None)
    dcg = sum(1.0 / math.log2(index + 1) for index, chunk_id in enumerate(ranked_ids[:10], start=1) if chunk_id in relevant_chunk_ids)
    ideal_hits = min(len(relevant_chunk_ids), 10)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return {
        "recall@5": recall_at(5),
        "recall@10": recall_at(10),
        "recall@20": recall_at(20),
        "precision@5": precision_at(5),
        "mrr@10": 1.0 / first_rank if first_rank else 0.0,
        "ndcg@10": dcg / idcg if idcg else 0.0,
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def build_retrieval_report(case_results: list[dict]) -> dict:
    """聚合检索指标、门控误判率、延迟分位数和切片统计。"""
    ranking = [item["ranking_metrics"] for item in case_results if item.get("ranking_metrics")]
    answerable = [item for item in case_results if item["case"]["answerable"]]
    unanswerable = [item for item in case_results if not item["case"]["answerable"]]
    recovered = [item for item in answerable if item["initial_action"] != "accept" and item["action"] == "accept"]
    recoverable = [item for item in answerable if item["initial_action"] != "accept"]

    slices: dict[str, list[dict]] = defaultdict(list)
    for item in case_results:
        case = item["case"]
        for key in ("subject", "grade", "complexity"):
            slices[f"{key}={case.get(key) or 'unknown'}"].append(item)
        slices[f"strategy={item.get('strategy') or 'unknown'}"].append(item)
        slices[f"retry_count={item.get('retry_count', 0)}"].append(item)

    def summarize_slice(items: list[dict]) -> dict:
        return {
            "sample_count": len(items),
            "accept_rate": round(_rate(sum(item["action"] == "accept" for item in items), len(items)), 4),
            "abstain_rate": round(_rate(sum(item["action"] == "abstain" for item in items), len(items)), 4),
        }

    metrics = {
        key: round(mean(item[key] for item in ranking), 4) if ranking else 0.0
        for key in ("recall@5", "recall@10", "recall@20", "precision@5", "mrr@10", "ndcg@10")
    }
    metrics.update(
        {
            "sample_count": len(case_results),
            "false_accept_rate": round(_rate(sum(item["action"] == "accept" for item in unanswerable), len(unanswerable)), 4),
            "false_reject_rate": round(_rate(sum(item["action"] != "accept" for item in answerable), len(answerable)), 4),
            "abstention_accuracy": round(_rate(sum(item["action"] == "abstain" for item in unanswerable), len(unanswerable)), 4),
            "retry_recovery_rate": round(_rate(len(recovered), len(recoverable)), 4),
            "latency_ms": {
                "retrieval_p50": round(_percentile([item["retrieval_latency_ms"] for item in case_results], 0.5), 3),
                "retrieval_p95": round(_percentile([item["retrieval_latency_ms"] for item in case_results], 0.95), 3),
                "rerank_p50": round(_percentile([item["rerank_latency_ms"] for item in case_results], 0.5), 3),
                "rerank_p95": round(_percentile([item["rerank_latency_ms"] for item in case_results], 0.95), 3),
                "total_p50": round(_percentile([item["total_latency_ms"] for item in case_results], 0.5), 3),
                "total_p95": round(_percentile([item["total_latency_ms"] for item in case_results], 0.95), 3),
            },
            "slices": {key: summarize_slice(items) for key, items in sorted(slices.items())},
            "cases": case_results,
        }
    )
    logger.info(
        "检索评估汇总: samples=%d, recall@5=%.4f, false_accept=%.4f, false_reject=%.4f",
        metrics["sample_count"],
        metrics["recall@5"],
        metrics["false_accept_rate"],
        metrics["false_reject_rate"],
    )
    return metrics


async def evaluate_retrieval_case(
    case: dict,
    *,
    vector_store: Any,
    reranker: CrossEncoderReranker,
) -> dict:
    """执行单条检索评估：只跑检索、重排和门控，不调用答案生成。"""
    plan = {"strategy": "initial", "queries": [case["question"]]}
    retry_count = 0
    initial_action = ""
    retrieval_latency_ms = 0.0
    rerank_latency_ms = 0.0
    total_started = time.perf_counter()
    docs: list[dict] = []
    decision: dict = {}

    while True:
        started = time.perf_counter()
        docs, _sub_queries = await hybrid_retrieve(
            vector_store=vector_store,
            query=case["question"],
            complexity=case.get("complexity", "medium"),
            subject=case.get("subject"),
            grade=case.get("grade"),
            retrieval_plan=plan,
            candidate_top_k=settings.RETRIEVAL_CANDIDATE_TOP_K,
        )
        retrieval_latency_ms += (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        try:
            docs = await reranker.rerank(case["question"], docs)
            reranker_available = True
        except RerankerUnavailableError:
            reranker_available = False
        rerank_latency_ms += (time.perf_counter() - started) * 1000
        decision = evaluate_retrieval_gate(
            docs,
            retry_count=retry_count,
            max_retries=settings.MAX_RETRIES,
            reranker_available=reranker_available,
            complexity=case.get("complexity", "medium"),
        )
        initial_action = initial_action or decision["action"]
        if decision["action"] != "retry":
            break
        retry_count += 1
        plan = await build_retry_plan(
            query=case["question"],
            next_retry_count=retry_count,
            decision=decision,
        )

    relevant_ids = {int(chunk_id) for chunk_id in case.get("relevant_chunk_ids", [])}
    logger.info(
        "检索样本完成: id=%s, action=%s, retry_count=%d, docs=%d",
        case.get("id"),
        decision["action"],
        retry_count,
        len(docs),
    )
    return {
        "case": case,
        "ranking_metrics": compute_ranking_metrics(relevant_chunk_ids=relevant_ids, docs=docs) if relevant_ids else None,
        "initial_action": initial_action,
        "action": decision["action"],
        "retry_count": retry_count,
        "strategy": plan.get("strategy", "initial"),
        "retrieval_latency_ms": round(retrieval_latency_ms, 3),
        "rerank_latency_ms": round(rerank_latency_ms, 3),
        "total_latency_ms": round((time.perf_counter() - total_started) * 1000, 3),
        "docs": docs,
    }


async def evaluate_retrieval_cases(
    cases: list[dict],
    *,
    vector_store: Any,
    reranker: CrossEncoderReranker | None = None,
) -> dict:
    """批量执行检索评估并返回聚合报告。"""
    reranker = reranker or CrossEncoderReranker()
    results = [
        await evaluate_retrieval_case(case, vector_store=vector_store, reranker=reranker)
        for case in cases
    ]
    return build_retrieval_report(results)


def calibrate_thresholds(case_results: list[dict], *, max_false_accept_rate: float = 0.05) -> dict:
    """在错误接受率预算下推荐 top1 与相关性阈值。"""
    answerable = [item for item in case_results if item.get("answerable", item.get("case", {}).get("answerable")) is True]
    unanswerable = [item for item in case_results if item.get("answerable", item.get("case", {}).get("answerable")) is False]
    if not answerable:
        raise ValueError("校准集必须包含 answerable=true 样本")
    if not unanswerable:
        raise ValueError("校准集必须包含 answerable=false 样本")

    candidates = []
    grid = [round(index / 20, 2) for index in range(21)]
    for top1_threshold in grid:
        for relevant_threshold in grid:
            def accepted(item: dict) -> bool:
                scores = [float(doc.get("rerank_score", 0.0)) for doc in item.get("docs", [])]
                return bool(scores) and max(scores) >= top1_threshold and any(score >= relevant_threshold for score in scores)

            false_accept_rate = _rate(sum(accepted(item) for item in unanswerable), len(unanswerable))
            if false_accept_rate > max_false_accept_rate:
                continue
            answerable_accept_rate = _rate(sum(accepted(item) for item in answerable), len(answerable))
            candidates.append(
                {
                    "top1_threshold": top1_threshold,
                    "relevant_threshold": relevant_threshold,
                    "false_accept_rate": round(false_accept_rate, 4),
                    "answerable_accept_rate": round(answerable_accept_rate, 4),
                }
            )
    if not candidates:
        raise ValueError("没有满足错误接受率预算的阈值组合")
    recommendation = min(
        candidates,
        key=lambda item: (
            -item["answerable_accept_rate"],
            item["false_accept_rate"],
            abs(item["top1_threshold"] - settings.RETRIEVAL_ACCEPT_TOP1_THRESHOLD)
            + abs(item["relevant_threshold"] - settings.RERANKER_RELEVANCE_THRESHOLD),
        ),
    )
    logger.info("阈值校准完成: %s", recommendation)
    return recommendation
