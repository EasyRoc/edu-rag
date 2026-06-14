"""在线检索质量门控。

这里有意只读取 `rerank_score`，不再混用 RRF、BM25 或余弦相似度。
RRF 只决定候选顺序，最终能否生成答案由重排分数决定。
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

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
    total_subquery_count: int
    covered_subquery_count: int
    missing_subqueries: list[str]
    weak_subqueries: list[str]
    subquery_metrics: list[dict]


class RetrievalDecision(TypedDict):
    action: Literal["accept", "retry", "abstain"]
    reason_codes: list[str]
    metrics: RetrievalMetrics
    suggested_strategy: str | None
    suggested_plan: NotRequired[dict | None]


def compute_retrieval_metrics(
    docs: list[dict],
    *,
    relevant_threshold: float | None = None,
    sub_queries: list[str] | None = None,
    subquery_top1_threshold: float | None = None,
) -> RetrievalMetrics:
    """基于归一化重排分数计算门控指标。

    各指标含义：
    ┌────────────────────┬──────────────────────────────────────────────────┐
    │ 指标                │ 含义                                              │
    ├────────────────────┼──────────────────────────────────────────────────┤
    │ candidate_count     │ 传入的候选文档总数（召回+重排后的全部文档数）         │
    │ relevant_count      │ rerank_score >= 相关阈值的文档数，                   │
    │                     │ 即“被认为有实际参考价值”的文档数                     │
    │ distinct_doc_count  │ 去重后的独立文档数（按 doc_id/id 去重），             │
    │                     │ 用于检测返回结果是否有大量重复内容                    │
    │ top1_score          │ 排名第一的文档的 rerank_score，                      │
    │                     │ 直接决定能否走快速放行通道                            │
    │ topk_mean_score     │ 前5名文档的平均 rerank_score，                       │
    │                     │ 反映文档集整体质量（top1 可能偶然命中，均值更稳定）     │
    │ top1_margin         │ top1 与 top2 的分数差，                             │
    │                     │ 差值越大说明 top1 是“碾压式胜出”，答案唯一性越高；     │
    │                     │ 差值越小说明多个文档争第一，答案可能非唯一              │
    │ coverage_ratio     │ 复杂问题中已覆盖子问题数 / 总子问题数；               │
    │                     │ 简单问题或未传子问题时为 None                        │
    └────────────────────┴──────────────────────────────────────────────────┘
    """
    threshold = (
        settings.RERANKER_RELEVANCE_THRESHOLD
        if relevant_threshold is None
        else relevant_threshold
    )
    # 所有文档的 rerank_score 降序排列
    scores = sorted(
        [float(doc["rerank_score"]) for doc in docs if doc.get("rerank_score") is not None],
        reverse=True,
    )
    top_scores = scores[:5]
    # 按 doc_id 去重，避免同一段内容被多次切块后霸占 topK
    distinct_docs = {
        _doc_identity(doc)
        for doc in docs
        if _doc_identity(doc)
    }
    clean_sub_queries = _dedupe_texts(sub_queries or [])
    subquery_metrics = _compute_subquery_metrics(
        docs,
        sub_queries=clean_sub_queries,
        relevant_threshold=threshold,
        top1_threshold=subquery_top1_threshold or threshold,
    )
    covered_count = sum(1 for item in subquery_metrics if item["status"] == "covered")
    total_subquery_count = len(subquery_metrics)
    return {
        "candidate_count": len(docs),
        "relevant_count": sum(score >= threshold for score in scores),
        "distinct_doc_count": len(distinct_docs),
        "top1_score": scores[0] if scores else 0.0,
        "topk_mean_score": sum(top_scores) / len(top_scores) if top_scores else 0.0,
        # top1_margin: 只有1条文档时 = 自身分数；0条 = 0.0
        "top1_margin": scores[0] - scores[1] if len(scores) > 1 else (scores[0] if scores else 0.0),
        "coverage_ratio": covered_count / total_subquery_count if total_subquery_count else None,
        "total_subquery_count": total_subquery_count,
        "covered_subquery_count": covered_count,
        "missing_subqueries": [
            item["query"] for item in subquery_metrics if item["status"] == "missing"
        ],
        "weak_subqueries": [
            item["query"] for item in subquery_metrics if item["status"] == "weak"
        ],
        "subquery_metrics": subquery_metrics,
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
    complexity: str = "medium",
    sub_queries: list[str] | None = None,
) -> RetrievalDecision:
    """在线检索质量门控 —— 决定检回来的文档能不能用、要不要重试、还是放弃回答。

    三种决策结果：
    - accept: 文档质量过关，直接进入 LLM 生成
    - retry:  质量不够但有重试配额，换策略重新检索
    - abstain: 质量不够且重试次数用尽，放弃回答（拒答）

    决策流程（五个分支，从上到下短路判断）：
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. 没召回任何文档                                                │
    │    → 有重试配额？retry（建议用 HyDE 策略重新检索）                  │
    │    → 没配额？abstain                                             │
    ├─────────────────────────────────────────────────────────────────┤
    │ 2. 重排器挂了（模型没加载成功 / 推理报错）                           │
    │    → observe 模式：accept（只记录，不阻断，线上观察用）              │
    │    → enforce 模式：abstain（严格模式，宁可拒答也不给不可靠的结果）    │
    ├─────────────────────────────────────────────────────────────────┤
    │ 3. 复杂问题子问题覆盖不足：某个 sub_query 没有达标证据              │
    │    → 有重试配额？retry（建议 complex_repair 定向修复）              │
    │    → 没配额？abstain                                             │
    ├─────────────────────────────────────────────────────────────────┤
    │ 4. 质量达标：top1 分数 >= 门限 且 至少1条相关文档                  │
    │    → accept                                                     │
    ├─────────────────────────────────────────────────────────────────┤
    │ 5. 质量不达标（走到这说明 1/2/3/4 都不满足）                       │
    │    → 有重试配额？retry                                           │
    │       · 无相关文档   → 建议 HyDE（生成假设答案来检索）              │
    │       · 有文档但分低  → 建议 step_back（回溯到更宽泛的查询）        │
    │    → 没配额？abstain                                             │
    └─────────────────────────────────────────────────────────────────┘
    """
    mode = gate_mode or settings.RETRIEVAL_GATE_MODE
    if complexity == "complex":
        default_relevant = settings.COMPLEX_RELEVANCE_THRESHOLD
        default_top1 = settings.COMPLEX_ACCEPT_TOP1_THRESHOLD
        if max_retries == settings.MAX_RETRIES:
            max_retries = settings.COMPLEX_MAX_RETRIES
    else:
        default_relevant = settings.RERANKER_RELEVANCE_THRESHOLD
        default_top1 = settings.RETRIEVAL_ACCEPT_TOP1_THRESHOLD

    # 每条文档要被认定为"相关"的最低 rerank_score
    relevant_min = default_relevant if relevant_threshold is None else relevant_threshold
    # top1 文档直接放行的最低分数（远高于 relevant_min，确保领头文档足够可靠）
    top1_min = default_top1 if accept_top1_threshold is None else accept_top1_threshold
    clean_sub_queries = _dedupe_texts(sub_queries or [])
    use_subquery_gate = complexity == "complex" and len(clean_sub_queries) >= 2
    metrics = compute_retrieval_metrics(
        docs,
        relevant_threshold=relevant_min,
        sub_queries=clean_sub_queries if use_subquery_gate else None,
        subquery_top1_threshold=top1_min,
    )

    # ── 分支 1: 空召回 ──
    if not docs:
        action = "retry" if retry_count < max_retries else "abstain"
        logger.info("检索门控: action=%s, reason=no_candidates", action)
        return {
            "action": action,
            "reason_codes": ["no_candidates"],
            "metrics": metrics,
            "suggested_strategy": "hyde",  # 空召回尝试生成假设答案来增强检索
            "suggested_plan": _build_complex_repair_plan(metrics) if use_subquery_gate else None,
        }

    # ── 分支 2: 重排器不可用，无法做质量判断 ──
    if not reranker_available:
        # observe 模式：线上灰度观察阶段，放行请求但记录异常
        # enforce 模式：正式拦截，拒绝回答
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
            "suggested_plan": None,
        }

    # ── 分支 3: 复杂问题必须覆盖每个子问题 ──
    if use_subquery_gate and metrics["covered_subquery_count"] < metrics["total_subquery_count"]:
        action = "retry" if retry_count < max_retries else "abstain"
        reasons = ["subquery_coverage_low"]
        if metrics["missing_subqueries"]:
            reasons.append("missing_subqueries")
        if metrics["weak_subqueries"]:
            reasons.append("weak_subqueries")
        logger.info(
            "检索门控: action=%s, reasons=%s, subquery_coverage=%d/%d",
            action,
            ",".join(reasons),
            metrics["covered_subquery_count"],
            metrics["total_subquery_count"],
        )
        return {
            "action": action,
            "reason_codes": reasons,
            "metrics": metrics,
            "suggested_strategy": "hyde" if metrics["missing_subqueries"] else "step_back",
            "suggested_plan": _build_complex_repair_plan(metrics),
        }

    # ── 分支 4: 质量达标，直接放行 ──
    # 默认阈值满足 top1_min > relevant_min，所以 top1 达标时至少有一条相关候选。
    if metrics["top1_score"] >= top1_min:
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
            "suggested_plan": None,
        }

    # ── 分支 5: 质量不达标 ──
    reasons = []
    if metrics["relevant_count"] == 0:
        reasons.append("no_relevant_docs")   # 所有文档 rerank 分都低于相关阈值
    if metrics["top1_score"] < top1_min:
        reasons.append("low_top1_score")     # 领头文档置信度不够
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
        # 全部不相关 → 原始查询词可能没命中，用 HyDE 生成假设答案来拉回相关文档
        # 文档存在但分低 → 查询范围太窄/表达不匹配，用 step_back 回溯到更宽泛的概念
        "suggested_strategy": "hyde" if metrics["relevant_count"] == 0 else "step_back",
        "suggested_plan": _build_complex_repair_plan(metrics) if use_subquery_gate else None,
    }


def _dedupe_texts(items: list[str]) -> list[str]:
    """按出现顺序去重，避免同一子问题重复拉低覆盖率。"""
    results: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in results:
            results.append(text)
    return results


def _doc_identity(doc: dict) -> str:
    """统一文档身份字段，避免缺少 id 时把不同片段算作同一条。"""
    for key in ("doc_id", "id", "chunk_id"):
        value = doc.get(key)
        if value is not None:
            return f"{key}:{value}"
    text = str(doc.get("text") or doc.get("content") or "")
    if text:
        return f"text:{text[:200]}"
    return ""


def _source_matches_query(source: str, query: str) -> bool:
    if not source or not query:
        return False
    parts = [item.strip() for item in source.split(";") if item.strip()]
    return query in parts or query in source


def _score_summary(docs: list[dict]) -> tuple[list[float], float, float]:
    scores = sorted(
        [float(doc["rerank_score"]) for doc in docs if doc.get("rerank_score") is not None],
        reverse=True,
    )
    top_scores = scores[:5]
    return scores, scores[0] if scores else 0.0, sum(top_scores) / len(top_scores) if top_scores else 0.0


def _compute_subquery_metrics(
    docs: list[dict],
    *,
    sub_queries: list[str],
    relevant_threshold: float,
    top1_threshold: float,
) -> list[dict]:
    """计算每个子问题自己的候选质量，用于复杂问题覆盖门控。"""
    results: list[dict] = []
    for query in sub_queries:
        query_docs = [
            doc
            for doc in docs
            if _source_matches_query(str(doc.get("source_sub_query", "")), query)
        ]
        scores, top1_score, topk_mean_score = _score_summary(query_docs)
        relevant_count = sum(score >= relevant_threshold for score in scores)
        if relevant_count == 0:
            status = "missing"
            repair = "hyde"
        elif top1_score < top1_threshold:
            status = "weak"
            repair = "step_back"
        else:
            status = "covered"
            repair = "direct"
        results.append(
            {
                "query": query,
                "candidate_count": len(query_docs),
                "relevant_count": relevant_count,
                "top1_score": top1_score,
                "topk_mean_score": topk_mean_score,
                "status": status,
                "repair": repair,
            }
        )
    return results


def _build_complex_repair_plan(metrics: RetrievalMetrics) -> dict:
    """把子问题质量诊断转换成 retry_planner 可执行的修复计划。"""
    return {
        "strategy": "complex_repair",
        "sub_queries": [item["query"] for item in metrics["subquery_metrics"]],
        "subqueries": [
            {
                "query": item["query"],
                "status": item["status"],
                "repair": item["repair"],
            }
            for item in metrics["subquery_metrics"]
        ],
    }
