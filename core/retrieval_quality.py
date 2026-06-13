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
    │ coverage_ratio     │ 暂未启用（预留字段，用于衡量检索覆盖了知识库中多少       │
    │                     │ 相关知识片段）                                      │
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
        # top1_margin: 只有1条文档时 = 自身分数；0条 = 0.0
        "top1_margin": scores[0] - scores[1] if len(scores) > 1 else (scores[0] if scores else 0.0),
        "coverage_ratio": None,  # 预留，后续可基于 chunk 总量计算覆盖率
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
) -> RetrievalDecision:
    """在线检索质量门控 —— 决定检回来的文档能不能用、要不要重试、还是放弃回答。

    三种决策结果：
    - accept: 文档质量过关，直接进入 LLM 生成
    - retry:  质量不够但有重试配额，换策略重新检索
    - abstain: 质量不够且重试次数用尽，放弃回答（拒答）

    决策流程（四个分支，从上到下短路判断）：
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. 没召回任何文档                                                │
    │    → 有重试配额？retry（建议用 HyDE 策略重新检索）                  │
    │    → 没配额？abstain                                             │
    ├─────────────────────────────────────────────────────────────────┤
    │ 2. 重排器挂了（模型没加载成功 / 推理报错）                           │
    │    → observe 模式：accept（只记录，不阻断，线上观察用）              │
    │    → enforce 模式：abstain（严格模式，宁可拒答也不给不可靠的结果）    │
    ├─────────────────────────────────────────────────────────────────┤
    │ 3. 质量达标：top1 分数 >= 门限 且 至少1条相关文档                  │
    │    → accept                                                     │
    ├─────────────────────────────────────────────────────────────────┤
    │ 4. 质量不达标（走到这说明 1/2/3 都不满足）                         │
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
    metrics = compute_retrieval_metrics(docs, relevant_threshold=relevant_min)

    # ── 分支 1: 空召回 ──
    if not docs:
        action = "retry" if retry_count < max_retries else "abstain"
        logger.info("检索门控: action=%s, reason=no_candidates", action)
        return {
            "action": action,
            "reason_codes": ["no_candidates"],
            "metrics": metrics,
            "suggested_strategy": "hyde",  # 空召回尝试生成假设答案来增强检索
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
        }

    # ── 分支 3: 质量达标，直接放行 ──
    # top1_min（0.60）> relevant_min（0.50），所以 top1 >= 0.60 时 relevant_count 必然 >= 1
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
        }

    # ── 分支 4: 质量不达标 ──
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
    }
