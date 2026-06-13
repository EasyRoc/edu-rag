"""策略化检索与纠正重试规划。

本模块只负责“怎么召回候选”，不负责“候选是否足够可靠”。
可靠性判断统一放在 `core.retrieval_quality`，避免多个阈值体系互相打架。
"""

from __future__ import annotations

from config import settings
from core.strategies import (
    StrategyType,
    decompose_query,
    generate_hypothetical_answer,
    generate_query_variants,
    generate_step_back_query,
    merge_sub_results,
    multi_query_fusion,
    select_strategy,
)
from core.vectorestore import K12VectorStore
from utils.logger import logger


def _top_k_for(complexity: str) -> int:
    """保留旧的复杂度到 top_k 映射，供直接检索兼容测试使用。"""
    return {"simple": 3, "medium": 5, "complex": 8}.get(complexity, 5)


def _annotate(docs: list[dict], strategy: str, query_variant: str) -> list[dict]:
    """给每个候选补充检索策略和查询变体，方便后续评估切片。"""
    results = []
    for doc in docs:
        item = dict(doc)
        item["retrieval_strategy"] = strategy
        item["query_variant"] = query_variant
        results.append(item)
    return results


def _search(
    vector_store: K12VectorStore,
    *,
    query: str,
    subject: str | None,
    grade: str | None,
    top_k: int,
    strategy: str,
) -> list[dict]:
    """执行一次底层混合检索，并附加来源策略信息。"""
    docs = vector_store.hybrid_search(query=query, subject=subject, grade=grade, top_k=top_k)
    return _annotate(docs, strategy, query)


def _direct_retrieve(
    vector_store: K12VectorStore,
    query: str,
    complexity: str,
    subject: str | None = None,
    grade: str | None = None,
    *,
    top_k: int | None = None,
) -> list[dict]:
    """直接混合检索，不依赖 LLM 生成额外查询。"""
    return _search(
        vector_store,
        query=query,
        subject=subject,
        grade=grade,
        top_k=top_k or _top_k_for(complexity),
        strategy=StrategyType.DIRECT.value,
    )


async def _multi_query_retrieve(
    vector_store: K12VectorStore,
    query: str,
    complexity: str,
    subject: str | None = None,
    grade: str | None = None,
    *,
    top_k: int | None = None,
) -> list[dict]:
    """多查询召回：原问题 + 改写问题分别检索，再通过 RRF 融合。"""
    variants = await generate_query_variants(query)
    if not variants:
        logger.info("多查询变体为空，降级为直接检索")
        return _direct_retrieve(vector_store, query, complexity, subject, grade, top_k=top_k)
    limit = top_k or _top_k_for(complexity)
    results = [
        _search(
            vector_store,
            query=item,
            subject=subject,
            grade=grade,
            top_k=limit,
            strategy=StrategyType.MULTI_QUERY.value,
        )
        for item in [query, *variants]
    ]
    return _annotate(multi_query_fusion(results, limit), StrategyType.MULTI_QUERY.value, query)


async def _decomposition_retrieve(
    vector_store: K12VectorStore,
    query: str,
    complexity: str,
    subject: str | None = None,
    grade: str | None = None,
    *,
    top_k: int | None = None,
) -> tuple[list[dict], list[str]]:
    """复杂问题拆解召回：子问题分别检索，再合并去重。"""
    sub_queries = await decompose_query(query)
    if len(sub_queries) <= 1:
        logger.info("问题分解结果不足，降级为多查询检索")
        docs = await _multi_query_retrieve(
            vector_store, query, complexity, subject, grade, top_k=top_k
        )
        return docs, [query]
    limit = top_k or _top_k_for(complexity)
    results = []
    for item in sub_queries:
        docs = _search(
            vector_store,
            query=item,
            subject=subject,
            grade=grade,
            top_k=max(3, limit // 2),
            strategy=StrategyType.DECOMPOSITION.value,
        )
        for doc in docs:
            doc["source_sub_query"] = item
        results.append(docs)
    return _annotate(merge_sub_results(results, limit), StrategyType.DECOMPOSITION.value, query), sub_queries


async def build_retry_plan(
    *,
    query: str,
    next_retry_count: int,
    decision: dict,
) -> dict:
    """根据门控失败原因规划下一轮纠正检索。

    渐进式重试 —— 第一次盲扩，后面对症下药：

    初始检索失败（策略由 select_strategy 按复杂度选择）
        │
        ▼
    【重试1 — query_variants】盲扩
        原因：还不知道为什么失败，先换几种问法试试覆盖面
        手段：生成3个同义改写变体，原始 query + 变体多路同时检索后 RRF 融合
        │
        ├── 成了 → 放行
        │
        └── 又败了 → 门控已经诊断出原因
                        │
                        ▼
                  【重试2 — 对症下药】

                  relevant_count == 0  → hyde
                    所有文档都不相关，说明查询词和知识库的表达方式不匹配
                    → 生成假设答案，用答案文本去检索（答案用词更贴近知识库）

                  top1_score 低       → step_back
                    有文档但质量不够，说明查询太窄或太具体
                    → 回溯到更宽泛的概念（浮力公式推导 → 浮力原理）
    """
    if next_retry_count == 1:
        # 第一次重试：盲扩——还不知道失败原因，用多查询变体扩大覆盖面
        variants = await generate_query_variants(query, n=3)
        queries = []
        for item in [query, *variants]:
            if item and item not in queries:
                queries.append(item)
        plan = {"strategy": "query_variants", "queries": queries[:4]}
        logger.info("生成第一次纠正检索计划: %s", plan)
        return plan
    # 第二次及以后：已有门控诊断，按病因选择 hyde 或 step_back
    strategy = decision.get("suggested_strategy") or "step_back"
    plan = {"strategy": strategy, "queries": [query]}
    logger.info("生成后续纠正检索计划: %s", plan)
    return plan


async def _retrieve_from_plan(
    vector_store: K12VectorStore,
    *,
    query: str,
    plan: dict,
    subject: str | None,
    grade: str | None,
    top_k: int,
) -> list[dict]:
    """按 retry_planner 输出的计划执行纠正检索。"""
    strategy = plan.get("strategy", "initial")
    queries = list(plan.get("queries") or [query])
    if strategy == "hyde":
        hypothetical = await generate_hypothetical_answer(query)
        queries = [query, *([hypothetical] if hypothetical else [])]
        logger.info("HyDE 纠正检索查询数: %d", len(queries))
    elif strategy == "step_back":
        step_back = await generate_step_back_query(query)
        queries = [query, *([step_back] if step_back else [])]
        logger.info("Step-Back 纠正检索查询数: %d", len(queries))

    results = [
        _search(
            vector_store,
            query=item,
            subject=subject,
            grade=grade,
            top_k=top_k,
            strategy=strategy,
        )
        for item in queries
    ]
    return _annotate(multi_query_fusion(results, top_k), strategy, query)


async def hybrid_retrieve(
    vector_store: K12VectorStore,
    query: str,
    complexity: str,
    intent: str = "educational",
    subject: str | None = None,
    grade: str | None = None,
    *,
    retrieval_plan: dict | None = None,
    candidate_top_k: int | None = None,
) -> tuple[list[dict], list[str]]:
    """召回候选文档；在线质量判断会在重排后完成。"""
    limit = candidate_top_k or settings.RETRIEVAL_CANDIDATE_TOP_K
    plan = retrieval_plan or {"strategy": "initial", "queries": [query]}
    if plan.get("strategy") != "initial":
        docs = await _retrieve_from_plan(
            vector_store,
            query=query,
            plan=plan,
            subject=subject,
            grade=grade,
            top_k=limit,
        )
        logger.info("纠正检索完成: strategy=%s, count=%d", plan.get("strategy"), len(docs))
        return docs, []

    strategy = select_strategy(intent, complexity, query)
    sub_queries: list[str] = []
    if strategy == StrategyType.DIRECT:
        docs = _direct_retrieve(vector_store, query, complexity, subject, grade, top_k=limit)
    elif strategy == StrategyType.MULTI_QUERY:
        docs = await _multi_query_retrieve(vector_store, query, complexity, subject, grade, top_k=limit)
    else:
        docs, sub_queries = await _decomposition_retrieve(
            vector_store, query, complexity, subject, grade, top_k=limit
        )
    logger.info("检索候选完成: strategy=%s, count=%d", strategy.value, len(docs))
    return docs, sub_queries
