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
    complexity: str = "medium",
    sub_queries: list[str] | None = None,
) -> dict:
    """根据门控失败原因规划下一轮纠正检索。

    普通问题沿用渐进式重试：第一次 query variants 盲扩，第二次根据门控
    建议选择 HyDE 或 Step-Back。复杂问题如果携带多个 `sub_queries`，则优先
    执行门控给出的 `complex_repair` 计划，按子问题分别 direct / hyde /
    step_back 修复，避免重试后丢失子问题感知重排与子答案合成能力。
    """
    clean_sub_queries = _dedupe_queries(sub_queries or [])
    suggested_plan = decision.get("suggested_plan")
    if complexity == "complex" and len(clean_sub_queries) >= 2:
        if suggested_plan and suggested_plan.get("strategy") == "complex_repair":
            plan = _normalize_complex_repair_plan(suggested_plan, clean_sub_queries)
        else:
            fallback_repair = (
                "query_variants"
                if next_retry_count == 1
                else decision.get("suggested_strategy") or "step_back"
            )
            plan = _normalize_complex_repair_plan(
                {
                    "strategy": "complex_repair",
                    "subqueries": [
                        {"query": item, "status": "unknown", "repair": fallback_repair}
                        for item in clean_sub_queries
                    ],
                },
                clean_sub_queries,
            )
        plan["queries"] = [query]
        logger.info("生成复杂问题纠正检索计划: %s", plan)
        return plan

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


async def _retrieve_complex_repair(
    vector_store: K12VectorStore,
    *,
    plan: dict,
    subject: str | None,
    grade: str | None,
    top_k: int,
) -> tuple[list[dict], list[str]]:
    """复杂问题纠正检索：按子问题分别修复，再保留子问题来源合并。"""
    sub_plans = list(plan.get("subqueries") or [])
    sub_queries = _dedupe_queries(plan.get("sub_queries") or [item.get("query", "") for item in sub_plans])
    if not sub_plans:
        sub_plans = [
            {"query": item, "status": "unknown", "repair": "direct"}
            for item in sub_queries
        ]
    per_sub_top_k = max(3, top_k // max(len(sub_plans), 1))
    per_sub_results: list[list[dict]] = []

    for sub_plan in sub_plans:
        sub_query = str(sub_plan.get("query", "")).strip()
        if not sub_query:
            continue
        repair = str(sub_plan.get("repair") or "direct")
        search_queries = await _build_repair_queries(sub_query, repair)
        query_results = []
        for search_query in search_queries:
            docs = _search(
                vector_store,
                query=search_query,
                subject=subject,
                grade=grade,
                top_k=per_sub_top_k,
                strategy="complex_repair",
            )
            for doc in docs:
                doc["source_sub_query"] = sub_query
                doc["query_variant"] = search_query
                doc["retry_repair_strategy"] = repair
            query_results.append(docs)
        fused = multi_query_fusion(query_results, per_sub_top_k) if len(query_results) > 1 else (query_results[0] if query_results else [])
        for doc in fused:
            doc["source_sub_query"] = sub_query
            doc["retrieval_strategy"] = "complex_repair"
            doc["retry_repair_strategy"] = repair
        per_sub_results.append(fused)

    docs = merge_sub_results(per_sub_results, top_k)
    logger.info(
        "复杂纠正检索完成: sub_queries=%d, sub_plans=%d, candidates=%d",
        len(sub_queries),
        len(sub_plans),
        len(docs),
    )
    return docs, sub_queries


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
    if plan.get("strategy") == "complex_repair":
        return await _retrieve_complex_repair(
            vector_store,
            plan=plan,
            subject=subject,
            grade=grade,
            top_k=limit,
        )
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


def _dedupe_queries(items: list[str]) -> list[str]:
    results: list[str] = []
    for item in items:
        query = str(item).strip()
        if query and query not in results:
            results.append(query)
    return results


def _normalize_complex_repair_plan(plan: dict, sub_queries: list[str]) -> dict:
    """补齐复杂修复计划，确保每个子问题都有明确 repair 动作。"""
    by_query = {
        str(item.get("query", "")).strip(): dict(item)
        for item in plan.get("subqueries", [])
        if str(item.get("query", "")).strip()
    }
    normalized_subqueries = []
    for query in sub_queries:
        item = by_query.get(query, {"query": query, "status": "unknown", "repair": "direct"})
        repair = item.get("repair") or "direct"
        if repair not in {"direct", "query_variants", "hyde", "step_back"}:
            repair = "step_back"
        normalized_subqueries.append(
            {
                "query": query,
                "status": item.get("status", "unknown"),
                "repair": repair,
            }
        )
    return {
        "strategy": "complex_repair",
        "sub_queries": sub_queries,
        "subqueries": normalized_subqueries,
    }


async def _build_repair_queries(query: str, repair: str) -> list[str]:
    """根据修复类型生成实际检索语句，始终保留原子问题兜底。"""
    candidates = [query]
    if repair == "hyde":
        hypothetical = await generate_hypothetical_answer(query)
        if hypothetical:
            candidates.append(hypothetical)
    elif repair == "step_back":
        step_back = await generate_step_back_query(query)
        if step_back:
            candidates.append(step_back)
    elif repair == "query_variants":
        variants = await generate_query_variants(query, n=2)
        candidates.extend(variants)
    return _dedupe_queries(candidates)
