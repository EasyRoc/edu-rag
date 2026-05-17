"""检索节点：根据查询复杂度执行不同策略的混合检索"""
from core.vectorestore import K12VectorStore
from core.strategies import (
    StrategyType,
    select_strategy,
    assess_retrieval_quality,
    should_apply_hyde,
    should_apply_step_back,
    generate_query_variants,
    multi_query_fusion,
    decompose_query,
    merge_sub_results,
    generate_step_back_query,
    generate_hypothetical_answer,
)
from utils.logger import logger


def _top_k_for(complexity: str) -> int:
    """根据复杂度返回检索数量"""
    top_k_map = {"simple": 3, "medium": 5, "complex": 8}
    return top_k_map.get(complexity, 5)


def _direct_retrieve(
    vector_store: K12VectorStore,
    query: str,
    complexity: str,
    subject: str | None = None,
    grade: str | None = None,
) -> list[dict]:
    """直接混合检索（无额外策略）"""
    return vector_store.hybrid_search(
        query=query, subject=subject, grade=grade, top_k=_top_k_for(complexity)
    )


async def _multi_query_retrieve(
    vector_store: K12VectorStore,
    query: str,
    complexity: str,
    subject: str | None = None,
    grade: str | None = None,
) -> list[dict]:
    """多查询策略：生成变体 → 多路检索 → RRF 融合"""
    variants = await generate_query_variants(query)
    if not variants:
        logger.info("多查询变体生成失败，降级为直接检索")
        return _direct_retrieve(vector_store, query, complexity, subject, grade)

    top_k = _top_k_for(complexity)
    all_queries = [query] + variants
    all_results = []
    for q in all_queries:
        results = vector_store.hybrid_search(
            query=q, subject=subject, grade=grade, top_k=top_k
        )
        all_results.append(results)

    return multi_query_fusion(all_results, top_k)


async def _decomposition_retrieve(
    vector_store: K12VectorStore,
    query: str,
    complexity: str,
    subject: str | None = None,
    grade: str | None = None,
) -> list[dict]:
    """复杂分解策略：拆解子问题 → 分别检索 → 合并去重"""
    sub_queries = await decompose_query(query)
    if len(sub_queries) <= 1:
        logger.info("问题分解未产生多个子问题，降级为直接检索")
        return _direct_retrieve(vector_store, query, complexity, subject, grade)

    top_k = _top_k_for(complexity)
    sub_results = []
    for sq in sub_queries:
        results = vector_store.hybrid_search(
            query=sq, subject=subject, grade=grade, top_k=max(3, top_k // 2)
        )
        sub_results.append(results)

    return merge_sub_results(sub_results, top_k)


async def _apply_supplementary(
    vector_store: K12VectorStore,
    docs: list[dict],
    query: str,
    complexity: str,
    subject: str | None = None,
    grade: str | None = None,
) -> list[dict]:
    """应用补充策略（Step-Back / HyDE）提升检索质量"""
    top_k = _top_k_for(complexity)
    supplements = []

    # HyDE 补充（适合定义/事实类查询得分低时）
    if should_apply_hyde(docs):
        hyde_answer = await generate_hypothetical_answer(query)
        if hyde_answer:
            hyde_results = vector_store.hybrid_search(
                query=hyde_answer, subject=subject, grade=grade, top_k=top_k
            )
            supplements.append(hyde_results)
            logger.info(f"HyDE 补充检索: {len(hyde_results)} 条结果")

    # Step-Back 补充（适合结果少/平均分低时）
    if should_apply_step_back(docs):
        step_back_query = await generate_step_back_query(query)
        if step_back_query:
            sb_results = vector_store.hybrid_search(
                query=step_back_query, subject=subject, grade=grade, top_k=top_k
            )
            supplements.append(sb_results)
            logger.info(f"Step-Back 补充检索: {len(sb_results)} 条结果")

    if not supplements:
        return docs

    # 将原结果和补充结果用 RRF 融合
    all_results = [docs] + supplements
    return multi_query_fusion(all_results, top_k)


async def hybrid_retrieve(
    vector_store: K12VectorStore,
    query: str,
    complexity: str,
    intent: str = "educational",
    subject: str | None = None,
    grade: str | None = None,
) -> list[dict]:
    """
    策略驱动的混合检索。

    根据意图和复杂度选择策略：
    - simple → DIRECT 直接检索
    - medium → MULTI_QUERY 多查询变体 + RRF 融合
    - complex → DECOMPOSITION 复杂问题拆解

    首轮检索后评估质量，必要时触发 Step-Back / HyDE 补充。
    """
    strategy = select_strategy(intent, complexity, query)
    logger.info(f"检索策略: {strategy.value}, complexity={complexity}, intent={intent}")

    # 执行主策略
    if strategy == StrategyType.DIRECT:
        docs = _direct_retrieve(vector_store, query, complexity, subject, grade)
    elif strategy == StrategyType.MULTI_QUERY:
        docs = await _multi_query_retrieve(vector_store, query, complexity, subject, grade)
    elif strategy == StrategyType.DECOMPOSITION:
        docs = await _decomposition_retrieve(vector_store, query, complexity, subject, grade)
    else:
        docs = _direct_retrieve(vector_store, query, complexity, subject, grade)

    # 评估质量，必要时补充
    if not assess_retrieval_quality(docs):
        logger.info("检索质量不足，尝试补充策略...")
        docs = await _apply_supplementary(vector_store, docs, query, complexity, subject, grade)

    logger.info(f"检索完成: 策略={strategy.value}, 最终 {len(docs)} 条结果")
    for i, doc in enumerate(docs):
        logger.debug(f"  结果[{i}]: score={doc['score']:.4f}, source={doc.get('_source', '?')}, "
                     f"text={doc['text'][:60]}...")

    return docs
