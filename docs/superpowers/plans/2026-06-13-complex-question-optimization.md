# 复杂问题 RAG 链路优化 Phase 1 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复复杂问题链路中 reranker 误伤子问题结果、缺少子答案中间推理、门控阈值不区分复杂度的三个核心问题，提升 complex query 的检索覆盖率和生成质量。

**Architecture:** 三阶段改造——(1) 分级门控阈值让复杂问题更宽松 accept，(2) 子问题感知重排改为两阶段 rerank（子问题独立 rerank + 原问题二次 rerank），(3) 子答案合成生成改为子答案并行生成 + synthesis 综合。所有变更通过 `complexity == "complex"` 分支隔离，不触动 simple/medium 路径。

**Tech Stack:** Python 3.13, LangGraph, sentence-transformers CrossEncoder, RAGAS 0.4.x

---

## File Structure

```
修改:
  config.py                          — 新增 6 个配置项
  core/retrieval_quality.py          — evaluate_retrieval_gate 新增 complexity 参数
  core/strategies/decomposition.py   — merge_sub_results 标注子问题来源，携带 sub_queries
  core/nodes/retriever.py            — _decomposition_retrieve 返回 sub_queries 到 state
  core/state.py                      — RAGState 新增 sub_queries 字段
  core/graph.py                      — rerank_node / generate_node / retrieval_gate_node 感知复杂度
  core/nodes/generator.py            — 新增 sub_answer_synthesis 函数

新增:
  test/test_complex_question.py      — 单元 + 集成测试
```

---

### Task 1: Config — 新增复杂问题相关配置项

**Files:**
- Modify: `config.py:70-75`

- [ ] **Step 1: 在 config.py 新增配置项**

在 `DECOMPOSITION_MAX_SUB` 下方插入：

```python
    # ---------- 复杂问题深度处理 ----------
    ENABLE_DEEP_COMPLEX_MODE: bool = os.getenv("ENABLE_DEEP_COMPLEX_MODE", "true").lower() in {"1", "true", "yes", "on"}

    # 复杂问题 rerank: 第一阶段各子问题独立 rerank 后取 top K，合并后再用原问题 rerank
    SUB_RERANK_TOP_K: int = int(os.getenv("SUB_RERANK_TOP_K", "6"))

    # 复杂问题门控阈值（比默认值宽松）
    COMPLEX_ACCEPT_TOP1_THRESHOLD: float = float(os.getenv("COMPLEX_ACCEPT_TOP1_THRESHOLD", "0.45"))
    COMPLEX_RELEVANCE_THRESHOLD: float = float(os.getenv("COMPLEX_RELEVANCE_THRESHOLD", "0.35"))
    COMPLEX_MAX_RETRIES: int = int(os.getenv("COMPLEX_MAX_RETRIES", "2"))

    # 子答案合成
    SUB_ANSWER_MAX_TOKENS: int = int(os.getenv("SUB_ANSWER_MAX_TOKENS", "512"))
    SYNTHESIS_MAX_TOKENS: int = int(os.getenv("SYNTHESIS_MAX_TOKENS", "4096"))
    COMPLEX_CONTEXT_TOP_K: int = int(os.getenv("COMPLEX_CONTEXT_TOP_K", "8"))
```

- [ ] **Step 2: 确认配置可导入**

```bash
cd /Users/zhouqiantalaogong/Downloads/cursor_project/edu-rag && python -c "from config import settings; print(settings.ENABLE_DEEP_COMPLEX_MODE, settings.COMPLEX_ACCEPT_TOP1_THRESHOLD)"
```
Expected: `True 0.45`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add complex question config entries (gate thresholds, sub-rerank, synthesis params)"
```

---

### Task 2: Retrieval Quality — 分级门控阈值 (Req 6)

**Files:**
- Modify: `core/retrieval_quality.py:88-137` (evaluate_retrieval_gate 签名 + 阈值选择)
- Modify: `core/graph.py:108-115` (retrieval_gate_node 传递 complexity)
- Test: `test/test_complex_question.py` (新增)

- [ ] **Step 1: 编写门控分级阈值测试**

```python
# test/test_complex_question.py
import pytest
from core.retrieval_quality import evaluate_retrieval_gate


def _make_docs(*scores: float) -> list[dict]:
    return [{"id": i, "text": f"doc{i}", "rerank_score": s} for i, s in enumerate(scores)]


class TestComplexityGradedGate:
    def test_complex_accepts_lower_top1_than_default(self):
        """复杂问题 top1=0.50: 低于默认 0.60 但高于 complex 0.45 → accept"""
        docs = _make_docs(0.50, 0.40, 0.30)
        decision = evaluate_retrieval_gate(docs, complexity="complex")
        assert decision["action"] == "accept"

    def test_complex_rejects_below_complex_threshold(self):
        """复杂问题 top1=0.30: 低于 complex 0.45 → retry/abstain"""
        docs = _make_docs(0.30, 0.25)
        decision = evaluate_retrieval_gate(docs, complexity="complex", retry_count=2, max_retries=2)
        assert decision["action"] == "abstain"

    def test_simple_uses_strict_threshold(self):
        """简单问题 top1=0.50: 低于默认 0.60 但简单问题不走宽阈值 → retry/abstain"""
        docs = _make_docs(0.50, 0.40)
        decision = evaluate_retrieval_gate(docs, complexity="simple", retry_count=2, max_retries=2)
        assert decision["action"] == "abstain"

    def test_medium_unchanged(self):
        """中等问题的门控行为不变"""
        docs = _make_docs(0.65, 0.55)
        decision = evaluate_retrieval_gate(docs, complexity="medium")
        assert decision["action"] == "accept"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /Users/zhouqiantalaogong/Downloads/cursor_project/edu-rag && python -m pytest test/test_complex_question.py::TestComplexityGradedGate -v
```
Expected: FAIL — `evaluate_retrieval_gate() got an unexpected keyword argument 'complexity'`

- [ ] **Step 3: 修改 evaluate_retrieval_gate 签名和阈值选择**

修改 `core/retrieval_quality.py:88-97`：

```python
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
```

修改阈值选择逻辑，在 `core/retrieval_quality.py:126-137` 替换原有 `relevant_min` 和 `top1_min` 的计算：

```python
    mode = gate_mode or settings.RETRIEVAL_GATE_MODE

    # 复杂度分级阈值：complex 使用更宽松的阈值
    if complexity == "complex":
        _default_relevant = settings.COMPLEX_RELEVANCE_THRESHOLD
        _default_top1 = settings.COMPLEX_ACCEPT_TOP1_THRESHOLD
        _default_max_retries = settings.COMPLEX_MAX_RETRIES
    else:
        _default_relevant = settings.RERANKER_RELEVANCE_THRESHOLD
        _default_top1 = settings.RETRIEVAL_ACCEPT_TOP1_THRESHOLD
        _default_max_retries = settings.MAX_RETRIES

    relevant_min = _default_relevant if relevant_threshold is None else relevant_threshold
    top1_min = _default_top1 if accept_top1_threshold is None else accept_top1_threshold
    if max_retries == 2 and complexity == "complex":
        max_retries = _default_max_retries

    metrics = compute_retrieval_metrics(docs, relevant_threshold=relevant_min)
```

- [ ] **Step 4: 修改 graph.py retrieval_gate_node 传递 complexity**

修改 `core/graph.py:108-115`：

```python
async def retrieval_gate_node(state: RAGState) -> dict:
    decision = evaluate_retrieval_gate(
        state.get("retrieved_docs", []),
        retry_count=state.get("retry_count", 0),
        max_retries=state.get("max_retries", settings.MAX_RETRIES),
        reranker_available=state.get("reranker_available", False),
        complexity=state.get("complexity", "medium"),
    )
```

- [ ] **Step 5: 运行测试确认通过**

```bash
cd /Users/zhouqiantalaogong/Downloads/cursor_project/edu-rag && python -m pytest test/test_complex_question.py::TestComplexityGradedGate -v
```
Expected: 4 PASS

- [ ] **Step 6: Commit**

```bash
git add core/retrieval_quality.py core/graph.py test/test_complex_question.py
git commit -m "feat: add complexity-graded retrieval gate thresholds"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 3: Decomposition — 子问题来源标注 + 携带 sub_queries (Req 1 前置)

**Files:**
- Modify: `core/strategies/decomposition.py:45-59` (merge_sub_results 标注来源)
- Modify: `core/state.py:9-31` (RAGState 新增 sub_queries)
- Modify: `core/nodes/retriever.py:103-129` (_decomposition_retrieve 返回 sub_queries)

- [ ] **Step 1: 编写测试**

在 `test/test_complex_question.py` 追加：

```python
class TestSubQueryAnnotation:
    def test_merge_annotates_sub_query_source(self):
        """merge_sub_results 给每个 doc 标注 source_sub_query"""
        from core.strategies.decomposition import merge_sub_results

        sub1 = [{"id": 1, "text": "a", "score": 0.9, "source_sub_query": "子问题A"}]
        sub2 = [{"id": 2, "text": "b", "score": 0.8, "source_sub_query": "子问题B"}]

        merged = merge_sub_results([sub1, sub2], top_k=10)
        sources = {doc.get("source_sub_query") for doc in merged}
        assert "子问题A" in sources
        assert "子问题B" in sources

    def test_state_has_sub_queries_field(self):
        """RAGState 包含 sub_queries 字段"""
        from core.state import RAGState
        state: RAGState = {
            "query": "test", "subject": None, "grade": None,
            "session_id": "s1", "intent": "educational", "complexity": "complex",
            "retrieved_docs": [], "answer": "", "retry_count": 0, "max_retries": 2,
            "conversation_history": [], "retrieval_plan": {}, "retrieval_attempts": [],
            "retrieval_metrics": {}, "retrieval_decision": {},
            "abstain_reason": "", "retrieval_latency_ms": 0.0, "rerank_latency_ms": 0.0,
            "reranker_available": True, "_queue_id": "",
            "sub_queries": [],
        }
        assert state["sub_queries"] == []
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /Users/zhouqiantalaogong/Downloads/cursor_project/edu-rag && python -m pytest test/test_complex_question.py::TestSubQueryAnnotation -v
```
Expected: FAIL — `merge_sub_results` doesn't preserve `source_sub_query`, `sub_queries` key not in RAGState

- [ ] **Step 3: 修改 RAGState 新增 sub_queries**

修改 `core/state.py:9-31`，在 `reranker_available` 下方插入：

```python
    reranker_available: bool
    sub_queries: list[str]          # DECOMPOSITION 拆解出的子问题列表
    _queue_id: str
```

- [ ] **Step 4: 修改 merge_sub_results 不覆盖 source_sub_query**

修改 `core/strategies/decomposition.py:45-59`，替换 `merge_sub_results`：

```python
def merge_sub_results(sub_results: list[list[dict]], top_k: int) -> list[dict]:
    """合并子问题检索结果，按 score 去重排序。
    
    保留每个 doc 的 source_sub_query 字段，用于后续子问题感知重排。
    """
    if not sub_results:
        return []

    seen: dict[int, dict] = {}
    for result_list in sub_results:
        for doc in result_list:
            chunk_key = doc.get("id", 0)
            if chunk_key not in seen or doc["score"] > seen[chunk_key]["score"]:
                seen[chunk_key] = dict(doc)
            # 若相同 chunk 被多个子问题检索到，追加 source_sub_query
            elif chunk_key in seen and doc.get("source_sub_query"):
                existing_sources = seen[chunk_key].get("source_sub_query", "")
                new_source = doc.get("source_sub_query", "")
                if new_source and new_source not in existing_sources:
                    seen[chunk_key]["source_sub_query"] = f"{existing_sources}; {new_source}"

    merged = sorted(seen.values(), key=lambda d: d["score"], reverse=True)
    logger.info(f"子结果合并: {len(sub_results)} 组 → {len(merged)} 条（去重后）")
    return merged[:top_k]
```

- [ ] **Step 5: 修改 _decomposition_retrieve 标注来源并返回 sub_queries**

修改 `core/nodes/retriever.py:103-129` 中 `_decomposition_retrieve`：

```python
async def _decomposition_retrieve(
    vector_store: K12VectorStore,
    query: str,
    complexity: str,
    subject: str | None = None,
    grade: str | None = None,
    *,
    top_k: int | None = None,
) -> tuple[list[dict], list[str]]:
    """复杂问题拆解召回：子问题分别检索，再合并去重。
    
    Returns:
        (docs, sub_queries): 合并后的候选列表 + 拆解出的子问题列表
    """
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
        # 标注来源子问题
        for doc in docs:
            doc["source_sub_query"] = item
        results.append(docs)

    merged = merge_sub_results(results, limit)
    return (
        _annotate(merged, StrategyType.DECOMPOSITION.value, query),
        sub_queries,
    )
```

修改 `hybrid_retrieve` 中调用 `_decomposition_retrieve` 的地方，`core/nodes/retriever.py:247-248`：

```python
    else:
        docs, sub_queries = await _decomposition_retrieve(
            vector_store, query, complexity, subject, grade, top_k=limit
        )
```

并在 `hybrid_retrieve` 返回时将 sub_queries 也返回。修改函数签名和返回值，`core/nodes/retriever.py:216-250`：

```python
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
    """召回候选文档；在线质量判断会在重排后完成。
    
    Returns:
        (docs, sub_queries): 候选文档列表 + 拆解出的子问题列表（非 DECOMPOSITION 策略返回空列表）
    """
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
```

`_direct_retrieve` 和 `_multi_query_retrieve` 返回值不动，`hybrid_retrieve` 在调用它们时自己包一层 `([docs], [])`。需要对这两个函数的调用做适配：

```python
    if strategy == StrategyType.DIRECT:
        docs = _direct_retrieve(vector_store, query, complexity, subject, grade, top_k=limit)
    elif strategy == StrategyType.MULTI_QUERY:
        docs = await _multi_query_retrieve(vector_store, query, complexity, subject, grade, top_k=limit)
    else:
        docs, sub_queries = await _decomposition_retrieve(
            vector_store, query, complexity, subject, grade, top_k=limit
        )
```

- [ ] **Step 6: 修改 graph.py retrieve_node 解包并写入 state**

修改 `core/graph.py:57-80` 的 `retrieve_node`：

```python
async def retrieve_node(state: RAGState, vector_store: K12VectorStore) -> dict:
    started = time.perf_counter()
    docs, sub_queries = await hybrid_retrieve(
        vector_store=vector_store,
        query=state["query"],
        complexity=state["complexity"],
        intent=state.get("intent", "educational"),
        subject=state.get("subject"),
        grade=state.get("grade"),
        retrieval_plan=state.get("retrieval_plan"),
        candidate_top_k=settings.RETRIEVAL_CANDIDATE_TOP_K,
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    logger.info(
        "retrieve: plan=%s, candidates=%d, sub_queries=%d, latency_ms=%.3f",
        state.get("retrieval_plan", {}).get("strategy", "initial"),
        len(docs),
        len(sub_queries),
        latency_ms,
    )
    return {
        "retrieved_docs": docs,
        "retrieval_latency_ms": latency_ms,
        "sub_queries": sub_queries,
    }
```

同时修改 `build_retry_plan` 调用处的返回值适配（`retriever.py:216-240` 的纠正检索分支 `return docs, []` 已经处理）。

检查 `evaluation/retrieval_evaluator.py:156-165` 中 `hybrid_retrieve` 的调用处，解包：

```python
docs, _sub_queries = await hybrid_retrieve(
    vector_store=vector_store,
    ...
)
```

- [ ] **Step 7: 运行测试确认通过**

```bash
cd /Users/zhouqiantalaogong/Downloads/cursor_project/edu-rag && python -m pytest test/test_complex_question.py::TestSubQueryAnnotation -v
```
Expected: 2 PASS

- [ ] **Step 8: Commit**

```bash
git add core/strategies/decomposition.py core/state.py core/nodes/retriever.py core/graph.py evaluation/retrieval_evaluator.py test/test_complex_question.py
git commit -m "feat: annotate sub-query source in decomposition, carry sub_queries in graph state"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 4: Graph — 子问题感知两阶段重排 (Req 1 核心)

**Files:**
- Modify: `core/graph.py:83-105` (rerank_node 分支: complex 走两阶段 rerank)
- Test: `test/test_complex_question.py` (追加)

- [ ] **Step 1: 编写两阶段 rerank 测试**

在 `test/test_complex_question.py` 追加：

```python
import asyncio

class TestTwoStageRerank:
    def test_two_stage_rerank_preserves_sub_query_sources(self):
        """两阶段重排后，来自不同子问题的 doc 仍然保留 source_sub_query"""
        from core.reranker import CrossEncoderReranker

        docs = [
            {"id": 1, "text": "勾股定理定义", "score": 0.9, "source_sub_query": "勾股定理是什么"},
            {"id": 2, "text": "相似三角形判定", "score": 0.85, "source_sub_query": "相似三角形判定定理"},
            {"id": 3, "text": "勾股定理应用例题", "score": 0.8, "source_sub_query": "勾股定理的例题"},
        ]
        sub_queries = ["勾股定理是什么", "相似三角形判定定理", "勾股定理的例题"]

        async def run_two_stage():
            reranker = CrossEncoderReranker()
            # Stage 1: 各子问题独立 rerank
            staged = {}
            for sq in sub_queries:
                sq_docs = [d for d in docs if d.get("source_sub_query") == sq]
                if sq_docs:
                    staged[sq] = await reranker.rerank(sq, sq_docs)

            # Stage 2: 合并去重后用原问题再 rerank
            seen_ids = set()
            merged = []
            for sq_docs in staged.values():
                for doc in sq_docs[:2]:  # 每子问题取 top 2
                    if doc["id"] not in seen_ids:
                        seen_ids.add(doc["id"])
                        merged.append(doc)

            original_query = "比较勾股定理和相似三角形的异同"
            final = await reranker.rerank(original_query, merged)
            return final

        final = asyncio.run(run_two_stage())
        assert len(final) > 0
        # 两个子方向的文档都应存在
        sources = {d.get("source_sub_query", "") for d in final}
        assert any("勾股定理" in s for s in sources)
        assert any("相似三角形" in s for s in sources)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /Users/zhouqiantalaogong/Downloads/cursor_project/edu-rag && python -m pytest test/test_complex_question.py::TestTwoStageRerank -v
```
Expected: PASS (测试逻辑自包含，应直接通过)

- [ ] **Step 3: 修改 graph.py rerank_node 加入两阶段逻辑**

修改 `core/graph.py:83-105` 的 `rerank_node`：

```python
async def rerank_node(state: RAGState, reranker: CrossEncoderReranker) -> dict:
    """本地 CrossEncoder 重排节点。
    
    DECOMPOSITION (complex) 策略走两阶段重排:
      Stage 1: 各子问题独立 rerank → 取 top SUB_RERANK_TOP_K
      Stage 2: 合并去重 → 用原问题 rerank
    其他策略走单阶段重排。
    """
    started = time.perf_counter()
    docs = state.get("retrieved_docs", [])
    sub_queries = state.get("sub_queries", [])
    complexity = state.get("complexity", "medium")

    try:
        if (
            complexity == "complex"
            and sub_queries
            and len(sub_queries) >= 2
            and settings.ENABLE_DEEP_COMPLEX_MODE
        ):
            docs = await _two_stage_rerank(
                state["query"], docs, sub_queries, reranker
            )
        else:
            docs = await reranker.rerank(state["query"], docs)
        available = True
    except RerankerUnavailableError as exc:
        logger.warning("本地重排不可用: %s", exc)
        docs = list(docs)
        available = False

    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    logger.info(
        "rerank: available=%s, docs=%d, top1=%.4f, latency_ms=%.3f",
        available,
        len(docs),
        docs[0].get("rerank_score", 0.0) if docs else 0.0,
        latency_ms,
    )
    return {
        "retrieved_docs": docs,
        "reranker_available": available,
        "rerank_latency_ms": latency_ms,
    }


async def _two_stage_rerank(
    original_query: str,
    docs: list[dict],
    sub_queries: list[str],
    reranker: CrossEncoderReranker,
) -> list[dict]:
    """两阶段重排: 子问题独立 rerank → 合并 → 原问题 rerank。"""
    from config import settings

    sub_top_k = settings.SUB_RERANK_TOP_K

    # Stage 1: 按 source_sub_query 分组，各组独立 rerank
    staged: dict[str, list[dict]] = {}
    for doc in docs:
        source = doc.get("source_sub_query", "")
        if not source:
            source = "__no_source__"
        staged.setdefault(source, []).append(doc)

    stage1_results: list[dict] = []
    for source_query, group_docs in staged.items():
        if source_query == "__no_source__":
            reranked = await reranker.rerank(original_query, group_docs)
        else:
            reranked = await reranker.rerank(source_query, group_docs)
        stage1_results.extend(reranked[:sub_top_k])

    # Stage 2: 合并去重后用原问题 rerank
    seen_ids: set[int] = set()
    merged: list[dict] = []
    for doc in sorted(stage1_results, key=lambda d: d.get("rerank_score", 0), reverse=True):
        doc_id = doc.get("id", 0)
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            merged.append(doc)

    logger.info(
        "两阶段重排: stage1_groups=%d, stage1_docs=%d, merged=%d",
        len(staged), len(stage1_results), len(merged),
    )
    return await reranker.rerank(original_query, merged)
```

- [ ] **Step 4: 运行完整测试确认通过**

```bash
cd /Users/zhouqiantalaogong/Downloads/cursor_project/edu-rag && python -m pytest test/test_complex_question.py -v
```
Expected: 7 PASS (4 from Task 2 + 2 from Task 3 + 1 from Task 4)

- [ ] **Step 5: Commit**

```bash
git add core/graph.py test/test_complex_question.py
git commit -m "feat: add two-stage sub-query-aware rerank for complex questions"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 5: Generator — 子答案合成生成 (Req 2)

**Files:**
- Modify: `core/nodes/generator.py` (新增 sub_answer_synthesis 函数)
- Modify: `core/graph.py:153-166` (generate_node 分支: complex 走 synthesis)

- [ ] **Step 1: 编写子答案合成测试**

在 `test/test_complex_question.py` 追加：

```python
class TestSubAnswerSynthesis:
    def test_sub_answer_prompt_structure(self):
        """子答案生成 prompt 包含子问题和上下文"""
        from core.nodes.generator import _build_sub_answer_prompt

        prompt = _build_sub_answer_prompt(
            sub_query="勾股定理是什么",
            context_docs=[{"text": "勾股定理是直角三角形斜边平方等于两直角边平方和"}],
        )
        assert "勾股定理是什么" in prompt
        assert "直角三角形" in prompt

    def test_synthesis_prompt_includes_sub_answers(self):
        """综合 prompt 包含子答案作为中间层"""
        from core.nodes.generator import _build_synthesis_prompt

        sub_answers = [
            ("勾股定理是什么", "勾股定理是a²+b²=c²"),
            ("相似三角形判定", "SSS/SAS/AA三种判定方法"),
        ]
        prompt = _build_synthesis_prompt(
            original_query="比较勾股定理和相似三角形",
            sub_answers=sub_answers,
            context_docs=[{"text": "勾股定理定义"}, {"text": "相似三角形判定"}],
        )
        assert "a²+b²=c²" in prompt
        assert "SSS/SAS/AA" in prompt
        assert "比较勾股定理和相似三角形" in prompt
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /Users/zhouqiantalaogong/Downloads/cursor_project/edu-rag && python -m pytest test/test_complex_question.py::TestSubAnswerSynthesis -v
```
Expected: FAIL — `_build_sub_answer_prompt` not defined

- [ ] **Step 3: 在 generator.py 新增子答案合成函数**

在 `core/nodes/generator.py` 末尾 `_mock_answer` 之后追加：

```python
# ======================================================================
# 复杂问题子答案合成 (Sub-Answer Synthesis)
# ======================================================================

SUB_ANSWER_SYSTEM = """你是一个 K12 教育助手。请根据上下文简要回答子问题。
只输出答案，不要引用来源编号，不要额外解释。"""

SUB_ANSWER_PROMPT = """## 子问题
{sub_query}

## 参考上下文
{context}

请简要回答："""

SYNTHESIS_SYSTEM = """你是一个专业的 K12 教育助手，名叫"知学助手"。
请基于子问题分析和参考资料，综合回答学生的原始问题。

## 要求
1. 综合利用子问题答案和原始上下文
2. 回答要全面覆盖原问题的所有角度
3. 简明易懂，适合 K12 学生的认知水平
4. 适当举例说明
5. 在回答末尾标注引用的参考来源序号（如 [1][2]）
6. 参考资料不足时明确说明"""

SYNTHESIS_PROMPT = """## 原始问题
{original_query}

## 子问题分析
{sub_answers_section}

## 参考资料
{context}

请综合回答原始问题："""


def _build_sub_answer_prompt(sub_query: str, context_docs: list[dict]) -> str:
    context_parts = []
    for i, doc in enumerate(context_docs):
        context_parts.append(f"[{i+1}] {doc['text']}")
    return SUB_ANSWER_PROMPT.format(
        sub_query=sub_query,
        context="\n\n".join(context_parts),
    )


def _build_synthesis_prompt(
    original_query: str,
    sub_answers: list[tuple[str, str]],
    context_docs: list[dict],
) -> str:
    parts = []
    for i, (sq, ans) in enumerate(sub_answers, 1):
        parts.append(f"### 子问题{i}: {sq}\n答案: {ans}")
    sub_answers_section = "\n\n".join(parts)

    ctx_parts = []
    for i, doc in enumerate(context_docs):
        ctx_parts.append(f"[{i+1}] {doc['text']}")

    return SYNTHESIS_PROMPT.format(
        original_query=original_query,
        sub_answers_section=sub_answers_section,
        context="\n\n".join(ctx_parts),
    )


async def generate_sub_answers(
    sub_queries: list[str],
    sub_docs_map: dict[str, list[dict]],
) -> list[tuple[str, str]]:
    """并行生成各子问题的中间答案。
    
    Args:
        sub_queries: 子问题列表
        sub_docs_map: {子问题文本: [关联docs]}
    
    Returns:
        [(子问题, 子答案), ...]，失败的子问题答案为空字符串
    """
    if not settings.LLM_API_KEY:
        return [(sq, "") for sq in sub_queries]

    async def _answer_one(sq: str) -> tuple[str, str]:
        docs = sub_docs_map.get(sq, [])[:3]
        prompt = _build_sub_answer_prompt(sq, docs)
        try:
            llm = get_chat_model(
                temperature=0.1,
                max_tokens=settings.SUB_ANSWER_MAX_TOKENS,
                timeout=30.0,
            )
            from langchain_core.messages import HumanMessage, SystemMessage
            response = await llm.ainvoke([
                SystemMessage(content=SUB_ANSWER_SYSTEM),
                HumanMessage(content=prompt),
            ])
            return (sq, response.content.strip())
        except Exception as e:
            logger.warning("子答案生成失败 [%s]: %s", sq[:30], e)
            return (sq, "")

    tasks = [_answer_one(sq) for sq in sub_queries]
    results = await asyncio.gather(*tasks)
    logger.info("子答案生成完成: %d/%d 成功", sum(1 for _, a in results if a), len(results))
    return results


async def synthesize_final_answer(
    original_query: str,
    sub_answers: list[tuple[str, str]],
    context_docs: list[dict],
) -> str:
    """基于子答案和原始上下文，综合生成最终答案。
    
    若 LLM 不可用，降级为直接 mock 回答。
    """
    if not settings.LLM_API_KEY:
        return _mock_answer(original_query, context_docs)

    prompt = _build_synthesis_prompt(original_query, sub_answers, context_docs)
    try:
        llm = get_chat_model(
            temperature=0.3,
            max_tokens=settings.SYNTHESIS_MAX_TOKENS,
            timeout=120.0,
        )
        from langchain_core.messages import HumanMessage, SystemMessage
        response = await llm.ainvoke([
            SystemMessage(content=SYNTHESIS_SYSTEM),
            HumanMessage(content=prompt),
        ])
        return response.content.strip()
    except Exception as e:
        logger.error("综合答案生成失败: %s，降级为直接生成", e)
        return await llm_generate(original_query, context_docs)
```

在 generator.py 顶部新增 import：

```python
import asyncio
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /Users/zhouqiantalaogong/Downloads/cursor_project/edu-rag && python -m pytest test/test_complex_question.py::TestSubAnswerSynthesis -v
```
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add core/nodes/generator.py test/test_complex_question.py
git commit -m "feat: add sub-answer synthesis for complex question generation"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 6: Graph — generate_node 分支 complex 走 synthesis 路径 (Req 2 集成)

**Files:**
- Modify: `core/graph.py:153-166` (generate_node)
- Test: `test/test_complex_question.py` (追加集成测试)

- [ ] **Step 1: 编写集成测试**

在 `test/test_complex_question.py` 追加：

```python
class TestComplexGenerateBranch:
    def test_synthesis_path_branch_condition(self):
        """验证 synthesis 路径触发条件：complex + sub_queries >= 2 + 开关开启"""
        # 条件满足 → 走 synthesis
        assert _should_use_synthesis("complex", ["q1", "q2", "q3"], True) is True
        # 中等复杂度 → 不走
        assert _should_use_synthesis("medium", ["q1", "q2"], True) is False
        # 子问题不足 → 不走
        assert _should_use_synthesis("complex", ["q1"], True) is False
        # 开关关闭 → 不走
        assert _should_use_synthesis("complex", ["q1", "q2"], False) is False


def _should_use_synthesis(complexity: str, sub_queries: list[str], deep_mode_enabled: bool) -> bool:
    return (
        complexity == "complex"
        and len(sub_queries) >= 2
        and deep_mode_enabled
    )
```

- [ ] **Step 2: 修改 generate_node**

修改 `core/graph.py:153-166`：

```python
async def generate_node(state: RAGState) -> dict:
    """答案生成节点。
    
    complex + 有 sub_queries 时走子答案合成路径，其他走直接生成。
    """
    from core.nodes.generator import generate_sub_answers, synthesize_final_answer

    complexity = state.get("complexity", "medium")
    sub_queries = state.get("sub_queries", [])
    full_answer = ""
    queue_id = state.get("_queue_id")

    if _should_use_synthesis(complexity, sub_queries, settings.ENABLE_DEEP_COMPLEX_MODE):
        # 复杂问题: 子答案合成路径
        all_docs = state.get("retrieved_docs", [])
        context_k = settings.COMPLEX_CONTEXT_TOP_K

        # 按 source_sub_query 分组文档
        sub_docs_map: dict[str, list[dict]] = {sq: [] for sq in sub_queries}
        for doc in all_docs:
            source = doc.get("source_sub_query", "")
            if source in sub_docs_map:
                sub_docs_map[source].append(doc)

        # 并行生成子答案
        sub_answers = await generate_sub_answers(sub_queries, sub_docs_map)

        # 综合生成最终答案
        context_docs = all_docs[:context_k]
        full_answer = await synthesize_final_answer(
            state["query"], sub_answers, context_docs
        )

        # 将完整答案一次性推入流队列
        await stream_queues.emit(queue_id, full_answer)

        logger.info(
            "generate (synthesis): sub_answers=%d, context_docs=%d, answer_chars=%d",
            len(sub_answers), len(context_docs), len(full_answer),
        )
        return {"answer": full_answer, "retrieved_docs": context_docs}

    # 简单/中等问题: 直接流式生成（原路径）
    docs = state.get("retrieved_docs", [])[: settings.GENERATION_CONTEXT_TOP_K]
    async for token in llm_generate_stream(
        query=state["query"],
        context_docs=docs,
        conversation_history=state.get("conversation_history", []),
    ):
        full_answer += token
        await stream_queues.emit(queue_id, token)
    logger.info("generate: context_docs=%d, answer_chars=%d", len(docs), len(full_answer))
    return {"answer": full_answer, "retrieved_docs": docs}


def _should_use_synthesis(complexity: str, sub_queries: list[str], deep_mode_enabled: bool) -> bool:
    """判断是否走子答案合成路径：complex + 至少有 2 个子问题 + 功能开关开启。"""
    return (
        complexity == "complex"
        and len(sub_queries) >= 2
        and deep_mode_enabled
    )
```

- [ ] **Step 3: 运行现有测试确认不退化**

```bash
cd /Users/zhouqiantalaogong/Downloads/cursor_project/edu-rag && python -m pytest test/test_complex_question.py test/test_graph_v1.py -v
```
Expected: 现有 graph 测试仍然 PASS

- [ ] **Step 4: Commit**

```bash
git add core/graph.py test/test_complex_question.py
git commit -m "feat: wire sub-answer synthesis into graph generate_node for complex queries"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 7: 端到端回归验证

**Files:**
- Test: `test/test_complex_question.py` (追加)

- [ ] **Step 1: 运行全量单元测试**

```bash
cd /Users/zhouqiantalaogong/Downloads/cursor_project/edu-rag && python -m pytest test/ -v --ignore=test/test_auto_eval_samples.py
```
Expected: 全部 PASS，无回归

- [ ] **Step 2: 运行 RAGAS 离线评估对比**

```bash
cd /Users/zhouqiantalaogong/Downloads/cursor_project/edu-rag && python evaluation/cli.py evaluate --from-db --limit 30 --metrics faithfulness,answer_relevancy
```
Expected: 评估正常运行，无报错

- [ ] **Step 3: 运行检索离线评估确认门控兼容**

```bash
cd /Users/zhouqiantalaogong/Downloads/cursor_project/edu-rag && python -c "
from evaluation.retrieval_evaluator import evaluate_retrieval_gate
# 验证旧接口兼容
result = evaluate_retrieval_gate([{'id': 1, 'text': 'test', 'rerank_score': 0.7}])
assert result['action'] == 'accept', f'unexpected: {result[\"action\"]}'
print('OK: retrieval_evaluator 兼容')
"
```
Expected: `OK: retrieval_evaluator 兼容`

- [ ] **Step 4: Commit**

```bash
git add test/test_complex_question.py
git commit -m "test: add end-to-end regression verification for complex question optimization"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

## Verification Checklist

- [ ] `test/test_complex_question.py` 全部通过 (≥ 9 tests)
- [ ] 现有 `test/test_graph_v1.py` 零回归
- [ ] 现有 `test/test_retrieval_v1.py` 零回归
- [ ] 现有 `test/test_strategies.py` 零回归
- [ ] `evaluation/cli.py evaluate --from-db` 正常运行
- [ ] retrieval_evaluator 兼容旧接口
- [ ] `ENABLE_DEEP_COMPLEX_MODE=false` 时 complex 照旧走单阶段 rerank + 直接生成
- [ ] 手工测试: 发送 150 字复杂问题，确认返回完整多角度回答
