# 核心四节点详解：retrieve → rerank → retrieval_gate → retry_planner

> 对应 `core/graph.py` 中 `build_rag_graph()` 注册的四个核心节点，按数据流向串联。

---

## 总览

```
retrieve ──→ rerank ──→ retrieval_gate ──┬── accept ──→ generate
             ↑                           │
             │                    retry ──┤
             │                           │
             └──── retry_planner ◄────────┘
```

| 节点 | 职责 | 一句话 |
|------|------|--------|
| **retrieve** | 召回候选文档 | 根据检索计划从 Milvus+BM25 捞文档，返回 `retrieved_docs` + `sub_queries` |
| **rerank** | 重排序 | 用 CrossEncoder 对候选文档精排，复杂问题走子问题感知重排 |
| **retrieval_gate** | 质量门控 | 判断重排后的文档质量是否过关：放行/重试/拒答 |
| **retry_planner** | 制定重试计划 | 根据门控失败原因，规划下一轮用什么策略重新检索 |

---

## 1. retrieve_node（召回）

### 入口

```python
# core/graph.py:57
async def retrieve_node(state: RAGState, vector_store: K12VectorStore) -> dict:
```

### 输入

| 来自 state | 含义 |
|-----------|------|
| `state["query"]` | 用户原始问题 |
| `state["complexity"]` | simple / medium / complex |
| `state["intent"]` | educational / chitchat / ... |
| `state["retrieval_plan"]` | 检索计划，首次为 `{"strategy": "initial", ...}` |

### 核心逻辑 — `hybrid_retrieve()` 入口分发

```
retrieval_plan.strategy 决定走哪条路：

  "complex_repair" ──→ _retrieve_complex_repair()     ← 复杂问题 retry 专用
        │
  "hyde" / "step_back" / "query_variants"
        │
        └──→ _retrieve_from_plan()                    ← 普通问题 retry 专用
        │
  "initial" (首次检索)
        │
        └──→ select_strategy(intent, complexity)      ← 按复杂度选策略
               │
               ├── simple  → _direct_retrieve()       直接 Dense+BM25
               ├── medium  → _multi_query_retrieve()   LLM 生成4个改写，多路检索
               └── complex → _decomposition_retrieve() LLM 拆解子问题，各自检索
```

### complex 路径核心：`_decomposition_retrieve()`

```
1. decompose_query(query) → ["子问题A", "子问题B", "子问题C"]
   拆解失败（≤1个）→ 降级为 _multi_query_retrieve

2. 对每个子问题单独 _search():
   - Dense+BM25 混合检索
   - top_k = max(3, 20 // 子问题数)  # 每个子问题分配额
   - 给每个 doc 标注 source_sub_query = "子问题A"

3. merge_sub_results(results, limit)
   - 按 doc.id 去重（同id保留最高分）
   - 按 score 降序 → 截到 limit(20)

4. _annotate(docs, "decomposition", query)
   - 给每个 doc 打 retrieval_strategy="decomposition"
```

### 输出

```
return {
    "retrieved_docs": docs,       # 最多20条候选，带 source_sub_query 标注
    "retrieval_latency_ms": ...,
    "sub_queries": sub_queries,   # ["子A","子B","子C"]，非complex为空[]
}
```

---

## 2. rerank_node（重排）

### 入口

```python
# core/graph.py:85
async def rerank_node(state: RAGState, reranker: CrossEncoderReranker) -> dict:
```

### 输入

| 来自 state | 含义 |
|-----------|------|
| `state["retrieved_docs"]` | retrieve 产出的候选（最多20条） |
| `state["complexity"]` | 决定走单阶段还是两阶段 |
| `state["sub_queries"]` | 子问题列表，两阶段重排需要 |

### 核心逻辑 — 两条路径

```
_should_use_two_stage_rerank?
  条件: complexity=="complex" + sub_queries>=2 + ENABLE_DEEP_COMPLEX_MODE=true

  YES → _two_stage_rerank()    ← 子问题感知重排
  NO  → reranker.rerank(query, docs)  ← 单阶段重排（原路径）
```

### 两阶段重排：`_two_stage_rerank()`

```
Stage 1: 子问题独立打分

  按 source_sub_query 把 docs 分组:
    子问题A → [chunk_1, chunk_3, chunk_5]
    子问题B → [chunk_2, chunk_4]
    __no_source__ → [chunk_6, chunk_7]   ← 没标注来源的兜底组

  每组独立 rerank:
    reranker.rerank("子问题A", [chunk_1, chunk_3, chunk_5])
    reranker.rerank("子问题B", [chunk_2, chunk_4])
    reranker.rerank(原问题,     [chunk_6, chunk_7])    ← 无来源组用原问题

  每组取 top SUB_RERANK_TOP_K(6) 进入下一轮

Stage 2: 合并去重

  各组 top 6 汇合 → 按 rerank_score 降序 → 按 doc_id 去重 → 直接返回
  **注意：不再用原问题二次 rerank**（会压低局部证据分）
```

为什么比单阶段好：

```
单阶段: reranker.rerank("比较勾股定理和相似三角形的异同，举例...", [A,B,C])
  → chunk_勾股定义(0.35) 被 "比较...异同" 压低

两阶段: reranker.rerank("勾股定理的定义", [A]) → 0.85 ✅
        reranker.rerank("相似三角形判定", [B]) → 0.82 ✅
  → 各子方向代表都在，不被原问题稀释
```

### 异常处理

```
CrossEncoder 加载/推理失败:
  → RerankerUnavailableError
  → docs 原样通过（不排序）
  → reranker_available = False  ← gate 据此决定观察放行还是拒答
```

### 输出

```
return {
    "retrieved_docs": docs,          # 重排后的候选（≤20条，按 rerank_score 降序）
    "reranker_available": True/False,
    "rerank_latency_ms": ...,
}
```

---

## 3. retrieval_gate_node（门控）

### 入口

```python
# core/graph.py:178
async def retrieval_gate_node(state: RAGState) -> dict:
```

### 输入

| 来自 state | 含义 |
|-----------|------|
| `state["retrieved_docs"]` | rerank 后的候选 |
| `state["retry_count"]` | 当前第几次重试 |
| `state["reranker_available"]` | 重排器是否可用 |
| `state["complexity"]` | 决定阈值等级 |
| `state["sub_queries"]` | 复杂问题的子问题列表，用于覆盖检查 |

### 核心逻辑 — 五分支决策

调用 `evaluate_retrieval_gate()`（`core/retrieval_quality.py`），按优先级短路判断：

```
分支1: 空召回（docs为空）
  → 有重试次数 → retry（建议 hyde）
  → 无重试次数 → abstain

分支2: 重排器不可用
  → observe 模式 → accept（仅记录）
  → enforce 模式 → abstain

分支3: 复杂问题子问题覆盖检查（use_subquery_gate=true 时）
  逐子问题检查:
    子问题A: 有doc得分≥0.35 → covered ✅
    子问题B: 有doc但得分<0.35 → weak ⚠️
    子问题C: 无相关doc       → missing ❌

  covered_count < total → retry（带 _build_complex_repair_plan）
  重试耗尽            → abstain

  阈值: complex 用 COMPLEX_RELEVANCE_THRESHOLD(0.35) / COMPLEX_ACCEPT_TOP1_THRESHOLD(0.45)
        simple/medium 用 0.50 / 0.60

分支4: 质量达标
  top1_score >= accept_top1_threshold → accept

分支5: 质量不达标（走到这说明 1/2/3/4 都不满足）
  relevant_count==0 → retry（建议 hyde）
  有文档但分低      → retry（建议 step_back）
  重试耗尽         → abstain
```

### 复杂问题覆盖检查详解（分支3）

```
_compute_subquery_metrics(docs, sub_queries):

  对每个子问题:
    1. 找到 source_sub_query 匹配该子问题的 docs
    2. 检查 rerank_score:
        无doc得分≥relevant_threshold → status="missing", repair="hyde"
        top1 < top1_threshold        → status="weak",    repair="step_back"
        top1 ≥ top1_threshold        → status="covered", repair="direct"

  返回 metrics 包含:
    total_subquery_count: 3
    covered_subquery_count: 1
    missing_subqueries: ["两者对比例题"]
    weak_subqueries: ["相似三角形判定"]
    subquery_metrics: [{query, status, repair, top1_score, ...}, ...]
```

### 复杂修复计划：`_build_complex_repair_plan()`

```
metrics.subquery_metrics:
  [{"query":"勾股定理定义",  "status":"covered", "repair":"direct"},
   {"query":"相似三角形判定","status":"weak",    "repair":"step_back"},
   {"query":"两者对比例题",  "status":"missing", "repair":"hyde"}]

→ 打包成 suggested_plan:
    {"strategy": "complex_repair",
     "subqueries": [...]}

→ retry_planner 拿到后按 repair 类型定向补检:
    covered → 跳过（已有结果直接复用）
    weak    → step_back 回溯宽泛概念重新检索
    missing → hyde 生成假设答案重新检索
```

### 输出

```
return {
    "retrieval_metrics": decision["metrics"],     # 含 subquery 覆盖详情
    "retrieval_decision": decision,               # {action, reason_codes, suggested_strategy, suggested_plan}
    "retrieval_attempts": [...],                  # 追加本次尝试记录
}
```

---

## 4. retry_planner_node（重试规划）

### 入口

```python
# core/graph.py:213
async def retry_planner_node(state: RAGState) -> dict:
```

### 触发条件

被 `_route_by_gate` 条件边路由到此处：`gate.action == "retry"`

```
retrieval_gate ──┬── accept  → generate
                 ├── retry   → retry_planner → retrieve（形成回路）
                 └── abstain → abstain
```

### 输入

| 来自 state | 含义 |
|-----------|------|
| `state["retry_count"]` | 当前重试次数 |
| `state["retrieval_decision"]` | gate 产出的决策（含 suggested_strategy 和 suggested_plan） |
| `state["complexity"]` | 决定走普通重试还是复杂修复 |
| `state["sub_queries"]` | 复杂问题的子问题列表 |

### 核心逻辑 — `build_retry_plan()` 三种策略

```
复杂问题（complexity=="complex" + sub_queries>=2）:

  gate 给出了 complex_repair 计划？
    YES → 直接用（_normalize_complex_repair_plan 补齐缺失项）
    NO  → 兜底构造: 所有子问题统一用 fallback_repair
            fallback: retry1→query_variants, retry2+→gate建议的strategy

  返回 plan: {"strategy": "complex_repair", "subqueries": [...], "queries": [...]}


普通问题:

  retry1: 盲扩 — query_variants（生成3个改写）
    → plan: {"strategy": "query_variants", "queries": [原问题, 改写1, 改写2, 改写3]}

  retry2+: 对症 — gate.suggested_strategy
    relevant_count==0 → "hyde"
    有文档但分低      → "step_back"
    → plan: {"strategy": "hyde"/"step_back", "queries": [原问题]}
```

### 输出

```
return {
    "retry_count": next_retry,      # retry_count + 1
    "retrieval_plan": plan,         # 下一轮检索计划
}
```

`retrieval_plan` 会替代初始的 `{"strategy": "initial"}`，下一轮 `retrieve_node` 读到后走 `complex_repair` / `hyde` / `step_back` / `query_variants` 分支。

---

## 数据流全景图

```
                      ┌─────────────────────────────────────────┐
                      │           RAGState 关键字段              │
                      └─────────────────────────────────────────┘

retrieve_node
  读: query, complexity, intent, retrieval_plan
  写: retrieved_docs, sub_queries, retrieval_latency_ms
      │
      ▼
rerank_node
  读: retrieved_docs, complexity, sub_queries
  写: retrieved_docs (rerank_score), reranker_available, rerank_latency_ms
      │
      ▼
retrieval_gate_node
  读: retrieved_docs, retry_count, reranker_available, complexity, sub_queries
  写: retrieval_metrics, retrieval_decision, retrieval_attempts
      │
      ├── action=accept → generate
      ├── action=abstain → abstain
      └── action=retry
            │
            ▼
      retry_planner_node
        读: retry_count, retrieval_decision, complexity, sub_queries
        写: retry_count (+1), retrieval_plan
            │
            └──→ retrieve_node（回路）
```

---

## 关键设计要点

1. **retrieve 不判断质量**，只负责「尽量多捞」。质量判断统一在 gate 做，避免多套阈值。

2. **retrieval_plan 是回路开关**：初始 `"initial"` → 正常策略选择；retry 后改写为 `"hyde"` / `"complex_repair"` 等 → `hybrid_retrieve` 入口直接走对应分支。

3. **sub_queries 贯穿全链路**：retrieve 产出 → rerank 用于分组重排 → gate 用于覆盖检查 → retry_planner 用于定向修复 → generate 用于子答案合成。任一环节丢失，下游能力退化为普通模式。

4. **复杂问题 retry 是定向修复而非全量重来**：`complex_repair` 按子问题的 status 分别决定 repair 策略（covered 跳过 / weak 回溯 / missing 生成假设答案），只补检缺失方向。
