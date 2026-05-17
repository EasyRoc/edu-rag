# 多策略混合检索

## 1. 目标

在现有混合检索（稠密+稀疏+RRF融合）基础上，引入5种查询策略，根据查询意图、复杂度和初次检索质量自动选择策略组合，提升检索召回率和答案质量。

## 2. 范围

### 包含
- **多查询生成（Multi-Query）**：对中等复杂度查询生成3-5个同义变体，分别检索后融合
- **多查询结果融合（RRF Fusion）**：将多查询的各路检索结果用RRF统一融合排序
- **复杂问题分解（Decomposition）**：将复杂查询拆解为2-4个子问题，逐个子问题检索后合并
- **Step-Back 回退**：生成更抽象/上游的"回退问题"，用其检索结果补充上下文
- **HyDE 假设文档嵌入**：生成假设答案，用假设答案的embedding替代原query进行检索
- 策略选择器：根据意图、复杂度、首轮检索质量自动决策

### 不包含
- 跨文档多跳推理（multi-hop reasoning）
- 检索结果重排序模型（cross-encoder reranker）
- 查询改写（语法纠错、实体替换）—— 可作为后续独立功能
- 策略执行的缓存/持久化

## 3. 架构设计

### 3.1 整体流程

```
retrieve_node
    |
    v
[策略选择器] ──→ 根据 intent + complexity 决定主策略
    |
    ├── complexity=simple → 直接 hybrid_retrieve（不触发额外策略）
    |
    ├── complexity=medium → [多查询策略]
    |       multi_query_generate() → 生成 N 个变体
    |       每个变体 → hybrid_retrieve()
    |       multi_query_fusion() → RRF 融合去重 → 结果集
    |
    ├── complexity=complex → [分解策略]
    |       decompose_query() → 拆成子问题
    |       每个子问题 → hybrid_retrieve()
    |       merge_sub_results() → 合并去重 → 结果集
    |
    v
[检索质量评估] ← 对所有结果集评估置信度
    |
    ├── 通过 → 直接输出
    ├── 置信度不足 → [补充策略]
    |       ├── step_back_retrieve() → 回退问题检索 → 补充结果
    |       └── hyde_retrieve() → 假设答案检索 → 补充结果
    v
最终结果集（去重、排序、截断 top_k）
```

### 3.2 模块划分

```
core/
├── strategies/
│   ├── __init__.py           # 策略注册 & 导出
│   ├── selector.py           # 策略选择器
│   ├── multi_query.py        # 多查询生成 + 融合
│   ├── decomposition.py      # 复杂问题分解
│   ├── step_back.py          # Step-Back 回退
│   └── hyde.py               # HyDE 假设文档嵌入
├── nodes/
│   └── retriever.py          # 修改：集成策略选择器
└── vectorestore.py           # 修改：暴露单次检索方法供策略调用
```

### 3.3 关键类/函数签名

```python
# core/strategies/selector.py
class StrategyType(Enum):
    DIRECT = "direct"           # 直接检索
    MULTI_QUERY = "multi_query"
    DECOMPOSITION = "decomposition"

def select_strategy(intent: str, complexity: str) -> StrategyType: ...
def assess_retrieval_quality(docs: list, threshold: float = 0.5) -> bool: ...

# core/strategies/multi_query.py
async def generate_query_variants(query: str, n: int = 4) -> list[str]: ...
def multi_query_fusion(all_results: list[list], top_k: int, rrf_k: int = 60) -> list: ...

# core/strategies/decomposition.py
async def decompose_query(query: str) -> list[str]: ...
def merge_sub_results(sub_results: list[list], top_k: int) -> list: ...

# core/strategies/step_back.py
async def generate_step_back_query(query: str) -> str: ...

# core/strategies/hyde.py
async def generate_hypothetical_answer(query: str) -> str: ...
```

### 3.4 策略选择决策表

| intent | complexity | 主策略 |
|--------|-----------|--------|
| 概念解释 (concept_explain) | simple | DIRECT |
| 概念解释 | medium | MULTI_QUERY |
| 概念解释 | complex | DECOMPOSITION |
| 解题方法 (problem_solving) | simple | DIRECT |
| 解题方法 | medium/complex | DECOMPOSITION |
| 知识总结 (knowledge_summary) | medium/complex | MULTI_QUERY |
| 对比分析 (comparison) | 任意 | DECOMPOSITION |
| 事实查询 (factual) | simple | DIRECT |
| 事实查询 | medium | MULTI_QUERY |
| 其他 | 任意 | DIRECT |

### 3.5 补充策略触发条件

首轮检索后评估：

| 条件 | 补充策略 |
|------|---------|
| top1 分数 < 0.4 | HyDE（适合定义类、事实类） |
| 结果数量 < 3 或 平均分 < 0.5 | Step-Back（适合需要背景知识的） |
| 结果多样性低（内容相似度 > 0.8） | Step-Back + HyDE 并行，取并集 |

## 4. 接口设计

### 4.1 修改 `retrieve_node`

```python
# core/nodes/retriever.py
async def retrieve_node(state: RAGState) -> dict:
    intent = state.get("intent", "other")
    complexity = state.get("complexity", "medium")
    query = state["query"]
    subject = state.get("subject")
    grade = state.get("grade")

    # 1. 选择主策略
    strategy = select_strategy(intent, complexity)

    # 2. 执行主策略检索
    if strategy == StrategyType.DIRECT:
        docs = hybrid_retrieve(query, subject, grade, complexity)
    elif strategy == StrategyType.MULTI_QUERY:
        variants = await generate_query_variants(query)
        all_results = [hybrid_retrieve(v, subject, grade, complexity) for v in [query] + variants]
        docs = multi_query_fusion(all_results, top_k_from_complexity(complexity))
    elif strategy == StrategyType.DECOMPOSITION:
        sub_queries = await decompose_query(query)
        sub_results = [hybrid_retrieve(sq, subject, grade, "simple") for sq in sub_queries]
        docs = merge_sub_results(sub_results, top_k_from_complexity(complexity))

    # 3. 评估检索质量，必要时补充
    if not assess_retrieval_quality(docs):
        docs = await apply_supplementary_strategies(query, docs, subject, grade, intent)

    return {"retrieved_docs": docs}
```

### 4.2 LLM 调用接口

所有策略共享同一个 LLM 调用模式——通过 `ChatOpenAI` 以 prompt 方式生成变体/子问题/假设答案：

```
# 多查询变体 prompt
"请将以下问题改写为 {n} 个不同角度的同义表述，用于提升文档检索召回率：\n{query}"

# 问题分解 prompt
"请将以下复杂问题拆解为 2-4 个简单的子问题，每个子问题可独立检索回答：\n{query}"

# Step-Back prompt
"请针对以下问题生成一个更抽象、更通用的'回退问题'，用于检索背景知识：\n{query}"

# HyDE prompt
"请针对以下问题写一段假设性的标准答案（即使不确定也尽量写），用于以其embedding来检索相关文档：\n{query}"
```

## 5. 数据模型

### 5.1 RAGState 扩展

```python
class RAGState(TypedDict):
    # ... 现有字段保持不变 ...
    query: str
    subject: str
    grade: str
    intent: str
    complexity: str
    retrieved_docs: list[Document]
    # 新增字段
    applied_strategies: list[str]        # 本次检索应用的策略列表
    retrieval_quality_score: float       # 检索质量评分
```

### 5.2 不涉及数据库/集合变更

Milvus 表结构、BM25 索引结构均不变。策略层只在查询侧工作。

## 6. 配置 & 环境变量

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `MULTI_QUERY_VARIANTS` | 4 | 多查询生成的变体数量 |
| `DECOMPOSITION_MAX_SUB` | 4 | 复杂问题最多拆解的子问题数 |
| `RETRIEVAL_QUALITY_THRESHOLD` | 0.5 | 检索质量最低置信度阈值 |
| `HYDE_MIN_SCORE` | 0.4 | 触发 HyDE 补充策略的最低分 |
| `STEP_BACK_MIN_DOCS` | 3 | 触发 Step-Back 的最少结果数 |
| `STRATEGY_LLM_MODEL` | (同现有 LLM) | 策略使用的 LLM 模型 |
| `STRATEGY_TIMEOUT` | 10 | 单次策略 LLM 调用超时(秒) |

## 7. 风险 & 边界条件

| 风险 | 处理策略 |
|------|---------|
| **LLM 调用增加延迟** | 多查询变体生成用单次 LLM 调用生成所有变体；子问题拆解也是单次调用；设置 10s 超时；策略失败时降级为 DIRECT |
| **HyDE 生成幻觉** | HyDE 仅用于检索，不直接展示给用户；若假设答案方向错误，检索结果也会偏，因此仅在首轮检索置信度很低时才触发 |
| **分解的子问题互相依赖** | 当前方案各子问题独立检索，不做串行依赖。若后续需要多跳推理再扩展 |
| **策略 LLM 调用成本** | 仅在 complexity=medium/complex 时触发额外 LLM 调用；simple 查询零额外开销 |
| **重复文档** | RRF 融合天然去重（按 chunk_id）；合并子结果时也按 doc_id 去重 |
| **流式响应延迟** | 策略执行在 retrieve_node 中完成，不阻塞 generate_node 的流式输出 |

## 8. 测试要点

### 8.1 单元测试
- `select_strategy()` 各 intent × complexity 组合返回正确策略
- `assess_retrieval_quality()` 对高分/低分/空结果正确判断
- `multi_query_fusion()` RRF 融合排序正确性
- `merge_sub_results()` 去重逻辑

### 8.2 集成测试
- 中等复杂度查询走 MULTI_QUERY 策略，结果数 ≥ DIRECT
- 复杂查询走 DECOMPOSITION，结果覆盖所有子问题
- 低质量检索触发 Step-Back / HyDE 补充
- simple 查询不触发额外 LLM 调用（延迟无增加）
- 策略 LLM 超时降级为 DIRECT 不报错

### 8.3 Mock 策略
- 测试时 mock LLM 调用，确保各策略路径可独立验证
