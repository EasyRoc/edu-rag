# 复杂问题 RAG 链路优化需求文档

## 背景

当前 RAG 系统对简单、中等复杂度的问题处理良好，但在面对 100~200 字的复杂问题（含比较、分析、推导、综合等多角度要求）时，存在 6 个结构性问题，导致检索覆盖不足、生成质量下降和误拒绝。

---

## 问题分析

### P0-1：Reranker 误伤子问题检索结果

**现象**：DECOMPOSITION 策略将复杂问题拆为子问题分别检索，但合并后 reranker 用原始复杂问题对各 chunk 统一打分。能覆盖某一子方向的 chunk 因不匹配原问题的全部要求，得分被压低。

**根因**：`graph.py:83-105` `rerank_node` 接收合并后的统一 doc 列表，`CrossEncoderReranker.rerank(original_query, docs)` 对每个 doc 打的是对原始 query 的分，不是对"当初检索它的那个子 query"的分。

**影响**：大量有效但仅覆盖局部子问题的 chunk 被挤出 top 5，最终上下文信息不完整。

### P0-2：缺少子答案中间推理

**现象**：拆解→各子问题检索→合并 chunk→一次性 LLM 生成。跳过了"各子问题先形成中间答案"这一推理步骤。

**根因**：`graph.py:153-166` `generate_node` 直接将合并后的 top 5 碎片塞入 prompt，LLM 需在单次生成中同步完成信息理解、多角度综合、引用标注。

**影响**：对 100~200 字的多角度复杂问题，LLM 容易遗漏某个子方向或生成表面化答案。

### P1-3：合并去重损失多样性

**现象**：`decomposition.py:45-59` `merge_sub_results` 仅按 chunk_id 去重保留最高分。多个子问题检索到的共性 chunk 占满 top 20，差异化结果被挤出。

**影响**：子问题的长尾信息丢失，最终 top 5 覆盖面偏向高频共性内容。

### P1-4：Token 预算限制回答深度

**现象**：`LLM_MAX_CONTEXT_TOKENS=8192`，`max_tokens=2048`。复杂问题自身 100~200 字，5 个 chunk (~1000 tokens) + system prompt + history，还剩约 5000 tokens。但 max_tokens=2048 限制了回答长度。

**影响**：要求"举例+分析+对比"的复杂问题，2048 token 回答不够充分。

### P2-5：拆解失败的降级链路弱

**现象**：`retriever.py:114-116` 当 LLM 拆解结果 ≤1 个子问题时，降级为 DIRECT 检索。

**影响**：200 字的复杂问题用单查询 DIRECT 检索，覆盖面大概率不足，注定触发 retry，凭空增加延迟。

### P2-6：门控阈值对复杂问题不友好

**现象**：`RETRIEVAL_ACCEPT_TOP1_THRESHOLD=0.60`，所有复杂度共用同一阈值。复杂问题拆解后的子问题 chunk 难在原问题上拿到 0.60+。

**影响**：有真实资料的复杂问题被误拒（false reject）。

---

## 优化目标

1. 复杂问题的检索覆盖率和上下文信息完整度提升到与简单/中等问题同等水平
2. 复杂问题的生成答案多角度覆盖度、深度明显改善
3. 端到端延迟增量控制在可接受范围（复杂问题允许比简单问题多 1~2 次 LLM 调用）
4. 降低复杂问题的误拒绝率
5. 不退化简单/中等问题现有表现

---

## 需求方案

### 需求 1：子问题感知重排 (Sub-Query-Aware Rerank) — 对应 P0-1

**描述**：将"合并后统一重排"改为"各子问题独立重排后加权合并"。

**流程变更**：

```
当前:
  子问题1 检索 → ┐
  子问题2 检索 → 合并去重 → rerank(原问题, 全部chunk) → top 5
  子问题3 检索 → ┘

改为:
  子问题1 检索 → rerank(子问题1, chunks_1)  → ┐
  子问题2 检索 → rerank(子问题2, chunks_2)  → 加权合并 → top N → rerank(原问题, top N) → top 5
  子问题3 检索 → rerank(子问题3, chunks_3)  → ┘
```

**关键设计**：
- 第一阶段：每个子问题的候选列表独立做 CrossEncoder rerank，保证子问题级别的相关性不被原问题稀释
- 第二阶段：各子问题 top K（如 K=5~8）合并去重后，用原问题做第二轮 rerank，保证最终结果与原问题的整体匹配度
- doc 携带 `source_sub_query` 标记，供第二阶段参考
- 考虑复用当前 `retriever.py` 中的 `_annotate` 函数思想，每个 doc 标注来源子问题

**涉及文件**：
- `core/nodes/retriever.py` — `_decomposition_retrieve` 改为子问题独立 rerank
- `core/reranker.py` — 可能需要批量 rerank 接口优化
- `core/graph.py` — rerank 节点可能需要感知策略类型

**验收标准**：
- 子问题检索结果的 rerank 得分不再被原问题整体复杂度压低
- RAGAS `context_recall` 在复杂问题测试集上提升 ≥10%
- 简单/中等问题走原路径，不受影响

---

### 需求 2：子答案合成生成 (Sub-Answer Synthesis) — 对应 P0-2

**描述**：在 DECOMPOSITION 策略下，先生成各子问题的中间答案，再综合为最终答案。

**流程变更**：

```
当前:
  子问题检索合并 → generate(原问题, top5 chunk) → 最终答案

改为:
  子问题1 chunk → generate(子问题1, chunk_1) → 子答案1 ┐
  子问题2 chunk → generate(子问题2, chunk_2) → 子答案2  → synthesize(原问题, 子答案们 + 原chunk) → 最终答案
  子问题3 chunk → generate(子问题3, chunk_3) → 子答案3 ┘
```

**关键设计**：
- 子答案生成使用轻量 prompt（"仅基于以下上下文简要回答子问题"），`max_tokens=512`，temperature 更低（0.1），追求准确而非发挥
- 子答案并行生成（asyncio.gather），不串行增加延迟
- synthesize 阶段将子答案作为结构化中间层传入 prompt：
  ```
  ## 子问题分析
  ### 子问题1: xxx
  答案: [子答案1]
  ### 子问题2: xxx
  答案: [子答案2]

  ## 原始上下文
  [完整上下文]
  ```
- 最终 synthesis 的 `max_tokens` 提升到 4096（仅 DECOMPOSITION 策略）
- 子答案生成失败时降级为当前行为（直接单次生成）

**涉及文件**：
- `core/nodes/generator.py` — 新增 `generate_sub_answers()` 和 `synthesize_final_answer()`
- `core/graph.py` — generate_node 分支：DECOMPOSITION 走子答案合成路径
- `config.py` — 新增 `SUB_ANSWER_MAX_TOKENS=512`、`SYNTHESIS_MAX_TOKENS=4096`

**验收标准**：
- 复杂问题的最终答案覆盖原问题的所有子方向
- RAGAS `answer_relevancy` 在复杂问题测试集上提升 ≥5%
- 子答案合成链路总延迟不超过当前链路 ×1.5

---

### 需求 3：多样性感知合并 (Diversity-Aware Merge) — 对应 P1-3

**描述**：合并子问题检索结果时，引入基于语义相似度的多样性控制，避免共性 chunk 挤占差异化结果。

**流程变更**：

```
merge_sub_results() 改为:
  1. 按 chunk_id 去重（保留最高分）
  2. 按 rerank_score 降序排列
  3. 多样性过滤：滑动窗口检查相邻 chunk 的向量余弦相似度
     - 若相似度 > 0.85 且来自不同子问题 → 保留高分者，低分者降到候选池尾部
     - 若相似度 > 0.85 且来自不同子问题且低分来自独特子问题 → 保留（不丢弃独有方向）
  4. 填充到 top K：从前半（高分）+ 后半（多样化）各取一定比例
```

**关键设计**：
- 利用已有的 embedding 向量做相似度计算（Milvus 存储了 512 维向量），无需额外模型
- 保留策略优先保证每个子问题至少有一个代表 chunk
- 新增 `MERGE_DIVERSITY_RATIO` 配置项，控制高分/多样比例（默认 0.6/0.4）

**涉及文件**：
- `core/strategies/decomposition.py` — `merge_sub_results` 增加多样性过滤
- `config.py` — 新增 `MERGE_DIVERSITY_RATIO=0.6`、`MERGE_SIMILARITY_THRESHOLD=0.85`

**验收标准**：
- 合并后的候选列表包含来自不同子问题的差异化 chunk
- 去重后 chunk 来自 ≥80% 子问题的比例提升
- RAGAS `context_precision` 不退化

---

### 需求 4：动态 Token 预算 — 对应 P1-4

**描述**：根据问题复杂度动态调整生成阶段的 `max_tokens` 和上下文 chunk 数量。

**关键设计**：

| 复杂度 | max_tokens | context_top_k | 说明 |
|--------|-----------|---------------|------|
| simple | 1024 | 3 | 事实性问题短答即可 |
| medium | 2048 | 5 | 当前默认值 |
| complex | 4096 | 8 | 复杂问题需要更多上下文和更长回答 |

- `generate_node` 中根据 `state["complexity"]` 查表决定参数
- complex 模式同时提升 `LLM_MAX_CONTEXT_TOKENS` 到 16384（对 DeepSeek 等 128K 窗口模型完全安全）

**涉及文件**：
- `core/nodes/generator.py` — `llm_generate_stream` 接受 complexity 参数
- `core/graph.py` — `generate_node` 传递 complexity
- `config.py` — 新增 `COMPLEX_MAX_TOKENS=4096`、`COMPLEX_CONTEXT_TOP_K=8`

**验收标准**：
- 复杂问题生成的回答不被 max_tokens 截断
- 简单/中等问题仍使用当前参数，不浪费 token

---

### 需求 5：拆解失败的多级降级 — 对应 P2-5

**描述**：当 DECOMPOSITION 拆解失败时，不直接退到 DIRECT，而是使用更合理的降级策略。

**降级链路**：

```
DECOMPOSITION 拆解
  ├── 拆解成功 (2~4 个子问题) → 正常子问题流程
  ├── 拆解为 1 个子问题 → MULTI_QUERY（用原问题生成 4 个变体，扩大覆盖面）
  └── 拆解完全失败/异常  → MULTI_QUERY + HyDE 并行（两条路同时检索，合并结果）
```

**涉及文件**：
- `core/nodes/retriever.py` — `_decomposition_retrieve` 失败分支改为 MULTI_QUERY

**验收标准**：
- 拆解失败时不再退到单查询 DIRECT
- 拆解失败的降级链路在相关文档存在时能正确召回

---

### 需求 6：按复杂度分级的门控阈值 — 对应 P2-6

**描述**：复杂问题使用更宽松的 accept 阈值，降低误拒绝率。

**关键设计**：

| 复杂度 | accept_top1_threshold | relevance_threshold | max_retries |
|--------|----------------------|--------------------|-------------|
| simple | 0.60 | 0.50 | 1 |
| medium | 0.60 | 0.50 | 2 (当前值) |
| complex | 0.45 | 0.35 | 2 |

- `evaluate_retrieval_gate()` 接受 `complexity` 参数，内部查表
- 检索离线评估（`retrieval_evaluator.py`）也按复杂度分别计算误判率

**涉及文件**：
- `core/retrieval_quality.py` — `evaluate_retrieval_gate` 新增 complexity 参数
- `core/graph.py` — `retrieval_gate_node` 传递 complexity
- `config.py` — 新增 `COMPLEX_ACCEPT_TOP1_THRESHOLD=0.45`、`COMPLEX_RELEVANCE_THRESHOLD=0.35`

**验收标准**：
- 复杂问题检索评估的 false_reject_rate 降低到与中等问题接近的水平
- simple/medium 门控行为不变

---

## 整体流程对比

### 当前流程 (complex 路径)

```
用户提问 (复杂)
  → classify (LLM ①)
  → decompose (LLM ②)
  → 各子问题 hybrid_search
  → merge (简单去重)
  → rerank (原问题)              ← 问题 1: 误伤子问题 chunk
  → retrieval_gate (统一阈值)    ← 问题 6: 阈值不友好
  → generate (top 5, max 2048)  ← 问题 2: 无子答案 / 问题 4: token 不够
     ↑ 若拆解失败 → DIRECT       ← 问题 5: 降级太弱
```

### 优化后流程 (complex 路径)

```
用户提问 (复杂)
  → classify (LLM ①)
  → decompose (LLM ②)
  │   ├── 成功 → 子问题流程
  │   └── 失败 → MULTI_QUERY + HyDE  (需求 5: 多级降级)
  │
  → 各子问题 hybrid_search
  → 各子问题独立 rerank(子问题)    (需求 1: 子问题感知重排)
  → 多样性感知合并                (需求 3: 保留差异化结果)
  → rerank(原问题, 合并结果)       (需求 1: 二阶段 rerank)
  → retrieval_gate(complexity)   (需求 6: 分级阈值)
  │
  → 子答案并行生成 (LLM ③)        (需求 2: 子答案中间层)
  → 综合合成 (LLM ④, max 4096)    (需求 2+4: 深度合成)
  → 最终答案
```

---

## 优先级与分期

### 第一期 (P0 — 解决最疼的问题)

| 需求 | 说明 | 预估工作量 |
|------|------|-----------|
| 需求 1：子问题感知重排 | 解决核心检索质量问题 | 2~3 天 |
| 需求 2：子答案合成生成 | 解决生成深度问题 | 2~3 天 |
| 需求 6：分级门控阈值 | 解决误拒绝问题 | 0.5 天 |

第一期先解决 P0 级别的三个问题，它们是当前复杂问题链路的最大瓶颈。

### 第二期 (P1 — 质量优化)

| 需求 | 说明 | 预估工作量 |
|------|------|-----------|
| 需求 3：多样性感知合并 | 提升上下文覆盖 | 1 天 |
| 需求 4：动态 Token 预算 | 利用大窗口模型优势 | 0.5 天 |

### 第三期 (P2 — 鲁棒性)

| 需求 | 说明 | 预估工作量 |
|------|------|-----------|
| 需求 5：多级降级 | 改善异常路径 | 1 天 |

---

## 测试方案

### 单元测试

- `test_decomposition.py`：拆解失败降级链路分支覆盖
- `test_reranker.py`：子问题感知重排两阶段输出验证
- `test_generator.py`：子答案合成 prompt 结构验证、并行生成路径
- `test_retrieval_quality.py`：分级阈值各复杂度分支
- `test_merge.py`：多样性合并的相似度过滤行为

### 集成测试

- 复杂问题端到端回归：`evaluation/cli.py evaluate --live` 用复杂问题测试集跑完整链路
- 简单/中等问题回归：确保优化后不退化
- RAGAS 四指标对比：优化前 vs 优化后，聚焦 `context_recall` 和 `answer_relevancy`

### 评估数据集

- 新增 `data/test_sets/complex_questions.jsonl`：30~50 道 100~200 字的复杂问题，覆盖多种类型（比较、分析、推导、综合）
- 每条包含：question、ground_truth、complexity=complex、预期覆盖的 chunk_id 列表

---

## 风险与注意事项

1. **延迟增加**：需求 1 增加一轮子问题独立 rerank，需求 2 增加子答案并行生成+synthesis。总延迟增量应在 1.5× 以内（需求 2 验收标准），若超标需考虑子答案的 eager cache 或流式 synthesis
2. **LLM 调用次数**：复杂问题的 LLM 调用从 3 次增加到 4~5 次（classify + decompose + 子答案×N + synthesize），API 成本上升，需在配置中可开关（`ENABLE_DEEP_COMPLEX_MODE`）
3. **简单/中等问题不受影响**：所有变更必须通过 `complexity == "complex"` 条件分支隔离，不能改动 DIRECT/MULTI_QUERY 策略的现有行为
4. **与现有 RAGAS 评估兼容**：新增逻辑不能破坏 `evaluation/pipeline.py` 和 `evaluation/ragas_evaluator.py` 的现有工作流
5. **重试回路兼容**：retry_planner 生成的 HyDE/Step-Back 纠正策略也需要兼容新的复杂度感知逻辑
