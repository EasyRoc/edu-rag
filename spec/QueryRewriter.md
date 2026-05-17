下面是一份可直接用于交给模型（如 Copilot、Cursor、Claude Engineering）生成代码的 **Spec 文档**。格式上强调**可执行路径、输入输出、约束条件和验收标准**。

---

# Spec: Query Rewriter 模块（RAG 场景）

## 1. 目标
实现一个 `QueryRewriter` 类，用于在 **RAG 检索前 / 检索后** 动态决定是否需要改写用户原始 Query，并提供多种改写策略。  
目标：**提高检索召回率与精确率，避免无脑使用所有策略导致成本与延迟上升**。

## 2. 输入
- `original_query: str`  
- `first_pass_docs: List[Document]`（可选，由第一次快速检索得到）  
- `config: QueryRewriterConfig`

## 3. 输出
- `final_queries: List[str]`（改写后的一个或多个查询）  
- `used_strategy: str`（记录最终采用的策略名）  
- `debug_info: dict`（可选，用于日志分析）

## 4. 核心逻辑（决策流程）

```text
1. 如果用户未提供 first_pass_docs
   → 先用原 query 快速检索一次（top_k=3）
   → 计算平均相关性得分（依赖 embedding similarity）

2. 如果平均相关性 ≥ threshold（默认 0.7）
   → 直接返回 [original_query]
   → used_strategy = "none"

3. 否则根据 query 特征选择策略（只选一个）：
   - 长度短 / 口语化     → query_expansion
   - 含多问 / “和/与/对比” → query_decomposition
   - 抽象 / 答案存在但表述不匹配 → hyde
   - “为什么/原理”      → step_back
   - “优缺点/比较”      → multi_query

4. 执行选中的策略，返回改写后的 queries
```

## 5. 各策略详细要求

### 5.1 Query Expansion
- 输入：`query`
- 输出：`[expanded_query]`（字符串）
- 实现方式：调用 LLM 生成 3～5 个同义词 / 相关词，拼接到原 query 后面
- LLM Prompt 示例（见附录）

### 5.2 Query Decomposition
- 输入：`query`
- 输出：`[sub_query_1, sub_query_2, ...]`
- 实现方式：LLM 将复合问题拆成多个独立子问题
- 约束：每个子问题必须能独立检索

### 5.3 HyDE
- 输入：`query`
- 输出：`[hyde_query]`（即生成的假设性答案）
- 实现方式：LLM 先生成一段假设性答案，再将该答案作为检索 query（不返回原 query）
- 注意：仅适用于非精确性检索（不适合查 ID、日期、数字）

### 5.4 Step‑back Prompting
- 输入：`query`
- 输出：`[step_back_query, original_query]`
- 实现方式：LLM 生成更抽象的高层问题，与原 query 一起检索

### 5.5 Multi‑query Retrieval
- 输入：`query`
- 输出：`[query1, query2, query3]`（不同角度）
- 实现方式：LLM 生成 3 个不同视角的查询

## 6. 外部依赖
- `LLM` 客户端（支持 `generate(prompt) -> str`）
- `Embeddings` 客户端（用于计算相关性，可选）
- `Document` 类型（与你现有 RAG 系统保持一致）

## 7. 配置参数（QueryRewriterConfig）
```python
@dataclass
class QueryRewriterConfig:
    relevance_threshold: float = 0.7       # 第一次检索的平均相关性阈值
    expansion_synonym_count: int = 3       # 扩展时生成的同义词数量
    hyde_enabled: bool = True              # 是否允许使用 HyDE
    multi_query_count: int = 3             # multi-query 生成数量
    fallback_strategy: str = "expansion"   # 无法判断时使用的默认策略
```

## 8. 非功能性要求
- **延迟**：决策 + 执行总时间 < 1.5 秒（不含 LLM 调用）
- **成本**：每次改写最多调用 LLM 一次（multi-query 除外，但可并行）
- **鲁棒性**：任何 LLM 调用失败 → 降级返回 `[original_query]`，记录告警

## 9. 接口定义（期望代码签名）
```python
class QueryRewriter:
    def __init__(self, llm, embeddings, config: QueryRewriterConfig):
        ...

    def rewrite(
        self, 
        original_query: str, 
        first_pass_docs: Optional[List[Document]] = None
    ) -> Tuple[List[str], str, Dict]:
        """
        Returns:
            - final_queries: 改写后的查询列表
            - used_strategy: 策略名
            - debug_info: 包含相关性得分、决策原因等
        """
        ...
```

## 10. 验收标准（模型生成代码后验证）
- [ ] 当第一次检索得分高时，不改写
- [ ] 短 query（如“怎么修bug”）能触发 expansion
- [ ] “A和B的区别”能触发 decomposition
- [ ] “为什么Transformer好”能触发 step_back
- [ ] 任何 LLM 错误不会导致程序崩溃
- [ ] 输出的 queries 长度不为 0

## 11. 附录：Prompt 示例（可直接内嵌）

### Expansion
```
请为以下问题生成 {count} 个同义词或相关词，用逗号分隔：
问题：{query}
只输出词语，不要解释。
```

### Decomposition
```
将以下复杂问题拆解成 2~4 个独立的、可以单独检索的子问题：
问题：{query}
输出格式：每行一个子问题
```

### HyDE
```
请针对以下问题写一段假设性的答案。答案不需要完全准确，仅用于检索相似文档。
问题：{query}
```

### Step‑back
```
请将以下具体问题抽象成一个更通用、更高层次的问题：
具体问题：{query}
高层次的通用问题：
```

### Multi‑query
```
请从 {count} 个不同角度重写以下问题，每个角度一行：
问题：{query}
```

---

## 12. 使用示例（期望效果）
```python
rewriter = QueryRewriter(llm, embeddings, config)

# 场景1：简单问题，直接返回
queries, strategy, _ = rewriter.rewrite("什么是Python", good_docs)
# queries = ["什么是Python"], strategy = "none"

# 场景2：抽象问题 → HyDE
queries, strategy, _ = rewriter.rewrite("最新AI芯片", bad_docs)
# queries = ["人工智能芯片的最新发展包括..."]
# strategy = "hyde"
```

---

**备注**：模型生成代码时，请假定 `llm.generate(prompt)` 和 `embeddings.similarity(query, doc)` 已存在。如果 `first_pass_docs` 为 `None`，内部自动调用一次检索。所有 LLM 调用需包含超时与重试（一次）。