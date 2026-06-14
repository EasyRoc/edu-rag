# Edu-RAG 系统流程文档

## 系统架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户交互层                               │
│   Web UI (static/index.html)  │  API (/api/v1/*)  │  CLI       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
    ┌─────────▼──────────┐          ┌──────────▼──────────┐
    │   离线：文档上传与入库  │          │   在线：检索与问答      │
    │   Ingestion Pipeline │          │   LangGraph Pipeline │
    └─────────┬──────────┘          └──────────┬──────────┘
              │                                 │
    ┌─────────▼──────────┐          ┌──────────▼──────────┐
    │  Milvus Lite        │          │  Embedding Model    │
    │  + BM25 混合存储     │◄─────────│  + Reranker + LLM   │
    └────────────────────┘          └─────────────────────┘
```

---

# 第一部分：离线流程 — 文档上传与入库

## 1.1 整体流程

```
用户上传文件
    │
    ▼
┌──────────────────────────────────────────────┐
│  Step 1: Load (加载)                          │
│  根据文件类型选择 Loader，读取为 Document 列表    │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Step 2: Pre-Split (预切分)                   │
│  将大段落按 chunk_size=1024 预拆分，提升清洗质量  │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Step 3: Clean (清洗)                         │
│  五阶段管道：规范化→去噪→结构修复→质量过滤→去重   │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Step 4: Chunk (切片)                         │
│  按策略切分：递归字符 / 语义 / Markdown 标题     │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Step 5: Embed + Insert (向量化+入库)          │
│  BGE Embedding → 存入 Milvus + 重建 BM25 索引  │
└──────────────────────────────────────────────┘
```

## 1.2 入口

| 入口 | 路径 | 说明 |
|------|------|------|
| **API 上传** | `POST /api/v1/documents/upload` | multipart 表单上传文件，附带学科/年级/章节/切片策略 |
| **SQL 导入** | `POST /api/v1/documents/import/sql` | 从外部数据库表导入内容 |
| **CLI 上传** | `edu-rag upload-samples` | 批量上传 `sample_docs/` 目录下的示例文档 |

核心服务类：`services/document_service.py:DocumentService`，核心管道：`ingestion/pipeline.py:IngestionPipeline`。

## 1.3 Step 1: Load（文档加载）

文件：`ingestion/loader.py`

根据文件扩展名选择对应的 LangChain Loader：

| 文件类型 | Loader | 说明 |
|---------|--------|------|
| **PDF** | `PyPDFLoader` | 逐页加载，每页生成一个 Document，metadata 包含 page、source_file、file_type |
| **Markdown** (.md) | `UnstructuredMarkdownLoader` | mode="single"，单文档加载 |
| **纯文本** (.txt) | `TextLoader` | UTF-8 编码加载 |

输出：`list[Document]`，每个 Document 包含 `page_content`（文本内容）和 `metadata`（来源元信息）。

## 1.4 Step 2: Pre-Split（预切分）

文件：`ingestion/cleaner.py:pre_split_docs()`

- 在清洗之前，对**超长段落**先做一次预切分（chunk_size=1024，按段落边界切）
- 目的：清洗模块对短文本处理效果更好，避免长文档的噪声特征被稀释

## 1.5 Step 3: Clean（清洗管道）

文件：`ingestion/cleaner.py:CleaningPipeline`

五阶段管道，顺序执行：

```
Normalize → Denoise → StructureRepair → QualityFilter → Dedup
```

### 1.5.1 Normalize（规范化）

- 去除不可见 Unicode 字符（U+0000-001F，保留 Tab 和换行）
- 去除零宽字符（zero-width space/joiner/non-joiner）
- 去除 BOM 头（`﻿`）
- 折叠多余空格（连续空格→单个空格）
- 折叠多余换行（>3 个连续换行→2 个）

### 1.5.2 Denoise（去噪）

- **页眉页脚**：移除独立页码行（如 "123"、"第X页"、"1/10"）
- **版权声明**：匹配常见版权句式并移除
- **目录页**：检测连续行首为数字+标题的模式，标记为目录并移除
- **PDF 断字符修复**：将英文单词行尾断字（`word-\nword`）修复为完整单词
- **高频内容检测**：统计高频行，过滤在多个文档中出现的内容（如固定模板文字）

### 1.5.3 StructureRepair（结构修复）

- **PDF/TXT 句内换行修复**：将段落内部的错误换行合并，恢复完整句子
- **Markdown 标记剥离**：
  - 去除 `###` 标题标记（保留标题文本）
  - 去除列表符号（`-`, `*`, 数字列表 `1.`）
  - 去除水平分割线 `---`
  - 链接 `[text](url)` → 仅保留 text
  - 去除加粗 `**text**`、斜体 `*text*`、行内代码 `` `code` ``（保留内容）

### 1.5.4 QualityFilter（质量过滤）

对每段文本打分（0.0 ~ 1.0），低于 0.5 的内容丢弃：

| 评分维度 | 权重 | 条件 |
|---------|------|------|
| 长度分 | 0.3 | 内容 > 50 字符 |
| 结构分 | 0.3 | 包含标点符号、合理段落长度 |
| 噪声比 | 0.4 | 噪声字符占比 < 5% |

### 1.5.5 Dedup（去重）

- 计算每段内容的 **MD5 哈希**
- 相同哈希的 Document 只保留第一次出现的
- 跨文档级别的去重（多次上传同一内容会被去重）

### 1.5.6 Metadata 构建

清洗完成后统一构建 metadata：

| 字段 | 说明 |
|------|------|
| `source` | 数据来源标识 |
| `source_id` | 来源 ID |
| `position` | 在原文档中的位置（页码/段落号） |
| `file_name` | 原始文件名 |
| `page` | 页码（PDF 来源） |
| `content_hash` | 内容 MD5 |
| `quality_score` | 清洗质量分 |
| `timestamp` | 处理时间戳 |

## 1.6 Step 4: Chunk（切片）

文件：`ingestion/chunker.py:split_documents()`

支持三种切片策略，由参数 `strategy` 控制：

### 1.6.1 recursive（递归字符切分，默认）

```
RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", "。", ".", " ", ""]
)
```

- 按分隔符优先级依次尝试切分：段落→换行→句号→英文句号→空格→字符
- 每个 chunk 不超过 512 字符，相邻 chunk 重叠 64 字符
- 适合**普通文本、试题、通用文档**

### 1.6.2 semantic（语义切分）

```
SemanticChunker(
    embedding=BAAI/bge-small-zh-v1.5,
    breakpoint_threshold_type="percentile"
)
```

**原理：**
1. 将文本按句子粒度拆分，逐句计算 embedding
2. 计算相邻句子对的余弦距离（1 - 余弦相似度）
3. 将所有相邻距离排序，取**高分位阈值**（percentile）
4. 余弦距离超过阈值的相邻句之间断开，形成语义边界

**兜底机制**：语义切分后，仍超过 `chunk_size=512` 的 chunk 会被 `_enforce_max_chunk_size()` 二次递归切分，解决两个问题：
- **长段落黑洞**：语义连贯的长段落找不到断点，整段变成一个巨大 chunk
- **语义距离漂移**：句1→句2→...→句50，相邻句相似但首尾已完全不同，局部相似掩盖全局漂移

适合**教材正文、概念讲解**等语义连贯的内容。

### 1.6.3 markdown（Markdown 标题切分）

```
MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "header1"),    # H1 标题
        ("##", "header2"),   # H2 标题
        ("###", "header3"),  # H3 标题
    ]
)
```

- 按 Markdown 标题层级拆分，每个标题+内容为一个 chunk
- **非 .md 文件**自动回退到 recursive 策略
- 超长 chunk 仍会二次递归切分
- 适合**结构化教案、笔记、带有层级标题的文档**

切片输出格式（统一封装为字典）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | str | 切片文本内容 |
| `doc_id` | str | 文档 UUID，同文档各切片共享 |
| `subject` | str | 学科（数学/语文/英语...） |
| `grade` | str | 年级（七年级/八年级...） |
| `chapter` | str | 章节名称 |
| `knowledge_point` | str | 知识点 |
| `chunk_type` | str | 切片类型（text） |
| `chunk_index` | int | 切片在文档内的序号 |
| `page` | int | 来源页码 |
| `source_file` | str | 来源文件名 |
| `file_type` | str | 文件类型（pdf/md/txt） |

## 1.7 Step 5: Embed + Insert（向量化与入库）

文件：`core/embeddings.py` + `core/vectorestore.py`

### 1.7.1 向量化（Embedding）

**模型**：`BAAI/bge-small-zh-v1.5`（通过 HuggingFace 加载，支持 HF Mirror 加速）

| 参数 | 值 |
|------|-----|
| 向量维度 | 512 |
| 设备 | CPU |
| 归一化 | `normalize_embeddings=True` |
| Query 前缀 | `为这个句子生成表示以用于检索相关文章：`（BGE 中文检索指令） |

调用 `embed_texts(texts)` 批量编码所有切片的 text 字段，一次性生成 512 维向量。

### 1.7.2 Milvus 存储

**存储引擎**：Milvus Lite（嵌入式，文件数据库 `./milvus_k12.db`）

**Collection Schema** (`k12_knowledge_base`)：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INT64 (auto_id) | 主键，自增 |
| `vector` | FLOAT_VECTOR[512] | 文本 embedding 向量，COSINE 度量 |
| `doc_id` | VARCHAR(64) | 所属文档 UUID |
| `chunk_text` | VARCHAR(8192) | 切片文本内容 |
| `subject` | VARCHAR(32) | 学科 |
| `grade` | VARCHAR(32) | 年级 |
| `chapter` | VARCHAR(128) | 章节 |
| `knowledge_point` | VARCHAR(128) | 知识点 |
| `chunk_type` | VARCHAR(32) | 切片类型 |
| (动态字段) | — | 支持 page、source_file、file_type 等扩展字段 |

**索引**：`IVF_FLAT`，`metric_type="COSINE"`，`nlist=128`

### 1.7.3 BM25 稀疏索引

每次插入/删除操作后，会全量重建 BM25 索引（`_rebuild_bm25_index()`）：

```
1. 从 Milvus 读取所有 chunk_text
2. 使用 bigram 分词器（中文字符滑动窗口，窗口大小=2）
3. 构建 BM25Okapi 索引
```

BM25 索引存储在内存中，用于混合检索（Milvus 向量 + BM25 关键字 → RRF 融合）。

---

# 第二部分：在线流程 — 检索与问答

## 2.1 整体流程

```
用户提问
    │
    ▼
┌──────────────────────────────────────────────┐
│  Node 1: classify (意图分类 + 难度分级)         │
│  Layer1 关键词匹配 → Layer2 LLM 分类            │
│  非教育类 → chitchat                          │
│  教育类   → 继续检索                           │
└──────────────┬───────────────────────────────┘
               │ (educational)
               ▼
┌──────────────────────────────────────────────┐
│  Node 2: retrieve (策略检索)                    │
│  simple → DIRECT 混合检索                       │
│  medium → MULTI_QUERY 多查询融合                │
│  complex → DECOMPOSITION 问题拆解 + 子问题标注   │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Node 3: rerank (重排序)                       │
│  CrossEncoder (bge-reranker-base) 逐对打分     │
│  普通问题单阶段；复杂问题可按子问题重排           │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Node 4: retrieval_gate (检索质量门控)          │
│  三分支决策：accept / retry / abstain           │
└──────┬──────────┬──────────┬─────────────────┘
       │          │          │
   accept      retry     abstain
       │          │          │
       ▼          ▼          ▼
┌──────────┐ ┌────────────┐ ┌──────────────────┐
│ generate │ │retry_planner│ │ abstain          │
│ 普通流式/ │ │ 制定重试策略 │ │ 固定拒答，不调用 │
│ 复杂合成  │ │ → 回到retrieve│ │ LLM 生成        │
└────┬─────┘ └────────────┘ └──────────────────┘
     │
     ▼
┌──────────────────────────────────────────────┐
│  finalize (更新对话历史)                         │
│  追加问答到 conversation_history               │
└──────────────────────────────────────────────┘
```

核心引擎：基于 **LangGraph StateGraph** 的节点图，在 `core/graph.py` 中构建并编译。Graph State 只保存可序列化数据；向量库、重排器等运行时对象通过闭包注入。

## 2.2 Node 1: Classify（意图分类 + 难度分级）

文件：`core/nodes/query_classifier.py`

### 2.2.1 意图分类（两层递进）

**Layer 1: 关键词匹配**（< 1ms）

按优先级顺序检查 5 个意图类别，命中即返回（confidence=1.0）：

| 优先级 | 意图 | 关键词示例 |
|--------|------|----------|
| 1 | `greeting`（问候） | 你好、早上好、hello、老师好 |
| 2 | `command`（指令） | 上传文档、切换学科、清空对话 |
| 3 | `educational`（教育） | 什么是、怎么做、解释一下、总结、概括、公式、定义 |
| 4 | `technical`（技术） | 报错、bug、卡顿、打不开 |
| 5 | `chitchat`（闲聊） | 天气、笑话、你喜欢、你的名字 |

**Layer 2: LLM 分类**（200~800ms，仅在关键词未命中时触发）

- 模型：ChatOpenAI(temperature=0, max_tokens=128, timeout=3s)
- 分类为 6 类：`educational | chitchat | technical | command | greeting | other`
- 正则提取 JSON 响应，带优雅降级
- 分类结果自动保存到 `data/intent_training_data.jsonl`，用于未来训练本地分类器

**路由逻辑**：
- `educational` → 继续进行检索
- 其他意图 → 跳转到 `chitchat` 节点（友好闲聊，引导回学习话题）

### 2.2.2 难度分级（两层递进）

**Layer 1: 规则匹配**

| 难度 | 关键词 |
|------|--------|
| `simple` | 是什么、定义、公式、概念、填空、选择、判断 |
| `complex` | 比较、对比、分析、推导、综合、论述、证明、为什么 |

关键词同时命中时，输出 `medium` 并交由 LLM 判定。

**Layer 2: LLM 判定**（仅在规则不确定时）

- 使用 JSON Mode 结构化输出 `{complexity: "simple|medium|complex", reasoning: "..."}`

输出 `complexity` 字段，用于后续策略选择。

## 2.3 Node 2: Retrieve（策略检索）

文件：`core/strategies/selector.py` + `core/nodes/retriever.py`

### 2.3.1 策略选择

根据意图和难度选择检索策略：

| 难度 | 策略 | 说明 |
|------|------|------|
| `simple` | `DIRECT` | 直接混合检索，单次查询 |
| `medium` | `MULTI_QUERY` | LLM 生成多个查询变体，分别检索后 RRF 融合 |
| `complex` | `DECOMPOSITION` | LLM 拆解为 2~4 个子问题，分别检索，标注来源并合并去重后返回 |

### 2.3.2 底层混合检索（K12VectorStore.hybrid_search）

文件：`core/vectorestore.py`

```
查询文本
    │
    ├──→ Dense 检索（Milvus ANN）
    │    embedding → COSINE 搜索 → top-k 结果
    │    可选过滤：DENSE_MIN_SIMILARITY 阈值
    │    可选过滤：subject / grade 标量过滤
    │
    └──→ Sparse 检索（BM25）
         bigram 分词 → BM25Okapi → top-k 结果
         │
         ▼
    RRF 融合（Reciprocal Rank Fusion, k=60）
    归一化得分 → 返回 top 20 (RETRIEVAL_CANDIDATE_TOP_K)
```

**RRF 公式**：`score(d) = Σ 1/(k + rank_i(d))`

### 2.3.3 高级策略详解

**MULTI_QUERY**（`strategies/multi_query.py`）：
1. LLM 生成 4 个不同表述的查询变体（`MULTI_QUERY_VARIANTS=4`）
2. 每个变体分别执行 `hybrid_search`
3. 所有结果按 RRF 融合，去重
4. 返回 top 20

**DECOMPOSITION**（`strategies/decomposition.py`）：
1. LLM 将复杂问题拆解为 2~4 个子问题（`DECOMPOSITION_MAX_SUB=4`）
2. 每个子问题独立执行 `hybrid_search`
3. 每个候选片段标注 `source_sub_query`，记录“这个 chunk 是被哪个子问题召回的”
4. 所有子问题结果合并，按 chunk id 去重；如果同一个 chunk 被多个子问题召回，合并保留多个来源
5. 返回 `(docs, sub_queries)`，Graph State 会携带 `sub_queries` 供后续子问题感知重排和子答案合成使用

拆解降级：

- 拆出 2 个以上子问题：正常执行 DECOMPOSITION。
- 拆解结果不足或异常：降级到 `MULTI_QUERY`，避免复杂长问题退化成单查询 DIRECT。

**HyDE**（`strategies/hyde.py`，用于重试）：
1. LLM 生成"假设性文档"（Hypothetical Document）——以文档的口吻回答用户问题
2. 用假设文档的 embedding 去检索，而非原始 query
3. 目的：桥接 query 和 document 之间的**词汇鸿沟**（术语不一致）

**Step-Back**（`strategies/step_back.py`，用于重试）：
1. LLM 将具体问题**抽象为更宽泛的概念问题**
   - 例："勾股定理在生活中怎么用？" → "勾股定理的基本概念和应用"
2. 用抽象问题去检索，再用原始问题对比筛选
3. 目的：当具体 query 匹配不到相关文档时，通过上溯概念扩大召回范围

## 2.4 Node 3: Rerank（重排序）

文件：`core/reranker.py` + `core/graph.py`

### 2.4.1 模型

| 配置项 | 值 |
|--------|-----|
| 模型 | `BAAI/bge-reranker-base` |
| 框架 | sentence-transformers CrossEncoder |
| 设备 | CPU |
| 批大小 | 16 |
| 开关 | `ENABLE_RERANKER=true` |

### 2.4.2 处理流程

普通问题路径：

```
候选文档列表 (最多20个)
    │
    ▼
构造 (query, doc_text) 配对
    │
    ▼
CrossEncoder 批量推理 → 原始 logits
    │
    ▼
Sigmoid 归一化 → [0, 1] 区间
    │
    ▼
按 rerank_score 降序排列
    │
    ▼
输出带 rerank_raw_score / rerank_score 的文档列表
```

- 整个过程在 `asyncio.to_thread()` 中执行，不阻塞事件循环
- 如果模型加载失败或被禁用，设置 `reranker_available=False`，文档原样通过

### 2.4.3 复杂问题子问题感知重排

触发条件：

```text
complexity == "complex"
and len(sub_queries) >= 2
and ENABLE_DEEP_COMPLEX_MODE == true
```

处理流程：

```text
DECOMPOSITION 候选 docs
    │
    ├── 按 source_sub_query 分组
    │
    ├── rerank(sub_query, docs_by_sub_query)
    │       每个子问题保留 SUB_RERANK_TOP_K 条
    │
    └── 合并去重
            输出候选列表，保留子问题 rerank_score
```

这样做的目的：避免“只覆盖某个子方向的有效 chunk”因为无法匹配原始复杂问题的全部要求而被压低分。当前实现不再对合并后的候选执行 `rerank(original_query, merged_docs)`，因此 `rerank_score` 保持为“候选对所属子问题的相关性分数”，后续子问题覆盖门控可以直接使用它。RRF 仍只负责候选排序，不参与质量判断。

## 2.5 Node 4: Retrieval Gate（检索质量门控）

文件：`core/retrieval_quality.py:evaluate_retrieval_gate()`

### 2.5.1 五分支决策

```
检索结果
    │
    ├── 无候选文档？ ────────────→ retry (有重试次数) / abstain
    │
    ├── Reranker 不可用？
    │   ├── observe 模式 → accept (仅记录日志)
    │   └── enforce 模式 → abstain
    │
    ├── 复杂问题子问题覆盖不足？
    │   └── YES → retry + complex_repair / abstain
    │
    ├── 质量合格？(达到当前复杂度的 top1 阈值且 relevant_count >= 1)
    │   └── YES → accept
    │
    └── 质量不合格？
        ├── 无相关文档 → retry + HyDE 建议 / abstain
        └── 文档存在但得分低 → retry + Step-Back 建议 / abstain
```

### 2.5.2 关键阈值

| 场景 | top1 接受阈值 | 相关候选阈值 | 重试次数 |
|------|----------------|--------------|----------|
| `simple` / `medium` | `RETRIEVAL_ACCEPT_TOP1_THRESHOLD=0.60` | `RERANKER_RELEVANCE_THRESHOLD=0.50` | `MAX_RETRIES=2` |
| `complex` | `COMPLEX_ACCEPT_TOP1_THRESHOLD=0.45` | `COMPLEX_RELEVANCE_THRESHOLD=0.35` | `COMPLEX_MAX_RETRIES=2` |

`RETRIEVAL_GATE_MODE` 控制重排器不可用时的行为：

- `enforce`：直接 `abstain`，避免无重排质量分时继续生成。
- `observe`：记录“本应拒答”的状态，但允许按旧流程生成，适合灰度观测。

### 2.5.3 质量指标

系统计算以下指标用于决策和日志：

| 指标 | 说明 |
|------|------|
| `candidate_count` | 候选文档数量 |
| `relevant_count` | 相关文档数量（得分 ≥ 当前复杂度相关候选阈值） |
| `distinct_doc_count` | 去重后的文档数 |
| `top1_score` | 最高重排序得分 |
| `topk_mean_score` | top-k 平均得分 |
| `top1_margin` | top1 与 top2 的得分差距 |
| `coverage_ratio` | 复杂问题已覆盖子问题数 / 总子问题数，普通问题为 `None` |
| `covered_subquery_count` | 有达标证据的子问题数量 |
| `missing_subqueries` | 没有达标相关证据的子问题 |
| `weak_subqueries` | 有候选但 top1 未达到复杂问题阈值的子问题 |

## 2.6 Node 5: Generate（流式生成）

文件：`core/nodes/generator.py` + `core/graph.py`

### 2.6.1 上下文构建

普通问题路径：

```
reranked 文档列表 (top 5 = GENERATION_CONTEXT_TOP_K)
    │
    ▼
格式化上下文：[1] chunk_text_1
              [2] chunk_text_2
              ...
    │
    ▼
构建消息列表：
  - SystemMessage: K12 教育助手系统提示词
  - HumanMessage / AIMessage: 对话历史
  - HumanMessage: 当前用户问题 + 上下文
```

复杂问题路径：

```
reranked 文档列表 (top 8 = COMPLEX_CONTEXT_TOP_K)
    │
    ├── 按 source_sub_query 分组
    │
    ├── generate_sub_answers(sub_queries, sub_docs_map)
    │      └── 每个子问题使用 SUB_ANSWER_MAX_TOKENS=512
    │
    └── synthesize_final_answer(original_query, sub_answers, context_docs)
           └── 最终合成使用 SYNTHESIS_MAX_TOKENS=4096
```

复杂路径的触发条件与子问题感知重排一致：`complexity == "complex"`、`sub_queries >= 2`、`ENABLE_DEEP_COMPLEX_MODE=true`。如果未配置 `LLM_API_KEY`，复杂路径会退回 `_mock_answer()`，保证本地 smoke 流程仍可运行。

### 2.6.2 Token 预算管理

`_trim_messages()` 函数：
- 使用 tiktoken (`cl100k_base`) 计算 token 数
- 如果 token 超限（`LLM_MAX_CONTEXT_TOKENS=8192`），**保留 SystemMessage 和最新 HumanMessage**，丢弃最早的历史消息
- 无法使用 tiktoken 时回退到字符数 / 2.5 的估算

### 2.6.3 系统提示词（核心指令）

模型定位为 **"知学助手"** K12 教育辅助 AI：
- 仅根据提供的参考资料回答
- 引用时标注 `[1][2]`
- 用 K12 学生能理解的语言解释
- 参考资料不足时明确说明
- 引导学生进一步探索

### 2.6.4 流式输出

```
LLM.astream() → 逐 token 输出
    │
    ▼
每个 token → StreamQueueRegistry 队列
    │
    ▼
SSE 事件 → 前端逐字渲染
```

- 模型：ChatOpenAI(temperature=0.3, max_tokens=2048, timeout=120s)
- 降级：无 API Key 时使用 `_mock_answer()`（拼接前 3 篇文档内容摘要）
- 复杂问题合成路径目前会把完整合成答案一次性写入流式队列；普通路径仍逐 token 输出。

## 2.7 Chitchat 分支（闲聊应答）

文件：`core/nodes/chitchat.py`

- 非教育类意图的统一处理节点
- 以"知学助手"身份友好回应
- 温和引导学生回到学习话题
- 同样支持流式输出和对话历史

## 2.8 Abstain 分支（拒答）

文件：`core/graph.py`

当检索质量不足且无剩余重试次数时，返回固定拒答：
> 抱歉，我暂时没有检索到足够可靠的资料来回答这个问题。你可以补充教材范围、年级或更具体的问题。

## 2.9 Finalize（对话终结）

文件：`core/graph.py`

- 将当前问答对追加到 `conversation_history`
- 裁剪到 `MAX_ROUNDS * 2 = 20` 条消息，防止状态无限膨胀
- 输出最终状态供 API 返回

## 2.10 重试回路

```
retrieve → rerank → retrieval_gate
              ↑         │
              │    retry │
              │         ▼
              └── retry_planner
                   (制定重试策略:
                    普通问题 Retry1: 原问题 + query variants
                    普通问题 Retry2: HyDE 或 Step-Back
                    复杂问题: complex_repair 按子问题定向修复)
```

最多重试 `MAX_RETRIES=2` 次。每次重试使用不同的策略，避免陷入同一种失败模式。

复杂问题的 `complex_repair` 会保留原始 `sub_queries`，并为每个子问题生成修复动作：

| 子问题状态 | 修复动作 |
|------------|----------|
| `covered` | `direct`，重新直接检索该子问题 |
| `missing` | `hyde`，用子问题生成假设答案补检 |
| `weak` | `step_back`，生成更抽象问题补检 |

修复后的候选仍带 `source_sub_query`，因此后续会继续走子问题感知重排和子答案合成，不会退化成普通单查询生成。

---

# 第三部分：关键技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| Web 框架 | FastAPI | 异步 API，自动生成 Swagger 文档 |
| 图编排 | LangGraph | 有状态图编排，MemorySaver 做本地 checkpoint |
| 向量数据库 | Milvus Lite | 嵌入式，文件存储，无需单独部署 |
| 稀疏检索 | BM25Okapi (rank_bm25) | 内存索引，bigram 中文分词 |
| 融合算法 | RRF (k=60) | 对偶融合，无需调权 |
| Embedding | BAAI/bge-small-zh-v1.5 | 512 维，CPU 推理 |
| Reranker | BAAI/bge-reranker-base | CrossEncoder，sigmoid 归一化 |
| LLM | OpenAI 兼容接口 | 通过 langchain-openai 调用；默认配置在 `config.py` / `.env` 中切换 |
| 数据库 | SQLite + aiosqlite | 业务数据持久化 |
| ORM | SQLAlchemy 2.0 (async) | 异步引擎 + session |
| 流式输出 | SSE (Server-Sent Events) | asyncio.Queue 中转 token |
| 评估 | RAGAS 0.4.x + 检索离线评估 | 答案质量、检索指标、门控校准、自动样本回归 |

---

# 第四部分：数据流向总结

```
                     离线流程
                     ========
  文件/数据库 ──→ Load ──→ PreSplit ──→ Clean ──→ Chunk ──→ Embed ──→ Milvus + BM25
                                                                          │
                                                                          │
                     在线流程                                              │
                     ========                                              │
  用户提问 ──→ 意图分类 ──→ 策略检索 ──→ 混合搜索 ◄─────────────────────────┘
                                    │
                                    ▼
                              重排序 (Reranker)
                                    │
                                    ▼
                              质量门控 ──→ accept ──→ 普通生成 / 复杂合成 ──→ SSE 返回用户
                                    │
                                    ├──→ retry ──→ 回到策略检索
                                    │
                                    └──→ abstain ──→ 拒答返回
                                    │
                                    └──→ 成功教育类 RAG 问答 ──→ auto_eval_samples
```

---

# 第五部分：RAGAS 评估体系

## 5.1 评估定位与架构

RAGAS 评估体系是整个 RAG 系统的**质量监控层**，对系统的检索质量和生成质量进行量化度量。评估分为两个维度：

| 维度 | 评估内容 | 评测方式 |
|------|---------|---------|
| **生成质量** | 答案是否忠实于上下文、是否与问题相关 | RAGAS 四大指标（LLM + Embedding 联合评估） |
| **检索质量** | 召回率、精确率、门控误判率、延迟分位数 | 离线标注集回放 + 排序指标计算 |

整体架构：

```
┌──────────────────────────────────────────────────────────────────┐
│                        RAGAS 评估体系                             │
│                                                                  │
│  ┌─────────────────────┐    ┌──────────────────────────────┐    │
│  │  测试集生成           │    │  评估执行                      │    │
│  │  TestSetGenerator    │    │  pipeline.py                 │    │
│  │                      │    │                              │    │
│  │  A. 向量库采样+LLM生成 │───→│  1. 加载 Dataset              │    │
│  │  B. QA历史筛选+补全   │    │  2. RAGASEvaluator 逐样本评估  │    │
│  │  C. 自动沉淀样本      │    │  3. 聚合评分 + 打印报告        │    │
│  │  D. 手动JSONL/JSON   │    │  4. 持久化到 EvaluationRecord  │    │
│  └─────────────────────┘    └──────────────┬───────────────┘    │
│                                            │                     │
│  ┌─────────────────────┐    ┌──────────────▼───────────────┐    │
│  │  检索离线评估         │    │  自动样本沉淀                  │    │
│  │  retrieval_evaluator │    │  AutoEvalSample              │    │
│  │                      │    │                              │    │
│  │  回放标注集            │    │  RAG 成功后自动写入样本        │    │
│  │  计算 recall/precision│   │  供后续批量评估使用             │    │
│  │  门控阈值校准          │    │                              │    │
│  └─────────────────────┘    └──────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

## 5.2 RAGAS 四大核心指标

文件：`evaluation/ragas_evaluator.py`

RAGAS (Retrieval Augmented Generation Assessment) 是一个专门评估 RAG 系统的开源框架，本项目使用 0.4.x 版本。四个指标需要 LLM 作为评判器（Judge），其中 `answer_relevancy` 还需要 Embedding 模型计算语义相似度。

### 5.2.1 Faithfulness（忠实度）

```
评估问题：生成的答案中有多少内容可以从检索到的上下文中推断出来？

评估流程：
  1. LLM 将回答拆解为一组独立的"陈述"（claims）
  2. 对每个陈述，LLM 判断其是否能从上下文中找到依据
  3. Faithfulness = 有依据的陈述数 / 总陈述数
```

| 特征 | 说明 |
|------|------|
| 需要 LLM | ✅ |
| 需要 Embedding | ❌ |
| 需要 ground_truth | ❌ |
| 典型阈值 | ≥ 0.80 |
| 低分原因 | LLM 幻觉、上下文无关编造、过度发挥 |

### 5.2.2 Answer Relevancy（答案相关性）

```
评估问题：生成的答案与用户问题的相关程度如何？

评估流程：
  1. LLM 根据回答反推出可能的几个"反向问题"（reverse questions）
  2. Embedding 计算每个反向问题与原始问题的余弦相似度
  3. Answer Relevancy = 所有反向问题相似度的均值
```

| 特征 | 说明 |
|------|------|
| 需要 LLM | ✅ |
| 需要 Embedding | ✅（BAAI/bge-small-zh-v1.5） |
| 需要 ground_truth | ❌ |
| 典型阈值 | ≥ 0.70 |
| 低分原因 | 答案跑题、答非所问、包含无关信息 |

### 5.2.3 Context Precision（上下文精度）

```
评估问题：检索到的上下文中，有多少是真正相关的？（信号噪声比）

评估流程：
  1. 在上下文中识别与 ground_truth 相关的片段
  2. 按检索排序位置加权：排名靠前的相关片段贡献更大
  3. Context Precision = Σ(相关片段在位置k的命中率) / 相关片段总数
```

| 特征 | 说明 |
|------|------|
| 需要 LLM | ✅ |
| 需要 Embedding | ❌ |
| 需要 ground_truth | ✅（必须有标准答案或参考答案） |
| 典型阈值 | ≥ 0.70 |
| 低分原因 | 检索噪声大、召回了大量无关文档、BM25+Dense 未能互补 |

### 5.2.4 Context Recall（上下文召回率）

```
评估问题：ground_truth 所需的信息，检索上下文是否都覆盖到了？

评估流程：
  1. LLM 分析 ground_truth，拆解出所需的关键信息点
  2. 判断每个信息点是否能在检索到的上下文中找到
  3. Context Recall = 能找到的信息点数 / 总信息点数
```

| 特征 | 说明 |
|------|------|
| 需要 LLM | ✅ |
| 需要 Embedding | ❌ |
| 需要 ground_truth | ✅（必须有标准答案或参考答案） |
| 典型阈值 | ≥ 0.70 |
| 低分原因 | 检索遗漏、切分策略不当导致关键信息丢失 |

### 5.2.5 指标选择策略

```
                    ┌─────────────────────────────┐
                    │   是否有 ground_truth？       │
                    └────────────┬────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │ YES                           │ NO
                 ▼                               ▼
    ┌────────────────────────┐    ┌────────────────────────────┐
    │ 全量指标 (4项)           │    │ 无需标准答案指标 (2项)        │
    │ - faithfulness          │    │ - faithfulness             │
    │ - answer_relevancy      │    │ - answer_relevancy         │
    │ - context_precision     │    │                            │
    │ - context_recall        │    │ (自动跳过 context_precision │
    └────────────────────────┘    │  和 context_recall)          │
                                  └────────────────────────────┘
```

代码实现：`_prepare_dataset_and_metric_names()` 自动检测 dataset 是否有 `reference`/`ground_truth` 列，无则自动跳过依赖标准答案的指标。

## 5.3 RAGAS 技术适配细节

### 5.3.1 LLM Judge 配置

RAGAS 使用项目本身的 LLM 作为评判器（Judge），通过 `ragas.llms.llm_factory` 封装：

```
项目配置                     RAGAS 适配
─────────                    ──────────
LLM_API_KEY  ──────────→  OpenAI client  ──→  llm_factory()  ──→  RAGAS Metrics
LLM_BASE_URL               (api_key, base_url)     (model, client,
LLM_MODEL                                            max_tokens=RAGAS_LLM_MAX_TOKENS)
```

关键配置 `RAGAS_LLM_MAX_TOKENS=8192`：Faithfulness 等指标需要 LLM 输出较长的 JSON 结构化判断结果，默认值太小会导致 JSON 截断、评估失败。

### 5.3.2 Embedding 适配层

文件：`ragas_evaluator.py:_LangChainStyleEmbeddingsAdapter`

RAGAS 0.4.x 的 `AnswerRelevancy` 指标内部使用了 LangChain 风格的 `embed_query`/`embed_documents` 接口，而项目加载的 `ragas.embeddings.HuggingFaceEmbeddings` 提供的是 `embed_text`/`embed_texts`。适配器做桥接转换：

```
ragas HuggingFaceEmbeddings          _LangChainStyleEmbeddingsAdapter
─────────────────────────────        ────────────────────────────────
embed_text(text) → list[float]  ──→  embed_query(text) → list[float]
embed_texts(texts) → list[...]  ──→  embed_documents(texts) → list[list[float]]
```

### 5.3.3 异步隔离执行

RAGAS 内部使用 `asyncio.run()`，如果在 FastAPI 的事件循环中直接调用会导致嵌套事件循环冲突。解决方法：

```python
result = await asyncio.to_thread(
    ragas_evaluate,   # 在独立线程中执行
    dataset=dataset,
    metrics=selected,
)
```

## 5.4 评估数据流

### 5.4.1 数据源

文件：`evaluation/dataset_builder.py:EvalDatasetBuilder`

评估数据集可从四个来源构建：

| 数据源 | 方法 | 说明 |
|--------|------|------|
| **业务数据库** | `from_db()` | 从 `qa_records` 表提取历史问答，支持按学科/用户/feedback 过滤 |
| **测试文件** | `from_file()` | JSON/JSONL 文件，需包含 question、answer、contexts，可含 ground_truth |
| **自动沉淀样本** | `from_auto_samples()` | 从 `auto_eval_samples` 表读取门控通过的 RAG 问答样本 |
| **手动构建** | `from_manual()` | 直接传入 question/answer/contexts 列表 |

最终输出统一为 HuggingFace `Dataset` 格式，包含列：`question`、`answer`、`contexts`（`list[str]`）、`ground_truth`/`reference`（可选）。

### 5.4.2 评估执行流程

文件：`evaluation/pipeline.py`

```
evaluation/pipeline.py:run_evaluation()
│
├── 1. 初始化 RAGASEvaluator
│       ├── 构建 LLM Judge（复用项目 LLM 配置）
│       └── 加载 Embedding 模型（BAAI/bge-small-zh-v1.5）
│
├── 2. 数据集预处理（_prepare_dataset_and_metric_names）
│       ├── 检测是否有 reference/ground_truth 列
│       ├── 无则跳过 context_precision / context_recall
│       └── ground_truth → 镜像为 reference（RAGAS 0.4.x 要求）
│
├── 3. 构建 RAGAS 指标实例（_build_ragas_metrics）
│       ├── Faithfulness(llm=ragas_llm)
│       ├── AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
│       ├── ContextPrecision(llm=ragas_llm)   ← 需 reference
│       └── ContextRecall(llm=ragas_llm)       ← 需 reference
│
├── 4. 执行 ragas.evaluate()（asyncio.to_thread 隔离）
│
├── 5. 解析 to_pandas() → EvalResult
│       ├── 聚合得分：scores = {metric_name: mean_value}
│       ├── 逐样本得分：samples = [EvalSample(question, answer, scores)]
│       └── 附加元信息：任务名、时间戳、耗时、模型配置
│
├── 6. 持久化到 EvaluationRecord 表
│       ├── task_name、metrics、scores、samples
│       ├── config_snapshot（评估时的系统配置快照）
│       └── elapsed_seconds
│
└── 7. 打印格式化报告（带进度条可视化）
```

### 5.4.3 实时评估模式

文件：`evaluation/pipeline.py:run_live_evaluation()`

区别于离线评估使用现成的 answer + contexts，实时评估是端到端的：

```
问题列表
  │
  ▼
对每个问题调用 RAG 系统完整流程:
  classify → retrieve → rerank → gate → generate
  │
  ▼
收集每个问题的 answer + contexts
  │
  ▼
构建 Dataset → run_evaluation() → RAGAS 评分
```

适用于：上线前全链路测试、夜间回归测试。

## 5.5 测试集生成

文件：`evaluation/testset_generator.py:TestSetGenerator`

### 5.5.1 三种生成方式

| 方式 | 说明 |
|------|------|
| **A. 向量库采样 + LLM 生成** | 从 Milvus 随机采样文档片段，LLM 据此生成 question + ground_truth |
| **B. QA 历史筛选 + 补全** | 从 qa_records 筛选高质量问答，LLM 补全 ground_truth |
| **C. 自动沉淀** | RAG 在线服务通过门控后，自动写入 auto_eval_samples 表 |

### 5.5.2 方式 A 详细流程

```
from_vectorestore()
│
├── 1. 从 Milvus 随机采样文档（支持 subject/grade 过滤）
│
├── 2. 每段文档调用 LLM，Prompt 包含：
│       - 文档内容（≤3000 字符）
│       - 学科/年级信息
│       - 三个难度要求：simple（事实检索）、medium（概念解释）、complex（综合推理）
│
├── 3. LLM 返回 JSON 数组：
│       [{"question": "...", "ground_truth": "...", "complexity": "simple/medium/complex", "question_type": "定义题/计算题/..."}]
│
└── 4. 解析 JSON 响应（支持 ```json 代码块、裸数组三种容错解析）
```

### 5.5.3 测试集校验

`TestSetGenerator.validate()` 执行以下检查：

| 校验项 | 说明 |
|--------|------|
| 问题去重 | 相同问题文本（忽略大小写）只保留第一次出现 |
| 空字段检测 | 统计缺失 question、ground_truth、contexts 的条目 |
| 分布统计 | 按 complexity / subject / grade / question_type 统计分布 |
| 输出报告 | 去重前后数量、各维度分布概况 |

## 5.6 检索离线评估与门控校准

文件：`evaluation/retrieval_evaluator.py`

这是独立于 RAGAS 生成质量评估的另一套评估维度，专注于**检索管道和质量门控**的性能。

### 5.6.1 检索排序指标

对于有 `relevant_chunk_ids` 标注的样本（已知哪些 chunk 是真正相关的），计算标准 IR 指标：

| 指标 | 计算方式 |
|------|---------|
| `recall@5` / `recall@10` / `recall@20` | top-k 结果中命中的相关 chunk 占全部相关 chunk 的比例 |
| `precision@5` | top-5 结果中相关 chunk 的比例 |
| `mrr@10` (Mean Reciprocal Rank) | 第一个相关 chunk 排名的倒数均值 |
| `ndcg@10` (Normalized DCG) | 折损累计增益，排名越靠前的相关 chunk 权重越高 |

### 5.6.2 门控决策评估

对标注为 `answerable=true` 和 `answerable=false` 的样本分别评估门控行为：

```
                  实际 answerable          实际 unanswerable
               ┌──────────────────┬──────────────────────┐
  门控 accept   │  True Positive   │  False Accept (误接受) │
  门控 abstain  │  False Reject    │  True Negative        │
  门控 retry    │  重试后可能恢复    │  重试后正确拒绝         │
               └──────────────────┴──────────────────────┘
```

| 聚合指标 | 说明 |
|---------|------|
| `false_accept_rate` | unanswerable 样本被门控放行的比例（越低越好） |
| `false_reject_rate` | answerable 样本被门控拒绝的比例（越低越好） |
| `abstention_accuracy` | unanswerable 样本被正确拒答的比例（越高越好） |
| `retry_recovery_rate` | 初次被拒但重试后恢复的比例 |

### 5.6.3 门控阈值校准

文件：`retrieval_evaluator.py:calibrate_thresholds()`

在给定错误接受率预算下，网格搜索最优阈值组合：

```
calibrate_thresholds(case_results, max_false_accept_rate=0.05)
│
├── 遍历 top1_threshold ∈ [0.00, 0.05, ..., 1.00]  (21 档)
│   └── 遍历 relevant_threshold ∈ [0.00, 0.05, ..., 1.00]  (21 档)
│       ├── 模拟门控决策
│       ├── 计算 false_accept_rate（必须 ≤ max_false_accept_rate）
│       └── 计算 answerable_accept_rate
│
└── 选择最优组合（最大化可回答接受率，最小化误接受率，偏向当前配置附近的值）
```

### 5.6.4 分维度切片统计

报告支持按 `subject`、`grade`、`complexity`、`strategy`、`retry_count` 五个维度切片，每个切片统计样本数和 accept/abstain 比率，便于定位特定场景下的门控短板。

## 5.7 自动样本沉淀

在线问答完成后，`RAGService` 会尝试把高质量教育类 RAG 问答写入 `auto_eval_samples` 表。采集失败只记录 warning，不影响用户问答响应。

采集条件：

| 条件 | 要求 |
|------|------|
| 意图 | `intent == "educational"` |
| 门控 | `retrieval_decision.action == "accept"` |
| 答案 | 最终答案非空，且不是固定拒答文案 |
| 引用 | `references` 至少包含一条带文本的引用 |
| 异常 | Graph 执行无异常 |

写入字段：

| 字段 | 说明 |
|------|------|
| `question` / `answer` / `contexts` | 问答核心数据 |
| `subject` / `grade` / `complexity` | 学科/年级/难度元信息 |
| `session_id` / `user_id` / `qa_record_id` | 回溯追踪信息 |
| `retrieval_decision` | 门控决策记录 |
| `retrieval_metrics` | 检索指标（candidate_count、relevant_count、top1_score 等） |
| `retrieval_attempts` | 重试历史（每次尝试的策略、候选数、指标和门控结果） |
| `latency_ms` | 端到端延迟 |

CLI 命令 `evaluation/cli.py evaluate-auto` 可直接从这批自动沉淀样本批量运行 RAGAS 评估：

```bash
python evaluation/cli.py evaluate-auto --limit 50 --subject math
```

项目根目录 CLI 对这个能力做了更短的封装：

```bash
./edu-rag as
./edu-rag as --limit 100 --subject 数学 --grade 七年级
./edu-rag as --metrics faithfulness,answer_relevancy --no-save
```

前端 **效果评估** 页面也支持 **手动输入** 与 **自动测试集** 两种模式切换；自动测试集模式调用 `/api/v1/evaluation/from-auto`，可按最近样本数量、学科、年级过滤。

## 5.8 CLI 命令速查

文件：`evaluation/cli.py`

| 命令 | 说明 |
|------|------|
| `evaluate --from-db --limit 50` | 从 QA 历史记录提取数据集并评估 |
| `evaluate --from-file test.jsonl` | 从 JSON/JSONL 文件加载数据集评估 |
| `evaluate --from-file test.jsonl --live` | 实时模式：先通过 RAG 系统回答再评估 |
| `evaluate-auto --limit 50` | 从自动沉淀样本批量评估 |
| `./edu-rag as --limit 50` | 根目录项目 CLI 封装，等价于从自动沉淀样本评估 |
| `generate --subject math --count 30` | LLM 生成测试集（question + ground_truth） |
| `validate --file test.jsonl` | 校验测试集格式、去重、输出分布统计 |
| `export --min-feedback 1 --limit 50` | 从 QA 历史导出测试集（LLM 补全 ground_truth） |
| `retrieval-evaluate --from-file cases.jsonl` | 离线检索评估（召回率、门控误判率） |
| `retrieval-calibrate --from-file cases.jsonl` | 门控阈值校准（网格搜索最优组合） |

### 5.8.1 使用示例

```bash
# 基于最近 50 条 QA 记录评估 faithfulness + answer_relevancy
python evaluation/cli.py evaluate --from-db --limit 50 \
    --metrics faithfulness,answer_relevancy --save

# 从自动沉淀样本评估全量指标（含 context precision/recall，如果样本有 ground_truth）
python evaluation/cli.py evaluate-auto --limit 100 --subject math

# 用 LLM 生成 30 道数学测试题
python evaluation/cli.py generate --subject math --grade junior --count 30 \
    --output data/test_sets/math_v2.jsonl

# 离线检索评估：回放标注集，检查召回率和门控行为
python evaluation/cli.py retrieval-evaluate --from-file data/test_sets/retrieval_cases.jsonl

# 门控阈值校准：在 5% 错误接受率预算下推荐最优阈值
python evaluation/cli.py retrieval-calibrate --from-file data/test_sets/retrieval_cases.jsonl \
    --max-false-accept-rate 0.05
```

## 5.9 RAGAS 评估体系的文件索引

| 文件 | 职责 |
|------|------|
| `evaluation/ragas_evaluator.py` | RAGAS 核心评估器：LLM/Embedding 适配、指标构建、批量+单样本评估 |
| `evaluation/pipeline.py` | 评估流水线：离线评估 + 实时评估 + 结果持久化 + 报告打印 |
| `evaluation/dataset_builder.py` | 数据集构建器：从 DB / 文件 / 自动样本 / 手动四种来源构建 Dataset |
| `evaluation/testset_generator.py` | 测试集生成器：LLM 从向量库文档生成问答对 + QA 历史补全 ground_truth |
| `evaluation/retrieval_evaluator.py` | 检索离线评估器：排序指标 + 门控决策评估 + 阈值校准 |
| `evaluation/schemas.py` | 评估数据模型：EvalSample、EvalResult、JSON 序列化函数 |
| `evaluation/cli.py` | 评估 CLI 入口：evaluate/evaluate-auto/generate/validate/export + 检索评估命令 |
| `models/db_models.py` | 持久化模型：AutoEvalSample（自动沉淀样本表）、EvaluationRecord（评估结果表） |
| `config.py` | 评估与门控配置：RAGAS_LLM_MAX_TOKENS、RETRIEVAL_ACCEPT_TOP1_THRESHOLD、COMPLEX_ACCEPT_TOP1_THRESHOLD、RETRIEVAL_GATE_MODE |
