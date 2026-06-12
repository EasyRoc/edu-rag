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
│  complex → DECOMPOSITION 问题拆解               │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Node 3: rerank (重排序)                       │
│  CrossEncoder (bge-reranker-base) 逐对打分     │
│  Sigmoid 归一化 → 按相关性排序                   │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Node 4: retrieval_gate (检索质量门控)          │
│  四分支决策：accept / retry / abstain           │
└──────┬──────────┬──────────┬─────────────────┘
       │          │          │
   accept      retry     abstain
       │          │          │
       ▼          ▼          ▼
┌──────────┐ ┌────────────┐ ┌──────────────────┐
│ generate │ │retry_planner│ │ abstain          │
│ 流式生成  │ │ 制定重试策略 │ │ "抱歉，暂无可靠   │
│ 答案     │ │ → 回到retrieve│ │  资料..."       │
└────┬─────┘ └────────────┘ └──────────────────┘
     │
     ▼
┌──────────────────────────────────────────────┐
│  Node 8: finalize (更新对话历史)                │
│  追加问答到 conversation_history               │
└──────────────────────────────────────────────┘
```

核心引擎：基于 **LangGraph StateGraph** 的 8 节点图，在 `core/graph.py` 中构建并编译。

## 2.2 Node 1: Classify（意图分类 + 难度分级）

文件：`nodes/query_classifier.py`

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

文件：`core/strategies/selector.py` + `nodes/retriever.py`

### 2.3.1 策略选择

根据意图和难度选择检索策略：

| 难度 | 策略 | 说明 |
|------|------|------|
| `simple` | `DIRECT` | 直接混合检索，单次查询 |
| `medium` | `MULTI_QUERY` | LLM 生成多个查询变体，分别检索后 RRF 融合 |
| `complex` | `DECOMPOSITION` | LLM 拆解为 2~4 个子问题，分别检索，合并去重后返回 |

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
3. 所有子问题结果合并，按 doc_id 去重
4. 返回 top 20

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

文件：`core/reranker.py`

### 2.4.1 模型

| 配置项 | 值 |
|--------|-----|
| 模型 | `BAAI/bge-reranker-base` |
| 框架 | sentence-transformers CrossEncoder |
| 设备 | CPU |
| 批大小 | 16 |
| 开关 | `ENABLE_RERANKER=true` |

### 2.4.2 处理流程

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

## 2.5 Node 4: Retrieval Gate（检索质量门控）

文件：`core/retrieval_quality.py:evaluate_retrieval_gate()`

### 2.5.1 四分支决策

```
检索结果
    │
    ├── 无候选文档？ ────────────→ retry (有重试次数) / abstain
    │
    ├── Reranker 不可用？
    │   ├── observe 模式 → accept (仅记录日志)
    │   └── enforce 模式 → abstain
    │
    ├── 质量合格？(top1_score >= 0.60 且 relevant_count >= 1)
    │   └── YES → accept
    │
    └── 质量不合格？
        ├── 无相关文档 → retry + HyDE 建议 / abstain
        └── 文档存在但得分低 → retry + Step-Back 建议 / abstain
```

### 2.5.2 关键阈值

| 阈值 | 默认值 | 说明 |
|------|--------|------|
| `RERANKER_RELEVANCE_THRESHOLD` | 0.50 | rerank 得分 ≥ 此值视为"相关" |
| `RETRIEVAL_ACCEPT_TOP1_THRESHOLD` | 0.60 | top1 得分 ≥ 此值快速放行 |
| `MAX_RETRIES` | 2 | 最大重试次数 |

### 2.5.3 质量指标

系统计算以下指标用于决策和日志：

| 指标 | 说明 |
|------|------|
| `candidate_count` | 候选文档数量 |
| `relevant_count` | 相关文档数量（得分 ≥ 0.50） |
| `distinct_doc_count` | 去重后的文档数 |
| `top1_score` | 最高重排序得分 |
| `topk_mean_score` | top-k 平均得分 |
| `top1_margin` | top1 与 top2 的得分差距 |
| `coverage_ratio` | 相关文档占候选的比例 |

## 2.6 Node 5: Generate（流式生成）

文件：`nodes/generator.py`

### 2.6.1 上下文构建

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

## 2.7 Node 6: Chitchat（闲聊应答）

文件：`nodes/chitchat.py`

- 非教育类意图的统一处理节点
- 以"知学助手"身份友好回应
- 温和引导学生回到学习话题
- 同样支持流式输出和对话历史

## 2.8 Node 7: Abstain（拒答）

文件：`graph.py`

当检索质量不足且无剩余重试次数时，返回固定拒答：
> 抱歉，我暂时没有检索到足够可靠的资料来回答这个问题。建议您换个问法试试，或者上传相关的教材资料后再次提问。

## 2.9 Node 8: Finalize（对话终结）

文件：`graph.py`

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
                    Retry1: 盲扩召回 (多查询融合)
                    Retry2+: HyDE 或 Step-Back)
```

最多重试 `MAX_RETRIES=2` 次。每次重试使用不同的策略，避免陷入同一种失败模式。

---

# 第三部分：关键技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| Web 框架 | FastAPI | 异步 API，自动生成 Swagger 文档 |
| 图编排 | LangGraph | 8 节点有状态图，MemorySaver 做 checkpoint |
| 向量数据库 | Milvus Lite | 嵌入式，文件存储，无需单独部署 |
| 稀疏检索 | BM25Okapi (rank_bm25) | 内存索引，bigram 中文分词 |
| 融合算法 | RRF (k=60) | 对偶融合，无需调权 |
| Embedding | BAAI/bge-small-zh-v1.5 | 512 维，CPU 推理 |
| Reranker | BAAI/bge-reranker-base | CrossEncoder，sigmoid 归一化 |
| LLM | DeepSeek-V4-Flash (兼容 OpenAI API) | 通过 langchain-openai 调用 |
| 数据库 | SQLite + aiosqlite | 业务数据持久化 |
| ORM | SQLAlchemy 2.0 (async) | 异步引擎 + session |
| 流式输出 | SSE (Server-Sent Events) | asyncio.Queue 中转 token |
| 评估 | RAGAS | retrieval/relevance 指标评估 |

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
                              质量门控 ──→ accept ──→ LLM 流式生成 ──→ SSE 返回用户
                                    │
                                    ├──→ retry ──→ 回到策略检索
                                    │
                                    └──→ abstain ──→ 拒答返回
```
