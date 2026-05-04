---
# K12 教育 RAG 系统 —— 面试讲解手册

> 面向面试官的项目复盘与讲解稿。建议讲解顺序：**自我介绍项目 → 架构总览 → 核心链路 → 技术亮点 → 难点与取舍 → 可扩展方向 → 兜底问答**。
> 建议总时长：10~15 分钟主讲 + 10 分钟追问。

---

## 目录

- [1. 一分钟电梯演讲（背熟）](#1-一分钟电梯演讲背熟)
- [2. 项目背景与业务价值](#2-项目背景与业务价值)
- [3. 技术选型与依据](#3-技术选型与依据)
- [4. 整体架构](#4-整体架构)
- [5. 核心链路一：文档入库 Pipeline](#5-核心链路一文档入库-pipeline)
- [6. 核心链路二：RAG 问答 Pipeline（重点）](#6-核心链路二rag-问答-pipeline重点)
- [7. 技术亮点深入](#7-技术亮点深入)
  - [7.1 混合检索 + RRF 融合](#71-混合检索--rrf-融合)
  - [7.2 Adaptive RAG（自适应检索）](#72-adaptive-rag自适应检索)
  - [7.3 Corrective RAG（纠正式检索）](#73-corrective-rag纠正式检索)
  - [7.4 LangGraph 工作流编排](#74-langgraph-工作流编排)
  - [7.5 SSE 流式输出](#75-sse-流式输出)
- [8. 难点与取舍](#8-难点与取舍)
- [9. 性能与指标](#9-性能与指标)
- [10. 个人贡献与扩展方向](#10-个人贡献与扩展方向)
- [11. 高频面试追问（准备好答案）](#11-高频面试追问准备好答案)

---

## 1. 一分钟电梯演讲（背熟）

> "我做了一个面向 K12 教育场景的 RAG 智能问答系统。技术栈是 **FastAPI + LangGraph + Milvus Lite + BGE Embedding + 阿里百炼 qwen-plus**。
>
> 它解决的核心问题是：传统关键词搜索不理解语义，比如学生搜『勾股定理的公式』，教材里写的是『直角三角形两直角边平方和等于斜边平方』，老搜索引擎是匹配不到的。
>
> 我用 RAG 把教材向量化存到 Milvus，问答时做**语义检索 + BM25 关键词检索的混合召回**，再用 **RRF 算法融合**结果，最后把 Top-K 片段喂给大模型生成回答。
>
> 在工程上还做了三件事让它不止是个 Demo：第一，**Adaptive RAG 根据问题复杂度动态调整 top_k**；第二，**Corrective RAG 自动评估回答质量**，不达标就扩大检索范围重试；第三，用 **LangGraph 把整个流程拆成节点**，有条件边做纠错分支，方便扩展。"

---

## 2. 项目背景与业务价值

### 2.1 业务痛点
| 痛点 | 传统方案的局限 |
|------|------------|
| 教材/试题库海量，学生查找困难 | Elasticsearch、数据库 `LIKE` 只能做关键词匹配，不理解"方程 ≈ 等式" |
| 纯大模型回答会**幻觉** | LLM 没见过学校私有教材，回答可能胡编；且训练数据截断后新内容无法获取 |
| 老师想做个性化推荐/学情分析 | 业务数据库没有语义信息，无法基于"薄弱知识点"做推荐 |

### 2.2 RAG 方案的核心价值
1. **知识可控**：回答基于私有教材库，源头是学校自己的数据
2. **降低幻觉**：LLM 看到原文再回答，瞎编概率大幅降低
3. **知识热更新**：换文档就行，不需要重新训练模型
4. **成本低**：不是微调，不需要 GPU 集群

> 讲解技巧：**业务价值要讲出"为什么这么做"**，不要只背技术名词。

---

## 3. 技术选型与依据

| 组件 | 选型 | 为什么选它 |
|------|------|-----------|
| Web 框架 | **FastAPI** | 异步原生支持；自动生成 Swagger；性能接近 Node.js |
| 向量数据库 | **Milvus Lite** | 单文件 `.db` 存储，零部署；API 与 Standalone 兼容，未来无缝升级分布式 |
| Embedding | **BGE-small-zh-v1.5** | 中文开源最佳之一；512 维比 large 版本快，效果接近；离线部署、数据不出本机 |
| LLM | **阿里百炼 qwen-plus** | OpenAI 兼容 API，国内调用稳定、低延迟、中文能力强 |
| 关键词检索 | **rank_bm25** | 纯 Python 实现，无需 ES；与向量检索互补 |
| 流程编排 | **LangGraph** | 用**有向图**建模 RAG 流程，条件边天然适合纠错分支，比链式 LangChain 更可控 |
| 业务存储 | **SQLite + SQLAlchemy（async）** | 单文件、零配置；`aiosqlite` 驱动适配 FastAPI 异步 |
| 文档解析 | `pypdf` / `unstructured` / `TextLoader` | 按后缀分派：PDF、MD、TXT 多格式覆盖 |

> 面试官常问："为什么不用 ChromaDB / FAISS？"  
> 回答：**Milvus Lite 兼顾轻量部署和生产可扩展性**——今天是文件，明天可以直接切到 Standalone 集群，代码不用改。FAISS 没有 filter 和元数据能力，ChromaDB 生产生态不如 Milvus。

---

## 4. 整体架构

### 4.1 分层架构

```
┌──────────────────────────────────────────────────┐
│  用户层：前端 UI（static/index.html）/ Swagger   │
├──────────────────────────────────────────────────┤
│  API 层（FastAPI Router）                         │
│   rag.py / documents.py / knowledge.py / analytics.py │
├──────────────────────────────────────────────────┤
│  服务层（Service）                                 │
│   RAGService / DocumentService / AnalyticsService │
├──────────────────────────────────────────────────┤
│  核心 RAG 引擎（core/）                            │
│   ┌──────────────────────────────────────────┐   │
│   │  LangGraph 工作流（graph.py）            │   │
│   │   ├── classify    Adaptive RAG 分类节点 │   │
│   │   ├── retrieve    混合检索节点          │   │
│   │   ├── generate    LLM 生成节点          │   │
│   │   ├── evaluate    质量评估节点          │   │
│   │   └── re_retrieve Corrective 重试节点   │   │
│   └──────────────────────────────────────────┘   │
├──────────────────────────────────────────────────┤
│  数据层                                           │
│   Milvus Lite（稠密向量）                         │
│   BM25（稀疏关键词）                              │
│   SQLite（业务数据）                              │
│   uploaded_docs/（原始文件）                     │
└──────────────────────────────────────────────────┘
```

### 4.2 两条核心数据流
- **写链路**：用户上传文档 → 加载 → 切片 → 向量化 → Milvus 入库 + BM25 重建 → 更新文档状态
- **读链路**：用户提问 → LangGraph 编排（分类 → 混合检索 → LLM 生成 → 评估 →（可能重试）→ 返回）

> 讲解时务必画出这两张图，能显著加分。

---

## 5. 核心链路一：文档入库 Pipeline

讲解路径：`api/documents.py` → `services/document_service.py` → `ingestion/pipeline.py` → `ingestion/loader.py` + `ingestion/chunker.py` → `core/vectorestore.py`

### 5.1 处理步骤

```
1. 保存文件到 uploaded_docs/   （避免原始文件丢失）
2. SQLite 建文档记录（status=processing）
3. loader.py  按后缀分派加载器：
     .pdf → PyPDFLoader
     .md  → UnstructuredMarkdownLoader
     .txt → TextLoader
4. chunker.py 三种切片策略：
     recursive（默认）：RecursiveCharacterTextSplitter，chunk=512、overlap=64
     markdown：按 #/##/### 标题层级切
     semantic：用 embedding 相似度找语义断点（可选）
5. embeddings.py 批量向量化（BGE, 512 维, 归一化）
6. vectorestore.insert_chunks：
     ① Milvus 批量 insert（schema 含 subject/grade/chapter/knowledge_point 元数据）
     ② 查询 Milvus 所有数据 → 分词 → 重建 BM25 索引
7. 更新文档状态 completed
```

### 5.2 关键设计细节（可展示）
- **元数据打平**：学科 / 年级 / 章节 / 知识点 作为 Milvus scalar field 存储，支持 `subject == '数学'` 这种结构化过滤
- **中文分词**：BM25 用**字符二元组（bigram）**分词，避免中文没空格带来的漂移问题（`_tokenize` 函数）
- **失败兜底**：加载、切片、入库任一步出错都会把状态更新为 `failed` 并写入错误信息

---

## 6. 核心链路二：RAG 问答 Pipeline（重点）

讲解路径：`api/rag.py` → `services/rag_service.py` → `core/graph.py` → 四个节点文件

### 6.1 LangGraph 工作流图

```
               ┌─────────────┐
               │  classify   │  ← Adaptive RAG
               └──────┬──────┘
                      │
                      ▼
               ┌─────────────┐
               │  retrieve   │  ← Dense + Sparse + RRF
               └──────┬──────┘
                      │
                      ▼
               ┌─────────────┐
               │  generate   │  ← qwen-plus
               └──────┬──────┘
                      │
                      ▼
               ┌─────────────┐
               │  evaluate   │  ← Corrective RAG
               └──────┬──────┘
          ┌──────────┴──────────┐
        accept              retry  give_up
          │                    │      │
          ▼                    ▼      ▼
         END           ┌─────────────┐ END
                       │ re_retrieve │
                       └──────┬──────┘
                              │
                              └──→ generate (循环)
```

### 6.2 全局状态（RAGState TypedDict）

```python
class RAGState(TypedDict):
    query: str                    # 用户原始问题
    subject: str | None           # 学科过滤
    grade: str | None             # 年级过滤
    complexity: str               # simple / medium / complex
    retrieved_docs: list          # 检索结果
    answer: str                   # 回答
    evaluation_reason: str        # 评估结论
    retry_count: int              # 重试次数
    max_retries: int              # 最大重试（默认 2）
```

### 6.3 单次完整调用流程

1. **classify_node**：规则匹配"是什么/公式"等关键词 → `simple`；"比较/分析/为什么"等 → `complex`；否则 `medium`
2. **retrieve_node**：按复杂度映射 `top_k = {simple:3, medium:5, complex:8}`，调用 `vector_store.hybrid_search`
3. **generate_node**：组装 `[1] ... [2] ...` 格式的上下文，用 System Prompt 约束"只能基于参考资料回答"，异步 httpx 调用 qwen-plus
4. **evaluate_node**：三维度评估——有无检索结果、回答是否过短、最高相关性得分是否 ≥ 0.4
5. **条件边 should_continue**：
   - `accept` → END（返回结果 + references）
   - `retry` 且未超限 → `re_retrieve`（top_k 升到 8）→ 重新 generate → 再 evaluate
   - `give_up` → END（返回"未找到相关信息"兜底）

---

## 7. 技术亮点深入

### 7.1 混合检索 + RRF 融合

**为什么要混合检索？**

| 检索方式 | 优势 | 劣势 |
|----------|------|------|
| 稠密向量（Milvus ANN） | 理解语义，同义词召回 | 对精确关键词（人名、公式符号）不敏感 |
| 稀疏 BM25 | 关键词精确匹配 | 不懂"方程 ≈ 等式" |

→ 两种都查，再用 RRF 融合，**取长补短**。

**RRF（Reciprocal Rank Fusion）公式**：

\[
\text{score}(d) = \sum_{i} \frac{1}{k + \text{rank}_i(d)}
\]

- `k` 是平滑参数，默认 60（业界经验值）
- `rank_i(d)` 是文档 d 在第 i 路检索中的排名
- **只依赖排名、不依赖原始分数**，避免了不同检索方式分数尺度不一致的问题

**代码位置**：`core/vectorestore.py` 中的 `_rrf_fusion` 方法。

> 亮点讲解："为什么不直接加权求和（0.7 × dense + 0.3 × sparse）？因为稠密是余弦相似度（0~1），BM25 是 TF-IDF 分数（可能几十几百），**量纲不一致直接加就废了**。RRF 巧妙在只看排名，跟绝对分数无关。"

### 7.2 Adaptive RAG（自适应检索）

**动机**：不是所有问题都需要检索 8 条文档。简单定义性问题（"什么是勾股定理"）1~3 条就够，复杂分析题（"比较直角三角形和等腰三角形的关系"）得多拿一些。

**实现**：基于规则的分类器（`core/nodes/query_classifier.py`）

```python
_SIMPLE_KEYWORDS = ["是什么", "什么是", "定义", "公式", "定理", ...]
_COMPLEX_KEYWORDS = ["比较", "对比", "分析", "为什么", "推导", "证明", ...]

# 含复杂词 & 长度 > 15 → complex
# 含简单词 or 长度 < 10  → simple
# 否则 medium
```

**效果**：
- 简单问题响应时间降 30%+（少拿文档少生成 token）
- 复杂问题得到更多上下文，回答更完整

> 面试追问："为什么用规则不用 LLM 分类？"  
> 答："成本权衡。用 LLM 分类要多调一次 API，延迟翻倍且花钱。**规则分类已经能覆盖 80% 的场景**。如果后期有需求，可以平滑切换成 LLM few-shot 分类，接口抽象已经留好了。"

### 7.3 Corrective RAG（纠正式检索）

**灵感**：来自 [CRAG 论文](https://arxiv.org/abs/2401.15884)。传统 RAG 是"一锤子买卖"——检索差就答得差。Corrective RAG 加入反馈闭环。

**本项目实现**（`core/nodes/evaluator.py`）：

```python
def evaluate_quality(query, answer, retrieved_docs, retry_count, max_retries=2):
    # 1. 没检索到 → give_up
    if not retrieved_docs: return "give_up", "未检索到相关文档"

    # 2. 回答太短/为空 → retry（未超限）或 give_up
    if not answer or len(answer.strip()) < 5:
        return "retry" if retry_count < max_retries else "give_up", ...

    # 3. 最高相关性得分 < 0.4 → retry
    max_score = max(doc.get("score", 0) for doc in retrieved_docs)
    if max_score < 0.4 and retry_count < max_retries:
        return "retry", f"检索相关性不足 (max_score={max_score:.3f})"

    return "accept", "回答质量合格"
```

**重试策略**（`core/graph.py` 的 `re_retrieve_node`）：**把复杂度强制提到 complex，把 top_k 升到 8**，扩大检索范围再生成。

最多重试 2 次，避免死循环浪费 token。

> 亮点讲解："这不是 LLM self-critic（让 LLM 自己骂自己那种），而是**基于确定性信号**（检索得分、回答长度）做决策，延迟可控、成本可控。"

### 7.4 LangGraph 工作流编排

**为什么用 LangGraph 而不是手写 if/else 或 LangChain Chain？**

| 方案 | 缺点 |
|------|------|
| 手写 `if/else` | 流程复杂后难维护；日志、重试、中间状态到处散落 |
| LangChain LCEL 链 | 只能线性串联，**没法表达条件分支和循环** |
| LangGraph | **有向图 + 条件边 + 全局 State + 内置 checkpointer**，天然适合 RAG 这种"有循环纠错"的流程 |

**本项目图结构**：
- 节点 5 个：classify / retrieve / generate / evaluate / re_retrieve
- 边 4 条普通边 + 1 条条件边（`should_continue`）
- 循环：`re_retrieve → generate → evaluate` 可回到 evaluate 再判断

**State 的闭包注入技巧**（代码实战点）：

```python
# vector_store 不适合放在 State 里（不可序列化、冗余）
# 所以用闭包包一层，把 store 注入到节点函数
async def retrieve_with_store(state: RAGState) -> dict:
    state["_vector_store"] = vector_store
    return await retrieve_node(state)
```

### 7.5 SSE 流式输出

为了提升 UI 交互体验，问答接口提供了两种模式：
- `/api/v1/rag/ask`：普通模式，等完整回答返回
- `/api/v1/rag/ask-stream`：**SSE 流式**，边生成边返回 token

实现要点（`services/rag_service.py` 的 `ask_stream`）：
- 用 `httpx.AsyncClient.stream(...)` 调用 LLM 的 `stream=true` 接口
- 解析 SSE 格式 `data: {"choices":[{"delta":{"content":"..."}}]}`，逐 token yield
- 封装了三种事件：`status`（状态进度）、`token`（内容片段）、`done`（完成含引用）

> 面试官可能问："为什么用 SSE 而不是 WebSocket？"  
> 答："RAG 是**单向的服务端推送**，不需要客户端回传，SSE 足够。SSE 基于 HTTP，**穿透代理、反向代理、负载均衡都更友好**，浏览器原生 EventSource 支持；WebSocket 是双向通信的杀鸡用牛刀。"

---

## 8. 难点与取舍

### 难点 1：Milvus Lite 在 asyncio 事件循环内启动失败

**现象**：`MilvusClient` 在 FastAPI lifespan 里初始化时报 subprocess 相关错误。

**原因**：Milvus Lite 会 fork 一个本地服务子进程，与 asyncio 事件循环的 signal handler 冲突。

**解决**（见 `main.py`）：
```python
# 关键：在 uvicorn.run 之前，同步初始化 Milvus 和 LangGraph
if __name__ == "__main__":
    _vector_store = init_vector_store_sync()     # 同步初始化
    _rag_graph = init_rag_graph_sync(_vector_store)
    uvicorn.run(app, ...)                         # 之后才起异步循环
```
在 `lifespan` 里把已初始化的全局实例挂到 `app.state`，异步环境拿现成的就行。

> 这个坑值得在面试里讲，**体现排查问题的能力**。

### 难点 2：中文 BM25 分词

**问题**：中文没有空格，`re.findall(r"[\w]+", text)` 只能得到长段文字，BM25 命中率低。

**解决**：**字符二元组（bigram）切分**——对长词再按相邻两字切分。例如"勾股定理" → ["勾股", "股定", "定理"]。无需引入 jieba 外部词典，成本低、对 K12 领域够用。

**更好的方案**（未来优化）：
- 用 `jieba` 或 `pkuseg` 专业分词
- 换成 `Milvus 2.5+ 原生稀疏向量`（SPLADE / BM25），让 Milvus 一站式搞定

### 难点 3：embed 模型首次下载卡住（国内网络）

**解决**：环境变量 `HF_ENDPOINT=https://hf-mirror.com` 走镜像；`config.py` 里自动设置。

### 取舍 1：是否用 LLM 做分类器

见 7.2 节——**性价比更高的规则分类**，保留了切换到 LLM 的接口。

### 取舍 2：是否用 LLM Judge 做评估

**没用**。LLM-as-a-judge 虽然效果好，但每次问答都要多调一次 LLM，成本翻倍，延迟也上升。选择了**基于检索得分的确定性评估**，90% 场景够用。

### 取舍 3：向量库为什么不用 Milvus 原生 Hybrid Search

Milvus 2.5+ 支持稀疏向量和 hybrid search，但 Lite 版支持有限；而且我想自己实现 RRF 作为**技术深度展示**，也便于后期换任何向量库都不绑死。

---

## 9. 性能与指标

> 以下为本机 M1 / CPU 模式下的粗略观测值，面试时照实说：

| 指标 | 数值 | 说明 |
|------|------|------|
| 文档入库（100 页 PDF） | ~30 s | 瓶颈在 embedding |
| 单次向量化（512 维） | ~30 ms/条 | CPU 模式 |
| Milvus ANN 搜索（万级） | < 10 ms | IVF_FLAT，nlist=128 |
| BM25 搜索（万级） | ~20 ms | 纯 Python |
| LLM 生成（qwen-plus） | 1~3 s | 非流式；流式首 token ~300ms |
| 端到端 RAG（不重试） | 2~4 s | 主要耗时在 LLM |
| 端到端 RAG（重试 1 次） | 4~7 s | 检索差时的兜底路径 |

### 性能优化已做
- `embedding` 单例加载（避免多次 init）
- `embed_texts` 批量调用（共享模型权重）
- BM25 仅在写操作后重建（读操作复用）
- LLM 异步 httpx 调用（不阻塞事件循环）

### 性能优化可做（未来）
- Embedding 切 ONNX Runtime / GPU
- 向量库从 Lite 迁 Standalone（IVF_FLAT → HNSW）
- BM25 换成 Milvus 稀疏向量索引
- 加 Redis 缓存常见 Query 的检索结果

---

## 10. 个人贡献与扩展方向

### 10.1 我做了什么（面试时按真实情况讲）

- 独立设计并实现整个 RAG 系统架构
- 核心算法实现：RRF 融合、Adaptive 分类、Corrective 评估
- LangGraph 工作流设计（5 节点 + 条件边 + 循环）
- FastAPI 异步架构 + 生命周期管理（踩过 Milvus + asyncio 冲突的坑）
- SSE 流式接口
- 业务功能：文档管理、学情分析、薄弱知识点分析

### 10.2 可扩展方向（展示技术视野）

1. **知识图谱增强**：为教材构建学科-章节-知识点的图谱，结合 GraphRAG
2. **Query 改写**：用户问"勾股定理"→ 改写成多个子问题（"公式"/"证明"/"应用"）分别检索再合并（Multi-Query RAG）
3. **Re-ranking**：召回后用 `bge-reranker` 做精排，提升 Top-K 质量
4. **长对话支持**：加 session 管理 + 多轮记忆，配合 LangGraph 的 checkpointer
5. **多模态**：教材里的公式、图表用 `nougat` / `Pix2Struct` 解析
6. **在线评估**：引入 RAGAS、TruLens 做无标注的自动化质量评测
7. **RAG + Agent**：让 LLM 自己决定是检索还是调用工具（计算器、画图）

---

## 11. 高频面试追问（准备好答案）

### Q1：RAG 和 Fine-tuning 的区别？什么场景选哪个？

**答**：
- **RAG**：外挂知识库。优势是知识热更新、成本低、可溯源；适合**知识频繁变化**、数据量大、要可解释来源的场景（法律、客服、教育）
- **Fine-tuning**：改模型权重。优势是**学习风格、能力**（口吻、推理模式）；适合**让模型掌握新技能或固定领域语气**
- **最佳实践**：先做 RAG 解决 80% 问题，少量用 Fine-tuning 让模型更懂你的领域 Prompt 格式

### Q2：Embedding 选型怎么考虑？

**答**：看 4 个维度——
1. **语言**：中文选 BGE / M3E / Qwen Embedding；英文 OpenAI / BGE-en
2. **维度**：维度越高检索越准但越慢。512 维是 K12 场景甜点
3. **上下文长度**：BGE-small 512 tokens，复杂文档要选 long context 模型（如 bge-m3 支持 8192）
4. **部署方式**：开源 vs API。我选开源是因为 **数据不出本机** 的合规需求

### Q3：Chunk size 怎么调？

**答**：trade-off——
- chunk **过小**：上下文不完整，回答碎片化
- chunk **过大**：相关性被稀释，检索得分低
- 经验值 **300~800 tokens**，中文约 **400~800 字**；本项目取 512 + 64 overlap
- 更优做法：**按语义边界切**（MarkdownHeader / SemanticChunker），比死板的固定大小好

### Q4：如果知识库有 1 亿条，你的系统还能扛吗？

**答**：这个架构的**代码完全不用改**，只需要替换两个地方——
1. Milvus Lite → Milvus Standalone/Cluster，索引从 IVF_FLAT 换成 HNSW
2. BM25 用 Python 进程内实现，这时必须换 **Elasticsearch** 或 **Milvus 2.5 原生稀疏索引**
3. Embedding 服务独立部署（用 Triton / vLLM），加 GPU
4. 前面加 **Redis 缓存**热查询
5. API 用 **Gunicorn + Uvicorn workers** 多进程

### Q5：幻觉怎么控制？

**答**：三层防线——
1. **Prompt 约束**：System Prompt 明确"只能基于参考资料回答，否则说未找到"
2. **Corrective RAG**：检索得分太低直接触发重试或兜底
3. **回答引用标注**：回答末尾要求标 `[1][2]`，用户能验证

进一步可做：**事实性检查**（用 NLI 模型判断回答是否被检索内容 entail）。

### Q6：向量检索的 `top_k` 怎么定？

**答**：
- 太小：召回不足，漏信息
- 太大：LLM context 被稀释、成本飙升（LLM 按 token 计费）
- 经验：**简单 3~5，复杂 8~10**
- 本项目做了 Adaptive——根据查询复杂度动态调

### Q7：RRF 的 `k=60` 是拍脑袋吗？

**答**：来自 [原论文](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)经验值。`k` 太小 → 前几名权重过大，融合退化；`k` 太大 → 所有文档得分差不多，融合失效。**60 是多次实验验证的稳健值**，几乎所有工业级 RRF 实现都用这个。

### Q8：为什么检索阶段不用重排（reranker）？

**答**：取舍。
- Reranker（cross-encoder）效果好但**很慢**，每条文档要跟 query 过一次模型，top_k=8 就是 8 次推理
- 本项目现阶段用 RRF 已经够用
- 如果要上，建议放在 Corrective RAG 的重试分支里（精度换成本）

### Q9：怎么保证系统的可观测性？

**答**：
- **日志**：每个节点进出都打日志（query、耗时、top_k、max_score）
- **问答记录**：SQLite 存 `QARecord`，含检索块、延迟、用户反馈（好评/差评）
- **健康检查**：`/health` 接口暴露 Milvus 行数、LLM 配置
- **生产化建议**：接 Prometheus + Grafana 监控 QPS/P99 延迟；接 OpenTelemetry 做 trace

### Q10：LangGraph 和 LangChain 的区别？

**答**：
- LangChain = **链式**（LCEL），适合线性"A → B → C"流程
- LangGraph = **图式**，节点 + 边 + 全局 State，支持**条件分支、循环、人工介入**（Human-in-the-Loop）
- 本项目 Corrective RAG 的"评估不通过就回去重新检索"是**循环**，LangChain LCEL 做不了，LangGraph 天然支持
- LangGraph 底层也是 LangChain 生态，**不冲突**，经常混用

---

## 附：现场演示脚本（按这个顺序走）

1. **起服务**：`python main.py` → 展示启动日志（Milvus 同步初始化、LangGraph 构建完成）
2. **打开首页**：`http://localhost:8000/` → 展示 UI
3. **上传文档**：在 UI 或 Swagger 上传一份教材 MD 文件，展示 `status: processing → completed` 和切片数
4. **提问 Demo 1（简单）**："什么是勾股定理？" → 展示 `complexity: simple, top_k=3`，返回带引用
5. **提问 Demo 2（复杂）**："比较勾股定理和余弦定理的关系" → 展示 `complexity: complex, top_k=8`
6. **提问 Demo 3（流式）**：用 `curl` 调 `/ask-stream` → 展示 token 逐字吐出
7. **展示学情分析**：`/analytics/weak-points/{user_id}` → 展示薄弱知识点列表
8. **展示 Swagger**：`http://localhost:8000/docs` → 证明所有接口有自动文档

---

## 最后：讲解心态与技巧

1. **先讲业务，再讲技术**：面试官不一定懂 RAG，先铺背景
2. **别背书，画图讲**：请求流程、LangGraph 图 一定要现场画
3. **主动暴露难点**：Milvus + asyncio 那个坑、BM25 中文分词，**承认自己踩过坑更可信**
4. **准备两个 Demo 问题**：一个简单、一个复杂，展示 Adaptive 效果
5. **留一个"高级话题"钩子**：比如提一嘴"我还想加 reranker 和 GraphRAG"，面试官感兴趣就能深入聊
