# K12 教育领域 RAG 知识库问答系统

面向 K12 教学内容的检索增强生成（RAG）服务：基于 **LangGraph** 编排问答流程，**Milvus Lite** 承载向量检索，**sentence-transformers / BGE** 提供 Embedding，LLM 通过 **OpenAI 兼容 HTTP API** 接入（默认 **阿里百炼** `qwen-plus`）。系统提供 **REST API**、**静态 Web 控制台**、**学情分析**，并集成 **RAGAS** 用于离线质量评估。**应用版本**：当前发布标识为 `1.0.0`（见 `main.py` 元数据）。

---

## 目录

- [概述](#概述)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [部署与本地运行](#部署与本地运行)
- [Web 控制台与 API](#web-控制台与-api)
- [RAGAS 评估](#ragas-评估)
- [数据持久化与本地资源](#数据持久化与本地资源)
- [测试集 manual_v1.jsonl](#manual_v1jsonl)
- [配置参考](#配置参考)
- [仓库结构](#仓库结构)
- [检索与流水线说明](#检索与流水线说明)
  - [问答主链路](#问答主链路)
  - [数据清洗流水线](#数据清洗流水线)
  - [SQL 数据导入](#sql-数据导入)
  - [RRF 融合公式](#rrf-融合公式)
- [安全与合规](#安全与合规)
- [已知限制与排查](#已知限制与排查)

---

## 概述

### 背景与目标

教材与教辅以自然语言为主，字面匹配检索难以覆盖同义表述与释义性描述。系统将文档切分为片段、完成向量化并入库存储；用户提问时通过 **稠密 + 稀疏混合检索** 召回相关片段，再由 LLM 在约束下生成答复，从而在私有语料范围内提升可解释性与可控性。

### 主要能力

| 特性 | 说明 |
|------|------|
| **混合检索** | 稠密向量（语义）+ BM25（关键词），RRF 融合 |
| **多策略检索** | 根据意图+复杂度自动选择检索策略：直接检索 / 多查询变体 / 问题分解 + HyDE / Step-Back 补充 |
| **意图与复杂度** | 教育 / 闲聊分流；教学问句再分 simple / medium / complex 调节检索深度；LLM 分类结果自动收集为后续分类器训练数据 |
| **Corrective RAG** | 生成后质量评估，不通过则扩检索并重试（可配置上限） |
| **数据清洗** | 文件导入时自动执行规范化→去噪→结构修复→校验四阶段流水线；支持去重、质量评分 |
| **LangGraph 编排** | 分类 → 检索（含策略选择）→ 生成 → 评估 → （重试）流程图式定义 |
| **Milvus Lite** | 本地单文件向量库（`pymilvus` + lite），免去单独部署向量服务 |
| **RAGAS 评估** | Faithfulness、Answer Relevancy 等指标；评估结果写入业务库并在控制台展示历史 |
| **Web 控制台** | `static/index.html`：问答、文档导入、学情、知识点维护、评估任务 |
| **SQL 数据导入** | 支持从 MySQL 等数据库直接导入表数据，经清洗→切片→向量化入库 |

### 应用场景与边界

**适用场景**

- 校本/机构内部 **教材与教辅** 的语义检索与自然语言问答，支持按 **学科、年级** 过滤召回范围。  
- **教学运营**：文档批量入库、切片策略可配，便于迭代知识库。  
- **质量度量**：通过 RAGAS 对回答进行多指标打分，并保留历史记录便于对比不同模型或语料版本。  
- **学情链路**：在持续产生 `user_id` 与问答历史的前提下，使用分析类接口观察薄弱点与推荐路径（具体统计逻辑以 `services/analytics_service.py` 为准）。

**使用边界（非目标）**

- 不提供多租户隔离、细粒度权限模型与审计全流程，生产环境需在上层网关或业务系统中补齐 **认证授权**与 **访问控制**。  
- Milvus **Lite** 面向单机与中小规模数据集；超高并发或多副本部署应考虑 **Milvus 集群形态**并做好客户端与运维配套。  
- 闲聊分支仅用于非教学类短对话分流，不适合作为通用开放域客服机器人基座。

---

## 系统架构

```mermaid
graph TB
    subgraph UI["用户层"]
        Web["静态 Web UI (/ )"]
        Swagger["Swagger UI (/docs)"]
    end

    subgraph API["API 层 FastAPI"]
        RAG_API["RAG 问答"]
        DOC_API["文档管理"]
        KNOW_API["知识点"]
        ANALYTICS_API["学情分析"]
        EVAL_API["RAGAS 评估"]
    end

    subgraph SVC["服务层"]
        RAG_SVC["RAGService"]
        DOC_SVC["DocumentService"]
        KNOW_SVC["KnowledgeService"]
        ANALYTICS_SVC["AnalyticsService"]
        INGEST["文档入库流水线"]
    end

    subgraph ENGINE["LangGraph RAG"]
        LANGGRAPH["工作流编排"]
        INTENT["意图 + 复杂度"]
        STRATEGY["策略选择"]
        HYBRID["混合检索 RRF"]
        LLM_GEN["LLM 生成"]
        CORR["Corrective 评估"]
        SUPP["补充策略<br/>HyDE / Step-Back"]
    end

    subgraph DATA["数据层"]
        MILVUS["Milvus Lite"]
        BM25["BM25 索引"]
        SQLITE["SQLite 业务库<br/>文档 / QA / 评估记录等"]
        EMBED["sentence-transformers<br/>BGE 等 Embedding"]
    end

    subgraph FILES["本地文件"]
        UP["uploaded_docs/"]
        CLEANER["Cleaner 清洗器"]
    end

    Web --> RAG_API
    Web --> DOC_API
    Web --> ANALYTICS_API
    Web --> KNOW_API
    Web --> EVAL_API
    Swagger --> API

    RAG_API --> RAG_SVC
    DOC_API --> DOC_SVC
    KNOW_API --> KNOW_SVC
    ANALYTICS_API --> ANALYTICS_SVC

    RAG_SVC --> LANGGRAPH
    DOC_SVC --> INGEST
    INGEST --> CLEANER["数据清洗"]
    LANGGRAPH --> INTENT --> STRATEGY --> HYBRID --> LLM_GEN --> CORR
    HYBRID --> MILVUS
    HYBRID --> BM25
    HYBRID --> EMBED
    CORR -.-> SUPP
    SUPP -.-> HYBRID
    INGEST --> UP
    CLEANER --> MILVUS
    CLEANER --> BM25

    RAG_SVC --> SQLITE
    DOC_SVC --> SQLITE
    KNOW_SVC --> SQLITE
    ANALYTICS_SVC --> SQLITE
    EVAL_API --> SQLITE

    EVAL_API -.->|pipeline 内复用 RAG 图| LANGGRAPH
```

### 文档入库（概要）

在 **系统架构** 图中，导入流水线由文档服务触发；数据流为：上传 PDF / Markdown / TXT / SQL → 解析 → **数据清洗（四阶段）** → 切片 → Embedding → Milvus 写入并同步稀疏索引（BM25）→ SQLite 更新文档状态。HTTP 入口：`POST /api/v1/documents/upload` 及 `POST /api/v1/documents/import/sql`。

---

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| 语言运行时 | Python 3.11+ | 建议在 3.11～3.13 |
| Web | FastAPI + Uvicorn | 异步 API，自带 OpenAPI |
| 编排 | LangGraph | RAG 状态机与工作流 |
| 向量库 | Milvus Lite（pymilvus） | 配置项见 `K12_MILVUS_URI` |
| Embedding | sentence-transformers | 默认 `BAAI/bge-small-zh-v1.5` |
| LLM | OpenAI SDK 兼容端点 | 默认百炼 Compatible Mode，`LLM_BASE_URL` + `LLM_MODEL` |
| 文档 | unstructured、pypdf | PDF / MD / TXT |
| 稀疏检索 | rank_bm25 | 与稠密向量互补 |
| 多策略检索 | LLM 驱动查询改写/分解 | Multi-Query、Decomposition、HyDE、Step-Back |
| 数据清洗 | 自定义四阶段流水线 | 规范化→去噪→修复→校验；多数据源适配 |
| 业务库 | SQLite + SQLAlchemy async | `k12_business.db` |
| 离线评估 | RAGAS + datasets | Instructor 结构化输出，`RAGAS_LLM_MAX_TOKENS` 可调 |
| 数据库导入 | SQLAlchemy + PyMySQL | 支持 MySQL 等关系型数据库流式导入 |

### 核心第三方组件

- **pymilvus**：向量检索客户端；Milvus Lite 模式下 URI 指向本地持久化文件。  
- **LangChain**：部分抽象与惯例；向量存储本项目以 **`K12VectorStore`**（`core/vectorestore.py`）为主。  
- **sentence-transformers**：加载 Embedding 编码器（如 BGE 系列）。  
- **rank_bm25**：稀疏检索打分。  
- **RAGAS / Hugging Face datasets**：离线评估流水线与数据集表示。  
- **SQLAlchemy / PyMySQL**：关系数据库连接与流式读取（SQL 数据导入）。

### 声明式依赖版本

以下与 [`requirements.txt`](requirements.txt) 一致，仅供快速浏览；安装与冲突解决请以该文件为准。

| 类别 | 包名（节选） | 说明 |
|------|----------------|------|
| 向量与检索 | `pymilvus[milvus_lite]>=2.4.2`、`milvus-lite>=2.4.0,<3.0.0` | Milvus 客户端与兼容现有单文件数据库的 Lite 运行时 |
| 应用框架 | `fastapi>=0.110.0`、`uvicorn[standard]>=0.27.0` | HTTP 服务 |
| LangGraph / LangChain | `langchain>=1.2.0`、`langchain-core`、`langchain-community`、`langchain-milvus`、`langchain-openai` | 编排与生态组件 |
| 向量化 | `sentence-transformers>=3.0.0` | Embedding 推理 |
| 文档 | `unstructured[pdf,md]`、`pypdf` | 解析与切分流水线 |
| 评估 | `ragas>=0.2.0`、`datasets>=3.0.0` | RAGAS 与数据集 |
| 持久化 | `sqlalchemy>=2.0.0`、`aiosqlite`、`greenlet`、`pymysql` | 异步 SQLite + MySQL 连接 |
| 其它 | `rank_bm25`、`httpx`、`python-dotenv`、`python-multipart` | BM25、HTTP 客户端、环境与上传 |

---

## 部署与本地运行

### 环境与依赖

- **Python**：3.11 及以上（推荐 3.11～3.13）。
- **Embedding**：需能访问模型权重（公网、[Hugging Face 镜像](https://hf-mirror.com)或本地缓存目录）。
- **LLM**：使用百炼或其它 OpenAI 兼容服务时配置 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`。

### 安装与启动

```bash
git clone <repository-url>
cd edu-rag

python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env         # 按环境修改变量
```

**环境变量摘要**（完整说明见 [.env.example](.env.example)）：

| 变量 | 说明 |
|------|------|
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | 大模型网关；默认值指向阿里百炼兼容接口 |
| `K12_MILVUS_URI` | Milvus Lite 数据库文件路径，勿与系统环境变量 **`MILVUS_URI`** 同名冲突 |
| `EMBEDDING_MODEL` / `EMBEDDING_DEVICE` | 向量模型名与运行设备 |
| `HF_ENDPOINT` | 国内下载模型常用镜像，如 `https://hf-mirror.com` |
| `RAGAS_LLM_MAX_TOKENS` | RAGAS 结构化输出单次生成上限（默认 8192，避免 Faithfulness 等 JSON 截断） |
| `DENSE_MIN_SIMILARITY` | 可选：稠密检索余弦下限过滤 |

可选：在当前 shell 设置 Hugging Face 端点后启动应用：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python main.py
```

- **控制台**：`http://localhost:8000/`
- **OpenAPI**：`http://localhost:8000/docs`
- **健康检查**：`GET /health`

**连通性自检示例**：

```bash
curl -s http://localhost:8000/health

curl -X POST http://localhost:8000/api/v1/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "一元一次方程怎么解？", "subject": "数学", "grade": "七年级", "user_id": "demo"}'
```

### 开发模式（热重载）

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

> **说明**：Milvus Lite 须在 **异步事件循环启动前** 完成初始化；`python main.py` 已按该顺序封装。若自行用 Uvicorn 多 worker 拉起进程，请先确认是否与 Milvus Lite 进程的 **单实例约束**相符。

---

## Web 控制台与 API

前端为单页应用，侧边栏功能模块如下：

| 模块 | 功能 |
|------|------|
| RAG 问答 | 多学科/年级筛选、引用来源面板 |
| 文档管理 | 上传（PDF/MD/TXT）、SQL 导入、列表、删除、切片策略 |
| 学情分析 | 与用户 ID 关联的统计分析（依赖历史数据） |
| 知识点 | 维护知识点树 |
| 效果评估 | JSON/JSONL 测试集上传 → RAG 生成答案 → RAGAS 打分；结果持久化并可查看历史记录 |

### 主要 HTTP 路由

以下为基础清单，**字段与错误码以运行实例中的 `/docs` 为准**。

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 返回 Web UI（`static/index.html`） |
| GET | `/health` | 健康检查 |
| POST | `/api/v1/rag/ask` | 问答 |
| POST | `/api/v1/rag/feedback` | 对某条 QA 点赞/差评 |
| POST | `/api/v1/documents/upload` | 上传文档并入库（PDF/MD/TXT） |
| POST | `/api/v1/documents/import/sql` | 从 MySQL 等数据库导入数据 |
| GET | `/api/v1/documents/list` | 文档列表 |
| DELETE | `/api/v1/documents/{id}` | 删除文档 |
| GET | `/api/v1/knowledge-points/tree` | 知识点树 |
| POST | `/api/v1/knowledge-points/` | 创建知识点 |
| GET | `/api/v1/analytics/weak-points/{user_id}` | 薄弱知识点等 |
| GET | `/api/v1/analytics/history/{user_id}` | 问答历史 |
| GET | `/api/v1/analytics/recommend/{user_id}` | 复习推荐 |
| POST | `/api/v1/evaluation/from-content` | 表单：`content`/`file` + `metrics`，实时 RAG + RAGAS |
| POST | `/api/v1/evaluation/from-history` | 按 QA 历史批量评估 |
| GET | `/api/v1/evaluation/history` | 评估记录列表 |
| GET | `/api/v1/evaluation/history/{id}` | 单条评估明细（含各题得分） |

---

## RAGAS 评估

- **指标**：与支持 RAGAS 当前版本的传统指标标识一致（如 `faithfulness`、`answer_relevancy`、`context_precision`；启用 `context_recall` 时建议在数据集中提供 `ground_truth`）。  
- **持久化**：评估完成后写入 SQLite 表 `evaluation_records`；控制台支持列表与明细查询；REST 响应在入库成功时可包含 `record_id`。  
- **命令行**：见 [`evaluation/cli.py`](evaluation/cli.py)：`evaluate`（`--from-db`、`--from-file`、`--live` 等）、`generate`、`validate`、`export` 等子命令。  
- **运行参数**：Faithfulness / Context 等依赖 Instructor 结构化输出，若单次生成长度过长导致截断，应提高环境变量 **`RAGAS_LLM_MAX_TOKENS`**；若检索上下文为空，上下文相关指标可能无效或分值异常，请核对知识库数据与检索过滤条件。

### 检索评估与阈值校准

检索评估与 RAGAS 并列运行，只衡量召回、排序、门控误判和重试收益。标注集采用 JSONL，每条包含 `question`、`answerable` 和 `relevant_chunk_ids`；不可回答问题将 `relevant_chunk_ids` 设为空数组。

```bash
python evaluation/cli.py retrieval-evaluate --from-file data/test_sets/retrieval_manual_v1.jsonl
python evaluation/cli.py retrieval-calibrate --from-file data/test_sets/retrieval_manual_v1.jsonl --max-false-accept-rate 0.05
```

输出包含 `Recall@5/10/20`、`Precision@5`、`MRR@10`、`nDCG@10`、错误接受率、错误拒绝率、拒答准确率、重试恢复率、延迟分位数和分组切片。示例文件中的 chunk ID 仅用于展示格式，正式运行前应基于当前向量库完成标注。

---

## 数据持久化与本地资源

| 路径 / 标识 | 内容 |
|-------------|------|
| `k12_business.db`（默认） | SQLite：文档元数据、问答记录、知识点、`evaluation_records` 等业务表 |
| `K12_MILVUS_URI` 指向的文件 | Milvus Lite 向量与索引数据 |
| `uploaded_docs/` | 用户上传文档的落盘副本（文件名通常带 UUID 前缀） |
| `data/intent_training_data.jsonl` | LLM 意图分类结果自动收集，用于后续分类器训练 |
| HuggingFace 缓存目录 | Embedding 权重缓存在本机用户目录下（取决于 `sentence-transformers` / `HF_HOME` 等环境） |
| 日志 | `k12_rag` 等 logger，默认级别见 `LOG_LEVEL` |

清空或迁移环境时：**先停服务**，再按需备份上述数据库文件与向量库文件；替换 Embedding 维度或集合结构后通常需 **重新入库**。

**随附示例资源**（便于联调与演示；不保证与生产语料一致）：

| 路径 | 说明 |
|------|------|
| [`sample_docs/`](sample_docs/) | 语文/数学等小样本教材片段，可配合「文档上传」走通入库与问答 |
| [`evaluation/sample_test.json`](evaluation/sample_test.json) | 含 `question` / `contexts` / `ground_truth` 的 JSON 示例，适合理解 RAGAS 输入形态 |
| [`data/test_sets/manual_v1.jsonl`](data/test_sets/manual_v1.jsonl) | 手工编写的多科问答测试集，见下节说明 |
| [`data/test_sets/retrieval_manual_v1.example.jsonl`](data/test_sets/retrieval_manual_v1.example.jsonl) | 检索标注格式示例，含不可回答问题 |
| [`spec/`](spec/) | 功能设计文档：数据清洗、引用优化 |
| [`test/`](test/) | 测试代码：多策略检索测试、数据清洗测试 |

### manual_v1.jsonl

文件路径：**[`data/test_sets/manual_v1.jsonl`](data/test_sets/manual_v1.jsonl)**。

**定位**：面向 RAG 联调与 RAGAS 回归的 **小规模人工基准集**（当前 **12** 条），覆盖数学、生物、物理，年级以初中、高中为主；题型包含定义、应用、对比与开放问答等，并标注了预期 **查询复杂度**（与系统内 `simple` / `medium` / `complex` 概念对齐，便于观察检索深度差异）。

**格式**：JSONL，**每行一个 JSON 对象**，字段如下。

| 字段 | 类型 | 说明 |
|------|------|------|
| `question` | string | 用户问题（必填；实时评估时由 RAG 生成 `answer` 与 `contexts`） |
| `ground_truth` | string | 参考答案（用于 `context_recall` 等需标准答案的指标） |
| `complexity` | string | 标注难度：`simple` / `medium` / `complex`（元数据，当前流水线以分类器结果为准，可用于筛题或扩展脚本） |
| `question_type` | string | 题型标签，如「定义题」「应用题」「对比题」「开放题」 |
| `subject` | string | 学科，如「数学」「生物」「物理」 |
| `grade` | string | 学段标签，如「初中」「高中」 |

**使用方式**：

- **Web 控制台**：在「效果评估」中粘贴该文件全文或上传文件；可选学科/年级与 RAGAS 指标。系统会按行调用 RAG，再对生成结果打分；含 `ground_truth` 时可勾选 **`context_recall`**。  
- **命令行**：例如 `python evaluation/cli.py evaluate --from-file data/test_sets/manual_v1.jsonl --live --metrics faithfulness,answer_relevancy`（`--live` 需已初始化向量库，参见 `evaluation/cli.py`）。

**注意**：评分质量强依赖知识库中是否已有与题目相关的入库文档；若检索为空，上下文类指标会失真。扩展该文件时请保持 **一行一 JSON**、键名与上表一致。

---

## 配置参考

除 `.env` / `.env.example` 所载变量外，下列参数在 **`config.py`** 中维护，可按需调整并重新部署：

| 参数 | 默认 / 含义 | 作用简述 |
|------|-------------|-----------|
| `TOP_K` | 5 | 单次检索返回的片段数量基数（复杂度会在节点内进一步调节） |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 512 / 64 | 文档切片字符规模与重叠 |
| `RRF_K` | 60 | Reciprocal Rank Fusion 平滑常数 \(k\) |
| `DENSE_WEIGHT` / `SPARSE_WEIGHT` | 0.7 / 0.3 | 定义于配置中；**混合检索主线**使用 RRF 融合（见 `core/vectorestore.py`），这两项现主要用于 **评估结果中的配置快照**（`evaluation/pipeline.py`），便于回溯实验环境 |
| `DENSE_MIN_SIMILARITY` | 0（可由环境变量设置） | 稠密检索余弦相似度下限，低于则丢弃 |
| `MAX_RETRIES` | 2 | Corrective RAG 最大重试轮次 |
| `LLM_TIMEOUT_SECONDS` / `ENABLE_LLM_FALLBACK` | 3 / true | 意图链路中的 LLM 分类超时与兜底开关 |
| `LLM_TIMEOUT_SECONDS` / `ENABLE_LLM_FALLBACK` | 3 / True | LLM 参与意图兜底时的超时与开关 |
| `MULTI_QUERY_VARIANTS` | 4 | 多查询策略生成的变体数量 |
| `DECOMPOSITION_MAX_SUB` | 4 | 复杂问题拆解的最多子问题数 |
| `RETRIEVAL_CANDIDATE_TOP_K` / `GENERATION_CONTEXT_TOP_K` | 20 / 5 | 重排前候选数与生成阶段上下文数 |
| `RERANKER_MODEL` / `RERANKER_DEVICE` | `BAAI/bge-reranker-base` / `cpu` | 本地 CrossEncoder 重排模型与设备 |
| `RERANKER_RELEVANCE_THRESHOLD` / `RETRIEVAL_ACCEPT_TOP1_THRESHOLD` | 0.50 / 0.60 | 在线门控相关候选与 top-1 接受阈值 |
| `RETRIEVAL_GATE_MODE` | `enforce` | `enforce` 会拒答，`observe` 仅记录不可用重排器 |
| `STRATEGY_TIMEOUT` | 10 | 策略 LLM 调用超时（秒） |

---

## 仓库结构

```
edu-rag/
├── main.py                     # FastAPI 入口（含 Milvus 同步初始化）
├── config.py                   # 全局配置
├── requirements.txt
├── .env.example
├── sample_docs/                # 入门用小型教材样例（TXT/MD）
├── data/
│   ├── intent_training_data.jsonl  # 意图分类微调数据（自动收集）
│   └── test_sets/
│       └── manual_v1.jsonl     # 手工问答测试集
├── spec/                       # 功能设计文档
│   ├── document_clean.md
│   └── source_citation_optimization.md
├── test/                       # 测试代码
│   ├── test_strategies.py      # 多策略检索测试
│   ├── test_cleaner.py         # 数据清洗测试
│   ├── multi_strategy_retrieval_test.md
│   └── data_cleaning_test.md
├── static/
│   └── index.html              # Web 控制台
├── core/
│   ├── embeddings.py
│   ├── vectorestore.py         # Milvus + BM25 + RRF
│   ├── reranker.py             # 本地 CrossEncoder 重排
│   ├── retrieval_quality.py    # 统一检索门控
│   ├── graph.py                # LangGraph 编排
│   ├── stream_queue.py         # 流式输出队列
│   ├── nodes/
│   │   ├── query_classifier.py # 复杂度 + 异步意图分流
│   │   ├── llm_classifier.py / keyword_matcher.py …
│   │   ├── chitchat.py
│   │   ├── retriever.py        # 策略驱动的混合检索
│   │   ├── generator.py
│   │   └── training_collector.py # 分类器训练数据自动收集
│   └── strategies/             # 多策略检索模块
│       ├── selector.py         # 初始检索策略选择器
│       ├── multi_query.py      # 多查询变体生成 + RRF 融合
│       ├── decomposition.py    # 复杂问题拆解 + 子结果合并
│       ├── hyde.py             # HyDE 假设答案生成
│       ├── step_back.py        # Step-Back 抽象回退
│       └── _llm.py             # 策略共享 LLM 调用工具
├── ingestion/
│   ├── loader.py               # PDF/MD/TXT 文档加载
│   ├── cleaner.py              # 数据清洗模块（四阶段 + 多数据源适配）
│   ├── chunker.py
│   └── pipeline.py             # 导入流水线（含文件/SQL 双入口）
├── evaluation/
│   ├── ragas_evaluator.py      # LLM / Embedding / 指标适配
│   ├── retrieval_evaluator.py  # 检索指标、门控回放与阈值校准
│   ├── pipeline.py             # run_evaluation / run_live_evaluation / 入库
│   ├── dataset_builder.py
│   ├── schemas.py
│   ├── testset_generator.py
│   ├── sample_test.json        # 离线评估示例数据
│   └── cli.py
├── api/
│   ├── rag.py / documents.py / knowledge.py / analytics.py / evaluation.py
├── services/
├── models/
└── utils/
```

---

## 检索与流水线说明

### 问答主链路

1. **意图与复杂度**：非教育类查询进入闲聊分支；教育类查询通过规则或模型划分为 `simple` / `medium` / `complex`，影响检索策略与广度（见 `core/nodes/query_classifier.py` 等）。LLM 分类的高置信度结果会自动写入 `data/intent_training_data.jsonl`，供后续训练本地分类器。
2. **策略选择**：根据意图和复杂度自动选择检索策略（见 `core/strategies/selector.py`）：
   - `simple` → **DIRECT**：直接混合检索
   - `medium` → **MULTI_QUERY**：LLM 生成多个查询变体，多路检索后 RRF 融合
   - `complex` → **DECOMPOSITION**：LLM 将复杂问题拆解为子问题，分别检索后合并去重
3. **混合检索**：每路查询在稠密向量（Milvus）与 BM25 上并行检索，使用 RRF 合并排序。
4. **本地重排**：候选片段经过 CrossEncoder，原始 logit 经 sigmoid 归一化为 `rerank_score`。RRF 仅负责候选排序，不参与质量判断。
5. **统一门控**：`retrieval_gate` 根据 `rerank_score` 给出 `accept` / `retry` / `abstain`。第一次重试使用原问题和最多三个改写变体；第二次根据失败原因使用 HyDE 或 Step-Back。
6. **生成或拒答**：只有 `accept` 才会生成答案；重试耗尽后进入固定拒答节点，避免低质量资料被包装成可信答案。

### 数据清洗流水线

文件导入时自动执行四阶段清洗（`ingestion/cleaner.py`），确保入库数据质量：

| 阶段 | 模块 | 说明 |
|------|------|------|
| 1. 规范化 | `Normalizer` | 编码统一、不可见字符移除、空格/换行规范化 |
| 2. 去噪 | `Denoiser` | 页码/页眉页脚移除、目录/版权声明过滤、短文本丢弃、高频噪声抑制 |
| 3. 结构修复 | `StructureRepairer` | 断句合并、残缺段落修复 |
| 4. 校验 | `Validator` | 长度校验、内容Hash去重、质量评分 |

清洗过程输出 `CleanStats`（含输入/输出数量、去重率、丢弃率等），并在处理结果中返回。

**数据源适配**：
- **文件源**：`FileSourceAdapter` 将 LangChain Document 转为清洗记录，清洗后回转为 Document
- **SQL 源**：`SQLSourceAdapter` 封装数据库连接，支持流式读取、字段映射、条件过滤

### SQL 数据导入

通过 `POST /api/v1/documents/import/sql` 从关系数据库导入数据：

```json
{
  "db_url": "mysql+pymysql://user:pass@host:3306/db",
  "table_name": "knowledge_items",
  "subject": "数学",
  "grade": "七年级",
  "field_map": {"title": "标题", "body": "正文"},
  "id_column": "id",
  "batch_size": 1000
}
```

后端使用 SQLAlchemy 流式读取 + 游标分页，每批数据经清洗流水线处理后切片入库。详细设计见 [`spec/document_clean.md`](spec/document_clean.md)。

### RRF 融合公式

对文档 \(d\) 在多路检索中的名次 \(rank_i(d)\)，采用：

\[
\mathrm{score}(d) = \sum_i \frac{1}{k + rank_i(d)}
\]

本项目默认 \(k=\) **`RRF_K`**（通常为 60，见 `config.py`）。

### LangGraph 节点拓扑（概要）

常规教育问答路径：`classify` → `retrieve` → `rerank` → `retrieval_gate`。门控通过后进入 `generate`；需要纠正时进入 `retry_planner` 后回到 `retrieve`；重试耗尽后进入 `abstain`。非教育意图经 `classify` → `chitchat` → `finalize`，不执行向量检索。

实现细节与条件边定义见 **`core/graph.py`**。行为变更以源代码及 **`GET /docs`** 为准。

---

## 安全与合规

- **密钥管理**：`LLM_API_KEY` 及其它敏感信息仅通过环境变量或私有配置注入，**不要将 `.env` 提交至版本库**（仓库已提供 `.env.example`）。
- **CORS**：默认 `allow_origins=["*"]`，面向公网部署时应在 `main.py` 中收紧为明确来源列表。
- **数据留存**：问答与评估记录落库于 SQLite，请按组织策略做 **保留周期、备份与脱敏**。
- **内容责任**：生成内容来自「检索上下文 + LLM」，教学场景下仍建议人工抽检与引用核对。

---

## 已知限制与排查

| 现象 | 可能原因 | 建议 |
|------|-----------|------|
| `/health` 中 `vector_store` 异常 | Milvus Lite 未就绪或锁文件占用 | 确认单进程初始化；检查 `.milvus*.db.lock` 等是否在异常退出后残留 |
| `too_many_pings` / gRPC 告警 | Milvus Lite 与客户端 HTTP/2 保活在高频请求下偶发 | 一般为瞬态；持续出现时降低并发或升级 `pymilvus` |
| 检索结果为空、上下文类指标偏低 | 知识库为空、筛选过严或向量未入库 | 检查文档状态、`subject`/`grade`、`DENSE_MIN_SIMILARITY` |
| Faithfulness 等结构化指标失败 | Instructor 输出被 `max_tokens` 截断 | 提高 **`RAGAS_LLM_MAX_TOKENS`** |
| 控制台与纯 API 表现不一致 | 流式输出经服务层队列推送 | 对照 `services/rag_service.py` 与 `core/stream_queue.py` |
| 多策略检索耗时过长 | 多查询/分解策略需多次 LLM 调用 | 调整 `MULTI_QUERY_VARIANTS`、`DECOMPOSITION_MAX_SUB` 降低变体数；或增大 `STRATEGY_TIMEOUT` |
| HyDE/Step-Back 未触发 | 门控已经接受结果，或尚处于第一次 query variants 重试 | 检查 `retrieval_attempts` 和 `RERANKER_*` / `RETRIEVAL_ACCEPT_TOP1_THRESHOLD` 配置 |
| 数据清洗丢弃了过多内容 | 去噪规则过于严格（页码/短文本/高频噪声） | 检查 `ingestion/cleaner.py` 中 `Denoiser` 的阈值和模式；观察 `CleanStats.dedup_rate` / `drop_rate` |
| SQL 导入连接超时 | 数据库不可达或防火墙限制 | 确认 `db_url` 格式正确、网络可达；减小 `batch_size` 降低单次压力 |
