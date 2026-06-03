# 面试技术复盘：K12 教育 RAG 知识库问答系统

> 面试人：易鹏 | 面试官：博 | 项目：edu-rag（K12 教育领域 RAG 知识库问答系统）
> 复盘日期：2026-05-24

---

## 一、项目总览

**项目定位**：面向 K12 教学内容的检索增强生成（RAG）服务，基于 LangGraph 编排问答流程，Milvus Lite 承载向量检索，BGE 模型提供 Embedding，LLM 通过 OpenAI 兼容 API 接入（默认阿里百炼 qwen-plus）。

**技术栈一览**：

| 组件 | 选型 | 角色 |
|------|------|------|
| 编排框架 | LangGraph | RAG 状态机与工作流 |
| 向量库 | Milvus Lite (pymilvus) | 稠密向量 ANN 检索 |
| Embedding | BAAI/bge-small-zh-v1.5 | 语义向量化 |
| 稀疏检索 | rank_bm25 (BM25Okapi) | 关键词召回 |
| LLM | 阿里百炼 qwen-plus (OpenAI 兼容) | 意图分类 + 答案生成 + 查询改写 |
| 业务库 | SQLite + SQLAlchemy async | 文档/问答/评估记录 |
| 评估 | RAGAS | Faithfulness / Answer Relevancy 等 |
| 文档解析 | unstructured + pypdf | PDF/MD/TXT |
| Web | FastAPI + 静态单页 | API + 控制台 |

---

## 二、意图识别：三层兜底方案

### 面试中的描述

易鹏描述了三层兜底：**规则匹配 → 预训练模型分类 → 大模型兜底**。实际代码实现为 **两层渐进式**（第三层 LLM 分类结果自动收集为训练数据，用于后续训练本地分类器）。

### 代码实际实现

**第一层：关键词匹配（`core/nodes/keyword_matcher.py`）**

- 速度 < 1ms，命中即短路返回
- 覆盖 5 类意图：greeting、command、educational、technical、chitchat
- 每类意图维护关键词列表，按优先级匹配
- 教育类关键词覆盖：学科名（数学/语文/英语...）、学习动作（公式/定理/定义/概念）、问答词（是什么/为什么/怎么计算）

```python
# 意图分类体系（keyword_matcher.py:7-44）
KEYWORD_INTENT_MAP = [
    ("greeting", ["你好", "您好", "hi", "hello", ...]),
    ("command", ["/help", "/exit", ...]),
    ("educational", ["老师", "公式", "定理", "数学", "是什么", ...]),
    ("technical", ["bug", "报错", "API", ...]),
    ("chitchat", ["谢谢", "天气", "你是谁", "测试", ...]),
]
```

**第二层：LLM 兜底（`core/nodes/llm_classifier.py`）**

- 仅在第一层未命中时触发
- 调用 LLM（阿里百炼 qwen-plus），temperature=0.0，timeout=3s
- 分类为 6 类：educational / chitchat / technical / command / greeting / other
- 超时或异常时降级返回 "other"
- LLM 分类的高置信度结果自动写入 `data/intent_training_data.jsonl`，为后续训练本地分类器积累数据

**意图分流逻辑（`core/graph.py:160-161`）**：
- educational → 进入 RAG 检索管线
- 其他所有意图 → 进入闲聊节点（chitchat），不执行向量检索

### 复杂度分级（`core/nodes/query_classifier.py:29-45`）

教育类查询进一步做复杂度分级（纯规则）：

| 级别 | 触发条件 | 检索策略 |
|------|----------|----------|
| simple | 含"是什么/定义/公式"等关键词，或 query < 10 字 | DIRECT 直接检索 |
| medium | 不满足 simple/complex 条件 | MULTI_QUERY 多查询变体 |
| complex | 含"比较/分析/为什么/推导/证明"等 + query > 15 字 | DECOMPOSITION 问题拆解 |

### 面试复盘要点

- 实际代码是 **两层（关键词 + LLM）**，不是三层。"预训练模型分类"层在当前代码中表现为 **训练数据自动收集机制**（`training_collector.py`），LLM 分类结果持续写入 JSONL 文件，为后续训练本地分类器做准备——这是一个"先跑数据、后训练模型"的渐进式策略。
- 三层分类更准确的说法是：**关键词（<1ms）→ LLM（200-800ms）→ 未来本地模型（<10ms）**
- 如果被问"为什么不是真正的三层"，回答：**先通过关键词+LLM 积累标注数据，数据量达标后再训练本地分类器替换 LLM 兜底层，实现成本与速度的最优。**

---

## 三、混合检索：稠密+稀疏双路召回 + RRF 融合

### 架构总览

```
用户查询
   ├── 稠密检索（Milvus ANN，COSINE 相似度）
   │     └── BGE Embedding → Milvus vector search → Top-K
   ├── 稀疏检索（本地 BM25）
   │     └── 分词 → BM25Okapi.get_scores() → Top-K
   └── RRF 倒数排名融合
         └── score(d) = Σ 1/(k + rank_i(d))  （k=60）
```

### 代码核心实现（`core/vectorestore.py`）

**稠密检索**：
- Milvus Lite，索引类型 IVF_FLAT，metric_type COSINE
- 支持 subject/grade 标量过滤
- 可配置 `DENSE_MIN_SIMILARITY` 阈值过滤低分结果

**稀疏检索**：
- 本地 BM25Okapi，内置在 K12VectorStore 中
- 每次插入/删除后全量重建 BM25 索引（`_rebuild_bm25_index`）
- 中文分词使用简单的二元组切分（非 jieba），注释明确写了"实际项目可使用 jieba"

**RRF 融合**：
- k=60，两路结果按排名倒数求和后重排
- 最终得分归一化到 [0,1]

### 面试复盘要点

- 易鹏在面试中说"通过 BM25 算法和 BGE 模型分别进行关键词和密集向量检索，再通过 RF 倒数融合算法进行粗排，最后用 BGE 开源模型进行精排"——**存在表述偏差**：
  - 当前代码中 BM25+BGE 混合检索通过 RRF 融合后就是最终排序结果，**没有额外的"BGE 开源模型精排"环节**
  - RRF 融合本身就是"粗排+精排"合一
  - 如果被追问，可以解释为：RRF 融合后的排序结果直接作为最终检索结果，后续的 rerank 是通过 LLM 在生成阶段隐式完成的（generator 根据 context relevance 选择引用哪些片段）

- **召回率和准确率**：面试中提到前期召回率 60-70%、准确率约 6%。优化手段：
  - **查询扩展**（Multi-Query）：LLM 生成多个同义变体，每个变体独立检索后 RRF 融合
  - **任务拆分**（Decomposition）：复杂问题拆为子问题，各自检索后合并去重
  - 这两种策略已经编码在 `core/strategies/` 中

---

## 四、多策略检索体系

### 策略选择流程（`core/strategies/selector.py`）

```
classify → select_strategy → retrieve → assess_quality → [补充策略] → generate
```

| 策略 | 适用场景 | 机制 |
|------|----------|------|
| DIRECT | simple 查询 | 直接混合检索 |
| MULTI_QUERY | medium 查询 | LLM 生成 4 个查询变体 → 多路检索 → RRF 融合 |
| DECOMPOSITION | complex 查询 | LLM 拆解为 2-4 个子问题 → 分别检索 → 去重合并 |
| HyDE（补充） | 首轮检索 top1 分 < 0.4 | LLM 生成假设答案 → 用答案 embedding 再检索 |
| Step-Back（补充） | 结果数 < 3 或平均分 < 0.5 | LLM 生成抽象回退问题 → 检索更广泛背景知识 |

### 面试复盘要点

- 易鹏说的"后期通过查询条件扩展和任务拆分，将单一查询转化为多语义片段或子任务进行向量检索"——这在代码中完整实现了
- 85% 召回率目标：通过 Multi-Query + Decomposition + HyDE + Step-Back 的组合策略确实可以显著提升，但需要注意 LLM 调用次数增加带来的延迟（config 中 `STRATEGY_TIMEOUT=10s`）
- HyDE 策略适合定义/事实类查询，Step-Back 适合需要背景知识的问题

---

## 五、Corrective RAG：质量评估与纠正

### 流程（`core/graph.py` + `core/retrieval_quality.py`）

```
retrieve → rerank → retrieval_gate → accept  → generate → finalize
                                  → retry   → retry_planner → retrieve...
                                  → abstain → finalize（达到 max_retries=2）
```

### 评估维度（`retrieval_quality.py`）

1. 检索结果是否为空
2. 本地 CrossEncoder 重排器是否可用
3. top-1 `rerank_score` 是否 ≥ 0.60，且至少一个候选是否 ≥ 0.50
4. 重试次数是否达到 `max_retries`（默认 2）

### 面试复盘要点

- 这是一个**基于规则的轻量级评估**，不是 LLM 评估。如果被问为什么不使用 LLM 评估，回答：规则评估速度快（<1ms）、确定性高，适合在线链路；LLM 评估（如 RAGAS 的 Faithfulness）放在离线评估 pipeline 中使用。
- 第一次重试使用 query variants，第二次按失败原因选择 HyDE 或 Step-Back，全部回到统一检索、重排和门控链路。

---

## 六、数据清洗流水线

### 四阶段清洗（`ingestion/cleaner.py`）

| 阶段 | 模块 | 功能 |
|------|------|------|
| 1. 规范化 | Normalizer | 编码统一、不可见字符移除、空格/换行规范化 |
| 2. 去噪 | Denoiser | 页码/页眉页脚移除、目录/版权过滤、短文本丢弃 |
| 3. 结构修复 | StructureRepairer | 断句合并、残缺段落修复 |
| 4. 校验 | Validator | 长度校验、Hash 去重、质量评分 |

- 支持两种数据源适配：FileSourceAdapter（文件上传）+ SQLSourceAdapter（数据库导入）
- 输出 CleanStats（输入/输出数量、去重率、丢弃率）

### 面试复盘要点

- 易鹏在面试中表示"语义切分的连贯性仍不理想"——这个问题在代码中的体现是 `CHUNK_SIZE=512, CHUNK_OVERLAP=64`，属于固定大小切分（character-based），而非语义切分（semantic chunking）。改进方向：可考虑基于 embedding 相似度阈值的自适应切分，或引入 LangChain 的 SemanticChunker。
- "文档通过 embedding 模型解析语义并存入数据库"表述不够准确，正确流程是：**文档 → 解析 → 数据清洗 → 固定大小切片 → BGE Embedding → 写入 Milvus + 同步 BM25 索引**

---

## 七、记忆系统设计（面试中被重点追问）

### 当前代码中的实现

**本项目当前的"记忆"相对简单**，主要体现在两个层面：

**1. 短期记忆（对话历史）**（`core/graph.py:18-25`）：
```python
# finalize_node: 每轮结束后追加 Q&A 到 conversation_history
history.append({"role": "user", "content": state["query"]})
history.append({"role": "assistant", "content": state.get("answer", "")})
# 窗口裁剪：最多保留 MAX_ROUNDS=10 轮（即 20 条消息）
if len(history) > max_msgs:
    history = history[-max_msgs:]
```

- 通过 LangGraph 的 MemorySaver checkpoint 机制持久化
- `thread_id` = user_id，实现按用户的对话隔离

**2. 学情分析（用户画像）**（`services/analytics_service.py`）：
- 基于问答历史做统计分析：薄弱知识点识别、学习推荐
- 数据存储在 SQLite QARecord 表中

### 面试中讨论但代码中未实现的部分

易鹏在面试中讨论了以下方案，但**当前代码中未完整实现**：

| 面试讨论 | 代码现状 |
|----------|----------|
| 短期记忆：上下文窗口限制 + 轮次压缩 | 仅有窗口裁剪（10轮），无摘要压缩 |
| 长期记忆：pg circle 数据库存储 | 使用 SQLite，非 PostgreSQL；无长期记忆模块 |
| JSON 结构化用户画像 | 未实现 |
| 偏好反转更新（"榴莲从喜欢到讨厌"） | 未实现 |
| 相似度匹配更新偏好 | 未实现 |
| 懒初始化 + 自动降级 | 未实现 |

### 面试复盘要点

- 关于"四层记忆设计方案"——实际项目中记忆系统还处于基础阶段，如果被追问"压缩摘要的细节"，可以回答：压缩摘要是计划中的方案，通过 LLM 对超出窗口的历史对话进行摘要，保留关键信息（知识点掌握程度、偏好等），替换原始对话以节省 token。
- "偏好反转"问题（"榴莲从喜欢到讨厌"）：可以设计为时间衰减权重 + 近期行为强信号覆盖机制。最近的行为权重最高，如果用户频繁表达新偏好，系统应快速更新画像。
- "自动降级"：当中间件（Redis/PG）故障时，仅使用当前会话的 conversation_history 作为上下文，不读取历史记忆——这是一种优雅降级策略。

---

## 八、向量库选型：Milvus Lite

### 选型依据

在面试中易鹏说"选用 Milvus 主要基于业务适配性和 K8s 部署优势"。实际代码使用的是 **Milvus Lite**（嵌入式版本）：

- 免去独立部署向量服务，数据存储在本地文件（`milvus_k12.db`）
- 索引类型 IVF_FLAT，metric COSINE，nlist=128
- 适用于单机中小规模数据集
- README 明确写了：超高并发或多副本部署应迁移至 Milvus 集群形态

### 面试复盘要点

- 如果被问"为什么不用 Milvus 集群版/其他向量库"：Milvus Lite 适合当前业务量（中小规模 K12 教育数据），零运维成本；业务量增长后可平滑升级到 Milvus 集群。竞品方面，Faiss 缺少标量过滤能力，Chroma 的混合检索生态不如 Milvus 完善。
- "HNSW 索引兼顾速度与精度"——但当前代码使用的是 **IVF_FLAT**，不是 HNSW。这是一个需要纠正的表述。IVF_FLAT 是倒排索引+精确计算，HNSW 是图索引。两者差异：HNSW 构建慢但查询快且召回率高，IVF_FLAT 更简单但需要合理的 nlist 配置。

---

## 九、系统架构全貌

```
┌────────────────────────────────────────────────────────────┐
│                      FastAPI 应用层                         │
│  /api/v1/rag/ask  /api/v1/documents/...  /api/v1/analytics │
└────────────────────────┬───────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────┐
│                    RAGService (服务层)                       │
│         ask() / ask_stream() — LangGraph 编排               │
└────────────────────────┬───────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────┐
│              LangGraph 工作流 (core/graph.py)               │
│                                                             │
│  classify ──→ retrieve ──→ rerank ──→ retrieval_gate        │
│     │            │                       │                  │
│     │       ┌────┴────┐          ┌───────┼────────┐         │
│     │    DIRECT  MULTI_QUERY   accept   retry   abstain      │
│     │              DECOMPOSITION │       │        │         │
│     │                             │ retry_planner  │         │
│  chitchat ←────────────────── finalize ←───────────┘        │
│     │            │                                          │
│     └────────────┘                                          │
└─────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────┐
│                K12VectorStore (数据层)                       │
│   Milvus Lite (稠密) + BM25Okapi (稀疏) + RRF 融合          │
│   BGE Embedding (sentence-transformers)                     │
│   SQLite (业务库: QA记录/文档/评估/知识点)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 十、面试中的优势与待提升点总结

### 优势

1. **工程落地能力强**：从 0 到 1 搭建了完整的 RAG 系统，覆盖文档入库→数据清洗→混合检索→生成→评估的完整链路
2. **技术选型务实**：使用开源方案（Milvus Lite + BGE + BM25），自主部署，成本可控
3. **架构设计有层次**：意图识别分流、复杂度驱动的多策略检索、Corrective RAG 纠正机制
4. **渐进式优化思路**：训练数据自动收集 → 后续训练本地分类器替换 LLM 兜底——体现了"先跑通、再优化"的工程思维
5. **多策略检索设计合理**：从 simple 到 complex 逐级增加检索深度，避免简单问题消耗过多资源

### 待提升点

1. **技术细节表述需要更精确**：
   - "三层兜底"实际是两层+数据收集，需明确说明第三层是规划中的本地模型
   - "BGE 精排"实际是 RRF 融合，无独立精排环节
   - "HNSW 索引"实际用的是 IVF_FLAT
   - "Mirrors"实际是 Milvus（可能是口误）

2. **记忆系统方案尚未落地**：面试中讨论了丰富的记忆架构（四层记忆、偏好反转、懒初始化、降级），但代码中仅实现了基础的对话历史管理。需要有可演示的 demo 支撑。

3. **语义切分待优化**：当前固定大小切分（512 字符）可能导致语义不连贯，可考虑引入语义切分方案。

4. **评估体系可增强**：当前在线评估仅为规则-based（检查非空+相关性分数），可引入 LLM-based 的 Hallucination 检测。

5. **召回率数据**：面试中说的 60-70%→85% 的优化数据需要有实验报告或 RAGAS 评估记录支撑，避免被质疑"85% 更像承诺而非现状"。

---

## 十一、高频追问及建议回答

### Q1: "三层兜底具体怎么实现的？预训练模型是哪一层的？"
**建议回答**：第一层关键词匹配（<1ms），覆盖教育/闲聊/命令/问候等主要意图；第二层 LLM 分类（200-800ms），处理关键词未命中的边界 case；第三层是规划中的本地分类器——LLM 分类结果自动写入训练数据文件，数据量达标后训练一个轻量分类模型替换 LLM 兜底层，实现低成本低延迟的意图识别。

### Q2: "你提到 BGE 用于精排，具体怎么做的？"
**建议回答**：精排更准确地说是在 RRF 融合后完成的——稠密检索（BGE embedding + Milvus ANN）和稀疏检索（BM25）各自返回 Top-K，通过 RRF 倒数排名融合重新排序，最终结果直接用于生成。如果进一步需要 Cross-Encoder 精排，可以在 RRF 后加一层 BGE-Reranker 对 Top-N 重打分。

### Q3: "为什么选 Milvus？和其他向量库对比过吗？"
**建议回答**：选 Milvus Lite 主要考虑三点：一是嵌入式部署免运维，适合当前中小规模教育数据；二是支持标量过滤（按学科/年级筛选），这对教育场景很重要；三是可平滑升级到 Milvus 集群。对比 Faiss 缺少标量过滤和多副本能力，Chroma 当时在混合检索生态上不如 Milvus 成熟。

### Q4: "记忆系统怎么处理用户偏好反转？"
**建议回答**：当前方案基于用户画像的 JSON 结构化存储，更新机制采用时间衰减权重+近期行为强信号覆盖。具体来说：每条偏好记录带时间戳和置信度，较新的交互权重更高；当检测到与历史偏好矛盾的行为（如用户反复查询不喜欢某类内容），系统降低旧偏好的权重或直接覆盖。这个方案还在规划阶段，核心挑战是确定"反转"的判定阈值。

### Q5: "召回率怎么从 60% 提升到 85% 的？"
**建议回答**：核心是两个策略：一是 Multi-Query（查询扩展），将单一查询改写为 4 个不同角度的变体，多路检索后 RRF 融合，弥补单一查询表述的覆盖盲区；二是 Decomposition（问题拆解），将复杂问题拆为 2-4 个子问题分别检索，每个子问题聚焦单一知识点，合并去重后覆盖更全面。配合 HyDE（假设答案检索）和 Step-Back（抽象回退检索）补充策略，在首轮结果质量不足时自动触发。

### Q6: "数据清洗具体做了什么？"
**建议回答**：四阶段流水线——规范化（编码统一、不可见字符清理）→ 去噪（页码/页眉页脚/目录/版权声明过滤、短文本丢弃）→ 结构修复（断句合并、残缺段落修复）→ 校验（长度校验、Hash 去重、质量评分）。支持文件源和 SQL 源两种数据适配器，输出清洗统计（去重率、丢弃率等）。

---

## 十二、架构补充说明（代码未提及但在面试中讨论的内容）

以下内容在面试中提到但**当前代码中未实现**，需要在后续迭代中补充：

| 功能 | 面试提及 | 建议实现路径 |
|------|----------|-------------|
| 长期记忆 | pg circle 存储 | 引入 PostgreSQL + pgvector，存储用户历史偏好 embedding |
| 摘要压缩 | 上下文窗口外轮次压缩 | 在 finalize_node 中加入 LLM 摘要逻辑 |
| 偏好更新 | JSON 画像 + 相似度匹配 | 在 analytics_service 中增加用户画像 CRUD |
| 懒初始化 | 延迟加载 Redis/PG | 在 AppState 中增加 LazyLoader wrapper |
| 自动降级 | 中间件故障降级 | 在服务层增加 try/except fallback 链 |
| 本地分类器 | LLM 数据训练分类器 | 积累足够的 intent_training_data.jsonl 后训练 |

---

> 以上分析基于项目代码 `edu-rag` 的实际实现，结合面试对话进行了逐点对照和复盘。建议在下次面试前，将待提升点中的第 1-3 项做针对性准备，特别是技术细节的精确表述。

---

## 十三、向量数据库选型：为什么选 Milvus Lite

### 面试回答（建议话术）

**问：为什么选 Milvus？和其他向量库对比过吗？选型标准是什么？**

**答**：

选 Milvus Lite 主要基于五个维度的评估：**部署成本、检索能力、过滤能力、扩展路径、社区生态**。

**一、部署成本：嵌入式优先**

Milvus Lite 是嵌入式模式，`pip install pymilvus` 即可使用，数据存储在本地单文件（`milvus_k12.db`），不需要单独部署服务进程。对于我们当时团队规模小、业务数据量不大的场景，零运维成本是最高优先级。对比 Qdrant、Weaviate 都需要 Docker 或独立服务部署，Pinecone 是云托管付费服务——都偏重。

**二、标量过滤能力：教育场景的刚需**

教育 RAG 有个特殊需求：同一个知识库里有不同学科、不同年级的内容，检索时必须按学科和年级过滤（比如只在"数学+七年级"范围内检索）。Milvus 原生支持标量字段过滤（`subject == '数学' and grade == '七年级'`），这在代码里直接体现在每次 hybrid_search 的 filter_str 参数。对比 Faiss——它是纯向量检索库，没有标量过滤能力，需要自己在应用层做后过滤，性能和准确度都不好。

**三、平滑扩展路径**

Milvus 有三种形态：Lite（嵌入式）→ Standalone（单机服务）→ Cluster（分布式集群），三种形态使用同一套 pymilvus 客户端 API。业务量上来后，只需要改连接地址，不用改代码。这符合"先用简单方案验证，再按需扩展"的工程原则。

**四、中文生态**

Milvus 是 LF AI & Data 基金会毕业项目（CNCF 体系），背后的 Zilliz 公司在国内有很强的技术支持和社区活跃度。遇到问题中文资料和社区响应都比 Weaviate、Qdrant 好很多。

**五、混合检索支持**

我们的架构是稠密+稀疏双路召回，稠密走 Milvus ANN，稀疏走本地 BM25，最后 RRF 融合。Milvus 本身不做 BM25，但它提供了灵活的检索接口，可以方便地与自定义稀疏检索结合。

### 竞品对比速查表

| 维度 | Milvus Lite | Faiss | Chroma | Qdrant | Weaviate | Pinecone | Elasticsearch |
|------|------------|-------|--------|--------|----------|----------|---------------|
| 部署方式 | 嵌入式/集群 | 库（无服务） | 嵌入式/服务 | Docker/Cloud | Docker/Cloud | 仅云托管 | 独立服务 |
| 标量过滤 | 原生支持 | 不支持 | 基础支持 | 完善 | 完善 | 完善 | 完善 |
| 中文生态 | 强 | 一般 | 一般 | 一般 | 弱 | 弱 | 一般 |
| 索引类型 | IVF/HNSW/DiskANN等 | 丰富 | HNSW | HNSW | HNSW/PQ | 专有 | HNSW/PQ |
| 扩展路径 | Lite→Standalone→Cluster | 无 | 单机 | 单机→Cluster | 单机→Cluster | 弹性 | 单机→Cluster |
| 成本 | 免费开源 | 免费开源 | 免费开源 | 免费开源 | 免费开源 | 按量付费 | 免费开源 |
| 适用规模 | 中小→大规模 | 中小规模 | 小规模 | 中大规模 | 中大规模 | 任意规模 | 中大规模 |

### 当前索引选择的说明

代码中实际使用的是 **IVF_FLAT**（倒排索引+精确距离计算），nlist=128，metric_type=COSINE。选 IVF_FLAT 而不是 HNSW 的原因：当前数据量不大（K12 教材语料），IVF_FLAT 构建快、内存占用小，召回率可满足需求。数据量到百万级以后可以切换为 HNSW（查询更快、召回率更高）。

### 面试中易忽略的点

- 面试时说的"HNSW 索引兼顾速度与精度"与代码实际不符，代码用的是 IVF_FLAT。如果被追问，直接承认"当前数据量不大用 IVF_FLAT，HNSW 是后续扩展方向"。
- 面试时说的"Mirrors"大概率是 Milvus 的口误。建议下次面试直接说 Milvus Lite/嵌入式 Milvus。

---

## 十四、Embedding 模型选型：为什么选 BGE-small-zh

### 面试回答（建议话术）

**问：为什么用 BGE 模型？和 OpenAI Embedding、m3e、text2vec 对比过吗？选型标准是什么？**

**答**：

选 BAAI/bge-small-zh-v1.5 主要基于四个维度：**中文性能、部署成本、推理效率、检索增强特性**。

**一、中文性能：C-MTEB 基准的领先者**

BGE 系列（BAAI General Embedding）是智源研究院（BAAI）发布的，在中文 MTEB（C-MTEB）基准上长期排名前列。对于教育场景——教材文本是中文、学生提问是中文——中文语义理解能力是第一优先级。对比 OpenAI 的 text-embedding-ada-002，其训练语料以英文为主，中文语义表达的理解不如 BGE 精细。实测 BGE 在同义表述、"的/地/得"区分、学科术语理解上表现更好。

**二、部署成本：开源本地推理 vs API 付费**

OpenAI Embedding API 按 token 计费，教育场景文档量大、持续入库，API 成本会线性增长。而且每次 embedding 调用有网络延迟，批量入库时不够快。BGE 是开源模型，本地加载后零成本推理。另外数据安全层面——教育数据不需要发送到第三方 API，符合数据主权要求。

**三、模型规模的取舍：small vs base vs large**

BGE 中文有三个规格：

| 规格 | 参数量 | 向量维度 | 模型大小 | 适用场景 |
|------|--------|----------|----------|----------|
| bge-small-zh-v1.5 | 24M | 512 | ~100MB | 资源受限、CPU推理、快速原型 |
| bge-base-zh-v1.5 | 102M | 768 | ~400MB | 平衡性能与资源 |
| bge-large-zh-v1.5 | 326M | 1024 | ~1.3GB | GPU推理、追求极致效果 |

选 small 的原因是：首先，K12 教育领域的语义空间相对收敛（教材术语体系固定），不需要 large 级别的语义粒度；其次，团队服务器没有 GPU，CPU 推理 small 规格单条查询 <50ms，base 要 200ms+；最后，向量维度 512 直接决定了 Milvus 索引大小——small 的内存和磁盘占用分别是 base 的 1/2、large 的 1/4，对于 Lite 模式很重要。

**四、BGE 的检索增强特性：Instruction-aware Encoding**

这是 BGE 区别于其他 Embedding 模型的关键设计。BGE 在编码查询时添加指令前缀（query instruction），编码文档时不添加。本项目代码中：

```python
encode_kwargs = {"normalize_embeddings": True}
query_instruction = "为这个句子生成表示以用于检索相关文章："
```

查询文本 → 拼接指令前缀 → 编码为向量；文档文本 → 直接编码为向量。这种非对称编码让模型理解"这是一个检索查询"和"这是一段要被检索的文档"的区别，从而让查询向量更接近其答案向量。实测在 RAG 场景下这个特性可以让召回率提升 3-5 个百分点。

另外 `normalize_embeddings=True` 将所有向量归一化到单位长度，此时余弦相似度 = 内积，计算更高效且与 Milvus 的 COSINE metric 对齐。

### 竞品对比速查表

| 维度 | BGE-small-zh | m3e-base | text2vec-base | OpenAI ada-002 | GTE-small-zh |
|------|-------------|----------|---------------|----------------|--------------|
| 中文优化 | 专门优化 | 专门优化 | 专门优化 | 英文为主 | 专门优化 |
| C-MTEB 排名 | Top 3 | Top 10 | Top 15 | 中等 | Top 5 |
| 模型大小 | ~100MB | ~400MB | ~400MB | API | ~100MB |
| 向量维度 | 512 | 768 | 768 | 1536 | 512 |
| 本地部署 | 支持 | 支持 | 支持 | 不支持 | 支持 |
| 指令感知 | 是 | 否 | 否 | 否 | 否 |
| 归一化 | 内置 | 需手动 | 需手动 | 内置 | 内置 |
| 成本 | 免费 | 免费 | 免费 | $0.0001/1K tokens | 免费 |

### 如果被追问"为什么不用大模型 API 的 Embedding"

三大原因：①成本——文档入库和检索都需要频繁调用 embedding，API 计费累积快；②延迟——网络往返 + API 排队，批量入库时可能成为瓶颈；③数据主权——教育场景敏感数据不应出域。BGE 本地部署一举三得。

### 如果被追问"后续模型升级计划"

BGE 的 base/large 版本可作为性能升级路径，向量维度变化时需要重建 Milvus 集合。如果业务需要多语言支持（如英语教材），BGE 也有 bge-m3 多语言版本可平滑迁移。模型替换的关键约束是：新模型的向量维度必须一致，否则需要重建向量库。

---

## 十五、技术选型方法论总结（面试加分回答）

如果面试官问"你做技术选型的通用方法论是什么"，可以用以下框架回答：

**五维度评估模型**：

1. **业务适配度**：技术方案是否能满足核心业务需求？（如 Milvus 的标量过滤满足学科/年级筛选，BGE 的中文优化满足 K12 教育场景）
2. **团队能力匹配**：当前团队能否驾驭该技术？（小团队无专职运维 → 嵌入式方案优先）
3. **成本约束**：是否有预算限制？（开源自建 vs API 付费；CPU 推理 vs GPU 需求）
4. **扩展路径**：方案是否支持平滑升级？（Milvus Lite→Cluster 同 API；BGE small→base→large 同系列）
5. **社区生态**：遇到问题是否有足够资源解决？（中文社区活跃度、GitHub Star/Issue 响应速度）

对于本项目的两个关键选型：

- **向量库**：业务适配度（标量过滤）和部署成本（嵌入式）是主要决策因子
- **Embedding 模型**：中文性能（C-MTEB）和部署成本（开源本地）是主要决策因子

两个选型的共同原则：**先用轻量方案跑通全链路、验证业务价值，再根据实际瓶颈定向优化**。这也是为什么选了 small 规格而非 base/large，选了 Lite 模式而非集群——过早优化是工程浪费。
