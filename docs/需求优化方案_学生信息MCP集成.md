# 知学助手 — K12 教育 RAG 系统需求优化方案

> 文档版本：v1.0  
> 日期：2026-05-10  
> 范围：系统能力分析 + 学生信息查询 MCP 集成方案

---

## 一、系统现状

### 1.1 当前架构

系统采用 **RAG（检索增强生成）** 架构，核心链路如下：

```
用户提问 → 查询分类 → 混合检索 → LLM 生成 → 质量评估 → 返回答案
                ↑                          ↑
          稠密+稀疏检索              Corrective 重试机制
```

**已实现的能力：**

| 模块 | 功能 | 状态 |
|------|------|------|
| 文档管理 | PDF/MD/TXT 上传、解析、切片、入库 | 完成 |
| 向量存储 | Milvus Lite + BGE Embedding + BM25 混合检索 | 完成 |
| 知识问答 | 基于 LangGraph 的 RAG 工作流，含分类-检索-生成-评估 | 完成 |
| 质量门控 | Corrective RAG：空回答检测、低分重试 | 完成 |
| 学情分析 | 薄弱知识点分析、问答历史、复习推荐 | 完成 |
| 数据持久化 | SQLite 存储文档、问答记录、知识点树 | 完成 |
| REST API | FastAPI，涵盖问答、文档、知识点、分析等 | 完成 |
| RAGAS 评估 | faithfulness/answer_relevancy/context_precision 离线评估 | v2 新增 |

### 1.2 数据模型中的学生信息

当前系统中与「学生」相关的数据分散存储，缺乏统一视图：

| 数据 | 存储位置 | 关联字段 | 说明 |
|------|----------|----------|------|
| 问答历史 | `qa_records` 表 | `user_id` | 记录每个学生的提问和回答 |
| 用户反馈 | `qa_records.feedback` | `user_id` | 1 好评 / -1 差评 |
| 薄弱知识点 | 实时计算（AnalyticsService） | `user_id` | 聚合分析生成，不持久化 |
| 学科信息 | `qa_records.subject` | `user_id` | 学生提问涉及哪些学科 |
| 年级信息 | `qa_records.grade` | `user_id` | 学生所属年级 |

### 1.3 现有局限

1. **学生信息无统一入口** — 学情分析需要通过 REST API 逐项查询，无法被 AI 自主调用
2. **无法自然语言查询学生** — 系统不支持「张三这个月数学进步了吗？」这类语义查询
3. **无跨会话上下文** — 每次提问独立执行，无法关联同一学生的历史数据做连贯分析
4. **数据服务仅限 HTTP 暴露** — 缺乏标准化的工具协议，无法被 AI Agent 生态直接消费
5. **无教师/家长视角** — 缺乏批量查看多个学生、对比分析的能力

---

## 二、优化目标

### 2.1 核心目标

在现有 RAG 答疑能力之上，增加 **学生信息智能查询** 能力，通过 **MCP（Model Context Protocol）** 协议暴露数据工具，使学生数据分析可被 AI 自主编排和查询。

### 2.2 具体目标

1. **统一学生数据视图** — 将分散在学生问答记录中的信息整合为可查询的结构化数据
2. **MCP 工具化** — 通过 MCP Server 暴露学生信息查询工具，支持 AI 自动调用
3. **语义化查询** — 支持自然语言到结构化查询的转换（如「小明最近哪科退步了？」）
4. **与 RAG 流程融合** — 学生数据可作为 RAG 问答的上下文补充，实现个性化回答

---

## 三、优化方案：学生信息查询 MCP 集成

### 3.1 什么是 MCP

MCP（Model Context Protocol）是 Anthropic 推出的开源协议，用于标准化 AI 模型与外部数据源/工具之间的交互。其核心概念：

```
┌─────────────┐     MCP 协议      ┌──────────────┐
│  AI 客户端   │ ◄──────────────► │  MCP Server  │
│ (Claude等)   │   Tools/Resources │              │
└─────────────┘                   │  ┌──────────┐ │
                                  │  │ 数据源 A  │ │
                                  │  ├──────────┤ │
                                  │  │ 数据源 B  │ │
                                  │  └──────────┘ │
                                  └──────────────┘
```

通过 MCP 集成，教师或学生可以用自然语言向 AI 询问学情，AI 自动调用 MCP 工具查询底层数据并返回分析结果。

### 3.2 架构设计

```
┌──────────────────────────────────────────────────────────┐
│                    用户交互层                              │
│  Claude Desktop / 自定义 AI Chat / VS Code AI             │
└─────────────────────┬────────────────────────────────────┘
                      │ MCP 协议 (JSON-RPC over stdio/SSE)
                      ▼
┌──────────────────────────────────────────────────────────┐
│                  MCP Server Layer                         │
│  ┌────────────────────────────────────────────────────┐  │
│  │  mcp_student_server.py                             │  │
│  │  +  FastMCP 服务实例                                │  │
│  │  +  工具注册 → 数据查询 → 结果返回                   │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────┬────────────────────────────────────┘
                      │ 内部调用
                      ▼
┌──────────────────────────────────────────────────────────┐
│                  业务服务层                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐│
│  │RAGService│ │Analytics │ │Knowledge│ │Document      ││
│  │          │ │Service   │ │Service  │ │Service       ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘│
└─────────────────────┬────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│                  数据持久层                                │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────┐   │
│  │  Milvus    │  │  SQLite    │  │  (可选) PostgreSQL│   │
│  │ 向量存储    │  │ 业务数据库  │  │  未来扩展         │   │
│  └────────────┘  └────────────┘  └──────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

**整合到现有项目的路径：**

```
edu-rag/
├── mcp/
│   ├── __init__.py
│   ├── server.py              # MCP Server 主入口
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── student_tools.py   # 学生信息查询工具
│   │   └── analytics_tools.py # 学情分析工具
│   └── config.py              # MCP 服务配置
├── services/
│   ├── analytics_service.py   # ← 增强：增加更多学生分析维度
│   └── student_service.py     # ← 新增：学生专属服务
├── models/
│   └── db_models.py           # ← 扩展：增加学生画像表
├── main.py                    # ← 可选：注册 MCP 路由
└── cli/
    └── run_mcp_server.py      # MCP Server 启动脚本
```

### 3.3 MCP 工具定义

#### 3.3.1 学生基础信息查询

| 工具名称 | 功能 | 输入参数 | 输出 |
|----------|------|----------|------|
| `get_student_profile` | 获取学生画像 | `user_id` | 年级、常考学科、活跃度、综合表现 |
| `list_recent_questions` | 查询近期提问 | `user_id`, `limit`, `subject?` | 问题列表含答案、评分、时间 |
| `get_student_stats` | 获取统计数据 | `user_id` | 总提问数、好评/差评率、平均延迟 |

#### 3.3.2 学情分析工具

| 工具名称 | 功能 | 输入参数 | 输出 |
|----------|------|----------|------|
| `get_weak_points` | 薄弱知识点分析 | `user_id`, `subject?` | 按薄弱程度排序的知识点列表 |
| `get_progress_trend` | 学习趋势分析 | `user_id`, `subject?`, `days` | 各学科趋势、得分变化、差评分布 |
| `recommend_review` | 复习推荐 | `user_id`, `subject?` | 针对薄弱点的复习材料 |
| `compare_students` | 学生对比 | `user_ids[]`, `subject?` | 多学生在同一学科上的对比 |

#### 3.3.3 知识库工具

| 工具名称 | 功能 | 输入参数 | 输出 |
|----------|------|----------|------|
| `query_knowledge_base` | 知识库检索 | `query`, `subject?`, `grade?` | 检索到的知识点片段 |
| `get_knowledge_tree` | 知识点树 | `subject?` | 学科知识点层级结构 |

#### 3.3.4 语义查询工具（高阶）

| 工具名称 | 功能 | 输入参数 | 输出 |
|----------|------|----------|------|
| `ask_about_student` | 自然语言问学生 | `user_id`, `question` | AI 理解意图后组合多工具查询并回答 |

### 3.4 MCP Server 核心实现

```python
# mcp/server.py
from mcp.server.fastmcp import FastMCP

# 创建 MCP 服务实例
mcp = FastMCP("K12 Student Assistant", version="1.0.0")


@mcp.tool()
async def get_weak_points(
    user_id: str,
    subject: str | None = None,
) -> list[dict]:
    """获取学生的薄弱知识点，按薄弱程度降序排列"""
    service = AnalyticsService(vector_store)
    return await service.get_weak_points(user_id, subject)


@mcp.tool()
async def get_student_stats(user_id: str) -> dict:
    """获取学生的整体学习统计数据"""
    service = StudentService()
    return await service.get_stats(user_id)


@mcp.resource("student://{user_id}/profile")
async def student_profile(user_id: str) -> str:
    """将学生画像暴露为可读资源"""
    service = StudentService()
    profile = await service.get_profile(user_id)
    return format_profile_markdown(profile)
```

### 3.5 需要扩展的数据模型

在现有 `qa_records` 基础上，新增学生维度数据：

```python
# models/db_models.py — 新增表

class StudentProfile(Base):
    """学生画像表"""
    __tablename__ = "student_profiles"

    user_id = Column(String(64), primary_key=True)
    name = Column(String(64), default="")
    grade = Column(String(32), default="")
    subjects = Column(JSON, default=list)       # ["数学", "语文", "英语"]
    teacher_id = Column(String(64), default="") # 关联教师
    extra = Column(JSON, default=dict)          # 扩展字段
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class StudentScore(Base):
    """学生考试成绩表（可选扩展）"""
    __tablename__ = "student_scores"

    id = Column(String(64), primary_key=True)
    user_id = Column(String(64), nullable=False, index=True)
    subject = Column(String(32), nullable=False)
    score = Column(Float, nullable=False)
    total_score = Column(Float, default=100.0)
    exam_name = Column(String(128), default="")
    exam_date = Column(DateTime, default=datetime.utcnow)
```

### 3.6 新增服务层

```python
# services/student_service.py
class StudentService:
    """学生信息服务：聚合多源数据，提供学生维度的统一查询接口"""

    async def get_profile(self, user_id: str) -> dict:
        """获取学生画像（含统计摘要）"""
        # 1. 查询基础信息（StudentProfile 表）
        # 2. 聚合 QA 记录统计
        # 3. 返回完整画像

    async def get_stats(self, user_id: str) -> dict:
        """获取统计数据"""
        # 总提问数、各学科分布、好评/差评率、活跃天数等

    async def get_progress_trend(self, user_id: str, subject: str | None = None, days: int = 30) -> list[dict]:
        """学习趋势：按时间序列分析反馈变化"""
        # 按天聚合 feedback 数据，计算滑动平均

    async def compare_students(self, user_ids: list[str], subject: str | None = None) -> list[dict]:
        """多学生对比分析"""
        # 对每个学生计算统计指标，输出对比表
```

### 3.7 集成方式

#### 方案 A：独立 MCP Server 进程（推荐）

```
# cli/run_mcp_server.py
# 启动独立的 MCP Server 进程，通过 stdio 通信

python cli/run_mcp_server.py
# 或集成到项目
python mcp/server.py
```

- 与 FastAPI 进程解耦，互不影响
- 可通过 stdio（本地）或 SSE（远程）两种传输协议访问
- 适合 Claude Desktop、VS Code 等 MCP 客户端直接连接

#### 方案 B：与 FastAPI 同进程

- MCP Server 作为 FastAPI 的子路由运行
- 共享内存中的向量存储和服务实例
- 适合统一部署、减少资源占用

#### 方案 C：混合模式

- MCP Server 独立进程，但共享 SQLite 数据库
- 利用 Milvus Lite 的文件级并发能力
- 兼顾独立性和数据一致性

**推荐方案 A**，原因：
1. MCP 客户端（Claude Desktop 等）原生支持 stdio 模式连接独立进程
2. 与 REST API 解耦，变更互不影响
3. 独立进程可以按需重启，不影响主服务

### 3.8 MCP 客户端配置

用户连接 MCP Server 只需在客户端配置中添加：

```json
{
  "mcpServers": {
    "k12-student-assistant": {
      "command": "python",
      "args": ["/path/to/edu-rag/mcp/server.py"],
      "env": {
        "K12_MILVUS_URI": "./milvus_k12.db",
        "LLM_API_KEY": "sk-xxx"
      }
    }
  }
}
```

### 3.9 与现有系统交互流程

```
用户问：小明最近数学学得怎么样？
                │
                ▼
  AI 理解意图 → 调用 MCP tool: get_weak_points("xiaoming", "数学")
                │
                ▼
  MCP Server  → AnalyticsService.get_weak_points()
                │
                ▼
            SQLite 查询 qa_records
            聚合计算薄弱知识点
                │
                ▼
  返回: [{"knowledge_point": "二次函数", "weakness_score": 0.85}, ...]
                │
                ▼
  AI 组合结果 + 知识库检索 → 生成可读回答：
  「小明同学在数学方面需要重点关注：
   1. 二次函数（薄弱分 0.85）
   2. 几何证明（薄弱分 0.72）
   建议：...」
```

### 3.10 安全与权限

| 维度 | 措施 |
|------|------|
| 身份认证 | MCP Server 启动时校验 API Token |
| 数据隔离 | `user_id` 维度过滤，一个学生只能查自己的数据 |
| 教师权限 | 教师 `teacher_id` 关联学生列表，可查名下学生 |
| 审计日志 | 所有 MCP 工具调用记录日志，包含时间、工具名、参数 |
| 敏感信息 | 返回结果自动脱敏，不暴露完整姓名 |

---

## 四、实施路线图

### 阶段一：基础建设（1-2 天）

| 任务 | 产出 | 前置依赖 |
|------|------|----------|
| 安装 MCP SDK | `pip install mcp` | 无 |
| 新增 `StudentProfile` 表 | DB migration | 现有 DB 模型 |
| 新增 `StudentService` | 学生画像/统计/趋势服务 | DB 表 |
| 增强 `AnalyticsService` | 增加趋势分析、对比分析 | 现有 QA 记录 |

### 阶段二：MCP Server（1-2 天）

| 任务 | 产出 | 前置依赖 |
|------|------|----------|
| 创建 `mcp/` 模块 | 目录结构和基础框架 | 阶段一 |
| 实现学生查询工具 | `get_student_profile`, `get_weak_points` 等 | StudentService |
| 实现学情分析工具 | `get_progress_trend`, `compare_students` | AnalyticsService |
| 实现知识库工具 | `query_knowledge_base` | RAGService |
| 启动脚本 | `cli/run_mcp_server.py` | 所有 tools |

### 阶段三：集成与测试（1 天）

| 任务 | 产出 | 前置依赖 |
|------|------|----------|
| 本地 MCP Server 调试 | Claude Desktop 成功连接 | 阶段二 |
| 端到端测试 | 覆盖所有工具的测试用例 | MCP Server |
| 错误处理 | 超时、空结果、非法参数处理 | 测试结果 |
| 文档 | 使用说明、API 文档 | 全部完成 |

### 阶段四：进阶能力（可选）

| 任务 | 说明 |
|------|------|
| 语义查询工具 `ask_about_student` | 利用 LLM 做意图识别，自动编排多工具 |
| 批量导入学生数据 | 支持 CSV/Excel 导入学生信息 |
| SSE 模式部署 | 远程 MCP Server，支持 HTTP 传输 |
| 对接外部系统 | 接入教务系统、考试系统的学生数据 |

---

## 五、收益评估

| 维度 | 当前状态 | 优化后 |
|------|----------|--------|
| 学生数据查询 | 需手动调 REST API，逐项查 | 自然语言一句话查询 |
| 学情分析深度 | 基础薄弱点 + 历史 | 趋势分析 + 跨学生对比 |
| AI 自动化程度 | 仅限问答 | 学生数据+问答融合编排 |
| 生态兼容性 | 仅 HTTP API | MCP 标准，任意 MCP 客户端可用 |
| 教师使用体验 | 需开发前端界面 | Claude Desktop 即可使用 |
| 数据利用效率 | 仅用于学情展示 | 可作为 RAG 上下文个性化回答 |

---

## 六、技术依赖

```
# 新增依赖
mcp>=1.0.0                    # MCP 协议 SDK (FastMCP)
```

MCP SDK 由 Anthropic 官方维护，轻量无侵入，与现有框架无冲突。

---

## 七、附录

### A. 现有 `AnalyticsService` 复用度分析

| 现有方法 | 是否可复用 | 改造说明 |
|----------|-----------|----------|
| `get_weak_points()` | ✅ 可直接复用 | 仅需适配 MCP 工具入参格式 |
| `get_history()` | ✅ 可直接复用 | 返回列表转 MCP 资源格式 |
| `recommend_review()` | ✅ 可直接复用 | 需额外调知识库检索 |

### B. MCP SDK 参考

- 官方文档：https://modelcontextprotocol.io/
- Python SDK：https://github.com/modelcontextprotocol/python-sdk
- FastMCP 快速开始：`pip install mcp && python -m mcp`
