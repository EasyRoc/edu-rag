"""
K12 教育 RAG 系统 — FastAPI 应用入口

启动方式:
    python main.py                 # 直接运行
    uvicorn main:app --reload      # 开发模式热重载

环境变量配置请参考 .env.example 文件。
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config import settings
from utils.logger import logger

# ==================== 全局变量 ====================
# 在模块级别初始化（同步），避免在 asyncio 事件循环内启动 Milvus Lite
_vector_store = None
_rag_graph = None


# ==================== 应用生命周期管理 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动和关闭时的生命周期管理"""
    logger.info("=========================================")
    logger.info("  K12 教育 RAG 系统 启动中...")
    logger.info(f"  Milvus 模式: Lite (文件: {settings.MILVUS_URI})")
    logger.info(f"  Embedding 模型: {settings.EMBEDDING_MODEL}")
    logger.info(f"  LLM 模型: {settings.LLM_MODEL}")
    logger.info(f"  日志级别: {settings.LOG_LEVEL}")
    logger.info("=========================================")

    # 使用模块级别已初始化的实例
    app.state.vector_store = _vector_store
    app.state.rag_graph = _rag_graph
    await init_services(app)
    await init_database()
    await register_routers(app)

    logger.info("系统启动完成，等待请求...")
    yield

    # 关闭时的清理
    logger.info("系统关闭中...")


# ==================== 应用初始化 ====================

app = FastAPI(
    title="K12 教育 RAG 系统",
    description="基于 RAG 技术的 K12 教育知识库问答系统，支持文档管理、智能问答、学情分析等功能。",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 配置（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def init_database():
    """初始化业务数据库"""
    from models.db_models import init_db
    logger.info("正在初始化业务数据库...")
    await init_db()
    logger.info("业务数据库就绪")


async def init_services(app: FastAPI):
    """初始化各业务服务"""
    from services.rag_service import RAGService
    from services.document_service import DocumentService
    from services.knowledge_service import KnowledgeService
    from services.analytics_service import AnalyticsService

    app.state.rag_service = RAGService(app.state.vector_store, app.state.rag_graph)
    app.state.document_service = DocumentService(app.state.vector_store)
    app.state.knowledge_service = KnowledgeService()
    app.state.analytics_service = AnalyticsService(app.state.vector_store)

    logger.info("业务服务初始化完成")


async def register_routers(app: FastAPI):
    """注册 API 路由"""
    from api import rag, documents, knowledge, analytics, evaluation

    # 注入服务实例
    rag.init_router(app.state.rag_service)
    documents.init_router(app.state.document_service)
    knowledge.init_router(app.state.knowledge_service)
    analytics.init_router(app.state.analytics_service)
    evaluation.init_router(app.state.vector_store)

    # 注册路由
    app.include_router(rag.router)
    app.include_router(documents.router)
    app.include_router(knowledge.router)
    app.include_router(analytics.router)
    app.include_router(evaluation.router)

    logger.info("API 路由注册完成")


# ==================== 根路径 ====================
@app.get("/")
async def root():
    """返回 UI 界面"""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {
        "app": "K12 教育 RAG 系统",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """健康检查接口"""
    try:
        vs = app.state.vector_store
        if vs is None:
            return {"status": "unhealthy", "error": "vector_store not initialized"}
        stats = vs.collection_stats
        return {
            "status": "healthy",
            "vector_store": stats,
            "llm_configured": bool(settings.LLM_API_KEY),
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# ==================== 启动入口 ====================

def init_vector_store_sync():
    """
    同步初始化向量存储（Milvus Lite）。
    必须在 asyncio 事件循环启动之前调用，因为 Milvus Lite 启动子进程时不兼容 asyncio。
    """
    from core.vectorestore import K12VectorStore
    logger.info("正在初始化向量存储（同步）...")
    vs = K12VectorStore()
    stats = vs.collection_stats
    logger.info(f"向量存储就绪，当前数据量: {stats.get('row_count', 0)} 条")
    return vs


def init_rag_graph_sync(vector_store):
    """同步初始化 LangGraph 工作流"""
    from core.graph import build_rag_graph
    logger.info("正在构建 RAG 工作流...")
    graph = build_rag_graph(vector_store)
    logger.info("RAG 工作流构建完成")
    return graph


if __name__ == "__main__":
    import uvicorn

    # 在 uvicorn 启动之前，同步初始化 Milvus Lite 和 LangGraph
    # 这是关键：Milvus Lite 启动本地服务进程，不能在 asyncio 循环内执行
    _vector_store = init_vector_store_sync()
    _rag_graph = init_rag_graph_sync(_vector_store)
    logger.info(f"启动服务器: {settings.APP_HOST}:{settings.APP_PORT}")
    uvicorn.run(
        app,
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower(),
    )
