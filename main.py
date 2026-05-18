"""
K12 教育 RAG 系统 — FastAPI 应用入口

启动方式:
    python main.py                 # 直接运行
    uvicorn main:app --reload      # 开发模式热重载

环境变量配置请参考 .env.example 文件。
"""

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config import settings
from utils.logger import logger


@dataclass
class AppState:
    """Runtime objects shared by API routes through app.state."""

    vector_store: Any
    rag_graph: Any
    rag_service: Any
    document_service: Any
    knowledge_service: Any
    analytics_service: Any


def init_vector_store_sync():
    """
    同步初始化向量存储（Milvus Lite）。
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


def build_app_state(vector_store: Any | None = None, rag_graph: Any | None = None) -> AppState:
    """Build all runtime services for the app."""
    from services.analytics_service import AnalyticsService
    from services.document_service import DocumentService
    from services.knowledge_service import KnowledgeService
    from services.rag_service import RAGService

    vector_store = vector_store or init_vector_store_sync()
    rag_graph = rag_graph or init_rag_graph_sync(vector_store)
    return AppState(
        vector_store=vector_store,
        rag_graph=rag_graph,
        rag_service=RAGService(vector_store, rag_graph),
        document_service=DocumentService(vector_store),
        knowledge_service=KnowledgeService(),
        analytics_service=AnalyticsService(vector_store),
    )


def attach_app_state(app: FastAPI, state: AppState) -> None:
    """Attach runtime services to app.state."""
    app.state.vector_store = state.vector_store
    app.state.rag_graph = state.rag_graph
    app.state.rag_service = state.rag_service
    app.state.document_service = state.document_service
    app.state.knowledge_service = state.knowledge_service
    app.state.analytics_service = state.analytics_service


async def init_database():
    """初始化业务数据库"""
    from models.db_models import init_db

    logger.info("正在初始化业务数据库...")
    await init_db()
    logger.info("业务数据库就绪")


def register_routers(app: FastAPI) -> None:
    """注册 API 路由"""
    from api import analytics, documents, evaluation, knowledge, rag

    app.include_router(rag.router)
    app.include_router(documents.router)
    app.include_router(knowledge.router)
    app.include_router(analytics.router)
    app.include_router(evaluation.router)
    logger.info("API 路由注册完成")


def create_app(
    app_state: AppState | None = None,
    *,
    initialize_runtime: bool = True,
) -> FastAPI:
    """Create the FastAPI application without import-time heavy initialization."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("=========================================")
        logger.info("  K12 教育 RAG 系统 启动中...")
        logger.info(f"  Milvus 模式: Lite (文件: {settings.MILVUS_URI})")
        logger.info(f"  Embedding 模型: {settings.EMBEDDING_MODEL}")
        logger.info(f"  LLM 模型: {settings.LLM_MODEL}")
        logger.info(f"  日志级别: {settings.LOG_LEVEL}")
        logger.info("=========================================")

        if app_state is not None:
            attach_app_state(app, app_state)
        elif initialize_runtime:
            attach_app_state(app, build_app_state())

        if initialize_runtime:
            await init_database()

        logger.info("系统启动完成，等待请求...")
        yield
        logger.info("系统关闭中...")

    app = FastAPI(
        title="K12 教育 RAG 系统",
        description="基于 RAG 技术的 K12 教育知识库问答系统，支持文档管理、智能问答、学情分析等功能。",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if app_state is not None:
        attach_app_state(app, app_state)

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
    async def health(request: Request):
        """健康检查接口"""
        try:
            vs = getattr(request.app.state, "vector_store", None)
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

    register_routers(app)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    runtime_app = create_app(app_state=build_app_state())
    logger.info(f"启动服务器: {settings.APP_HOST}:{settings.APP_PORT}")
    uvicorn.run(
        runtime_app,
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower(),
    )
