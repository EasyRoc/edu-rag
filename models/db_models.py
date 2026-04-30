"""SQLAlchemy ORM 模型：业务数据表结构"""
from sqlalchemy import Column, String, Integer, Float, Text, DateTime, ForeignKey, JSON, create_engine
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncAttrs
import uuid
from datetime import datetime

from config import settings
from utils.logger import logger


class Base(AsyncAttrs, DeclarativeBase):
    pass



# ==================== 文档表 ====================

class Document(Base):
    __tablename__ = "documents"
    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(256), nullable=False)
    doc_type = Column(String(32), nullable=False)  # pdf, docx, md, txt
    subject = Column(String(32), nullable=False, index=True)
    grade = Column(String(32), default="", index=True)
    chapter = Column(String(128), default="")
    file_path = Column(String(512), nullable=False)
    chunk_count = Column(Integer, default=0)
    status = Column(String(16), default="pending")  # pending, processing, completed, failed
    error_message = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ==================== 知识点表 ====================

class KnowledgePoint(Base):
    """知识点层级表"""
    __tablename__ = "knowledge_points"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(128), nullable=False)
    subject = Column(String(32), nullable=False, index=True)
    parent_id = Column(String(64), ForeignKey("knowledge_points.id"), nullable=True)
    level = Column(Integer, default=0)
    description = Column(Text, default="")
    sort_order = Column(Integer, default=0)

    children = relationship("KnowledgePoint", backref="parent", remote_side=[id], lazy="selectin")


# ==================== 问答记录表 ====================

class QARecord(Base):
    """问答历史记录表"""
    __tablename__ = "qa_records"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(64), nullable=False, index=True)
    query = Column(Text, nullable=False)
    answer = Column(Text, default="")
    subject = Column(String(32), default="")
    grade = Column(String(32), default="")
    complexity = Column(String(16), default="medium")
    retrieved_chunks = Column(JSON, default=list)
    relevance_score = Column(Float, default=0.0)
    feedback = Column(Integer, default=0)                # 1: 好评, -1: 差评, 0: 未反馈
    latency_ms = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

# ==================== 数据库初始化 ====================
_engine = None
_session_maker = None

def get_engine():
    """获取数据库引擎（单例）"""
    global _engine
    if _engine is None:
        db_url = settings.DATABASE_URL
        logger.info(f"初始化数据库连接: {db_url}")
        _engine = create_async_engine(db_url, echo=False)
    return _engine

def get_session_maker():
    """获取异步 Session 工厂"""
    global _session_maker
    if _session_maker is None:
        engine = get_engine()
        _session_maker = async_sessionmaker(engine, expire_on_commit=False)
    return _session_maker

async def init_db():
    """初始化数据库表结构"""
    engine = get_engine()
    logger.info("正在创建数据库表...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("数据库表创建完成")








