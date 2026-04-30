"""Pydantic 数据模型：API 请求/响应 Schema"""

from pydantic import BaseModel, Field
from typing import Any


# ==================== 问答接口 ====================

class AskRequest(BaseModel):
    """问答请求"""
    query: str = Field(..., description="用户问题", min_length=1, max_length=2000)
    subject: str | None = Field(None, description="学科过滤（如：数学）")
    grade: str | None = Field(None, description="年级过滤（如：七年级）")
    user_id: str | None = Field(None, description="用户 ID")
    session_id: str | None = Field(None, description="会话 ID")
    stream: bool = Field(False, description="是否流式输出")


class Reference(BaseModel):
    """引用来源"""
    chunk_id: int | None = None
    text: str = ""
    source: str = ""
    score: float = 0.0
    subject: str = ""
    grade: str = ""


class AskResponse(BaseModel):
    """问答响应"""
    code: int = 0
    message: str = "success"
    data: dict[str, Any] | None = None


# ==================== 文档管理接口 ====================

class DocumentUploadResponse(BaseModel):
    """文档上传响应"""
    code: int = 0
    message: str = "success"
    data: dict | None = None


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    code: int = 0
    data: list[dict] = []
    total: int = 0


# ==================== 知识点接口 ====================

class KnowledgePointCreate(BaseModel):
    """知识点创建请求"""
    name: str = Field(..., description="知识点名称")
    subject: str = Field(..., description="所属学科")
    parent_id: str | None = Field(None, description="父知识点 ID")
    description: str = ""
    sort_order: int = 0


class KnowledgePointResponse(BaseModel):
    """知识点响应"""
    code: int = 0
    data: list[dict] | dict | None = None


# ==================== 学情分析接口 ====================

class AnalyticsResponse(BaseModel):
    """学情分析响应"""
    code: int = 0
    data: dict | None = None
