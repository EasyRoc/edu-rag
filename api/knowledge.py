"""知识点管理接口：知识点的增删查"""

from fastapi import APIRouter, HTTPException
from models.schemas import KnowledgePointCreate, KnowledgePointResponse
from services.knowledge_service import KnowledgeService
from utils.logger import logger

router = APIRouter(prefix="/api/v1/knowledge-points", tags=["知识点管理"])

knowledge_service: KnowledgeService | None = None


def init_router(service: KnowledgeService):
    global knowledge_service
    knowledge_service = service


@router.get("/tree", response_model=KnowledgePointResponse)
async def get_knowledge_tree(subject: str | None = None):
    """获取知识点树（支持按学科过滤）"""
    if knowledge_service is None:
        raise HTTPException(status_code=503, detail="知识点服务未初始化")
    tree = await knowledge_service.get_knowledge_tree(subject)
    return KnowledgePointResponse(data=tree)


@router.post("/", response_model=KnowledgePointResponse)
async def create_knowledge_point(data: KnowledgePointCreate):
    """创建知识点"""
    if knowledge_service is None:
        raise HTTPException(status_code=503, detail="知识点服务未初始化")
    kp = await knowledge_service.create_knowledge_point(data.model_dump())
    return KnowledgePointResponse(data=kp)


@router.delete("/{kp_id}")
async def delete_knowledge_point(kp_id: str):
    """删除知识点"""
    if knowledge_service is None:
        raise HTTPException(status_code=503, detail="知识点服务未初始化")
    success = await knowledge_service.delete_knowledge_point(kp_id)
    if not success:
        raise HTTPException(status_code=404, detail="知识点不存在")
    return {"code": 0, "message": "删除成功"}
