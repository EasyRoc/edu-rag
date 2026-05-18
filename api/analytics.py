"""学情分析接口：薄弱知识点、问答历史、复习推荐"""

from fastapi import APIRouter, HTTPException, Request
from models.schemas import AnalyticsResponse

router = APIRouter(prefix="/api/v1/analytics", tags=["学情分析"])


def get_analytics_service(request: Request):
    service = getattr(request.app.state, "analytics_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="学情分析服务未初始化")
    return service


@router.get("/weak-points/{user_id}", response_model=AnalyticsResponse)
async def get_weak_points(user_id: str, request: Request, subject: str | None = None):
    """获取学生薄弱知识点"""
    analytics_service = get_analytics_service(request)
    weak = await analytics_service.get_weak_points(user_id, subject)
    return AnalyticsResponse(data={"user_id": user_id, "weak_points": weak})


@router.get("/history/{user_id}", response_model=AnalyticsResponse)
async def get_history(user_id: str, request: Request, limit: int = 20):
    """获取学生问答历史"""
    analytics_service = get_analytics_service(request)
    history = await analytics_service.get_history(user_id, limit)
    return AnalyticsResponse(data={"user_id": user_id, "history": history, "total": len(history)})


@router.get("/recommend/{user_id}", response_model=AnalyticsResponse)
async def recommend_review(user_id: str, request: Request, subject: str | None = None):
    """获取复习推荐内容"""
    analytics_service = get_analytics_service(request)
    rec = await analytics_service.recommend_review(user_id, subject)
    return AnalyticsResponse(data={"user_id": user_id, **rec})
