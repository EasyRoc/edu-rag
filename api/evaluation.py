"""RAGAS 评估 API：离线评估、查看历史评估结果"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from config import settings
from evaluation.dataset_builder import EvalDatasetBuilder
from evaluation.pipeline import run_evaluation, run_live_evaluation
from evaluation.schemas import eval_result_to_dict
from models.db_models import EvaluationRecord, get_session_maker
from models.schemas import AskResponse
from sqlalchemy import select, desc
from utils.logger import logger

router = APIRouter(prefix="/api/v1/evaluation", tags=["RAGAS 评估"])

_vector_store = None


def init_router(vector_store):
    global _vector_store
    _vector_store = vector_store


# ----------------------------------------------------------------
# 从 QA 历史记录评估
# ----------------------------------------------------------------
@router.post("/from-history", response_model=AskResponse)
async def evaluate_from_history(
    limit: int = Query(50, description="取最近多少条 QA 记录进行评估"),
    subject: str | None = Query(None, description="按学科过滤"),
    metrics: str | None = Query(None, description="评估指标，逗号分隔 (faithfulness,answer_relevancy,context_precision)"),
):
    """基于历史问答记录运行 RAGAS 评估"""
    logger.info("API 触发评估: from_history, limit=%d, subject=%s", limit, subject)

    metric_list = metrics.split(",") if metrics else None
    dataset = await EvalDatasetBuilder.from_db(limit=limit, subject=subject)
    if len(dataset) == 0:
        return AskResponse(code=1, message="没有找到可评估的问答记录", data=None)

    result = await run_evaluation(
        dataset=dataset,
        name=f"from_history_{subject or 'all'}",
        metrics=metric_list,
    )
    return AskResponse(data=eval_result_to_dict(result))


# ----------------------------------------------------------------
# 上传测试集文件并评估
# ----------------------------------------------------------------
@router.post("/from-file", response_model=AskResponse)
async def evaluate_from_file(
    file_path: str = Query(..., description="测试集 JSON/JSONL 文件路径"),
    metrics: str | None = Query(None, description="评估指标，逗号分隔"),
):
    """上传测试集 JSON 文件并运行评估"""
    metric_list = metrics.split(",") if metrics else None
    dataset = EvalDatasetBuilder.from_file(file_path)
    result = await run_evaluation(
        dataset=dataset,
        name=f"from_file_{file_path}",
        metrics=metric_list,
    )
    return AskResponse(data=eval_result_to_dict(result))


# ----------------------------------------------------------------
# 实时问答 + 评估
# ----------------------------------------------------------------
@router.post("/live", response_model=AskResponse)
async def evaluate_live(
    questions: list[str] = Query(..., description="待评估的问题列表"),
    subject: str | None = Query(None),
    grade: str | None = Query(None),
    metrics: str | None = Query(None),
):
    """实时问答 + 评估：先让 RAG 系统回答问题，再用 RAGAS 评估"""
    if _vector_store is None:
        raise HTTPException(status_code=503, detail="向量存储未初始化")
    metric_list = metrics.split(",") if metrics else None
    result = await run_live_evaluation(
        questions=questions,
        vector_store=_vector_store,
        subject=subject,
        grade=grade,
        metrics=metric_list,
        name="live_eval",
    )
    return AskResponse(data=eval_result_to_dict(result))


# ----------------------------------------------------------------
# 查看历史评估结果
# ----------------------------------------------------------------
@router.get("/history", response_model=AskResponse)
async def list_evaluations(
    limit: int = Query(10, description="返回最近的 N 条评估记录"),
):
    """查看历史评估记录"""
    session_maker = get_session_maker()
    async with session_maker() as session:
        query = (
            select(EvaluationRecord)
            .order_by(desc(EvaluationRecord.created_at))
            .limit(limit)
        )
        rows = (await session.execute(query)).scalars().all()

    data = [
        {
            "id": r.id,
            "task_name": r.task_name,
            "metrics": r.metrics,
            "scores": r.scores,
            "sample_count": r.sample_count,
            "elapsed_seconds": r.elapsed_seconds,
            "created_at": r.created_at.isoformat() if r.created_at else "",
        }
        for r in rows
    ]
    return AskResponse(data={"records": data, "total": len(data)})


@router.get("/history/{record_id}", response_model=AskResponse)
async def get_evaluation_detail(record_id: str):
    """查看单条评估记录的详细信息"""
    session_maker = get_session_maker()
    async with session_maker() as session:
        query = select(EvaluationRecord).where(EvaluationRecord.id == record_id)
        row = (await session.execute(query)).scalar_one_or_none()

    if row is None:
        raise HTTPException(status_code=404, detail="评估记录不存在")

    return AskResponse(data={
        "id": row.id,
        "task_name": row.task_name,
        "metrics": row.metrics,
        "scores": row.scores,
        "sample_count": row.sample_count,
        "samples": row.samples,
        "config_snapshot": row.config_snapshot,
        "elapsed_seconds": row.elapsed_seconds,
        "created_at": row.created_at.isoformat() if row.created_at else "",
    })
