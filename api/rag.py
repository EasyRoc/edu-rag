"""RAG 问答接口：处理用户提问并返回回答"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from models.schemas import AskRequest, AskResponse
from utils.logger import logger

router = APIRouter(prefix="/api/v1/rag", tags=["RAG 问答"])


def get_rag_service(request: Request):
    service = getattr(request.app.state, "rag_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="RAG 服务未初始化")
    return service


@router.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest, request: Request):
    """RAG 问答接口：根据用户问题检索知识库并生成回答"""
    rag_service = get_rag_service(request)

    logger.info(f"收到问答请求: query='{req.query[:50]}', subject={req.subject}, grade={req.grade}")

    try:
        result = await rag_service.ask(
            query=req.query,
            subject=req.subject,
            grade=req.grade,
            user_id=req.user_id,
            session_id=req.session_id,
            stream=req.stream,
        )

        return AskResponse(data={
            "answer": result["answer"],
            "references": result["references"],
            "latency_ms": result["latency_ms"],
            "complexity": result.get("complexity", "medium"),
            "record_id": result.get("record_id"),
            "session_id": result["session_id"],
        })

    except Exception as e:
        logger.error(f"问答处理异常: {e}")
        return AskResponse(code=500, message=f"服务器内部错误: {str(e)}")


@router.post("/ask-stream")
async def ask_question_stream(req: AskRequest, request: Request):
    """流式 RAG 问答接口（SSE），逐 token 返回回答"""
    rag_service = get_rag_service(request)

    logger.info(f"收到流式问答请求: query='{req.query[:50]}', subject={req.subject}, grade={req.grade}")

    session_id = rag_service.resolve_session_id(req.session_id, req.user_id)
    return StreamingResponse(
        rag_service.ask_stream(
            query=req.query,
            subject=req.subject,
            grade=req.grade,
            user_id=req.user_id,
            session_id=session_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Session-ID": session_id,
        },
    )


@router.post("/feedback")
async def submit_feedback(record_id: str, feedback: int):
    """提交问答反馈（1: 好评, -1: 差评）"""
    from models.db_models import QARecord, get_session_maker
    session_maker = get_session_maker()
    async with session_maker() as session:
        record = await session.get(QARecord, record_id)
        if not record:
            raise HTTPException(status_code=404, detail="问答记录不存在")
        record.feedback = feedback
        await session.commit()
        logger.info(f"反馈已提交: record_id={record_id}, feedback={feedback}")
        return {"code": 0, "message": "反馈成功"}
