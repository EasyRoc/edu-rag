"""闲聊节点：调用 LLM 以友好方式回应非教育类查询"""

from core.stream_queue import _registry as _stream_queues
from core.nodes.generator import llm_generate_stream
from utils.logger import logger

_CHITCHAT_SYSTEM_PROMPT = (
    "你是一个友好的 K12 学习助手，名叫「知学助手」。"
    "你可以和学生闲聊、打招呼、回答日常问题，但请始终保持友好、鼓励的语气。"
    "如果学生问学习相关的问题，引导他们提出具体的学科问题。"
    "回答要简短自然，不要长篇大论。"
)


async def chitchat_node(state):
    """闲聊节点：调用 LLM 以友好的方式回应非教育类查询"""
    logger.info(f"[节点] chitchat: query='{state['query'][:50]}'")
    queue_id = state.get("_queue_id")
    stream_queue = _stream_queues.get(queue_id) if queue_id else None
    full_answer = ""
    try:
        async for token in llm_generate_stream(
            query=state["query"],
            context_docs=[],
            system_prompt=_CHITCHAT_SYSTEM_PROMPT,
        ):
            full_answer += token
            if stream_queue is not None:
                await stream_queue.put(token)
    finally:
        if stream_queue is not None:
            await stream_queue.put(None)
    return {"answer": full_answer}
