"""闲聊节点：调用 LLM 以友好方式回应非教育类查询"""

from core.stream_queue import stream_queues
from core.nodes.generator import llm_generate_stream
from utils.logger import logger

_CHITCHAT_SYSTEM_PROMPT = (
    "你是一个友好的 K12 学习助手，名叫「知学助手」。"
    "你可以和学生闲聊、打招呼、回答日常问题，但请始终保持友好、鼓励的语气。"
    "如果学生问学习相关的问题，引导他们提出具体的学科问题。"
    "回答要简短自然，不要长篇大论。"
    "请记住对话中用户告诉你的信息（如名字、偏好等），并在后续对话中自然引用这些信息。"
)


async def chitchat_node(state):
    """闲聊节点：调用 LLM 以友好的方式回应非教育类查询"""
    logger.info(f"[节点] chitchat: query='{state['query'][:50]}'")
    queue_id = state.get("_queue_id")
    full_answer = ""
    async for token in llm_generate_stream(
        query=state["query"],
        context_docs=[],
        system_prompt=_CHITCHAT_SYSTEM_PROMPT,
        conversation_history=state.get("conversation_history", []),
    ):
        full_answer += token
        await stream_queues.emit(queue_id, token)
    return {"answer": full_answer}
