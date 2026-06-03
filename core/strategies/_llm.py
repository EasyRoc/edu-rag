"""策略模块共享的 LLM 调用工具"""

from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from core.llm import get_chat_model
from utils.logger import logger


async def llm_complete(system_prompt: str, user_prompt: str, timeout: float = 10.0) -> str:
    """非流式 LLM 调用，返回文本内容。失败时返回空字符串。"""
    if not settings.LLM_API_KEY:
        logger.warning("未配置 LLM_API_KEY，策略 LLM 调用不可用")
        return ""

    try:
        llm = get_chat_model(temperature=0.3, max_tokens=1024, timeout=timeout)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = await llm.ainvoke(messages)
        content = response.content
        logger.debug(f"策略 LLM 返回: {content[:100]}...")
        return content
    except Exception as e:
        logger.error(f"策略 LLM 调用异常: {e}")
        return ""
