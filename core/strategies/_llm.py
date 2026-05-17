"""策略模块共享的 LLM 调用工具"""
import httpx
from config import settings
from utils.logger import logger


async def llm_complete(system_prompt: str, user_prompt: str, timeout: float = 10.0) -> str:
    """非流式 LLM 调用，返回文本内容。失败时返回空字符串。"""
    if not settings.LLM_API_KEY:
        logger.warning("未配置 LLM_API_KEY，策略 LLM 调用不可用")
        return ""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{settings.LLM_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.LLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.LLM_MODEL,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 1024,
                },
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            logger.debug(f"策略 LLM 返回: {content[:100]}...")
            return content
    except httpx.HTTPStatusError as e:
        logger.error(f"策略 LLM API 错误: {e.response.status_code}")
        return ""
    except Exception as e:
        logger.error(f"策略 LLM 调用异常: {e}")
        return ""
