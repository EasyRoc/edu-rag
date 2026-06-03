"""LLM 调用工厂：统一使用 langchain-openai 的 ChatOpenAI"""

from langchain_openai import ChatOpenAI

from config import settings


def get_chat_model(
    temperature: float = 0.3,
    max_tokens: int = 1024,
    timeout: float = 60.0,
) -> ChatOpenAI:
    """返回配置好的 ChatOpenAI 实例，兼容所有 OpenAI API 格式的服务"""
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=1,
    )
