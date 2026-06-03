"""生成节点：基于检索结果，调用 LLM 生成回答"""
from typing import AsyncGenerator

from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from core.llm import get_chat_model
from utils.logger import logger

# 系统 Prompt 模板 —— 约束 LLM 仅基于检索内容回答
SYSTEM_PROMPT_TEMPLATE = """你是一个专业的 K12 教育助手，名叫"知学助手"。
请根据以下提供的参考资料，回答学生的问题。

## 要求
1. 仅基于参考资料中的内容回答，不要编造事实
2. 如果参考资料不足以回答问题，请明确说明"参考资料中未找到相关信息"
3. 回答要简明易懂，适合 K12 学生的认知水平
4. 适当举例说明，帮助理解
5. 在回答末尾标注引用的参考来源序号（如 [1][2]）

## 参考资料
{context}

## 问题
{query}
"""


async def llm_generate(query: str, context_docs: list[dict]) -> str:
    """
    调用 LLM 生成回答。

    使用 ChatOpenAI 调用兼容 OpenAI API 的服务（如 GPT-4o、Claude、DeepSeek 等）。
    可配置通过 LLM_BASE_URL 切换到任意兼容服务。
    """
    if not settings.LLM_API_KEY:
        logger.warning("未配置 LLM_API_KEY，使用模拟回答模式")
        return _mock_answer(query, context_docs)

    context_parts = []
    for i, doc in enumerate(context_docs):
        context_parts.append(f"[{i+1}] {doc['text']}")
    context = "\n\n".join(context_parts)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(context=context, query=query)),
        HumanMessage(content=query),
    ]

    logger.info(f"调用 LLM: model={settings.LLM_MODEL}, context_docs={len(context_docs)}")
    logger.debug(f"Prompt 长度: {sum(len(m.content) for m in messages)} 字符")

    try:
        llm = get_chat_model(temperature=0.3, max_tokens=2048, timeout=60.0)
        response = await llm.ainvoke(messages)
        answer = response.content
        logger.info(f"LLM 回答生成完成，长度: {len(answer)} 字符")
        return answer
    except Exception as e:
        logger.error(f"LLM 调用异常: {e}")
        return _mock_answer(query, context_docs)


async def llm_generate_stream(
    query: str,
    context_docs: list[dict],
    system_prompt: str | None = None,
    conversation_history: list[dict] | None = None,
) -> AsyncGenerator[str, None]:
    """
    流式调用 LLM，逐个 token 产出回答内容。

    system_prompt: 自定义系统提示词，不传则使用默认的 K12 教育模板
    conversation_history: 历史对话 [{"role": ..., "content": ...}]
    """
    if not settings.LLM_API_KEY:
        logger.warning("未配置 LLM_API_KEY，直接 yield 模拟回答")
        yield _mock_answer(query, context_docs)
        return

    if system_prompt:
        messages = [SystemMessage(content=system_prompt)]
    else:
        context_parts = []
        for i, doc in enumerate(context_docs):
            context_parts.append(f"[{i+1}] {doc['text']}")
        context = "\n\n".join(context_parts)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(context=context, query=query)),
        ]

    if conversation_history:
        for msg in conversation_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=query))

    logger.info(f"流式调用 LLM: model={settings.LLM_MODEL}, context_docs={len(context_docs)}")

    try:
        llm = get_chat_model(temperature=0.3, max_tokens=2048, timeout=120.0)
        async for chunk in llm.astream(messages):
            if chunk.content:
                yield chunk.content
    except Exception as e:
        logger.error(f"LLM 流式调用异常: {e}")
        yield f"\n[LLM 调用异常: {e}]"
        yield _mock_answer(query, context_docs)


def _mock_answer(query: str, context_docs: list[dict]) -> str:
    """
    模拟回答（当未配置 API Key 时使用）。
    简单提取上下文中的片段作为回答，方便测试流程。
    """
    if not context_docs:
        return "抱歉，未找到与该问题相关的参考资料。请尝试换个问法，或联系老师添加相关知识库内容。"

    parts = [f"根据检索到的资料，以下是与「{query}」相关的信息：\n"]
    for i, doc in enumerate(context_docs[:3]):
        text = doc["text"][:200]
        parts.append(f"[{i+1}] {text}")

    parts.append(f"\n（共检索到 {len(context_docs)} 条相关记录，请配置 LLM_API_KEY 以启用智能生成）")
    return "\n\n".join(parts)
