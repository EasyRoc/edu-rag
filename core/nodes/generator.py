"""生成节点：基于检索结果，调用 LLM 生成回答"""
from __future__ import annotations

from typing import AsyncGenerator

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from config import settings
from core.llm import get_chat_model
from utils.logger import logger

# token 计数器：优先用 tiktoken，不可用时回退到字符估算
try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except Exception:
    def _count_tokens(text: str) -> int:
        # 中英文混合估算：中文 ~1.5 char/token，英文 ~4 char/token
        # 取粗略平均 2.5 char/token
        return max(1, len(text) // 2)


def _trim_messages(messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    """裁剪消息列表到 max_tokens 以内，保留 system 消息和最近的对话。

    规则：
    - SystemMessage（首条）始终保留，因为包含参考资料和系统指令
    - 从旧到新丢弃中间的 history 消息，保证最新的消息不丢
    - 最坏情况只保留 system + 当前 query（最后一条 HumanMessage）
    """
    if not messages:
        return messages

    sys_msg = messages[0]
    # 分离首条 system 消息和后续消息
    body = messages[1:] if isinstance(sys_msg, SystemMessage) else messages
    system = [sys_msg] if isinstance(sys_msg, SystemMessage) else []

    # 固定开销：system prompt + 当前 query（最后一条）+ 预留响应 token
    fixed = _count_tokens(sys_msg.content if system else "")
    last = messages[-1]  # 当前 query
    fixed += _count_tokens(last.content if hasattr(last, "content") else str(last))
    reserve = settings.LLM_API_KEY and 2048 + int(max_tokens * 0.05) or 0  # max_tokens + 5% 余量
    budget = max_tokens - fixed - reserve

    if budget <= 0:
        # 连 system + query 都快撑满了，只能保留最小集
        logger.warning(
            "上下文窗口紧张: max=%d, fixed=%d, 丢弃全部历史消息",
            max_tokens, fixed,
        )
        return system + [last]

    # 从旧到新丢弃 body 消息，直到剩余消息落在 budget 内
    total = sum(_count_tokens(m.content if hasattr(m, "content") else str(m)) for m in body)
    if total <= budget:
        return messages  # 没超，原样返回

    for cutoff in range(0, len(body) - 1):
        candidate = body[cutoff:]
        candidate_tokens = sum(
            _count_tokens(m.content if hasattr(m, "content") else str(m)) for m in candidate
        )
        if candidate_tokens <= budget:
            trimmed_count = cutoff
            logger.info(
                "上下文窗口裁剪: max=%d, fixed=%d, budget=%d, 裁剪历史 %d 条消息",
                max_tokens, fixed, budget, trimmed_count,
            )
            return system + candidate

    # 全丢了也装不下，只保留 system + 当前 query
    logger.warning(
        "上下文窗口不足: max=%d, fixed=%d, 丢弃全部 %d 条历史消息",
        max_tokens, fixed, len(body),
    )
    return system + [last]

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

    # 上下文窗口裁剪：历史消息过长时从旧到新丢弃，保证不超出模型上限
    max_context = settings.LLM_MAX_CONTEXT_TOKENS
    messages = _trim_messages(messages, max_context)

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
