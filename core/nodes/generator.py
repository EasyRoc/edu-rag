"""生成节点：基于检索结果，调用 LLM 生成回答"""
from __future__ import annotations

import asyncio
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

SUB_ANSWER_SYSTEM = """你是一个严谨的 K12 教育助手。
请只根据参考资料回答当前子问题，不要扩展到原问题之外。
如果资料不足，请直接说明该子问题资料不足。"""

SUB_ANSWER_PROMPT = """## 子问题
{sub_query}

## 参考资料
{context}

请用 2-5 句话回答该子问题，并尽量保留关键条件、公式或概念。"""

SYNTHESIS_SYSTEM = """你是一个专业的 K12 教育助手，擅长把多个子问题的答案合成为清晰、完整的总回答。
请严格基于子问题答案和参考资料作答，不要编造事实。"""

SYNTHESIS_PROMPT = """## 原问题
{original_query}

## 子问题答案
{sub_answers}

## 参考资料
{context}

请将以上子问题答案合成为对原问题的完整回答：
1. 先直接回应原问题
2. 再按逻辑说明每个关键点
3. 对 K12 学生保持简明易懂
4. 资料不足的部分要明确说明
5. 末尾标注可对应的参考来源序号（如 [1][2]）"""


def _format_context(context_docs: list[dict]) -> str:
    """把检索片段格式化为带序号的上下文，便于生成阶段引用。"""
    if not context_docs:
        return "（无可用参考资料）"
    parts = []
    for i, doc in enumerate(context_docs):
        text = str(doc.get("text", "")).strip()
        if text:
            parts.append(f"[{i + 1}] {text}")
    return "\n\n".join(parts) or "（无可用参考资料）"


def _build_sub_answer_prompt(sub_query: str, context_docs: list[dict]) -> str:
    """构建子问题回答 prompt。"""
    return SUB_ANSWER_PROMPT.format(
        sub_query=sub_query,
        context=_format_context(context_docs),
    )


def _build_synthesis_prompt(
    original_query: str,
    sub_answers: list[tuple[str, str]],
    context_docs: list[dict],
) -> str:
    """构建复杂问题最终合成 prompt。"""
    if sub_answers:
        sub_answer_text = "\n\n".join(
            f"{idx}. 子问题：{sub_query}\n答案：{answer or '资料不足，未生成有效子答案。'}"
            for idx, (sub_query, answer) in enumerate(sub_answers, start=1)
        )
    else:
        sub_answer_text = "（无可用子问题答案）"

    return SYNTHESIS_PROMPT.format(
        original_query=original_query,
        sub_answers=sub_answer_text,
        context=_format_context(context_docs),
    )


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


async def generate_sub_answers(
    sub_queries: list[str],
    sub_docs_map: dict[str, list[dict]],
) -> list[tuple[str, str]]:
    """逐个子问题生成中间答案，供复杂问题最终合成使用。"""
    clean_queries = [item for item in sub_queries if item]
    if not clean_queries:
        return []
    if not settings.LLM_API_KEY:
        logger.warning("未配置 LLM_API_KEY，跳过复杂问题子答案生成")
        return [(item, "") for item in clean_queries]

    async def _generate_one(sub_query: str) -> tuple[str, str]:
        docs = sub_docs_map.get(sub_query, [])
        messages = [
            SystemMessage(content=SUB_ANSWER_SYSTEM),
            HumanMessage(content=_build_sub_answer_prompt(sub_query, docs)),
        ]
        messages = _trim_messages(messages, settings.LLM_MAX_CONTEXT_TOKENS)
        try:
            llm = get_chat_model(
                temperature=0.2,
                max_tokens=settings.SUB_ANSWER_MAX_TOKENS,
                timeout=60.0,
            )
            response = await llm.ainvoke(messages)
            answer = str(response.content).strip()
            logger.debug("复杂问题子答案生成完成: sub_query=%s, chars=%d", sub_query[:40], len(answer))
            return sub_query, answer
        except Exception as exc:
            logger.warning("复杂问题子答案生成失败: sub_query=%s, err=%s", sub_query[:40], exc)
            return sub_query, ""

    return await asyncio.gather(*(_generate_one(item) for item in clean_queries))


async def synthesize_final_answer(
    original_query: str,
    sub_answers: list[tuple[str, str]],
    context_docs: list[dict],
) -> str:
    """把多个子答案合成为最终回答。"""
    if not settings.LLM_API_KEY:
        logger.warning("未配置 LLM_API_KEY，复杂问题合成使用模拟回答")
        return _mock_answer(original_query, context_docs)

    messages = [
        SystemMessage(content=SYNTHESIS_SYSTEM),
        HumanMessage(content=_build_synthesis_prompt(original_query, sub_answers, context_docs)),
    ]
    messages = _trim_messages(messages, settings.LLM_MAX_CONTEXT_TOKENS)
    try:
        llm = get_chat_model(
            temperature=0.3,
            max_tokens=settings.SYNTHESIS_MAX_TOKENS,
            timeout=120.0,
        )
        response = await llm.ainvoke(messages)
        answer = str(response.content).strip()
        logger.info("复杂问题最终合成完成，长度: %d 字符", len(answer))
        return answer
    except Exception as exc:
        logger.error("复杂问题最终合成失败，回退普通生成: %s", exc)
        return await llm_generate(original_query, context_docs)


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
