"""生成节点：基于检索结果，调用 LLM 生成回答"""
from typing import AsyncGenerator

import httpx
import json

from config import Settings, settings
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

    使用 httpx 调用兼容 OpenAI API 的服务（如 GPT-4o、Claude、DeepSeek 等）。
    可配置通过 LLM_BASE_URL 切换到任意兼容服务。
    """
    if not settings.LLM_API_KEY:
        logger.warning("未配置 LLM_API_KEY，使用模拟回答模式")
        return _mock_answer(query, context_docs)
    # 组装上下文
    context_parts = []
    for i, doc in enumerate(context_docs):
        context_parts.append(f"[{i+1}] {doc['text']}")
    context = "\n\n".join(context_parts)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(context=context, query=query)},
        {"role": "user", "content": query},
    ]
    logger.info(f"调用 LLM: model={settings.LLM_MODEL}, context_docs={len(context_docs)}")
    logger.debug(f"Prompt 长度: {sum(len(m['content']) for m in messages)} 字符")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
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
                    "max_tokens": 2048,
                },
            )
            response.raise_for_status()
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            logger.info(f"LLM 回答生成完成，长度: {len(answer)} 字符")
            return answer
    except httpx.HTTPStatusError as e:
        logger.error(f"LLM API 返回错误: {e.response.status_code} {e.response.text[:200]}")
        return _mock_answer(query, context_docs)
    except Exception as e:
        logger.error(f"LLM 调用异常: {e}")
        return _mock_answer(query, context_docs)



async def llm_generate_stream(
    query: str,
    context_docs: list[dict],
    system_prompt: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    流式调用 LLM，逐个 token 产出回答内容。

    system_prompt: 自定义系统提示词，不传则使用默认的 K12 教育模板
    """
    if not settings.LLM_API_KEY:
        logger.warning("未配置 LLM_API_KEY，直接 yield 模拟回答")
        yield _mock_answer(query, context_docs)
        return

    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
    else:
        context_parts = []
        for i, doc in enumerate(context_docs):
            context_parts.append(f"[{i+1}] {doc['text']}")
        context = "\n\n".join(context_parts)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(context=context, query=query)},
            {"role": "user", "content": query},
        ]

    logger.info(f"流式调用 LLM: model={settings.LLM_MODEL}, context_docs={len(context_docs)}")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{settings.LLM_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.LLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.LLM_MODEL,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 2048,
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    except httpx.HTTPStatusError as e:
        logger.error(f"LLM API 流式调用返回错误: {e.response.status_code}")
        yield f"\n[LLM API 错误: {e.response.status_code}]"
        yield _mock_answer(query, context_docs)
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