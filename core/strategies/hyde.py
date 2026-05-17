"""HyDE 策略：生成假设性答案，用假设答案的 embedding 进行检索"""
from core.strategies._llm import llm_complete
from utils.logger import logger

HYDE_SYSTEM = """你是一个 K12 教育助手。请针对用户的问题，写一段假设性的标准答案。

要求：
1. 内容应像教科书或教辅资料中的标准解释
2. 包含关键概念和知识点
3. 即使不确定细节，也尽量写出合理的假设性答案
4. 长度控制在 100-200 字
5. 这个答案将用于以其 embedding 检索相关文档，不会被直接展示给用户

只输出假设答案本身，不要任何解释。"""


async def generate_hypothetical_answer(query: str) -> str:
    """生成假设性答案用于 HyDE 检索，失败时返回空字符串"""
    result = await llm_complete(HYDE_SYSTEM, query)
    if result:
        logger.info(f"HyDE 生成: {query[:30]}... → 假设答案 {len(result)} 字")
    else:
        logger.warning("HyDE 假设答案生成失败")
    return result.strip() if result else ""
