"""Step-Back 回退策略：生成更抽象的上游问题，用其检索结果补充上下文"""
from core.strategies._llm import llm_complete
from utils.logger import logger

STEP_BACK_SYSTEM = """你是一个问题抽象助手。请针对用户的具体问题，生成一个更抽象、更通用的"回退问题"。

回退问题应该：
1. 比原问题高一个抽象层次（如从"勾股定理计算"回退到"直角三角形性质"）
2. 能够帮助检索到更广泛的背景知识
3. 保持与原问题相关的知识领域
4. 是一个独立的、可直接用于检索的问题

只输出回退问题本身，不要任何解释。"""


async def generate_step_back_query(query: str) -> str:
    """生成 step-back 回退问题，失败时返回空字符串"""
    result = await llm_complete(STEP_BACK_SYSTEM, query)
    if result:
        logger.info(f"Step-Back 回退: {query[:30]}... → {result[:30]}...")
    else:
        logger.warning("Step-Back 回退生成失败")
    return result.strip() if result else ""
