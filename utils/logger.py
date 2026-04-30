"""日志工具模块：统一日志输出格式"""

import logging
import sys
from config import settings


def setup_logger(name: str = "k12_rag") -> logging.Logger:
    """创建并返回一个配置好的日志器"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # 如果已经配置过 handler，不再重复添加
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# 全局默认日志器
logger = setup_logger()
