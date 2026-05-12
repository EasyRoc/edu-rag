"""Adaptive RAG 查询分类节点：意图识别 + 复杂度分级路由

1. 先判断查询意图（教育相关 / 闲聊）
2. 再对教育类查询进行 simple / medium / complex 三级复杂度分类
"""
from typing import Literal

from utils.logger import logger

# ==================== 闲聊/非教育类意图关键词 ====================
_CHITCHAT_KEYWORDS = [
    # 问候
    "你好", "您好", "hi", "hello", "hey", "嗨",
    "早上好", "下午好", "晚上好", "早安", "晚安",
    "在吗", "在不在",
    # 告别
    "再见", "拜拜", "bye", "明天见",
    # 寒暄
    "谢谢", "感谢", "thank", "不客气",
    "辛苦了", "好的", "ok",
    # 聊天
    "天气", "今天天气", "吃饭", "吃了没",
    "高兴", "开心", "哈哈", "好玩",
    "无聊", "没意思", "难受",
    # 关于助手自身
    "你是谁", "你叫什么", "who are you",
    "你能做什么", "你会什么", "你擅长什么",
    "你多大了", "你几岁",
    "你喜欢什么",
    # 无意义/测试
    "测试", "test", "试一下", "试试",
    "111", "222", "aaa",
]

# 教育类强信号关键词——如果命中则优先判定为教育意图
_EDUCATIONAL_KEYWORDS = [
    "老师", "学生", "同学", "家长",
    "上课", "考试", "作业", "习题", "题目", "答案",
    "公式", "定理", "定义", "概念",
    "数学", "语文", "英语", "物理", "化学", "生物", "历史", "地理", "政治",
    "作文", "阅读", "背诵", "默写",
    "分数", "成绩", "排名",
    "千克", "公里", "厘米", "方程", "函数", "三角形",
    "古诗", "文言文", "成语",
    "语法", "单词", "时态",
]

# 简单查询关键词：以事实性、定义类问题为主
_SIMPLE_KEYWORDS = [
    "是什么", "什么是", "定义", "公式", "定理", "等于", "多少",
    "谁", "哪一年", "在哪里", "什么时候",
]

# 复杂查询关键词：需要比较、分析、综合
_COMPLEX_KEYWORDS = [
    "比较", "对比", "区别", "异同", "关系", "分析", "为什么",
    "如何影响", "原理", "推导", "证明", "总结",
]


def classify_intent(query: str) -> Literal["educational", "chitchat"]:
    """
    识别用户查询意图：教育相关还是闲聊。

    规则（按优先级）:
    1. 命中教育强信号关键词 → educational
    2. 命中闲聊关键词且未命中教育关键词 → chitchat
    3. 同时命中两者或均未命中 → 根据查询长度和内容特征判断
    """
    query_lower = query.strip().lower()
    query_len = len(query_lower)

    has_educational = any(kw in query_lower for kw in _EDUCATIONAL_KEYWORDS)
    has_chitchat = any(kw in query_lower for kw in _CHITCHAT_KEYWORDS)

    logger.debug(
        f"意图识别: query='{query[:50]}', "
        f"educational={has_educational}, chitchat={has_chitchat}"
    )

    # 优先级1：命中教育强信号 → 教育类
    if has_educational:
        logger.info(f"意图识别结果: educational (命中教育关键词)")
        return "educational"

    # 优先级2：命中闲聊关键词 → 闲聊
    if has_chitchat:
        logger.info(f"意图识别结果: chitchat (命中闲聊关键词)")
        return "chitchat"

    # 优先级3：短查询（2字以内）无任何关键词 → 视为闲聊
    if query_len <= 2:
        logger.info(f"意图识别结果: chitchat (过短查询，无法识别为教育意图)")
        return "chitchat"

    # 优先级4：较长查询但仍无明确关键词 → 默认教育类（宁放过不误杀）
    logger.info(f"意图识别结果: educational (无明确闲聊信号，默认教育类)")
    return "educational"


def classify_query(query: str) -> Literal["simple", "medium", "complex"]:
    """
    基于规则对教育类查询进行复杂度分类。

    规则：
    - simple: 事实性、定义类查询，长度短且包含简单关键词
    - complex: 需要比较、分析、推理的查询
    - medium: 介于两者之间
    """

    query_lower = query.strip().lower()
    query_len = len(query_lower)
    logger.info(f"查询复杂度分类: query='{query[:50]}', 长度={query_len}")

    # 复杂查询检测
    has_complex = any(kw in query_lower for kw in _COMPLEX_KEYWORDS)
    if has_complex and query_len > 15:
        logger.info(f"复杂度结果: complex (包含分析/比较关键词)")
        return "complex"

    # 简单查询检测
    has_simple = any(kw in query_lower for kw in _SIMPLE_KEYWORDS)
    if has_simple or query_len < 10:
        logger.info(f"复杂度结果: simple (简单定义/事实查询)")
        return "simple"

    # 默认中等复杂度
    logger.info(f"复杂度结果: medium")
    return "medium"


