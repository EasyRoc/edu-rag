"""第一层意图识别：关键词匹配器（速度 < 1ms，命中则短路返回）"""

import time

# 按优先级排列：越靠前的意图优先级越高，任一关键词命中即返回
KEYWORD_INTENT_MAP: list[tuple[str, list[str]]] = [
    ("greeting", [
        "你好", "您好", "hi", "hello", "hey", "嗨",
        "早上好", "下午好", "晚上好", "早安", "晚安",
        "在吗", "在不在", "再见", "拜拜", "bye", "明天见",
    ]),
    ("command", [
        "/help", "/exit", "/start", "/model",
        "功能列表", "命令",
    ]),
    ("educational", [
        "老师", "学生", "同学", "家长",
        "上课", "考试", "作业", "习题", "题目", "答案",
        "公式", "定理", "定义", "概念",
        "数学", "语文", "英语", "物理", "化学", "生物", "历史", "地理", "政治",
        "作文", "阅读", "背诵", "默写", "分数", "成绩", "排名",
        "千克", "公里", "厘米", "方程", "函数", "三角形",
        "古诗", "文言文", "成语", "语法", "单词", "时态",
        "学习", "教程", "如何", "什么是",
        "是什么", "等于", "多少", "为什么", "推导", "证明",
        "怎么计算", "怎么做", "怎么解", "怎么求",
    ]),
    ("technical", [
        "bug", "报错", "安装失败", "部署",
        "API", "接口", "配置", "环境变量", "依赖",
        "异常", "崩溃", "超时",
    ]),
    ("chitchat", [
        "谢谢", "感谢", "thank", "不客气", "帮助",
        "辛苦了", "好的", "ok", "哈哈", "好玩",
        "天气", "今天天气", "吃饭", "吃了没",
        "高兴", "开心", "无聊", "没意思", "难受",
        "笑话", "心情", "怎么样",
        "你是谁", "你叫什么", "who are you",
        "你能做什么", "你会什么", "你擅长什么",
        "你多大了", "你几岁", "你喜欢什么",
        "测试", "test", "试一下", "试试",
    ]),
]


def match_keywords(query: str) -> dict | None:
    """
    第一层：关键词匹配。

    返回:
        {"intent": str, "confidence": 1.0} 或 None（未命中）
    """
    start = time.perf_counter()
    query_lower = query.strip().lower()

    for intent, keywords in KEYWORD_INTENT_MAP:
        for kw in keywords:
            if kw in query_lower:
                elapsed_ms = (time.perf_counter() - start) * 1000
                return {
                    "intent": intent,
                    "confidence": 1.0,
                    "source": "keyword",
                    "processing_time_ms": round(elapsed_ms, 2),
                }

    return None
