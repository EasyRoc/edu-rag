"""流式 token 队列注册表，供 graph 节点和 rag_service 共享"""

import asyncio

# key: queue_id, value: asyncio.Queue
_registry: dict[str, asyncio.Queue] = {}
