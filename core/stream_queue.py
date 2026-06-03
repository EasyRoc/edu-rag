"""流式 token 队列注册表，供 graph 节点和 rag_service 共享"""

import asyncio
import uuid


class StreamQueueRegistry:
    """Small runtime wrapper around streaming queues."""

    def __init__(self):
        self._queues: dict[str, asyncio.Queue] = {}
        self._closed: set[str] = set()

    def create(self) -> tuple[str, asyncio.Queue]:
        queue_id = str(uuid.uuid4())
        queue: asyncio.Queue = asyncio.Queue()
        self._queues[queue_id] = queue
        self._closed.discard(queue_id)
        return queue_id, queue

    def get(self, queue_id: str | None) -> asyncio.Queue | None:
        if not queue_id:
            return None
        return self._queues.get(queue_id)

    async def emit(self, queue_id: str | None, token: str) -> None:
        queue = self.get(queue_id)
        if queue is not None and queue_id not in self._closed:
            await queue.put(token)

    async def close(self, queue_id: str | None) -> None:
        queue = self.get(queue_id)
        if queue is not None and queue_id not in self._closed:
            self._closed.add(queue_id)
            await queue.put(None)

    def remove(self, queue_id: str | None) -> None:
        if queue_id:
            self._queues.pop(queue_id, None)
            self._closed.discard(queue_id)


stream_queues = StreamQueueRegistry()
