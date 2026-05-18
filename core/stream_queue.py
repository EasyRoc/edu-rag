"""流式 token 队列注册表，供 graph 节点和 rag_service 共享"""

import asyncio
import uuid


class StreamQueueRegistry:
    """Small runtime wrapper around streaming queues."""

    def __init__(self):
        self._queues: dict[str, asyncio.Queue] = {}

    def create(self) -> tuple[str, asyncio.Queue]:
        queue_id = str(uuid.uuid4())
        queue: asyncio.Queue = asyncio.Queue()
        self._queues[queue_id] = queue
        return queue_id, queue

    def get(self, queue_id: str | None) -> asyncio.Queue | None:
        if not queue_id:
            return None
        return self._queues.get(queue_id)

    def remove(self, queue_id: str | None) -> None:
        if queue_id:
            self._queues.pop(queue_id, None)


stream_queues = StreamQueueRegistry()

# Backward-compatible view for older graph code/tests.
_registry = stream_queues._queues
