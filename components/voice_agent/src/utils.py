import asyncio
from typing import Any, AsyncIterator, TypeVar


T = TypeVar("T")


async def merge_async_iters(*aiters: AsyncIterator[T]) -> AsyncIterator[T]:
    """Merge multiple async iterators, yielding items as they become available."""
    queue: asyncio.Queue[Any] = asyncio.Queue()
    sentinel = object()

    async def producer(aiter: AsyncIterator[Any]) -> None:
        async for item in aiter:
            await queue.put(item)
        await queue.put(sentinel)

    tasks = [asyncio.create_task(producer(aiter)) for aiter in aiters]
    try:
        finished = 0
        while finished < len(aiters):
            item = await queue.get()
            if item is sentinel:
                finished += 1
            else:
                yield item
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
