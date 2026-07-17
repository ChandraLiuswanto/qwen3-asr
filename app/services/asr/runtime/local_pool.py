# -*- coding: utf-8 -*-
"""Small async pool for per-request ASR engines."""

from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass
class _PoolState(Generic[T]):
    queue: asyncio.Queue[T]


class LocalEnginePool(Generic[T]):
    """Fixed-size lazy engine pool backed by ``asyncio.Queue``."""

    def __init__(self, size: int, factory: Callable[[], T]):
        self._size = max(1, size)
        self._factory = factory
        self._state: Optional[_PoolState[T]] = None
        self._init_lock = threading.Lock()

    def _ensure_state(self) -> _PoolState[T]:
        if self._state is not None:
            return self._state

        with self._init_lock:
            if self._state is None:
                self._state = _PoolState(queue=asyncio.Queue(maxsize=self._size))
                for _ in range(self._size):
                    self._state.queue.put_nowait(self._factory())
        return self._state

    def warmup(self) -> None:
        self._ensure_state()

    async def acquire(self) -> T:
        state = self._ensure_state()
        return await state.queue.get()

    async def release(self, engine: T) -> None:
        state = self._ensure_state()
        await state.queue.put(engine)


class ThreadedEnginePool(Generic[T]):
    """Thread-safe twin of :class:`LocalEnginePool` backed by ``queue.Queue``.

    For engines used *synchronously from executor threads* (diarization runs
    inside ``run_sync``); the async pool above cannot serve that path. An
    instance checked out here is exclusively owned by the caller until
    ``release`` — the pool itself is the per-instance mutex.

    Invariant (see DIARIZATION_POOL_SIZE in app/core/config.py): a blocking
    ``acquire()`` holds its executor-thread slot while waiting, so the number
    of concurrent callers must not exceed the pool size.

    Release contract: ``release`` never constructs the pool and never blocks.
    Releasing before any acquire is caller misuse and raises ``RuntimeError``;
    releasing into an already-full pool raises ``queue.Full`` loudly instead
    of wedging the calling thread forever.
    """

    def __init__(self, size: int, factory: Callable[[], T]):
        self._size = max(1, size)
        self._factory = factory
        self._queue: Optional[queue.Queue[T]] = None
        self._init_lock = threading.Lock()

    @property
    def size(self) -> int:
        return self._size

    def _ensure_queue(self) -> "queue.Queue[T]":
        if self._queue is not None:
            return self._queue
        with self._init_lock:
            if self._queue is None:
                q: "queue.Queue[T]" = queue.Queue(maxsize=self._size)
                # Sequential construction, on purpose: factories may touch
                # global registries (modelscope) and download models.
                for _ in range(self._size):
                    q.put_nowait(self._factory())
                self._queue = q
        return self._queue

    def warmup(self) -> None:
        self._ensure_queue()

    def acquire(self) -> T:
        return self._ensure_queue().get()

    def release(self, engine: T) -> None:
        # Never construct the pool here: an engine can only exist via
        # acquire(), so a missing queue means caller misuse.
        q = self._queue
        if q is None:
            raise RuntimeError(
                "ThreadedEnginePool.release() called before any acquire()"
            )
        # put_nowait: an over-full pool is a bug; fail loudly, never block.
        q.put_nowait(engine)
