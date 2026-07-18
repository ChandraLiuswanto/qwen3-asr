# -*- coding: utf-8 -*-
"""Parent-side manager for diarization worker processes (Change D).

The worker process is the instance: exclusivity is structural (a
ProcessPoolExecutor worker runs one task at a time). This replaces the
in-process thread-pool checkout mutex, which could not scale past the GIL.
"""

from __future__ import annotations

import multiprocessing
import threading
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor
from concurrent.futures import wait as futures_wait
from concurrent.futures.process import BrokenProcessPool
from typing import List, Optional, Tuple

from loguru import logger

from ..core.exceptions import DefaultServerErrorException
from . import diarization_worker


class DiarizationProcessPool:
    def __init__(self, size: int, boot_timeout_s: float):
        self._size = max(1, size)
        self._boot_timeout_s = boot_timeout_s
        self._executor: Optional[ProcessPoolExecutor] = None
        self._lock = threading.Lock()
        self._starting = False
        # Set once by close(); makes the pool un-resurrectable and lets a
        # concurrent start() (boot or rebuild) detect a close that landed
        # mid-build and tear down its own fresh executor instead of orphaning
        # a fresh batch of CUDA worker processes past shutdown.
        self._closing = False
        # Handle to the most recent off-thread rebuild worker. Exposed so a
        # request that hit BrokenProcessPool does not block on the rebuild;
        # tests join it to synchronize deterministically.
        self._rebuild_thread: Optional[threading.Thread] = None

    @property
    def size(self) -> int:
        return self._size

    def start(self) -> None:
        """Spawn all N workers and block until every pipeline is built.

        Structural fan-out: N probe submits each force a spawn, because every
        already-spawned worker is parked on the barrier inside _worker_init
        (never idle, so ProcessPoolExecutor cannot reuse it). Barrier parties
        = N + 1 (the parent); release therefore means all N pipelines exist —
        the barrier alone is the proof (no PID counting: probes have no
        task-to-worker affinity and a PID check is scheduling-flaky).

        FAIL FAST on worker death: a worker whose initializer raises never
        reaches the barrier, and its probe future fails with
        BrokenProcessPool almost immediately — a watcher thread aborts the
        barrier on FIRST_EXCEPTION so the parent does not sit out the full
        boot timeout. The timeout covers only genuinely wedged spawns.

        The blocking warmup runs OUTSIDE self._lock (a rebuild must not make
        close() wait minutes for the lock); the lock only guards publishing.
        Raises on failure — the caller (boot) must NOT swallow this.

        Publish is atomic vs close(): the _closing check and the publish share
        ONE lock acquisition, so either start() publishes before close() flips
        the flag (close() then tears the published pool down) or close() flips
        first (start() sees it under the lock and tears down its OWN fresh
        executor) — never an orphan.
        """
        with self._lock:
            if self._executor is not None or self._starting or self._closing:
                return
            self._starting = True
        try:
            ctx = multiprocessing.get_context("spawn")
            barrier = ctx.Barrier(self._size + 1)
            executor = ProcessPoolExecutor(
                max_workers=self._size,
                mp_context=ctx,
                initializer=diarization_worker._worker_init,
                initargs=(barrier, self._boot_timeout_s),
            )
            try:
                probes = [
                    executor.submit(diarization_worker._worker_probe)
                    for _ in range(self._size)
                ]

                def _abort_on_worker_death() -> None:
                    done_or_failed = futures_wait(
                        probes, return_when=FIRST_EXCEPTION
                    )
                    if any(f.exception() is not None for f in done_or_failed.done):
                        barrier.abort()

                watcher = threading.Thread(
                    target=_abort_on_worker_death, daemon=True
                )
                watcher.start()
                barrier.wait(timeout=self._boot_timeout_s)
                for p in probes:
                    p.result(timeout=self._boot_timeout_s)
            except Exception:
                barrier.abort()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            with self._lock:
                if self._closing:
                    # close() landed while we were building. Tear down the
                    # fresh, un-published executor instead of orphaning its
                    # workers; close() saw _executor=None so it is our job.
                    executor.shutdown(wait=False, cancel_futures=True)
                    return
                self._executor = executor
            logger.info("Diarization process pool ready: {} workers", self._size)
        finally:
            with self._lock:
                self._starting = False

    def diarize(self, audio_path: str) -> List[Tuple[float, float, int]]:
        """Submit one diarization to a worker and block for the result.

        Worker exceptions re-raise here with their original type and message
        (pickled), so SpeakerDiarizer's "too short" string match still works.
        BrokenProcessPool triggers ONE rebuild for future requests; the
        request that hit the break FAILS FAST — it raises immediately and the
        rebuild runs on a background daemon thread, so this request never
        blocks up to DIARIZATION_BOOT_TIMEOUT_S (nor holds its admission slot)
        while N workers respawn and reload models. No silent retry.
        """
        executor = self._executor
        if executor is None:
            raise DefaultServerErrorException(
                "说话人分离进程池未启动（boot 预热缺失）"
            )
        try:
            return executor.submit(
                diarization_worker._worker_diarize, audio_path
            ).result()
        except BrokenProcessPool as e:
            logger.critical(
                "Diarization worker pool broke ({}); rebuilding off-thread. "
                "Worker traceback is on worker stderr.",
                e,
            )
            # Rebuild off the request thread. _rebuild's compare-and-swap
            # keeps rebuild-once: even if several broken requests each launch
            # a thread, only the one that finds _executor still == this broken
            # executor rebuilds; the rest are cheap no-ops. daemon=True so a
            # rebuild in flight can never block process exit.
            thread = threading.Thread(
                target=self._rebuild, args=(executor,), daemon=True
            )
            self._rebuild_thread = thread
            thread.start()
            raise DefaultServerErrorException(
                "说话人分离进程池崩溃，本请求失败（池已尝试重建）"
            ) from e

    def _rebuild(self, broken: ProcessPoolExecutor) -> None:
        """Tear down `broken` and start a fresh pool, once. Racing callers
        that lost the swap do nothing (their `broken` is already replaced)."""
        with self._lock:
            if self._executor is not broken:
                return
            self._executor = None
        broken.shutdown(wait=False, cancel_futures=True)
        try:
            self.start()
        except Exception as rebuild_error:
            logger.critical(
                "Diarization pool rebuild FAILED — pool stays down: {}",
                rebuild_error,
            )

    def close(self) -> None:
        # Flip _closing and swap the executor out under ONE lock acquisition.
        # This is the other half of start()'s atomic publish: if a rebuild is
        # mid-build, either we see and tear down its published executor here,
        # or it sees _closing under the lock at publish and tears down its own
        # fresh executor — never an orphaned CUDA-worker batch past shutdown.
        with self._lock:
            self._closing = True
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
