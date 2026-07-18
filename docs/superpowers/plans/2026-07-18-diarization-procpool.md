# Diarization Process Pool (Change D) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move CAM++ speaker diarization from the in-process `ThreadedEnginePool` (GIL-bound, measured dead end) to spawn-based worker processes, one pipeline per worker, with loud boot warmup.

**Architecture:** A new worker-side module (`diarization_worker.py`, runs in spawned children) and a parent-side manager (`diarization_pool.py`, `DiarizationProcessPool`). `SpeakerDiarizer.diarize()` submits an audio path and gets back native-typed `(start_sec, end_sec, label)` triples. Boot warms all N workers via a `multiprocessing.Barrier(N+1)` rendezvous inside the initializer and fails the boot loudly on any warmup failure. Spec: `docs/superpowers/specs/2026-07-18-procpool-diarization-asyncllm-design.md` (Change D sections; Change C is a separate later plan).

**Tech Stack:** Python 3.11, `concurrent.futures.ProcessPoolExecutor` (spawn context), modelscope/funasr (worker side only), unittest.

## Global Constraints

- Branch: `feat/diarization-procpool` (created). NEVER commit to main; merge is gated on H100 gates G0+G1 (spec).
- Tests are **unittest, NOT pytest**: `DEVICE=cpu .venv/bin/python -m unittest discover -s tests` must be green after every task.
- Task tracking in **bd** (issue to be filed at execution start), not TodoWrite/markdown.
- `DIARIZATION_POOL_SIZE` keeps its name and default (4); its meaning becomes "worker process count". Production value is chosen at gate G1 from measured VRAM — do not change the default.
- No throughput numbers anywhere in code/comments/commits — measurement gates only (project rule).
- Worker processes must never import the vLLM/engine stack; `app/utils/diarization_worker.py` top level stays light (heavy imports inside functions).
- The "too short" fallback contract: modelscope raises `AssertionError('modelscope error: The effective audio duration is too short.')`; the parent-side string match in `SpeakerDiarizer.diarize` must keep working (exceptions pickle across `.result()` with message preserved).

---

### Task 1: Worker module — marshalling core

**Files:**
- Create: `app/utils/diarization_worker.py`
- Test: `tests/test_diarization_worker.py`

**Interfaces:**
- Produces: `diarization_worker._worker_diarize(audio_path: str) -> list[tuple[float, float, int]]` (native Python types only); module global `_pipeline`; `diarization_worker._FakeWorkerPipeline` (test/fake-mode pipeline, modelscope-shaped output with numpy scalars).
- Consumes: `app.utils.speaker_diarizer._suppress_empty_cache` (existing, imported lazily inside the function).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_diarization_worker.py
"""Worker-side marshalling: modelscope pipeline output (lists containing
numpy scalars) must cross the process boundary as native-typed triples.
Getting this wrong is SILENT (empty result -> per-request VAD fallback),
so types are asserted exactly, through a real pickle round trip."""

import pickle
import unittest

import numpy as np

from app.utils import diarization_worker as dw


class _FakePipeline:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, audio_path):
        self.calls.append(audio_path)
        return self.result


class WorkerDiarizeTest(unittest.TestCase):
    def tearDown(self):
        dw._pipeline = None

    def test_numpy_scalars_become_native_types(self):
        dw._pipeline = _FakePipeline(
            {"text": [[np.float64(0.0), np.float64(1.5), np.int64(0)],
                      [np.float64(1.5), np.float64(3.0), np.int64(1)]]}
        )
        triples = dw._worker_diarize("/tmp/x.wav")
        self.assertEqual(triples, [(0.0, 1.5, 0), (1.5, 3.0, 1)])
        for s, e, label in triples:
            self.assertIs(type(s), float)
            self.assertIs(type(e), float)
            self.assertIs(type(label), int)
        # The contract is "small picklable triples": prove it round-trips.
        self.assertEqual(pickle.loads(pickle.dumps(triples)), triples)

    def test_malformed_segments_are_skipped_not_fatal(self):
        dw._pipeline = _FakePipeline(
            {"text": [[0.0, 1.5, 0], ["bad", "seg"], [1.5, "x", 1], None]}
        )
        self.assertEqual(dw._worker_diarize("/tmp/x.wav"), [(0.0, 1.5, 0)])

    def test_non_dict_result_uses_text_attribute(self):
        class R:
            text = [[0.0, 2.0, 3]]
        dw._pipeline = _FakePipeline(R())
        self.assertEqual(dw._worker_diarize("/tmp/x.wav"), [(0.0, 2.0, 3)])

    def test_use_before_init_raises(self):
        with self.assertRaises(RuntimeError):
            dw._worker_diarize("/tmp/x.wav")

    def test_pipeline_exception_propagates_untouched(self):
        dw._pipeline = _FakePipeline(None)
        dw._pipeline.__class__.__call__ = lambda self, p: (_ for _ in ()).throw(
            AssertionError("modelscope error: The effective audio duration is too short.")
        )
        with self.assertRaises(AssertionError) as ctx:
            dw._worker_diarize("/tmp/x.wav")
        self.assertIn("too short", str(ctx.exception))

    def test_fake_worker_pipeline_shape(self):
        result = dw._FakeWorkerPipeline()("/tmp/x.wav")
        self.assertIn("text", result)
        self.assertEqual(len(result["text"]), 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_diarization_worker -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'app.utils.diarization_worker'`

- [ ] **Step 3: Implement the module**

```python
# app/utils/diarization_worker.py
# -*- coding: utf-8 -*-
"""Diarization worker-process entrypoints (Change D).

Runs INSIDE spawn-based worker processes owned by
app.utils.diarization_pool.DiarizationProcessPool. The module top level must
stay light: the parent imports it only to reference these functions by
qualified name for pickling, and every spawned child re-imports it. Heavy
imports (modelscope, torch via speaker_diarizer) happen inside functions,
in the child only. This module must NEVER import the vLLM/engine stack.
"""

from __future__ import annotations

import os
import sys
from typing import Any, List, Tuple

# One pipeline per worker process. The process IS the instance: a
# ProcessPoolExecutor worker runs one task at a time, so funasr's
# mutate-shared-state-per-call hazard cannot cross requests.
_pipeline: Any = None


class _FakeWorkerPipeline:
    """DIARIZATION_WORKER_FAKE=1 stand-in (tests only): modelscope-shaped
    output including numpy scalar types, so spawn-integration tests exercise
    the real marshalling and pickle path without loading models."""

    def __call__(self, audio_path: str):
        import numpy as np

        return {
            "text": [
                [np.float64(0.0), np.float64(1.5), np.int64(0)],
                [np.float64(1.5), np.float64(3.0), np.int64(1)],
            ]
        }


def _configure_worker_logging() -> None:
    """Sinks are not inherited across spawn. Route loguru to stderr so
    [diarization-profile] lines and init tracebacks are visible in the
    parent's captured stderr — gate G1 reads exactly these lines."""
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level=os.getenv("DIARIZATION_WORKER_LOG_LEVEL", "INFO"))


def _build_pipeline() -> Any:
    if os.getenv("DIARIZATION_WORKER_FAKE") == "1":
        return _FakeWorkerPipeline()
    # Force modelscope task registration BEFORE building — without this the
    # speaker-diarization task can be unregistered and pipeline() falls
    # through to transformers ("Unknown task speaker-diarization", the
    # qwen3-asr-9nk boot failure).
    import modelscope.pipelines.audio  # noqa: F401

    from app.utils.speaker_diarizer import _build_diarization_pipeline

    return _build_diarization_pipeline()


def _worker_init(barrier: Any = None, barrier_timeout_s: float = 300.0) -> None:
    """ProcessPoolExecutor initializer. Builds this worker's pipeline, then
    rendezvouses on the boot barrier (parties = N workers + 1 parent), so
    barrier release means every worker exists AND finished building. An
    exception here surfaces to the parent as BrokenProcessPool with NO cause
    text — the real traceback is on this process's stderr."""
    global _pipeline
    _configure_worker_logging()
    _pipeline = _build_pipeline()
    if barrier is not None:
        barrier.wait(timeout=barrier_timeout_s)


def _worker_probe() -> int:
    """Warmup no-op; returns the worker's PID for the distinct-N assertion."""
    return os.getpid()


def _worker_diarize(audio_path: str) -> List[Tuple[float, float, int]]:
    """Run this worker's pipeline; return native-typed triples.

    Conversion to native float/int happens HERE (worker side): the pipeline
    emits lists with numpy scalars, and the cross-process contract is small
    plain-Python triples. Malformed segments are skipped (same policy as the
    old parent-side parse). Pipeline exceptions propagate untouched so the
    parent's "too short" string match keeps working.
    """
    if _pipeline is None:
        raise RuntimeError("diarization worker used before _worker_init")

    from app.utils.speaker_diarizer import _suppress_empty_cache

    with _suppress_empty_cache():
        result = _pipeline(audio_path)

    if isinstance(result, dict):
        raw = result.get("text", [])
    else:
        raw = getattr(result, "text", []) or []

    triples: List[Tuple[float, float, int]] = []
    for seg in raw:
        if isinstance(seg, (list, tuple)) and len(seg) == 3:
            try:
                triples.append((float(seg[0]), float(seg[1]), int(seg[2])))
            except (TypeError, ValueError):
                continue
    return triples
```

- [ ] **Step 4: Run tests to verify pass**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_diarization_worker -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Run the full suite, then commit**

Run: `DEVICE=cpu .venv/bin/python -m unittest discover -s tests` → green.

```bash
git add app/utils/diarization_worker.py tests/test_diarization_worker.py
git commit -m "feat: diarization worker-process module — native-typed triple marshalling"
```

---

### Task 2: Parent-side DiarizationProcessPool

**Files:**
- Create: `app/utils/diarization_pool.py`
- Test: `tests/test_diarization_process_pool.py`

**Interfaces:**
- Consumes: `diarization_worker._worker_init`, `._worker_probe`, `._worker_diarize`, `._FakeWorkerPipeline` (Task 1).
- Produces: `DiarizationProcessPool(size: int, boot_timeout_s: float)` with `.size: int`, `.start() -> None` (blocking, raises on failure), `.diarize(audio_path: str) -> list[tuple[float, float, int]]`, `.close() -> None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_diarization_process_pool.py
"""Parent-side pool manager. Two layers:
- REAL spawn integration (DIARIZATION_WORKER_FAKE=1): start() must bring up
  N distinct worker processes and diarize() must return native triples.
- Fake-executor unit tests for the BrokenProcessPool rebuild-once contract.
"""

import os
import unittest
from concurrent.futures.process import BrokenProcessPool
from unittest import mock

from app.core.exceptions import DefaultServerErrorException
from app.utils.diarization_pool import DiarizationProcessPool


class SpawnIntegrationTest(unittest.TestCase):
    """Real processes, fake pipeline. Slow-ish (~seconds); still CPU-safe."""

    def setUp(self):
        self._env = mock.patch.dict(os.environ, {"DIARIZATION_WORKER_FAKE": "1"})
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_start_then_diarize_end_to_end(self):
        pool = DiarizationProcessPool(2, boot_timeout_s=120)
        pool.start()
        self.addCleanup(pool.close)
        triples = pool.diarize("/tmp/anything.wav")
        self.assertEqual(triples, [(0.0, 1.5, 0), (1.5, 3.0, 1)])
        for s, e, label in triples:
            self.assertIs(type(s), float)
            self.assertIs(type(label), int)

    def test_start_is_idempotent(self):
        pool = DiarizationProcessPool(1, boot_timeout_s=120)
        pool.start()
        self.addCleanup(pool.close)
        pool.start()  # second call must be a no-op, not a respawn


class PoolContractTest(unittest.TestCase):
    def test_diarize_before_start_raises_server_error(self):
        pool = DiarizationProcessPool(2, boot_timeout_s=5)
        with self.assertRaises(DefaultServerErrorException):
            pool.diarize("/tmp/x.wav")

    def test_broken_pool_rebuilds_once_and_fails_the_request(self):
        pool = DiarizationProcessPool(2, boot_timeout_s=5)
        broken_executor = mock.Mock()
        broken_executor.submit.side_effect = BrokenProcessPool("worker died")
        pool._executor = broken_executor
        with mock.patch.object(pool, "start") as restart:
            with self.assertRaises(DefaultServerErrorException):
                pool.diarize("/tmp/x.wav")
        restart.assert_called_once_with()
        broken_executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    def test_rebuild_skipped_if_another_thread_already_rebuilt(self):
        pool = DiarizationProcessPool(2, boot_timeout_s=5)
        broken_executor = mock.Mock()
        broken_executor.submit.side_effect = BrokenProcessPool("worker died")
        pool._executor = broken_executor
        replacement = mock.Mock()

        def swap(broken):
            # simulate a racing thread having already swapped the executor
            pool._executor = replacement
            return DiarizationProcessPool._rebuild(pool, broken)

        with mock.patch.object(pool, "_rebuild", side_effect=swap):
            with self.assertRaises(DefaultServerErrorException):
                pool.diarize("/tmp/x.wav")
        # the raced rebuild must NOT tear down the replacement
        replacement.shutdown.assert_not_called()

    def test_worker_exception_propagates_raw(self):
        pool = DiarizationProcessPool(2, boot_timeout_s=5)
        fake_future = mock.Mock()
        fake_future.result.side_effect = AssertionError(
            "modelscope error: The effective audio duration is too short."
        )
        fake_executor = mock.Mock()
        fake_executor.submit.return_value = fake_future
        pool._executor = fake_executor
        with self.assertRaises(AssertionError):
            pool.diarize("/tmp/x.wav")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_diarization_process_pool -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.utils.diarization_pool'`

- [ ] **Step 3: Implement the manager**

```python
# app/utils/diarization_pool.py
# -*- coding: utf-8 -*-
"""Parent-side manager for diarization worker processes (Change D).

The worker process is the instance: exclusivity is structural (a
ProcessPoolExecutor worker runs one task at a time). This replaces the
ThreadedEnginePool checkout mutex, which could not scale past the GIL.
"""

from __future__ import annotations

import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor
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

    @property
    def size(self) -> int:
        return self._size

    def start(self) -> None:
        """Spawn all N workers and block until every pipeline is built.

        Structural fan-out: N probe submits each force a spawn, because every
        already-spawned worker is parked on the barrier inside _worker_init
        (never idle, so ProcessPoolExecutor cannot reuse it). Barrier parties
        = N + 1 (the parent); release therefore means all N pipelines exist.
        The distinct-PID check is belt and braces on that mechanism.
        Raises on timeout/failure — the caller (boot) must NOT swallow this.
        """
        with self._lock:
            if self._executor is not None:
                return
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
                barrier.wait(timeout=self._boot_timeout_s)
                pids = {p.result(timeout=self._boot_timeout_s) for p in probes}
                if len(pids) != self._size:
                    raise RuntimeError(
                        "diarization pool warmup: expected "
                        f"{self._size} distinct workers, got {len(pids)}"
                    )
            except Exception:
                barrier.abort()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            self._executor = executor
            logger.info("Diarization process pool ready: {} workers", self._size)

    def diarize(self, audio_path: str) -> List[Tuple[float, float, int]]:
        """Submit one diarization to a worker and block for the result.

        Worker exceptions re-raise here with their original type and message
        (pickled), so SpeakerDiarizer's "too short" string match still works.
        BrokenProcessPool triggers ONE rebuild for future requests; the
        request that hit the break fails — no silent retry.
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
                "Diarization worker pool broke ({}); attempting one rebuild. "
                "Worker traceback is on worker stderr.",
                e,
            )
            self._rebuild(executor)
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
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_diarization_process_pool -v`
Expected: all tests PASS (spawn tests take a few seconds).

- [ ] **Step 5: Full suite, then commit**

Run: `DEVICE=cpu .venv/bin/python -m unittest discover -s tests` → green.

```bash
git add app/utils/diarization_pool.py tests/test_diarization_process_pool.py
git commit -m "feat: DiarizationProcessPool — barrier warmup, rebuild-once, spawn integration tests"
```

---

### Task 3: Rewire SpeakerDiarizer onto the process pool

**Files:**
- Modify: `app/utils/speaker_diarizer.py` (imports at :22, pool global at :368-376, `diarize()` body at :395-451; keep the "too short" except-path at :453-477 verbatim)
- Modify: `app/core/config.py` (add `DIARIZATION_BOOT_TIMEOUT_S`, update the `DIARIZATION_POOL_SIZE` comment block at :94-101)
- Test: rewrite `tests/test_diarization_pool_wiring.py`

**Interfaces:**
- Consumes: `DiarizationProcessPool` (Task 2).
- Produces: module global `speaker_diarizer._diarization_pool: DiarizationProcessPool`; `warmup_diarization_pool() -> int` (unchanged signature, now starts processes); `SpeakerDiarizer.diarize()` (unchanged signature/return type `List[SpeakerSegment]`); `settings.DIARIZATION_BOOT_TIMEOUT_S: int` (default 300).

- [ ] **Step 1: Rewrite the wiring tests (failing first)**

Replace the whole of `tests/test_diarization_pool_wiring.py` with:

```python
# tests/test_diarization_pool_wiring.py
"""Structure-only tests: SpeakerDiarizer.diarize() delegates to the
process-pool manager and maps native triples onto SpeakerSegment. The
empty_cache guard and per-instance exclusivity now live INSIDE workers
(tests/test_diarization_worker.py, tests/test_diarization_process_pool.py);
scripts/h100/test_offline_mixing.py stays the GPU-concurrency gate."""

import os
import subprocess
import sys
import unittest
from unittest import mock

from app.utils import speaker_diarizer as sd
from app.utils.diarization_pool import DiarizationProcessPool


class DiarizationWiringTest(unittest.TestCase):
    def test_diarize_maps_triples_to_segments(self):
        with mock.patch.object(
            sd._diarization_pool, "diarize", return_value=[(0.0, 1.5, 0), (1.5, 3.0, 1)]
        ) as d:
            segments = sd.SpeakerDiarizer().diarize("/tmp/x.wav")
        d.assert_called_once_with("/tmp/x.wav")
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].start_ms, 0)
        self.assertEqual(segments[0].end_ms, 1500)
        self.assertEqual(segments[0].speaker_id, "说话人1")
        self.assertEqual(segments[1].speaker_id, "说话人2")

    def test_too_short_fallback_survives_the_process_boundary(self):
        err = AssertionError(
            "modelscope error: The effective audio duration is too short."
        )
        with mock.patch.object(sd._diarization_pool, "diarize", side_effect=err), \
             mock.patch.object(sd.librosa, "get_duration", return_value=5.0):
            segments = sd.SpeakerDiarizer().diarize("/tmp/x.wav")
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].speaker_id, "说话人1")
        self.assertEqual(segments[0].end_ms, 5000)

    def test_other_worker_errors_become_server_errors(self):
        with mock.patch.object(
            sd._diarization_pool, "diarize", side_effect=RuntimeError("cuda meltdown")
        ):
            with self.assertRaises(Exception) as ctx:
                sd.SpeakerDiarizer().diarize("/tmp/x.wav")
        self.assertIn("说话人分离失败", str(ctx.exception))

    def test_pool_is_process_pool_sized_from_settings(self):
        self.assertIsInstance(sd._diarization_pool, DiarizationProcessPool)
        # Fresh interpreter: the module-level pool reads settings at import.
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from app.utils.speaker_diarizer import _diarization_pool; "
                "import sys; "
                "sys.exit(0 if _diarization_pool.size == 2 else 1)",
            ],
            env={**os.environ, "DIARIZATION_POOL_SIZE": "2", "DEVICE": "cpu"},
            cwd=repo_root,
        )
        self.assertEqual(result.returncode, 0)

    def test_warmup_starts_pool_and_returns_size(self):
        with mock.patch.object(sd._diarization_pool, "start") as start:
            n = sd.warmup_diarization_pool()
        start.assert_called_once_with()
        self.assertEqual(n, sd._diarization_pool.size)

    def test_threaded_engine_pool_is_gone_from_this_module(self):
        self.assertFalse(hasattr(sd, "ThreadedEnginePool"))

    def test_boot_timeout_setting_exists(self):
        from app.core.config import settings
        self.assertGreaterEqual(settings.DIARIZATION_BOOT_TIMEOUT_S, 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_diarization_pool_wiring -v`
Expected: FAILures (pool is still `ThreadedEnginePool`, no `DIARIZATION_BOOT_TIMEOUT_S`, `diarize` still does checkout).

- [ ] **Step 3: Modify `app/core/config.py`**

In the class body, replace the `DIARIZATION_POOL_SIZE` comment block (keep the field and default) and add the timeout field:

```python
    # CAM++ 说话人分离 worker 进程数（change D：每个 worker 进程独立持有
    # 一份 pipeline，进程即实例 —— GIL 不再是并行上限）。生产取值必须以
    # H100 上 nvidia-smi 实测的单 worker 显存增量为准（gate G1，见
    # docs/superpowers/specs/2026-07-18-procpool-diarization-asyncllm-design.md），
    # 不要凭空调大。
    # 不变量：保持 VLLM_OFFLINE_CONCURRENCY <= DIARIZATION_POOL_SIZE，
    # 否则超额请求在进程池队列中排队（不再占死 executor 线程，但仍是队头阻塞）。
    DIARIZATION_POOL_SIZE: int = 4
    # 进程池 boot 预热超时（秒）：全部 worker 建完 pipeline 的 Barrier 等待
    # 上限。冷缓存首次下载模型时最慢 —— 超时会导致 boot 失败（响亮，不降级）。
    DIARIZATION_BOOT_TIMEOUT_S: int = 300
```

In `_load_from_env`, next to the existing `DIARIZATION_POOL_SIZE` load, add:

```python
        self.DIARIZATION_BOOT_TIMEOUT_S = int(
            os.getenv("DIARIZATION_BOOT_TIMEOUT_S", str(self.DIARIZATION_BOOT_TIMEOUT_S))
        )
```

- [ ] **Step 4: Modify `app/utils/speaker_diarizer.py`**

(a) Delete line 22 (`from ..services.asr.runtime.local_pool import ThreadedEnginePool`) and add instead:

```python
from .diarization_pool import DiarizationProcessPool
```

(b) Replace the pool global and warmup (current lines 368-376) with:

```python
_diarization_pool: DiarizationProcessPool = DiarizationProcessPool(
    settings.DIARIZATION_POOL_SIZE, settings.DIARIZATION_BOOT_TIMEOUT_S
)


def warmup_diarization_pool() -> int:
    """Spawn and warm ALL worker processes (startup). Raises on failure —
    the boot path must let that propagate (fail loud, never degrade)."""
    _diarization_pool.start()
    return _diarization_pool.size
```

(c) Replace the body of `SpeakerDiarizer.diarize` from `try:` down to the end of the segment-parsing loop (current lines 406-451) with the following — the `except` block below it (the "too short" fallback, lines 453-477) stays byte-identical:

```python
        try:
            logger.info(f"开始说话人分离: {audio_path}")
            # The worker process is the instance: exclusivity is structural
            # (one task per worker), so there is no checkout/release here.
            # Bind the pool once: a module reload rebinds the global.
            pool = _diarization_pool
            triples = pool.diarize(audio_path)

            segments = [
                SpeakerSegment(
                    start_ms=int(start_sec * 1000),
                    end_ms=int(end_sec * 1000),
                    speaker_id=f"说话人{label + 1}",
                )
                for (start_sec, end_sec, label) in triples
            ]

            logger.info(f"说话人分离完成，原始片段数: {len(segments)}")
            for i, seg in enumerate(segments[:20]):
                logger.debug(
                    f"[CAM++原始] #{i}: {seg.start_sec:.2f}-{seg.end_sec:.2f}s "
                    f"({seg.duration_sec:.2f}s) {seg.speaker_id}"
                )
            return segments
```

(d) The functions `_build_diarization_pipeline`, `_create_modelscope_pipeline`, `_enable_batched_sv`, `_enable_stage_timing`, `_install_empty_cache_guard`, `_suppress_empty_cache` all STAY in this module — they now execute only inside worker processes (imported lazily by `diarization_worker`). Update `_build_diarization_pipeline`'s docstring first line to: `"""Build ONE independent CAM++ pipeline instance — runs INSIDE a diarization worker process (see app/utils/diarization_worker.py)."""`

- [ ] **Step 5: Run tests to verify pass**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_diarization_pool_wiring -v`
Expected: all PASS.

- [ ] **Step 6: Full suite; fix any test that still fakes the old checkout**

Run: `DEVICE=cpu .venv/bin/python -m unittest discover -s tests`
Expected: green, EXCEPT tests that patch the old `acquire`/`release` API — if any fail, they are asserting the retired contract; update them to patch `sd._diarization_pool.diarize` (pattern in Step 1). Do not weaken unrelated assertions.

- [ ] **Step 7: Commit**

```bash
git add app/utils/speaker_diarizer.py app/core/config.py tests/test_diarization_pool_wiring.py
git commit -m "feat: SpeakerDiarizer delegates to the diarization process pool"
```

---

### Task 4: Loud boot warmup in model_loader

**Files:**
- Modify: `app/utils/model_loader.py` (step 5 block, lines 533-550)
- Test: `tests/test_model_loader_diarization_boot.py` (create)

**Interfaces:**
- Consumes: `warmup_diarization_pool` (Task 3).
- Produces: `model_loader._preload_diarization_pool(result: dict, progress) -> None` — raises on warmup failure (boot dies loudly).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model_loader_diarization_boot.py
"""Boot MUST fail loudly when the diarization worker pool cannot warm up —
this replaces the old swallow-and-degrade (log + continue) that produced the
silent lazy-build stall (bd qwen3-asr-9nk)."""

import unittest
from unittest import mock

from app.utils import model_loader


def _result_dict():
    return {"speaker_diarization_model": {"loaded": False, "error": None}}


class DiarizationBootTest(unittest.TestCase):
    def test_success_marks_loaded(self):
        result = _result_dict()
        progress = mock.Mock()
        with mock.patch(
            "app.utils.speaker_diarizer.warmup_diarization_pool", return_value=4
        ):
            model_loader._preload_diarization_pool(result, progress)
        self.assertTrue(result["speaker_diarization_model"]["loaded"])

    def test_failure_records_error_and_reraises(self):
        result = _result_dict()
        progress = mock.Mock()
        boom = RuntimeError("BrokenProcessPool: see worker stderr")
        with mock.patch(
            "app.utils.speaker_diarizer.warmup_diarization_pool", side_effect=boom
        ):
            with self.assertRaises(RuntimeError):
                model_loader._preload_diarization_pool(result, progress)
        self.assertIn(
            "BrokenProcessPool", result["speaker_diarization_model"]["error"]
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_model_loader_diarization_boot -v`
Expected: FAIL — `_preload_diarization_pool` does not exist.

- [ ] **Step 3: Implement — extract step 5 into a helper and make it raise**

In `app/utils/model_loader.py`, replace the step-5 block (lines 533-550) with a call:

```python
        # 5. Preload the required speaker diarization worker pool (CAM++).
        _preload_diarization_pool(result, progress)
```

and add the module-level helper (near the other helpers):

```python
def _preload_diarization_pool(result: dict, progress) -> None:
    """Warm ALL diarization worker processes. FAIL THE BOOT on any error:
    a swallowed failure here previously meant the first request paid a lazy
    N-process build (or worse, silent VAD-only degradation). An initializer
    crash surfaces as BrokenProcessPool with no cause text — the real
    traceback is on the worker's stderr; the log line says so.
    """
    progress.update("加载说话人分离模型(CAM++ worker 进程池)")
    from ..utils.speaker_diarizer import warmup_diarization_pool

    try:
        pool_size = warmup_diarization_pool()
        result["speaker_diarization_model"]["loaded"] = True
        logger.info("说话人分离进程池预热完成: %d workers", pool_size)
    except Exception as e:
        result["speaker_diarization_model"]["error"] = str(e)
        logger.critical(
            "说话人分离进程池预热失败，boot 中止（worker 真实 traceback 在其 stderr）: %s",
            e,
        )
        raise
    finally:
        progress.advance("已完成说话人分离模型(CAM++)")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_model_loader_diarization_boot -v` → PASS.

- [ ] **Step 5: Full suite, commit**

Run: `DEVICE=cpu .venv/bin/python -m unittest discover -s tests` → green.

```bash
git add app/utils/model_loader.py tests/test_model_loader_diarization_boot.py
git commit -m "feat: boot fails loudly when diarization worker pool warmup fails"
```

---

### Task 5: Retire ThreadedEnginePool; docs and env

**Files:**
- Modify: `app/services/asr/runtime/local_pool.py` (delete the `ThreadedEnginePool` class, lines 52-108; keep `LocalEnginePool`)
- Delete: `tests/test_threaded_engine_pool.py`
- Modify: `.env.example` (DIARIZATION_POOL_SIZE comment + new DIARIZATION_BOOT_TIMEOUT_S + DIARIZATION_WORKER_LOG_LEVEL)

**Interfaces:**
- Consumes: nothing. Produces: nothing (pure removal + docs). Pre-verified: the only importers of `ThreadedEnginePool` were `speaker_diarizer.py` (removed in Task 3) and its own test file.

- [ ] **Step 1: Verify no remaining importers**

Run: `grep -rn "ThreadedEnginePool" app tests --include=*.py | grep -v local_pool.py | grep -v test_threaded_engine_pool.py`
Expected: no output. If anything appears, STOP — do not delete; report the importer.

- [ ] **Step 2: Delete the class and its tests**

Remove lines 52-108 of `app/services/asr/runtime/local_pool.py` (the whole `ThreadedEnginePool` class) and `git rm tests/test_threaded_engine_pool.py`.

- [ ] **Step 3: Update `.env.example`**

Replace the `DIARIZATION_POOL_SIZE=16` block with:

```bash
# CAM++ diarization worker PROCESSES (change D). Each worker holds one
# pipeline; the process is the instance, so the GIL no longer caps
# parallelism. Production value MUST come from gate G1's measured
# per-worker VRAM (spec 2026-07-18) — do not guess upward. Default 4.
DIARIZATION_POOL_SIZE=4

# Boot warmup barrier timeout (seconds) for spawning + building all
# workers. Cold model caches are the slow case. Timeout fails the boot.
DIARIZATION_BOOT_TIMEOUT_S=300

# Log level inside diarization worker processes (their loguru sink goes to
# stderr; DIARIZATION_STAGE_TIMINGS lines appear there).
DIARIZATION_WORKER_LOG_LEVEL=INFO
```

- [ ] **Step 4: Full suite, commit**

Run: `DEVICE=cpu .venv/bin/python -m unittest discover -s tests` → green.

```bash
git add -A
git commit -m "chore: retire ThreadedEnginePool; document process-pool env knobs"
```

---

### Task 6: Final review pass and bd bookkeeping

**Files:** none new.

- [ ] **Step 1: Whole-branch self-check**

Run all of:
- `DEVICE=cpu .venv/bin/python -m unittest discover -s tests` → green, count recorded.
- `grep -rn "ThreadedEnginePool\|_diarization_inference_semaphore" app` → no output.
- `.venv/bin/python -c "import app.utils.diarization_worker"` → imports clean and fast (top level must not pull torch/modelscope — verify by adding `-X importtime` and checking no modelscope entry).

- [ ] **Step 2: Request code review**

Per house process: dispatch the final whole-branch review (superpowers:requesting-code-review) before declaring the local scope done.

- [ ] **Step 3: bd bookkeeping + commit**

```bash
bd create --title="Change D: diarization process pool (plan 2026-07-18)" \
  --description="Implements spec docs/superpowers/specs/2026-07-18-procpool-diarization-asyncllm-design.md change D via plan docs/superpowers/plans/2026-07-18-diarization-procpool.md. Local tasks 1-6 done on feat/diarization-procpool. H100 gates G0+G1 BLOCK the merge." \
  --type=feature --priority=1
git add .beads/ && git commit -m "chore: beads export (change D local scope)"
```

---

### Task 7: H100 gates G0 + G1 (OPERATOR TASK — blocks merge)

Not executable on the dev box. On the H100, from this branch:

- [ ] **G0 baseline (current main):** restart server from main with `DIARIZATION_POOL_SIZE=4 VLLM_OFFLINE_CONCURRENCY=4`, run `scripts/h100/bench.sh` levels [1,2,4,8,10,16]; record wall/req/s/p50/p95.
- [ ] **G1 (this branch):** deploy branch, same knobs, `DIARIZATION_STAGE_TIMINGS=true`. Verify boot log shows the pool-ready line with 4 workers (and that a deliberately broken config — e.g. `DIARIZATION_BOOT_TIMEOUT_S=1` — fails the boot loudly, then restore). Run the same bench plus `scripts/h100/test_offline_mixing.py`. Pass criteria (spec): per-call `preprocess` at n=10 ≈ its n=1 time (GIL-escape proof, read from worker stderr `[diarization-profile]` lines); zero failures; per-worker VRAM recorded via `nvidia-smi` deltas across pool start → choose production `DIARIZATION_POOL_SIZE` from it.
- [ ] Record both verdicts in the bd issue; merge only after both recorded.

---

## Self-Review (done at write time)

- **Spec coverage:** worker module incl. logging/registration/fake-mode (Task 1-2), barrier warmup + distinct-PID + timeout (Task 2), marshalling contract + parent parse replacement (Tasks 1, 3), rebuild-once (Task 2), loud boot (Task 4), knob reuse + comments + invariant text (Tasks 3, 5), ThreadedEnginePool retirement (Task 5), import discipline (Task 1 design + Task 6 importtime check), gates (Task 7). Change C intentionally excluded — separate plan after its Task-0 verifications.
- **Known judgment calls:** builders stay in `speaker_diarizer.py` (spec allowed either move-or-delete-import; deleting the `local_pool` import satisfies the discipline with less churn). `_suppress_empty_cache` wraps the pipeline call inside the worker (same window as before).
- **Type consistency:** `diarize(audio_path) -> List[Tuple[float, float, int]]` used identically in Tasks 1, 2, 3; `start()/close()` names consistent; `DIARIZATION_BOOT_TIMEOUT_S` defined (Task 3) before first use in the same task's pool construction.
