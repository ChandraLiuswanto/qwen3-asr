# Diarization Throughput — Implementation Plan

Date: 2026-07-17
Status: approved — Tasks 1-6 implemented on feat/diarization-throughput; Tasks 7-8 (H100) pending

**Design spec (the what/why this plan implements):** [`docs/superpowers/specs/2026-07-17-diarization-throughput-design.md`](../specs/2026-07-17-diarization-throughput-design.md)

> Section references below of the form `spec §"..."` point into that file. This plan deliberately does not restate the spec's problem statement, measurements, or the thread-safety analysis — read the spec first.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Change 1 (thread-local `torch.cuda.empty_cache` guard) and Change 2 (pool of N CAM++ pipeline instances), preserving per-instance serialization, with the H100 VRAM measurement gating the production pool size.

**Architecture:** A `queue.Queue`-backed thread-safe pool class (`ThreadedEnginePool`, twin of `LocalEnginePool`) lives beside the existing async pool. `app/utils/speaker_diarizer.py` builds N pipeline instances sequentially, sequenced by the pool's own init lock (`ThreadedEnginePool._init_lock`) — `_diarization_pipeline_lock` is used only by the one-time `_install_empty_cache_guard` install, which must not be called while it is held (corrected in review: an earlier draft claimed construction ran under `_diarization_pipeline_lock`, which would deadlock with the guard install) — each patched with `_enable_batched_sv`; the request path checks an instance out (that checkout IS the mutex) and back in via try/finally. The `empty_cache` guard is a process-wide wrapper installed idempotently, skipping only on threads that set a `threading.local` flag for the duration of a diarization call.

**Tech Stack:** Python stdlib (`queue`, `threading`, `contextlib`), torch, modelscope/funasr (already in `.venv`), `unittest` (NOT pytest).

### Global Constraints

- Dev box is a **non-CUDA AMD APU**: vLLM does not install, real diarization cannot run, VRAM/performance cannot be measured locally. Every step below is tagged **[LOCAL]** or **[H100-ONLY]**. Do not attempt an [H100-ONLY] step on the dev box.
- Local unit tests use fakes and verify **STRUCTURE ONLY** (checkout semantics, idempotence, config validation, warmup counts). They prove nothing about real concurrency, VRAM, or throughput. Only the H100 tasks validate behavior.
- Test runner: `DEVICE=cpu .venv/bin/python -m unittest discover -s tests` — currently GREEN at **134 tests** (corrected: plan was written against a stale count of 127); must stay green (test count will grow).
- Per-instance serialization is load-bearing (spec §"The mutex is load-bearing"). No task may allow one pipeline instance to be visible to two concurrent requests.
- Diarization stays synchronous. Do NOT make `SpeakerDiarizer.diarize` or anything under it async.
- Do NOT hardcode the production pool size on faith: default `DIARIZATION_POOL_SIZE=4` in code, but the **production value is gated on Task 7's measured VRAM delta** (spec §"Change 2", blocking step).
- Workers make **NO git commits**. Skip any commit habit; leave changes in the working tree.
- No `git pull`/fetch/rebase during the branch (spec §"Success criteria").
- Env setup if `.venv` is missing/stale: `./scripts/sync_cpu_env.sh`

### Task Order and Dependencies

```
Task 1 (config knob)            [LOCAL]  ─┐
Task 2 (ThreadedEnginePool)     [LOCAL]  ─┼─ independent, can run in parallel
Task 3 (empty_cache guard)      [LOCAL]  ─┘
Task 4 (pool wiring in speaker_diarizer)  [LOCAL]  depends on 1, 2, 3
Task 5 (warmup + test symbol move)        [LOCAL]  depends on 4
Task 6 (stage profiling, optional)        [LOCAL code, H100 readout]  independent of 4-5
Task 7 (VRAM measurement — BLOCKING)      [H100-ONLY]  depends on 4, 5; gates production N
Task 8 (bench + mixing verification)      [H100-ONLY]  depends on 7
```

---

### Task 1: `DIARIZATION_POOL_SIZE` config knob [LOCAL]

**Files:**
- Modify: `app/core/config.py` (default near line 74; env load near line 137)
- Test: create `tests/test_diarization_pool_config.py`

**Interfaces:**
- Produces: `settings.DIARIZATION_POOL_SIZE: int` (default 4, validated `>=1` at boot via the existing `_positive_int_from_env`). Task 4 consumes it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_diarization_pool_config.py`:

```python
"""DIARIZATION_POOL_SIZE must default to 4, be env-configurable, and reject
nonsense at boot via _positive_int_from_env (a 0-size pool means every
diarization call blocks forever)."""

import os
import unittest
from unittest import mock

from app.core.config import Settings


class DiarizationPoolConfigTest(unittest.TestCase):
    def test_default_is_4(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DIARIZATION_POOL_SIZE", None)
            self.assertEqual(Settings().DIARIZATION_POOL_SIZE, 4)

    def test_env_override(self) -> None:
        with mock.patch.dict(os.environ, {"DIARIZATION_POOL_SIZE": "2"}):
            self.assertEqual(Settings().DIARIZATION_POOL_SIZE, 2)

    def test_zero_rejected_at_boot(self) -> None:
        with mock.patch.dict(os.environ, {"DIARIZATION_POOL_SIZE": "0"}):
            with self.assertRaises(ValueError):
                Settings()

    def test_negative_rejected_at_boot(self) -> None:
        with mock.patch.dict(os.environ, {"DIARIZATION_POOL_SIZE": "-1"}):
            with self.assertRaises(ValueError):
                Settings()


if __name__ == "__main__":
    unittest.main()
```

Note: check `app/core/config.py` for the actual settings class name — if the module only exposes a singleton `settings` and the class is named differently (e.g. `Config`), import that name instead. Also confirm `_positive_int_from_env` raises `ValueError` (read its body around line 166-190); if it raises a different exception, assert that one.

- [ ] **Step 2: Run test to verify it fails**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_diarization_pool_config -v`
Expected: FAIL — `AttributeError: ... DIARIZATION_POOL_SIZE` (or default mismatch).

- [ ] **Step 3: Implement**

In `app/core/config.py`, next to `VLLM_OFFLINE_CONCURRENCY: int = 4` (line ~88), add:

```python
    # CAM++ 说话人分离 pipeline 池大小。每个实例独立持有模型权重与 CUDA
    # 缓存 —— 生产环境的取值必须以 H100 上 nvidia-smi 实测的单实例显存增量
    # 为准（见 docs/superpowers/specs/2026-07-17-diarization-throughput-design.md），
    # 不要凭空调大。
    # 不变量：必须保持 VLLM_OFFLINE_CONCURRENCY <= DIARIZATION_POOL_SIZE，
    # 否则 executor 线程会阻塞在 queue.Queue.get() 上等 pipeline，
    # 占着并发额度饿死其他请求（与 change A 治理过的饥饿同类）。
    DIARIZATION_POOL_SIZE: int = 4
```

In `_load_from_env`, next to the `VLLM_OFFLINE_CONCURRENCY` block (line ~151), add:

```python
        self.DIARIZATION_POOL_SIZE = self._positive_int_from_env(
            "DIARIZATION_POOL_SIZE", self.DIARIZATION_POOL_SIZE
        )
        if self.VLLM_OFFLINE_CONCURRENCY > self.DIARIZATION_POOL_SIZE:
            # Documented invariant (spec: blocking queue.Queue.get() holds an
            # executor slot). Warn loudly rather than refuse to boot: the
            # operator may run with diarization disabled per-request.
            import logging

            logging.getLogger(__name__).warning(
                "VLLM_OFFLINE_CONCURRENCY (%d) > DIARIZATION_POOL_SIZE (%d): "
                "diarization-enabled requests may block executor threads "
                "waiting for a pipeline instance",
                self.VLLM_OFFLINE_CONCURRENCY,
                self.DIARIZATION_POOL_SIZE,
            )
```

(Use `loguru.logger` instead if that is what `config.py` already imports — match the file's existing logging style; if it imports neither, the stdlib form above is fine.)

Also add a matching one-line comment at `VLLM_OFFLINE_CONCURRENCY`'s definition pointing at the invariant, so both knobs document it (spec: "enforce or document ... where both knobs are defined").

- [ ] **Step 4: Run tests**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_diarization_pool_config -v` → PASS
Run: `DEVICE=cpu .venv/bin/python -m unittest discover -s tests` → all green (134 + 4 new = 138) (corrected: plan was written against a stale count of 127).

---

### Task 2: `ThreadedEnginePool` — thread-safe twin of `LocalEnginePool` [LOCAL]

**Files:**
- Modify: `app/services/asr/runtime/local_pool.py` (append the new class; do not touch `LocalEnginePool`)
- Test: create `tests/test_threaded_engine_pool.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ThreadedEnginePool(size: int, factory: Callable[[], T])` with synchronous `warmup() -> None`, `acquire() -> T`, `release(engine: T) -> None`, and `size` property. Task 4 consumes it. Construction of instances happens inside `_ensure_queue` under the init lock — i.e. **sequential**, which is exactly what the spec's "construction must be sequential" requires as long as the factory itself does the modelscope work.

- [ ] **Step 1: Write the failing test**

Create `tests/test_threaded_engine_pool.py`:

```python
"""Structure-only tests for ThreadedEnginePool.

These verify checkout semantics with fakes on a CPU dev box. They prove
NOTHING about real GPU concurrency; the H100 mixing test is the real gate.
"""

import queue
import threading
import unittest

from app.services.asr.runtime.local_pool import ThreadedEnginePool


class ThreadedEnginePoolTest(unittest.TestCase):
    def test_lazy_builds_exactly_n_instances_sequentially(self) -> None:
        build_log = []

        def factory():
            build_log.append(threading.get_ident())
            return object()

        pool = ThreadedEnginePool(3, factory)
        self.assertEqual(build_log, [])  # lazy: nothing built yet
        pool.warmup()
        self.assertEqual(len(build_log), 3)
        # Sequential under the init lock: all built on one thread.
        self.assertEqual(len(set(build_log)), 1)

    def test_warmup_is_idempotent(self) -> None:
        count = [0]

        def factory():
            count[0] += 1
            return object()

        pool = ThreadedEnginePool(2, factory)
        pool.warmup()
        pool.warmup()
        self.assertEqual(count[0], 2)

    def test_checked_out_instance_is_exclusive(self) -> None:
        pool = ThreadedEnginePool(2, object)
        a = pool.acquire()
        b = pool.acquire()
        self.assertIsNot(a, b)
        # Pool exhausted: a third acquire must block, not hand out a dup.
        result: "queue.Queue[object]" = queue.Queue()
        t = threading.Thread(target=lambda: result.put(pool.acquire()))
        t.start()
        with self.assertRaises(queue.Empty):
            result.get(timeout=0.2)  # still blocked -> exclusivity holds
        pool.release(a)
        c = result.get(timeout=2.0)
        t.join(timeout=2.0)
        self.assertIs(c, a)  # released instance is the one handed out
        pool.release(b)
        pool.release(c)

    def test_size_floor_is_one(self) -> None:
        pool = ThreadedEnginePool(0, object)
        pool.warmup()
        self.assertIsNotNone(pool.acquire())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_threaded_engine_pool -v`
Expected: FAIL — `ImportError: cannot import name 'ThreadedEnginePool'`.

- [ ] **Step 3: Implement**

Append to `app/services/asr/runtime/local_pool.py` (add `import queue` at top; `threading` is already imported):

```python
class ThreadedEnginePool(Generic[T]):
    """Thread-safe twin of :class:`LocalEnginePool` backed by ``queue.Queue``.

    For engines used *synchronously from executor threads* (diarization runs
    inside ``run_sync``); the async pool above cannot serve that path. An
    instance checked out here is exclusively owned by the caller until
    ``release`` — the pool itself is the per-instance mutex.

    Invariant (see DIARIZATION_POOL_SIZE in app/core/config.py): a blocking
    ``acquire()`` holds its executor-thread slot while waiting, so the number
    of concurrent callers must not exceed the pool size.
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
        self._ensure_queue().put(engine)
```

- [ ] **Step 4: Run tests**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_threaded_engine_pool -v` → PASS
Run: full suite → green.

---

### Task 3: Change 1 — thread-local `empty_cache` guard, installed idempotently [LOCAL]

**Files:**
- Modify: `app/utils/speaker_diarizer.py` (module level, near the existing lock definitions at lines 22-25)
- Test: create `tests/test_empty_cache_guard.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `_install_empty_cache_guard() -> None` (idempotent) and context manager `_suppress_empty_cache()` in `app/utils/speaker_diarizer.py`. Task 4 wraps the `pipeline(audio_path)` call with `_suppress_empty_cache()` and calls `_install_empty_cache_guard()` from the pipeline factory.

- [ ] **Step 1: Write the failing test**

Create `tests/test_empty_cache_guard.py`:

```python
"""Structure-only tests for the torch.cuda.empty_cache guard.

funasr calls torch.cuda.empty_cache() after every inference
(funasr/auto/auto_model.py:410-417), flushing the allocator on the GPU vLLM
shares. The guard skips it only on threads inside a diarization call.
On this CPU dev box empty_cache is a no-op, so these tests verify wiring
(idempotence, thread-locality, restore-on-exit) with a sentinel — not any
CUDA behavior.
"""

import threading
import unittest

import torch

from app.utils import speaker_diarizer as sd


class EmptyCacheGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        # Isolate each test from prior installs.
        self._original = torch.cuda.empty_cache
        self.calls = []
        torch.cuda.empty_cache = lambda: self.calls.append(True)
        sd._empty_cache_guard_installed = False

    def tearDown(self) -> None:
        torch.cuda.empty_cache = self._original
        sd._empty_cache_guard_installed = False

    def test_install_is_idempotent_no_nested_wrappers(self) -> None:
        sd._install_empty_cache_guard()
        first = torch.cuda.empty_cache
        sd._install_empty_cache_guard()
        self.assertIs(torch.cuda.empty_cache, first)  # not re-wrapped

    def test_passthrough_outside_diarization(self) -> None:
        sd._install_empty_cache_guard()
        torch.cuda.empty_cache()
        self.assertEqual(len(self.calls), 1)

    def test_skipped_inside_suppress_and_restored_after(self) -> None:
        sd._install_empty_cache_guard()
        with sd._suppress_empty_cache():
            torch.cuda.empty_cache()
        self.assertEqual(len(self.calls), 0)
        torch.cuda.empty_cache()
        self.assertEqual(len(self.calls), 1)

    def test_restored_even_on_exception(self) -> None:
        sd._install_empty_cache_guard()
        with self.assertRaises(RuntimeError):
            with sd._suppress_empty_cache():
                raise RuntimeError("boom")
        torch.cuda.empty_cache()
        self.assertEqual(len(self.calls), 1)

    def test_flag_is_thread_local(self) -> None:
        sd._install_empty_cache_guard()
        other_thread_calls = []

        def other() -> None:
            torch.cuda.empty_cache()
            other_thread_calls.append(len(self.calls))

        with sd._suppress_empty_cache():
            t = threading.Thread(target=other)
            t.start()
            t.join(timeout=5.0)
        # The other thread was NOT suppressed.
        self.assertEqual(other_thread_calls, [1])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_empty_cache_guard -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_install_empty_cache_guard'`.

- [ ] **Step 3: Implement**

In `app/utils/speaker_diarizer.py`, after the existing module-level locks (line ~25), add (`contextlib` import goes at the top of the file; `threading` and `torch` are already imported):

```python
import contextlib

# --- torch.cuda.empty_cache guard -------------------------------------------
# funasr runs `torch.cuda.empty_cache()` after EVERY inference
# (funasr/auto/auto_model.py:410-417): a device sync plus an allocator flush
# on the same GPU vLLM is using, inside our per-request diarization critical
# path. `empty_cache` is resolved at call time as `torch.cuda.empty_cache`,
# so it cannot be patched per-module; instead we install ONE wrapper that
# skips only when the calling thread is inside a diarization call.
# Thread-local, not a module flag: diarization runs on executor threads while
# other threads use CUDA concurrently.
_empty_cache_tls = threading.local()
_empty_cache_guard_installed = False


def _install_empty_cache_guard() -> None:
    """Wrap torch.cuda.empty_cache once. Idempotent: never nests wrappers."""
    global _empty_cache_guard_installed
    with _diarization_pipeline_lock:
        if _empty_cache_guard_installed or getattr(
            torch.cuda.empty_cache, "_diarization_guard", False
        ):
            _empty_cache_guard_installed = True
            return

        real_empty_cache = torch.cuda.empty_cache

        def _guarded_empty_cache() -> None:
            if getattr(_empty_cache_tls, "skip", False):
                return
            real_empty_cache()

        _guarded_empty_cache._diarization_guard = True  # type: ignore[attr-defined]
        torch.cuda.empty_cache = _guarded_empty_cache
        _empty_cache_guard_installed = True
        logger.info("已安装 torch.cuda.empty_cache 线程局部守卫（跳过 diarization 内的调用）")


@contextlib.contextmanager
def _suppress_empty_cache():
    """Skip torch.cuda.empty_cache on THIS thread for the duration."""
    _empty_cache_tls.skip = True
    try:
        yield
    finally:
        _empty_cache_tls.skip = False
```

Note the double idempotence check: the module flag catches the normal path; the `_diarization_guard` attribute on the current `torch.cuda.empty_cache` catches module reloads (fresh module globals, already-wrapped torch) — spec risk "reload, tests, multiple workers".

- [ ] **Step 4: Run tests**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_empty_cache_guard -v` → PASS
Run: full suite → green.

---

### Task 4: Change 2 — replace singleton+semaphore with the pool in `speaker_diarizer.py` [LOCAL]

**Files:**
- Modify: `app/utils/speaker_diarizer.py` (lines 22-25 module globals; `get_global_diarization_pipeline` at 214-248; `SpeakerDiarizer.diarize` at 279-283)
- Test: create `tests/test_diarization_pool_wiring.py`

**Interfaces:**
- Consumes: `ThreadedEnginePool` (Task 2), `settings.DIARIZATION_POOL_SIZE` (Task 1), `_install_empty_cache_guard`/`_suppress_empty_cache` (Task 3).
- Produces, in `app/utils/speaker_diarizer.py`:
  - `_build_diarization_pipeline() -> Any` — module-level factory; body is the current `get_global_diarization_pipeline` load logic (create pipeline, `_enable_batched_sv`), NO caching, NO lock (the pool's init lock provides sequencing), plus `_install_empty_cache_guard()` first.
  - `_diarization_pool: ThreadedEnginePool[Any]` — module-level, size `settings.DIARIZATION_POOL_SIZE`.
  - `warmup_diarization_pool() -> int` — builds all N, returns pool size. Task 5's warmup and test consume this exact name.
  - `get_global_diarization_pipeline` is **deleted** (Task 5 migrates its two remaining consumers; grep confirmed only three call sites exist).

- [ ] **Step 1: Write the failing test**

Create `tests/test_diarization_pool_wiring.py`:

```python
"""Structure-only tests: the diarization request path checks a pipeline out
of the pool for the duration of the pipeline call and returns it in a
finally. These use fakes on CPU; they cannot prove GPU-concurrency safety —
scripts/h100/test_offline_mixing.py is the real gate for that.
"""

import unittest
from unittest import mock

from app.utils import speaker_diarizer as sd


class _FakePipeline:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, audio_path):
        self.calls.append(audio_path)
        return {"text": [[0.0, 1.5, 0], [1.5, 3.0, 1]]}


class DiarizationPoolWiringTest(unittest.TestCase):
    def test_diarize_acquires_calls_and_releases(self) -> None:
        fake = _FakePipeline()
        with mock.patch.object(sd._diarization_pool, "acquire", return_value=fake) as acq, \
             mock.patch.object(sd._diarization_pool, "release") as rel:
            segments = sd.SpeakerDiarizer().diarize("/tmp/x.wav")
        acq.assert_called_once_with()
        rel.assert_called_once_with(fake)
        self.assertEqual(fake.calls, ["/tmp/x.wav"])
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].speaker_id, "说话人1")

    def test_release_happens_even_when_pipeline_raises(self) -> None:
        broken = mock.Mock(side_effect=RuntimeError("cuda meltdown"))
        with mock.patch.object(sd._diarization_pool, "acquire", return_value=broken), \
             mock.patch.object(sd._diarization_pool, "release") as rel:
            with self.assertRaises(Exception):
                sd.SpeakerDiarizer().diarize("/tmp/x.wav")
        rel.assert_called_once_with(broken)

    def test_pool_sized_from_settings(self) -> None:
        from app.core.config import settings

        self.assertEqual(sd._diarization_pool.size, settings.DIARIZATION_POOL_SIZE)

    def test_global_singleton_accessor_is_gone(self) -> None:
        self.assertFalse(hasattr(sd, "get_global_diarization_pipeline"))

    def test_warmup_builds_all_n(self) -> None:
        with mock.patch.object(sd._diarization_pool, "warmup") as warm:
            n = sd.warmup_diarization_pool()
        warm.assert_called_once_with()
        self.assertEqual(n, sd._diarization_pool.size)


if __name__ == "__main__":
    unittest.main()
```

Caution: `test_global_singleton_accessor_is_gone` will fail until Task 5 also lands (model_loader still imports the old name → ImportError at import time elsewhere). Run Tasks 4 and 5 back-to-back; the suite is only required green at the end of Task 5.

- [ ] **Step 2: Run test to verify it fails**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_diarization_pool_wiring -v`
Expected: FAIL — `AttributeError: ... _diarization_pool`.

- [ ] **Step 3: Implement**

In `app/utils/speaker_diarizer.py`:

(a) Delete line 25 (`_diarization_inference_semaphore = threading.BoundedSemaphore(1)`) — grep confirms nothing else references it. Keep `_diarization_pipeline_lock` (the guard install in Task 3 uses it, and it still guards nothing else — that's fine).

(b) Replace the whole `get_global_diarization_pipeline` function (lines 214-248) with:

```python
def _build_diarization_pipeline() -> Any:
    """Build ONE independent CAM++ pipeline instance (weights + batched SV).

    Called by the pool factory under the pool's init lock, so construction is
    sequential — modelscope `pipeline()` touches global registries and may
    download on first run; building N concurrently is a race.

    Each instance is fully independent (own sv/vad/change_locator children,
    per-instance MethodType patch). funasr mutates shared per-instance state
    on every call (auto_model.py kwargs, fsmn_vad is_final toggling), so an
    instance must NEVER be visible to two concurrent requests: the pool
    checkout is the mutex that replaced _diarization_inference_semaphore.
    """
    _install_empty_cache_guard()
    try:
        from modelscope.utils.constant import Tasks
        from ..infrastructure.model_utils import resolve_model_path

        model_id = 'iic/speech_campplus_speaker-diarization_common'
        model_path = resolve_model_path(model_id)
        modelscope_device = _resolve_modelscope_device()

        logger.info(
            "正在加载 CAM++ 说话人分离模型: {}, device={}",
            model_path,
            modelscope_device,
        )
        pipeline_instance = _create_modelscope_pipeline(
            task=Tasks.speaker_diarization,
            model=model_path,
            modelscope_device=modelscope_device,
        )
        pipeline_instance = _enable_batched_sv(
            pipeline_instance,
            modelscope_device,
            max_batch_size=settings.DIARIZATION_SV_BATCH_SIZE,
        )
        logger.info("CAM++ 模型加载成功（已启用 batched SV）")
        return pipeline_instance
    except Exception as e:
        logger.error(f"CAM++ 模型加载失败: {e}")
        raise DefaultServerErrorException(f"说话人分离模型加载失败: {str(e)}")


from ..services.asr.runtime.local_pool import ThreadedEnginePool  # noqa: E402

_diarization_pool: ThreadedEnginePool[Any] = ThreadedEnginePool(
    settings.DIARIZATION_POOL_SIZE, _build_diarization_pipeline
)


def warmup_diarization_pool() -> int:
    """Eagerly build all N pipeline instances (startup warmup). Returns N."""
    _diarization_pool.warmup()
    return _diarization_pool.size
```

(Move the `from ..services.asr.runtime.local_pool import ThreadedEnginePool` up to the file's import block if it imports cleanly there — verify no circular import first by running the suite; if circular, keep it as a function-local import inside a small `_make_pool()` helper or leave the late module-level import with the `noqa`.)

(c) In `SpeakerDiarizer.diarize`, replace lines 279-283:

```python
            pipeline = get_global_diarization_pipeline()

            logger.info(f"开始说话人分离: {audio_path}")
            with _diarization_inference_semaphore:
                result = pipeline(audio_path)
```

with:

```python
            logger.info(f"开始说话人分离: {audio_path}")
            # Checkout = per-instance mutex: this instance is ours alone
            # until release. funasr mutates instance state per call, so the
            # instance must not be shared; the finally makes leaks impossible
            # even when the pipeline raises (e.g. "too short" audio).
            pipeline = _diarization_pool.acquire()
            try:
                with _suppress_empty_cache():
                    result = pipeline(audio_path)
            finally:
                _diarization_pool.release(pipeline)
```

- [ ] **Step 4: Run the new tests**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_diarization_pool_wiring -v`
Expected: the four pool tests PASS; expect `tests/test_preload_models_config_repair.py` to now FAIL (patches the deleted symbol) and `model_loader.py` import to fail — that is Task 5's job. Do not "fix" this by resurrecting the old name.

---

### Task 5: Migrate warmup + test to the pool [LOCAL]

**Files:**
- Modify: `app/utils/model_loader.py:533-546`
- Modify: `tests/test_preload_models_config_repair.py:44-47`

**Interfaces:**
- Consumes: `warmup_diarization_pool() -> int` (Task 4).
- Produces: green full suite; warmup builds ALL N instances at startup.

- [ ] **Step 1: Update the warmup block**

In `app/utils/model_loader.py`, replace lines 535-542:

```python
        try:
            from ..utils.speaker_diarizer import get_global_diarization_pipeline

            diarization_pipeline = get_global_diarization_pipeline()
            if diarization_pipeline:
                result["speaker_diarization_model"]["loaded"] = True
            else:
                result["speaker_diarization_model"]["error"] = "说话人分离模型加载后返回None"
```

with:

```python
        try:
            from ..utils.speaker_diarizer import warmup_diarization_pool

            # Build ALL N pool instances now, sequentially: modelscope
            # pipeline() touches global registries and may download, and a
            # lazy build at request time would pay N model loads on the
            # first N requests.
            pool_size = warmup_diarization_pool()
            if pool_size >= 1:
                result["speaker_diarization_model"]["loaded"] = True
            else:
                result["speaker_diarization_model"]["error"] = "说话人分离池为空"
```

Also update the progress strings if desired (optional): `progress.update("加载说话人分离模型(CAM++ 池)")` — cosmetic, skip if it risks breaking log-grepping tests (grep tests/ for the string first; none matched at plan time).

- [ ] **Step 2: Update the patched symbol in the test**

In `tests/test_preload_models_config_repair.py`, replace lines 44-47:

```python
            mock.patch(
                "app.utils.speaker_diarizer.get_global_diarization_pipeline",
                return_value=object(),
            ),
```

with:

```python
            mock.patch(
                "app.utils.speaker_diarizer.warmup_diarization_pool",
                return_value=4,
            ),
```

- [ ] **Step 3: Run the full suite**

Run: `DEVICE=cpu .venv/bin/python -m unittest discover -s tests`
Expected: ALL green, including `test_preload_models_config_repair` and every Task 1-4 test (134 originals + 19 new = 153; corrected: plan was written against a stale count of 127). If anything else red, fix before proceeding — the suite green is a hard gate.

- [ ] **Step 4: Import smoke check**

Run: `DEVICE=cpu .venv/bin/python -c "import app.utils.speaker_diarizer, app.utils.model_loader; print('ok')"`
Expected: `ok` (verifies no circular import from the ThreadedEnginePool import placement).

---

### Task 6 (optional, non-blocking): Stage profiling instrumentation [LOCAL to write, H100-ONLY to read]

Spec §"Ordering" item 2: profile the 0.883s to decide follow-on work (VAD vs change-locator vs nothing). Blocks nothing in this plan; do it if time allows, ideally before Task 8 so the bench run also yields the profile.

**Files:**
- Modify: `app/utils/speaker_diarizer.py` (inside `SpeakerDiarizer.diarize`, or as a wrapper installed next to `_enable_batched_sv`)

- [ ] **Step 1 [LOCAL]:** In `_build_diarization_pipeline` (or a helper `_enable_stage_timing(pipeline_instance)` called after `_enable_batched_sv`), wrap the pipeline's `preprocess`, `forward` (already ours), and `postprocess` methods via `types.MethodType` in the `_enable_batched_sv` style, logging `time.perf_counter()` deltas per stage at INFO with a `[diarization-profile]` prefix. Gate on `os.getenv("DIARIZATION_STAGE_TIMINGS", "").lower() == "true"` so production logs stay quiet. Verify locally only that the suite stays green and import works — the timings themselves are meaningless on CPU.
- [ ] **Step 2 [H100-ONLY]:** Run one request with `DIARIZATION_STAGE_TIMINGS=true`, record the per-stage split of the ~0.883s in the bench notes. Decision input for future work (Silero/process pool), not for this branch.

---

### Task 7: H100 VRAM measurement — BLOCKING gate for production N [H100-ONLY]

Spec §"Change 2": *"VRAM per instance is UNMEASURED — measure it on the H100 before fixing N. This is a blocking step, not a formality."* Two of the four model components have never been loaded anywhere measurable. **Do not deploy with `DIARIZATION_POOL_SIZE=4` (or anything >1) until this task's numbers exist.** Do not attempt any part of this task on the dev box — no CUDA, and two model dirs are not even cached.

**Files:** none in-repo (a scratch script on the H100; record numbers in the deploy notes / PR description).

- [ ] **Step 1: Deploy the branch to the H100** (usual deploy path; no git operations beyond what deploy already does — no pull/rebase on the branch).

- [ ] **Step 2: Measure the per-instance delta.** With the service STOPPED (so vLLM isn't confounding), on the H100:

Because the first instance's delta includes the one-time CUDA context, the script ALSO builds a second instance in the same process and records its (marginal) delta — that marginal number is the true per-additional-instance cost. It then runs one real diarization call (`p(path_to_2min_wav)`) at the production `DIARIZATION_SV_BATCH_SIZE` (128) before the final reading, so activation/allocator peak is included. All of this happens in ONE process — a fresh interpreter would lose the already-built instances and the CUDA context, so the steps must not be split across separate runs:

```bash
# scratch script: measure_diarization_vram.py (run with the service venv)
export DIARIZATION_POOL_SIZE=1
python - << 'EOF'
import subprocess, torch

def used_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    return int(out)

before = used_mib()
from app.utils.speaker_diarizer import _build_diarization_pipeline
p = _build_diarization_pipeline()   # one instance, _enable_batched_sv applied
torch.cuda.synchronize()
after_build = used_mib()
print(f"before={before} MiB  after_build={after_build} MiB  "
      f"delta={after_build - before} MiB (includes CUDA context + throwaway "
      f"duplicate SV pipeline from SegmentationClusteringPipeline.__init__)")

# Second instance, same process: marginal = true per-additional-instance cost
# (the first delta above includes the one-time CUDA context).
before2 = used_mib()
p2 = _build_diarization_pipeline()
torch.cuda.synchronize()
print(f"marginal per-instance delta = {used_mib() - before2} MiB")

# One real diarization call at production DIARIZATION_SV_BATCH_SIZE (128) so
# the final reading includes the activation/allocator peak.
# EDIT the path below on the H100 before running:
# p(path_to_2min_wav)
torch.cuda.synchronize()
print(f"final used = {used_mib()} MiB (after real call: includes activation peak)")
EOF
```

- [ ] **Step 3: Compute available headroom.** Record what vLLM + the forced aligner already reserve: `nvidia-smi` with the FULL service running (pool size 1). Headroom = 80GB card − reserved − safety margin (≥10%).

- [ ] **Step 4: Fix production N.** `N = min(4, floor(headroom / marginal_per_instance_delta))`, also respecting `N >= VLLM_OFFLINE_CONCURRENCY` (=4 default) per the documented invariant — if VRAM only affords N<4, LOWER `VLLM_OFFLINE_CONCURRENCY` to match rather than shipping waiters. Set `DIARIZATION_POOL_SIZE` in the production env accordingly and record: before/after/delta MiB, marginal delta, reserved-by-vLLM, chosen N, in the PR/deploy notes. **This step is the gate: no bench (Task 8) with N>1 until these numbers are written down.**

---

### Task 8: H100 verification — mixing test, bench, success criteria [H100-ONLY]

Depends on Task 7's N. All spec §"Success criteria" items checked here.

- [ ] **Step 1: Start the service** with the chosen `DIARIZATION_POOL_SIZE` and production `DIARIZATION_SV_BATCH_SIZE=128`. Confirm from the startup log that warmup built all N instances (N × "CAM++ 模型加载成功" lines) and the empty_cache guard install line appears once.

- [ ] **Step 2: Mixing test (MANDATORY).** Run `scripts/h100/test_offline_mixing.py` against the running service. Expected: PASS — zero cross-request transcript/speaker mixing. If it fails, STOP: per-instance serialization is broken; do not proceed to the bench, go back to Task 4's checkout logic.

- [ ] **Step 3: Bench.** `scripts/h100/bench.sh --audio <the same 2-minute file as the baseline>` at n=10, diarization ON. Compare against the recorded baseline: **0.78 req/s, 12.8s wall, diarization_s 4.299s**. Success per spec: `diarization_s` drops materially from 4.299s (no target multiple — do not invent one), throughput moves toward the 1.47 req/s ceiling, and **zero fails**. Also do one n=1 run to see whether Change 1 moved the 0.883s service time.

- [ ] **Step 4: VRAM stability check.** During the n=10 bench, watch `nvidia-smi` — no growth trend across runs (an instance leak would show as monotonic growth), no OOM.

- [ ] **Step 5: Record results** (numbers, not adjectives) in the PR description: n=1 and n=10 diarization_s, req/s, wall, fails, VRAM figures from Task 7, and the Task 6 stage profile if collected. If the win is under ~1.3x, say so plainly — the spec hedges the GIL ceiling and forbids inventing forecasts.

---

### Self-review notes

- Spec coverage: Change 1 → Task 3+4(c); Change 2 → Tasks 1,2,4,5; VRAM blocking measurement → Task 7 (explicitly gates N); profiling (spec Ordering §2, non-blocking) → Task 6; bench/mixing/success criteria → Task 8; invariant `VLLM_OFFLINE_CONCURRENCY <= DIARIZATION_POOL_SIZE` → documented at both knobs + boot warning (Task 1) + Task 7 Step 4 operationalizes it.
- All three `get_global_diarization_pipeline` call sites are moved: request path (Task 4c), warmup (Task 5.1), test patch (Task 5.2); the symbol is deleted and its absence asserted (Task 4 test).
- Known transient red between Task 4 and Task 5 is called out; suite-green is gated at end of Task 5.
- Local tests are labeled structure-only in their own docstrings; nothing local claims concurrency proof.

---

### Post-review amendments (final whole-branch review, 2026-07-17)

- **Task 6 narrowing, now explicit:** the instrumentation wraps `preprocess`/`forward`/`postprocess` (3 wrap points), not the spec's five named stages. Decision inputs survive: VAD≈preprocess, embed≈forward, change_locator is inside postprocess, clustering is spec-verified cheap; chunk+cluster residual = `diarization_s` − (sum of stages). **Collect the profile at n=1 only** — profile lines carry no request tag and interleave unattributably under concurrency.
- **Task 8 additional operator watch items:**
  1. A boot line "CAM++ 模型加载失败" is a HARD STOP: warmup failure is swallowed at boot, and the lazy fallback then builds all N instances from an executor thread on the first diarization request while holding an admission permit — serializing and stalling offline traffic, corrupting every measurement (tracked as a bd issue).
  2. Expect ~N× longer boot (sequential builds). N × "CAM++ 模型加载成功" + exactly one guard-install line is the proof the pool is populated (Step 1 already requires this).
  3. Run the Step 4 VRAM watch long enough to see the allocator plateau: with `empty_cache` suppressed on diarization threads, the CAM++ allocator cache grows to a steady state it never previously reached (N instances × batch-128 SV activations).
  4. The boot warning for `VLLM_OFFLINE_CONCURRENCY <= DIARIZATION_POOL_SIZE` only sees the vLLM offline knob. Other engine families (FUNASR_WORKERS, QWEN_RUST_CPU_WORKERS, a second vLLM model id) add concurrent diarize callers it cannot see — re-derive the bound before adding families. Single-process deployment assumed; multiple uvicorn workers multiply pool VRAM by worker count on top of Task 7's arithmetic.
