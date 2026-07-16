from __future__ import annotations

import asyncio
import os
import threading
import time
import unittest
from unittest import mock

from app.services.asr.runtime.router import RuntimeFamily, RuntimeRouter


class SharedEngineAdmissionTest(unittest.IsolatedAsyncioTestCase):
    def _router_with_fake_shared_engine(self):
        # Constructing RuntimeRouter() resolves the default ASR model via
        # ModelManager._load_models_config -> get_default_model_id, which
        # raises RuntimeError on boxes with no runnable Qwen3-ASR model
        # (e.g. non-CUDA dev boxes without the QwenASR rust build). None of
        # that resolution matters here: the test replaces _resolve_family
        # and _get_shared_engine right below, so stub the model lookup to
        # keep router construction environment-independent.
        import app.services.asr.manager as manager_mod

        with mock.patch(
            "app.services.asr.manager.get_default_model_id",
            return_value="qwen3-asr-test",
        ):
            with mock.patch.object(manager_mod, "_model_manager", None):
                router = RuntimeRouter()
        engine = object()
        semaphore = asyncio.Semaphore(2)
        router._resolve_family = lambda _m: RuntimeFamily.QWEN_VLLM  # type: ignore[method-assign]
        router._get_shared_engine = lambda _f, _m: (engine, semaphore)  # type: ignore[method-assign]
        return router, engine, semaphore

    async def test_websocket_lease_consumes_no_permit(self) -> None:
        router, engine, semaphore = self._router_with_fake_shared_engine()
        lease = await router.lease_shared_engine("qwen3-asr-test")
        self.assertIs(lease.engine, engine)
        self.assertEqual(semaphore._value, 2)  # untouched
        await lease.close()
        self.assertEqual(semaphore._value, 2)  # close is a no-op on permits

    async def test_offline_lease_still_consumes_permit(self) -> None:
        router, _engine, semaphore = self._router_with_fake_shared_engine()
        lease = await router.acquire_engine("qwen3-asr-test")
        self.assertEqual(semaphore._value, 1)
        await lease.close()
        self.assertEqual(semaphore._value, 2)

    def test_semaphore_values_come_from_settings(self) -> None:
        # Assert the class defaults and the env-override mechanism, NOT the
        # live singleton's values — a runner with VLLM_*_CONCURRENCY exported
        # would otherwise fail this spuriously.
        from app.core.config import Settings

        self.assertEqual(Settings.VLLM_OFFLINE_CONCURRENCY, 4)
        self.assertEqual(Settings.VLLM_WS_DECODE_CONCURRENCY, 4)

        with mock.patch.dict(
            os.environ,
            {"VLLM_OFFLINE_CONCURRENCY": "7", "VLLM_WS_DECODE_CONCURRENCY": "9"},
        ):
            s = Settings()
            self.assertEqual(s.VLLM_OFFLINE_CONCURRENCY, 7)
            self.assertEqual(s.VLLM_WS_DECODE_CONCURRENCY, 9)

    async def test_ws_decode_dispatch_bounded_by_decode_semaphore(self) -> None:
        # Item 2's mechanism: at most VLLM_WS_DECODE_CONCURRENCY decode
        # dispatches in flight, regardless of how many sessions stream.
        import app.services.qwen3_websocket_asr as ws

        mu = threading.Lock()
        state = {"active": 0, "max_active": 0}

        def fake_decode() -> None:
            with mu:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.02)
            with mu:
                state["active"] -= 1

        await asyncio.gather(*(ws._run_decode(fake_decode) for _ in range(12)))
        # Bound assertions, not an exact peak: a slow/loaded executor may never
        # reach 4 concurrently. The bound is what matters; >1 proves the calls
        # do overlap, so an accidentally-serializing impl still fails.
        self.assertLessEqual(state["max_active"], 4)  # VLLM_WS_DECODE_CONCURRENCY default
        self.assertGreater(state["max_active"], 1)

    def test_executor_default_workers_formula(self) -> None:
        # Tests the FORMULA via the pure function, not the box's cpu_count —
        # `_MAX_WORKERS >= 8` would already pass on any >=8-thread machine
        # and would never exercise the sum term. Fails first with
        # ImportError: compute_default_workers does not exist yet.
        from app.core.executor import compute_default_workers

        # sum term dominates on a small box: 4 offline + 4 ws decode
        self.assertEqual(compute_default_workers(2, 4, 4), 8)
        # cpu_count dominates on a big box
        self.assertEqual(compute_default_workers(64, 4, 4), 64)
        # raised knobs raise the floor with them
        self.assertEqual(compute_default_workers(2, 16, 8), 24)
        # absolute floor of 4
        self.assertEqual(compute_default_workers(1, 1, 1), 4)


if __name__ == "__main__":
    unittest.main()
