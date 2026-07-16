# tests/test_branch_review_fixes.py
"""Whole-branch review fixes for change A.

These are fake-based structural/unit tests. They pin the mechanisms the
review asked for (ITN never locking on the loop thread, /health taking no
admission permit, concurrency config rejected at boot). They do NOT and
cannot prove concurrency safety of the real vLLM/wetext backends.
"""
from __future__ import annotations

import asyncio
import os
import threading
import unittest
from unittest import mock


class ItnOffLoopTest(unittest.IsolatedAsyncioTestCase):
    async def test_qwen3_ws_normalize_runs_off_the_event_loop(self) -> None:
        # The hazard: normalize_asr_text takes a blocking process-wide lock.
        # On the loop thread it would freeze every connection. Assert the call
        # lands on some OTHER thread than the loop's.
        import app.services.qwen3_websocket_asr as ws

        loop_thread = threading.get_ident()
        seen: list[int] = []

        def fake_normalize(text: str, enable_itn: bool) -> str:
            seen.append(threading.get_ident())
            return text + "!"

        ctx = mock.MagicMock()
        ctx.params = {"enable_inverse_text_normalization": True}

        with mock.patch.object(ws, "normalize_asr_text", fake_normalize):
            handler = ws.Qwen3ASRService.__new__(ws.Qwen3ASRService)
            out = await ws.Qwen3ASRService._normalize_output_text(handler, "一二三", ctx)

        self.assertEqual(out, "一二三!")
        self.assertEqual(len(seen), 1)
        self.assertNotEqual(seen[0], loop_thread, "ITN ran on the event loop thread")

    async def test_qwen3_ws_normalize_skips_dispatch_when_itn_disabled(self) -> None:
        import app.services.qwen3_websocket_asr as ws

        called: list[str] = []

        ctx = mock.MagicMock()
        ctx.params = {"enable_inverse_text_normalization": False}

        with mock.patch.object(
            ws, "normalize_asr_text", lambda *a, **k: called.append("x")
        ):
            handler = ws.Qwen3ASRService.__new__(ws.Qwen3ASRService)
            out = await ws.Qwen3ASRService._normalize_output_text(handler, "abc", ctx)

        self.assertEqual(out, "abc")
        self.assertEqual(called, [], "ITN dispatched despite enable_itn=False")

    async def test_ws_decode_semaphore_not_consumed_by_itn(self) -> None:
        # ITN is CPU work on its own lock; it must not spend a GPU decode
        # permit. Assert the ws-decode semaphore is untouched by normalize.
        import app.services.qwen3_websocket_asr as ws

        sem = asyncio.Semaphore(2)
        ctx = mock.MagicMock()
        ctx.params = {"enable_inverse_text_normalization": True}

        with mock.patch.object(ws, "_get_ws_decode_semaphore", lambda: sem):
            with mock.patch.object(ws, "normalize_asr_text", lambda t, e: t):
                handler = ws.Qwen3ASRService.__new__(ws.Qwen3ASRService)
                await ws.Qwen3ASRService._normalize_output_text(handler, "abc", ctx)

        self.assertEqual(sem._value, 2)


class ItnLockContractTest(unittest.TestCase):
    def test_get_normalizer_rejects_unlocked_caller(self) -> None:
        import app.utils.text_processing as tp

        with self.assertRaises(AssertionError):
            tp._get_normalizer()

    def test_warmup_itn_builds_normalizer_before_any_request(self) -> None:
        import app.utils.text_processing as tp

        constructed: list[object] = []

        class _FakeNormalizer:
            def __init__(self, lang="zh", operator="itn") -> None:
                constructed.append(self)

            def normalize(self, text: str) -> str:
                return text

        fake_wetext = mock.MagicMock()
        fake_wetext.Normalizer = _FakeNormalizer

        with mock.patch.dict("sys.modules", {"wetext": fake_wetext}):
            tp._wetext_normalizer = None
            try:
                self.assertTrue(tp.warmup_itn())
                self.assertEqual(len(constructed), 1)
                # A subsequent request must reuse the warm singleton, so the
                # multi-second init never runs under load.
                tp.apply_itn_to_text("一百二十三")
                self.assertEqual(len(constructed), 1)
            finally:
                tp._wetext_normalizer = None


def _import_asr_api():
    """Import app.api.v1.asr without depending on this box's model plan.

    app/api/v1/__init__.py imports openai_compatible, which resolves the
    active Qwen model at MODULE level and raises on a machine with no runnable
    Qwen3-ASR (this dev box). Stub that one resolution for the duration of the
    import; none of it is what this test measures.
    """
    import app.services.asr.manager as manager_mod

    with mock.patch(
        "app.services.asr.model_plan.get_active_qwen_model",
        # A declared model id, so the import-time model_selection lookups
        # resolve; which one is irrelevant to what this test measures.
        return_value="qwen3-asr-0.6b",
    ):
        with mock.patch.object(manager_mod, "_model_manager", None):
            import app.api.v1.asr as asr_api

            return asr_api


class HealthAdmissionTest(unittest.IsolatedAsyncioTestCase):
    async def test_health_check_consumes_no_offline_permit(self) -> None:
        asr_api = _import_asr_api()

        engine = mock.MagicMock()
        engine.device = "cuda:0"
        semaphore = asyncio.Semaphore(4)

        router = mock.MagicMock()
        router.resolve_model_id.return_value = "qwen3-asr-test"
        router.get_memory_usage.return_value = {"gpu_memory": None}
        router.get_loaded_model_ids.return_value = ["qwen3-asr-test"]

        from app.services.asr.runtime.router import RuntimeEngineLease

        async def lease_shared_engine(model_id=None):
            return RuntimeEngineLease(engine=engine, release_callback=lambda: None)

        async def acquire_engine(model_id=None):
            await semaphore.acquire()
            return RuntimeEngineLease(engine=engine, release_callback=semaphore.release)

        router.lease_shared_engine = lease_shared_engine
        router.acquire_engine = acquire_engine

        # Drain every permit, as VLLM_OFFLINE_CONCURRENCY in-flight
        # transcriptions would. Health must still answer immediately.
        for _ in range(4):
            await semaphore.acquire()
        self.assertEqual(semaphore._value, 0)

        with mock.patch.object(asr_api, "get_runtime_router", lambda: router):
            with mock.patch.object(asr_api, "validate_token", lambda r: (True, "")):
                result = await asyncio.wait_for(
                    asr_api.health_check(mock.MagicMock()), timeout=5.0
                )

        self.assertEqual(result["status"], "healthy")
        self.assertEqual(result["device"], "cuda:0")
        self.assertEqual(semaphore._value, 0, "health took an offline permit")


class ConcurrencyConfigValidationTest(unittest.TestCase):
    def _settings_with(self, **env):
        from app.core.config import Settings

        with mock.patch.dict(os.environ, env):
            return Settings()

    def test_zero_offline_concurrency_fails_at_boot(self) -> None:
        # Semaphore(0) would make every offline request await forever with no
        # log and no error. Refuse to boot instead.
        with self.assertRaises(ValueError) as cm:
            self._settings_with(VLLM_OFFLINE_CONCURRENCY="0")
        self.assertIn("VLLM_OFFLINE_CONCURRENCY", str(cm.exception))

    def test_negative_ws_decode_concurrency_fails_at_boot(self) -> None:
        with self.assertRaises(ValueError) as cm:
            self._settings_with(VLLM_WS_DECODE_CONCURRENCY="-1")
        self.assertIn("VLLM_WS_DECODE_CONCURRENCY", str(cm.exception))

    def test_zero_ws_decode_concurrency_fails_at_boot(self) -> None:
        with self.assertRaises(ValueError):
            self._settings_with(VLLM_WS_DECODE_CONCURRENCY="0")

    def test_negative_offline_concurrency_fails_at_boot(self) -> None:
        with self.assertRaises(ValueError):
            self._settings_with(VLLM_OFFLINE_CONCURRENCY="-4")

    def test_non_integer_fails_at_boot(self) -> None:
        with self.assertRaises(ValueError):
            self._settings_with(VLLM_OFFLINE_CONCURRENCY="lots")

    def test_valid_values_still_load(self) -> None:
        s = self._settings_with(
            VLLM_OFFLINE_CONCURRENCY="1", VLLM_WS_DECODE_CONCURRENCY="16"
        )
        self.assertEqual(s.VLLM_OFFLINE_CONCURRENCY, 1)
        self.assertEqual(s.VLLM_WS_DECODE_CONCURRENCY, 16)


if __name__ == "__main__":
    unittest.main()
