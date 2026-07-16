from __future__ import annotations

import asyncio
import threading
import time
import unittest
from unittest import mock

from app.services.asr.engines import ASRFullResult
from app.services.asr.runtime.router import (
    OfflineASRRequest,
    RuntimeFamily,
    RuntimeRouter,
)


class _StatefulEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.current_audio_path = ""

    def transcribe_long_audio(
        self, *, audio_path: str, **_kwargs: object
    ) -> ASRFullResult:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.current_audio_path = audio_path
        time.sleep(0.01)
        with self._lock:
            result = self.current_audio_path
            self.active -= 1
        return ASRFullResult(text=result, segments=[], duration=0.0)


class RuntimeRouterTest(unittest.IsolatedAsyncioTestCase):
    async def test_vllm_offline_requests_overlap_up_to_semaphore(self) -> None:
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

        engine = _StatefulEngine()
        semaphore = asyncio.Semaphore(4)
        router._resolve_family = lambda _model_id: RuntimeFamily.QWEN_VLLM  # type: ignore[method-assign]
        router._get_shared_engine = lambda _family, _model_id: (  # type: ignore[method-assign]
            engine,
            semaphore,
        )

        requests = [
            OfflineASRRequest(
                model_id="qwen3-asr-test",
                audio_path=f"request-{index}",
            )
            for index in range(8)
        ]
        results = await asyncio.gather(
            *(router.run_offline(request) for request in requests)
        )

        # The router no longer serializes; the semaphore is the only bound.
        self.assertGreater(engine.max_active, 1)
        self.assertLessEqual(engine.max_active, 4)
        self.assertEqual(len(results), 8)
