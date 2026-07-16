"""Regression tests for Qwen3 WebSocket result de-duplication."""

import asyncio
import json
import unittest
from typing import Callable
from unittest.mock import patch

import numpy as np

from app.services.qwen3_websocket_asr import ConnectionContext, Qwen3ASRService


async def _run_inline(
    func: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> object:
    return func(*args, **kwargs)


class _FakeStreamingState:
    def __init__(self, text: str = "repeated partial"):
        self.last_text = text
        self.last_language = "Chinese"
        self.chunk_count = 0


class _FakeQwenEngine:
    supports_realtime = True
    device = "cpu"

    def init_streaming_state(self, **kwargs: object) -> _FakeStreamingState:
        return _FakeStreamingState()

    def streaming_transcribe(
        self,
        audio: np.ndarray,
        state: _FakeStreamingState,
    ) -> _FakeStreamingState:
        state.chunk_count += 1
        return state

    def finish_streaming_transcribe(
        self,
        state: _FakeStreamingState,
    ) -> _FakeStreamingState:
        return state


class _FakeLease:
    def __init__(self, engine: _FakeQwenEngine):
        self.engine = engine

    async def close(self) -> None:
        return None


class _FakeRuntimeRouter:
    def __init__(self, engine: _FakeQwenEngine):
        self._engine = engine

    async def acquire_engine(self, model_id: str) -> _FakeLease:
        assert model_id == "qwen3-asr-0.6b"
        return _FakeLease(self._engine)

    async def lease_shared_engine(self, model_id: str) -> _FakeLease:
        assert model_id == "qwen3-asr-0.6b"
        return _FakeLease(self._engine)


class _FakeWebSocket:
    def __init__(self, messages: list[dict[str, object]]):
        self._messages = messages
        self.events: list[dict[str, object]] = []
        self.accepted = False

    async def accept(self) -> None:
        self.accepted = True

    async def receive(self) -> dict[str, object]:
        return self._messages.pop(0)

    async def send_json(self, event: dict[str, object]) -> None:
        self.events.append(event)


class Qwen3WebSocketDedupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        asyncio.get_running_loop().slow_callback_duration = 1.0

    async def test_truncate_returns_each_confirmed_segment_once(self) -> None:
        service = Qwen3ASRService()
        engine = _FakeQwenEngine()
        ctx = ConnectionContext(
            params={"enable_inverse_text_normalization": True},
            engine=engine,
            streaming_state=_FakeStreamingState("unique segment"),
        )
        websocket = _FakeWebSocket([])

        with patch(
            "app.services.qwen3_websocket_asr.run_sync",
            new=_run_inline,
        ):
            await service._truncate(websocket, ctx, "task", "silence")

        segment_end = websocket.events[0]
        self.assertEqual(segment_end["type"], "segment_end")
        self.assertEqual(segment_end["result"]["text"], "unique segment")
        self.assertEqual(segment_end["confirmed_texts"], ["unique segment"])

    async def test_repeated_partial_text_is_sent_once(self) -> None:
        engine = _FakeQwenEngine()
        websocket = _FakeWebSocket(
            [
                {
                    "type": "websocket.receive",
                    "text": json.dumps(
                        {
                            "type": "start",
                            "payload": {
                                "format": "pcm",
                                "sample_rate": 16000,
                                "chunk_size_sec": 0.00025,
                            },
                        }
                    ),
                },
                {
                    "type": "websocket.receive",
                    "bytes": np.full(8, 2000, dtype=np.int16).tobytes(),
                },
                {"type": "websocket.receive", "text": json.dumps({"type": "stop"})},
            ]
        )
        service = Qwen3ASRService()

        with (
            patch(
                "app.services.qwen3_websocket_asr.get_runtime_router",
                return_value=_FakeRuntimeRouter(engine),
            ),
            patch(
                "app.services.qwen3_websocket_asr.validate_realtime_model_id",
                return_value="qwen3-asr-0.6b",
            ),
            patch("app.services.qwen3_websocket_asr.Qwen3ASREngine", _FakeQwenEngine),
            patch("app.services.qwen3_websocket_asr.run_sync", new=_run_inline),
        ):
            await service.handle_connection(websocket, "task")

        result_events = [
            event for event in websocket.events if event["type"] == "result"
        ]
        self.assertTrue(websocket.accepted)
        self.assertEqual(len(result_events), 1)
        self.assertEqual(len(result_events[0]["results"]), 1)
        self.assertEqual(result_events[0]["results"][0]["text"], "repeated partial")


if __name__ == "__main__":
    unittest.main()
