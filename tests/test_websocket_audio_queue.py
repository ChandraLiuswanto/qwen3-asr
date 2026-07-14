"""Regression tests for bounded WebSocket audio ingress."""

import asyncio
import json
import unittest
import uuid
from unittest.mock import patch

import numpy as np

from app.services.websocket_asr import AliyunWebSocketASRService, _BoundedAudioQueue


class _FakeEngine:
    supports_realtime = True


class _FakeLease:
    engine = _FakeEngine()

    async def close(self) -> None:
        return None


class _FakeRuntimeRouter:
    async def acquire_engine(self, model_id: str) -> _FakeLease:
        assert model_id == "paraformer-large"
        return _FakeLease()


class _FailingWebSocket:
    def __init__(self, start_message: dict[str, object]):
        self._messages = [
            {"type": "websocket.receive", "text": json.dumps(start_message)}
        ]
        self.sent_texts: list[str] = []
        self.close_calls: list[tuple[int, str]] = []
        self.receive_cancelled = False

    async def receive(self) -> dict[str, object]:
        if self._messages:
            return self._messages.pop(0)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.receive_cancelled = True
            raise
        raise AssertionError("Unreachable")

    async def send_text(self, data: str) -> None:
        self.sent_texts.append(data)

    async def close(self, code: int, reason: str) -> None:
        self.close_calls.append((code, reason))


class BoundedAudioQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_queue_applies_backpressure_without_dropping_audio(self) -> None:
        queue = _BoundedAudioQueue(max_samples=4)
        first = np.array([1, 2, 3, 4], dtype=np.float32)
        second = np.array([5, 6, 7, 8], dtype=np.float32)

        await queue.put(first)
        producer = asyncio.create_task(queue.put(second))
        await asyncio.sleep(0)
        self.assertFalse(producer.done())

        np.testing.assert_array_equal(await queue.get(), first)
        self.assertGreater(await asyncio.wait_for(producer, timeout=0.1), 0)
        np.testing.assert_array_equal(await queue.get(), second)

    async def test_finished_queue_drains_before_returning_none(self) -> None:
        queue = _BoundedAudioQueue(max_samples=4)
        audio = np.array([1, 2], dtype=np.float32)

        await queue.put(audio)
        await queue.finish()

        np.testing.assert_array_equal(await queue.get(), audio)
        self.assertIsNone(await queue.get())

    async def test_rejects_oversized_frame(self) -> None:
        queue = _BoundedAudioQueue(max_samples=4)

        with self.assertRaises(ValueError):
            await queue.put(np.zeros(5, dtype=np.float32))

    async def test_abort_unblocks_a_backpressured_producer(self) -> None:
        queue = _BoundedAudioQueue(max_samples=4)
        await queue.put(np.ones(4, dtype=np.float32))
        producer = asyncio.create_task(queue.put(np.ones(4, dtype=np.float32)))
        await asyncio.sleep(0)

        await queue.abort()

        with self.assertRaises(RuntimeError):
            await producer

    async def test_worker_failure_sends_task_failed_without_waiting_for_more_audio(
        self,
    ) -> None:
        task_id = uuid.uuid4().hex
        websocket = _FailingWebSocket(
            {
                "header": {
                    "message_id": uuid.uuid4().hex,
                    "task_id": task_id,
                    "namespace": "SpeechTranscriber",
                    "name": "StartTranscription",
                },
                "payload": {"format": "pcm", "sample_rate": 16000},
            }
        )
        service = AliyunWebSocketASRService()

        async def fail_worker(*args: object) -> None:
            raise RuntimeError("model inference failed")

        setattr(service, "_run_audio_worker", fail_worker)
        with self.assertLogs("app.services.websocket_asr", level="ERROR"):
            with (
                patch(
                    "app.services.websocket_asr.validate_websocket_token",
                    return_value=(True, ""),
                ),
                patch(
                    "app.services.websocket_asr.get_runtime_router",
                    return_value=_FakeRuntimeRouter(),
                ),
            ):
                await asyncio.wait_for(
                    service.handle_connection(websocket, task_id), timeout=0.2
                )

        event_names = [
            json.loads(data)["header"]["name"] for data in websocket.sent_texts
        ]
        self.assertEqual(
            event_names,
            ["TranscriptionStarted", "TaskFailed"],
        )
        self.assertEqual(websocket.close_calls, [(1011, "Audio processing failed")])
        self.assertTrue(websocket.receive_cancelled)


if __name__ == "__main__":
    unittest.main()
