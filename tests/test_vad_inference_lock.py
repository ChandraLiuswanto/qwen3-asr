from __future__ import annotations

import threading
import time
import unittest
from unittest import mock


class _RecordingVadModel:
    def __init__(self) -> None:
        self._mu = threading.Lock()
        self.active = 0
        self.max_active = 0

    def generate(self, input, cache):
        with self._mu:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.01)
        with self._mu:
            self.active -= 1
        return [{"value": [[0, 1000]]}]


class VadInferenceLockTest(unittest.TestCase):
    def test_concurrent_get_vad_segments_serialized(self) -> None:
        from app.utils.audio_splitter import AudioSplitter

        model = _RecordingVadModel()
        splitter = AudioSplitter(device="cpu")
        with mock.patch(
            "app.services.asr.engines.get_global_vad_model", return_value=model
        ):
            threads = [
                threading.Thread(target=splitter.get_vad_segments, args=("/fake.wav",))
                for _ in range(8)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(model.max_active, 1)
