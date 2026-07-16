from __future__ import annotations

import threading
import time
import unittest
from unittest import mock


class _RecordingNormalizer:
    def __init__(self) -> None:
        self._mu = threading.Lock()
        self.active = 0
        self.max_active = 0

    def normalize(self, text: str) -> str:
        with self._mu:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.005)
        with self._mu:
            self.active -= 1
        return text


class ItnThreadSafetyTest(unittest.TestCase):
    def test_single_init_and_serialized_normalize_under_contention(self) -> None:
        import app.utils.text_processing as tp

        constructed: list[_RecordingNormalizer] = []

        class _FakeNormalizerCls:
            def __new__(cls, lang="zh", operator="itn"):
                time.sleep(0.005)  # widen the init race window
                instance = _RecordingNormalizer()
                constructed.append(instance)
                return instance

        fake_wetext = mock.MagicMock()
        fake_wetext.Normalizer = _FakeNormalizerCls

        with mock.patch.dict("sys.modules", {"wetext": fake_wetext}):
            tp._wetext_normalizer = None  # reset the singleton
            try:
                threads = [
                    threading.Thread(target=tp.apply_itn_to_text, args=("一百二十三",))
                    for _ in range(8)
                ]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
            finally:
                tp._wetext_normalizer = None

        self.assertEqual(len(constructed), 1, "normalizer constructed more than once")
        self.assertEqual(constructed[0].max_active, 1, "normalize() not serialized")
