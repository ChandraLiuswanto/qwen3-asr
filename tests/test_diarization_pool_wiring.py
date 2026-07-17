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
