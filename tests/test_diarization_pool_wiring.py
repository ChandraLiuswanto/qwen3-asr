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

    def test_close_hook_delegates_to_pool(self):
        with mock.patch.object(sd._diarization_pool, "close") as close:
            sd.close_diarization_pool()
        close.assert_called_once_with()

    def test_threaded_engine_pool_is_gone_from_this_module(self):
        self.assertFalse(hasattr(sd, "ThreadedEnginePool"))

    def test_boot_timeout_setting_exists(self):
        from app.core.config import settings
        self.assertGreaterEqual(settings.DIARIZATION_BOOT_TIMEOUT_S, 1)


if __name__ == "__main__":
    unittest.main()
