"""Boot MUST fail loudly when the diarization worker pool cannot warm up —
this replaces the old swallow-and-degrade (log + continue) that produced the
silent lazy-build stall (bd qwen3-asr-9nk)."""

import unittest
from unittest import mock

from app.utils import model_loader


def _result_dict():
    return {"speaker_diarization_model": {"loaded": False, "error": None}}


class DiarizationBootTest(unittest.TestCase):
    def test_success_marks_loaded(self):
        result = _result_dict()
        progress = mock.Mock()
        with mock.patch(
            "app.utils.speaker_diarizer.warmup_diarization_pool", return_value=4
        ):
            model_loader._preload_diarization_pool(result, progress)
        self.assertTrue(result["speaker_diarization_model"]["loaded"])

    def test_failure_records_error_and_reraises(self):
        result = _result_dict()
        progress = mock.Mock()
        boom = RuntimeError("BrokenProcessPool: see worker stderr")
        with mock.patch(
            "app.utils.speaker_diarizer.warmup_diarization_pool", side_effect=boom
        ):
            with self.assertRaises(RuntimeError):
                model_loader._preload_diarization_pool(result, progress)
        self.assertIn(
            "BrokenProcessPool", result["speaker_diarization_model"]["error"]
        )


if __name__ == "__main__":
    unittest.main()
