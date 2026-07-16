# tests/test_stage_timings.py
from __future__ import annotations

import logging
import unittest
from unittest import mock


class StageTimingLogTest(unittest.TestCase):
    def test_stage_timing_log_line_shape(self) -> None:
        # Unit-test the formatting helper directly; the full pipeline
        # needs real models and is H100-only.
        from app.services.asr.engines.base import format_stage_timings

        line = format_stage_timings(
            task_id="t1",
            total_s=12.345,
            diarization_s=3.0,
            vad_split_s=0.0,
            inference_s=8.5,
            alignment_s=0.0,
            segments=5,
            batches=2,
        )
        self.assertIn("ASR_STAGE_TIMINGS", line)
        self.assertIn("task_id=t1", line)
        self.assertIn("diarization_s=3.000", line)
        self.assertIn("vad_split_s=0.000", line)
        self.assertIn("inference_s=8.500", line)
        self.assertIn("batches=2", line)
        # Unmarked lines are successes, so existing consumers keep working.
        self.assertIn("status=success", line)

    def test_error_status_marker_preserves_existing_field_contract(self) -> None:
        from app.services.asr.engines.base import format_stage_timings

        kwargs = dict(
            task_id="t2",
            total_s=1.5,
            diarization_s=0.25,
            vad_split_s=0.5,
            inference_s=0.75,
            alignment_s=0.0,
            segments=0,
            batches=0,
        )
        ok = format_stage_timings(**kwargs)
        err = format_stage_timings(**kwargs, status="error")

        self.assertIn("status=error", err)
        # The runbook parses this line: every pre-existing field must keep its
        # exact text and position, with status appended last.
        self.assertEqual(err, ok.replace("status=success", "status=error"))
        self.assertTrue(err.startswith("ASR_STAGE_TIMINGS task_id=t2 total_s=1.500 "))
        self.assertEqual(err.split()[-1], "status=error")
        self.assertEqual(err.split()[-2], "batches=0")
