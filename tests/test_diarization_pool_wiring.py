"""Structure-only tests: the diarization request path checks a pipeline out
of the pool for the duration of the pipeline call — acquire, then call (with
torch.cuda.empty_cache suppressed for exactly that call), then release, in
that order, the release in a finally. The ordering and the suppression window
are asserted below with a shared event log and an empty_cache sentinel.
These use fakes on CPU; they cannot prove GPU-concurrency safety —
scripts/h100/test_offline_mixing.py is the real gate for that.
"""

import os
import subprocess
import sys
import unittest
from unittest import mock

import torch

from app.utils import speaker_diarizer as sd


class _FakePipeline:
    """Records call order AND probes the empty_cache guard mid-call, the way
    funasr does (it calls torch.cuda.empty_cache() after every inference)."""

    def __init__(self, events) -> None:
        self.events = events
        self.calls = []

    def __call__(self, audio_path):
        self.events.append("call")
        self.calls.append(audio_path)
        # funasr-style: empty_cache during the pipeline call. The guard must
        # suppress this (the sentinel installed in setUp must NOT record).
        torch.cuda.empty_cache()
        return {"text": [[0.0, 1.5, 0], [1.5, 3.0, 1]]}


class DiarizationPoolWiringTest(unittest.TestCase):
    def setUp(self) -> None:
        # Sentinel + guard arrangement (same as tests/test_empty_cache_guard):
        # replace torch.cuda.empty_cache with a recording lambda, force a
        # fresh guard install over it, restore everything afterwards.
        self._original_empty_cache = torch.cuda.empty_cache
        self.sentinel_calls = []
        torch.cuda.empty_cache = lambda: self.sentinel_calls.append(True)
        sd._empty_cache_guard_installed = False
        sd._install_empty_cache_guard()
        self.addCleanup(self._restore_empty_cache)

    def _restore_empty_cache(self) -> None:
        torch.cuda.empty_cache = self._original_empty_cache
        sd._empty_cache_guard_installed = False

    def test_diarize_acquires_calls_and_releases(self) -> None:
        events = []
        fake = _FakePipeline(events)

        def fake_acquire():
            events.append("acquire")
            return fake

        def fake_release(pipeline):
            self.assertIs(pipeline, fake)
            events.append("release")

        with mock.patch.object(
            sd._diarization_pool, "acquire", side_effect=fake_acquire
        ) as acq, mock.patch.object(
            sd._diarization_pool, "release", side_effect=fake_release
        ) as rel:
            segments = sd.SpeakerDiarizer().diarize("/tmp/x.wav")

        acq.assert_called_once_with()
        rel.assert_called_once_with(fake)
        # Exact ordering: checkout, then the pipeline call, then check-in.
        self.assertEqual(events, ["acquire", "call", "release"])
        # empty_cache was suppressed DURING the pipeline call...
        self.assertEqual(len(self.sentinel_calls), 0)
        # ...and is NOT suppressed outside diarize.
        torch.cuda.empty_cache()
        self.assertEqual(len(self.sentinel_calls), 1)
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

    def test_factory_installs_guard_and_stage_timing(self) -> None:
        """_build_diarization_pipeline must install the empty_cache guard and
        run the pipeline through _enable_stage_timing (setUp reset the guard
        onto a fresh, unguarded sentinel, so the attribute check below can
        only pass if the factory path itself performed the install)."""
        # Undo setUp's install so the factory has to do it itself.
        torch.cuda.empty_cache = lambda: self.sentinel_calls.append(True)
        sd._empty_cache_guard_installed = False
        self.assertFalse(
            getattr(torch.cuda.empty_cache, "_diarization_guard", False)
        )

        fake_pipeline = object()
        with mock.patch(
            "app.infrastructure.model_utils.resolve_model_path",
            return_value="/models/campplus",
        ), mock.patch.object(
            sd, "_resolve_modelscope_device", return_value="cpu"
        ), mock.patch.object(
            sd, "_create_modelscope_pipeline", return_value=fake_pipeline
        ), mock.patch.object(
            sd, "_enable_batched_sv", side_effect=lambda p, d, max_batch_size: p
        ), mock.patch.object(
            sd, "_enable_stage_timing", side_effect=lambda p: p
        ) as timing_spy:
            built = sd._build_diarization_pipeline()

        self.assertIs(built, fake_pipeline)
        self.assertIs(
            getattr(torch.cuda.empty_cache, "_diarization_guard", False), True
        )
        timing_spy.assert_called_once_with(fake_pipeline)

    def test_pool_sized_from_settings(self) -> None:
        # Subprocess, not in-process: the module-level pool is built from
        # settings at import time, so only a fresh interpreter with the env
        # var set can prove the size is wired to DIARIZATION_POOL_SIZE
        # rather than hardcoded.
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

    def test_global_singleton_accessor_is_gone(self) -> None:
        self.assertFalse(hasattr(sd, "get_global_diarization_pipeline"))

    def test_warmup_builds_all_n(self) -> None:
        with mock.patch.object(sd._diarization_pool, "warmup") as warm:
            n = sd.warmup_diarization_pool()
        warm.assert_called_once_with()
        self.assertEqual(n, sd._diarization_pool.size)


if __name__ == "__main__":
    unittest.main()
