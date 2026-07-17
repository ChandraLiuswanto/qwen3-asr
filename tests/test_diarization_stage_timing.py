"""Structure-only tests for _enable_stage_timing (diarization stage profiling).

The instrumentation is gated by DIARIZATION_STAGE_TIMINGS and meaningless on
CPU; these tests only verify the patching structure, delegation, and logging
prefix using a fake pipeline object (no real modelscope pipeline).
"""

import unittest
from unittest import mock

import app.utils.speaker_diarizer as sd


class FakePipeline:
    """Minimal stand-in with the three modelscope pipeline stages."""

    def __init__(self):
        self.calls = []

    def preprocess(self, *args, **kwargs):
        self.calls.append(("preprocess", args, kwargs))
        return "pre-result"

    def forward(self, *args, **kwargs):
        self.calls.append(("forward", args, kwargs))
        return "fwd-result"

    def postprocess(self, *args, **kwargs):
        self.calls.append(("postprocess", args, kwargs))
        return "post-result"


class EnableStageTimingTest(unittest.TestCase):
    def _capture_logs(self):
        messages = []
        handler_id = sd.logger.add(
            lambda m: messages.append(str(m)), level="INFO"
        )
        self.addCleanup(sd.logger.remove, handler_id)
        return messages

    def test_env_off_returns_instance_unpatched(self):
        fake = FakePipeline()
        for value in ("", "false", "TRUE_ISH"):
            with mock.patch.dict(
                "os.environ", {"DIARIZATION_STAGE_TIMINGS": value}
            ):
                result = sd._enable_stage_timing(fake)
            self.assertIs(result, fake)
            # No instance-level overrides: stages still resolve to the class.
            self.assertNotIn("preprocess", vars(fake))
            self.assertNotIn("forward", vars(fake))
            self.assertNotIn("postprocess", vars(fake))

    def test_env_on_wraps_delegates_and_logs_prefix(self):
        fake = FakePipeline()
        with mock.patch.dict(
            "os.environ", {"DIARIZATION_STAGE_TIMINGS": "true"}
        ):
            result = sd._enable_stage_timing(fake)
        self.assertIs(result, fake)

        messages = self._capture_logs()

        # Wrapped methods delegate: call through and return underlying result.
        self.assertEqual(fake.preprocess("in", key=1), "pre-result")
        self.assertEqual(fake.forward([[0, 1, "wav"]]), "fwd-result")
        self.assertEqual(fake.postprocess("emb"), "post-result")
        self.assertEqual(
            [c[0] for c in fake.calls], ["preprocess", "forward", "postprocess"]
        )
        self.assertEqual(fake.calls[0], ("preprocess", ("in",), {"key": 1}))

        # One [diarization-profile] INFO line per wrapped stage.
        profile_lines = [m for m in messages if "[diarization-profile]" in m]
        self.assertEqual(len(profile_lines), 3)
        for stage in ("preprocess", "forward", "postprocess"):
            self.assertTrue(
                any(stage in line for line in profile_lines),
                f"no profile line for stage {stage}",
            )

    def test_env_off_after_env_change_does_not_recheck_per_call(self):
        """Env is read once at patch time: patched instance keeps logging
        even if the env var is cleared afterwards."""
        fake = FakePipeline()
        with mock.patch.dict(
            "os.environ", {"DIARIZATION_STAGE_TIMINGS": "true"}
        ):
            sd._enable_stage_timing(fake)

        messages = self._capture_logs()
        with mock.patch.dict("os.environ", {"DIARIZATION_STAGE_TIMINGS": ""}):
            self.assertEqual(fake.forward([]), "fwd-result")
        self.assertTrue(any("[diarization-profile]" in m for m in messages))


if __name__ == "__main__":
    unittest.main()
