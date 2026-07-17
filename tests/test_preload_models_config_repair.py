"""Pins the exception contract of `preload_models`' CAM++ config repair block.

`fix_camplusplus_config` raising RuntimeError means the offline config rewrite
failed, which would leave diarization silently reaching for modelscope.cn at
request time. `preload_models` must let that escape. Every other failure mode is
non-fatal and must not block startup.

The half of this contract inside `fix_camplusplus_config` is covered by
`test_readonly_override_raises_when_offline`; this file covers the half that
sits on the actual startup path.
"""

import unittest
from unittest import mock

from app.core import model_paths
from app.utils.model_loader import preload_models


class PreloadModelsConfigRepairTest(unittest.TestCase):
    def setUp(self) -> None:
        model_paths.reset_override_cache()
        self.addCleanup(model_paths.reset_override_cache)

    def _stub_model_loading(self) -> None:
        """Neutralize the real model loading that follows the CAM++ block.

        Every step after the config repair is individually wrapped in its own
        `except Exception` and records the failure into the result dict, so
        stubbing only these four entry points keeps `preload_models` on its
        normal path without touching disk, network, or a GPU.
        """
        # No declared models -> the ASR warmup loop and the paraformer-only
        # punctuation steps are skipped entirely.
        patches = [
            mock.patch(
                "app.services.asr.manager.get_model_manager",
                side_effect=RuntimeError("stubbed out"),
            ),
            mock.patch(
                "app.services.asr.engines.get_global_vad_model",
                return_value=object(),
            ),
            mock.patch(
                "app.utils.speaker_diarizer.warmup_diarization_pool",
                return_value=4,
            ),
            mock.patch("app.utils.text_processing.warmup_itn", return_value=True),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_runtime_error_from_config_repair_propagates(self) -> None:
        # An unfixed config under HF_HUB_OFFLINE is fatal: startup must not
        # swallow it. The repair block sits ahead of all model loading, so a
        # passing run never reaches the stubs; they are here so that a
        # regression fails on the assertion below instead of hanging on a real
        # model load.
        self._stub_model_loading()

        with mock.patch(
            "app.utils.download_models.fix_camplusplus_config",
            side_effect=RuntimeError("read-only config, offline"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                preload_models()

        self.assertIn("read-only config, offline", str(ctx.exception))

    def test_other_errors_from_config_repair_are_swallowed(self) -> None:
        self._stub_model_loading()

        with mock.patch(
            "app.utils.download_models.fix_camplusplus_config",
            side_effect=ValueError("some non-fatal config quirk"),
        ):
            result = preload_models()

        # Startup continued past the failure and produced its normal report.
        self.assertEqual(result["asr_models"], {})
        self.assertTrue(result["speaker_diarization_model"]["loaded"])


if __name__ == "__main__":
    unittest.main()
