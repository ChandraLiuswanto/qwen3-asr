import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.core import model_paths
from app.core.config import settings
from app.infrastructure.model_utils import (
    find_huggingface_snapshot_dir,
    resolve_model_path,
)
from app.services.asr.model_capabilities import get_slugged_assets


class SluggedAssetsTest(unittest.TestCase):
    def test_support_models_expose_stable_slugs(self) -> None:
        slugs = {asset.slug: asset.model_id for asset in get_slugged_assets()}

        self.assertEqual(
            slugs,
            {
                "VAD": "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                "PUNC": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                "PUNC_REALTIME": (
                    "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
                ),
                "CAMPP_DIARIZATION": "iic/speech_campplus_speaker-diarization_common",
                "CAMPP_SV": "damo/speech_campplus_sv_zh-cn_16k-common",
                "CAMPP_TRANSFORMER": (
                    "damo/speech_campplus-transformer_scl_zh-cn_16k-common"
                ),
            },
        )

    def test_every_slug_is_unique(self) -> None:
        slugs = [asset.slug for asset in get_slugged_assets()]

        self.assertEqual(len(slugs), len(set(slugs)))


def _make_model_dir(root: str, name: str) -> str:
    path = Path(root) / name
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    return str(path)


class OverrideEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        model_paths.reset_override_cache()
        self.addCleanup(model_paths.reset_override_cache)

    def test_no_overrides_yields_empty_map(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(model_paths.get_model_path_overrides(), {})

    def test_valid_override_maps_slug_to_model_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = _make_model_dir(temp_dir, "vad")
            with mock.patch.dict(os.environ, {"MODEL_PATH_VAD": model_dir}, clear=True):
                overrides = model_paths.get_model_path_overrides()

            self.assertEqual(
                overrides,
                {"damo/speech_fsmn_vad_zh-cn-16k-common-pytorch": Path(model_dir).resolve()},
            )

    def test_models_json_key_becomes_slug(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = _make_model_dir(temp_dir, "qwen")
            with mock.patch.dict(
                os.environ, {"MODEL_PATH_QWEN3_ASR_1_7B": model_dir}, clear=True
            ):
                overrides = model_paths.get_model_path_overrides()

            self.assertEqual(overrides, {"Qwen/Qwen3-ASR-1.7B": Path(model_dir).resolve()})

    def test_forced_aligner_has_one_shared_slug(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = _make_model_dir(temp_dir, "aligner")
            with mock.patch.dict(
                os.environ, {"MODEL_PATH_FORCED_ALIGNER": model_dir}, clear=True
            ):
                overrides = model_paths.get_model_path_overrides()

            self.assertEqual(
                overrides, {"Qwen/Qwen3-ForcedAligner-0.6B": Path(model_dir).resolve()}
            )

    def test_empty_value_is_treated_as_unset(self) -> None:
        with mock.patch.dict(os.environ, {"MODEL_PATH_VAD": "   "}, clear=True):
            self.assertEqual(model_paths.get_model_path_overrides(), {})

    def test_missing_path_raises_naming_var_and_path(self) -> None:
        with mock.patch.dict(
            os.environ, {"MODEL_PATH_VAD": "/nonexistent/vad"}, clear=True
        ):
            with self.assertRaises(ValueError) as ctx:
                model_paths.get_model_path_overrides()

        self.assertIn("MODEL_PATH_VAD", str(ctx.exception))
        self.assertIn("/nonexistent/vad", str(ctx.exception))

    def test_file_instead_of_directory_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "weights.bin"
            file_path.write_bytes(b"x")
            with mock.patch.dict(
                os.environ, {"MODEL_PATH_VAD": str(file_path)}, clear=True
            ):
                with self.assertRaises(ValueError) as ctx:
                    model_paths.get_model_path_overrides()

        self.assertIn("not a directory", str(ctx.exception))

    def test_empty_directory_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            empty = Path(temp_dir) / "empty"
            empty.mkdir()
            with mock.patch.dict(os.environ, {"MODEL_PATH_VAD": str(empty)}, clear=True):
                with self.assertRaises(ValueError) as ctx:
                    model_paths.get_model_path_overrides()

        self.assertIn("empty", str(ctx.exception))

    def test_unknown_slug_raises_and_lists_valid_slugs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = _make_model_dir(temp_dir, "whatever")
            with mock.patch.dict(
                os.environ, {"MODEL_PATH_NOT_A_MODEL": model_dir}, clear=True
            ):
                with self.assertRaises(ValueError) as ctx:
                    model_paths.get_model_path_overrides()

        message = str(ctx.exception)
        self.assertIn("MODEL_PATH_NOT_A_MODEL", message)
        self.assertIn("MODEL_PATH_VAD", message)

    def test_two_slugs_one_model_id_different_paths_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first = _make_model_dir(temp_dir, "a")
            second = _make_model_dir(temp_dir, "b")
            registry = {"SLUG_A": "same/model", "SLUG_B": "same/model"}
            with mock.patch.object(
                model_paths, "build_slug_registry", return_value=(registry, {})
            ):
                with mock.patch.dict(
                    os.environ,
                    {"MODEL_PATH_SLUG_A": first, "MODEL_PATH_SLUG_B": second},
                    clear=True,
                ):
                    with self.assertRaises(ValueError) as ctx:
                        model_paths.get_model_path_overrides()

        self.assertIn("same/model", str(ctx.exception))

    def test_expands_user_and_env_vars(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = _make_model_dir(temp_dir, "vad")
            with mock.patch.dict(
                os.environ,
                {"MODEL_ROOT": temp_dir, "MODEL_PATH_VAD": "$MODEL_ROOT/vad"},
                clear=True,
            ):
                overrides = model_paths.get_model_path_overrides()

            self.assertEqual(
                overrides["damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"],
                Path(model_dir).resolve(),
            )


class SlugRegistryTest(unittest.TestCase):
    def test_slug_derivation_from_models_json_keys(self) -> None:
        registry, _ = model_paths.build_slug_registry()

        self.assertEqual(registry["QWEN3_ASR_1_7B"], "Qwen/Qwen3-ASR-1.7B")
        self.assertEqual(registry["QWEN3_ASR_0_6B"], "Qwen/Qwen3-ASR-0.6B")
        self.assertEqual(
            registry["PARAFORMER_LARGE"],
            "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        )

    def test_slugify_rules(self) -> None:
        self.assertEqual(model_paths._slugify("qwen3-asr-1.7b"), "QWEN3_ASR_1_7B")
        self.assertEqual(model_paths._slugify("paraformer-large"), "PARAFORMER_LARGE")

    def test_entry_with_both_offline_and_realtime_gets_suffixed_slugs(self) -> None:
        entries = {
            "dual-model": {
                "name": "Dual",
                "engine": "qwen3",
                "models": {"offline": "org/dual-offline", "realtime": "org/dual-realtime"},
            }
        }
        registry, _ = model_paths._registry_from_entries(entries)

        self.assertEqual(registry["DUAL_MODEL_OFFLINE"], "org/dual-offline")
        self.assertEqual(registry["DUAL_MODEL_REALTIME"], "org/dual-realtime")
        self.assertNotIn("DUAL_MODEL", registry)

    def test_new_models_json_entry_needs_no_code_change(self) -> None:
        entries = {
            "brand-new-model-2.0": {
                "name": "New",
                "engine": "qwen3",
                "models": {"offline": "org/brand-new"},
            }
        }
        registry, _ = model_paths._registry_from_entries(entries)

        self.assertEqual(registry["BRAND_NEW_MODEL_2_0"], "org/brand-new")

    def test_conflicting_aligners_mark_slug_ambiguous(self) -> None:
        entries = {
            "a": {
                "name": "A",
                "engine": "qwen3",
                "models": {"offline": "org/a"},
                "extra_kwargs": {"forced_aligner_path": "org/aligner-one"},
            },
            "b": {
                "name": "B",
                "engine": "qwen3",
                "models": {"offline": "org/b"},
                "extra_kwargs": {"forced_aligner_path": "org/aligner-two"},
            },
        }
        _, ambiguous = model_paths._registry_from_entries(entries)

        self.assertEqual(ambiguous["FORCED_ALIGNER"], {"org/aligner-one", "org/aligner-two"})


_VAD_ID = "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"


class ResolutionWithOverridesTest(unittest.TestCase):
    def setUp(self) -> None:
        model_paths.reset_override_cache()
        self.addCleanup(model_paths.reset_override_cache)

    def test_modelscope_override_wins_over_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = _make_model_dir(temp_dir, "vad")
            with mock.patch.dict(os.environ, {"MODEL_PATH_VAD": model_dir}, clear=True):
                resolved = resolve_model_path(_VAD_ID)

            self.assertEqual(resolved, str(Path(model_dir).resolve()))

    def test_without_override_modelscope_resolution_is_unchanged(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            resolved = resolve_model_path(_VAD_ID)

        # No override and (in a clean test env) no cache entry: falls through to
        # the bare id exactly as it does today.
        self.assertIn(resolved, {_VAD_ID, str(Path(settings.MODELSCOPE_PATH) / _VAD_ID)})

    def test_huggingface_override_returns_flat_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = _make_model_dir(temp_dir, "qwen")
            with mock.patch.dict(
                os.environ, {"MODEL_PATH_QWEN3_ASR_1_7B": model_dir}, clear=True
            ):
                resolved = find_huggingface_snapshot_dir("Qwen/Qwen3-ASR-1.7B")

            self.assertEqual(resolved, Path(model_dir).resolve())
