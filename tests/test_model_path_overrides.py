import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.bootstrap import validate_model_path_overrides
from app.core import model_paths
from app.core.config import settings
from app.infrastructure.model_utils import (
    find_huggingface_snapshot_dir,
    resolve_model_path,
)
from app.services.asr.manager import ModelManager
from app.services.asr.model_capabilities import (
    get_camplusplus_replacement_paths,
    get_slugged_assets,
)
from app.utils.download_models import (
    check_all_models,
    fix_camplusplus_config,
)
from app.utils.download_models import download_models as run_download_models
from app.utils.model_loader import _build_required_model_integrity_specs


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
        # The ModelScope cache genuinely contains the VAD model, so this only
        # passes if the override is consulted BEFORE the cache lookup.
        with tempfile.TemporaryDirectory() as cache_dir, tempfile.TemporaryDirectory() as override_root:
            cache_model_dir = _make_model_dir(cache_dir, _VAD_ID)
            model_dir = _make_model_dir(override_root, "vad")
            self.assertTrue(Path(cache_model_dir).is_dir())

            with mock.patch.object(settings, "MODELSCOPE_PATH", cache_dir):
                with mock.patch.dict(
                    os.environ, {"MODEL_PATH_VAD": model_dir}, clear=True
                ):
                    resolved = resolve_model_path(_VAD_ID)

            self.assertEqual(resolved, str(Path(model_dir).resolve()))
            self.assertNotEqual(resolved, cache_model_dir)

    def test_without_override_modelscope_resolution_is_unchanged(self) -> None:
        # With no override set, the ModelScope cache entry is returned exactly
        # as it is today.
        with tempfile.TemporaryDirectory() as cache_dir:
            cache_model_dir = _make_model_dir(cache_dir, _VAD_ID)

            with mock.patch.object(settings, "MODELSCOPE_PATH", cache_dir):
                with mock.patch.dict(os.environ, {}, clear=True):
                    resolved = resolve_model_path(_VAD_ID)

            self.assertEqual(resolved, cache_model_dir)

    def test_huggingface_override_returns_flat_dir(self) -> None:
        # The HF cache genuinely contains a resolvable snapshot for the model,
        # so this only passes if the override is consulted BEFORE the cache.
        with tempfile.TemporaryDirectory() as cache_dir, tempfile.TemporaryDirectory() as override_root:
            base_dir = Path(cache_dir) / "models--Qwen--Qwen3-ASR-1.7B"
            snapshot_name = "abc123def456"
            snapshot_dir = base_dir / "snapshots" / snapshot_name
            snapshot_dir.mkdir(parents=True)
            (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")
            refs_dir = base_dir / "refs"
            refs_dir.mkdir(parents=True)
            (refs_dir / "main").write_text(snapshot_name, encoding="utf-8")

            model_dir = _make_model_dir(override_root, "qwen")

            with mock.patch.dict(
                os.environ,
                {
                    "MODEL_PATH_QWEN3_ASR_1_7B": model_dir,
                    "HF_HUB_CACHE": cache_dir,
                },
                clear=True,
            ):
                resolved = find_huggingface_snapshot_dir("Qwen/Qwen3-ASR-1.7B")

            self.assertEqual(resolved, Path(model_dir).resolve())
            self.assertNotEqual(resolved, snapshot_dir.resolve())


class OverridesSkipStartupChecksTest(unittest.TestCase):
    def setUp(self) -> None:
        model_paths.reset_override_cache()
        self.addCleanup(model_paths.reset_override_cache)
        # Both checks route through the runtime model plan, which on a CPU box
        # without the Rust extension resolves no Qwen model and raises. Pretend
        # the extension is present so the plan is buildable under DEVICE=cpu.
        patcher = mock.patch(
            "app.services.asr.qwenasr_rust.is_qwenasr_rust_available",
            return_value=True,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_overridden_model_is_not_integrity_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = _make_model_dir(temp_dir, "vad")
            with mock.patch.dict(os.environ, {"MODEL_PATH_VAD": model_dir}, clear=True):
                specs = _build_required_model_integrity_specs()

        descriptions = [spec.description for spec in specs]
        self.assertNotIn("VAD", descriptions)

    def test_non_overridden_models_are_still_integrity_checked(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            specs = _build_required_model_integrity_specs()

        self.assertIn("VAD", [spec.description for spec in specs])

    def test_overridden_model_is_not_reported_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = _make_model_dir(temp_dir, "vad")
            with mock.patch.dict(os.environ, {"MODEL_PATH_VAD": model_dir}, clear=True):
                missing_ids = [item[0] for item in check_all_models()]

        self.assertNotIn(_VAD_ID, missing_ids)

    def test_non_overridden_models_are_still_reported_missing(self) -> None:
        # Forces every model to look absent from the cache so the assertion does
        # not depend on what this machine has downloaded. Without this, an
        # inverted guard (skipping the non-overridden models) would go unnoticed.
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = _make_model_dir(temp_dir, "vad")
            with mock.patch.dict(os.environ, {"MODEL_PATH_VAD": model_dir}, clear=True):
                with mock.patch(
                    "app.utils.download_models.check_model_exists",
                    return_value=(False, ""),
                ):
                    missing_ids = [item[0] for item in check_all_models()]

        self.assertNotIn(_VAD_ID, missing_ids)
        self.assertIn("iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch", missing_ids)


_CAMPP_SV_ID = "damo/speech_campplus_sv_zh-cn_16k-common"


class CamppReplacementPathsTest(unittest.TestCase):
    def setUp(self) -> None:
        model_paths.reset_override_cache()
        self.addCleanup(model_paths.reset_override_cache)

    def test_replacement_paths_reflect_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = _make_model_dir(temp_dir, "campp-sv")
            with mock.patch.dict(
                os.environ, {"MODEL_PATH_CAMPP_SV": model_dir}, clear=True
            ):
                replacements = get_camplusplus_replacement_paths()

            self.assertEqual(replacements[_CAMPP_SV_ID], str(Path(model_dir).resolve()))

    def test_replacement_paths_use_cache_without_override(self) -> None:
        # With no override set, the map must resolve to the ModelScope cache
        # entry. Asserting the VALUE (not just the hardcoded key) is what makes
        # this fail if resolution ever stops consulting the cache.
        with tempfile.TemporaryDirectory() as cache_dir:
            cache_model_dir = _make_model_dir(cache_dir, _CAMPP_SV_ID)

            with mock.patch.object(settings, "MODELSCOPE_PATH", cache_dir):
                with mock.patch.dict(os.environ, {}, clear=True):
                    replacements = get_camplusplus_replacement_paths()

            self.assertEqual(replacements[_CAMPP_SV_ID], cache_model_dir)


class CamppRewriteTargetTest(unittest.TestCase):
    """Guards spec test 11b and the fail-loud contract on a read-only override."""

    def setUp(self) -> None:
        model_paths.reset_override_cache()
        self.addCleanup(model_paths.reset_override_cache)

    def _overridden_campp(self, temp_dir: str) -> dict[str, str]:
        """Build a diarization override whose config needs a real replacement."""
        diar_dir = Path(temp_dir) / "diar"
        diar_dir.mkdir()
        (diar_dir / "configuration.json").write_text(
            json.dumps({"model": {"speaker_model": _CAMPP_SV_ID}}), encoding="utf-8"
        )
        sv_dir = _make_model_dir(temp_dir, "campp-sv")
        return {
            "MODEL_PATH_CAMPP_DIARIZATION": str(diar_dir),
            "MODEL_PATH_CAMPP_SV": sv_dir,
        }

    def test_rewrite_targets_the_overridden_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = self._overridden_campp(temp_dir)
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertTrue(fix_camplusplus_config())

            written = json.loads(
                (Path(env["MODEL_PATH_CAMPP_DIARIZATION"]) / "configuration.json")
                .read_text(encoding="utf-8")
            )

        self.assertEqual(
            written["model"]["speaker_model"],
            str(Path(env["MODEL_PATH_CAMPP_SV"]).resolve()),
        )

    def test_readonly_override_raises_when_offline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = self._overridden_campp(temp_dir)
            env["HF_HUB_OFFLINE"] = "1"
            with mock.patch.dict(os.environ, env, clear=True):
                with mock.patch(
                    "app.utils.download_models.json.dump",
                    side_effect=OSError("read-only file system"),
                ):
                    with self.assertRaises(RuntimeError) as ctx:
                        fix_camplusplus_config()

        self.assertIn("read-only file system", str(ctx.exception))

    def test_readonly_override_only_warns_when_online(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = self._overridden_campp(temp_dir)
            with mock.patch.dict(os.environ, env, clear=True):
                with mock.patch(
                    "app.utils.download_models.json.dump",
                    side_effect=OSError("read-only file system"),
                ):
                    self.assertFalse(fix_camplusplus_config())


class DeclaredEntryExistsFlagTest(unittest.TestCase):
    def setUp(self) -> None:
        model_paths.reset_override_cache()
        self.addCleanup(model_paths.reset_override_cache)

    def _entries(self, env: dict, cache_dir: str) -> dict:
        # Picking a default model needs a runnable Qwen build, which a plain CPU
        # box without the Rust extension does not have; the constructor would
        # raise before reaching the flags under test. A fresh instance is used
        # rather than the process singleton so this test cannot leak into others.
        with mock.patch(
            "app.services.asr.qwenasr_rust.is_qwenasr_rust_available",
            return_value=True,
        ):
            manager = ModelManager()
        # MODELSCOPE_PATH is pinned at an empty temp dir so the flags depend on
        # the override alone, not on whatever this machine happens to have cached.
        with mock.patch.object(settings, "MODELSCOPE_PATH", cache_dir):
            with mock.patch.dict(os.environ, env, clear=True):
                entries = manager.list_declared_entries()
        return {item["id"]: item for item in entries}

    def test_override_marks_offline_model_as_present(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = _make_model_dir(temp_dir, "qwen")
            cache_dir = os.path.join(temp_dir, "cache")
            os.makedirs(cache_dir)
            entries = self._entries(
                {"MODEL_PATH_QWEN3_ASR_1_7B": model_dir}, cache_dir
            )

        self.assertTrue(entries["qwen3-asr-1.7b"]["offline_model"]["exists"])
        # The un-overridden sibling still reads False, so the True above is
        # caused by the override rather than by the flag being hardcoded.
        self.assertFalse(entries["qwen3-asr-0.6b"]["offline_model"]["exists"])

    def test_override_marks_realtime_model_as_present(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = _make_model_dir(temp_dir, "paraformer")
            cache_dir = os.path.join(temp_dir, "cache")
            os.makedirs(cache_dir)
            entries = self._entries(
                {"MODEL_PATH_PARAFORMER_LARGE": model_dir}, cache_dir
            )

        self.assertTrue(entries["paraformer-large"]["realtime_model"]["exists"])

    def test_without_override_cache_presence_still_drives_the_flag(self) -> None:
        realtime_id = (
            "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
        )
        with tempfile.TemporaryDirectory() as cache_dir:
            _make_model_dir(cache_dir, realtime_id)
            entries = self._entries({}, cache_dir)

        self.assertTrue(entries["paraformer-large"]["realtime_model"]["exists"])
        self.assertFalse(entries["qwen3-asr-1.7b"]["offline_model"]["exists"])


class ExportRefusesOverridesTest(unittest.TestCase):
    def setUp(self) -> None:
        model_paths.reset_override_cache()
        self.addCleanup(model_paths.reset_override_cache)

    def test_export_refuses_when_overrides_are_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = _make_model_dir(temp_dir, "vad")
            export_dir = Path(temp_dir) / "export"
            with mock.patch.dict(os.environ, {"MODEL_PATH_VAD": model_dir}, clear=True):
                # check_all_models is mocked so this test can never reach a real
                # snapshot_download or a multi-GB copytree. The guard under test
                # sits BEFORE this call; if it regresses, the test fails fast and
                # offline rather than hammering the network.
                with mock.patch(
                    "app.utils.download_models.check_all_models", return_value=[]
                ):
                    result = run_download_models(
                        auto_mode=True, export_dir=str(export_dir)
                    )

        self.assertFalse(result)
        self.assertFalse(export_dir.exists())

    def _assert_reaches_check_all_models(self, **kwargs) -> None:
        """Assert download_models gets past the guard and into check_all_models.

        check_all_models is stubbed to raise a sentinel, so execution stops the
        instant the guard is cleared. Nothing downloads, nothing is copied, and
        the real CAM++ config on this machine is never rewritten.
        """

        class _Reached(Exception):
            pass

        with mock.patch(
            "app.utils.download_models.check_all_models", side_effect=_Reached
        ):
            with self.assertRaises(_Reached):
                run_download_models(auto_mode=True, **kwargs)

    def test_export_without_overrides_is_not_refused(self) -> None:
        # The guard is conditional on overrides, not on export_dir alone.
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir) / "export"
            with mock.patch.dict(os.environ, {}, clear=True):
                self._assert_reaches_check_all_models(export_dir=str(export_dir))

    def test_overrides_do_not_block_a_plain_download(self) -> None:
        # No export_dir means no cache-relative copying, so overrides are fine.
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = _make_model_dir(temp_dir, "vad")
            with mock.patch.dict(os.environ, {"MODEL_PATH_VAD": model_dir}, clear=True):
                self._assert_reaches_check_all_models()


class BootstrapValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        model_paths.reset_override_cache()
        self.addCleanup(model_paths.reset_override_cache)

    def test_bad_override_raises_rather_than_being_swallowed(self) -> None:
        with mock.patch.dict(
            os.environ, {"MODEL_PATH_VAD": "/nonexistent/vad"}, clear=True
        ):
            with self.assertRaises(ValueError):
                validate_model_path_overrides()

    def test_valid_overrides_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = _make_model_dir(temp_dir, "vad")
            with mock.patch.dict(os.environ, {"MODEL_PATH_VAD": model_dir}, clear=True):
                validate_model_path_overrides()  # must not raise

    def test_ensure_models_downloaded_does_not_swallow_bad_override(self) -> None:
        from app.bootstrap import ensure_models_downloaded

        with mock.patch.dict(
            os.environ, {"MODEL_PATH_VAD": "/nonexistent/vad"}, clear=True
        ):
            with self.assertRaises(ValueError):
                ensure_models_downloaded(interactive=False)


class ValidationFiresWithoutBootstrapTest(unittest.TestCase):
    """Spec test 13: the WORKERS>1 and standalone-CLI paths never call bootstrap."""

    def setUp(self) -> None:
        model_paths.reset_override_cache()
        self.addCleanup(model_paths.reset_override_cache)

    def test_integrity_check_rejects_bad_override(self) -> None:
        # The WORKERS>1 path: start.py skips preflight, main.py's lifespan calls
        # verify_required_models_integrity, which must still refuse to start.
        with mock.patch(
            "app.services.asr.qwenasr_rust.is_qwenasr_rust_available", return_value=True
        ):
            with mock.patch.dict(
                os.environ, {"MODEL_PATH_VAD": "/nonexistent/vad"}, clear=True
            ):
                with self.assertRaises(ValueError):
                    _build_required_model_integrity_specs()

    def test_download_cli_rejects_bad_override(self) -> None:
        with mock.patch(
            "app.services.asr.qwenasr_rust.is_qwenasr_rust_available", return_value=True
        ):
            with mock.patch.dict(
                os.environ, {"MODEL_PATH_VAD": "/nonexistent/vad"}, clear=True
            ):
                with self.assertRaises(ValueError):
                    run_download_models(auto_mode=True)


class DotenvReachesEveryEntrypointTest(unittest.TestCase):
    """Guards `load_dotenv()` living in app/core/config.py.

    IMPORTANT — how python-dotenv actually resolves `.env`, and why this test
    is built the way it is:

    `load_dotenv()` -> `find_dotenv()` picks its search root one of two ways.
    If `__main__` has no `__file__` (a REPL, or `python -c ...`), it treats the
    process as interactive and searches upward from **os.getcwd()**. Otherwise
    -- which is every real entrypoint: `python start.py`, `python -m
    app.utils.download_models`, `uvicorn app.main:app` -- it searches upward
    from the directory of the file that *called* `load_dotenv()`, i.e. from
    `dirname(app/core/config.py)` up through `app/` to the repo root. So in
    production, `.env` resolution is **config.py-relative and completely
    independent of cwd**. A `.env` sitting in the cwd is NOT read.

    Do not "simplify" this into a `python -c` subprocess: that silently flips
    find_dotenv onto the interactive/cwd branch and tests a path production
    never takes. Hence: the runner is a real script FILE (so `__main__.__file__`
    exists), and it imports `app` through a symlinked fake repo root so the
    upward walk lands on a temp `.env` we control. `os.path.abspath` does not
    resolve symlinks, so config.py's apparent location -- and therefore the
    walk -- stays inside the temp tree. That keeps the real repo untouched: no
    `.env` is ever written into it.
    """

    def _run_import_in_fake_root(self, temp_dir: str, extra_env: dict) -> str:
        """Import app.core.config from a script file, in a cwd far from the .env.

        Layout:
          <temp>/fake_root/.env      <- the "repo" .env; the ONLY one that may win
          <temp>/fake_root/app       -> symlink to the real app package
          <temp>/cwd/.env            <- decoy; production must ignore this
          <temp>/runner.py           <- gives __main__ a __file__

        Returns the subprocess's stdout (the resolved MODEL_PATH_VAD).
        """
        repo_root = Path(__file__).resolve().parent.parent
        fake_root = Path(temp_dir) / "fake_root"
        fake_root.mkdir(exist_ok=True)
        (fake_root / "app").symlink_to(repo_root / "app")

        runner = Path(temp_dir) / "runner.py"
        runner.write_text(
            "import app.core.config, os\n"
            "print(os.getenv('MODEL_PATH_VAD'))\n",
            encoding="utf-8",
        )

        env = {k: v for k, v in os.environ.items() if not k.startswith("MODEL_PATH_")}
        env["PYTHONPATH"] = str(fake_root)
        env.update(extra_env)

        result = subprocess.run(
            [sys.executable, str(runner)],
            cwd=str(Path(temp_dir) / "cwd"),
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def _write_env_files(self, temp_dir: str, repo_value: str, cwd_value: str) -> None:
        (Path(temp_dir) / "cwd").mkdir()
        (Path(temp_dir) / "cwd" / ".env").write_text(
            f"MODEL_PATH_VAD={cwd_value}\n", encoding="utf-8"
        )
        (Path(temp_dir) / "fake_root").mkdir(exist_ok=True)
        (Path(temp_dir) / "fake_root" / ".env").write_text(
            f"MODEL_PATH_VAD={repo_value}\n", encoding="utf-8"
        )

    def test_importing_settings_loads_dotenv(self) -> None:
        # Importing settings alone -- no start.py -- must make MODEL_PATH_VAD
        # visible. This is what makes the standalone CLI and bare uvicorn honor
        # .env at all. Drop load_dotenv() from config.py and this prints None.
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_value = _make_model_dir(temp_dir, "vad-repo")
            cwd_value = _make_model_dir(temp_dir, "vad-cwd")
            self._write_env_files(temp_dir, repo_value, cwd_value)

            resolved = self._run_import_in_fake_root(temp_dir, {})

        # The repo's .env won; the one in cwd was never read. Both halves matter:
        # equality proves load_dotenv ran, inequality proves it was not cwd-based.
        self.assertEqual(resolved, repo_value)
        self.assertNotEqual(resolved, cwd_value)

    def test_dotenv_in_unrelated_cwd_is_ignored(self) -> None:
        # No .env anywhere on config.py's upward walk, only one in the cwd.
        # Production must resolve nothing -- pinning that cwd cannot inject
        # config. (This test alone has no teeth against load_dotenv removal;
        # test_importing_settings_loads_dotenv is the one that does.)
        with tempfile.TemporaryDirectory() as temp_dir:
            cwd_value = _make_model_dir(temp_dir, "vad-cwd")
            (Path(temp_dir) / "cwd").mkdir()
            (Path(temp_dir) / "cwd" / ".env").write_text(
                f"MODEL_PATH_VAD={cwd_value}\n", encoding="utf-8"
            )

            resolved = self._run_import_in_fake_root(temp_dir, {})

        self.assertEqual(resolved, "None")

    def test_real_environment_wins_over_dotenv(self) -> None:
        # Docker sets `environment:` explicitly; load_dotenv must not override it.
        with tempfile.TemporaryDirectory() as temp_dir:
            from_dotenv = _make_model_dir(temp_dir, "vad-dotenv")
            from_cwd = _make_model_dir(temp_dir, "vad-cwd")
            from_env = _make_model_dir(temp_dir, "vad-env")
            self._write_env_files(temp_dir, from_dotenv, from_cwd)

            resolved = self._run_import_in_fake_root(
                temp_dir, {"MODEL_PATH_VAD": from_env}
            )

        self.assertEqual(resolved, from_env)
