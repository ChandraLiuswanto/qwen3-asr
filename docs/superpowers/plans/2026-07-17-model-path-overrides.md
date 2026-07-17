# Per-Model Path Overrides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an operator point any individual model at an explicit directory from `.env` via `MODEL_PATH_<SLUG>`, with zero code changes needed to add new models.

**Architecture:** A new `app/core/model_paths.py` turns `MODEL_PATH_*` environment variables into a validated `{model_id: Path}` map, built lazily on first consult and cached. Slugs come from `models.json` keys (so new models are free) plus an optional `slug` field on `ModelAsset` for the six support models that aren't in `models.json`. Every site that resolves a model path consults the map first — there are nine of them, not one.

**Tech Stack:** Python 3, stdlib only (`os`, `json`, `pathlib`, `threading`, `re`). Tests are `unittest`.

**Spec:** `docs/superpowers/specs/2026-07-17-model-path-env-design.md` (revision 2)

## Global Constraints

- **Tests are `unittest`, NOT pytest.** Run: `DEVICE=cpu .venv/bin/python -m unittest discover -s tests`. `pytest` is not a dependency; do not add it. Do not use `tmp_path` (a pytest fixture) — use `tempfile.TemporaryDirectory`.
- **The `.venv` may exist but be empty.** If every test fails at import, run `./scripts/sync_cpu_env.sh` first. That is an environment problem, not a code problem.
- **Defaults must not change.** With no `MODEL_PATH_*` set, behavior is byte-for-byte what it is today. Every task preserves this.
- **Env var prefix is exactly `MODEL_PATH_`.** Slug rule: `models.json` key uppercased, every run of non-alphanumeric characters → `_`. `qwen3-asr-1.7b` → `QWEN3_ASR_1_7B`.
- **Override paths are direct model directories.** The model id is never appended.
- **Fail loudly:** bad path, unknown slug, or ambiguous slug → `ValueError` naming the variable.
- **Never import `app.services.*` at module level from `app/core/model_paths.py`.** `model_capabilities.py:10` imports `get_model_manager` at module scope, which reaches `app.infrastructure` via `engines/funasr.py:10`. A module-level import creates the cycle `model_utils → model_paths → model_capabilities → manager → engines → funasr → app.infrastructure → model_utils`. Use function-local imports (the established pattern at `model_loader.py:251-259`).

## Spec coverage: the nine resolution sites

Each site from the spec's "Resolution sites" table, and the task that handles it:

| Site | Task | How |
|---|---|---|
| `model_utils.py:85` `resolve_model_path` | 3 | Consults the override map first |
| `model_utils.py:47` `find_huggingface_snapshot_dir` | 3 | Consults the override map first |
| `model_loader.py:215` `_build_modelscope_spec` | 4 | Overridden models are skipped before the spec is built |
| `model_loader.py:222` `_build_huggingface_spec` | 4 | Same |
| `manager.py:126,132` | 6 | `_declared_model_exists` checks the override |
| `download_models.py:52` `_get_cache_path` | 4, 6 | Left cache-only *by design*: both callers are guarded — `check_all_models` skips overridden models (4), export refuses to run with overrides (6). The function is never reached for an overridden model. |
| `download_models.py:109` CAM++ config location | 5 | Resolved through `resolve_model_path` |
| `download_models.py:182` download target | 4 | Left cache-only *by design*: the spec predicted "downloads next to, not into, override", but Task 4 removes overridden models from `missing`, so they never reach the download loop. Line 182 remains a display string. |

The two sites left cache-only are deliberate and load-bearing on Tasks 4 and 6. If a
later change lets an overridden model reach either one, it will silently use the cache.

## Deviations from the spec (deliberate, noted for the reviewer)

1. **`ModelAsset.slug` is `Optional[str] = None`, not required.** The spec's Risks section called for a required field touching all nine construction sites. Only six assets need slugs; the Qwen sites (`model_capabilities.py:138,156`) and the Paraformer realtime asset (line 71) are already addressable via `models.json` keys, and giving them a second slug would create two slugs for one model id. Optional is less churn and avoids the collision.
2. **`model_paths` reaches `ModelAsset` slugs via a function-local import**, rather than not importing `model_capabilities` at all as the spec's Layering note says. The spec's intent was avoiding the import cycle; a deferred import achieves that while keeping `ModelAsset` the single source of truth for support-model ids.

3. **The `FORCED_ALIGNER` slug is sourced from `models.json`'s `extra_kwargs.forced_aligner_path`**, not from the dynamic `ModelAsset` at `model_capabilities.py:156` as the spec's Risks section implies. Equivalent today (both Qwen entries declare the same aligner id, `models.json:46,79`), and reading it from `models.json` keeps `_registry_from_entries` a pure function of the config file — which is what makes the registry unit-testable without the manager stack. If a future entry declares a different aligner, `_record` marks the slug ambiguous and any override of it raises.

Two smaller judgment calls, flagged rather than buried:

- **Empty value means unset.** `MODEL_PATH_VAD=` is skipped, not treated as a bad path. Mirrors `API_KEY` (`config.py:97`) and keeps a commented `.env` template usable. It is a deliberate exception to fail-loud: an empty value expresses no intent.
- **`_slugify` generalizes the spec's "`-`/`.` → `_`" rule** to "any run of non-alphanumerics → `_`, strip edges". A superset; it cannot change the slug of any current key.
- **An entry declaring zero models** produces no slug and is silently unaddressable. No current entry does this. Not worth a guard, but do not be surprised by it.

---

### Task 1: Add the `slug` field to `ModelAsset`

**Files:**
- Modify: `app/services/asr/model_capabilities.py:17-25` (dataclass), `:28-99` (six assets), append new accessor
- Test: `tests/test_model_path_overrides.py` (create)

**Interfaces:**
- Produces: `ModelAsset.slug: Optional[str]`; `get_slugged_assets() -> list[ModelAsset]` returning only assets with a non-None slug.

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_path_overrides.py`:

```python
import unittest

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_model_path_overrides -v`
Expected: FAIL — `ImportError: cannot import name 'get_slugged_assets'`

- [ ] **Step 3: Add the field**

In `app/services/asr/model_capabilities.py`, add `slug` to the dataclass (after `description`, before `revision`, so existing positional construction is unaffected — all current sites use keywords, but keep the safe order anyway):

```python
@dataclass(frozen=True)
class ModelAsset:
    source: ModelSource
    model_id: str
    description: str
    slug: Optional[str] = None
    revision: Optional[str] = None
    required_patterns: tuple[str, ...] = ()
    alternative_required_patterns: tuple[tuple[str, ...], ...] = ()
    min_total_size_bytes: int = 0
```

- [ ] **Step 4: Set slugs on the six support assets**

Add exactly one `slug=` line to each. `_VAD_ASSETS` (line ~28): `slug="VAD"`. In `_DIARIZATION_ASSETS`: `slug="CAMPP_DIARIZATION"` on `iic/speech_campplus_speaker-diarization_common`, `slug="CAMPP_SV"` on `damo/speech_campplus_sv_zh-cn_16k-common`, `slug="CAMPP_TRANSFORMER"` on `damo/speech_campplus-transformer_scl_zh-cn_16k-common`. In `_REALTIME_PARAFORMER_ASSETS`: `slug="PUNC"` on the `settings.PUNC_MODEL` asset **only**. On `_REALTIME_PUNC_ASSET`: `slug="PUNC_REALTIME"`.

**Do NOT** add a slug to the Paraformer realtime asset (`model_capabilities.py:71`, model id `iic/speech_paraformer-large_asr_nat-...-online`) — it is already addressable as `MODEL_PATH_PARAFORMER_LARGE` via its `models.json` key, and a second slug for the same model id is exactly what Task 2's ambiguity check rejects. Same for the dynamic Qwen assets at lines ~138 and ~156.

- [ ] **Step 5: Add the accessor**

Append to `app/services/asr/model_capabilities.py`:

```python
def get_slugged_assets() -> list[ModelAsset]:
    """Return support-model assets that expose a MODEL_PATH_<SLUG> override name.

    Only models absent from models.json carry a slug; entries declared there are
    addressed by their models.json key instead.
    """
    return [
        asset
        for asset in (
            *_VAD_ASSETS,
            *_DIARIZATION_ASSETS,
            *_REALTIME_PARAFORMER_ASSETS,
            _REALTIME_PUNC_ASSET,
        )
        if asset.slug is not None
    ]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_model_path_overrides -v`
Expected: PASS (2 tests)

- [ ] **Step 7: Run the full suite (no regressions)**

Run: `DEVICE=cpu .venv/bin/python -m unittest discover -s tests`
Expected: same pass/fail set as before this task.

- [ ] **Step 8: Commit**

```bash
git add app/services/asr/model_capabilities.py tests/test_model_path_overrides.py
git commit -m "feat: add override slugs to support model assets"
```

---

### Task 2: The override registry and validator

**Files:**
- Create: `app/core/model_paths.py`
- Test: `tests/test_model_path_overrides.py` (extend)

**Interfaces:**
- Consumes: `get_slugged_assets()` from Task 1.
- Produces:
  - `get_model_path_overrides() -> dict[str, Path]` — model id → resolved dir. Lazy, cached, thread-safe.
  - `get_override(model_id: str) -> Optional[Path]`
  - `is_overridden(model_id: str) -> bool`
  - `build_slug_registry() -> tuple[dict[str, str], dict[str, set[str]]]` — (slug → model_id, ambiguous slug → ids)
  - `reset_override_cache() -> None` — test hook.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_model_path_overrides.py`:

```python
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.core import model_paths


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_model_path_overrides -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.core.model_paths'`

- [ ] **Step 3: Write the module**

Create `app/core/model_paths.py`:

```python
# -*- coding: utf-8 -*-
"""Per-model path overrides driven by MODEL_PATH_<SLUG> environment variables.

An operator sets MODEL_PATH_VAD=/mnt/disk/vad to load that directory as-is; the
model id is never appended. Resolution order for any model is: explicit override,
then the ModelScope cache, then the HuggingFace cache, then the bare model id.

Overrides are validated on first consult and cached. Validation is deliberately
fatal: an override is a statement of intent, so a bad path must not silently
degrade into loading different weights from a cache.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Optional

from app.core.config import settings

_PREFIX = "MODEL_PATH_"

_cache: Optional[dict[str, Path]] = None
_cache_lock = threading.Lock()


def _slugify(key: str) -> str:
    """Turn a models.json key into an env var slug: qwen3-asr-1.7b -> QWEN3_ASR_1_7B."""
    return re.sub(r"[^A-Z0-9]+", "_", key.upper()).strip("_")


def _expand_path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def _record(
    registry: dict[str, str],
    ambiguous: dict[str, set[str]],
    slug: str,
    model_id: str,
) -> None:
    """Register slug -> model_id, tracking slugs that resolve to more than one id."""
    if slug in ambiguous:
        ambiguous[slug].add(model_id)
        return
    existing = registry.get(slug)
    if existing is not None and existing != model_id:
        ambiguous[slug] = {existing, model_id}
        del registry[slug]
        return
    registry[slug] = model_id


def _registry_from_entries(
    entries: dict[str, dict],
) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Build the slug registry from models.json entry dicts."""
    registry: dict[str, str] = {}
    ambiguous: dict[str, set[str]] = {}

    for key, entry in entries.items():
        base = _slugify(key)
        declared = {
            kind: model_id
            for kind, model_id in (entry.get("models") or {}).items()
            if model_id
        }
        # One declared model takes the bare key slug; an entry declaring both an
        # offline and a realtime model needs one slug per model.
        if len(declared) == 1:
            _record(registry, ambiguous, base, next(iter(declared.values())))
        else:
            for kind, model_id in declared.items():
                _record(registry, ambiguous, f"{base}_{kind.upper()}", model_id)

        aligner = str(
            (entry.get("extra_kwargs") or {}).get("forced_aligner_path") or ""
        ).strip()
        if aligner:
            # Every entry currently declares the same aligner, so it gets one
            # shared slug rather than a per-entry one.
            _record(registry, ambiguous, "FORCED_ALIGNER", aligner)

    return registry, ambiguous


def build_slug_registry() -> tuple[dict[str, str], dict[str, set[str]]]:
    """Return (slug -> model_id, ambiguous slug -> ids) for every addressable model."""
    with open(settings.models_config_path, "r", encoding="utf-8") as handle:
        entries = json.load(handle).get("models", {})

    registry, ambiguous = _registry_from_entries(entries)

    # Imported here, not at module scope: model_capabilities imports the model
    # manager, which reaches app.infrastructure via engines/funasr.py, which
    # imports this module's consumer. A module-level import would be a cycle.
    from app.services.asr.model_capabilities import get_slugged_assets

    for asset in get_slugged_assets():
        _record(registry, ambiguous, asset.slug, asset.model_id)

    return registry, ambiguous


def _validated_dir(var: str, raw: str) -> Path:
    path = _expand_path(raw)
    if not path.exists():
        raise ValueError(f"{var}={raw!r} does not exist")
    if not path.is_dir():
        raise ValueError(f"{var}={raw!r} is not a directory")
    if not any(path.iterdir()):
        raise ValueError(f"{var}={raw!r} is an empty directory")
    return path.resolve()


def _build_overrides() -> dict[str, Path]:
    registry, ambiguous = build_slug_registry()
    overrides: dict[str, Path] = {}
    source_var: dict[str, str] = {}

    for var, raw_value in sorted(os.environ.items()):
        if not var.startswith(_PREFIX):
            continue
        raw = (raw_value or "").strip()
        if not raw:
            continue  # Empty means unset, matching API_KEY handling in config.py.

        slug = var[len(_PREFIX):]
        if slug in ambiguous:
            ids = ", ".join(sorted(ambiguous[slug]))
            raise ValueError(
                f"{var} is ambiguous: slug {slug!r} maps to several models ({ids}). "
                f"Give these models distinct slugs before overriding them."
            )
        if slug not in registry:
            valid = ", ".join(f"{_PREFIX}{name}" for name in sorted(registry))
            raise ValueError(
                f"{var} does not name a known model. Valid variables: {valid}"
            )

        path = _validated_dir(var, raw)
        model_id = registry[slug]
        previous = overrides.get(model_id)
        if previous is not None and previous != path:
            raise ValueError(
                f"{var} and {source_var[model_id]} both override model "
                f"{model_id!r} with different paths ({path} vs {previous})"
            )
        overrides[model_id] = path
        source_var[model_id] = var

    return overrides


def get_model_path_overrides() -> dict[str, Path]:
    """Return the validated {model_id: path} override map, building it once."""
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                _cache = _build_overrides()
    return _cache


def get_override(model_id: Optional[str]) -> Optional[Path]:
    if not model_id:
        return None
    return get_model_path_overrides().get(model_id)


def is_overridden(model_id: Optional[str]) -> bool:
    return get_override(model_id) is not None


def reset_override_cache() -> None:
    """Drop the cached map. For tests and for bootstrap re-validation."""
    global _cache
    with _cache_lock:
        _cache = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_model_path_overrides -v`
Expected: PASS (all tests from Task 1 and Task 2)

- [ ] **Step 5: Verify no import cycle**

Run: `DEVICE=cpu .venv/bin/python -c "from app.core.model_paths import build_slug_registry; r, a = build_slug_registry(); print(sorted(r)); print(a)"`
Expected: prints a list containing `CAMPP_DIARIZATION`, `CAMPP_SV`, `CAMPP_TRANSFORMER`, `FORCED_ALIGNER`, `PARAFORMER_LARGE`, `PUNC`, `PUNC_REALTIME`, `QWEN3_ASR_0_6B`, `QWEN3_ASR_1_7B`, `VAD`, then `{}`. No `ImportError` and no hang.

- [ ] **Step 6: Commit**

```bash
git add app/core/model_paths.py tests/test_model_path_overrides.py
git commit -m "feat: add MODEL_PATH override registry and validation"
```

---

### Task 3: Wire overrides into path resolution

**Files:**
- Modify: `app/infrastructure/model_utils.py:47` (`find_huggingface_snapshot_dir`), `:85` (`resolve_model_path`)
- Test: `tests/test_model_path_overrides.py` (extend)

**Interfaces:**
- Consumes: `get_override` from Task 2.
- Produces: no signature changes. `resolve_model_path` returns the override dir as a string; `find_huggingface_snapshot_dir` returns it as a `Path`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_model_path_overrides.py`:

```python
from app.infrastructure.model_utils import (
    find_huggingface_snapshot_dir,
    resolve_model_path,
)

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
```

Add `from app.core.config import settings` to the test imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_model_path_overrides -v`
Expected: FAIL — `test_modelscope_override_wins_over_cache` and `test_huggingface_override_returns_flat_dir` fail; the override is ignored.

- [ ] **Step 3: Wire `resolve_model_path`**

In `app/infrastructure/model_utils.py`, replace the body of `resolve_model_path` (line 85 onward):

```python
def resolve_model_path(model_id: Optional[str]) -> str:
    if not model_id:
        raise ValueError("model_id is required")

    override = get_override(model_id)
    if override is not None:
        logger.info("Using MODEL_PATH override for %s: %s", model_id, override)
        return str(override)

    local_path = Path(settings.MODELSCOPE_PATH) / model_id

    if local_path.exists() and local_path.is_dir():
        resolved = str(local_path)
        logger.info("Using local ModelScope cache for %s: %s", model_id, resolved)
        return resolved

    logger.warning("ModelScope cache missing for %s; runtime may download it", model_id)
    return model_id
```

- [ ] **Step 4: Wire `find_huggingface_snapshot_dir`**

Insert at the top of `find_huggingface_snapshot_dir` (before the existing `raw_path` check at line 48):

```python
def find_huggingface_snapshot_dir(model_ref_or_path: str) -> Optional[Path]:
    override = get_override(model_ref_or_path)
    if override is not None:
        logger.info(
            "Using MODEL_PATH override for %s: %s", model_ref_or_path, override
        )
        return override

    raw_path = Path(model_ref_or_path).expanduser()
    ...
```

- [ ] **Step 5: Add the import**

`model_paths` imports only `app.core.config`, and `model_utils` is not in its import chain, so this one is safe at module scope. Add to the imports at the top of `app/infrastructure/model_utils.py`:

```python
from app.core.model_paths import get_override
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_model_path_overrides tests.test_huggingface_model_utils -v`
Expected: PASS. `test_huggingface_model_utils` must still pass unchanged — it is the guard that default resolution is untouched.

- [ ] **Step 7: Commit**

```bash
git add app/infrastructure/model_utils.py tests/test_model_path_overrides.py
git commit -m "feat: honor MODEL_PATH overrides in model path resolution"
```

---

### Task 4: Stop overrides from aborting boot (integrity + download checks)

This is the task the spec's revision 2 exists for. `verify_required_models_integrity` (`model_loader.py:301`) is called by `app/main.py:83-86`, which raises `RuntimeError` on any invalid model. Its specs are built from cache paths and match `snapshots/*/config.json`; a flat override dir has `config.json` at its root, so **a correct override would abort startup** without this task.

**Files:**
- Modify: `app/utils/model_loader.py:250-298` (`_build_required_model_integrity_specs`)
- Modify: `app/utils/download_models.py:74-96` (`check_all_models`)
- Test: `tests/test_model_path_overrides.py` (extend)

**Interfaces:**
- Consumes: `is_overridden` from Task 2.
- Produces: no signature changes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_model_path_overrides.py`:

```python
from app.utils.download_models import check_all_models
from app.utils.model_loader import _build_required_model_integrity_specs


class OverridesSkipStartupChecksTest(unittest.TestCase):
    def setUp(self) -> None:
        model_paths.reset_override_cache()
        self.addCleanup(model_paths.reset_override_cache)

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_model_path_overrides -v`
Expected: FAIL — `test_overridden_model_is_not_integrity_checked` finds `VAD` in the specs.

Note on `test_overridden_model_is_not_reported_missing`: it only goes red on a machine whose ModelScope cache lacks VAD. If the cache is already populated (any box that has run the app), `check_model_exists` returns True and VAD is not in `missing` regardless — so the test passes before the change. That is expected, not a mistake; it still guards the behavior afterward. Do not "fix" it by deleting it.

- [ ] **Step 3: Skip overridden models in the integrity specs**

In `app/utils/model_loader.py`, inside `_build_required_model_integrity_specs`, add the import alongside the other function-local imports (after line 258):

```python
    from ..core.model_paths import is_overridden
```

Then guard both loops. The ModelScope loop becomes:

```python
    for asset in get_runtime_required_modelscope_assets(
        include_realtime_punc=True,
    ):
        if is_overridden(asset.model_id):
            logger.info(
                "Skipping integrity check for %s: MODEL_PATH override in use",
                asset.model_id,
            )
            continue
        specs.append(
            _build_modelscope_spec(
                asset.model_id,
                asset.description,
                asset.required_patterns,
                alternative_required_patterns=asset.alternative_required_patterns,
                min_total_size_bytes=asset.min_total_size_bytes,
            )
        )
```

And the HuggingFace loop:

```python
    for asset in get_enabled_qwen_huggingface_assets(
        include_forced_aligner=_should_check_qwen_forced_aligner(
            resolved_device=resolved_device,
            using_cpu_qwen_rust=using_cpu_qwen_rust,
        ),
    ):
        if is_overridden(asset.model_id):
            logger.info(
                "Skipping integrity check for %s: MODEL_PATH override in use",
                asset.model_id,
            )
            continue
        specs.append(
            _build_huggingface_spec(
                asset.model_id,
                asset.description,
                asset.required_patterns,
                alternative_required_patterns=asset.alternative_required_patterns,
                min_total_size_bytes=asset.min_total_size_bytes,
            )
        )
```

The override's guarantee is Task 2's boot validation (exists, is a directory, is non-empty) — weaker than pattern matching, and the accepted cost of the direct-model-dir decision.

- [ ] **Step 4: Skip overridden models in `check_all_models`**

In `app/utils/download_models.py`, add to the imports at the top:

```python
from app.core.model_paths import is_overridden
```

Then guard both loops in `check_all_models` (lines 85-94):

```python
    # Check ModelScope models.
    for asset in ms_assets:
        if is_overridden(asset.model_id):
            continue
        exists, _ = check_model_exists(asset.model_id, source="modelscope")
        if not exists:
            missing.append((asset.model_id, asset.description, "modelscope", asset.revision))

    # Check Hugging Face models. HF assets currently do not use pinned revisions.
    for asset in hf_assets:
        if is_overridden(asset.model_id):
            continue
        exists, _ = check_model_exists(asset.model_id, source="huggingface")
        if not exists:
            missing.append((asset.model_id, asset.description, "huggingface", None))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_model_path_overrides tests.test_model_integrity -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/utils/model_loader.py app/utils/download_models.py tests/test_model_path_overrides.py
git commit -m "fix: exempt overridden models from cache-shaped startup checks"
```

---

### Task 5: CAM++ offline config rewrite

`fix_camplusplus_config()` rewrites the diarization `configuration.json` so offline runs don't reach for modelscope.cn. Two bugs under overrides: the replacement paths are built from `MODELSCOPE_PATH`, and the rewrite *target file* is located in the cache too — so an overridden diarization model's config is never fixed and the function silently returns `False`.

**Files:**
- Modify: `app/services/asr/model_capabilities.py:167-173` (`get_camplusplus_replacement_paths`)
- Modify: `app/utils/download_models.py:99-145` (`fix_camplusplus_config`), `:120` (caller)
- Test: `tests/test_model_path_overrides.py` (extend)

**Interfaces:**
- Consumes: `resolve_model_path` (Task 3), `is_overridden` (Task 2).
- Produces: `get_camplusplus_replacement_paths() -> dict[str, str]` — **the `cache_dir: str` parameter is removed**.

**Decision (operator did not answer; cheap to flip):** the rewrite writes into the model directory, which fails if an override points at a read-only mount. On write failure: warn, but raise when `HF_HUB_OFFLINE=1`, where an unfixed config means diarization breaks at runtime.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_model_path_overrides.py`:

```python
from app.services.asr.model_capabilities import get_camplusplus_replacement_paths

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
        with mock.patch.dict(os.environ, {}, clear=True):
            replacements = get_camplusplus_replacement_paths()

        self.assertIn(_CAMPP_SV_ID, replacements)


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
```

Add `import json` and `from app.utils.download_models import fix_camplusplus_config` to the test imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_model_path_overrides -v`
Expected: FAIL — `TypeError: get_camplusplus_replacement_paths() missing 1 required positional argument: 'cache_dir'`, and `test_readonly_override_raises_when_offline` fails because the raise is swallowed by the function's own `except Exception`. That second failure is the whole point of this task: it is the guard on the dead-code trap described in Step 4.

- [ ] **Step 3: Rewrite `get_camplusplus_replacement_paths`**

Replace it in `app/services/asr/model_capabilities.py`:

```python
def get_camplusplus_replacement_paths() -> dict[str, str]:
    """Return the CAM++ offline replacement map, honoring MODEL_PATH overrides."""
    from app.infrastructure.model_utils import resolve_model_path

    return {
        model_id: resolve_model_path(model_id)
        for model_id in (
            "damo/speech_campplus_sv_zh-cn_16k-common",
            "damo/speech_campplus-transformer_scl_zh-cn_16k-common",
            "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        )
    }
```

The `resolve_model_path` import is function-local: `app.infrastructure.model_utils` imports `app.core.model_paths`, which imports this module inside `build_slug_registry`. A module-level import here would close that loop.

- [ ] **Step 4: Fix the rewrite target and write handling**

**The trap:** the existing function ends in `except Exception as e: print(...); return False` (`download_models.py:143-145`). A `raise RuntimeError` placed inside that `try` is caught by the function's own handler, printed as a warning, and turned into `return False` — dead code. The write must sit **outside** the catch-all. Step 6 removes the second suppression.

In `app/utils/download_models.py`, replace the whole body of `fix_camplusplus_config` after its docstring:

```python
    from app.infrastructure.model_utils import resolve_model_path

    diarization_id = "iic/speech_campplus_speaker-diarization_common"

    try:
        config_file = Path(resolve_model_path(diarization_id)) / "configuration.json"

        if not config_file.exists():
            return False

        # Read the config file.
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Model id to local path replacements.
        replacements = get_camplusplus_replacement_paths()

        # Check whether any replacement is needed.
        modified = False
        if "model" in config:
            for key in ["speaker_model", "change_locator", "vad_model"]:
                if key in config["model"]:
                    old_value = config["model"][key]
                    if old_value in replacements:
                        new_value = replacements[old_value]
                        # Check whether the local path exists.
                        if Path(new_value).exists():
                            config["model"][key] = new_value
                            modified = True

        if not modified:
            return False

    except Exception as e:
        print(f"⚠️  修复 CAM++ 配置文件失败: {e}")
        return False

    # The write sits outside the catch-all above on purpose: an offline rewrite
    # failure must be able to abort startup, and that except Exception would
    # turn the RuntimeError back into a warning.
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    except OSError as e:
        # An override may point at a read-only mount. Offline runs need this
        # rewrite or diarization reaches for modelscope.cn; online runs can
        # still fetch, so they only warn.
        message = (
            f"Cannot rewrite {config_file} for offline use: {e}. "
            f"Make the directory writable, or pre-patch the config."
        )
        if is_huggingface_offline():
            raise RuntimeError(message) from e
        print(f"⚠️  {message}")
        return False

    return True
```

- [ ] **Step 5: Stop `preload_models` from swallowing the offline failure**

`model_loader.py:393-397` wraps the call in `except Exception: pass`, so even a correctly-raised `RuntimeError` dies there. Replace it:

```python
    # Fix CAM++ config files for offline environments.
    try:
        from .download_models import fix_camplusplus_config
        fix_camplusplus_config()
    except RuntimeError:
        # Offline repair failure is fatal: diarization would silently reach for
        # modelscope.cn at request time. See fix_camplusplus_config.
        raise
    except Exception:
        pass  # Other config repair failures should not block startup.
```

- [ ] **Step 6: Fix the caller's now-stale argument**

`download_models.py:120` passed `str(cache_dir)`. The `cache_dir` local (line 109) is gone; confirm no other reference remains.

Run: `grep -rn "get_camplusplus_replacement_paths\|ms_cache_dir\|cache_dir" app/utils/download_models.py app/services/asr/model_capabilities.py`
Expected: no call passes an argument to `get_camplusplus_replacement_paths`; `ms_cache_dir` at line ~182 (a display string in `download_models`) is untouched and still valid.

- [ ] **Step 7: Run tests to verify they pass**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_model_path_overrides tests.test_model_integrity -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add app/services/asr/model_capabilities.py app/utils/download_models.py app/utils/model_loader.py tests/test_model_path_overrides.py
git commit -m "fix: honor overrides in CAM++ offline config rewrite"
```

---

### Task 6: Model API `exists` flags and export refusal

`manager.py:124-134` computes the `exists` flags the models API returns; without this an overridden, working model reports `exists: false`. `--export-dir` copies from cache paths and computes `relative_to(get_huggingface_cache_root())`, which raises `ValueError` for any path outside the cache — so export refuses to run with overrides rather than silently exporting stale weights.

**Files:**
- Modify: `app/services/asr/manager.py:121-134`
- Modify: `app/utils/download_models.py:148-168` (`download_models`, export guard)
- Test: `tests/test_model_path_overrides.py` (extend)

**Interfaces:**
- Consumes: `is_overridden`, `get_override` (Task 2).
- Produces: no signature changes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_model_path_overrides.py`:

```python
from app.services.asr.manager import get_model_manager
from app.utils.download_models import download_models as run_download_models


class DeclaredEntryExistsFlagTest(unittest.TestCase):
    def setUp(self) -> None:
        model_paths.reset_override_cache()
        self.addCleanup(model_paths.reset_override_cache)

    def test_override_marks_offline_model_as_present(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = _make_model_dir(temp_dir, "qwen")
            with mock.patch.dict(
                os.environ, {"MODEL_PATH_QWEN3_ASR_1_7B": model_dir}, clear=True
            ):
                entries = get_model_manager().list_declared_entries()

        entry = next(item for item in entries if item["id"] == "qwen3-asr-1.7b")
        self.assertTrue(entry["offline_model"]["exists"])


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
```

**Do not drop that mock.** Step 2 runs this test *before* the guard exists, so an unmocked run would fall through to `check_all_models()` — and with `clear=True` wiping `HF_HUB_OFFLINE`, `download_models` would either start real multi-GB downloads (`download_models.py:217,242`) or `copytree` every cached model into the temp dir (`:262-300`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_model_path_overrides -v`
Expected: FAIL — `exists` is `False`; export proceeds instead of refusing.

- [ ] **Step 3: Make the `exists` flags override-aware**

In `app/services/asr/manager.py`, add to the module imports:

```python
from ...core.model_paths import get_override
```

Then replace the two path checks inside `list_declared_entries` (lines 124-134):

```python
            if config.offline_model_path:
                offline_path_exists = _declared_model_exists(config.offline_model_path)

            if config.realtime_model_path:
                realtime_path_exists = _declared_model_exists(config.realtime_model_path)
```

And add this helper above the `ModelManager` class (after `_supports_qwen_realtime_on_device`):

```python
def _declared_model_exists(model_id: str) -> bool:
    """Report whether a declared model is present locally.

    Pre-existing quirk preserved: HF-hosted ids (Qwen/...) are checked against the
    ModelScope cache, so they read False unless overridden. Fixing that is out of
    scope here; this only ensures an explicit override reads True.
    """
    override = get_override(model_id)
    if override is not None:
        return True
    return (Path(settings.MODELSCOPE_PATH) / model_id).exists()
```

- [ ] **Step 4: Refuse export when overrides are set**

In `app/utils/download_models.py`, insert at the start of `download_models`, immediately after `import shutil`:

```python
    overrides = get_model_path_overrides()
    if export_dir and overrides:
        print("❌ --export-dir cannot be combined with MODEL_PATH overrides.")
        print("   Export copies from the model caches; these models live elsewhere:")
        for model_id, path in sorted(overrides.items()):
            print(f"     - {model_id}: {path}")
        print("   Unset the MODEL_PATH_* variables to export, or copy them by hand.")
        return False
```

Update the import added in Task 4 to bring in both names:

```python
from app.core.model_paths import get_model_path_overrides, is_overridden
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_model_path_overrides -v`
Expected: PASS

- [ ] **Step 6: Run the full suite**

Run: `DEVICE=cpu .venv/bin/python -m unittest discover -s tests`
Expected: PASS, no regressions.

- [ ] **Step 7: Commit**

```bash
git add app/services/asr/manager.py app/utils/download_models.py tests/test_model_path_overrides.py
git commit -m "feat: report overridden models as present, refuse export with overrides"
```

---

### Task 7: Make `.env` reach every entrypoint, plus boot validation, Docker, and docs

**The premise bug.** `load_dotenv()` is called in exactly one place — `start.py:10-12`. So `python -m app.utils.download_models` (which `bootstrap.py:35` tells operators to run) and a direct `uvicorn app.main:app` **never read `.env` at all**. A feature whose whole premise is "set the path in `.env`" is silently inert there: the CLI would cheerfully re-download models the operator already supplied. Task 2's lazy validation fires on those paths, but with an empty environment it has nothing to validate — validation that never sees the config cannot protect it.

Moving `load_dotenv()` into `app/core/config.py` fixes it for every entrypoint at once, because everything imports `settings`. `load_dotenv` does not override variables already set in the real environment, so Docker's explicit `environment:` entries still win and no current behavior changes.

Validation is lazy (Task 2), so every entrypoint also validates on first resolution — including `WORKERS>1` (`start.py:76-78` skips preflight), which reaches it via `main.py:83-86` → `verify_required_models_integrity` → Task 4's `is_overridden`. This task adds an *early, clean* failure for the common CLI path on top of that, placed **outside** `ensure_models_downloaded`'s `except Exception` (`bootstrap.py:40-42`), which would otherwise downgrade the `ValueError` to a `⚠️` print.

**Files:**
- Modify: `app/core/config.py` (add `load_dotenv`), `start.py:10-12` (drop the now-redundant call)
- Modify: `app/bootstrap.py:9-42`
- Modify: `docker-compose.yml`, `docker-compose-cpu.yml`
- Modify: `.env.example`
- Test: `tests/test_model_path_overrides.py` (extend)

**Interfaces:**
- Consumes: `get_model_path_overrides` (Task 2), `is_overridden` (Task 4).
- Produces: `validate_model_path_overrides() -> None` in `app/bootstrap.py`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_model_path_overrides.py`:

```python
import subprocess
import sys

from app.bootstrap import validate_model_path_overrides
from app.utils.model_loader import _build_required_model_integrity_specs


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


class ValidationFiresWithoutBootstrapTest(unittest.TestCase):
    """Spec test 13: the WORKERS>1 and standalone-CLI paths never call bootstrap."""

    def setUp(self) -> None:
        model_paths.reset_override_cache()
        self.addCleanup(model_paths.reset_override_cache)

    def test_integrity_check_rejects_bad_override(self) -> None:
        # The WORKERS>1 path: start.py skips preflight, main.py's lifespan calls
        # verify_required_models_integrity, which must still refuse to start.
        with mock.patch.dict(
            os.environ, {"MODEL_PATH_VAD": "/nonexistent/vad"}, clear=True
        ):
            with self.assertRaises(ValueError):
                _build_required_model_integrity_specs()

    def test_download_cli_rejects_bad_override(self) -> None:
        with mock.patch.dict(
            os.environ, {"MODEL_PATH_VAD": "/nonexistent/vad"}, clear=True
        ):
            with self.assertRaises(ValueError):
                run_download_models(auto_mode=True)


class DotenvReachesEveryEntrypointTest(unittest.TestCase):
    def test_importing_settings_loads_dotenv(self) -> None:
        # A subprocess with a .env in cwd and nothing in the environment: importing
        # settings alone must make MODEL_PATH_VAD visible. This is what makes the
        # standalone CLI and bare uvicorn honor .env at all.
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = _make_model_dir(temp_dir, "vad")
            (Path(temp_dir) / ".env").write_text(
                f"MODEL_PATH_VAD={model_dir}\n", encoding="utf-8"
            )
            env = {k: v for k, v in os.environ.items() if not k.startswith("MODEL_PATH_")}
            env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)

            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import app.core.config, os; print(os.getenv('MODEL_PATH_VAD'))",
                ],
                cwd=temp_dir,
                env=env,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.stdout.strip(), model_dir, result.stderr)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_model_path_overrides -v`
Expected: FAIL — `ImportError: cannot import name 'validate_model_path_overrides'`. After Step 3 lands, `test_importing_settings_loads_dotenv` still fails (prints `None`) until Step 4.

- [ ] **Step 3: Add early validation to bootstrap**

In `app/bootstrap.py`, add above `ensure_models_downloaded`:

```python
def validate_model_path_overrides() -> None:
    """Validate MODEL_PATH_* overrides, raising on the first bad one.

    Deliberately outside ensure_models_downloaded's try/except: that block turns
    every exception into a warning print, which is the opposite of the fail-loud
    contract these overrides carry.
    """
    from app.core.model_paths import get_model_path_overrides

    get_model_path_overrides()
```

Then call it first in `ensure_models_downloaded`, before the `try`:

```python
def ensure_models_downloaded(interactive: bool) -> bool:
    """Ensure declared deployment models exist locally, downloading if needed."""
    validate_model_path_overrides()

    try:
        from app.infrastructure import is_huggingface_offline
        ...
```

- [ ] **Step 4: Load `.env` from `app/core/config.py`**

At the top of `app/core/config.py`, after the existing imports and before the `Settings` class:

```python
from dotenv import load_dotenv

# Loaded here rather than in start.py alone: `python -m app.utils.download_models`
# and a bare `uvicorn app.main:app` never run start.py, so .env was invisible to
# them. Everything imports settings, so this reaches every entrypoint. Real
# environment variables already set take precedence (load_dotenv does not
# override), so Docker's explicit `environment:` entries still win.
load_dotenv()
```

Then remove the now-redundant `load_dotenv()` and its import from `start.py:10-12`. Confirm `python-dotenv` is already a dependency (it is — `start.py:10` imports it today).

Run: `grep -n "dotenv" pyproject.toml environments/*/pyproject.toml 2>/dev/null; grep -rn "dotenv" start.py app/core/config.py`
Expected: `load_dotenv` now appears only in `app/core/config.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_model_path_overrides -v`
Expected: PASS, including `test_importing_settings_loads_dotenv`.

- [ ] **Step 6: Add `env_file` to both compose files**

`MODEL_PATH_*` cannot be enumerated under `environment:` — Compose has no globbing, and a per-model entry is the per-model code change goal #2 forbids. `required: false` matters: a bare `env_file: .env` is a hard error when the file is absent, and both files run fine without one today. Needs Compose v2.24+.

In `docker-compose.yml` and `docker-compose-cpu.yml`, add above `environment:`:

```yaml
    env_file:
      - path: .env
        required: false
```

- [ ] **Step 7: Verify compose still parses without a `.env`**

Run: `docker compose -f docker-compose.yml config >/dev/null && docker compose -f docker-compose-cpu.yml config >/dev/null && echo OK`
Expected: `OK`. If it errors with `env_file` unsupported syntax, the local Compose is below v2.24 — report that rather than reverting to `env_file: .env`, which breaks the no-`.env` case.

- [ ] **Step 8: Document in `.env.example`**

Replace the "Model cache behavior" section at the end of `.env.example`:

```bash
# -----------------------------------------------------------------------------
# Model cache behavior.
# Default: download missing models at startup.
# -----------------------------------------------------------------------------
# HF_HUB_OFFLINE=1
# HF_ENDPOINT=https://hf-mirror.com

# -----------------------------------------------------------------------------
# Per-model paths.
#
# Point any single model at a directory you control. The path is the model
# directory itself -- the model id is NOT appended:
#
#   MODEL_PATH_VAD=/mnt/models/fsmn-vad     loads /mnt/models/fsmn-vad
#
# A model with an override is never downloaded, and is skipped by the startup
# integrity check (the check assumes cache layout; your directory is flat).
# A path that does not exist, is not a directory, or is empty aborts startup.
# An unrecognized MODEL_PATH_* variable also aborts startup, so typos surface
# immediately instead of silently loading cached weights.
#
# ASR models are named after their key in app/services/asr/models.json,
# uppercased with - and . replaced by _. Adding a model there needs no code
# change: qwen3-asr-1.7b -> MODEL_PATH_QWEN3_ASR_1_7B
#
# MODEL_PATH_QWEN3_ASR_1_7B=
# MODEL_PATH_QWEN3_ASR_0_6B=
# MODEL_PATH_PARAFORMER_LARGE=
# MODEL_PATH_FORCED_ALIGNER=
#
# Support models:
# MODEL_PATH_VAD=
# MODEL_PATH_PUNC=
# MODEL_PATH_PUNC_REALTIME=
# MODEL_PATH_CAMPP_DIARIZATION=
# MODEL_PATH_CAMPP_SV=
# MODEL_PATH_CAMPP_TRANSFORMER=
#
# Docker: an override alone is not enough -- the container cannot see host paths.
# Bind-mount the directory at the SAME path inside the container, e.g. for
# MODEL_PATH_VAD=/mnt/models/fsmn-vad add to docker-compose.yml:
#
#   volumes:
#     - /mnt/models/fsmn-vad:/mnt/models/fsmn-vad:ro
#
# Note: MODEL_PATH_CAMPP_DIARIZATION needs a writable mount (drop :ro) when
# HF_HUB_OFFLINE=1 -- its configuration.json is rewritten for offline use.
#
# --export-dir cannot be combined with overrides; it copies from the caches.
# -----------------------------------------------------------------------------
```

- [ ] **Step 9: Run the full suite**

Run: `DEVICE=cpu .venv/bin/python -m unittest discover -s tests`
Expected: PASS

- [ ] **Step 10: Verify defaults are genuinely untouched**

Run: `DEVICE=cpu env -u MODEL_PATH_VAD .venv/bin/python -c "
from app.core.model_paths import get_model_path_overrides
from app.infrastructure.model_utils import resolve_model_path
assert get_model_path_overrides() == {}, 'expected no overrides'
print(resolve_model_path('damo/speech_fsmn_vad_zh-cn-16k-common-pytorch'))
"`
Expected: prints the cache path or the bare model id — whichever it printed before this branch. No exception.

- [ ] **Step 11: Commit**

```bash
git add app/core/config.py start.py app/bootstrap.py docker-compose.yml docker-compose-cpu.yml .env.example tests/test_model_path_overrides.py
git commit -m "feat: load .env everywhere, validate overrides at boot, document"
```

---

## Verification

After Task 7, confirm end-to-end with a real override rather than trusting the tests:

```bash
mkdir -p /tmp/fake-vad && echo '{}' > /tmp/fake-vad/config.json
DEVICE=cpu MODEL_PATH_VAD=/tmp/fake-vad .venv/bin/python -c "
from app.infrastructure.model_utils import resolve_model_path
print(resolve_model_path('damo/speech_fsmn_vad_zh-cn-16k-common-pytorch'))
"
```
Expected: `/tmp/fake-vad`

```bash
DEVICE=cpu MODEL_PATH_TYPO=/tmp/fake-vad .venv/bin/python -c "
from app.bootstrap import validate_model_path_overrides
validate_model_path_overrides()
"
```
Expected: `ValueError` naming `MODEL_PATH_TYPO` and listing valid variables.

## Open decision (cheap to flip)

Task 5 warns when the CAM++ config rewrite fails on a read-only override mount, but raises when `HF_HUB_OFFLINE=1`. If the operator prefers a plain warning in every case, delete the `if is_huggingface_offline(): raise` branch in `fix_camplusplus_config`.
