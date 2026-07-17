# Per-Model Path Overrides via `.env`

**Date:** 2026-07-17
**Status:** Approved design, pending implementation plan

## Problem

Model storage location is not configurable. Models resolve to two hardcoded cache
roots:

- **ModelScope** — `MODELSCOPE_PATH`, hardcoded at `app/core/config.py:35` to
  `~/.cache/modelscope/hub/models`. Read directly in five places:
  `manager.py:126,132`, `model_loader.py:215`, `download_models.py:52,109,182`,
  `infrastructure/model_utils.py:89`.
- **HuggingFace** — `HF_HOME` / `HF_HUB_CACHE`, set only inside `Dockerfile.gpu:15`
  and `Dockerfile.cpu:6`.

`docker-compose.yml` bind-mounts `./models/modelscope` and `./models/huggingface`
onto those container paths; both are hardcoded there too.

An operator who keeps model weights on a separate disk, or shares one weight
directory across deployments, has no supported way to say so. The deployment will
grow to many models, so the mechanism must not require a code change per model.

## Goals

1. Point any individual model at an explicit directory from `.env`.
2. Adding a new model costs one `.env` line and **zero code changes**.
3. Work identically for a local `python start.py` run and `docker compose up`.
4. A misconfigured override fails at boot, loudly, rather than silently loading
   different weights.

## Non-Goals

- Changing the default cache layout. With no override set, behavior is byte-for-byte
  what it is today.
- Per-model *revision* pinning (already handled by `ModelAsset.revision`).
- Making Docker discover host paths automatically — impossible; see Risks.

## Decisions

Settled during brainstorming:

| Decision | Choice | Rationale |
|---|---|---|
| Granularity | Per-model, not one root | Operator requirement: different models on different disks. |
| Path meaning | Direct model directory | `MODEL_PATH_VAD=/mnt/disk/vad` loads that folder as-is; the model id is **not** appended. What you point at is what loads. |
| Bad path | Fail at boot | Mirrors `_positive_int_from_env` (`config.py:143`). An override is a statement of intent; ignoring it loads weights the operator did not ask for. |
| Unknown var | Fail at boot | A typo'd `MODEL_PATH_FOO` that is silently ignored is the exact failure mode being designed against. |
| Scope | Docker and local | One mechanism, both paths. |

## Architecture

### The chokepoint

All ModelScope resolution funnels through `resolve_model_path(model_id)`
(`app/infrastructure/model_utils.py:85`); all HuggingFace resolution through
`find_huggingface_snapshot_dir(model_ref_or_path)` (same file, line 47). Overrides
are applied at these two functions, so every existing caller inherits them with no
plumbing changes.

`find_huggingface_snapshot_dir` already returns a direct path when one exists
(lines 48–50), so a flat override directory integrates with its contract as-is.

### New module: `app/core/model_paths.py`

Owns one job: turn the environment into a validated `{model_id: Path}` map.

```
build_overrides(registry: dict[str, str]) -> dict[str, Path]
```

- Scans `os.environ` for the `MODEL_PATH_` prefix.
- Resolves each var's slug against the registry to a model id.
- Validates: path exists, is a directory, is non-empty. Otherwise `raise ValueError`
  naming both the variable and the path.
- Unknown slug → `raise ValueError` listing the valid slugs.
- Expands `~` and `$VARS` via the existing `_expand_path` helper.

Built once at boot, cached in a module global. `resolve_model_path` and
`find_huggingface_snapshot_dir` consult it before touching any cache.

### The slug registry

Model ids (`Qwen/Qwen3-ASR-1.7B`) make unusable env names
(`MODEL_PATH_QWEN_QWEN3_ASR_1_7B`). Slugs come from two sources instead:

**1. `models.json` keys** — uppercased, `-` and `.` → `_`. This is the rule that
makes goal #2 hold: a new `models.json` entry gets its override for free.

| `.env` var | model id | source |
|---|---|---|
| `MODEL_PATH_QWEN3_ASR_1_7B` | `Qwen/Qwen3-ASR-1.7B` | `models.json` key `qwen3-asr-1.7b` |
| `MODEL_PATH_QWEN3_ASR_0_6B` | `Qwen/Qwen3-ASR-0.6B` | `models.json` key `qwen3-asr-0.6b` |
| `MODEL_PATH_PARAFORMER_LARGE` | `iic/speech_paraformer-large_asr_nat-...-online` | `models.json` key `paraformer-large` |

Forced aligners are declared per-entry in `extra_kwargs.forced_aligner_path`, so they
take the entry slug plus a suffix: `MODEL_PATH_QWEN3_ASR_1_7B_FORCED_ALIGNER`.

**2. A new explicit `slug` field on `ModelAsset`** (`model_capabilities.py:17`) for
the support models that are not in `models.json`:

| `.env` var | model id |
|---|---|
| `MODEL_PATH_VAD` | `damo/speech_fsmn_vad_zh-cn-16k-common-pytorch` |
| `MODEL_PATH_PUNC` | `iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch` |
| `MODEL_PATH_PUNC_REALTIME` | `iic/punc_ct-transformer_...-vad_realtime-...` |
| `MODEL_PATH_CAMPP_DIARIZATION` | `iic/speech_campplus_speaker-diarization_common` |
| `MODEL_PATH_CAMPP_SV` | `damo/speech_campplus_sv_zh-cn_16k-common` |
| `MODEL_PATH_CAMPP_TRANSFORMER` | `damo/speech_campplus-transformer_scl_zh-cn_16k-common` |

The slug is an **explicit field, not derived from `description`**. Descriptions are
human prose (`"CAM++ Diarization"`) — deriving env var names from them would produce
awkward slugs and would silently rename an operator's variable if someone reworded a
string.

### Resolution order

Per model, first hit wins:

1. **Explicit override** — `MODEL_PATH_<SLUG>`, loaded as-is.
2. **ModelScope cache** — `MODELSCOPE_PATH / model_id` (today's behavior).
3. **HuggingFace cache** — snapshot lookup (today's behavior).
4. **Bare model id** — runtime downloads it (today's behavior).

### Boot validation

`app/bootstrap.py:ensure_models_downloaded` runs before model load. Override
validation is invoked at its start, before `check_all_models()`, so a bad path fails
before any download work begins. `start.py` reaches this via `run_cli_preflight`.

### Download and integrity interaction

`check_all_models()` (`download_models.py:74`) must **skip overridden models**. Two
reasons: the operator has already supplied the weights, and the HF integrity patterns
(`snapshots/*/config.json`, `model_loader.py`) assume the HF cache's `snapshots/`
layout — a flat override directory has `config.json` at its root and would fail the
check despite being valid. Boot validation already proved the directory exists and is
non-empty; that is the guarantee overridden models get.

### CAM++ config rewrite

`fix_camplusplus_config()` (`download_models.py:99`) rewrites the diarization
`configuration.json` to point at local paths for offline use, using
`get_camplusplus_replacement_paths(cache_dir)` (`model_capabilities.py:167`), which
today string-formats `MODELSCOPE_PATH` against three hardcoded ids.

That signature changes: drop the `cache_dir` parameter and build each replacement via
`resolve_model_path(model_id)`, so overrides are honored. Without this, overriding
CAM++ SV, CAM++ Transformer, or VAD breaks diarization in offline mode — the rewritten
config would point at a cache path that holds nothing.

### Docker

- Add `env_file: .env` to `docker-compose.yml` and `docker-compose-cpu.yml`. Compose
  cannot glob env vars into `environment:`, so enumerating `MODEL_PATH_*` per model
  would reintroduce the per-model edit that goal #2 forbids.
- The host directory must also be bind-mounted **at the same path inside the
  container**, since the container cannot see host paths otherwise. Documented in
  `.env.example` with a worked example.

### `.env.example`

Extend the existing "Model cache behavior" section with the `MODEL_PATH_*` mechanism,
the slug rule, the full table of currently valid slugs, and the Docker mount caveat.

## Testing

Project convention: `unittest`, not pytest (`DEVICE=cpu .venv/bin/python -m unittest
discover -s tests`). New file `tests/test_model_path_overrides.py`, using `tmp_path`
dirs and `unittest.mock.patch.dict(os.environ, ...)`:

1. No `MODEL_PATH_*` set → resolution is unchanged from today (the regression guard on
   the "defaults untouched" promise).
2. Override set to a valid dir → `resolve_model_path` returns it, cache is not consulted.
3. Override for an HF model → `find_huggingface_snapshot_dir` returns the flat dir.
4. Path missing → boot raises `ValueError` naming the var and the path.
5. Path exists but is a file, and path is an empty dir → both raise.
6. Unknown `MODEL_PATH_FOO` → raises, message lists valid slugs.
7. Slug derivation: `qwen3-asr-1.7b` → `QWEN3_ASR_1_7B`.
8. A synthetic `models.json` entry gets a working override with no code change (the
   goal #2 guard).
9. Overridden model is excluded from `check_all_models()`.
10. `get_camplusplus_replacement_paths()` reflects an override.

## Risks

- **Docker mount mismatch.** `.env` alone cannot make a host path visible in the
  container; a bind-mount is still required and cannot be automated. Mitigated by
  boot validation — the container fails immediately with the unreadable path named,
  rather than silently downloading a second copy.
- **`ModelAsset` gains a required field.** All existing construction sites
  (`model_capabilities.py:29,40,54,61,71,84,93,138,156`) must be updated. The two
  dynamic Qwen sites (138, 156) derive their slug from the `models.json` entry key.
- **`get_camplusplus_replacement_paths` is a signature change.** Callers:
  `download_models.py:120` and any test referencing it.

## Rollback

Unset every `MODEL_PATH_*` variable. Resolution returns to today's cache-only
behavior with all new validation still in place — the override map is empty and every
lookup falls through to step 2.
