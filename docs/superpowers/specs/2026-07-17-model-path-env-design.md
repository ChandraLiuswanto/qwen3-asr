# Per-Model Path Overrides via `.env`

**Date:** 2026-07-17
**Status:** Approved design, pending implementation plan
**Revision:** 2 — rev 1 claimed `resolve_model_path` was a universal chokepoint. It is
not; see "Resolution sites". Rev 2 also reworks boot validation, the forced-aligner
slug, and the CAM++ rewrite.

## Problem

Model storage location is not configurable. Models resolve to two hardcoded cache
roots:

- **ModelScope** — `MODELSCOPE_PATH`, hardcoded at `app/core/config.py:35` to
  `~/.cache/modelscope/hub/models`. `_load_from_env` (`config.py:81`) never reads it,
  so no env var reaches it today.
- **HuggingFace** — `HF_HOME` / `HF_HUB_CACHE`, set only inside `Dockerfile.gpu:15-16`
  and `Dockerfile.cpu:6-7`.

`docker-compose.yml` bind-mounts `./models/modelscope` and `./models/huggingface`
onto those container paths; both are hardcoded there too.

An operator who keeps model weights on a separate disk, or shares one weight
directory across deployments, has no supported way to say so. The deployment will
grow to many models, so the mechanism must not require a code change per model.

## Goals

1. Point any individual model at an explicit directory from `.env`.
2. Adding a new model costs one `.env` line and **zero code changes**.
3. Work identically for a local `python start.py` run and `docker compose up`.
4. A misconfigured override fails loudly at boot, rather than silently loading
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
| Path meaning | Direct model directory | `MODEL_PATH_VAD=/mnt/disk/vad` loads that folder as-is; the model id is **not** appended. |
| Bad path | Fail loudly | An override is a statement of intent; ignoring it loads weights the operator did not ask for. |
| Unknown var | Fail loudly | A typo'd `MODEL_PATH_FOO` silently ignored is the exact failure mode being designed against. |
| Scope | Docker and local | One mechanism, both paths. |

## Architecture

### Resolution sites

There is **no single chokepoint**. `resolve_model_path` (`app/infrastructure/model_utils.py:85`)
has only five callers — `engines/global_models.py:50,85,119`, `engines/funasr.py:36`,
`utils/speaker_diarizer.py:225`. Every other site builds `MODELSCOPE_PATH / model_id`
itself and would silently bypass overrides. The full set that must become
override-aware:

| Site | What it does | Consequence if missed |
|---|---|---|
| `infrastructure/model_utils.py:85` `resolve_model_path` | ModelScope resolution for FunASR/VAD/punc/diarization | Overrides ignored for 5 callers |
| `infrastructure/model_utils.py:47` `find_huggingface_snapshot_dir` | HF resolution (Qwen, aligner) | Overrides ignored for Qwen |
| `model_loader.py:215` `_build_modelscope_spec` | Integrity spec paths | **Boot aborts** (see below) |
| `model_loader.py:222` `_build_huggingface_spec` | Integrity spec paths | **Boot aborts** |
| `manager.py:126,132` | `exists` flags in the models API | API reports `exists: false` for working models |
| `download_models.py:52` `_get_cache_path` | Existence check + export | Redundant re-download; bad export |
| `download_models.py:109` | CAM++ `configuration.json` location | Offline diarization silently unfixed |
| `download_models.py:182` | Download target dir | Downloads next to, not into, override |

**The boot-abort trap.** `verify_required_models_integrity` (`model_loader.py:301`) is
called from `app/main.py:83-86`, which raises `RuntimeError` on any invalid model. Its
specs come from `_build_modelscope_spec`/`_build_huggingface_spec`, which read the
caches directly. So a *correct* override would abort startup. Rev 1 missed this
entirely and exempted only `check_all_models()`.

Note also that the pattern-based checking (`required_patterns`,
`snapshots/*/config.json`) lives in `verify_required_models_integrity`, **not** in
`check_all_models` (`download_models.py:74-96`), which does pure existence checks.

### New module: `app/core/model_paths.py`

Owns one job: turn the environment into a validated `{model_id: Path}` map.

- Scans `os.environ` for the `MODEL_PATH_` prefix.
- Resolves each var's slug against the registry to a model id.
- Validates: path exists, is a directory, is non-empty → else `ValueError` naming both
  the variable and the path.
- Unknown slug → `ValueError` listing valid slugs.
- Two slugs mapping to one model id with **different** paths → `ValueError`.
- Expands `~` and `$VARS` (mirror `_expand_path`, `model_utils.py:18`).

**Built lazily on first consult, cached in a module global** — not "once at boot".
Rev 1 put validation in `bootstrap.ensure_models_downloaded`, which is wrong twice
over: that function wraps everything in `except Exception` (`bootstrap.py:40-42`) and
would downgrade the `ValueError` to a `⚠️` print; and it is skipped entirely when
`WORKERS>1` (`start.py:71-76`), under `python -m app.utils.download_models`
(`download_models.py:321`), and under direct `uvicorn app.main:app`. Lazy-with-cache
means every entrypoint validates, because every entrypoint resolves a model path.

Bootstrap additionally calls the builder early and **outside** its `try/except`, so
the common CLI path still gets a clean, early error.

**Layering.** `model_paths` parses `models.json` directly (path via
`settings.models_config_path`). It must not import `app.services.asr.manager` or
`model_capabilities`: `model_utils` → `model_paths` → `manager` → `engines` →
`engines/funasr.py:10` → `app.infrastructure` is a cycle, and `app/core` depending on
`app/services` inverts layering.

### The slug registry

Model ids (`Qwen/Qwen3-ASR-1.7B`) make unusable env names. Slugs come from two sources:

**1. `models.json` keys** — uppercased, `-`/`.` → `_`. This rule is what makes goal #2
hold: a new entry gets its override for free.

| `.env` var | model id |
|---|---|
| `MODEL_PATH_QWEN3_ASR_1_7B` | `Qwen/Qwen3-ASR-1.7B` |
| `MODEL_PATH_QWEN3_ASR_0_6B` | `Qwen/Qwen3-ASR-0.6B` |
| `MODEL_PATH_PARAFORMER_LARGE` | `iic/speech_paraformer-large_asr_nat-...-online` |

An entry may declare both `offline` and `realtime` ids (`manager.py:54-56`), so one
key can address two models. No current entry does, but the rule is fixed **now**, not
later: when an entry declares both, the slugs are `<KEY>_OFFLINE` and
`<KEY>_REALTIME`; when it declares one, the bare `<KEY>` is used.

**2. A new explicit `slug` field on `ModelAsset`** (`model_capabilities.py:17`) for
support models absent from `models.json`:

| `.env` var | model id |
|---|---|
| `MODEL_PATH_VAD` | `damo/speech_fsmn_vad_zh-cn-16k-common-pytorch` |
| `MODEL_PATH_PUNC` | `iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch` |
| `MODEL_PATH_PUNC_REALTIME` | `iic/punc_ct-transformer_...-vad_realtime-...` |
| `MODEL_PATH_CAMPP_DIARIZATION` | `iic/speech_campplus_speaker-diarization_common` |
| `MODEL_PATH_CAMPP_SV` | `damo/speech_campplus_sv_zh-cn_16k-common` |
| `MODEL_PATH_CAMPP_TRANSFORMER` | `damo/speech_campplus-transformer_scl_zh-cn_16k-common` |

The slug is an **explicit field, not derived from `description`**. Descriptions are
human prose (`"CAM++ Diarization"`); deriving from them yields awkward names and would
silently rename an operator's variable if someone reworded a string.

**Forced aligner: one shared slug.** Both `qwen3-asr-1.7b` and `qwen3-asr-0.6b`
declare the same `forced_aligner_path` (`models.json:46,79`) =
`Qwen/Qwen3-ForcedAligner-0.6B`. Rev 1 proposed per-entry slugs
(`MODEL_PATH_QWEN3_ASR_1_7B_FORCED_ALIGNER`), which is incoherent: two slugs, one
model id, undefined result. There is a single **`MODEL_PATH_FORCED_ALIGNER`**. If a
future entry declares a *different* aligner, the generic same-id/different-path check
raises, and the slug rule is revisited then.

### Resolution order

Per model, first hit wins:

1. **Explicit override** — `MODEL_PATH_<SLUG>`, loaded as-is.
2. **ModelScope cache** — `MODELSCOPE_PATH / model_id` (today's behavior).
3. **HuggingFace cache** — snapshot lookup (today's behavior).
4. **Bare model id** — runtime downloads it (today's behavior).

### Download and integrity interaction

Overridden models are **skipped by both** `check_all_models()` (`download_models.py:74`)
and `verify_required_models_integrity()` (`model_loader.py:301`).

The operator supplied the weights, and the integrity patterns assume cache layout —
HF specs match `snapshots/*/config.json`, but a flat override dir has `config.json` at
its root, so a valid override fails the check and aborts boot. Rather than teach the
matcher a second layout, overridden models get a weaker but explicit guarantee: boot
validation proved the directory exists and is a non-empty directory. This is a real
reduction in safety for overridden models and is the accepted cost of the
direct-model-dir decision.

`--export-dir` mode (`download_models.py:279-291`) is **out of scope**: it computes
`cache_path.relative_to(get_huggingface_cache_root())` (line 284), which raises
`ValueError` for any path outside the cache. Export refuses to run with overrides set,
with a message saying so, rather than exporting stale weights.

### CAM++ config rewrite

`fix_camplusplus_config()` (`download_models.py:99`) rewrites the diarization
`configuration.json` for offline use. Two changes:

- `get_camplusplus_replacement_paths` (`model_capabilities.py:167`) drops its
  `cache_dir` parameter and builds each replacement via `resolve_model_path(model_id)`.
  Sole caller: `download_models.py:120`.
- The **rewrite target itself** (`download_models.py:109-110`) must resolve through the
  override too, or an overridden diarization model's config is never fixed and the
  function silently returns `False`.

**Open question for the operator:** this writes into the model directory. If an
override points at a read-only mount, the rewrite fails. Proposal: warn on failure,
but fail loudly when `HF_HUB_OFFLINE=1` (where an unfixed config means diarization
reaches for modelscope.cn and breaks).

### Docker

- Add `env_file: [{path: .env, required: false}]` to `docker-compose.yml` and
  `docker-compose-cpu.yml`. Compose cannot glob env vars into `environment:`, so
  enumerating `MODEL_PATH_*` per model reintroduces the per-model edit goal #2 forbids.
  `required: false` matters — a bare `env_file: .env` is a hard error when the file is
  absent, and today both compose files run fine without one.
- The host directory must also be bind-mounted **at the same path inside the
  container**. Documented in `.env.example` with a worked example.

### `.env.example`

Extend the "Model cache behavior" section with the mechanism, the slug rule, the table
of valid slugs, and the Docker mount caveat.

## Testing

Convention: `unittest`, not pytest — `DEVICE=cpu .venv/bin/python -m unittest discover
-s tests`. Use `tempfile.TemporaryDirectory` (existing pattern,
`tests/test_model_integrity.py:10`); `tmp_path` is a pytest fixture and does not exist
here. Env via `unittest.mock.patch.dict(os.environ, ...)`, clearing the module cache
between cases. New file `tests/test_model_path_overrides.py`:

1. No `MODEL_PATH_*` set → resolution unchanged (regression guard on "defaults untouched").
2. Valid override → `resolve_model_path` returns it; cache not consulted.
3. HF override → `find_huggingface_snapshot_dir` returns the flat dir.
4. Missing path → `ValueError` naming the var and the path.
5. Path is a file; path is an empty dir → both raise.
6. Unknown `MODEL_PATH_FOO` → raises, message lists valid slugs.
7. Slug derivation: `qwen3-asr-1.7b` → `QWEN3_ASR_1_7B`.
8. Synthetic `models.json` entry gets a working override with no code change (goal #2 guard).
9. Overridden model excluded from `check_all_models()`.
10. **Overridden model excluded from `verify_required_models_integrity()`** — the
    regression guard on the rev-1 boot-abort trap.
11. `get_camplusplus_replacement_paths()` reflects an override; rewrite targets the
    overridden dir.
12. Two slugs → same model id, different paths → raises.
13. Validation fires on the `WORKERS>1` and standalone-CLI paths (the lazy-build guard).

`tests/test_huggingface_model_utils.py` exercises `find_huggingface_snapshot_dir` and
must keep passing with no overrides set. No existing test constructs `ModelAsset`
directly, so the new required field does not break tests.

## Risks

- **Docker mount mismatch.** `.env` alone cannot make a host path visible in the
  container; the bind-mount cannot be automated. Mitigated by validation — the
  container fails immediately with the unreadable path named.
- **`ModelAsset` gains a required field.** All construction sites
  (`model_capabilities.py:29,40,54,61,71,84,93,138,156`) must be updated. Sites 138 and
  156 are dynamic: 138 is the Qwen offline asset (slug from the `models.json` key), 156
  is the forced aligner (the shared `FORCED_ALIGNER` slug).
- **`get_camplusplus_replacement_paths` signature change.** Caller: `download_models.py:120`.
- **Weakened integrity checking for overridden models** — accepted, documented above.

## Rollback

Unset every `MODEL_PATH_*` variable. Resolution returns to today's cache-only behavior
with all new validation still in place — the override map is empty and every lookup
falls through to step 2.
