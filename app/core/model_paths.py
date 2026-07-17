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
