# -*- coding: utf-8 -*-
"""Diarization worker-process entrypoints (Change D).

Runs INSIDE spawn-based worker processes owned by
app.utils.diarization_pool.DiarizationProcessPool. The module top level must
stay light: the parent imports it only to reference these functions by
qualified name for pickling, and every spawned child re-imports it. Heavy
imports (modelscope, torch via speaker_diarizer) happen inside functions,
in the child only. This module must NEVER import the vLLM/engine stack.
"""

from __future__ import annotations

import os
import sys
from typing import Any, List, Tuple

# One pipeline per worker process. The process IS the instance: a
# ProcessPoolExecutor worker runs one task at a time, so funasr's
# mutate-shared-state-per-call hazard cannot cross requests.
_pipeline: Any = None


class _FakeWorkerPipeline:
    """DIARIZATION_WORKER_FAKE=1 stand-in (tests only): modelscope-shaped
    output including numpy scalar types, so spawn-integration tests exercise
    the real marshalling and pickle path without loading models."""

    def __call__(self, audio_path: str):
        import numpy as np

        return {
            "text": [
                [np.float64(0.0), np.float64(1.5), np.int64(0)],
                [np.float64(1.5), np.float64(3.0), np.int64(1)],
            ]
        }


def _configure_worker_logging() -> None:
    """Sinks are not inherited across spawn. Route loguru to stderr so
    [diarization-profile] lines and init tracebacks are visible in the
    parent's captured stderr — gate G1 reads exactly these lines."""
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level=os.getenv("DIARIZATION_WORKER_LOG_LEVEL", "INFO"))


def _build_pipeline() -> Any:
    if os.getenv("DIARIZATION_WORKER_FAKE") == "1":
        if os.getenv("DEVICE", "").lower() != "cpu":
            raise RuntimeError(
                "DIARIZATION_WORKER_FAKE=1 is test-only and refuses to run "
                "with DEVICE != cpu — a stray production value must fail the "
                "boot, not silently fake diarization."
            )
        return _FakeWorkerPipeline()
    # Force modelscope task registration BEFORE building — without this the
    # speaker-diarization task can be unregistered and pipeline() falls
    # through to transformers ("Unknown task speaker-diarization", the
    # qwen3-asr-9nk boot failure).
    import modelscope.pipelines.audio  # noqa: F401

    from app.utils.speaker_diarizer import _build_diarization_pipeline

    return _build_diarization_pipeline()


def _worker_init(barrier: Any = None, barrier_timeout_s: float = 300.0) -> None:
    """ProcessPoolExecutor initializer. Builds this worker's pipeline, then
    rendezvouses on the boot barrier (parties = N workers + 1 parent), so
    barrier release means every worker exists AND finished building. An
    exception here surfaces to the parent as BrokenProcessPool with NO cause
    text — the real traceback is on this process's stderr."""
    global _pipeline
    _configure_worker_logging()
    _pipeline = _build_pipeline()
    if barrier is not None:
        barrier.wait(timeout=barrier_timeout_s)


def _worker_probe() -> int:
    """Warmup no-op; returns the worker's PID for the distinct-N assertion."""
    return os.getpid()


def _worker_diarize(audio_path: str) -> List[Tuple[float, float, int]]:
    """Run this worker's pipeline; return native-typed triples.

    Conversion to native float/int happens HERE (worker side): the pipeline
    emits lists with numpy scalars, and the cross-process contract is small
    plain-Python triples. Malformed segments are skipped (same policy as the
    old parent-side parse). Pipeline exceptions propagate untouched so the
    parent's "too short" string match keeps working.
    """
    if _pipeline is None:
        raise RuntimeError("diarization worker used before _worker_init")

    from app.utils.speaker_diarizer import _suppress_empty_cache

    with _suppress_empty_cache():
        result = _pipeline(audio_path)

    if isinstance(result, dict):
        raw = result.get("text", [])
    else:
        raw = getattr(result, "text", []) or []

    triples: List[Tuple[float, float, int]] = []
    for seg in raw:
        if isinstance(seg, (list, tuple)) and len(seg) == 3:
            try:
                triples.append((float(seg[0]), float(seg[1]), int(seg[2])))
            except (TypeError, ValueError):
                continue
    return triples
