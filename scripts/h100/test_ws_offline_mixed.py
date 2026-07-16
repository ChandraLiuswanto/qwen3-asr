#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H100-ONLY. Mixed websocket + offline concurrency test, and the websocket
decode-latency baseline (p50/p95).

WHY THE SHAPE OF THIS TEST IS THE WHOLE POINT
---------------------------------------------
Read this before changing anything here, or you will "simplify" it into a test
that cannot fail.

The ITN path (`app/utils/text_processing.normalize_asr_text`) is guarded by an
ITN lock. To exercise that guard you need two threads in ITN at once. You
CANNOT get that from concurrent websocket sessions alone:

  * A websocket session's decode runs on an executor thread (`await
    _run_decode(...)`), but its ITN call (`_normalize_output_text`) runs AFTER
    that await returns — i.e. back ON THE EVENT-LOOP THREAD.
  * One event loop runs one callback at a time. So websocket ITN calls are
    serialized by the event loop itself and CANNOT race each other, no matter
    how many sessions you open.
  * A test built purely from concurrent websocket sessions would therefore
    NEVER fail, guard or no guard. It would be theatre.

The only live ITN race is:

    a websocket's event-loop-thread ITN call
        vs.
    an OFFLINE request's executor-thread ITN call

Hence the mixed shape: websocket sessions AND offline requests, together. If
you drop the offline half of this test, you have deleted its reason to exist.

WHAT IT CHECKS
--------------
  1. Per-channel keyword integrity across BOTH channels, same rule as
     `test_offline_mixing.py`: every websocket transcript and every offline
     response must contain its own audio's keyword and no other channel's.
  2. Websocket decode latency p50/p95 — the BASELINE from which a threshold can
     be set. This script does not enforce a threshold; there is no data yet to
     set one from. It produces that data.

INPUTS
------
  ASR_BASE_URL       default http://localhost:8000 (ws URL derived: http->ws)
  ASR_TEST_AUDIO_DIR distinct-content audio, `<label>_<keyword>.<ext>` (see
                     test_offline_mixing.py). The WEBSOCKET half additionally
                     requires those files to be 16 kHz mono 16-bit PCM WAV --
                     the ws protocol takes raw PCM, not containers.
  ASR_API_KEY        optional bearer token for the offline half.

Offline ITN: `offline_transcription_service.transcribe` hardcodes
`enable_itn=True`; there is no form field to set. The offline half of this test
therefore always exercises ITN. The websocket half sets
`enable_inverse_text_normalization: true` explicitly in its `start` payload.

USAGE
-----
    python scripts/h100/test_ws_offline_mixed.py | tee /tmp/ws_offline_mixed.txt
    python scripts/h100/test_ws_offline_mixed.py --iterations 20 --ws 4 --offline 4

Exit codes: 0 = no contamination observed, 1 = contamination / failure,
2 = harness or setup error.

Dependencies: stdlib + `websockets`. Imports `test_offline_mixing.py` (same
directory) for the shared keyword/integrity logic — keep the two files together.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_offline_mixing import (  # noqa: E402
    BOTH,
    FOREIGN,
    MISSING_OWN,
    OK,
    AudioCase,
    check_integrity,
    is_contaminated,
    load_cases,
    summarize,
    transcribe,
    validate_cases,
)

WS_SAMPLE_RATE = 16000
WS_CHUNK_SEC = 2.0


# --------------------------------------------------------------------------
# Pure logic. Unit-tested locally in tests/test_ws_offline_mixed_stats.py.
# Covers the percentile arithmetic ONLY. It says nothing about whether the
# service actually races, or whether the measured latencies are correct.
# --------------------------------------------------------------------------


def percentile(values: list[float], fraction: float) -> Optional[float]:
    """Nearest-rank percentile. `fraction` in [0, 1]. Pure.

    Nearest-rank (not interpolated) is chosen deliberately: every reported
    number is an ACTUAL OBSERVED latency, not a synthetic value between two
    samples. For small N that matters — an interpolated p95 of 12 samples is a
    number nobody ever measured.

    Returns None for an empty input rather than 0.0: "no data" must not be
    reportable as "zero latency".
    """
    if not values:
        return None
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction!r}")
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(fraction * len(ordered))))
    return ordered[rank - 1]


def describe_latencies(values: list[float]) -> dict[str, Optional[float]]:
    """p50/p95/min/max/mean over observed latencies. Pure."""
    if not values:
        return {"count": 0, "p50": None, "p95": None, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


# --------------------------------------------------------------------------
# Websocket client
# --------------------------------------------------------------------------


@dataclass
class WsResult:
    case: AudioCase
    text: str = ""
    latencies: list[float] = field(default_factory=list)
    silent_sends: int = 0
    error: Optional[str] = None


def ws_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.startswith("https://"):
        return "wss://" + url[len("https://"):] + "/ws/v1/asr/qwen"
    if url.startswith("http://"):
        return "ws://" + url[len("http://"):] + "/ws/v1/asr/qwen"
    raise SystemExit(f"cannot derive a websocket URL from {base_url!r}")


def read_pcm16(path: str) -> bytes:
    """Read a 16 kHz mono 16-bit PCM WAV as raw bytes for the ws protocol."""
    with wave.open(path, "rb") as handle:
        if handle.getnchannels() != 1:
            raise SystemExit(f"{path}: websocket half needs MONO, got {handle.getnchannels()} channels")
        if handle.getsampwidth() != 2:
            raise SystemExit(f"{path}: websocket half needs 16-bit PCM, got {handle.getsampwidth() * 8}-bit")
        if handle.getframerate() != WS_SAMPLE_RATE:
            raise SystemExit(f"{path}: websocket half needs {WS_SAMPLE_RATE} Hz, got {handle.getframerate()} Hz")
        return handle.readframes(handle.getnframes())


async def ws_session(base_url: str, case: AudioCase, chunk_sec: float, recv_timeout: float) -> WsResult:
    """Stream one file over the qwen websocket, measuring per-chunk latency.

    Protocol (from app/services/qwen3_websocket_asr.py):
      -> {"type": "start", "payload": {...}}     <- {"type": "started"}
      -> <raw pcm bytes>                          <- {"type": "result", "results": [...]}
                                                  <- {"type": "segment_end"/"segment_start"}
      -> {"type": "stop"}                         <- {"type": "final", "result": {...}}

    LATENCY DEFINITION AND ITS LIMITS — read before quoting these numbers.
    We send exactly one chunk_size_sec chunk, then wait for the matching
    `result`. Latency = send -> first `result`. This is LOCKSTEP: it does not
    pipeline sends the way a real client would, so it measures per-decode
    turnaround under this harness's traffic shape, not end-to-end streaming lag
    of a production client. It is a comparable baseline, not a user-facing SLA.

    The server suppresses a `result` when the text did not change
    (`if full != ctx.last_partial_text`). Those sends yield NO message; they are
    counted as `silent_sends` and EXCLUDED from the latency sample rather than
    recorded as a timeout. Excluding them biases the sample toward chunks that
    produced new text -- state that when reporting.
    """
    try:
        import websockets
    except ImportError:
        return WsResult(case=case, error="the `websockets` package is not installed in this image")

    pcm = read_pcm16(case.path)
    chunk_bytes = int(chunk_sec * WS_SAMPLE_RATE) * 2
    result = WsResult(case=case)

    try:
        async with websockets.connect(ws_url(base_url), max_size=None) as socket:
            await socket.send(
                json.dumps(
                    {
                        "type": "start",
                        "payload": {
                            "format": "pcm",
                            "sample_rate": WS_SAMPLE_RATE,
                            "enable_inverse_text_normalization": True,
                            "chunk_size_sec": chunk_sec,
                        },
                    }
                )
            )
            started = json.loads(await asyncio.wait_for(socket.recv(), timeout=recv_timeout))
            if started.get("type") != "started":
                result.error = f"expected 'started', got {started!r}"
                return result

            for offset in range(0, len(pcm), chunk_bytes):
                sent_at = time.perf_counter()
                await socket.send(pcm[offset:offset + chunk_bytes])
                # Drain until this chunk's `result` arrives. segment_start /
                # segment_end are bookkeeping and are not the decode reply.
                while True:
                    try:
                        raw = await asyncio.wait_for(socket.recv(), timeout=recv_timeout)
                    except asyncio.TimeoutError:
                        result.silent_sends += 1
                        break
                    message = json.loads(raw)
                    kind = message.get("type")
                    if kind == "result":
                        result.latencies.append(time.perf_counter() - sent_at)
                        break
                    if kind == "error":
                        result.error = f"server error: {message!r}"
                        return result
                    # segment_start / segment_end: keep waiting.

            await socket.send(json.dumps({"type": "stop"}))
            deadline = time.perf_counter() + recv_timeout
            while time.perf_counter() < deadline:
                message = json.loads(await asyncio.wait_for(socket.recv(), timeout=recv_timeout))
                if message.get("type") == "final":
                    result.text = str(message.get("result", {}).get("full_text", ""))
                    return result
                if message.get("type") == "error":
                    result.error = f"server error: {message!r}"
                    return result
            result.error = "no 'final' message before timeout"
    except Exception as exc:  # noqa: BLE001 - never mask a failure as a pass
        result.error = repr(exc)
    return result


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


async def run_iteration(
    base_url: str,
    ws_cases: list[AudioCase],
    offline_cases: list[AudioCase],
    api_key: Optional[str],
    timeout: float,
    chunk_sec: float,
) -> tuple[list[WsResult], list, list[str]]:
    """Fire all websocket sessions and all offline requests TOGETHER.

    They must overlap in time; that overlap is the entire experiment. Offline
    requests are blocking urllib calls, so they go to threads via to_thread —
    which is also what puts their ITN call on an executor thread, opposite the
    websockets' event-loop-thread ITN calls. That opposition is the race.
    """
    ws_tasks = [asyncio.create_task(ws_session(base_url, case, chunk_sec, timeout)) for case in ws_cases]
    offline_tasks = [
        asyncio.create_task(asyncio.to_thread(transcribe, base_url, case, api_key, timeout))
        for case in offline_cases
    ]
    ws_results = await asyncio.gather(*ws_tasks)
    offline_results = await asyncio.gather(*offline_tasks)

    all_keywords = [case.keyword for case in ws_cases] + [case.keyword for case in offline_cases]
    verdicts: list[str] = []
    for result in list(ws_results) + list(offline_results):
        if result.error is not None:
            verdicts.append(MISSING_OWN)
            continue
        others = [kw for kw in all_keywords if kw != result.case.keyword]
        verdict, _foreign = check_integrity(result.text, result.case.keyword, others)
        verdicts.append(verdict)
    return list(ws_results), list(offline_results), verdicts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--ws", type=int, default=4, help="concurrent websocket sessions (default 4)")
    parser.add_argument("--offline", type=int, default=4, help="concurrent offline requests (default 4)")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--chunk-sec", type=float, default=WS_CHUNK_SEC)
    parser.add_argument("--base-url", default=os.environ.get("ASR_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--audio-dir", default=os.environ.get("ASR_TEST_AUDIO_DIR", ""))
    args = parser.parse_args()

    if not args.audio_dir:
        print("ASR_TEST_AUDIO_DIR is not set (or pass --audio-dir).", file=sys.stderr)
        return 2
    if args.ws < 1 or args.offline < 1:
        print(
            "BOTH --ws and --offline must be >= 1. With either at 0 this test loses "
            "the ONLY shape that can catch the ITN race (event-loop thread vs "
            "executor thread) and becomes theatre. Refusing to run.",
            file=sys.stderr,
        )
        return 2
    if args.iterations < 20:
        print(f"[warn] --iterations={args.iterations} is below the minimum of 20; the race is probabilistic.")

    try:
        cases = load_cases(args.audio_dir)
        validate_cases(cases, args.ws + args.offline)
    except SystemExit as exc:
        print(f"SETUP ERROR: {exc}", file=sys.stderr)
        return 2

    wav_cases = [case for case in cases if case.path.lower().endswith(".wav")]
    if len(wav_cases) < args.ws:
        print(
            f"SETUP ERROR: the websocket half needs at least {args.ws} 16kHz mono 16-bit "
            f"PCM .wav files; found {len(wav_cases)} .wav in {args.audio_dir}.",
            file=sys.stderr,
        )
        return 2

    ws_cases = wav_cases[: args.ws]
    remaining = [case for case in cases if case not in ws_cases]
    if len(remaining) < args.offline:
        print(
            f"[warn] only {len(remaining)} files left for {args.offline} offline slots; reusing files. "
            "Two slots sharing a keyword cannot visibly contaminate each other — detection is WEAKENED."
        )
        remaining = remaining or cases
    offline_cases = [remaining[i % len(remaining)] for i in range(args.offline)]

    print("=" * 78)
    print("Mixed websocket + offline concurrency — H100 / deployed service only")
    print("=" * 78)
    print(f"base url   : {args.base_url}")
    print(f"ws url     : {ws_url(args.base_url)}")
    print(f"ws cases   : {[(c.label, c.keyword) for c in ws_cases]}")
    print(f"off cases  : {[(c.label, c.keyword) for c in offline_cases]}")
    print(f"iterations : {args.iterations}   ws={args.ws}  offline={args.offline}")
    print()
    print("This mixed shape is deliberate: websocket ITN runs on the EVENT LOOP")
    print("thread, so ws<->ws ITN calls are serialized by the loop and cannot race.")
    print("Only ws-vs-offline (event loop vs executor thread) can. Do not reduce")
    print("this to a websocket-only test; it would never fail.")
    print()

    totals: dict[str, int] = {}
    latencies: list[float] = []
    silent_sends = 0
    failed_iterations = 0

    for index in range(1, args.iterations + 1):
        ws_results, offline_results, verdicts = asyncio.run(
            run_iteration(
                args.base_url,
                ws_cases,
                offline_cases,
                os.environ.get("ASR_API_KEY"),
                args.timeout,
                args.chunk_sec,
            )
        )
        counts = summarize(verdicts)
        for key, value in counts.items():
            totals[key] = totals.get(key, 0) + value
        for result in ws_results:
            latencies.extend(result.latencies)
            silent_sends += result.silent_sends

        bad = is_contaminated(counts) or counts.get(MISSING_OWN, 0) > 0
        if bad:
            failed_iterations += 1
        print(
            f"iteration {index:3d}/{args.iterations}: {'FAIL' if bad else 'PASS'}  "
            f"ok={counts[OK]} missing_own={counts[MISSING_OWN]} "
            f"foreign={counts[FOREIGN]} both={counts[BOTH]}"
        )
        if bad:
            for result, verdict in zip(list(ws_results) + list(offline_results), verdicts):
                if verdict == OK:
                    continue
                channel = "ws " if isinstance(result, WsResult) else "off"
                if result.error is not None:
                    print(f"    [{verdict}] {channel} {Path(result.case.path).name}: error: {result.error}")
                else:
                    others = [c.keyword for c in cases if c.keyword != result.case.keyword]
                    _v, foreign = check_integrity(result.text, result.case.keyword, others)
                    print(
                        f"    [{verdict}] {channel} {Path(result.case.path).name}: "
                        f"expected {result.case.keyword!r}; foreign={foreign or 'none'}; "
                        f"text={result.text[:160]!r}"
                    )

    stats = describe_latencies(latencies)
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"iterations with a fault : {failed_iterations}/{args.iterations}")
    print(f"responses OK            : {totals.get(OK, 0)}")
    print(f"responses MISSING_OWN   : {totals.get(MISSING_OWN, 0)}")
    print(f"responses FOREIGN       : {totals.get(FOREIGN, 0)}   <- mixing / ITN corruption signature")
    print(f"responses BOTH          : {totals.get(BOTH, 0)}   <- mixing / ITN corruption signature")
    print()
    print("WEBSOCKET DECODE LATENCY (send chunk -> matching `result`)")
    if stats["count"] == 0:
        print("  NO SAMPLES. Every send was silent or errored — there is no baseline here.")
        print("  Do not record a latency number. Investigate before quoting anything.")
    else:
        print(f"  samples : {stats['count']}   (silent sends excluded: {silent_sends})")
        print(f"  p50     : {stats['p50'] * 1000:.1f} ms")
        print(f"  p95     : {stats['p95'] * 1000:.1f} ms")
        print(f"  min/max : {stats['min'] * 1000:.1f} / {stats['max'] * 1000:.1f} ms")
        print(f"  mean    : {stats['mean'] * 1000:.1f} ms")
    print()
    print("HOW TO READ THE LATENCY NUMBERS — do not quote them without this:")
    print("  * This is a BASELINE to SET a threshold from, not a threshold check.")
    print("    No threshold is enforced here because no data existed to set one.")
    print(f"  * Measured under load: {args.ws} ws sessions + {args.offline} offline requests")
    print("    concurrently. Numbers taken under a different mix are not comparable.")
    print("  * The client is LOCKSTEP (one chunk, wait for its result). A pipelining")
    print("    production client will see different end-to-end lag.")
    print("  * Silent sends (server suppressed an unchanged partial) are EXCLUDED, so")
    print("    the sample is biased toward chunks that produced new text.")
    print("  * To compare against `main`, run this same script against a `main` deploy")
    print("    with the same audio and the same --ws/--offline. Absolute numbers across")
    print("    different hardware or audio mean nothing.")
    print()
    print("KNOWN INSTRUMENTATION GAP: `ASR_STAGE_TIMINGS` is logged only on the SUCCESS")
    print("path (app/services/asr/engines/base.py) — a request that raises emits no stage")
    print("line. If you cross-reference server-side stage timings with this run, they")
    print("describe successful requests only and are biased accordingly. Reconcile the")
    print("stage-line count against the request count before drawing conclusions.")

    if is_contaminated(totals):
        print()
        print("VERDICT: CONTAMINATION OBSERVED across the mixed ws/offline load.")
        return 1
    if totals.get(MISSING_OWN, 0) > 0:
        print()
        print("VERDICT: no foreign keywords, but responses/sessions missed their OWN")
        print("keyword or errored. Not proof of a race — but not a pass either.")
        print("Investigate (see the per-iteration detail above) before recording.")
        return 1
    print()
    print("VERDICT: no contamination observed.")
    print("EVIDENCE, NOT PROOF: the race is probabilistic and this ran")
    print(f"{args.iterations} rounds. It does not establish that the race cannot occur.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
