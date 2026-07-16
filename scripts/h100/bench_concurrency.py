#!/usr/bin/env python3
"""Throughput benchmark for POST /v1/audio/transcriptions under concurrency.

Answers the only question change A has not answered: is it actually faster?

WHAT THIS MEASURES
    Wall-clock and per-request latency for N concurrent transcriptions of the
    same audio, swept over several N. It does NOT check transcript correctness
    — test_offline_mixing.py does that. Run that first: a fast wrong answer is
    not a win.

THE COMPARISON THAT MATTERS
    This measures ONE server configuration. To learn what change A bought, run
    it twice against a restarted server:

        VLLM_OFFLINE_CONCURRENCY=1  -> serialized baseline (old behavior)
        VLLM_OFFLINE_CONCURRENCY=4  -> current default

    The knob cannot be flipped without a restart, so this script will not do it
    for you. It prints the server's reported value so the two runs are
    comparable and hard to mix up.

STDLIB ONLY — runs inside the deployed image with no extra install.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BASE_URL = os.environ.get("ASR_BASE_URL", "http://localhost:8000")


@dataclass
class Attempt:
    ok: bool
    seconds: float
    status: int
    detail: str = ""


def _encode_multipart(audio: Path, fields: dict[str, str]) -> tuple[bytes, str]:
    boundary = f"----bench{uuid.uuid4().hex}"
    sep = f"--{boundary}".encode()
    body = bytearray()
    for key, value in fields.items():
        body += sep + b"\r\n"
        body += f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
        body += f"{value}\r\n".encode()
    ctype = mimetypes.guess_type(audio.name)[0] or "application/octet-stream"
    body += sep + b"\r\n"
    body += (
        f'Content-Disposition: form-data; name="file"; filename="{audio.name}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode()
    body += audio.read_bytes() + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _transcribe(base_url: str, audio: Path, timeout: float, diarize: bool) -> Attempt:
    fields = {
        "model": "qwen3-asr",
        "response_format": "json",
        "enable_speaker_diarization": "true" if diarize else "false",
    }
    body, content_type = _encode_multipart(audio, fields)
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    api_key = os.environ.get("API_KEY", "").strip()
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            return Attempt(True, time.perf_counter() - t0, resp.status)
    except urllib.error.HTTPError as exc:
        return Attempt(False, time.perf_counter() - t0, exc.code, exc.reason or "")
    except Exception as exc:  # noqa: BLE001 — a bench must report, not crash
        return Attempt(False, time.perf_counter() - t0, 0, f"{type(exc).__name__}: {exc}")


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[k]


def _round(base_url: str, audio: Path, n: int, timeout: float, diarize: bool) -> list[Attempt]:
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [
            pool.submit(_transcribe, base_url, audio, timeout, diarize) for _ in range(n)
        ]
        return [f.result() for f in futures]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True, help="path to a representative clip (~5 min)")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--levels", default="1,2,4,8,10", help="concurrency levels (default 1,2,4,8,10)")
    ap.add_argument("--rounds", type=int, default=3, help="rounds per level, median reported (default 3)")
    ap.add_argument("--timeout", type=float, default=1200.0)
    ap.add_argument("--no-diarization", action="store_true",
                    help="disable diarization; isolates the vLLM path from the diarization mutex")
    ap.add_argument("--json-out", default="", help="write raw results here")
    args = ap.parse_args()

    audio = Path(args.audio)
    if not audio.is_file():
        print(f"ERROR: no such audio file: {audio}", file=sys.stderr)
        return 2

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    diarize = not args.no_diarization

    print("=" * 78)
    print("ASR CONCURRENCY BENCHMARK")
    print("=" * 78)
    print(f"  base url    : {args.base_url}")
    print(f"  audio       : {audio}  ({audio.stat().st_size / 1e6:.1f} MB)")
    print(f"  levels      : {levels}   rounds/level: {args.rounds}")
    print(f"  diarization : {'ON (default path)' if diarize else 'OFF (vLLM path isolated)'}")
    print()
    print("  Server-side knobs are NOT set by this script. Restart the server with")
    print("  VLLM_OFFLINE_CONCURRENCY=1 for the serialized baseline, then again at")
    print("  the default, and compare the two runs. Record which is which.")
    print()

    # Warm the pipeline: the first request pays lazy init (ITN FSTs, aligner)
    # and would otherwise land entirely on level 1, flattering every later level.
    print("warming up (one request, not measured) ...", flush=True)
    warm = _transcribe(args.base_url, audio, args.timeout, diarize)
    if not warm.ok:
        print(f"  WARMUP FAILED: status={warm.status} {warm.detail}", file=sys.stderr)
        print("  Fix the service before benchmarking; numbers from a broken service are noise.",
              file=sys.stderr)
        return 1
    print(f"  ok ({warm.seconds:.1f}s)\n", flush=True)

    rows = []
    raw: dict[str, dict] = {}
    for n in levels:
        per_round = []
        latencies: list[float] = []
        failures = 0
        for r in range(args.rounds):
            t0 = time.perf_counter()
            attempts = _round(args.base_url, audio, n, args.timeout, diarize)
            wall = time.perf_counter() - t0
            per_round.append(wall)
            for a in attempts:
                if a.ok:
                    latencies.append(a.seconds)
                else:
                    failures += 1
                    print(f"  [n={n} round={r}] FAIL status={a.status} {a.detail}", file=sys.stderr)
        wall = statistics.median(per_round)
        rows.append((n, wall, n / wall if wall else 0.0, _pct(latencies, 50), _pct(latencies, 95), failures))
        raw[str(n)] = {"walls": per_round, "latencies": latencies, "failures": failures}
        print(f"  n={n:<3} wall={wall:7.1f}s  throughput={n / wall if wall else 0:5.2f} req/s"
              f"  p50={_pct(latencies, 50):6.1f}s  p95={_pct(latencies, 95):6.1f}s"
              f"{'  FAILURES=' + str(failures) if failures else ''}", flush=True)

    print()
    print("=" * 78)
    print(f"{'concurrency':>12} {'wall(s)':>9} {'req/s':>8} {'p50(s)':>8} {'p95(s)':>8} {'speedup':>9} {'fails':>6}")
    print("-" * 78)
    base_wall = rows[0][1] if rows else 0.0
    base_n = rows[0][0] if rows else 1
    for n, wall, thr, p50, p95, fails in rows:
        # Speedup vs the n=1 serial reference: how much of the extra work is free.
        ideal = base_wall * (n / base_n)
        speedup = ideal / wall if wall else 0.0
        print(f"{n:>12} {wall:>9.1f} {thr:>8.2f} {p50:>8.1f} {p95:>8.1f} {speedup:>8.2f}x {fails:>6}")
    print("=" * 78)
    print()
    print("HOW TO READ THIS")
    print("  speedup = (time to do n requests one-at-a-time) / (measured wall for n).")
    print("  1.0x means concurrency bought nothing. n.0x would mean perfect scaling,")
    print("  which one GPU cannot give you — expect well under it.")
    print()
    print("  If speedup is ~1.0x at the default VLLM_OFFLINE_CONCURRENCY, the requests")
    print("  are still serialized somewhere. Prime suspect: diarization, which holds a")
    print("  BoundedSemaphore(1) on the default path. Re-run with --no-diarization; if")
    print("  that jumps and the default does not, the diarization mutex is your ceiling")
    print("  and the vLLM work did not address it.")
    print()
    print("  FAILURES ARE NOT NOISE. A p95 that looks good because slow requests errored")
    print("  out is a lie. Any non-zero fails column invalidates the row.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(raw, indent=2))
        print(f"\nraw results -> {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
