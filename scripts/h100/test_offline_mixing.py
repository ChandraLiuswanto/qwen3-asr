#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H100-ONLY. Offline transcription cross-contamination ("mixing") regression test.

WHAT THIS IS
------------
An instrument, not a proof. It fires N iterations x C concurrent
`POST /v1/audio/transcriptions` requests, each carrying a DIFFERENT audio file,
and checks that every response contains its OWN file's expected keyword and
NONE of the other files' keywords.

THE BUG IT TARGETS
------------------
`Qwen3VLLMBackend` (`app/services/asr/qwen3_vllm.py`) holds ONE shared
`vllm.LLM`. `_run_generate` reassembles results positionally:

    for output, (_audio, _context, language) in zip(outputs, audio_items):

If two threads call `generate()` concurrently on that one `LLM`, each caller
drains the shared engine and collects a MIX of both callers' outputs. `zip`
then silently truncates to the shorter side and pairs request B's transcripts
onto request A's segments. The result is WRONG TEXT WITH NO EXCEPTION — no
traceback, no 500, no log line. Only a content check like this one can see it.
The guard under test is `self._llm_lock` in `_run_generate`.

*** THIS TEST IS WORTHLESS UNTIL YOU HAVE SEEN IT FAIL ***
-----------------------------------------------------------
A green run against the fixed build proves NOTHING on its own. A test that
cannot fail is not evidence. Before you record a PASS as meaningful you MUST
complete the must-fail runbook (`--check-detects-bug` prints it):

  1. Deploy a build with the guard REMOVED — comment out `with self._llm_lock:`
     in `_run_generate` (dedent its body). Run this script. EXPECT FAIL.
  2. Restore the real build. Run this script. EXPECT 20/20 PASS.

If step 1 PASSES, this test does not detect the bug it was written for. Do not
record step 2 as a verification — debug the instrument first (see the
troubleshooting notes under `--check-detects-bug`).

THE RACE IS PROBABILISTIC — THIS IS EVIDENCE, NOT PROOF
-------------------------------------------------------
Interleaving depends on scheduler timing, batch composition and audio length.
An unguarded build can pass by luck; a single iteration proves nothing either
way. That is why the default is N=20 concurrent rounds. Even 20/20 PASS means
"no contamination observed in 20 rounds", NOT "the race is impossible".
Report it in those words.

INPUTS
------
  ASR_BASE_URL       default http://localhost:8000
  ASR_TEST_AUDIO_DIR directory of DISTINCT-CONTENT audio files. Each filename
                     encodes the keyword its transcript must contain, as
                     `<label>_<keyword>.<ext>`, e.g. `alpha_银行.wav` must
                     transcribe to text containing `银行`.
  ASR_API_KEY        optional; sent as `Authorization: Bearer <key>`.

The keywords MUST be mutually exclusive: no file's keyword may legitimately
appear in another file's audio, or a "contamination" hit is a false positive.
Verify this by running with `--concurrency 1` first (see below).

USAGE
-----
    python scripts/h100/test_offline_mixing.py | tee /tmp/offline_mixing.txt
    python scripts/h100/test_offline_mixing.py --iterations 20 --concurrency 8
    python scripts/h100/test_offline_mixing.py --concurrency 1   # sanity/baseline
    python scripts/h100/test_offline_mixing.py --check-detects-bug

Exit codes: 0 = no contamination observed, 1 = CONTAMINATION (or a keyword the
model never produced), 2 = harness/setup error.

Self-contained: stdlib only. Imports nothing from this repo.
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import random
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json

DEFAULT_BASE_URL = "http://localhost:8000"
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".mp4", ".webm"}

# Verdicts from the pure integrity check.
OK = "OK"
MISSING_OWN = "MISSING_OWN"
FOREIGN = "FOREIGN"
BOTH = "BOTH"


# --------------------------------------------------------------------------
# Pure logic. Unit-tested locally in tests/test_offline_mixing_check.py.
# This is ONLY the matching arithmetic — it says nothing about whether the
# real service mixes. That question can only be answered on the H100.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioCase:
    """One test audio file and the keyword its transcript must contain."""

    label: str
    keyword: str
    path: str


def parse_case(path: str) -> Optional[AudioCase]:
    """Parse `<label>_<keyword>.<ext>` into an AudioCase.

    Returns None if the filename does not encode a keyword. Splits on the LAST
    underscore, so labels may themselves contain underscores.
    """
    stem = Path(path).stem
    if "_" not in stem:
        return None
    label, _, keyword = stem.rpartition("_")
    if not label or not keyword:
        return None
    return AudioCase(label=label, keyword=keyword, path=path)


def check_integrity(text: str, own_keyword: str, other_keywords: list[str]) -> tuple[str, list[str]]:
    """Classify one response body against its own and its peers' keywords.

    Returns (verdict, foreign_keywords_found).

      OK           own keyword present, no foreign keyword present
      MISSING_OWN  own keyword absent, no foreign keyword — truncation/empty,
                   or simply a bad transcript. Suspicious, not proof of mixing.
      FOREIGN      another request's keyword appeared in this response. This is
                   the mixing signature.
      BOTH         own keyword missing AND a foreign keyword present — the
                   clearest form of the mixing signature.

    Pure: text in, verdict out. Substring matching, deliberately: the keywords
    are chosen to be mutually exclusive, so a substring hit is the signal.
    """
    has_own = own_keyword in text
    foreign = [keyword for keyword in other_keywords if keyword != own_keyword and keyword in text]
    if has_own and not foreign:
        return OK, []
    if has_own and foreign:
        return FOREIGN, foreign
    if not has_own and foreign:
        return BOTH, foreign
    return MISSING_OWN, []


def summarize(verdicts: list[str]) -> dict[str, int]:
    """Count verdicts by kind. Pure."""
    counts = {OK: 0, MISSING_OWN: 0, FOREIGN: 0, BOTH: 0}
    for verdict in verdicts:
        counts[verdict] = counts.get(verdict, 0) + 1
    return counts


def is_contaminated(counts: dict[str, int]) -> bool:
    """True iff a foreign keyword was observed anywhere. Pure.

    MISSING_OWN alone does NOT count as contamination: it can mean the model
    simply produced a poor transcript. It is reported separately and loudly,
    because a truncating `zip` also produces it.
    """
    return counts.get(FOREIGN, 0) > 0 or counts.get(BOTH, 0) > 0


# --------------------------------------------------------------------------
# HTTP (stdlib multipart; no requests dependency).
# --------------------------------------------------------------------------


@dataclass
class Response:
    case: AudioCase
    text: str = ""
    error: Optional[str] = None
    elapsed_s: float = 0.0
    raw: dict = field(default_factory=dict)


def _encode_multipart(fields: dict[str, str], filename: str, content: bytes) -> tuple[bytes, str]:
    boundary = f"----h100mixing{uuid.uuid4().hex}"
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n".encode()
    )
    parts.append(content)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def transcribe(base_url: str, case: AudioCase, api_key: Optional[str], timeout: float) -> Response:
    """One blocking POST /v1/audio/transcriptions with default parameters.

    Defaults are deliberate: this must exercise the ordinary production path
    (verbose_json, speaker diarization on), which is where the shared `LLM` and
    the positional `zip` live. Offline ITN is hardcoded on in
    `offline_transcription_service.transcribe` — there is no form field for it.
    """
    content = Path(case.path).read_bytes()
    body, content_type = _encode_multipart(
        {"response_format": "verbose_json"},
        Path(case.path).name,
        content,
    )
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/audio/transcriptions",
        data=body,
        method="POST",
    )
    request.add_header("Content-Type", content_type)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as handle:
            payload = json.loads(handle.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        return Response(case=case, error=f"HTTP {exc.code}: {detail}", elapsed_s=time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001 - the harness must never mask a failure
        return Response(case=case, error=repr(exc), elapsed_s=time.perf_counter() - started)

    elapsed = time.perf_counter() - started
    if not isinstance(payload, dict):
        return Response(case=case, error=f"unexpected body: {payload!r:.200}", elapsed_s=elapsed)
    return Response(case=case, text=str(payload.get("text", "")), elapsed_s=elapsed, raw=payload)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def load_cases(directory: str) -> list[AudioCase]:
    root = Path(directory)
    if not root.is_dir():
        raise SystemExit(f"ASR_TEST_AUDIO_DIR is not a directory: {directory}")
    cases: list[AudioCase] = []
    skipped: list[str] = []
    for path in sorted(root.iterdir()):
        if path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        case = parse_case(str(path))
        if case is None:
            skipped.append(path.name)
            continue
        cases.append(case)
    for name in skipped:
        print(f"[warn] skipping {name!r}: filename does not encode `<label>_<keyword>.<ext>`")
    return cases


def validate_cases(cases: list[AudioCase], concurrency: int) -> None:
    if len(cases) < 2:
        raise SystemExit(
            f"need at least 2 distinct audio files, found {len(cases)}. "
            "This test cannot detect mixing with fewer than 2 concurrent inputs."
        )
    keywords = [case.keyword for case in cases]
    duplicates = {kw for kw in keywords if keywords.count(kw) > 1}
    if duplicates:
        raise SystemExit(
            f"keywords must be mutually exclusive; these repeat across files: {sorted(duplicates)}. "
            "Duplicate keywords make cross-contamination undetectable."
        )
    for outer in cases:
        for inner in cases:
            if outer is not inner and outer.keyword in inner.keyword:
                raise SystemExit(
                    f"keyword {outer.keyword!r} is a substring of {inner.keyword!r}. "
                    "Substring overlap causes false contamination hits. Rename the files."
                )
    if len(cases) < concurrency:
        print(
            f"[warn] only {len(cases)} audio files for concurrency {concurrency}; "
            "files will be reused within an iteration, which WEAKENS detection "
            "(two slots sharing a keyword cannot contaminate each other visibly). "
            "Provide at least as many distinct files as the concurrency."
        )


def run_iteration(
    base_url: str,
    cases: list[AudioCase],
    concurrency: int,
    api_key: Optional[str],
    timeout: float,
) -> tuple[list[Response], list[str]]:
    chosen = [cases[i % len(cases)] for i in range(concurrency)]
    random.shuffle(chosen)
    all_keywords = [case.keyword for case in chosen]

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        # Submit all before collecting any: the requests must be in flight
        # together or there is no race to observe.
        futures = [pool.submit(transcribe, base_url, case, api_key, timeout) for case in chosen]
        responses = [future.result() for future in futures]

    verdicts: list[str] = []
    for response in responses:
        if response.error is not None:
            verdicts.append(MISSING_OWN)
            continue
        others = [kw for kw in all_keywords if kw != response.case.keyword]
        verdict, _foreign = check_integrity(response.text, response.case.keyword, others)
        verdicts.append(verdict)
    return responses, verdicts


DETECTS_BUG_RUNBOOK = """
================================================================================
MANDATORY MUST-FAIL RUNBOOK — do this BEFORE trusting any PASS from this script
================================================================================
A passing test that has never failed is not evidence. It may be passing because
the guard works, or because the harness is broken, the audio is too short to
overlap, the concurrency never actually overlapped, or the keywords never
appear. You cannot tell those apart from a green run. So force a red one.

STEP 1 — reintroduce the bug, and watch this script FAIL.
  In `app/services/asr/qwen3_vllm.py`, in `_run_generate`, comment out the
  guard and dedent its body:

      # with self._llm_lock:
      outputs = self.llm.generate(...)        # <- dedented, now unguarded

  (Comment out the `with self._llm_lock:` at the aligner/tokenizer call site
  too if the run stays green — the mixing path is the generate call.)
  Rebuild/redeploy, then:

      python scripts/h100/test_offline_mixing.py --iterations 20 --concurrency 8

  EXPECT: FAIL, exit 1, with FOREIGN/BOTH verdicts naming another request's
  keyword. Save the output. THIS OUTPUT IS THE EVIDENCE. A PASS here means the
  test does not detect the bug -- see troubleshooting below.

STEP 2 — restore the real build, and watch it PASS.
      git checkout app/services/asr/qwen3_vllm.py
  Rebuild/redeploy, then run the exact same command.
  EXPECT: 20/20 PASS, exit 0.

STEP 3 — record BOTH outputs. Step 2 alone is not a result. The pair is.

RESTORE THE GUARD. Do not leave a build with the lock removed anywhere near a
deploy path.

IF STEP 1 PASSES (the test failed to detect the bug) — troubleshoot, do not
proceed:
  - Are the requests actually concurrent? Check the server log timestamps and
    the per-request elapsed times printed below; if they are serialized end to
    end, something upstream (an admission semaphore set to 1, a proxy, a single
    worker) is serializing them and no race can occur.
  - Is the audio long enough? Very short clips may complete inside one engine
    step and never overlap. Use clips of several seconds or more.
  - Raise --concurrency and --iterations.
  - Are the keywords actually produced at all? Run with `--concurrency 1`. If
    that reports MISSING_OWN, the model is not producing your keywords and
    every result from this script is meaningless.

THE RESULT IS EVIDENCE, NOT PROOF. The race is probabilistic. 20/20 PASS means
"no contamination observed in 20 rounds under this load", not "the race cannot
happen". Write it up that way.
================================================================================
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=20, help="rounds of concurrent requests (default 20)")
    parser.add_argument("--concurrency", type=int, default=8, help="concurrent requests per round (default 8)")
    parser.add_argument("--timeout", type=float, default=600.0, help="per-request timeout seconds (default 600)")
    parser.add_argument("--base-url", default=os.environ.get("ASR_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--audio-dir", default=os.environ.get("ASR_TEST_AUDIO_DIR", ""))
    parser.add_argument(
        "--check-detects-bug",
        action="store_true",
        help="print the mandatory must-fail runbook and exit",
    )
    args = parser.parse_args()

    if args.check_detects_bug:
        print(DETECTS_BUG_RUNBOOK)
        return 0

    if not args.audio_dir:
        print("ASR_TEST_AUDIO_DIR is not set (or pass --audio-dir).", file=sys.stderr)
        return 2
    if args.iterations < 20:
        print(
            f"[warn] --iterations={args.iterations} is below the required minimum of 20. "
            "The race is probabilistic; fewer rounds is not evidence."
        )

    try:
        cases = load_cases(args.audio_dir)
        validate_cases(cases, args.concurrency)
    except SystemExit as exc:
        print(f"SETUP ERROR: {exc}", file=sys.stderr)
        return 2

    print("=" * 78)
    print("Offline mixing regression — H100 / deployed service only")
    print("=" * 78)
    print(f"base url    : {args.base_url}")
    print(f"audio dir   : {args.audio_dir}")
    print(f"cases       : {[(c.label, c.keyword) for c in cases]}")
    print(f"iterations  : {args.iterations}   concurrency: {args.concurrency}")
    print()
    print("REMINDER: a PASS here is meaningless unless you have ALSO seen this")
    print("script FAIL against a build with `_llm_lock` removed from")
    print("`_run_generate`. Run --check-detects-bug for the mandatory runbook.")
    print()

    totals: dict[str, int] = {}
    failed_iterations = 0
    errors: list[str] = []

    for index in range(1, args.iterations + 1):
        responses, verdicts = run_iteration(
            args.base_url, cases, args.concurrency, os.environ.get("ASR_API_KEY"), args.timeout
        )
        counts = summarize(verdicts)
        for key, value in counts.items():
            totals[key] = totals.get(key, 0) + value

        bad = is_contaminated(counts) or counts.get(MISSING_OWN, 0) > 0
        if bad:
            failed_iterations += 1
        status = "FAIL" if bad else "PASS"
        slowest = max((r.elapsed_s for r in responses), default=0.0)
        fastest = min((r.elapsed_s for r in responses), default=0.0)
        print(
            f"iteration {index:3d}/{args.iterations}: {status}  "
            f"ok={counts[OK]} missing_own={counts[MISSING_OWN]} "
            f"foreign={counts[FOREIGN]} both={counts[BOTH]}  "
            f"elapsed {fastest:.2f}-{slowest:.2f}s"
        )
        if bad:
            for response, verdict in zip(responses, verdicts):
                if verdict == OK:
                    continue
                if response.error is not None:
                    detail = f"request error: {response.error}"
                    errors.append(detail)
                else:
                    others = [c.keyword for c in cases if c.keyword != response.case.keyword]
                    _v, foreign = check_integrity(response.text, response.case.keyword, others)
                    detail = (
                        f"expected {response.case.keyword!r}; "
                        f"foreign keywords found: {foreign or 'none'}; "
                        f"text={response.text[:160]!r}"
                    )
                print(f"    [{verdict}] {Path(response.case.path).name}: {detail}")

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"iterations run          : {args.iterations} x {args.concurrency} concurrent requests")
    print(f"iterations with a fault : {failed_iterations}")
    print(f"responses OK            : {totals.get(OK, 0)}")
    print(f"responses MISSING_OWN   : {totals.get(MISSING_OWN, 0)}")
    print(f"responses FOREIGN       : {totals.get(FOREIGN, 0)}   <- mixing signature")
    print(f"responses BOTH          : {totals.get(BOTH, 0)}   <- mixing signature")
    if errors:
        print(f"transport/HTTP errors   : {len(errors)} (first: {errors[0][:200]})")

    contaminated = is_contaminated(totals)
    if contaminated:
        print()
        print("VERDICT: CONTAMINATION OBSERVED. A response carried another request's")
        print("keyword. This is the mixing bug. It is a positive detection and is")
        print("conclusive in the direction of failure — unlike a PASS, one FAIL is enough.")
        return 1

    if totals.get(MISSING_OWN, 0) > 0:
        print()
        print("VERDICT: NO foreign keywords, but some responses lacked their OWN keyword.")
        print("This is NOT proof of mixing — the model may simply have transcribed poorly,")
        print("or a request errored. But a truncating `zip` ALSO produces exactly this.")
        print("Investigate before recording a pass: run with --concurrency 1 and see")
        print("whether the same files still miss their keyword. If they do, the audio or")
        print("the keywords are wrong and this whole run is uninformative.")
        return 1

    print()
    print("VERDICT: no contamination observed.")
    print()
    print("READ THIS BEFORE RECORDING IT AS A PASS:")
    print(f"  * This is EVIDENCE, NOT PROOF. {args.iterations} rounds without contamination")
    print("    does not make the race impossible; it is probabilistic.")
    print("  * This result is worthless on its own unless you have ALSO recorded a")
    print("    FAIL from a build with `_llm_lock` removed (--check-detects-bug).")
    print("    Without that pair, a green run may just mean the test cannot fail.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
