#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H100-ONLY. Before/after entity-accuracy and prompt-leak comparison for the
context wrapper wording.

WHY THIS EXISTS
---------------
The context wrapper in _build_chat_prompt changed from
"Use this context when resolving named entities:" to
"Use this context when transcribing:". Whether the new wording steers entity
recognition as well as the old one CANNOT be established by reading code
(recorded lesson: perf-estimates-from-code-are-unreliable). This script
measures it. The wording is compiled into the deployed build, so you run the
script once against the OLD build and once against the NEW build, then compare.

THE METRIC MUST NOT REWARD PROMPT LEAK
--------------------------------------
The spec's named failure mode is the model echoing the context INTO the
transcript instead of using it as a hint. Under the `hotwords` condition every
request carries every keyword, so a leaking build makes every transcript
contain every keyword — a naive `keyword in text` metric scores that ~100% and
the gate passes hardest exactly when the change is worst. So, same rule as
test_offline_mixing.py's contamination check:

  hit  = clip's OWN keyword present AND NO foreign keyword present
  leak = any OTHER clip's keyword present (the echo signature)

leak_rate is reported per condition and a leak increase on `hotwords` BLOCKS
in `compare`. Keywords must be mutually exclusive across clips (enforced), or
a leak hit is a false positive.

Matching is casefold()ed on both sides (Latin names come back case-shifted).
Casefold does NOT undo ITN rewriting — choose alphabetic proper nouns as
keywords, never numbers or dates.

USAGE
-----
  # against a running service (wording = whatever build is deployed):
  python scripts/h100/bench_context_prompt.py run \
      --audio-dir /path/to/clips --out /tmp/wording_old.json

  # after redeploying the other build:
  python scripts/h100/bench_context_prompt.py run \
      --audio-dir /path/to/clips --out /tmp/wording_new.json

  # verdict:
  python scripts/h100/bench_context_prompt.py compare \
      /tmp/wording_old.json /tmp/wording_new.json

INPUTS
------
  ASR_BASE_URL  default http://localhost:8000
  ASR_API_KEY   optional; sent as `Authorization: Bearer <key>`
  Audio files:  `<label>_<keyword>.<ext>` — keyword must be a named entity
                actually SPOKEN in the clip. Keywords should be entities the
                model plausibly misspells without a hint (proper nouns,
                domain jargon), or the measurement has no headroom.

Each clip is transcribed under four conditions per iteration:
  none      no context at all (floor / sanity)
  hotwords  context = all keywords space-joined (the vocabulary_id shape).
            This is the BLOCKING condition.
  sentence  context = --sentence-context with {keyword} replaced per clip
            (keyword-in-a-sentence steering)
  topic     context = --topic-context, one keyword-free background sentence
            shared by all clips — the spec's motivating use case. Measured
            and recorded; NON-blocking (the spec claims no improvement).

Exit codes: run — 0 ok, 2 setup error. compare — 0 PASS, 1 FAIL (blocked),
2 harness error (e.g. mismatched clip sets between the two runs).
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from urllib import request as urlrequest

DEFAULT_BASE_URL = "http://localhost:8000"
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".mp4", ".webm"}
CONDITIONS = ("none", "hotwords", "sentence", "topic")
# Hit-rate tolerance band for the blocking rule; rationale in the plan
# (Task 4 Step 4): with ~100 Bernoulli trials per side, a strict `new >= old`
# rule fails ~half the time on truly identical wordings.
HIT_RATE_TOLERANCE = 0.10

# Float artifact guard: BOTH sides of the band comparison need rounding.
# The threshold side: 0.80 - 0.10 == 0.7000000000000001. The candidate side:
# avg() accumulates error, e.g. (0.70 + 0.70 + 0.70) / 3 == 0.6999999999999998,
# so a multi-clip candidate at exactly the documented band edge (80% -> 70%)
# would block against the stated ">= baseline - 0.10" rule. A single-clip run
# has no accumulation error, which masks the bug in small-fixture tests —
# real runs (>= 10 clips) hit it. Round both sides before comparing.
_BAND_EPSILON_PLACES = 9


def _contains(text: str, keyword: str) -> bool:
    return keyword.casefold() in text.casefold()


def classify(text: str, own_keyword: str, foreign_keywords: list[str]) -> tuple[bool, list[str]]:
    """(hit, leaked_keywords). A leaked response is never a hit — counting it
    would invert the gate (see module docstring). Pure; mirror of
    test_offline_mixing.check_integrity."""
    leaked = [k for k in foreign_keywords if k != own_keyword and _contains(text, k)]
    hit = _contains(text, own_keyword) and not leaked
    return hit, leaked


def load_cases(audio_dir: str) -> list[tuple[Path, str]]:
    root = Path(audio_dir)
    # Setup errors exit 2 per this script's contract; a bare iterdir() on a
    # missing dir would raise and exit 1, which the docstring does not promise.
    if not root.is_dir():
        print(f"[setup] --audio-dir is not a directory: {audio_dir}", file=sys.stderr)
        sys.exit(2)
    cases = []
    for path in sorted(root.iterdir()):
        if path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        stem = path.stem
        if "_" not in stem:
            print(f"[skip] {path.name}: no _keyword in filename", file=sys.stderr)
            continue
        keyword = stem.rsplit("_", 1)[1]
        cases.append((path, keyword))
    return cases


def transcribe(base_url: str, api_key: str, path: Path, prompt: str, timeout: float) -> str:
    boundary = uuid.uuid4().hex
    parts = []
    # NOTE: no `model` field — the endpoint declares none
    # (app/api/v1/openai_compatible.py:463-503); sending one would imply
    # model selection works when it does not exist.
    fields = {"response_format": "json"}
    if prompt:
        fields["prompt"] = prompt
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        )
    parts.append(
        (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{path.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + path.read_bytes()
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urlrequest.Request(
        f"{base_url.rstrip('/')}/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8")).get("text", "")


def cmd_run(args: argparse.Namespace) -> int:
    cases = load_cases(args.audio_dir)
    if len(cases) < 3:
        print(f"need >= 3 keyword-named clips, found {len(cases)}", file=sys.stderr)
        return 2
    keywords = [keyword for _path, keyword in cases]
    duplicates = {k for k in keywords if keywords.count(k) > 1}
    if duplicates:
        print(f"keywords must be mutually exclusive; duplicates: {sorted(duplicates)}", file=sys.stderr)
        return 2
    for outer in keywords:
        for inner in keywords:
            if outer != inner and _contains(inner, outer):
                print(
                    f"keyword {outer!r} is a substring of {inner!r}; "
                    "substring overlap causes false leak hits. Rename the files.",
                    file=sys.stderr,
                )
                return 2
    named = [k for k in keywords if _contains(args.topic_context, k)]
    if named:
        print(
            f"--topic-context must name NO keyword (it measures topic-only "
            f"background) but contains: {named}",
            file=sys.stderr,
        )
        return 2
    hotword_list = " ".join(keywords)
    api_key = os.environ.get("ASR_API_KEY", "")
    results = {c: {} for c in CONDITIONS}
    for iteration in range(args.iterations):
        for path, keyword in cases:
            foreign = [k for k in keywords if k != keyword]
            contexts = {
                "none": "",
                "hotwords": hotword_list,
                # str.replace, not str.format: a user-supplied template with a
                # literal `{` or `}` must not raise KeyError/ValueError.
                "sentence": args.sentence_context.replace("{keyword}", keyword),
                "topic": args.topic_context,
            }
            for condition, prompt in contexts.items():
                text = transcribe(args.base_url, api_key, path, prompt, args.timeout)
                hit, leaked = classify(text, keyword, foreign)
                results[condition].setdefault(path.name, []).append(
                    {"hit": hit, "leak": bool(leaked)}
                )
                leak_note = f" LEAK={leaked}" if leaked else ""
                print(
                    f"[iter {iteration + 1}/{args.iterations}] {path.name} "
                    f"{condition:8s} hit={hit}{leak_note} text={text[:80]!r}"
                )
                time.sleep(args.pause)

    def rate(per_clip: dict, key: str) -> dict:
        return {name: sum(obs[key] for obs in observations) / len(observations)
                for name, observations in per_clip.items()}

    summary = {
        "base_url": args.base_url,
        "audio_dir": str(args.audio_dir),
        "iterations": args.iterations,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
        "hit_rate": {c: rate(results[c], "hit") for c in CONDITIONS},
        "leak_rate": {c: rate(results[c], "leak") for c in CONDITIONS},
    }
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    for condition in CONDITIONS:
        hits = list(summary["hit_rate"][condition].values())
        leaks = list(summary["leak_rate"][condition].values())
        print(
            f"  {condition:8s} hit rate: {sum(hits) / len(hits):.2%}   "
            f"leak rate: {sum(leaks) / len(leaks):.2%}"
        )
    return 0


class _CompareInputError(Exception):
    """A harness/setup problem with `compare`'s inputs. Never a verdict:
    cmd_compare translates this to exit 2, so exit 1 stays reserved for a
    genuine computed regression — a harness error must never impersonate a
    verdict."""


def _load_summary(label: str, path_str: str) -> dict:
    """Load and structurally validate one `run` output file. Any failure —
    unreadable path, non-JSON content, or a shape `run` would never emit —
    raises _CompareInputError (exit 2), not a traceback (exit 1)."""
    try:
        doc = json.loads(Path(path_str).read_text(encoding="utf-8"))
    except OSError as exc:
        raise _CompareInputError(f"cannot read {label} file {path_str}: {exc}")
    except ValueError as exc:
        # ValueError covers both json.JSONDecodeError (not JSON) and
        # UnicodeDecodeError (not UTF-8 text, e.g. an audio file passed by
        # mistake). Both are harness errors, never a verdict.
        raise _CompareInputError(f"{label} file {path_str} is not valid JSON: {exc}")
    if not isinstance(doc, dict) or "timestamp" not in doc:
        raise _CompareInputError(
            f"MALFORMED {label}: not a `run` summary (missing 'timestamp'). "
            "Both files must come from this script's `run`; no verdict."
        )
    for condition in CONDITIONS:
        # A missing/mis-typed condition table is a malformed file, not a
        # verdict. Guard it here: indexing it in the scoring loop would raise
        # and exit 1, which this plan defines as "regression confirmed, do
        # not merge".
        missing = [
            k for k in ("hit_rate", "leak_rate")
            if not isinstance(doc.get(k), dict) or not isinstance(doc[k].get(condition), dict)
        ]
        if missing:
            raise _CompareInputError(
                f"MALFORMED {label}: {condition!r} absent (or not a per-clip "
                f"table) in {missing}. Both files must come from this "
                "script's `run`; no verdict."
            )
        for key in ("hit_rate", "leak_rate"):
            # An EMPTY per-clip table is a shape `run` can never emit (it
            # enforces >= 3 clips at startup) — and letting it through would
            # let a zero-data file score as PASS. A verdict, especially on the
            # blocking `hotwords` condition, must never be computed from zero
            # observations; no verdict, exit 2.
            if not doc[key][condition]:
                raise _CompareInputError(
                    f"MALFORMED {label}: {key} for {condition!r} has no clips. "
                    "A real `run` records >= 3 clips per condition; a PASS "
                    "must never be computed from zero observations. No verdict."
                )
            bad = [
                clip for clip, value in doc[key][condition].items()
                if not isinstance(value, (int, float)) or isinstance(value, bool)
            ]
            if bad:
                raise _CompareInputError(
                    f"MALFORMED {label}: non-numeric {key} for clips {sorted(bad)} "
                    f"in {condition!r}; no verdict."
                )
        if set(doc["hit_rate"][condition]) != set(doc["leak_rate"][condition]):
            raise _CompareInputError(
                f"MALFORMED {label}: hit_rate and leak_rate list different "
                f"clips in {condition!r}. Both files must come from this "
                "script's `run`; no verdict."
            )
    return doc


def cmd_compare(args: argparse.Namespace) -> int:
    try:
        old = _load_summary("baseline", args.baseline)
        new = _load_summary("candidate", args.candidate)
    except _CompareInputError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"baseline : {args.baseline} ({old['timestamp']})")
    print(f"candidate: {args.candidate} ({new['timestamp']})")
    # Refuse mismatched clip sets rather than scoring a missing clip as 0.0,
    # which would manufacture a fake regression (or hide a real one). Checked
    # on BOTH tables: _load_summary guarantees hit_rate/leak_rate agree
    # within one file, and this guarantees they agree across the two files.
    for condition in CONDITIONS:
        for key in ("hit_rate", "leak_rate"):
            old_clips = set(old[key][condition])
            new_clips = set(new[key][condition])
            if old_clips != new_clips:
                print(
                    f"CLIP SET MISMATCH in {condition!r} ({key}): "
                    f"only in baseline {sorted(old_clips - new_clips)}, "
                    f"only in candidate {sorted(new_clips - old_clips)}. "
                    "Both runs must use the identical audio dir; rerun.",
                    file=sys.stderr,
                )
                return 2

    def avg(rates: dict) -> float:
        return sum(rates.values()) / len(rates) if rates else 0.0

    verdict_pass = True
    for condition in CONDITIONS:
        old_hit, new_hit = avg(old["hit_rate"][condition]), avg(new["hit_rate"][condition])
        old_leak, new_leak = avg(old["leak_rate"][condition]), avg(new["leak_rate"][condition])
        marker = ""
        if condition == "hotwords":
            if new_leak > old_leak:
                verdict_pass = False
                marker += "  <-- BLOCKING: leak rate increased (context echoed into transcript)"
            if round(new_hit, _BAND_EPSILON_PLACES) < round(
                old_hit - HIT_RATE_TOLERANCE, _BAND_EPSILON_PLACES
            ):
                verdict_pass = False
                marker += "  <-- BLOCKING: hit rate dropped beyond the 10pp band"
        print(
            f"\n{condition}: hit {old_hit:.2%} -> {new_hit:.2%}   "
            f"leak {old_leak:.2%} -> {new_leak:.2%}{marker}"
        )
        for clip in sorted(old["hit_rate"][condition]):
            o, n = old["hit_rate"][condition][clip], new["hit_rate"][condition][clip]
            ol, nl = old["leak_rate"][condition][clip], new["leak_rate"][condition][clip]
            flag = " !" if (n < o or nl > ol) else ""
            print(f"    {clip}: hit {o:.2%} -> {n:.2%}  leak {ol:.2%} -> {nl:.2%}{flag}")
    print(
        "\nVERDICT:",
        "PASS — no leak increase and hotword hit rate within the 10pp band"
        if verdict_pass
        else "FAIL — hotword condition regressed (leak and/or hit rate); do not merge",
    )
    return 0 if verdict_pass else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--base-url", default=os.environ.get("ASR_BASE_URL", DEFAULT_BASE_URL))
    run_p.add_argument("--audio-dir", required=True)
    run_p.add_argument("--iterations", type=int, default=10)
    run_p.add_argument("--timeout", type=float, default=120.0)
    run_p.add_argument("--pause", type=float, default=0.5)
    run_p.add_argument(
        "--sentence-context",
        default="Topik rekaman: {keyword}.",
        help="per-clip context; the literal text {keyword} is replaced with the clip's keyword",
    )
    run_p.add_argument(
        "--topic-context",
        default="The recording is a business news broadcast about companies and markets.",
        help="keyword-free background context shared by all clips (topic-only condition)",
    )
    run_p.add_argument("--out", required=True)
    run_p.set_defaults(func=cmd_run)
    cmp_p = sub.add_parser("compare")
    cmp_p.add_argument("baseline")
    cmp_p.add_argument("candidate")
    cmp_p.set_defaults(func=cmd_compare)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
