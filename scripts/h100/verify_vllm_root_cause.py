#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H100-ONLY probe. Run inside the deployed image (vllm[audio]==0.19.0).

WHAT THIS IS
------------
The concurrency design for this repo rests on an UNVERIFIED hypothesis about
vLLM 0.19.0. This script is the instrument that confirms or refutes it. It
does not assume the hypothesis is true, and it is designed to be capable of
returning REFUTED.

THE HYPOTHESIS (Q-a)
--------------------
`vllm.entrypoints.llm.LLM.generate` adds every prompt to a shared `LLMEngine`
and then drains that engine in a loop, collecting whatever finishes, WITHOUT
filtering the collected outputs down to the request IDs that this particular
`generate()` call submitted. If that is how it works, two threads calling
`generate()` on one `LLM` each collect a mix of both callers' outputs, and
`Qwen3VLLMBackend._run_generate`'s positional `zip(outputs, audio_items)`
silently pairs the wrong transcript with the wrong audio. Wrong text, no
exception.

IMPORTANT CAVEAT — DO NOT PATTERN-MATCH
---------------------------------------
Upstream historically SORTS the drained outputs by request id before
returning. Sorting does NOT rescue concurrent callers (the engine's request-id
counter is shared, so ownership still interleaves) but it DOES mean the drain
loop is not literally shaped like the hypothesis's pseudocode. Seeing a sort
is therefore NOT evidence of filtering, and its absence is not evidence of the
hypothesis. This probe classifies on the presence or absence of an
OWNERSHIP FILTER only, and prints the real source so a human can overrule it.

DECISION RULE (stated up front so the reader need not interpret)
---------------------------------------------------------------
Let D be the drain loop that `generate()` relies on (`LLM._run_engine`, or
whatever `generate` actually calls at this version).

  CONFIRMED    D appends engine step outputs to the returned list based only
               on "is this output finished?", with no comparison against a
               set/list/dict of request IDs captured by THIS generate() call.
  REFUTED      D restricts what it returns to this caller's own request IDs
               (an explicit request-id membership test, a per-caller output
               map keyed by the ids this call added, or a per-caller queue).
  INCONCLUSIVE Source unavailable, or its shape matches neither pattern.
               INCONCLUSIVE is a real outcome. Do not "round" it to
               CONFIRMED. Read the printed source and decide by hand.

Sorting by request id, batching, tqdm bookkeeping, and `n`-sampling fan-out
are all IRRELEVANT to this rule and are ignored.

ALSO ANSWERED
-------------
Q-b: does `vllm.v1.engine.async_llm.AsyncLLM` import at this version, and does
     the `AsyncLLMEngine` compatibility alias still exist? (Upstream carries a
     TODO to delete the proxy; the v0 engine is gone.)
Q-c: does the async engine expose POOLING / `encode`? The forced aligner uses
     `runner="pooling"` + `aligner.encode(...)`. If async pooling does not
     exist, the aligner CANNOT migrate to the async engine and must stay
     serialized behind its lock, regardless of Q-a's verdict.

USAGE
-----
    python scripts/h100/verify_vllm_root_cause.py | tee /tmp/vllm_root_cause.txt

Exit codes: 0 = CONFIRMED, 1 = REFUTED, 2 = INCONCLUSIVE, 3 = probe error.

Self-contained: stdlib + vllm only. Imports nothing from this repo.
"""

from __future__ import annotations

import inspect
import re
import sys
import traceback
from typing import Any, Optional

CONFIRMED = "CONFIRMED"
REFUTED = "REFUTED"
INCONCLUSIVE = "INCONCLUSIVE"

_EXIT_CODES = {CONFIRMED: 0, REFUTED: 1, INCONCLUSIVE: 2}


# --------------------------------------------------------------------------
# Pure classification logic (unit-tested locally; no vllm import needed).
# --------------------------------------------------------------------------

# An ownership filter: the drain restricts results to ids THIS call submitted.
# Each entry is (regex, human-readable description).
_FILTER_PATTERNS: list[tuple[str, str]] = [
    (
        r"\.request_id\s+(?:not\s+)?in\s+\w+",
        "membership test of an output's request_id against a caller-held collection",
    ),
    (
        r"if\s+\w*(?:req|request)_ids?\b[^\n]*\bin\b",
        "conditional keyed on a request-id collection",
    ),
    (
        r"\brequest_ids\s*=\s*(?:set|frozenset)\(",
        "generate() captures its own request ids into a set for later filtering",
    ),
    (
        r"\.pop\(\s*\w*(?:output|out)\.request_id\s*\)",
        "per-caller output map keyed by request_id",
    ),
    (
        r"\bper_caller\b|\bcaller_queue\b|\boutput_queue\s*\[",
        "per-caller output queue",
    ),
]

# The unfiltered drain: collect anything the engine reports as finished.
_UNFILTERED_PATTERNS: list[tuple[str, str]] = [
    (
        r"if\s+\w*(?:output|out)\.finished\s*:",
        "outputs are collected on `finished` alone, with no ownership check",
    ),
    (
        r"\bstep_outputs\b",
        "the loop appends raw engine step outputs",
    ),
    (
        r"outputs\.append\(\s*\w*(?:output|out)\s*\)",
        "engine step outputs are appended verbatim to the returned list",
    ),
]

# Informational only. Deliberately NOT part of the verdict — see caveat above.
_SORT_PATTERNS: list[tuple[str, str]] = [
    (
        r"\.sort\(\s*key\s*=[^\n]*request_id",
        "outputs sorted by request_id before return (does NOT rescue concurrent callers)",
    ),
    (
        r"sorted\(\s*outputs[^\n]*request_id",
        "outputs sorted by request_id before return (does NOT rescue concurrent callers)",
    ),
]


def _scan(source: str, patterns: list[tuple[str, str]]) -> list[str]:
    """Return descriptions of every pattern that matches `source`."""
    hits: list[str] = []
    for pattern, description in patterns:
        match = re.search(pattern, source)
        if match:
            hits.append(f"{description}  [matched: {match.group(0).strip()!r}]")
    return hits


def classify_drain(source: Optional[str]) -> tuple[str, list[str], list[str]]:
    """Classify the drain-loop source against the decision rule above.

    Returns (verdict, evidence, notes). Pure: takes source text, no imports.

    An ownership filter always wins: if the drain restricts results to the
    caller's own request ids, the hypothesis is refuted no matter what else
    the loop does.
    """
    notes: list[str] = []
    if not source or not source.strip():
        return INCONCLUSIVE, ["No drain-loop source was available to read."], notes

    filter_hits = _scan(source, _FILTER_PATTERNS)
    unfiltered_hits = _scan(source, _UNFILTERED_PATTERNS)
    notes.extend(_scan(source, _SORT_PATTERNS))

    if filter_hits:
        evidence = ["Ownership filter present:"] + [f"  - {hit}" for hit in filter_hits]
        if unfiltered_hits:
            evidence.append(
                "  (unfiltered-shaped lines also matched, but an explicit "
                "ownership filter overrides them)"
            )
        return REFUTED, evidence, notes

    if unfiltered_hits:
        evidence = [
            "No ownership filter found, and the drain has the unfiltered shape:",
        ] + [f"  - {hit}" for hit in unfiltered_hits]
        return CONFIRMED, evidence, notes

    return (
        INCONCLUSIVE,
        [
            "The drain source matched neither an ownership filter nor the "
            "unfiltered-collection shape. Its structure is unrecognized — read "
            "the printed source and decide by hand. Do NOT default to CONFIRMED."
        ],
        notes,
    )


# --------------------------------------------------------------------------
# H100-only probing (requires a real vllm install).
# --------------------------------------------------------------------------


def _safe_source(obj: Any, label: str) -> Optional[str]:
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError) as exc:
        print(f"  !! could not read source of {label}: {exc}")
        return None


def _print_block(title: str, body: str) -> None:
    print(f"\n===== {title} =====")
    print(body)


def probe_q_a() -> tuple[str, list[str], list[str]]:
    """Q(a): does the generate() drain filter to the caller's request ids?"""
    from vllm.entrypoints.llm import LLM

    generate_src = _safe_source(LLM.generate, "LLM.generate")
    if generate_src:
        _print_block("LLM.generate source", generate_src)

    # The drain may not be named _run_engine at this version. Find what
    # generate actually calls rather than assuming the name.
    drain_candidates = ("_run_engine", "_run_engine_v1", "_engine_step", "_drain")
    drain_name: Optional[str] = None
    if generate_src:
        for name in drain_candidates:
            if f"self.{name}(" in generate_src:
                drain_name = name
                break
    if drain_name is None:
        for name in drain_candidates:
            if getattr(LLM, name, None) is not None:
                drain_name = name
                print(
                    f"\n[warn] generate() does not visibly call any known drain; "
                    f"falling back to LLM.{name} by name. Verify this is really "
                    f"the drain before trusting the verdict."
                )
                break

    drain_src: Optional[str] = None
    if drain_name is not None:
        drain_src = _safe_source(getattr(LLM, drain_name), f"LLM.{drain_name}")
        if drain_src:
            _print_block(f"LLM.{drain_name} source (THE DRAIN — verdict is based on this)", drain_src)
    else:
        print(
            "\n[warn] No drain method found on LLM. Verdict will be INCONCLUSIVE; "
            "read LLM.generate above and trace the drain by hand."
        )

    # Context only: how requests get ids. Not part of the verdict, but it is
    # what makes sorting insufficient (shared counter => interleaved ids).
    for name in ("_validate_and_add_requests", "_add_request"):
        fn = getattr(LLM, name, None)
        if fn is not None:
            src = _safe_source(fn, f"LLM.{name}")
            if src:
                _print_block(f"LLM.{name} source (context: request-id assignment)", src)

    return classify_drain(drain_src)


def probe_q_b() -> None:
    """Q(b): AsyncLLM import + the AsyncLLMEngine compatibility alias."""
    try:
        from vllm.v1.engine.async_llm import AsyncLLM

        print(f"  AsyncLLM import: OK -> {AsyncLLM}")
    except Exception as exc:
        print(f"  AsyncLLM import: FAILED -> {exc!r}")

    try:
        from vllm import AsyncLLMEngine

        print(f"  AsyncLLMEngine alias: OK -> {AsyncLLMEngine}")
        try:
            from vllm.v1.engine.async_llm import AsyncLLM as _AsyncLLM

            same = AsyncLLMEngine is _AsyncLLM
            print(f"  AsyncLLMEngine is vllm.v1.engine.async_llm.AsyncLLM: {same}")
            if not same:
                print(
                    f"  (alias resolves elsewhere: {getattr(AsyncLLMEngine, '__module__', '?')}"
                    f".{getattr(AsyncLLMEngine, '__qualname__', '?')} — it may be a proxy/subclass)"
                )
        except Exception:
            pass
    except Exception as exc:
        print(f"  AsyncLLMEngine alias: FAILED (proxy may have been removed) -> {exc!r}")


def probe_q_c() -> None:
    """Q(c): pooling / encode surface on the async engine.

    The forced aligner uses runner="pooling" + encode(pooling_task=...). If
    AsyncLLM has no encode, the aligner cannot migrate and stays serialized.
    """
    try:
        from vllm.v1.engine.async_llm import AsyncLLM
    except Exception as exc:
        print(f"  cannot inspect AsyncLLM: {exc!r}")
        return

    interesting = ("generate", "encode", "pooling", "add_request")
    present = [name for name in interesting if hasattr(AsyncLLM, name)]
    print(f"  AsyncLLM members present: {present}")
    print(f"  AsyncLLM members ABSENT:  {[n for n in interesting if n not in present]}")

    encode = getattr(AsyncLLM, "encode", None)
    if encode is None:
        print(
            "  => AsyncLLM has NO encode(). The forced aligner CANNOT migrate to "
            "the async engine; it must stay behind its serializing lock."
        )
        return

    try:
        signature = inspect.signature(encode)
        print(f"  AsyncLLM.encode signature: {signature}")
        params = list(signature.parameters)
        print(f"  encode accepts pooling_task: {'pooling_task' in params}")
    except (TypeError, ValueError) as exc:
        print(f"  could not read encode signature: {exc}")

    src = _safe_source(encode, "AsyncLLM.encode")
    if src:
        _print_block("AsyncLLM.encode source", src)
        # Does the async path reject multimodal or pooling outright?
        for marker in ("NotImplementedError", "not supported", "multi_modal", "pooling_task"):
            if marker in src:
                print(f"  [note] AsyncLLM.encode source mentions {marker!r} — read the source above.")


def main() -> int:
    print("=" * 72)
    print("vLLM root-cause probe — H100 / deployed image only")
    print("=" * 72)
    try:
        import vllm

        print(f"vllm version: {getattr(vllm, '__version__', 'UNKNOWN')}")
        if getattr(vllm, "__version__", "") != "0.19.0":
            print(
                "[warn] This probe's verdict is only claimed for 0.19.0. The "
                "installed version differs — record the actual version with the verdict."
            )
    except Exception as exc:
        print(f"FATAL: vllm is not importable here: {exc!r}")
        print("This probe MUST run inside the deployed image on the H100.")
        return 3

    print("\n" + "-" * 72)
    print("Q(a) Does LLM.generate's drain filter outputs to the caller's request ids?")
    print("-" * 72)
    try:
        verdict, evidence, notes = probe_q_a()
    except Exception:
        print("Q(a) probe raised:")
        traceback.print_exc()
        verdict, evidence, notes = INCONCLUSIVE, ["The Q(a) probe crashed; see traceback."], []

    print("\n" + "-" * 72)
    print("Q(b) AsyncLLM import / AsyncLLMEngine alias")
    print("-" * 72)
    try:
        probe_q_b()
    except Exception:
        traceback.print_exc()

    print("\n" + "-" * 72)
    print("Q(c) Async pooling / encode with multimodal audio (aligner migration)")
    print("-" * 72)
    try:
        probe_q_c()
    except Exception:
        traceback.print_exc()

    print("\n" + "=" * 72)
    print("DECISION RULE")
    print("  CONFIRMED    drain collects finished outputs with NO check against")
    print("               the request ids this generate() call submitted.")
    print("  REFUTED      drain restricts results to this caller's request ids.")
    print("  INCONCLUSIVE unreadable or unrecognized; decide by hand from the")
    print("               source printed above. Do NOT round up to CONFIRMED.")
    print("  (Sorting by request id is NOT filtering and does not affect this.)")
    print("=" * 72)
    print(f"VERDICT (Q-a, hypothesis that LLM.generate mixes concurrent callers): {verdict}")
    for line in evidence:
        print(f"  {line}")
    for note in notes:
        print(f"  [note] {note}")
    if verdict == REFUTED:
        print(
            "\n  *** REFUTED IS A SIGNIFICANT RESULT ***\n"
            "  The design's stated root cause is WRONG. Before any further work:\n"
            "    1. Changes already landed on the strength of this hypothesis may\n"
            "       be unnecessary — re-review them rather than building on them.\n"
            "    2. Change A's lock is NOT justified by output mixing. It may still\n"
            "       stand on the separate, documented grounds that `LLM` is not\n"
            "       thread-safe and on the tokenizer / ITN / VAD races — but that\n"
            "       must be argued on its own, not inherited from this hypothesis.\n"
            "    3. Change C (async engine) must be RE-SCOPED from scratch; its\n"
            "       premise is void.\n"
            "  Record this verdict in the spec changelog before proceeding."
        )
    if verdict == INCONCLUSIVE:
        print(
            "\n  Do not proceed on a guess. Either read the source above and record\n"
            "  a hand verdict with a quoted excerpt, or fix this probe and re-run."
        )
    print(
        "\nRecord in the spec changelog: the Q-a verdict, the vllm version, and the\n"
        "Q-b / Q-c answers."
    )
    return _EXIT_CODES[verdict]


if __name__ == "__main__":
    sys.exit(main())
