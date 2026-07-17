# Broaden the Transcription Context Prompt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reframe caller-supplied `context` in the vLLM chat prompt as general background ("Use this context when transcribing:") instead of a named-entity list, fix three stale doc/annotation sites on the OpenAI endpoint that claim `prompt` is inert, and **measure** the wording change on real audio before merge.

**Architecture:** One f-string changes in `_build_chat_prompt` (`app/services/asr/qwen3_vllm.py`). Both of its call sites (offline batch at `:283`, realtime/alignment at `:459`) and all three public surfaces (OpenAI `prompt`, Alibaba `vocabulary_id`, WebSocket `context`) inherit the new wording with zero signature changes. A test in `tests/test_vllm_mixing_fidelity.py` uses the old wording as a parsing anchor and must change in the same commit. A new pure-function test file pins the prompt shape. A measurement task on real hardware gates the merge.

**Tech Stack:** Python 3, stdlib. Tests are `unittest`. Measurement uses the running service's HTTP API via `urllib` (same pattern as `scripts/h100/test_offline_mixing.py`).

**Spec:** `docs/superpowers/specs/2026-07-17-transcription-context-prompt-design.md`

## Global Constraints

- **Tests are `unittest`, NOT pytest.** Run: `DEVICE=cpu .venv/bin/python -m unittest discover -s tests`. Do not add pytest or use pytest fixtures (`tmp_path` etc.).
- **Spec items 1 and 2 are ATOMIC — same commit.** `tests/test_vllm_mixing_fidelity.py` exercises the real `_run_generate`; its `_marker()` (`:204-212`) slices context out of the built prompt with `text.index("resolving named entities: ")`. Changing the prompt without the needle makes `.index()` raise `ValueError` and the suite go red. Never leave a commit between them.
- **vLLM/CUDA only.** The CPU/Rust backend discards caller context (`app/services/asr/qwen3_engine.py:234`, `_ = (hotwords, ...)`) and is an explicit non-goal. No Rust work, no changes outside `qwen3_vllm.py` for behavior.
- **No API surface changes.** No new parameters, no renames. `hotwords` internal names and public `vocabulary_id` stay as they are (spec "Deliberately not renamed").
- **Measurement (Task 4) BLOCKS merge.** This repo has a recorded lesson (`bd` knowledge: `perf-estimates-from-code-are-unreliable`) that confident code-reading estimates were wrong three times running. The spec explicitly makes **no claim** that the new wording steers as well as the old, especially for bare hotword lists (`vocabulary_id`). Do not merge to main until Task 4's before/after comparison is recorded and the decision rule passes — both halves of it: no prompt-leak increase AND no hotword hit-rate drop beyond the band (Task 4 Step 4).
- **Exact new string:** `f"Use this context when transcribing: {context.strip()}"`. Copy it verbatim; do not "improve" it.
- **Work on a feature branch.** Tasks 1–3 commit to the branch; the branch merges only after Task 4's verdict.

## Spec coverage

| Spec item | Site (verified 2026-07-17) | Task |
|---|---|---|
| 1. Prompt wording | `app/services/asr/qwen3_vllm.py:81` | 1 |
| 2. Test needle | `tests/test_vllm_mixing_fidelity.py:210` | 1 (same commit) |
| 3. 暂不支持 doc line | `app/api/v1/openai_compatible.py:413` | 2 |
| 4. `_ = (prompt, ...)` annotation | `app/api/v1/openai_compatible.py:506` (`prompt` is genuinely read at `:561`, `hotwords=prompt or ""`) | 2 |
| 5. Form description | `app/api/v1/openai_compatible.py:496` | 2 |
| New `tests/test_chat_prompt.py` (spec "Testing" cases 1–5) | create | 1 |
| Risk: "steering quality is unverified … validation is a blocking task" | `scripts/h100/` runbook | 3 (instrument), 4 (run + verdict) |

Line numbers above were re-verified against the working tree while writing this plan.

---

### Task 1: Prompt wording + test needle + new unit tests (one atomic commit)

**Files:**
- Modify: `app/services/asr/qwen3_vllm.py:81`
- Modify: `tests/test_vllm_mixing_fidelity.py:210`
- Create: `tests/test_chat_prompt.py`

**Interfaces:**
- Consumes: `_build_chat_prompt(context: str = "", language: Optional[str] = None) -> str` (`qwen3_vllm.py:74`) — pure function, unchanged signature.
- Produces: system block now reads `... Use this context when transcribing: <context>`. Task 3's measurement script and Task 4's runbook depend on exactly this string being live in the deployed build.

Import safety (pre-verified, per spec): `qwen3_vllm.py` has no module-level `vllm` import — only `importlib.util.find_spec("vllm")` at `:52` — so `from app.services.asr.qwen3_vllm import _build_chat_prompt` works in the CPU venv. The module does import `librosa`/`numpy`/`app.infrastructure` at import time; that is fine in `.venv`, but it means these tests need the synced venv, not a bare interpreter.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chat_prompt.py`:

```python
# -*- coding: utf-8 -*-
"""Unit tests for _build_chat_prompt (vLLM chat prompt construction).

Pins the prompt shape after the context wrapper was broadened from
"when resolving named entities" to "when transcribing". Pure-function tests;
no vLLM install required (qwen3_vllm has no module-level vllm import).
"""

import unittest

from app.services.asr.qwen3_vllm import _build_chat_prompt


def _system_text(prompt: str) -> str:
    """Slice the system block out of the full chat-template string."""
    start = prompt.index("<|im_start|>system\n") + len("<|im_start|>system\n")
    return prompt[start:prompt.index("<|im_end|>", start)]


class BuildChatPromptTest(unittest.TestCase):
    def test_context_and_language_yields_both_instructions_language_first(self) -> None:
        prompt = _build_chat_prompt(context="Danantara dana kelolaan", language="Indonesian")

        self.assertEqual(
            _system_text(prompt),
            "Transcribe the speech in Indonesian. "
            "Use this context when transcribing: Danantara dana kelolaan",
        )

    def test_context_without_language_uses_accurately_preamble(self) -> None:
        prompt = _build_chat_prompt(context="阿里巴巴 腾讯")

        self.assertEqual(
            _system_text(prompt),
            "Transcribe the speech accurately. "
            "Use this context when transcribing: 阿里巴巴 腾讯",
        )

    def test_empty_and_whitespace_context_emit_no_context_clause(self) -> None:
        for context in ("", "   ", "\n\t"):
            with self.subTest(context=repr(context)):
                prompt = _build_chat_prompt(context=context)

                self.assertEqual(_system_text(prompt), "Transcribe the speech accurately.")
                self.assertNotIn("Use this context", prompt)

    def test_language_alias_id_normalizes_to_indonesian(self) -> None:
        # Callers pass through _normalize_language_name; mirror that here.
        from app.services.asr.qwen3_vllm import _normalize_language_name

        prompt = _build_chat_prompt(context="x", language=_normalize_language_name("id"))

        self.assertIn("Transcribe the speech in Indonesian.", prompt)

    def test_old_named_entity_wording_is_gone(self) -> None:
        prompt = _build_chat_prompt(context="anything", language="Chinese")

        self.assertNotIn("resolving named entities", prompt)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new tests to verify they fail on the wording**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_chat_prompt -v`
Expected: FAIL — `test_context_and_language_yields_both_instructions_language_first`, `test_context_without_language_uses_accurately_preamble`, and `test_old_named_entity_wording_is_gone` fail (the prompt still says `resolving named entities`). The empty-context and language-alias tests already pass; that is expected — they pin behavior that must not change.

If instead every test errors at import: the venv is not synced. Run `./scripts/sync_cpu_env.sh` first — environment problem, not a code problem.

- [ ] **Step 3: Change the prompt string**

In `app/services/asr/qwen3_vllm.py:81`, change:

```python
        instructions.append(f"Use this context when resolving named entities: {context.strip()}")
```

to:

```python
        instructions.append(f"Use this context when transcribing: {context.strip()}")
```

- [ ] **Step 4: Swap the parsing needle in the fidelity test — do NOT skip to running tests first**

In `tests/test_vllm_mixing_fidelity.py:210`, inside `_marker()`, change:

```python
        needle = "resolving named entities: "
```

to:

```python
        needle = "when transcribing: "
```

This is a mechanical anchor swap only. `_marker()` uses the wrapper text purely to slice the caller's context back out of the built prompt to prove which request produced which text; the test's guarantee (no cross-caller mixing) is unchanged.

- [ ] **Step 5: Run both test modules to verify they pass**

Run: `DEVICE=cpu .venv/bin/python -m unittest tests.test_chat_prompt tests.test_vllm_mixing_fidelity -v`
Expected: PASS (all tests in both modules).

- [ ] **Step 6: Run the full suite (no regressions)**

Run: `DEVICE=cpu .venv/bin/python -m unittest discover -s tests`
Expected: same pass/fail set as before this task. In particular, no `ValueError: substring not found` anywhere — that error means Step 3 and Step 4 got out of sync.

- [ ] **Step 7: Commit — all three files together**

```bash
git add app/services/asr/qwen3_vllm.py tests/test_vllm_mixing_fidelity.py tests/test_chat_prompt.py
git commit -m "feat: frame caller context as transcription background, not a named-entity list"
```

---

### Task 2: Fix the OpenAI endpoint's stale `prompt` documentation

The endpoint's own OpenAPI description tells users `prompt` is inert, while line 561 passes it through as `hotwords=prompt or ""`. Pre-existing documentation bug; independent of Task 1 but touches the same parameter. Doc-only — no behavior change, no new tests.

**Files:**
- Modify: `app/api/v1/openai_compatible.py:413,496,506`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: nothing consumed later. Purely OpenAPI/doc text and an unused-var annotation.

- [ ] **Step 1: Drop `prompt` from the unsupported-parameters doc line**

At `app/api/v1/openai_compatible.py:413`, change:

```
`prompt`、`temperature`、`timestamp_granularities` 参数已保留但暂不生效
```

to:

```
`temperature`、`timestamp_granularities` 参数已保留但暂不生效
```

- [ ] **Step 2: Update the `prompt` Form description**

At `app/api/v1/openai_compatible.py:496`, change:

```python
    prompt: Optional[str] = Form(None, description="提示文本，作为命名实体上下文注入转写提示（hotwords）"),
```

to:

```python
    prompt: Optional[str] = Form(None, description="上下文提示文本，作为背景信息注入转写提示（如主题、领域、专有名词）"),
```

- [ ] **Step 3: Remove `prompt` from the unused-parameters annotation**

At `app/api/v1/openai_compatible.py:506`, change:

```python
    _ = (prompt, temperature, timestamp_granularities)
```

to:

```python
    _ = (temperature, timestamp_granularities)
```

`prompt` is genuinely read at `:561` (`hotwords=prompt or ""`), so it must not carry a "deliberately unused" marker. Leave the surrounding comment about 兼容性参数 alone. Do NOT touch `temperature` or `timestamp_granularities` handling — they genuinely are inert (spec non-goal).

- [ ] **Step 4: Verify the module still imports and nothing else references the old text**

Run: `DEVICE=cpu .venv/bin/python -c "import app.api.v1.openai_compatible; print('ok')"`
Expected: `ok`

Run: `grep -rn "命名实体\|resolving named entities" app/ tests/`
Expected: no matches. (If `app/` matches somewhere new, a site was missed — stop and check it against the spec's Changes table before proceeding.)

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `DEVICE=cpu .venv/bin/python -m unittest discover -s tests`
Expected: same pass set as after Task 1.

- [ ] **Step 6: Commit**

```bash
git add app/api/v1/openai_compatible.py
git commit -m "docs: stop declaring the OpenAI prompt parameter inert; it is wired through"
```

---

### Task 3: Build the measurement instrument

The spec refuses to claim the new wording steers as well as the old — particularly for bare hotword lists, which the old wording described precisely and the new one only generally. This task builds the instrument; Task 4 runs it on real hardware. The instrument follows the house pattern of `scripts/h100/test_offline_mixing.py`: stdlib-only, hits the running service over HTTP, filenames encode expected keywords.

**Files:**
- Create: `scripts/h100/bench_context_prompt.py`

**Interfaces:**
- Consumes: the running service's `POST /v1/audio/transcriptions` with the `prompt` form field (which becomes `context` via `hotwords=prompt or ""`).
- Produces: a JSON results file per run (`--out`), and a `--compare BASELINE.json CANDIDATE.json` mode that prints the verdict Task 4 records. Task 4 depends on the CLI exactly as defined here.

Audio-file convention (same as `test_offline_mixing.py`): each file is named `<label>_<keyword>.<ext>`, where `<keyword>` is a named entity actually spoken in the clip (e.g. `meeting_Danantara.wav`, `earnings_阿里巴巴.wav`). Keywords must be mutually exclusive across clips — that exclusivity is what makes the leak check below meaningful. The script sends each clip four times per iteration: `none` (no context), `hotwords` (all keywords space-joined, the `vocabulary_id` shape), `sentence` (a per-clip background sentence naming the keyword), and `topic` (one keyword-free background context shared by all clips — the spec's motivating use case).

**The metric must not reward prompt leak.** The spec's central fear is that weaker framing makes the model echo the context into the transcript instead of treating it as a hint. Under `hotwords`, every clip's request carries every keyword — so a leaking build produces transcripts containing all keywords, and a naive `keyword in text` metric scores that ~100%: the gate would pass hardest exactly when the change is worst. The script therefore reuses the contamination logic of `test_offline_mixing.py:148-149`: a **hit** counts only when the clip's own keyword appears AND no foreign keyword (any other clip's keyword) appears; a **leak** (foreign keyword present) is tracked as a separate per-condition `leak_rate`, and a leak increase on `hotwords` is a blocking failure in `compare` — not an eyeball step.

- [ ] **Step 1: Write the script**

Create `scripts/h100/bench_context_prompt.py`:

```python
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
```

- [ ] **Step 2: Smoke-check the CLI (no service needed)**

Run: `.venv/bin/python scripts/h100/bench_context_prompt.py --help`
Expected: usage text with `run` and `compare` subcommands, exit 0.

Run: `.venv/bin/python scripts/h100/bench_context_prompt.py run --audio-dir /tmp --out /tmp/x.json`
Expected: exits 2 with `need >= 3 keyword-named clips, found 0` (no audio in `/tmp` — proves case loading and argument plumbing work without touching the network).

- [ ] **Step 3: Verify `compare` and `classify` end-to-end with synthetic fixtures — including the leak rule**

```bash
.venv/bin/python - <<'EOF'
import json, subprocess, sys

sys.path.insert(0, "scripts/h100")
from bench_context_prompt import classify

# The pure metric must not reward prompt leak, and must normalize case.
assert classify("kata Danantara hari ini", "Danantara", ["Prabowo"]) == (True, [])
assert classify("kata DANANTARA hari ini", "Danantara", ["Prabowo"]) == (True, [])   # casefold
assert classify("Danantara Prabowo 阿里巴巴", "Danantara", ["Prabowo", "阿里巴巴"]) == (False, ["Prabowo", "阿里巴巴"])  # echo != hit
assert classify("tidak ada apa-apa", "Danantara", ["Prabowo"]) == (False, [])

def doc(hot_hit, hot_leak, clip="a.wav", ts="t"):
    hr = {c: {clip: 0.5} for c in ("none", "sentence", "topic")}
    lr = {c: {clip: 0.0} for c in ("none", "sentence", "topic")}
    hr["hotwords"], lr["hotwords"] = {clip: hot_hit}, {clip: hot_leak}
    return {"timestamp": ts, "hit_rate": hr, "leak_rate": lr}

fixtures = {
    "old":  doc(0.80, 0.0),
    "good": doc(0.75, 0.0),              # within the 10pp band -> PASS
    "drop": doc(0.60, 0.0),              # >10pp hit drop -> FAIL
    "leak": doc(0.75, 0.40),             # hit within band BUT leaking -> FAIL (inverted-gate guard)
    "mm":   doc(0.80, 0.0, clip="b.wav"),  # different clip set -> harness error
}
for name, data in fixtures.items():
    open(f"/tmp/wording_{name}.json", "w").write(json.dumps(data))

def cmp(a, b):
    return subprocess.run([sys.executable, "scripts/h100/bench_context_prompt.py",
                           "compare", f"/tmp/wording_{a}.json", f"/tmp/wording_{b}.json"]).returncode

assert cmp("old", "good") == 0, "in-band candidate must PASS"
assert cmp("old", "drop") == 1, "hit drop beyond band must FAIL"
assert cmp("old", "leak") == 1, "leak increase must FAIL even with acceptable hit rate"
assert cmp("old", "mm") == 2, "mismatched clip sets must be a harness error, not a fake regression"
print("compare verdict logic OK, including the leak guard")
EOF
```

Expected: PASS for `good`, FAIL exit 1 for both `drop` and `leak`, exit 2 for the clip-set mismatch, then `compare verdict logic OK, including the leak guard`. A comparator that cannot fail is not evidence — this proves it can fail on both blocking rules, and specifically that a leaking build cannot buy a PASS with an inflated hit rate.

- [ ] **Step 4: Commit**

```bash
git add scripts/h100/bench_context_prompt.py
git commit -m "test: add before/after entity-accuracy bench for the context wording change"
```

---

### Task 4: Run the measurement on real hardware — MERGE GATE

This task cannot run on the dev box (non-CUDA; vLLM is not installable there — see `scripts/h100/README.md`). It runs on the H100 deployment, exactly like the other `scripts/h100/` instruments. **The branch does not merge until this task's verdict is recorded.** Do not skip it, do not mark it done from reasoning about the prompt — the whole point is that reasoning about prompts has been wrong here before.

**Files:**
- No code changes. Output: two JSON result files and a recorded verdict (in the PR description or `bd` issue, per session-close convention).

**Interfaces:**
- Consumes: `scripts/h100/bench_context_prompt.py` from Task 3; the old build (main) and new build (this branch) deployed in turn.

- [ ] **Step 1: Prepare the audio set**

Assemble **at least 5 Indonesian and 5 Chinese clips** into one directory, named `<label>_<keyword>.<ext>`:
- Each keyword is a named entity **actually spoken** in its clip (proper noun / org name / domain jargon the model plausibly gets wrong without a hint — e.g. `rapat_Danantara.wav`, `berita_Prabowo.wav`, `caijing_阿里巴巴.wav`, `xinwen_腾讯.wav`).
- Keywords must be mutually exclusive across clips (same rule as `test_offline_mixing.py`): no clip's keyword may legitimately occur in another clip's audio, and no keyword may be a substring of another. `run` enforces this at startup; a foreign-keyword "leak" from dirty fixtures is a false positive.
- Keywords must be alphabetic words, not numbers/dates — ITN rewrites digits and defeats the (casefolded) substring match.
- 10–60 s per clip. Real speech, not TTS, if at all possible.

Why at least 10 clips: the model decodes near-deterministically, so repeated iterations of the same clip are highly correlated — the effective sample size is closer to the number of **distinct clips** than to clips × iterations. More clips buys real resolution; more iterations mostly buys robustness against nondeterminism. See the decision rule in Step 4 for what this sample size can and cannot detect.

Sanity-check headroom first: run once with the OLD build and inspect the `none` condition — if every keyword already hits at 100% with no context, the clips are too easy and the comparison measures nothing. Swap in harder entities until `none` < `hotwords` on at least some clips.

- [ ] **Step 2: Baseline run against the OLD wording**

Deploy the build from `main` (pre-Task-1 commit). Confirm the deployed wording before measuring — **inside the running container, not the host checkout**. Per `scripts/h100/README.md` the service runs from a deployed Docker image; host/container drift is exactly the failure this check exists to catch, and importing from the host checkout would happily "confirm" a wording the service is not running (the host's bare `python` may not even have `qwen3_vllm`'s import-time deps, librosa/numpy):

```bash
docker compose exec qwen3-asr python -c \
  "from app.services.asr.qwen3_vllm import _build_chat_prompt; \
   assert 'resolving named entities' in _build_chat_prompt(context='x'), 'this is NOT the old build'; \
   print('old wording confirmed')"
```

(If the deployment is not compose-managed, use `docker exec <container> python -c ...` against whatever container actually serves port 8000 — the point is the artifact behind `ASR_BASE_URL`, never the host filesystem.)

Then, from the host:

```bash
python scripts/h100/bench_context_prompt.py run \
    --audio-dir /path/to/clips --iterations 10 --out /tmp/wording_old.json
```

- [ ] **Step 3: Candidate run against the NEW wording**

Deploy this branch. Confirm, again inside the running container:

```bash
docker compose exec qwen3-asr python -c \
  "from app.services.asr.qwen3_vllm import _build_chat_prompt; \
   assert 'when transcribing' in _build_chat_prompt(context='x'), 'this is NOT the new build'; \
   print('new wording confirmed')"
```

Then:

```bash
python scripts/h100/bench_context_prompt.py run \
    --audio-dir /path/to/clips --iterations 10 --out /tmp/wording_new.json
```

Same audio dir, same iteration count, same machine, back-to-back — anything that drifts between runs is noise in the comparison. (`compare` exits 2 if the clip sets differ, rather than scoring missing clips as regressions.)

- [ ] **Step 4: Compare and record the verdict**

```bash
python scripts/h100/bench_context_prompt.py compare /tmp/wording_old.json /tmp/wording_new.json
```

**Decision rule (blocking), evaluated on the `hotwords` condition** — the `vocabulary_id` shape, the surface documented as hotwords-only whose clients never opted into a semantics change. A hit counts only when the clip's own keyword is present AND no foreign keyword is; foreign-keyword presence is the leak (echo) signature. Both parts are enforced by `compare` (exit 1), not by eyeball:

1. **Leak rule — zero tolerance:** candidate `leak_rate` must not exceed baseline `leak_rate`. With clean fixtures, a foreign keyword can only reach a transcript by the model echoing the prompt as content — the spec's named failure mode. Same epistemics as `FOREIGN` in `test_offline_mixing.py`: one leak is conclusive in the direction of failure, so any increase blocks.

   **"Clean" is only half machine-enforced.** Startup checks keyword-vs-keyword uniqueness and non-substring collisions. It cannot check the other half: that no clip's *audio* naturally speaks another clip's keyword, and that no keyword is a casefolded substring of ordinary words in the transcript language. Those stay manual (Step 1's fixture rules). So a dirty fixture set can false-block on rule 1. If a leak fires, inspect the printed `text=` before believing it — confirm the leak is echo, not a fixture defect, then rerun.
2. **Hit rule — 10-percentage-point band:** candidate overall hit rate must be `>= baseline − 0.10`.

**Why a band, and what it can and cannot see — do not overclaim this.** Each side yields ~100 Bernoulli trials nominally (10 clips × 10 iterations), but near-deterministic decoding correlates iterations of the same clip, so the honest effective n is closer to the ~10 distinct clips. Numbers below use the nominal n=100 per side; the truth is between the two and closer to the pessimistic end:
- A strict `new >= old` rule at this n fails on noise roughly half the time when the two wordings are truly identical — a coin-flip gate, worse than no gate because it manufactures false regressions.
- With the 10pp band, the false-block rate on truly identical wordings is roughly 5–10% at nominal n (SE of a hit-rate difference near p≈0.7 is ~6.5pp at n=100/side; substantially wider at effective n≈10 clips).
- **Minimum detectable effect: a true drop needs to be on the order of 20pp before this gate blocks it reliably (~80%+); at the pessimistic effective n it is more like 30pp.** A 5–15pp real regression will usually pass. That is a real, accepted limitation: detecting a 5pp drop with confidence would need on the order of several hundred *distinct* clips per condition, which is not practical here. The gate is designed to catch the catastrophic modes — prompt echo (rule 1, where a single event is signal and sample size is irrelevant) and gross steering loss (rule 2) — not to certify fine-grained equivalence. Record the verdict in those words; never write "proven equivalent".

**One echo mode the gate cannot see.** Rule 1 catches echo of the *context* — under `hotwords` the context carries every keyword, so any echo of it drags foreign keywords in and trips the leak rule. It does **not** catch echo of the *wrapper instruction itself* (e.g. the literal words "Use this context when transcribing" surfacing in a transcript): no keyword moves, so hit and leak are both unchanged and `compare` returns PASS. There is no automated discriminator for this at the instrument's fidelity. The mitigation is manual and mandatory: the per-response `text=` printouts exist for this: **read a sample of them before recording a PASS.** Do not treat exit 0 alone as evidence the transcripts are clean.

- Exit 0 (PASS): record both JSON files' summaries and the compare output in the PR / `bd` issue, explicitly stating the MDE caveat above. Merge may proceed.
- Exit 1 (FAIL): **do not merge.** Follow the spec's rollback: revert item 1 to the original string (item 2's needle reverts with it — one commit, since they are atomic). Task 2's doc fixes may still land: they describe behavior that is already true today under either wording. File a `bd` issue with the numbers so the next wording attempt starts from data.
- Exit 2 (harness error, e.g. mismatched clip sets): fix the setup and rerun both sides; there is no verdict.
- Marginal (hit delta inside the band but consistently negative across most clips): add more **distinct clips** — not iterations — and rerun both sides before deciding.

Also record the `topic` condition (keyword-free background — the spec's motivating "general background context" use case) and the `sentence` condition in the write-up: hit and leak rates, old vs new. Neither is a merge criterion — the spec makes no improvement claim and this plan does not smuggle one in — but the topic numbers are the first real data on the use case that motivated the change, and a leak appearing under `topic` or `sentence` on the new wording is worth a `bd` issue even though only `hotwords` blocks.

- [ ] **Step 5: Merge (only after PASS)**

With the verdict recorded, merge the branch per the normal flow (`superpowers:finishing-a-development-branch`), then complete the CLAUDE.md session-close protocol: update `bd` issues, `git pull --rebase && git push`, verify `git status` shows up to date.

---

## Self-review notes

- **Spec coverage:** all five Changes-table items land in Tasks 1–2; the spec's five listed unit tests for `_build_chat_prompt` are Task 1 Step 1 (test 1 → `test_context_and_language...`, 2 → `test_context_without_language...`, 3 → `test_empty_and_whitespace...`, 4 → `test_language_alias_id...`, 5 → `test_old_named_entity_wording_is_gone`); the spec's blocking-validation risk is Tasks 3–4. Rollback path is written into Task 4 Step 4.
- **Atomicity:** items 1+2 are a single commit (Task 1 Step 7). Docs are a separate commit (Task 2). Measurement gates the merge, not the commits — committing to the feature branch before measuring is fine; merging is not.
- **Realtime surface:** the wording change reaches `qwen3_vllm.py:459` (streaming) automatically; no separate task because there is no separate code. Task 4 measures the offline HTTP path only — the WebSocket `context` param has no documented contract to regress against (spec Risks), and the wrapper string is identical on both paths.
- **Gate design:** the metric cannot be gamed by prompt leak (a leaked response is never a hit, and leak_rate blocks independently — Task 3); the deploy check inspects the running container, not the host checkout (Task 4 Steps 2–3); the decision rule states its minimum detectable effect and does not claim statistical power it lacks (Task 4 Step 4); the motivating topic-only use case is measured (`topic` condition), non-blocking per the spec.
