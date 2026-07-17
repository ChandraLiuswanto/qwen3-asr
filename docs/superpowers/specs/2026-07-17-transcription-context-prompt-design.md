# Broaden the Transcription Context Prompt

**Date:** 2026-07-17
**Status:** Draft design, pending review
**Scope:** One prompt string in `_build_chat_prompt`, plus four stale doc/annotation
sites on the OpenAI endpoint. No API surface changes.

## Problem

`_build_chat_prompt` (`app/services/asr/qwen3_vllm.py:74`) frames all caller-supplied
context as a named-entity list:

```python
if context.strip():
    instructions.append(f"Use this context when resolving named entities: {context.strip()}")
```

The parameter is named `context` at every layer, and callers want to supply general
background about the audio — topic, setting, speaker mix, jargon domain. The trailing
three words `when resolving named entities` narrow that to a noun list. A caller sending
`Rapat mingguan tim Danantara, membahas dana kelolaan` has that sentence labelled as a
set of named entities, which is not what it is.

The narrowing is the whole problem. The wrapper sentence itself is load-bearing (see
"Why not raw injection").

## Goals

1. Caller-supplied context is framed as background, whatever its shape.
2. Existing hotword-list callers keep working unchanged.
3. No new parameters, no API changes.

## Non-Goals

- **Output directives** ("don't translate", "no punctuation"). The broadened wrapper
  tolerates them awkwardly but does not target them. If they become a real requirement,
  that is a separate field and a separate spec.
- **Whisper `prompt` parity.** Rejected below.
- Touching `temperature` / `timestamp_granularities`, which genuinely are inert.
- **The CPU/Rust backend.** This change lives in `qwen3_vllm.py` and is a no-op there.
  Rust discards caller context outright (`qwen3_engine.py:234`,
  `_ = (hotwords, enable_punctuation, sample_rate)`) and is not wired for `language`
  either, though `qwenasr_rust.py:318` accepts it. So all three surfaces below gain the
  new framing **on vLLM/CUDA only**. Deliberately out of scope.

## The convergence

One prompt string serves three public surfaces. All three feed the same `context`
argument, under three different names. Paths below are relative to the repo root;
note there are two `asr.py` files (`app/api/v1/` and `app/models/`).

| Surface | Caller-facing name | Path to `context` | Documented shape |
|---|---|---|---|
| OpenAI `/v1/audio/transcriptions` | `prompt` | `app/api/v1/openai_compatible.py:496` → `hotwords=prompt or ""` (`:561`) → `app/services/asr/qwen3_engine.py:371` `context=hotwords or ""` | "提示文本…（hotwords）" |
| Alibaba `/stream/v1/asr` | `vocabulary_id` | `app/api/v1/asr.py:237` `hotwords=params.vocabulary_id or ""` → same | Bare word list — `阿里巴巴 腾讯` (`app/api/v1/asr.py:172`) |
| WebSocket `/ws/v1/asr/qwen` | `context` | `app/services/qwen3_websocket_asr.py:216-217` and `:279,291` → `qwen3_engine.py:541` | Undocumented |

There is no fourth surface. `app/services/asr/runtime/router.py:221`
(`hotwords=request.hotwords`) is an internal hop on the offline chain, not an entry
point.

The WebSocket surface has **two** `init_streaming_state` sites, not one: `:291` on
initial start, and `:216-217` on segment re-init after truncation. Both pass `context`;
both change behavior together.

Both `_build_chat_prompt` call sites are affected: `app/services/asr/qwen3_vllm.py:283`
(offline batch) and `:459` (realtime/alignment). `qwen3_engine.py:371,426,490` all pass
`context=hotwords or ""`.

**`vocabulary_id` is the constraint that decides the design.** It is documented in
`README.md:287`, `docs/README_zh.md:285`, `app/api/v1/asr.py:112,172`, and
`app/models/asr.py:45` as a bare space-separated hotword list, capped at 512 chars. Any
change must keep `阿里巴巴 腾讯` working as a hint rather than as content.

## Decision

Change one line — `qwen3_vllm.py:81`:

```python
instructions.append(f"Use this context when transcribing: {context.strip()}")
```

Resulting system prompt with `language=id`:

```
Transcribe the speech in Indonesian. Use this context when transcribing: Rapat mingguan tim Danantara, membahas dana kelolaan
```

And with a `vocabulary_id` hotword list, still coherent:

```
Transcribe the speech accurately. Use this context when transcribing: 阿里巴巴 腾讯
```

`language` and `context` remain independent slots joined into one `instructions` list;
neither constrains the decoder. Both are soft prompt steering. That is unchanged.

### Why not raw injection

The rejected alternative was `Transcribe the speech in Indonesian. {context}` —
dropping the wrapper entirely.

The wrapper sentence is what tells the model the text is *background rather than
content*. Without it, a bare noun phrase sits in the system block immediately before
the model transcribes audio in that same language, and the expected failure is that it
echoes the context into the transcript — the prompt-leak mode Whisper is known for, to
which an instruction-following LLM reading the whole system block as text is at least
as exposed. `vocabulary_id`'s documented `阿里巴巴 腾讯` becomes a dangling fragment with
no stated purpose, which is a silent regression on a documented, shipped parameter.

Raw injection buys caller control over instructions. Goal-wise that is a non-goal here.

### Why not Whisper parity

OpenAI's `whisper-1` `prompt` is decoder conditioning — text treated as if it preceded
the audio, steering by imitation, capped near 224 tokens. Qwen3-ASR is an
instruction-following LLM with a chat template; `context` lands in an `<|im_start|>system`
block. The mechanisms differ, so string-level compatibility does not produce behavioral
compatibility. Parity would help callers who use Whisper's `prompt` incorrectly (as
instructions) and hurt those who use it correctly (as a style exemplar). Rejected.

## Changes

| # | Site | Change |
|---|---|---|
| 1 | `app/services/asr/qwen3_vllm.py:81` | `when resolving named entities:` → `when transcribing:` |
| 2 | `tests/test_vllm_mixing_fidelity.py:210` | `needle = "resolving named entities: "` → `"when transcribing: "` |
| 3 | `app/api/v1/openai_compatible.py:413` | Drop `prompt` from the 暂不支持的参数 list — it works. Line becomes `` `temperature`、`timestamp_granularities` 参数已保留但暂不生效 ``. |
| 4 | `app/api/v1/openai_compatible.py:506` | `_ = (prompt, temperature, timestamp_granularities)` → `_ = (temperature, timestamp_granularities)`; `prompt` is read at `:561`. |
| 5 | `app/api/v1/openai_compatible.py:496` | Form description `"提示文本，作为命名实体上下文注入转写提示（hotwords）"` → `"上下文提示文本，作为背景信息注入转写提示（如主题、领域、专有名词）"`. |

**Items 1 and 2 are atomic — they must land in the same commit.**
`test_vllm_mixing_fidelity.py` calls the real `_run_generate` (`:285`), so item 1 without
item 2 makes `_marker()` raise `ValueError` on `.index()`. Items 3–5 are independent and
may land separately.

Items 3–5 are a pre-existing documentation bug: the endpoint's own OpenAPI description
tells users `prompt` is inert while line 561 passes it through. They are independent of
item 1 but touch the same parameter.

**Deliberately not renamed:** `hotwords` as an internal argument name
(`app/services/asr/offline_transcription_service.py:29`, `engines/base.py:92,105,129,402`,
`engines/funasr.py:57,71`) and the public `vocabulary_id`. Both are wider blast radius
than this change earns, and `funasr.py` discards the value anyway (`:65,77`).

## Testing

Convention: `unittest`, not pytest. `DEVICE=cpu .venv/bin/python -m unittest discover -s tests`.

`test_vllm_mixing_fidelity.py:204-212` uses the wrapper text only as a *parsing anchor* —
`_marker()` slices the caller's context back out of the built prompt to prove which
request produced which text. It does not assert the wording semantically, so item 2 is a
mechanical needle swap and the test's guarantee is unchanged.

New `tests/test_chat_prompt.py`, direct unit tests on `_build_chat_prompt`. Importable
without a vLLM install — `qwen3_vllm.py` has no module-level `vllm` import, only
`importlib.util.find_spec` at `:52`. Verified by executing
`from app.services.asr.qwen3_vllm import _build_chat_prompt` in the repo venv. The
module does pull in `librosa`/`numpy`/`app.infrastructure` at import time, so the
function is pure but its module is not free to import.

1. Context + language → both instructions present, language first.
2. Context, no language → `Transcribe the speech accurately.` + context clause.
3. Empty / whitespace-only context → no context clause emitted (guards the
   `context.strip()` branch).
4. Language alias: `id` → `Indonesian` via `_normalize_language_name`.
5. Prompt no longer contains `resolving named entities` (regression guard).

## Risks

- **Steering quality is unverified.** Whether `when transcribing:` frames context as
  well as `when resolving named entities:` did — particularly for the hotword-list
  callers that the old wording described precisely and the new one describes only
  generally — cannot be established by reading code. This repo has a recorded lesson
  (`perf-estimates-from-code-are-unreliable`) that confident code-reading estimates were
  wrong three times running. **No claim of improvement is made here.** Validation is a
  blocking task in the implementation plan, not an assumption of this spec: run real
  Indonesian and Chinese clips with a hotword list through both wordings and compare
  entity accuracy before merging.
- **Blast radius is three surfaces, one of them documented as hotwords-only.** A
  regression on `vocabulary_id` would hit Alibaba-protocol clients that never opted into
  a semantics change. This is the main thing measurement must cover.
- **Realtime path shares the wrapper** (`qwen3_vllm.py:459`). Streaming context behavior
  changes too, and `/ws/v1/asr/qwen`'s `context` param has no documented contract to
  regress against.

## Rollback

Revert item 1 to the original string. The wrapper is one line with no state, no
migration, and no persisted artifacts; item 2 reverts with it. Items 3–5 are
documentation-only and can stand independently — they describe behavior that is already
true today.
