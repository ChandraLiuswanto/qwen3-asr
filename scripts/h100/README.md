# H100 runbook

**Status: NOT YET RUN. Nothing here is verified.** This directory holds
instruments, not results. Whoever runs them on an H100 records the verdicts.

| script | question it answers | section |
|--------|--------------------|---------|
| `verify_vllm_root_cause.py` | Does vLLM 0.19.0's drain actually mix concurrent callers? | [below](#part-1--verify-the-vllm-root-cause-hypothesis) |
| `test_offline_mixing.py` | Do concurrent offline requests return each other's transcripts? | [Part 2](#part-2--offline-mixing-regression-test_offline_mixingpy) |
| `test_ws_offline_mixed.py` | Does the ws-vs-offline ITN race corrupt output? What is ws decode p50/p95? | [Part 3](#part-3--mixed-websocket--offline-test_ws_offline_mixedpy) |

Everything under Part 2 and Part 3 exists because **Change A is verified only by
unit tests against FAKES on a non-CUDA box.** No test has yet demonstrated the
real concurrency bug is fixed. These scripts are the only instruments that can.
Until they are run and their output recorded, the fix is unverified.

Full execution order is in [Part 4](#part-4--execution-runbook-after-change-a-is-deployed).

---

# Part 1 — verify the vLLM root-cause hypothesis

## Why this exists

The concurrency design for this repo rests on a hypothesis about vLLM 0.19.0
that **nobody has ever checked against the source**, because vLLM cannot be
installed on the dev box (AMD APU, non-CUDA). The hypothesis:

> `Qwen3VLLMBackend` holds one `vllm.LLM`
> (`app/services/asr/qwen3_vllm.py`). `_run_generate` reassembles results
> positionally via `zip(outputs, audio_items)`. `LLM.generate` is believed to
> add every prompt to a shared `LLMEngine` and drain it, collecting whatever
> finishes, **without filtering to the request IDs this caller submitted**. Two
> concurrent callers would each collect a mix of both requests' outputs, and
> `zip` would silently truncate the mismatch — pairing request B's transcripts
> with request A's segments. Wrong text, no exception.

The probe reads the **actual installed source** and classifies it. It is built
to be able to say REFUTED.

## The caveat that makes this non-trivial

Upstream historically **sorts** the drained outputs by request id before
returning. Sorting does *not* rescue concurrent callers — the engine's
request-id counter is shared, so ownership still interleaves — but it does mean
the real drain loop is **not literally shaped like the hypothesis's
pseudocode**. Do not let that mismatch read as either confirmation or
refutation. The probe classifies on the presence or absence of an **ownership
filter** only, and reports any sort as an informational note.

Read the printed source. The probe's verdict is a reading aid, not an oracle;
if the source contradicts it, the source wins and you record that.

## Run it

Inside the deployed image (`vllm[audio]==0.19.0`), on the H100:

```
python scripts/h100/verify_vllm_root_cause.py | tee /tmp/vllm_root_cause.txt
```

The script is self-contained (stdlib + vllm; no repo imports). Exit codes:

| code | meaning |
|------|---------|
| 0 | CONFIRMED |
| 1 | REFUTED |
| 2 | INCONCLUSIVE |
| 3 | probe error (vllm not importable — you are on the wrong machine) |

## Decision rule

Let D be the drain loop `generate()` actually relies on (`LLM._run_engine`, or
whatever it calls at this version — the probe resolves the name from
`generate`'s source rather than assuming it).

- **CONFIRMED** — D appends engine step outputs based only on *"is this output
  finished?"*, with no comparison against a set/list/dict of request IDs
  captured by this `generate()` call.
- **REFUTED** — D restricts its return to this caller's own request IDs: an
  explicit request-id membership test, a per-caller output map keyed by the ids
  this call added, or a per-caller queue.
- **INCONCLUSIVE** — source unreadable, or shape matches neither pattern.

Sorting, batching, tqdm bookkeeping, and `n`-sampling fan-out are irrelevant to
this rule.

**INCONCLUSIVE is a real outcome.** Do not round it up to CONFIRMED. Read the
source, record a hand verdict with a quoted excerpt, or fix the probe and rerun.

## What each verdict means for the work

### CONFIRMED
The stated root cause holds. Change A's lock is justified on mixing grounds;
change C proceeds as scoped. Record the verdict, the vllm version, and the
quoted drain source in the spec changelog.

### REFUTED — this is a significant result, not a formality
The design's root cause is **wrong**. Before any further work:

1. **Changes already landed on the strength of this hypothesis may be
   unnecessary.** Re-review them on their own merits; do not build on them.
2. **Change A's lock is not justified by output mixing.** It may still stand on
   the separate, documented grounds that `LLM` is not thread-safe, plus the
   tokenizer / ITN / VAD races — but that case must be argued independently,
   not inherited from a refuted hypothesis.
3. **Change C (async engine) must be re-scoped from scratch.** Its premise is
   void.

### INCONCLUSIVE
Do not proceed on a guess. Resolve it to CONFIRMED or REFUTED by hand, with
evidence, before touching change C.

## The other two questions (both in scope, both gate change C)

- **Q(b)** — does `vllm.v1.engine.async_llm.AsyncLLM` import at 0.19.0, and does
  the `AsyncLLMEngine` alias still exist? `AsyncLLMEngine` is a compatibility
  alias for the v1 engine; the v0 implementation is gone and upstream carries a
  TODO to remove the proxy. If the alias is gone, code depending on it breaks.
- **Q(c)** — does `AsyncLLM` support **pooling** / `encode` with multimodal
  audio? The forced aligner uses `runner="pooling"` and
  `aligner.encode(..., pooling_task="token_classify")`. **If async pooling does
  not exist, the aligner cannot migrate to the async engine and stays
  serialized behind `_aligner_lock`** — independently of Q(a)'s verdict. Scope
  change C accordingly.

Record all three answers plus the vllm version in the spec changelog.

## Local testing

The probe's classifier (`classify_drain`) is pure — source text in, verdict out
— and is unit-tested in `tests/test_verify_vllm_root_cause.py`, including that
a sort does not flip the verdict and that missing/unrecognized source yields
INCONCLUSIVE rather than CONFIRMED.

Everything else in the probe requires a real vllm on a CUDA box and is **not**
tested locally. Mocking `inspect.getsource` would prove nothing about vLLM
0.19.0. Those tests would be theatre; they are deliberately absent.

Run: `DEVICE=cpu .venv/bin/python -m unittest discover -s tests`

---

# Test audio fixtures (required by Parts 2 and 3)

Both integration scripts read `ASR_TEST_AUDIO_DIR`: a directory of
**distinct-content** audio where each filename encodes the keyword its
transcript must contain, as `<label>_<keyword>.<ext>`:

```
alpha_银行.wav      # must transcribe to text containing 银行
beta_机场.wav       # must transcribe to text containing 机场
gamma_医院.wav      # ...
```

Rules the scripts enforce at startup (they refuse to run otherwise):

- **At least 2 files.** Mixing cannot be detected with one input.
- **Keywords mutually exclusive.** No keyword may be a substring of another
  (`银行` inside `银行卡` would flag contamination on every clean run).
- **At least as many files as the concurrency**, or slots reuse files — two
  slots sharing a keyword cannot visibly contaminate each other, which silently
  weakens detection. The script warns; take the warning seriously.

Additionally, **Part 3's websocket half needs 16 kHz mono 16-bit PCM `.wav`** —
the ws protocol takes raw PCM, not containers. Convert:

```
ffmpeg -i input.m4a -ar 16000 -ac 1 -c:a pcm_s16le alpha_银行.wav
```

**Validate the fixtures before trusting any run:**

```
ASR_TEST_AUDIO_DIR=/path/to/audio python scripts/h100/test_offline_mixing.py \
    --iterations 20 --concurrency 1
```

Concurrency 1 has no race. If it reports `MISSING_OWN`, the model does not
produce your keywords and **every result from both scripts is uninformative** —
fix the audio or the keywords first.

---

# Part 2 — offline mixing regression (`test_offline_mixing.py`)

## The bug

`Qwen3VLLMBackend` holds one shared `vllm.LLM`. `_run_generate` reassembles
results positionally with `zip(outputs, audio_items)`. If two threads call
`generate()` concurrently, each drains the shared engine and collects a mix of
both callers' outputs; `zip` truncates to the shorter side and pairs request
B's transcripts onto request A's segments. **Wrong text, no exception** — no
traceback, no 500, no log line. Only a content check catches it. The guard under
test is `self._llm_lock` in `_run_generate`.

## Run it

```
export ASR_BASE_URL=http://localhost:8000
export ASR_TEST_AUDIO_DIR=/path/to/audio
export ASR_API_KEY=...        # optional
python scripts/h100/test_offline_mixing.py --iterations 20 --concurrency 8 \
    | tee /tmp/offline_mixing.txt
```

Stdlib only. Exit codes: `0` = no contamination observed, `1` = contamination
(or keywords missing), `2` = setup error.

## MANDATORY: the must-fail runbook

**A green run from this script is worthless on its own.** It may be green
because the guard works — or because the harness is broken, the audio is too
short to overlap, or the keywords never appear. A green run cannot distinguish
those. So force a red one first. `--check-detects-bug` prints this:

1. **Reintroduce the bug.** In `app/services/asr/qwen3_vllm.py` `_run_generate`,
   comment out `with self._llm_lock:` and dedent its body. Rebuild, redeploy,
   run the script. **EXPECT FAIL** (exit 1, `FOREIGN`/`BOTH` verdicts naming
   another request's keyword). **Save this output — it is the evidence.**
2. **Restore.** `git checkout app/services/asr/qwen3_vllm.py`, rebuild,
   redeploy, run the identical command. **EXPECT 20/20 PASS.**
3. **Record BOTH outputs.** Step 2 alone is not a result; the pair is.

**Restore the guard.** Never leave a lock-removed build near a deploy path.

**If step 1 PASSES**, the test does not detect the bug it was written for. Do
not proceed to step 2 — troubleshoot: are the requests actually overlapping
(check server log timestamps and the printed per-request elapsed times)? Is an
admission semaphore set to 1, or a proxy, serializing them? Is the audio long
enough to overlap? Raise `--concurrency`/`--iterations`.

## This is EVIDENCE, NOT PROOF

The race is probabilistic — interleaving depends on scheduler timing, batch
composition and audio length. An unguarded build can pass by luck. **20/20 PASS
means "no contamination observed in 20 rounds under this load", not "the race
cannot happen."** Write it up in exactly those words.

Note `FOREIGN`/`BOTH` (a foreign keyword appeared) is the mixing signature and
one hit is conclusive in the direction of failure. `MISSING_OWN` alone is *not*
claimed as mixing — it can just be a poor transcript — but a truncating `zip`
also produces it, so it fails the run and must be investigated.

---

# Part 3 — mixed websocket + offline (`test_ws_offline_mixed.py`)

## Why the SHAPE of this test is the whole point

Do not "simplify" this into a websocket-only test. It would never fail.

- A websocket session's decode runs on an executor thread (`await
  _run_decode(...)`), but its ITN call (`_normalize_output_text`) runs **after**
  that await returns — i.e. back **on the event-loop thread**.
- One event loop runs one callback at a time, so **websocket↔websocket ITN calls
  are serialized by the loop and cannot race each other**, no matter how many
  sessions you open.
- **The only live ITN race is a websocket's event-loop-thread ITN call against
  an OFFLINE request's executor-thread ITN call.**

Hence the mixed shape. Drop the offline half and you have deleted the test's
reason to exist. The script refuses to run with `--ws 0` or `--offline 0`.

Offline ITN is hardcoded `enable_itn=True` in
`offline_transcription_service.transcribe` (no form field), so the offline half
always exercises ITN. The ws half sets
`enable_inverse_text_normalization: true` explicitly.

## Run it

```
python scripts/h100/test_ws_offline_mixed.py --iterations 20 --ws 4 --offline 4 \
    | tee /tmp/ws_offline_mixed.txt
```

Needs stdlib + `websockets`. Exit codes as Part 2.

## What it produces

1. **Per-channel keyword integrity** across both channels, same rule as Part 2.
2. **Websocket decode latency p50/p95** — the baseline the spec's threshold must
   be set *from*. **No threshold is enforced**, because no data existed to set
   one. This run produces that data; agree the threshold from it.

## How to read the latency numbers — do not quote them without this

- Measured **under the mixed load** (`--ws` sessions + `--offline` requests
  concurrently). Numbers from a different mix are not comparable.
- The client is **lockstep** (send one chunk, wait for its `result`). A
  pipelining production client sees different end-to-end lag. This is a
  comparable baseline, not a user-facing SLA.
- **Silent sends are excluded.** The server suppresses a `result` when the text
  did not change (`if full != ctx.last_partial_text`). Those sends produce no
  message and are counted separately, so the latency sample is **biased toward
  chunks that produced new text**.
- Percentiles are **nearest-rank**, so every number reported is an actually
  observed sample, not an interpolation between two.
- **To compare against `main`**, run the identical command against a `main`
  deploy with the same audio and the same `--ws`/`--offline`. Absolute numbers
  across different hardware or audio are meaningless.

---

# Part 4 — execution runbook (after Change A is deployed)

Run in this order and **record every number**. Nothing below is verified until
its output is pasted into the spec changelog.

1. **Part 1 probe** — `verify_vllm_root_cause.py`. If REFUTED, stop: the root
   cause is wrong and Parts 2–3 are testing a hypothesis that does not hold.
2. **Fixture validation** — `test_offline_mixing.py --concurrency 1`. Must be
   clean before anything else counts.
3. **Part 2 must-fail runbook** — guard removed → **expect FAIL**; guard restored
   → **expect 20/20 PASS**. Record both. *Without the FAIL, the PASS is not
   evidence.* Restore the guard.
4. **Part 3** — `--iterations 20 --ws 4 --offline 4`. Record integrity result and
   the p50/p95 baseline. Agree the latency threshold from these numbers.
5. **Semaphore / overlap check** — 10 concurrent long (~5 min) offline requests.
   Compare wall-clock against a `main` deploy and grep `ASR_STAGE_TIMINGS` in the
   logs to confirm stages actually overlap. The spec expects improvement; there
   is **no target multiple** — record what you measure.

## KNOWN GAP — `ASR_STAGE_TIMINGS` is success-path only

`format_stage_timings` is emitted only after a successful transcription
(`app/services/asr/engines/base.py`, inside the success branch). **A failed
request logs no stage line at all.** Any measurement built from these lines
therefore describes *successful requests only* and is biased accordingly — under
concurrency, if the slowest or most contended requests are the ones that fail or
time out, they vanish from the data and the timings look better than reality.

Before drawing any conclusion from stage timings: **reconcile the stage-line
count against the number of requests you actually sent.** If they differ,
requests failed silently and the timing data is incomplete. Say so in the
write-up rather than reporting the mean of the survivors.

## Local testing of Parts 2 and 3

Only the pure logic is unit-tested locally:

- `tests/test_offline_mixing_check.py` — filename→keyword parsing, the
  contamination verdict rules, and the fixture-overlap guard.
- `tests/test_ws_offline_mixed_stats.py` — percentile/statistics arithmetic and
  ws-URL derivation.

**What those tests do NOT cover, and cannot:** whether the service actually
mixes concurrent requests; whether `_llm_lock` or the ITN lock fix anything;
whether the latencies are measured correctly; whether the HTTP/ws clients or the
concurrency work at all. A green local run means "the detector's arithmetic is
what we intend" — **it is not evidence about the service.**

Faking the concurrency locally to produce a green test is the exact failure mode
this design exists to prevent. There are deliberately no such tests.

Run: `DEVICE=cpu .venv/bin/python -m unittest discover -s tests`
