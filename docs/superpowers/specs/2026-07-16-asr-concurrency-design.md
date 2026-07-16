# ASR Concurrency — Design

Date: 2026-07-16
Status: awaiting approval
Revision: 7 — shared tokenizer added to A's lock scope; ITN race mechanism corrected. Reviewed GO at r6. See Changelog.

## Spec

### Problem

`POST /v1/audio/transcriptions` serializes completely under concurrent load. Single-request latency is acceptable. Production is an H100 (80GB); the target workload is 10 concurrent users with ~5-minute audio.

The direct cause is the lock added in `a0e6900` ("fix: serialize vllm offline inference"), at `app/services/asr/runtime/router.py:196`:

```python
async def run_offline(self, request: OfflineASRRequest) -> ASRFullResult:
    model_id = self.resolve_model_id(request.model_id)
    if self._resolve_family(model_id) == RuntimeFamily.QWEN_VLLM:
        lock = self._vllm_offline_locks.setdefault(model_id, asyncio.Lock())
        async with lock:
            return await self._run_offline(request, model_id)
    return await self._run_offline(request, model_id)
```

`_run_offline` (`router.py:204-222`) wraps the **entire pipeline** in one `run_sync(engine.transcribe_long_audio, ...)` job. So the lock is held across diarization, splitting, every inference batch, and alignment. `asyncio.Semaphore(_VLLM_SHARED_CONCURRENCY)` (`router.py:21,146`) is acquired *inside* it (via `acquire_engine` from `_run_offline:209`), so the offline path can never use more than one permit. Raising it has no effect. Websocket traffic does consume the permits, so the semaphore is not dead globally — only on this path.

**The lock's breadth is the central problem.** It exists to protect one thing (a non-thread-safe vLLM engine) but serializes everything. Narrowing it to what it actually needs to protect is the change that unblocks all other concurrency work.

### No throughput projections in this spec

> r3 carried a stage-share table and Amdahl-derived speedup figures. Adversarial review found them structurally wrong — not merely imprecise — and they have been **removed rather than re-estimated**. Two prior attempts at these numbers were both wrong, in different directions. This spec states bottlenecks and fixes; the numbers come from measurement on the H100 (task 2), not from reasoning about the code.
>
> What the r3 numbers got wrong, recorded so it is not repeated:
> - They budgeted stages that never co-occur (see Request shapes below).
> - They applied `1/(S + (1-S)/N)` with N=10, which assumes each parallelized stage runs 10× faster with 10 users. Every stage shares **one** H100. Concurrency exposes idle GPU headroom; it does not multiply GPU capacity. Continuous-batching throughput scales sublinearly and approaches a ceiling.
> - They projected a "diarization-only" scenario that is unimplementable, because the router lock already serializes diarization (see Ordering).

### Request shapes — the stages are conditional, not additive

Verified in `app/services/asr/engines/base.py:152-167`:

```python
if enable_speaker_diarization:
    diarizer = SpeakerDiarizer()
    speaker_segments = diarizer.split_audio_by_speakers(audio_path)
    if not speaker_segments:
        logger.warning(f"{task_prefix}说话人分离未检测到片段，fallback 到 VAD 分割")

if not speaker_segments:
    splitter = AudioSplitter(device=self.device)
    audio_segments = splitter.split_audio_file(audio_path)
```

- **FunASR VAD is a fallback**, not a stage. It runs only when diarization is disabled or returns nothing. `enable_speaker_diarization` defaults `True` (`router.py:38`), so on default traffic it does not run at all.
- **Alignment runs only when `word_timestamps=True`**, which defaults `False` (`router.py:39`; `qwen3_vllm.py:302,349` short-circuit otherwise). Zero on default traffic.
- The **default request is: diarization → ASR decode.** Those two are the only stages that matter for default load. VAD and alignment matter only for their respective non-default traffic shapes.
- Note the CAM++ pipeline contains its **own internal modelscope VAD child** (`speaker_diarizer.py:133-139`) — a different model from the FunASR VAD. Do not conflate them.

Structural sizing facts: `MAX_SEGMENT_SEC = 60.0` (`config.py:68`) splits 5-minute audio into a handful of segments; `ASR_BATCH_SIZE = 4` (`config.py:64`) means only a few sequential GPU batches per request (`engines/base.py:186`). A batch of 4 short segments is a small workload for an 80GB H100 — there is headroom to pack concurrent requests into those forward passes. How much is a measurement question.

### Root cause — LOAD-BEARING AND NOT YET VERIFIED

> **Hypothesis, not established fact.** vLLM is not installed in the dev environment (`pyproject.toml:33` pins `vllm[audio]==0.19.0`, Linux/CUDA-only), so `vllm/entrypoints/llm.py` has not been read. This characterization comes from upstream's documented "`LLM` is not thread-safe" stance and historical `_run_engine` behavior. **Task 1 verifies it. If false, changes A and C must be revised.**

`Qwen3VLLMBackend` holds one `vllm.LLM` (`app/services/asr/qwen3_vllm.py:201`). Inference dispatches through `loop.run_in_executor` on `max(4, os.cpu_count())` threads (`app/core/executor.py:34-35`), so without the router lock multiple requests call `generate` on different OS threads simultaneously.

`_run_generate` (`qwen3_vllm.py:255`) reassembles results **positionally**:

```python
outputs = self._llm.generate(prompts, sampling_params=self._sampling_params, use_tqdm=False)
for output, (_audio, _context, language) in zip(outputs, audio_items):
```

The hypothesis: `generate` adds every prompt to the shared `LLMEngine`, then drains it, collecting whatever finishes — without filtering to the request IDs this caller submitted. Two concurrent callers each collect a mix of both requests' outputs; `zip` silently truncates the mismatch, pairing request B's transcripts with request A's segments. Wrong text, no exception. Concurrent stepping also races scheduler internals assuming a single stepper.

This is the "cross-request mixing" that `ad05bc4` chased, `91f812e` reverted, and `a0e6900` re-landed. Each fixed it at the router; if the hypothesis holds, the defect belongs to `Qwen3VLLMBackend`.

### The engines and their call sites

**Two independent vLLM engines, not one:** `self._llm` (`qwen3_vllm.py:201`, ASR model) and `self._forced_aligner` (`qwen3_vllm.py:233`, `runner="pooling"`). No shared state; cross-engine mixing is impossible.

| Site | Line | Engine | Path |
|---|---|---|---|
| `_run_generate` → `self._llm.generate` | `qwen3_vllm.py:268` | main | offline transcription |
| `_decode_stream` → `self._llm.generate` | `qwen3_vllm.py:455` | main | realtime / websocket |
| `align_transcript` → `aligner.encode` | `qwen3_vllm.py:390` | aligner | offline alignment (`word_timestamps` only) |
| `_get_forced_aligner` → `self._llm_cls(...)` | `qwen3_vllm.py:233` | aligner (constructs it) | lazy init, any path |

The router lock covers the offline sites only. The websocket path calls `acquire_engine` directly (`app/services/qwen3_websocket_asr.py:85`, `app/services/websocket_asr.py:179` — under `app/services/`, not `app/services/asr/`; do not confuse with the schema file `app/models/websocket_asr.py`), bypasses `run_offline`, and reaches `_decode_stream` guarded only by the 8-permit semaphore. **Verified: two websocket sessions, or one websocket plus one offline request, hit `self._llm.generate` concurrently today.** `90f1c76` ("fix: deduplicate qwen websocket results") may be a symptom — speculation, not established.

`_get_forced_aligner` (`:222-249`) is an unsynchronized double-checked init constructing a CUDA engine. Defused today only because `_warmup_forced_aligner` (`qwen3_engine.py:159,329-333`) preloads at startup, and only when `forced_aligner_path` is configured. Load-bearing and undocumented; needs a guard, not a comment. **Note it is called from inside `align_transcript` (`:387`) immediately before `encode` (`:390`)** — see A's scope item 2 for why that ordering makes lock granularity a correctness matter, not a style one.

For completeness of the census: `app/api/v1/asr.py:346` also acquires an engine lease directly, but only reads `.device` — it never reaches `generate`/`encode`, so it needs no guard.

### Bottleneck 2 — diarization

One shared modelscope CAM++ pipeline, serialized by `threading.BoundedSemaphore(1)` (`app/utils/speaker_diarizer.py:25`) — a mutex:

```python
pipeline = get_global_diarization_pipeline()
with _diarization_inference_semaphore:
    result = pipeline(audio_path)
```

Default-on, so every default request passes through it. **But it is not the binding constraint today** — the router lock already permits only one request into the pipeline at a time. The mutex only starts to matter once the router lock is narrowed. Provenance: it arrived in `34f0719`, a refactor, not a bugfix; no commit demonstrates it fixed observed corruption. Defensive against a stateful singleton — reasonable but unproven.

`_enable_batched_sv` is **already applied unconditionally** (`speaker_diarizer.py:238`); there is nothing to enable.

### Bottleneck 3 — FunASR VAD: latent today, opened by change A

`_vad_inference_lock` (`app/services/asr/engines/global_models.py:21`) is exposed via `get_vad_inference_lock()` at line 71 and has **zero callers** (verified). `app/utils/audio_splitter.py:97-102` calls the shared model directly. The PUNC analogues *are* locked — an accidental omission.

**The race is LATENT, not live.** r3 and r4 both got this wrong, in opposite directions. Verified:
- The only FunASR-family entry is `paraformer-large` (`app/services/asr/models.json`), which declares **only** `"realtime"` — there is no offline FunASR model.
- The offline pipeline never accepts a caller-supplied model id: `offline_transcription_service.py:89` always calls `get_default_offline_model_id()`, and `model_selection.py:41-50` restricts that to entries with `has_offline_model` — i.e. only `qwen3-asr-*`. `run_offline` has exactly one production caller.
- The global FunASR VAD is reachable only through `transcribe_long_audio`'s fallback (`base.py:163-167`). No FunASR-family request can reach it. The FunASR websocket path uses the PUNC lock (`websocket_asr.py:723`), never the VAD splitter.

So no cross-family race exists today. **r4's "exists now" claim was false.**

**But the inversion matters more: change A opens the first real concurrent path to the unguarded VAD model.** Today the router lock means at most one Qwen offline request is inside `transcribe_long_audio`. After A narrows it, up to 8 overlap (the semaphore's permits), and any two requests with `enable_speaker_diarization=False` — or whose diarization returns no segments — will concurrently hit `vad_model.generate(input=audio_path, cache={})` (`audio_splitter.py:102`) with the orphan lock still uncalled.

**Therefore wiring `get_vad_inference_lock()` into `audio_splitter.py:102` is part of change A, not change B.** It is the one thing the wide router lock is accidentally protecting, so narrowing the lock without it introduces a bug. A one-line fix, but load-bearing.

Caveat: that FunASR VAD `generate` with a fresh `cache={}` is genuinely unsafe is asserted by the author's comment, not demonstrated. If it is provably safe, the lock is unnecessary and A can skip it — but the default must be to guard.

### Design principle

Every stage becomes concurrent by one of two mechanisms: **one object made safe for many callers**, or **many objects**. Model size decides which.

- Large model (ASR, dominates VRAM) → one engine, many callers. Replication is unaffordable and would fragment the KV cache that continuous batching depends on.
- Small models (CAM++, FunASR VAD, tens of MB) → an instance pool. Follows `LocalEnginePool` (imported `router.py:19`, used `router.py:189`), already serving the rust path.

### The changes

**A. Narrow the lock, and guard every call site.** *(correctness + unblocks everything else)*
- Replace the router-level `asyncio.Lock` with **two `threading.Lock`s inside `Qwen3VLLMBackend` — one per engine**. The aligner is separate; a single lock would make alignment block generation for no correctness reason.
- Delete `_vllm_offline_locks` from the router.
- **This is the key restructure.** r3 kept the router lock until C landed, which stranded B (see Ordering). Narrowing here means diarization, splitting, and response assembly of different requests overlap immediately, and it closes the live websocket race on `_decode_stream`.

**A's scope is wider than "move the lock." Four things must land together or A introduces bugs:**

1. **Main-engine lock** — guards `_run_generate:268` and `_decode_stream:455`.
   **The lock must start before `:450`, not at `:455`.** `_decode_stream` uses the shared HF tokenizer at `:450` (`self._tokenizer.encode`) and `:453` (`.decode`) *before* reaching `generate`. `self._tokenizer` (`qwen3_vllm.py:188`) is shared across all sessions, and HF fast tokenizers (Qwen → `Qwen2TokenizerFast`) raise `RuntimeError: Already borrowed` under concurrent `encode` — each call re-enters `set_truncation_and_padding`, taking a Rust `borrow_mut`. Up to 8 concurrent websocket sessions call this from executor threads **today**, and would continue to after A if the lock covered only `:455`. Guarding the tokenizer + generate as one critical section closes it. (This is websocket-only — the router lock never protected it — but it falls inside this spec's goal of making the websocket path safe.)

2. **Aligner lock — one critical section, not two. `threading.Lock` is not reentrant.**
   `align_transcript` calls `_get_forced_aligner()` at `qwen3_vllm.py:387`, three lines before `aligner.encode(...)` at `:390`. Guarding both listed sites naively **nests the acquisition and self-deadlocks on the first alignment call.** (Today this would be masked by the startup warmup pre-populating the aligner — the exact undocumented accident this spec sets out to remove, so it must not be relied on.)
   **`_get_forced_aligner` has three callers, not one** — `align_transcript:387`, `ensure_forced_aligner_loaded:251-253` (the startup warmup, from `qwen3_engine.py:159`), and any future path. So "take the lock in `align_transcript` and let `_get_forced_aligner` assume it is held" is **wrong**: the warmup would then init with no lock, recreating exactly the unguarded-init-saved-by-startup-timing accident this spec refuses to rely on.
   Required shape: **a separate init-only lock inside `_get_forced_aligner`, never held across `encode`**, plus the engine lock around `encode`. That is safe for all three callers and cannot nest. `RLock` is an acceptable alternative but must be a deliberate choice, not a workaround for accidental nesting.

3. **Wire the orphan VAD lock** into `audio_splitter.py:102`. A is what makes concurrent VAD reachable (see Bottleneck 3). Without it, A trades one bug for another.

3b. **Guard the wetext ITN singleton.** `_wetext_normalizer` (`app/utils/text_processing.py:12-29`) is a module-level lazy singleton with an **unsynchronized double-checked init and no lock at all** — weaker than the VAD/PUNC models in `global_models.py`, which at least lock construction — wrapping a pynini/OpenFST `Normalizer`. It has **no startup warmup**, so unlike the forced aligner nothing masks the init race.
   It is squarely on the default offline path: `offline_transcription_service.py` hard-codes `enable_itn=True` into every `OfflineASRRequest`, and every transcript flows through `normalize_asr_text` inside the backend (`qwen3_vllm.py:289,301,348`) — i.e. inside the `transcribe_long_audio` call the router lock currently serializes.
   *Failure after A without this:* two concurrent offline requests both observe `_wetext_normalizer is None`, both construct FST normalizers (duplicate multi-second FST load, torn global), then concurrently call `.normalize()` on one shared FST whose thread safety is unverified.
   Guard the init and — absent proof `.normalize()` is thread-safe — the call, same treatment as VAD.
   *A live race exists today, but not the obvious one.* The websocket ITN calls (`qwen3_websocket_asr.py:107` via `:135,:340,:412`; `websocket_asr.py:854`) run **after** `await run_sync(...)` returns — i.e. on the **event-loop thread**, not an executor thread. So websocket↔websocket ITN calls are serialized by the single event loop and **cannot** race each other. The only live race today is a websocket event-loop call against an offline request's executor thread inside `normalize_asr_text`. A adds the offline↔offline exposure. One guard fixes both — but **a regression test built around concurrent websocket sessions alone would never fail**, so target the mixed websocket + offline shape instead.

4. **Resolve the executor/semaphore interaction** (see Risks — starvation bites at A, not B). Today's 8-permit semaphore admits 8 concurrent requests; before A they queue on an `asyncio.Lock` holding **no** executor thread, but after A each admitted request occupies one for its entire duration, most blocked inside the new backend lock. With the pool at `max(4, os.cpu_count())` this can exhaust the pool and starve websocket decode. A must either size the executor pool against the semaphore, reduce the admission limit, or restructure `_run_offline` so blocked requests do not hold threads.
   **Permits are not all request-scoped — do not size as if they were.** The Qwen websocket lease takes a permit at session start (`qwen3_websocket_asr.py:85`) and releases it only at connection teardown (`:388`), so a permit is held for the **entire connection lifetime, including silence**. Two consequences: (a) 8 concurrent websocket connections permanently exhaust the semaphore and block *all* offline traffic regardless of executor sizing; (b) "reduce the admission limit" would directly cap concurrent websocket sessions — a product-visible regression, not a tuning knob. Any resolution must either rescope websocket permits to per-decode or size against `offline_requests + concurrent_ws_sessions`.
   **This also resolves the Goals item on `_VLLM_SHARED_CONCURRENCY`:** under A the semaphore becomes the real admission bound (no longer shadowed by the router lock), so it stops being a dead knob — but its value must be chosen against both the executor pool size and the websocket permit lifetime, not left at 8 by inheritance.

- Items 1-3 are justified by evidence already in hand, independent of task 1's outcome and of any measurement. Item 4 needs a decision, not a measurement.

**What else the wide lock accidentally protects — A's central bet.**

Two shared-mutable singletons on the path, both requiring guards in A: the **VAD fallback** (item 3) and the **wetext ITN normalizer** (item 3b).

Confirmed *not* to need guards: temp files are `NamedTemporaryFile` with unique names; diarization has its own mutex (`speaker_diarizer.py:25,282`); the CAM++ monkeypatch is per-instance (`speaker_diarizer.py:202`); the Qwen offline path never touches the PUNC model (the vLLM branches at `qwen3_engine.py:368-373` and `:487-494` never pass `enable_punctuation` down); `app/api/v1/asr.py:346` takes an engine lease but only reads `.device`; `transcribe_long_audio`'s own state is function-local.

> **Confidence note.** r5 asserted this audit found "only the VAD fallback" and called it verified. Review found the ITN singleton it had missed — on the default path, with no warmup to mask it. The audit has now been performed twice and grown an item each time. **Treat this list as the best current knowledge, not as proven-exhaustive.** The implementer should re-grep for module-level mutable state on the `transcribe_long_audio` path before landing A, rather than trusting this list. The general pattern to search for: module-level `_x = None` plus a lazy `if _x is None:` initializer.

**B. Pool the small models.** *(throughput; depends on A)*
- Diarization: N CAM++ instances via a pool; delete `_diarization_inference_semaphore`. Nothing shared, so no lock needed.
  - *Verified feasible:* `_enable_batched_sv` monkeypatches **per instance** — `types.MethodType(batched_forward, pipeline_instance)` (`speaker_diarizer.py:202`), with the closure capturing per-call `modelscope_device`/`max_batch_size`. The `_batched_sv_enabled` guard (line 112) is also per-instance. No class- or module-level mutation; N instances are independent.
  - *Unverified:* whether `modelscope.pipelines.pipeline()` **construction** is thread-safe (global registry, config resolution, possible downloads in `resolve_model_path`). **Pool warmup must be serialized** unless proven otherwise.
- VAD: pool the FunASR VAD model *only if* measurement shows the fallback path (diarization disabled/empty) is hot enough to matter. **A already makes it correct** via the inference lock; B's pooling would only make it concurrent. Lower priority than diarization, which is on the default path.
- Pool size N: start at 4, config-driven, **sized against the executor pool** (see Risks).
- **VRAM budget required before implementing.** Each pool instance is *four* models — the main pipeline plus rebuilt `sv_pipeline`, `vad_pipeline`, `change_locator_pipeline` (`speaker_diarizer.py:127-146`). N×4 modelscope models plus N large SV batches must be budgeted against what remains after vLLM's `gpu_memory_utilization` reservation across two engines (`qwen3_vllm.py:121`).

**C. Async engine for the ASR model.** *(throughput; the largest change)*
- Eliminates the mixing bug **by construction** — results route by request ID, not list position — so A's main-engine lock disappears rather than moving. Enables continuous batching.
- **API target unverified.** `AsyncLLMEngine` is a compatibility alias for the v1 engine `vllm.v1.engine.async_llm.AsyncLLM`; the v0 implementation is gone and upstream carries a TODO to remove the proxy. The real target is `AsyncLLM`. Whether the alias still imports at 0.19.0, and whether `multi_modal_data` audio prompts behave identically on the async path, must be confirmed in the deployed image alongside task 1.
- **C does not address the aligner.** `aligner.encode(..., pooling_task="token_classify")` (`qwen3_vllm.py:390`) is a sync pooling engine. After C it stays behind A's lock permanently, becoming the new global serialization point for `word_timestamps=True` traffic. Whether v1 `AsyncLLM` supports pooling/`encode` with multimodal audio at 0.19.0 is **unknown and must be established before C is scoped.** If it does not, aligner concurrency needs a pool or stays serial — an accepted limitation for non-default traffic.
- Blocked on task 1.

### Ordering

1. **Verify the root cause** (task 1) — read `vllm/entrypoints/llm.py` in the deployed image; confirm/refute for 0.19.0. Also check the `AsyncLLM` API surface and pooling support. Blocks C.
2. **Instrument the stage breakdown** (task 2) — supplies the numbers this spec refuses to guess. Does not block A.
3. **A — narrow the lock, guard all call sites.** No dependencies.
4. **B — pool diarization.** *Depends on A.* (VAD pooling is optional and measurement-gated — A already makes VAD correct.)
5. **Measure.**
6. **C — async engine**, if inference is the ceiling.

**Correction to r3's ordering claim.** r3 asserted A and B each "stand alone" and no task is wasted. False for B: with the router lock intact, one request is inside diarization at a time no matter how many CAM++ instances exist, so B lands for **zero** gain and the post-B measurement would falsely condemn it. B's payoff requires the lock narrowing — which r3 deferred to C. Moving the narrowing into A resolves the inversion: B now depends only on A, not on the vLLM migration.

**Correction to r4's independence claim.** r4 said "the VAD pooling half of B is genuinely independent (the cross-family race is live today)." Both halves of that are wrong: the race is latent, not live (no offline FunASR model exists), and VAD *guarding* is not independent of A — it is a prerequisite *within* A, because A is what makes concurrent VAD reachable. VAD *pooling* remains optional and measurement-gated.

A stands alone on correctness, with the wider scope stated above. C is the only change gated on an unverified hypothesis.

### Landed already

`DIARIZATION_SV_BATCH_SIZE` (`config.py:65`, default 32, env override at `config.py:120-122`) is now passed at the diarization pipeline's construction site (`speaker_diarizer.py:238-242`). `_enable_batched_sv`'s own `max_batch_size: int = 32` default (`speaker_diarizer.py:98`) is **retained** so the function stays independently testable; the sole production caller overrides it from settings. The old value was chosen to avoid OOM on an unspecified GPU and was unreachable from config; on an H100 with a tens-of-MB model it is conservative. Tuning it is now an env change, not a code change.

This is a knob, not a fix — it reduces sequential passes within one request's diarization; it does nothing for concurrency. Not added to `.env.example`, which is a curated minimal file that also omits `ASR_BATCH_SIZE`, `MAX_SEGMENT_SEC`, and the worker counts.

### Goals

- 10 concurrent 5-minute requests use H100 capacity rather than queueing behind global locks.
- Cross-request mixing impossible on the offline path **and** the websocket path.
- `_VLLM_SHARED_CONCURRENCY = 8` either governs something real or is removed — no dead knobs.

### Non-goals

- `ASR_BATCH_SIZE` and the sequential inter-batch loop (`engines/base.py:186`) stay as-is — that governs single-request latency.
- The rust/CPU path is untouched: `LocalEnginePool`, per-request exclusive engines, no shared state, no bug. Unreachable on a CUDA host (`router.py:93`).
- Device placement: VAD, diarization, and alignment were verified already on CUDA (`global_models.py:56`, `speaker_diarizer.py:233`; the aligner is a vLLM engine by construction).

### Rejected alternatives

- **Raise `_VLLM_SHARED_CONCURRENCY`.** No effect — acquired inside the router lock. It permits the concurrency that causes the bug; it does not prevent it.
- **Coalescing dispatcher** — one thread owning the engine, merging concurrent calls into one batched `generate`. Real batching without the async migration, but bespoke machinery (queue, batch-window policy, result routing) reimplementing what the async engine already does. Fallback if C proves intractable.
- **Multiprocessing / multiple uvicorn workers.** The GIL is not the bottleneck — PyTorch releases it during CUDA compute; shared singletons are. Each process needs its own model copy plus a CUDA context, and splitting `gpu_memory_utilization` N ways shrinks each engine's KV cache, competing with continuous batching for the same VRAM. On 80GB this caps at ~2-3 workers. (vLLM itself uses processes for tensor parallelism across GPUs, deliberately not for concurrent requests on one GPU.)
- **Disabling diarization by default.** Out of scope, but if much traffic does not need speaker labels, `enable_speaker_diarization=False` on those requests is the cheapest available win and should be evaluated independently.

### Risks

- **The root cause may be wrong** (task 1). Largest risk; C depends on it entirely.
- **GPU saturation caps everything.** All stages share one H100. Pooling and continuous batching expose headroom; they do not add capacity. The realistic ceiling is unknown until task 2.
- **Websocket latency.** `_decode_stream` is unguarded today and is fast *because* it is racy. A serializes it against offline work; C should recover this. Measure with a stated threshold.
- **Executor pool starvation — bites at A, and A is where it must be handled.** r4 attributed this to B; that was wrong. Today the 9 queued requests wait on an `asyncio.Lock` holding **no** executor thread — only one `run_sync(transcribe_long_audio)` thread is live. After A, all admitted requests (semaphore = 8) enter `_run_offline` concurrently and each occupies a `ThreadPoolExecutor` thread for its whole duration (`router.py:209-222`), most of them **blocked inside the new backend lock**. A converts non-thread-holding async waiting into thread-holding blocking. With the pool at `max(4, os.cpu_count())` (`executor.py:34-35`), a low-core container exhausts it the moment A lands, starving websocket decode — which needs executor threads via `run_sync` — with zero B code present. See A's scope item 4.
  B then compounds it: `diarize()` blocks on pool acquisition while holding an executor thread (`speaker_diarizer.py:282`, inside `run_sync`). B must size its pool against the executor pool, not independently.
- **VRAM.** See B's budget requirement.

### Success criteria

- **The mixing regression test runs on the H100, not in CI.** vLLM is absent from dev/CI; a mocked engine cannot reproduce a scheduler race and a green mocked test would be worthless. H100-only integration check, inherently racy — run N≥20, treat as evidence, not proof. Must fail against a build with the guard removed, or it proves nothing.
- Equivalent concurrent-websocket and mixed websocket+offline tests.
- 10 concurrent 5-minute requests measurably improve against `main` on the H100. **No target multiple is set** — task 2 establishes what is achievable; a number invented now would repeat r3's error.
- Websocket p50/p95 decode latency does not regress beyond a threshold agreed at task 2.
- `tests/test_runtime_router.py` updated — it asserts `engine.max_active == 1` (line 59), the router-level locking A removes.
- No `git pull`/fetch/rebase during the branch; single commit.

### Development environment

The dev box is an **AMD APU, not CUDA**. `router.py:93` routes non-CUDA to `QWEN_RUST_CPU`, which uses `LocalEnginePool` with per-request exclusive engines — **no shared engine, no mixing bug, no serialization**. vLLM will not install there (`pyproject.toml:33`).

Consequence: **none of this is reproducible or verifiable locally.** Tasks 1, 2, and every integration test require the H100. Locally you can develop structure — router logic, the pool pattern, service refactoring, unit tests with a fake engine — but none of it can be trusted until it runs on the H100.

If ROCm is installed, `torch.cuda.is_available()` may return `True`, making `detect_device` return `"cuda:0"` and routing to the vLLM path, which then fails at import — a confusing error rather than a clean fallback. Set `DEVICE=cpu` explicitly locally to force the rust path.

## Changelog

**r7** — the r6 review returned **GO**; these were its remaining non-blocking findings, applied:
- **Shared HF tokenizer added to A item 1.** `_decode_stream` uses `self._tokenizer` at `:450`/`:453` *before* the `generate` at `:455`. A lock starting at `:455` leaves it exposed; `Qwen2TokenizerFast` raises `RuntimeError: Already borrowed` under concurrent `encode`. Websocket-only (the router lock never covered it), but inside this spec's websocket-safety goal.
- **ITN "live today" mechanism corrected.** r6 said the websocket ITN calls run on executor threads; they run on the **event-loop thread** (after `await run_sync` returns), so websocket↔websocket ITN cannot race. The only live race is websocket vs offline. Remedy unchanged, but a websocket-only regression test would never fail — noted so the test targets the mixed shape.
- PUNC evidence re-cited to the vLLM branches (`qwen3_engine.py:368-373`, `:487-494`); `qwen3_engine.py:234` was the rust path.
- Websocket permit release is `qwen3_websocket_asr.py:388`, not `:387`.
- The r6 review's independent exhaustive sweep for module-level mutable state on the `transcribe_long_audio` path found **no third missed singleton** — the audit list (VAD + ITN) is complete as of r6.

**r6** — following adversarial review of r5:
- **A's audit was incomplete: it missed the wetext ITN singleton.** `_wetext_normalizer` (`text_processing.py:12-29`) is a module-level lazy singleton with no lock and no startup warmup, on the **default** offline path (`enable_itn=True` hard-coded; `qwen3_vllm.py:289,301,348`). r5 called the audit "verified" and "A's central bet, and it holds" — it did not. Added as A scope item 3b. Also live on the websocket paths today (`qwen3_websocket_asr.py:108`, `websocket_asr.py:854`); the same guard fixes both.
- **The audit is no longer claimed exhaustive.** It has been run twice and grown an item each time. It is now labelled best-current-knowledge with an instruction to re-grep for module-level lazy singletons before landing A.
- **Semaphore permit lifetime corrected.** r5's item 4 sized permits as if request-scoped. The Qwen websocket lease holds a permit for the whole *connection* (`qwen3_websocket_asr.py:85` → `:387`), so 8 concurrent websocket sessions block all offline traffic regardless of executor sizing, and "reduce the admission limit" is a product-visible cap on websocket sessions rather than a tuning knob.
- **Aligner lock shape corrected.** `_get_forced_aligner` has three callers, including the startup warmup (`qwen3_vllm.py:251-253` ← `qwen3_engine.py:159`). r5's first suggested option ("`_get_forced_aligner` assumes the lock is held") would leave the warmup unguarded. Now mandates a separate init-only lock never held across `encode`.
- Ordering step 4 no longer contradicts the VAD-pooling demotion.

**r5** — following adversarial review of r4:
- **The "live cross-family VAD race" was false.** Verified: `paraformer-large` (`models.json`) declares only `"realtime"`, there is no offline FunASR model, and `offline_transcription_service.py:89` always resolves via `get_default_offline_model_id()` → `qwen3-asr-*` only. No FunASR request can reach the VAD fallback. r3 said VAD was "safe by accident"; r4 over-corrected to "live now"; both wrong. It is **latent**.
- **Inverted consequence: change A opens the race.** Narrowing the router lock is what first makes concurrent VAD reachable (two `enable_speaker_diarization=False` requests, or empty diarization). **Wiring the orphan VAD lock moved from B into A** — it is the one thing the wide lock accidentally protects, so A without it trades one bug for another. r4's claim that A "stands alone on correctness" was true only by accident.
- **Executor starvation reattributed from B to A.** Before A, queued requests wait on an `asyncio.Lock` holding no executor thread; after A they block inside a `threading.Lock` while holding one. A converts non-thread-holding waits into thread-holding blocks and can exhaust `max(4, nproc)` with zero B code present. Added to A's scope, along with the `_VLLM_SHARED_CONCURRENCY` resolution the Goals demanded but The Changes never delivered.
- **Aligner lock granularity specified.** `align_transcript:387` calls `_get_forced_aligner()` before `encode:390`; guarding both naively nests a non-reentrant `threading.Lock` → self-deadlock on first alignment, masked only by the startup warmup this spec wants to stop relying on. Init and encode must share one critical section.
- **B's VAD pooling demoted** to optional and measurement-gated — A makes VAD *correct*; pooling would only make it *concurrent*, and it is off the default path.
- Recorded the audit result that nothing *else* depends on the wide lock (temp files, diarization mutex, per-instance monkeypatch, PUNC unused by Qwen offline, all-local state) — A's central bet, verified.
- "Landed already" corrected: the `max_batch_size=32` default at `speaker_diarizer.py:98` is *retained* for testability and overridden by the caller, not replaced.
- Citations: `MAX_SEGMENT_SEC` at `config.py:68`; diarization mutex at `speaker_diarizer.py:282`; added the `asr.py:346` health-check lease to the census.

**r4** — following adversarial review of r3:
- **Removed all throughput projections.** The r3 stage table and Amdahl rows were structurally wrong, not imprecise. Two attempts at these numbers were both wrong; the third is a measurement, not an estimate. The specific errors are recorded so they are not repeated.
- **Request shapes corrected.** VAD is a *fallback* (`base.py:161`), not a stage, and does not run on the default path; alignment requires `word_timestamps=True`, default `False`. r3 budgeted four stages that never co-occur.
- **Lock narrowing moved into A.** r3 kept the router lock until C, which made B a guaranteed 0× — one request at a time in diarization regardless of pool size. r3's "each change stands alone / no task is wasted" claim was false for B. B now depends on A alone, not on the vLLM migration.
- **"VAD safe only by accident" corrected.** The router lock is `QWEN_VLLM`-only; a FunASR-family request races a Qwen request's VAD fallback *today*, unlocked. Conditional on a FunASR model actually being served — to be confirmed.
- **C's API target flagged unverified.** `AsyncLLMEngine` is an alias for v1 `AsyncLLM`; r3 asserted the API without checking, the same error r3 called out elsewhere.
- **C does not cover the aligner** — a sync pooling engine that becomes the new serialization point for `word_timestamps` traffic after C. r3 never asked whether async pooling exists.
- **B feasibility part-verified:** the `_enable_batched_sv` monkeypatch is per-instance, so replication works. Pipeline *construction* thread-safety is unverified → serialize pool warmup. VRAM budget (N×4 models) added as a precondition.
- **Executor starvation tied to B's pool sizing**, where it bites first; r3 named it only generically.
- Citation fixes: `LocalEnginePool` import at `router.py:19`; the `max_active` assertion at `test_runtime_router.py:59`.
- Recorded `DIARIZATION_SV_BATCH_SIZE` as landed.

**r3** — restructured around three coordinated changes; added the (now removed) estimates table; added the size-based design principle and pooling; recorded `_enable_batched_sv` as already-on; added rejected alternatives; added Development Environment.

**r2 corrected r1** — two engines not one (r1 specified a single lock; the aligner is separate); root cause demoted to unverified hypothesis; lazy aligner init promoted from comment to guarded site; H100 incorporated; diarization mutex promoted to in-scope; executor starvation named; success criteria made executable; websocket file paths corrected.

## Plan

# ASR Concurrency — Implementation Plan (Tasks 1, 2, Change A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **NO COMMITS.** Workers make no git commits. Skip every "commit" step you would normally add; the branch owner integrates as a single commit at the end (see Success criteria: "single commit").

**Goal:** Verify the vLLM root-cause hypothesis (task 1), instrument the offline stage breakdown (task 2), and land change A — narrow the router lock to per-engine backend locks with every accidental-protection site guarded — per the Spec's "The changes → A" section.

**Architecture:** See Spec sections "Root cause", "The engines and their call sites", and "The changes → A". This plan does not restate them; each task cites the spec item it implements.

**Tech Stack:** Python 3 / FastAPI / vLLM 0.19.0 (H100 only) / `threading.Lock` / `unittest` (`tests/` uses `unittest.IsolatedAsyncioTestCase`, run via `python -m pytest tests/ -v`).

**Deliberately out of scope:** Changes **B** (small-model pooling) and **C** (async engine) are NOT planned here. C is gated on task 1's verdict; B is gated on A landing and task 2's measurements showing where the time goes. Plan them after tasks 1 and 2 have run on the H100.

### Global Constraints

- **Dev box is an AMD APU, non-CUDA. vLLM will not install** (`pyproject.toml:33`). Every task below is tagged **[LOCAL]** (structural work + unit tests with fakes, runnable on the dev box) or **[H100-ONLY]** (requires the deployed image; nobody has access yet — these tasks produce runbooks/scripts now and are *executed* later). Do not attempt to run a `[H100-ONLY]` step locally; do not claim a `[LOCAL]` task "verifies" concurrency behavior — locally you can only verify lock structure with fakes (Spec: "Development environment").
- Set `DEVICE=cpu` in any local run to avoid the ROCm false-CUDA trap (Spec: "Development environment").
- TDD wherever a fake-engine unit test can express the behavior; the mixing/race behavior itself is only testable on the H100 (Spec: "Success criteria").
- No `git pull`/fetch/rebase during the branch; single final commit by the branch owner; **workers commit nothing**.
- All of A's guards (Tasks 3–7) must land **together** — removing the router lock (Task 6) with any guard missing trades one bug for another (Spec: "A's scope").
- Run tests with: `python -m pytest tests/ -v` from the repo root.

### Task dependency graph

```
Task 1 [H100-ONLY runbook now, execute later]  — independent; gates future C
Task 2 [LOCAL instrumentation] → Task 2b [H100-ONLY measurement run] — independent of A
Task 3 (backend locks)   ─┐
Task 4 (VAD lock wire)    ├─ [LOCAL, parallel with each other and with 1/2]
Task 5 (ITN guard)        │
Task 7 (semaphore/executor)┘
Task 8 (singleton re-grep audit) — after 3,4,5; before 6
Task 6 (remove router lock + fix test_runtime_router) — LAST of the A tasks; depends on 3,4,5,7,8
Task 9 [H100-ONLY] integration test scripts — written any time after 3–7; executed on H100
```

Tasks 1, 2, 3, 4, 5, 7 can run in parallel (disjoint files). Task 6 must be last among A tasks.

---

### Task 1: Verify the vLLM root-cause hypothesis [H100-ONLY execution; runbook + probe script written locally]

Implements Spec "Root cause — LOAD-BEARING AND NOT YET VERIFIED" and Ordering step 1. Gates change C (out of scope here); does **not** gate A.

**Files:**
- Create: `scripts/h100/verify_vllm_root_cause.py`

**Interfaces:**
- Produces: a written verdict (CONFIRMED / REFUTED, recorded in the spec's changelog by the branch owner) on: (a) does `LLM.generate` at 0.19.0 drain the shared `LLMEngine` without filtering to the caller's request IDs? (b) does `vllm.v1.engine.async_llm.AsyncLLM` import at 0.19.0, and does the `AsyncLLMEngine` alias still exist? (c) does `AsyncLLM` support pooling/`encode` with multimodal audio (for the aligner question in C)?

- [ ] **Step 1: Write the probe script (local, structural only — it will not run locally)**

```python
# scripts/h100/verify_vllm_root_cause.py
"""H100-ONLY. Run inside the deployed image (vllm[audio]==0.19.0).

Answers the three questions gating change C. Prints source excerpts; the
human reads them and records CONFIRMED/REFUTED in the spec changelog.
"""
import inspect
import sys


def main() -> int:
    import vllm
    print(f"vllm version: {vllm.__version__}")

    # Q(a): does LLM.generate/_run_engine filter outputs to this caller's request ids?
    from vllm.entrypoints.llm import LLM
    print("\n===== LLM.generate source =====")
    print(inspect.getsource(LLM.generate))
    for name in ("_run_engine", "_validate_and_add_requests"):
        fn = getattr(LLM, name, None)
        if fn is not None:
            print(f"\n===== LLM.{name} source =====")
            print(inspect.getsource(fn))
    # Read the printed source. CONFIRMED if _run_engine collects
    # engine.step() outputs without restricting to the request ids this
    # generate() call added; REFUTED if it filters/queues per caller.

    # Q(b): AsyncLLM availability and the alias.
    try:
        from vllm.v1.engine.async_llm import AsyncLLM
        print("\nAsyncLLM import: OK", AsyncLLM)
    except ImportError as exc:
        print("\nAsyncLLM import FAILED:", exc)
    try:
        from vllm import AsyncLLMEngine  # compatibility alias
        print("AsyncLLMEngine alias: OK", AsyncLLMEngine)
    except ImportError as exc:
        print("AsyncLLMEngine alias FAILED:", exc)

    # Q(c): pooling/encode surface on the async engine (aligner concern for C).
    try:
        from vllm.v1.engine.async_llm import AsyncLLM
        members = [m for m in dir(AsyncLLM) if m in ("encode", "pooling", "generate")]
        print("AsyncLLM relevant members:", members)
        if hasattr(AsyncLLM, "encode"):
            print("\n===== AsyncLLM.encode signature =====")
            print(inspect.signature(AsyncLLM.encode))
    except Exception as exc:
        print("AsyncLLM inspection failed:", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2 [H100-ONLY]: Run it in the deployed image**

Run: `python scripts/h100/verify_vllm_root_cause.py | tee /tmp/vllm_root_cause.txt`
Expected: script completes; source of `LLM.generate`/`_run_engine` printed.

- [ ] **Step 3 [H100-ONLY]: Read the output and record the verdict**

Decision rule (from Spec "Root cause"): the hypothesis is CONFIRMED iff the drain loop collects finished outputs without filtering to the caller's own request IDs. Record CONFIRMED/REFUTED plus the AsyncLLM answers. If REFUTED, changes A's *lock necessity for mixing* is weakened but A still stands on the documented "`LLM` is not thread-safe" stance and the tokenizer/ITN/VAD races — note this and re-review C's premise before scoping C.

---

### Task 2: Instrument the offline stage breakdown [LOCAL structural + H100-ONLY measurement]

Implements Spec Ordering step 2 ("supplies the numbers this spec refuses to guess"). Independent of A. Informs whether C is worth it and B's priorities.

**Files:**
- Modify: `app/services/asr/engines/base.py` — `transcribe_long_audio` (the diarization/split/batch/align pipeline around lines 152–186)
- Test: `tests/test_stage_timings.py`

**Interfaces:**
- Produces: one structured log line per offline request:
  `ASR_STAGE_TIMINGS task_id=<id> total_s=<f> diarization_s=<f> vad_split_s=<f> inference_s=<f> alignment_s=<f> segments=<n> batches=<n>` emitted via the module `logger` at INFO. Stages that did not run report `0.000` (Spec "Request shapes": stages are conditional). No behavior change.
- **Known limitation (deliberate):** `alignment_s` is ALWAYS `0.000` in this task. Alignment runs *inside* the backend's `transcribe_batch`, per segment, interleaved within each batch call (`qwen3_vllm.py:352` — `align_transcript` is invoked from the batch loop in `qwen3_vllm.py`, not from `base.py`), so `base.py`'s batch loop cannot separate it from inference. Consequently, for `word_timestamps=True` requests, `inference_s` INCLUDES alignment time. Default traffic (`word_timestamps=False`, the shape task 2b measures) has zero alignment, so its numbers are exact. If alignment ever needs its own number, that requires threading timing accumulators out of the backend's `transcribe_batch` return value — out of scope here; record it as a follow-up if `word_timestamps` traffic becomes a measurement target. The field is kept in the format so the log-line schema does not change later.

- [ ] **Step 1: Read `app/services/asr/engines/base.py` in full** and locate: the `enable_speaker_diarization` block (~152), the `AudioSplitter` fallback (~161–167), the batch inference loop (~186), and whatever alignment/post-processing follows. Adjust the exact insertion points below to what you find — the spec's line numbers are verified but the file may have drifted.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_stage_timings.py
from __future__ import annotations

import logging
import unittest
from unittest import mock


class StageTimingLogTest(unittest.TestCase):
    def test_stage_timing_log_line_shape(self) -> None:
        # Unit-test the formatting helper directly; the full pipeline
        # needs real models and is H100-only.
        from app.services.asr.engines.base import format_stage_timings

        line = format_stage_timings(
            task_id="t1",
            total_s=12.345,
            diarization_s=3.0,
            vad_split_s=0.0,
            inference_s=8.5,
            alignment_s=0.0,
            segments=5,
            batches=2,
        )
        self.assertIn("ASR_STAGE_TIMINGS", line)
        self.assertIn("task_id=t1", line)
        self.assertIn("diarization_s=3.000", line)
        self.assertIn("vad_split_s=0.000", line)
        self.assertIn("inference_s=8.500", line)
        self.assertIn("batches=2", line)
```

- [ ] **Step 3: Run it — must fail** with `ImportError: cannot import name 'format_stage_timings'`.

Run: `python -m pytest tests/test_stage_timings.py -v`

- [ ] **Step 4: Implement**

Add to `app/services/asr/engines/base.py` (module level):

```python
def format_stage_timings(
    *,
    task_id: str,
    total_s: float,
    diarization_s: float,
    vad_split_s: float,
    inference_s: float,
    alignment_s: float,
    segments: int,
    batches: int,
) -> str:
    return (
        "ASR_STAGE_TIMINGS "
        f"task_id={task_id} total_s={total_s:.3f} "
        f"diarization_s={diarization_s:.3f} vad_split_s={vad_split_s:.3f} "
        f"inference_s={inference_s:.3f} alignment_s={alignment_s:.3f} "
        f"segments={segments} batches={batches}"
    )
```

Inside `transcribe_long_audio`: initialize `import time` accumulators at the top (`_t_total = time.perf_counter()`, `diarization_s = vad_split_s = inference_s = 0.0`, `batches = 0`); wrap the diarizer call, the `AudioSplitter.split_audio_file` call, and each inference batch (the loop at ~:186 — accumulate into `inference_s` and increment `batches`) with `t0 = time.perf_counter()` / `stage_s += time.perf_counter() - t0`. **Do NOT attempt to time alignment separately** — it happens inside the backend's `transcribe_batch` per segment (`qwen3_vllm.py:352`) and is not observable from `base.py`; pass `alignment_s=0.0` always (see the Interfaces limitation above — `inference_s` includes alignment when `word_timestamps=True`). Just before returning, emit `logger.info(format_stage_timings(task_id=str(task_id or ""), total_s=time.perf_counter() - _t_total, ..., alignment_s=0.0))`. Do not change any control flow.

- [ ] **Step 5: Run the test — must pass.** Then run the full suite: `python -m pytest tests/ -v` — no regressions.

- [ ] **Step 6 [H100-ONLY]: Measurement runbook** — on the H100, after deploy: submit 1, then 10 concurrent, 5-minute-audio `POST /v1/audio/transcriptions` requests (defaults: diarization on, no word timestamps); grep the service log for `ASR_STAGE_TIMINGS`; compute per-stage shares and p50/p95 websocket decode latency baseline (the threshold Spec "Risks → Websocket latency" demands). These numbers gate B and C scoping. Record them; do not extrapolate from them locally.

---

### Task 3: Per-engine backend locks in `Qwen3VLLMBackend` [LOCAL]

Implements Spec "The changes → A", scope items **1** and **2**: main-engine `threading.Lock` covering tokenizer+generate in `_decode_stream` and generate in `_run_generate`; aligner engine lock around `encode`; a **separate init-only lock** inside `_get_forced_aligner` (three callers — Spec item 2 explains why naive nesting self-deadlocks and why "caller holds the lock" is wrong for the warmup path).

**Files:**
- Modify: `app/services/asr/qwen3_vllm.py` — `__init__` (:161–211), `_get_forced_aligner` (:222–249), `_run_generate` (:255–279), `align_transcript` (:376–423), `_decode_stream` (:447–474)
- Test: `tests/test_qwen3_vllm_locks.py`

**Interfaces:**
- Produces: three instance attributes on `Qwen3VLLMBackend`: `self._llm_lock: threading.Lock` (main engine + shared tokenizer), `self._aligner_lock: threading.Lock` (aligner `encode`), `self._aligner_init_lock: threading.Lock` (aligner construction only, never held across `encode`). Task 6 relies on these existing before the router lock is removed.
- Note: `Qwen3VLLMBackend.__init__` imports `vllm` via `importlib` (:170) — it cannot be constructed locally. Tests build the object with `Qwen3VLLMBackend.__new__` and set attributes by hand. This is the sanctioned local pattern; it verifies lock *structure*, not scheduler races (those are Task 9, H100).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_qwen3_vllm_locks.py
from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from app.services.asr.qwen3_vllm import Qwen3VLLMBackend, VLLMRealtimeState


class _FakeOutputToken:
    text = "hello"


class _FakeOutput:
    outputs = [_FakeOutputToken()]


class _ConcurrencyRecorder:
    """Fake vLLM LLM that records overlapping calls into ONE shared counter.

    The tokenizer fake shares this recorder (see _FakeTokenizer), so
    max_active counts tokenizer+generate overlap COMBINED. This is what makes
    the test non-vacuous: an implementation that locks only the generate at
    :455 leaves tokenizer calls at :450/:453 overlapping other threads'
    generate — combined max_active > 1 — and MUST fail here (spec A item 1:
    tokenizer + generate are one critical section).
    """

    def __init__(self) -> None:
        self._mu = threading.Lock()
        self.active = 0
        self.max_active = 0

    def _enter(self) -> None:
        with self._mu:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def _exit(self) -> None:
        with self._mu:
            self.active -= 1

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        self._enter()
        time.sleep(0.01)
        self._exit()
        return [_FakeOutput() for _ in prompts]


class _FakeTokenizer:
    """Shares the LLM's overlap recorder so a lock covering only generate
    (starting at :455 instead of before :450) fails the combined assertion."""

    def __init__(self, recorder: _ConcurrencyRecorder) -> None:
        self._recorder = recorder

    def encode(self, text, add_special_tokens=False):
        self._recorder._enter()
        time.sleep(0.005)  # widen the overlap window
        self._recorder._exit()
        return [1, 2, 3]

    def decode(self, ids, skip_special_tokens=False):
        self._recorder._enter()
        time.sleep(0.005)
        self._recorder._exit()
        return "abc"


class _FakeSamplingParamsCls:
    def __init__(self, **kwargs):
        pass


def _bare_backend() -> Qwen3VLLMBackend:
    backend = Qwen3VLLMBackend.__new__(Qwen3VLLMBackend)
    backend._llm = _ConcurrencyRecorder()
    backend._tokenizer = _FakeTokenizer(backend._llm)  # SHARED recorder — see class docstrings
    backend._sampling_params = object()
    backend._sampling_params_cls = _FakeSamplingParamsCls
    backend._max_inference_batch_size = 4
    backend._forced_aligner_path = "/fake/aligner"
    backend._forced_aligner = None
    backend._timestamp_token_id = None
    backend._timestamp_segment_time = None
    backend._llm_lock = threading.Lock()
    backend._aligner_lock = threading.Lock()
    backend._aligner_init_lock = threading.Lock()
    return backend


def _hammer(fn, n=8):
    # daemon=True: if the code under test deadlocks, the watchdog's timeout
    # assertion must be able to fail AND the interpreter must exit — non-daemon
    # hammer threads would block process exit and hang the whole suite instead
    # of reporting the failure.
    threads = [threading.Thread(target=fn, daemon=True) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


class MainEngineLockTest(unittest.TestCase):
    def test_run_generate_serialized(self) -> None:
        backend = _bare_backend()
        audio = np.zeros(160, dtype=np.float32)
        _hammer(lambda: backend._run_generate([(audio, "", None)]))
        self.assertEqual(backend._llm.max_active, 1)

    def test_decode_stream_serialized_including_tokenizer(self) -> None:
        backend = _bare_backend()

        def one_session() -> None:
            state = VLLMRealtimeState(
                prompt_raw="p",
                language="",
                chunk_size_sec=2.0,
                unfixed_chunk_num=0,  # forces the tokenizer branch at :450
                unfixed_token_num=2,
                max_new_tokens=8,
                raw_decoded="seed",
                audio_accum=np.zeros(160, dtype=np.float32),
            )
            backend._decode_stream(state)

        _hammer(one_session)
        # Combined tokenizer+generate overlap (shared recorder). A lock
        # covering only generate (:455) leaves tokenizer encode/decode
        # (:450/:453) racing other threads' generate → max_active > 1 → FAIL.
        self.assertEqual(backend._llm.max_active, 1)

    def test_decode_stream_and_run_generate_share_one_lock(self) -> None:
        backend = _bare_backend()
        audio = np.zeros(160, dtype=np.float32)

        def offline() -> None:
            backend._run_generate([(audio, "", None)])

        def ws() -> None:
            state = VLLMRealtimeState(
                prompt_raw="p", language="", chunk_size_sec=2.0,
                unfixed_chunk_num=0, unfixed_token_num=2, max_new_tokens=8,
                raw_decoded="seed", audio_accum=audio,
            )
            backend._decode_stream(state)

        threads = [threading.Thread(target=offline if i % 2 else ws) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(backend._llm.max_active, 1)


class _Sentinel(Exception):
    """Raised by the fake aligner's encode so align_transcript stops after
    the code under test (init + lock acquisition + encode) has run."""


class AlignerLockTest(unittest.TestCase):
    def test_align_transcript_does_not_self_deadlock_and_serializes_encode(self) -> None:
        # Spec item 2: the init-only lock and the encode lock must not nest
        # a non-reentrant Lock. With no warmup (aligner starts None), the
        # first align_transcript call exercises the REAL _get_forced_aligner
        # init path plus encode together, from 8 threads at once.
        backend = _bare_backend()
        recorder = _ConcurrencyRecorder()
        init_calls: list[int] = []
        init_mu = threading.Lock()

        class _FakeHfConfig:
            timestamp_token_id = 99
            timestamp_segment_time = 0.02

        class _FakeAlignerEngine:
            class llm_engine:  # noqa: N801 — mimics vLLM attribute shape
                class vllm_config:
                    class model_config:
                        hf_config = _FakeHfConfig()

            def encode(self, prompts, pooling_task=""):
                recorder.generate(prompts)  # overlap accounting only
                raise _Sentinel()

        class _FakeLLMCls:
            """Injected as backend._llm_cls: the aligner constructor."""

            def __new__(cls, **kwargs):
                with init_mu:
                    init_calls.append(1)
                time.sleep(0.005)  # widen the init race window
                return _FakeAlignerEngine()

        backend._llm_cls = _FakeLLMCls
        backend._gpu_memory_utilization = 0.5
        errors: list[BaseException] = []

        def run() -> None:
            try:
                backend.align_transcript(
                    audio_path="x",
                    text="hello world",
                    audio=np.zeros(160, dtype=np.float32),
                )
            except _Sentinel:
                pass
            except BaseException as exc:
                errors.append(exc)

        done = threading.Event()

        def guard() -> None:
            _hammer(run)
            done.set()

        watchdog = threading.Thread(target=guard, daemon=True)
        watchdog.start()
        # A nested non-reentrant lock shows up as a hang, not an exception.
        self.assertTrue(done.wait(timeout=10.0), "align_transcript deadlocked")
        self.assertFalse(errors, errors)
        self.assertEqual(len(init_calls), 1, "aligner constructed more than once")
        self.assertEqual(recorder.max_active, 1, "encode not serialized")
```

(Note: this drives the **production** `_get_forced_aligner` — `_FakeLLMCls` is injected as `backend._llm_cls`, so the real double-checked init, the real `_get_forced_aligner_gpu_memory_utilization` call, and the real config extraction all execute. If `_resolve_forced_aligner_gpu_memory_utilization` reads settings it must run cleanly on the dev box; it does — it is pure config math. Add `backend._gpu_memory_utilization = 0.5` to `_bare_backend` if you prefer it there.)

- [ ] **Step 2: Run — must fail.** `python -m pytest tests/test_qwen3_vllm_locks.py -v`
Expected: `AttributeError` on the lock attributes and/or `max_active > 1` assertions failing (the fakes bypass `__init__`, so the *first* failures come from the production code not taking the locks the tests hand it — `max_active == 1` assertions fail).

- [ ] **Step 3: Implement in `app/services/asr/qwen3_vllm.py`**

Add `import threading` to the imports. In `__init__`, after `self._timestamp_segment_time = None` (:211):

```python
        # Per-engine serialization (spec change A, items 1-2).
        # _llm_lock: the ASR engine AND the shared HF tokenizer — one
        # critical section, because _decode_stream touches the tokenizer
        # before generate and Qwen2TokenizerFast raises "Already borrowed"
        # under concurrent encode.
        # _aligner_lock: the pooling engine's encode().
        # _aligner_init_lock: aligner construction ONLY; never held across
        # encode (three callers of _get_forced_aligner; nesting a
        # non-reentrant Lock with _aligner_lock would self-deadlock).
        self._llm_lock = threading.Lock()
        self._aligner_lock = threading.Lock()
        self._aligner_init_lock = threading.Lock()
```

In `_run_generate`, wrap only the generate call (prompt building stays outside):

```python
        with self._llm_lock:
            outputs = self._llm.generate(
                prompts,
                sampling_params=self._sampling_params,
                use_tqdm=False,
            )
```

In `_decode_stream`, the lock must start **before :450** (tokenizer use), not at :455 — wrap the whole body from `prefix = ""` computation through the `generate` call. Concretely: indent lines 448–467 (from `prefix = ""` through the `[0]` of `generate`) under `with self._llm_lock:`; the post-processing from `generated = str(...)` onward is state-local and stays outside.

In `_get_forced_aligner`, replace the unsynchronized double-checked init (:226–247) with the init-only lock:

```python
        if self._forced_aligner is None:
            with self._aligner_init_lock:
                if self._forced_aligner is None:
                    ... existing construction body (:227-247) unchanged ...
        return self._forced_aligner
```

(Assign `self._forced_aligner` **last** in the construction body, after `_timestamp_token_id`/`_timestamp_segment_time` are set, so a concurrent reader outside the lock never sees a half-initialized aligner. Use a local variable during construction and publish it at the end.)

In `align_transcript`, wrap only the `encode` call (:390–393) with the engine lock — init already happened at :387 outside any encode-scoped lock, so no nesting:

```python
        with self._aligner_lock:
            outputs = aligner.encode(
                [{"prompt": prompt, "multi_modal_data": {"audio": audio_array}}],
                pooling_task="token_classify",
            )
```

- [ ] **Step 4: Run — must pass.** `python -m pytest tests/test_qwen3_vllm_locks.py -v`, then `python -m pytest tests/ -v` (no regressions).

---

### Task 4: Wire the orphan VAD inference lock [LOCAL]

Implements Spec "The changes → A", scope item **3** and "Bottleneck 3": `get_vad_inference_lock()` (`app/services/asr/engines/global_models.py:71`) has zero callers; `audio_splitter.py:102` calls the shared FunASR VAD model unguarded. A makes this path concurrently reachable.

**Files:**
- Modify: `app/utils/audio_splitter.py:93-102` (`get_vad_segments`)
- Test: `tests/test_vad_inference_lock.py`

**Interfaces:**
- Consumes: `get_vad_inference_lock()` from `app.services.asr.engines` (re-exported; verify the re-export exists in `app/services/asr/engines/__init__.py` — `get_global_vad_model` is already imported from there at `audio_splitter.py:94`; if `get_vad_inference_lock` is not re-exported, add it to that `__init__.py`'s imports/`__all__`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vad_inference_lock.py
from __future__ import annotations

import threading
import time
import unittest
from unittest import mock


class _RecordingVadModel:
    def __init__(self) -> None:
        self._mu = threading.Lock()
        self.active = 0
        self.max_active = 0

    def generate(self, input, cache):
        with self._mu:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.01)
        with self._mu:
            self.active -= 1
        return [{"value": [[0, 1000]]}]


class VadInferenceLockTest(unittest.TestCase):
    def test_concurrent_get_vad_segments_serialized(self) -> None:
        from app.utils.audio_splitter import AudioSplitter

        model = _RecordingVadModel()
        splitter = AudioSplitter(device="cpu")
        with mock.patch(
            "app.services.asr.engines.get_global_vad_model", return_value=model
        ):
            threads = [
                threading.Thread(target=splitter.get_vad_segments, args=("/fake.wav",))
                for _ in range(8)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(model.max_active, 1)
```

(If `AudioSplitter(device="cpu")` needs other constructor args, read `app/utils/audio_splitter.py`'s `__init__` and supply them; the patch target must match the import site inside `get_vad_segments` — it imports `from ..services.asr.engines import get_global_vad_model` *inside the method*, so patch `app.services.asr.engines.get_global_vad_model` as shown.)

- [ ] **Step 2: Run — must fail** with `max_active > 1`. `python -m pytest tests/test_vad_inference_lock.py -v`

- [ ] **Step 3: Implement** in `app/utils/audio_splitter.py` — change the method-local import at :94 and the call at :102:

```python
            from ..services.asr.engines import get_global_vad_model
            from ..services.asr.engines.global_models import get_vad_inference_lock
            ...
            with get_vad_inference_lock():
                result = vad_model.generate(input=audio_path, cache={})
```

(This mirrors the existing PUNC pattern at `websocket_asr.py:723`. Spec caveat noted: if VAD `generate` with fresh `cache={}` is someday proven thread-safe the lock can be dropped, but the default is to guard.)

- [ ] **Step 4: Run — must pass.** Then full suite.

---

### Task 5: Guard the wetext ITN singleton [LOCAL]

Implements Spec "The changes → A", scope item **3b**: `_wetext_normalizer` (`app/utils/text_processing.py:12-29`) has an unsynchronized double-checked init and no call guard; on the default offline path via `enable_itn=True` → `normalize_asr_text` (`qwen3_vllm.py:289,301,348`); no warmup masks it. Guard both init and `.normalize()` (thread safety of the FST unproven).

**Files:**
- Modify: `app/utils/text_processing.py`
- Test: `tests/test_itn_thread_safety.py`

**Interfaces:**
- Produces: module-level `_wetext_lock = threading.Lock()` guarding both `_get_normalizer()` construction and the `normalizer.normalize(text)` call in `apply_itn_to_text`. Public API (`apply_itn_to_text`, `normalize_asr_text`) unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_itn_thread_safety.py
from __future__ import annotations

import threading
import time
import unittest
from unittest import mock


class _RecordingNormalizer:
    def __init__(self) -> None:
        self._mu = threading.Lock()
        self.active = 0
        self.max_active = 0

    def normalize(self, text: str) -> str:
        with self._mu:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.005)
        with self._mu:
            self.active -= 1
        return text


class ItnThreadSafetyTest(unittest.TestCase):
    def test_single_init_and_serialized_normalize_under_contention(self) -> None:
        import app.utils.text_processing as tp

        constructed: list[_RecordingNormalizer] = []

        class _FakeNormalizerCls:
            def __new__(cls, lang="zh", operator="itn"):
                time.sleep(0.005)  # widen the init race window
                instance = _RecordingNormalizer()
                constructed.append(instance)
                return instance

        fake_wetext = mock.MagicMock()
        fake_wetext.Normalizer = _FakeNormalizerCls

        with mock.patch.dict("sys.modules", {"wetext": fake_wetext}):
            tp._wetext_normalizer = None  # reset the singleton
            try:
                threads = [
                    threading.Thread(target=tp.apply_itn_to_text, args=("一百二十三",))
                    for _ in range(8)
                ]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
            finally:
                tp._wetext_normalizer = None

        self.assertEqual(len(constructed), 1, "normalizer constructed more than once")
        self.assertEqual(constructed[0].max_active, 1, "normalize() not serialized")
```

- [ ] **Step 2: Run — must fail** (duplicate construction and/or `max_active > 1`; note `apply_itn_to_text` swallows exceptions, so assert on the recorders, as above — not on raised errors).

- [ ] **Step 3: Implement** in `app/utils/text_processing.py`:

```python
import logging
import threading

logger = logging.getLogger(__name__)

_wetext_normalizer = None
_wetext_lock = threading.Lock()  # guards init AND normalize(): FST thread safety unproven


def _get_normalizer():
    """获取wetext标准化器实例（单例模式，caller must hold _wetext_lock）"""
    global _wetext_normalizer
    if _wetext_normalizer is None:
        try:
            from wetext import Normalizer
            _wetext_normalizer = Normalizer(lang="zh", operator="itn")
            logger.info("WeText ITN模块初始化成功")
        except ImportError as e:
            logger.error(f"导入wetext失败: {e}")
            raise ImportError("请安装wetext库: pip install wetext")
        except Exception as e:
            logger.error(f"初始化wetext失败: {e}")
            raise
    return _wetext_normalizer
```

and in `apply_itn_to_text`, replace the two lines at :47–48 with:

```python
        with _wetext_lock:
            normalizer = _get_normalizer()
            result = normalizer.normalize(text)
```

(One lock for init + call keeps it simple and cannot nest — `_get_normalizer` is only called with the lock held; grep first to confirm `_get_normalizer` has no other callers: `grep -rn "_get_normalizer" app/ tests/`.)

- [ ] **Step 4: Run — must pass.** Then full suite.

- [ ] **Step 5: Note for Task 9** — the H100 regression test for ITN must use the **mixed websocket + offline** shape; websocket-only ITN calls run on the event-loop thread and can never race each other (Spec item 3b, r7 changelog).

---

### Task 6: Remove the router-level lock; update `tests/test_runtime_router.py` [LOCAL] — LAST of the A tasks

Implements Spec "The changes → A" bullets 1–2 (delete `_vllm_offline_locks`; the backend locks from Task 3 take over) and the Success-criteria requirement that `tests/test_runtime_router.py:59` (`engine.max_active == 1`) be updated — it encodes exactly the serialization A removes. **Do not start until Tasks 3, 4, 5, 7, 8 are done.**

**Files:**
- Modify: `app/services/asr/runtime/router.py:82,196-202`
- Modify: `tests/test_runtime_router.py`

**Interfaces:**
- Consumes: Task 3's backend locks (mixing protection), Task 4/5 guards (accidental-protection sites), Task 7's admission changes.
- Produces: `run_offline` with no vLLM special-case lock; offline concurrency bounded only by the semaphore (Task 7's value).

- [ ] **Step 1: Rewrite the router test first (TDD — the new contract)**

Replace `test_vllm_offline_requests_do_not_overlap` in `tests/test_runtime_router.py` with:

```python
    async def test_vllm_offline_requests_overlap_up_to_semaphore(self) -> None:
        engine = _StatefulEngine()
        router = RuntimeRouter()
        semaphore = asyncio.Semaphore(4)
        router._resolve_family = lambda _model_id: RuntimeFamily.QWEN_VLLM  # type: ignore[method-assign]
        router._get_shared_engine = lambda _family, _model_id: (  # type: ignore[method-assign]
            engine,
            semaphore,
        )

        requests = [
            OfflineASRRequest(
                model_id="qwen3-asr-test",
                audio_path=f"request-{index}",
            )
            for index in range(8)
        ]
        results = await asyncio.gather(
            *(router.run_offline(request) for request in requests)
        )

        # The router no longer serializes; the semaphore is the only bound.
        self.assertGreater(engine.max_active, 1)
        self.assertLessEqual(engine.max_active, 4)
        self.assertEqual(len(results), 8)
```

Drop the `result.text == request.audio_path` assertion — `_StatefulEngine.current_audio_path` is a deliberate shared-state race detector that the *router* lock used to hide; positional integrity under concurrency is the backend lock's job (Task 3) and the H100 mixing test's job (Task 9), not the router's. Keep `_StatefulEngine` otherwise as-is.

- [ ] **Step 2: Run — must fail** (`engine.max_active` is 1 while the router lock still exists). `python -m pytest tests/test_runtime_router.py -v`

- [ ] **Step 3: Implement** in `app/services/asr/runtime/router.py`:

Delete line 82 (`self._vllm_offline_locks: dict[str, asyncio.Lock] = {}`) and collapse `run_offline` (:196–202) to:

```python
    async def run_offline(self, request: OfflineASRRequest) -> ASRFullResult:
        model_id = self.resolve_model_id(request.model_id)
        return await self._run_offline(request, model_id)
```

Confirm nothing else references `_vllm_offline_locks`: `grep -rn "_vllm_offline_locks" app/ tests/` → only the deleted lines.

- [ ] **Step 4: Run — must pass.** Then full suite: `python -m pytest tests/ -v`.

- [ ] **Step 5: Sanity re-read** — with the router lock gone, re-check the A checklist against the diff: backend `_llm_lock` covers `_run_generate` and `_decode_stream` incl. tokenizer (Task 3), aligner encode + init locks (Task 3), VAD lock wired (Task 4), ITN guarded (Task 5), semaphore/executor resolved (Task 7), re-grep clean (Task 8). All six present or A is not landable.

---

### Task 7: Resolve the executor-pool / semaphore interaction [LOCAL]

Implements Spec "The changes → A", scope item **4**, and the Goals item on `_VLLM_SHARED_CONCURRENCY`. Key constraint (Spec r6): Qwen websocket permits are **connection-scoped** (`qwen3_websocket_asr.py:85` acquire → `:388` release at teardown), so 8 idle websocket connections would permanently exhaust the semaphore and block all offline traffic; and after A each admitted offline request holds an executor thread for its whole duration.

**Decision taken by this plan** (the spec's A item 4 sanctions exactly two resolutions: "rescope websocket permits to **per-decode** or size against `offline_requests + concurrent_ws_sessions`"; this plan takes the **per-decode** option — the spec explicitly rejects capping websocket sessions as product-visible):

1. **Websocket *connection* leases stop consuming semaphore permits.** Add `RuntimeRouter.lease_shared_engine(model_id)` that returns a `RuntimeEngineLease` with a no-op release and **no** permit; `qwen3_websocket_asr.py:85` switches to it. An idle connection holds no executor thread and should hold no permit. The FunASR websocket (`websocket_asr.py:179`, `"paraformer-large"`) resolves to the pool path and never touched this semaphore — unchanged.
2. **Websocket *decodes* acquire a permit from a NEW, SEPARATE per-decode semaphore.** Active decodes are NOT free: each streaming decode is an `await run_sync(engine.streaming_transcribe / finish_streaming_transcribe, ...)` (`qwen3_websocket_asr.py:124,130,334,401,407`) that occupies an executor thread, mostly blocked on Task 3's `_llm_lock`. Without a bound, N concurrent streaming sessions occupy N executor threads and can exhaust the pool. So: add `asyncio.Semaphore(settings.VLLM_WS_DECODE_CONCURRENCY)` (config-driven, default **4**, alongside `VLLM_OFFLINE_CONCURRENCY`), held only for the duration of each decode dispatch — acquired before `run_sync`, released when it returns. Implement as one module-level helper in `qwen3_websocket_asr.py` (`_run_decode(fn, *args)` = `async with _ws_decode_semaphore(): return await run_sync(fn, *args)`, semaphore created lazily at first use) and route all five decode `run_sync` call sites through it.
   **Why a separate semaphore, not a shared one:** offline permits are held for a whole request (minutes of audio); ws-decode permits are held per chunk-decode (sub-second). Sharing one semaphore would let 4 long offline requests starve every websocket decode for minutes — and a burst of decodes would conversely block offline admission. Two independent bounds, two knobs.
3. **The old shared semaphore becomes offline-request-scoped admission**, value from config: `VLLM_OFFLINE_CONCURRENCY`, default **4** (was hardcoded 8). Rationale: each admitted offline request holds one executor thread for its full duration after A.
4. **Executor floor derived from BOTH enforced bounds** — not from an assumed headroom constant. Default workers become `max(4, cpu_count, VLLM_OFFLINE_CONCURRENCY + VLLM_WS_DECODE_CONCURRENCY)`. Every term of the formula is now backed by a mechanism: at most `VLLM_OFFLINE_CONCURRENCY` executor threads are occupied by offline requests (item 3's semaphore) and at most `VLLM_WS_DECODE_CONCURRENCY` by websocket decodes (item 2's semaphore), so under defaults (4+4=8) the pool can always run every admitted **vLLM** job — neither vLLM class can starve the other.
   **Scope limit — state it, do not overclaim:** the FunASR/paraformer websocket path dispatches its own decodes via `run_sync` (`app/services/websocket_asr.py:642,681,726`) under **no** admission bound, so N realtime paraformer sessions can occupy executor threads outside both semaphores. The floor guarantee therefore holds only across the two vLLM classes. This is pre-existing behavior and outside this spec's scope (the spec treats the FunASR ws path as separate), and paraformer decodes are short per-chunk rather than blocking — but on a small box a burst of paraformer sessions can still delay vLLM ws decodes. Do not claim the pool is starvation-proof in general. Extract the formula as a pure function `compute_default_workers(cpu_count, offline_concurrency, ws_decode_concurrency)` in `executor.py` so it is unit-testable independent of the box (see Step 1). Still overridable via `INFERENCE_THREAD_POOL_SIZE`. (`executor.py` must not import `app.core.config` — keep it env-only, matching its current style: read both knobs via `os.getenv` at module load for the default calculation.)

**Files:**
- Modify: `app/services/asr/runtime/router.py:21,146,177-194`
- Modify: `app/core/config.py` (add `VLLM_OFFLINE_CONCURRENCY: int = 4` and `VLLM_WS_DECODE_CONCURRENCY: int = 4` + env overrides, following the `QWEN_RUST_CPU_WORKERS` pattern at :71,:128-130)
- Modify: `app/core/executor.py:34-35`
- Modify: `app/services/qwen3_websocket_asr.py` (`:85` lease call; the five decode `run_sync` sites at `:124,:130,:334,:401,:407` routed through `_run_decode`)
- Test: `tests/test_shared_engine_admission.py`

**Interfaces:**
- Produces: `RuntimeRouter.lease_shared_engine(self, model_id: Optional[str] = None) -> RuntimeEngineLease` (no permit); offline semaphore value `settings.VLLM_OFFLINE_CONCURRENCY`; ws-decode semaphore `settings.VLLM_WS_DECODE_CONCURRENCY` wrapped by `qwen3_websocket_asr._run_decode`; executor pure function `compute_default_workers(cpu_count, offline_concurrency, ws_decode_concurrency) -> int` returning `max(4, cpu_count, offline_concurrency + ws_decode_concurrency)`, with `_DEFAULT_WORKERS = compute_default_workers(os.cpu_count() or 4, int(os.getenv("VLLM_OFFLINE_CONCURRENCY", "4")), int(os.getenv("VLLM_WS_DECODE_CONCURRENCY", "4")))`.
- Consumed by: Task 6's test (semaphore bound), Task 9's H100 scripts.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_shared_engine_admission.py
from __future__ import annotations

import asyncio
import unittest

from app.services.asr.runtime.router import (
    RuntimeFamily,
    RuntimeRouter,
    _VLLM_SHARED_CONCURRENCY,  # noqa: F401 — removed in this task; see step 3
)


class SharedEngineAdmissionTest(unittest.IsolatedAsyncioTestCase):
    def _router_with_fake_shared_engine(self):
        router = RuntimeRouter()
        engine = object()
        semaphore = asyncio.Semaphore(2)
        router._resolve_family = lambda _m: RuntimeFamily.QWEN_VLLM  # type: ignore[method-assign]
        router._get_shared_engine = lambda _f, _m: (engine, semaphore)  # type: ignore[method-assign]
        return router, engine, semaphore

    async def test_websocket_lease_consumes_no_permit(self) -> None:
        router, engine, semaphore = self._router_with_fake_shared_engine()
        lease = await router.lease_shared_engine("qwen3-asr-test")
        self.assertIs(lease.engine, engine)
        self.assertEqual(semaphore._value, 2)  # untouched
        await lease.close()
        self.assertEqual(semaphore._value, 2)  # close is a no-op on permits

    async def test_offline_lease_still_consumes_permit(self) -> None:
        router, _engine, semaphore = self._router_with_fake_shared_engine()
        lease = await router.acquire_engine("qwen3-asr-test")
        self.assertEqual(semaphore._value, 1)
        await lease.close()
        self.assertEqual(semaphore._value, 2)

    def test_semaphore_values_come_from_settings(self) -> None:
        # Assert the class defaults and the env-override mechanism, NOT the
        # live singleton's values — a runner with VLLM_*_CONCURRENCY exported
        # would otherwise fail this spuriously.
        from app.core.config import Settings

        self.assertEqual(Settings.VLLM_OFFLINE_CONCURRENCY, 4)
        self.assertEqual(Settings.VLLM_WS_DECODE_CONCURRENCY, 4)

        with mock.patch.dict(
            os.environ,
            {"VLLM_OFFLINE_CONCURRENCY": "7", "VLLM_WS_DECODE_CONCURRENCY": "9"},
        ):
            s = Settings()
            self.assertEqual(s.VLLM_OFFLINE_CONCURRENCY, 7)
            self.assertEqual(s.VLLM_WS_DECODE_CONCURRENCY, 9)

    async def test_ws_decode_dispatch_bounded_by_decode_semaphore(self) -> None:
        # Item 2's mechanism: at most VLLM_WS_DECODE_CONCURRENCY decode
        # dispatches in flight, regardless of how many sessions stream.
        import app.services.qwen3_websocket_asr as ws

        mu = threading.Lock()
        state = {"active": 0, "max_active": 0}

        def fake_decode() -> None:
            with mu:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.02)
            with mu:
                state["active"] -= 1

        await asyncio.gather(*(ws._run_decode(fake_decode) for _ in range(12)))
        # Bound assertions, not an exact peak: a slow/loaded executor may never
        # reach 4 concurrently. The bound is what matters; >1 proves the calls
        # do overlap, so an accidentally-serializing impl still fails.
        self.assertLessEqual(state["max_active"], 4)  # VLLM_WS_DECODE_CONCURRENCY default
        self.assertGreater(state["max_active"], 1)

    def test_executor_default_workers_formula(self) -> None:
        # Tests the FORMULA via the pure function, not the box's cpu_count —
        # `_MAX_WORKERS >= 8` would already pass on any >=8-thread machine
        # and would never exercise the sum term. Fails first with
        # ImportError: compute_default_workers does not exist yet.
        from app.core.executor import compute_default_workers

        # sum term dominates on a small box: 4 offline + 4 ws decode
        self.assertEqual(compute_default_workers(2, 4, 4), 8)
        # cpu_count dominates on a big box
        self.assertEqual(compute_default_workers(64, 4, 4), 64)
        # raised knobs raise the floor with them
        self.assertEqual(compute_default_workers(2, 16, 8), 24)
        # absolute floor of 4
        self.assertEqual(compute_default_workers(1, 1, 1), 4)
```

(Add `import threading`, `import time` to the test module's imports for the ws-decode test.)

Remove the `_VLLM_SHARED_CONCURRENCY` import line once step 3 deletes the constant — the first failing run uses it to prove the module still has the dead knob.

- [ ] **Step 2: Run — must fail** (`lease_shared_engine` does not exist; no `VLLM_OFFLINE_CONCURRENCY`/`VLLM_WS_DECODE_CONCURRENCY` settings; `_run_decode` does not exist; `compute_default_workers` does not exist — ImportError).

- [ ] **Step 3: Implement**

`app/core/config.py` — next to `QWEN_RUST_CPU_WORKERS` (:71):

```python
    VLLM_OFFLINE_CONCURRENCY: int = 4
    VLLM_WS_DECODE_CONCURRENCY: int = 4
```

and in the env-override block (pattern at :128–133):

```python
        self.VLLM_OFFLINE_CONCURRENCY = int(
            os.getenv("VLLM_OFFLINE_CONCURRENCY", str(self.VLLM_OFFLINE_CONCURRENCY))
        )
        self.VLLM_WS_DECODE_CONCURRENCY = int(
            os.getenv("VLLM_WS_DECODE_CONCURRENCY", str(self.VLLM_WS_DECODE_CONCURRENCY))
        )
```

(Not added to `.env.example` — it is a curated minimal file; Spec "Landed already".)

`app/services/asr/runtime/router.py` — delete `_VLLM_SHARED_CONCURRENCY = 8` (:21); at :146 use `asyncio.Semaphore(settings.VLLM_OFFLINE_CONCURRENCY)`; add below `acquire_engine`:

```python
    async def lease_shared_engine(
        self, model_id: Optional[str] = None
    ) -> RuntimeEngineLease:
        """Connection-lifetime lease on the shared vLLM engine, WITHOUT an
        admission permit. Websocket sessions hold their lease for the whole
        connection (including silence); charging them a permit would let 8
        idle connections block all offline traffic. Serialization of actual
        inference is the backend's per-engine locks, not the semaphore.
        """
        resolved_model_id = self.resolve_model_id(model_id)
        family = self._resolve_family(resolved_model_id)
        if family != RuntimeFamily.QWEN_VLLM:
            return await self.acquire_engine(model_id)
        engine, _semaphore = self._get_shared_engine(family, resolved_model_id)
        return RuntimeEngineLease(engine=engine, release_callback=lambda: None)
```

`app/services/qwen3_websocket_asr.py` — at `:85`:

```python
        ctx.engine_lease = await runtime_router.lease_shared_engine(model)
```

and add the per-decode bound (module level, near the other imports):

```python
_ws_decode_semaphore: Optional[asyncio.Semaphore] = None


def _get_ws_decode_semaphore() -> asyncio.Semaphore:
    """Per-DECODE admission (spec A item 4, per-decode option). Bounds how
    many websocket decode dispatches occupy executor threads at once.
    Connection leases are free (idle sessions hold no thread); each decode
    is not. Lazy init: created on the event loop at first decode; all
    acquires happen on the single event-loop thread, so no init race."""
    global _ws_decode_semaphore
    if _ws_decode_semaphore is None:
        from app.core.config import settings
        _ws_decode_semaphore = asyncio.Semaphore(settings.VLLM_WS_DECODE_CONCURRENCY)
    return _ws_decode_semaphore


async def _run_decode(fn, *args):
    """All streaming-decode executor dispatches go through here."""
    async with _get_ws_decode_semaphore():
        return await run_sync(fn, *args)
```

then route the five decode dispatches through it — at `:124,:130,:334,:401,:407`, change `await run_sync(engine.streaming_transcribe, ...)` / `await run_sync(engine.finish_streaming_transcribe, ...)` to `await _run_decode(engine.streaming_transcribe, ...)` / `await _run_decode(engine.finish_streaming_transcribe, ...)` (same arguments). Confirm no decode `run_sync` remains: `grep -n "run_sync" app/services/qwen3_websocket_asr.py` → only inside `_run_decode`.

`app/core/executor.py:34` — extract the formula as a testable pure function:

```python
def compute_default_workers(
    cpu_count: int, offline_concurrency: int, ws_decode_concurrency: int
) -> int:
    """Executor floor derived from BOTH admission bounds (spec A item 4):
    at most `offline_concurrency` threads held by offline requests and
    `ws_decode_concurrency` by websocket decodes — each term enforced by
    its own semaphore — so the pool can always run every admitted job."""
    return max(4, cpu_count, offline_concurrency + ws_decode_concurrency)


_DEFAULT_WORKERS = compute_default_workers(
    os.cpu_count() or 4,
    int(os.getenv("VLLM_OFFLINE_CONCURRENCY", "4")),
    int(os.getenv("VLLM_WS_DECODE_CONCURRENCY", "4")),
)
```

(Env reads, not `app.core.config` imports — `executor.py` stays config-free. The env values here must default to the same `4`s as `config.py` so the formula and the semaphores agree when the env is unset.)

- [ ] **Step 4: Run — must pass** (after removing the dead `_VLLM_SHARED_CONCURRENCY` import from the test). Then full suite. Grep for stale references: `grep -rn "_VLLM_SHARED_CONCURRENCY" app/ tests/` → none.

---

### Task 8: Re-grep audit for module-level lazy singletons [LOCAL] — before Task 6

Implements the Spec's Confidence note ("The implementer should re-grep for module-level mutable state on the `transcribe_long_audio` path before landing A" — the audit missed a singleton twice).

**Files:** none modified unless a finding appears.

- [ ] **Step 1: Run the pattern sweeps**

```bash
cd /home/chandraliuswanto/Desktop/qwen3-asr
# module-level lazy singletons: `_x = None` + `if _x is None:` initializer
grep -rn --include='*.py' -E '^_[a-zA-Z0-9_]+(: [^=]+)? = None' app/ | sort
grep -rn --include='*.py' -E 'if _[a-zA-Z0-9_]+ is None' app/ | sort
# module-level mutable containers and `global` writers
grep -rn --include='*.py' -E '^[a-zA-Z_]+ = (\{\}|\[\])' app/
grep -rn --include='*.py' -E '^\s+global _' app/
```

- [ ] **Step 2: For each hit, decide reachable-from-`transcribe_long_audio`-or-websocket-decode and guarded-or-not.** Known-good (already guarded or being guarded by this plan): `global_models.py` VAD/PUNC (locked), `text_processing.py` ITN (Task 5), `router.py` `_runtime_router` (locked, :226), `executor.py` `_executor` (event-loop-thread only — `get_executor` is called from `run_sync` on the loop thread, single-threaded; confirm no executor-thread caller before waving it through). Trace anything else to its callers before dismissing it (`Grep` for the accessor name).

- [ ] **Step 3: Outcome.** If a new unguarded singleton on the concurrent path is found: guard it with the Task 5 pattern (module `threading.Lock`, init+call), add a Task-5-style test, and record it in the plan's checklist here. If none: state "re-grep clean, N hits reviewed" in the worker report. Task 6 may not start until this task reports.

---

### Task 9: H100 integration test scripts [LOCAL authoring, H100-ONLY execution]

Implements Spec "Success criteria". These CANNOT run in CI or locally — vLLM is absent and a mocked engine cannot reproduce a scheduler race; a green mocked test would be worthless. Scripts live under `scripts/h100/` and are run manually on the H100 after deploy.

**Files:**
- Create: `scripts/h100/test_offline_mixing.py`
- Create: `scripts/h100/test_ws_offline_mixed.py`

**Interfaces:**
- Consumes: a running service URL (env `ASR_BASE_URL`, default `http://localhost:8000`), a directory of distinct-content test audio files (env `ASR_TEST_AUDIO_DIR`) where each file's expected transcript keyword is encoded in its filename (e.g. `alpha_银行.wav` must transcribe to text containing `银行`).

- [ ] **Step 1: Write `scripts/h100/test_offline_mixing.py`** — the mixing regression test. N≥20 iterations (Spec: "inherently racy — run N≥20, treat as evidence, not proof"); each iteration fires 8 concurrent `POST /v1/audio/transcriptions` (multipart upload, defaults) with distinct audio files and asserts each response's text contains its own file's keyword and none of the other files' keywords. Print a per-iteration PASS/FAIL and a final summary; exit nonzero on any cross-contamination. Include a `--check-detects-bug` note in the docstring: **the run proves nothing unless it FAILS against a build with the Task 3 `_llm_lock` removed** (Spec success criteria) — the runbook is: (1) deploy a build with `with self._llm_lock:` commented out in `_run_generate`, expect FAIL; (2) deploy the real build, expect 20/20 PASS.

- [ ] **Step 2: Write `scripts/h100/test_ws_offline_mixed.py`** — the mixed websocket + offline shape (the ONLY shape that can catch the ITN race — websocket-only never fails, Spec item 3b): drive 4 concurrent websocket sessions (protocol per `app/services/qwen3_websocket_asr.py` — read `_handle` / the API route in `app/api/v1/` for the message framing) streaming distinct audio with `enable_inverse_text_normalization=true`, while 4 offline requests with `enable_itn=true` run concurrently; assert per-channel keyword integrity as in step 1, and record websocket p50/p95 decode latency (time from audio-chunk send to matching partial result) to compare against the Task 2b baseline threshold.

- [ ] **Step 3 [H100-ONLY]: Execution runbook** — after A is deployed: run both scripts (each N≥20 where applicable); run the semaphore-behavior check implicitly via 10 concurrent 5-minute offline requests + `ASR_STAGE_TIMINGS` overlap in logs (compare wall-clock vs `main` per Success criteria — improvement expected, no target multiple); confirm websocket p50/p95 within the threshold agreed at task 2b. Record all numbers.

---

### Self-review notes (performed per the writing-plans skill)

- **Spec coverage (in-scope items):** task 1 → Task 1; task 2 → Tasks 2/2b; A item 1 (main lock incl. tokenizer at `:450`) → Task 3; A item 2 (aligner init-only lock, three callers, no nesting) → Task 3; A item 3 (VAD lock) → Task 4; A item 3b (ITN) → Task 5; A item 4 (executor/semaphore, per-decode websocket rescope, `_VLLM_SHARED_CONCURRENCY` dead-knob resolution) → Task 7; router-lock deletion + `test_runtime_router.py:59` → Task 6; re-grep instruction → Task 8; success-criteria H100 tests (mixing, mixed ws+offline, latency threshold, must-fail-without-guard) → Task 9. B and C: deliberately out of scope, stated up front.
- **Decision this plan made where the spec offered options (A item 4):** the **per-decode** rescope — no-permit connection leases + a separate per-decode websocket semaphore (`VLLM_WS_DECODE_CONCURRENCY`, default 4) + config-driven offline admission (`VLLM_OFFLINE_CONCURRENCY`, default 4) + an executor floor of `max(4, cpu_count, offline + ws_decode)` in which every term is enforced by one of those two semaphores — rather than sizing permits against connection counts. Grounds: the spec itself calls the admission-limit route "a product-visible regression, not a tuning knob". Two separate semaphores (not one shared) because offline permits are minutes-long and ws-decode permits sub-second; sharing would let either class starve the other.
- **Types/names consistent:** `_llm_lock`/`_aligner_lock`/`_aligner_init_lock` (Tasks 3, 6, 9), `lease_shared_engine`/`_run_decode` (Tasks 7, 9 consumers), `VLLM_OFFLINE_CONCURRENCY`/`VLLM_WS_DECODE_CONCURRENCY` (Tasks 6 semaphore bound via settings, 7), `compute_default_workers` (Task 7), `format_stage_timings` (Task 2).
