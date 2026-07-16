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
