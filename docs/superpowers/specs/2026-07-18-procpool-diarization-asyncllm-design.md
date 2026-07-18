# Process-Pool Diarization + AsyncLLM Migration — Design

Date: 2026-07-18
Status: draft — awaiting review
Scope: two changes in one spec, implemented as two sequential phases:
- **Change D** — diarization moves from an in-process thread pool to spawn-based worker processes. `DIARIZATION_POOL_SIZE` is reused as the worker-process count.
- **Change C** — the vLLM offline `LLM` API + `_llm_lock` is replaced by vLLM v1 `AsyncLLM` (continuous batching, per-request streams), for both the ASR engine and the forced aligner.

Driving requirement: the API must serve **≥16 concurrent users**. Neither change alone gets there; together they remove both measured serialization walls.

## Problem — measured, not estimated

Two independent walls, both measured on the production H100 with 2-minute audio:

**Wall 1 — diarization is GIL-bound; thread pools cannot scale it.**
Benchmarked 2026-07-18 with the thread pool live (`DIARIZATION_POOL_SIZE=16`, `VLLM_OFFLINE_CONCURRENCY=16`, n=10): per-stage timings (`DIARIZATION_STAGE_TIMINGS`) showed `preprocess` at **15.3–17.7s per call, ten calls completing within ~2.6s of each other** — the signature of N threads round-robining on the GIL (≈ 10 × the single-request time, all finishing together). The whole pipeline takes ~0.9s uncontended. `preprocess` is FSMN-VAD: a per-frame Python state machine (~12k GIL-holding iterations per 2-minute file). `clustering`/`postprocess` are CPU numpy/scipy; only the batched SV `forward` meaningfully uses the GPU (sub-second). Net: pool instances multiply, the interpreter doesn't. Effective thread-parallel ceiling ≈ 2–4; n=10 wall was 35.5s vs the 12.8s serialized baseline. Aggravator: funasr sets **process-global** `torch.set_num_threads(4)` (`funasr/auto/auto_model.py:209-210`), shared by all instances in one process.

**Wall 2 — vLLM generation is serialized by `_llm_lock`.**
The 2026-07-17 bench measured a **1.47 req/s ceiling with `--no-diarization`** (~52% of the request still serialized). `_llm_lock` exists because vLLM 0.19.0's offline `LLM.generate`/`encode` drain a shared engine with no per-caller request-id filtering (verified in source; see bd memory `vllm-generate-not-threadsafe`) — concurrent callers collect each other's outputs and `zip` silently mispairs text onto the wrong segments. The lock is correct for the offline API; the offline API is the wrong API for an online service.

Ordering note: today the diarization bottleneck staggers arrivals and masks Wall 2. Fixing D unmasks C; that is why both are in scope and D lands first.

## Verified facts this design relies on

- **vLLM 0.19.0 (pinned; `pyproject.toml:33`) already ships v1 `AsyncLLM` with both APIs**: `async def generate(..., request_id)` at `vllm/v1/engine/async_llm.py:529` and `async def encode(..., pooling_params, request_id)` at `:777` (verified against tag `v0.19.0`, previously shown byte-identical to the PyPI wheel). Outputs are delivered on **per-request streams**, so the shared-drain mixing defect of the offline API does not exist in this API by construction. **No vLLM version bump is required.**
- The forced aligner is a second, separate vLLM engine with its own `gpu_memory_utilization` slice (`qwen3_vllm.py:342`), used via `encode` (`:515`). It migrates to its own `AsyncLLM` instance; generate-mode and pooling-mode engines stay separate.
- `_build_chat_prompt` renders with `tokenize=False` (pure Jinja) and already runs outside `_llm_lock` by documented design — it needs no new protection.
- The direct `self._tokenizer.encode(...)` call in `_decode_stream` (`qwen3_vllm.py:576`) is a real concurrency hazard (`Qwen2TokenizerFast` raises "Already borrowed" under concurrent encode) that `_llm_lock` currently covers; removing the lock requires a replacement guard.
- CAM++ per-instance serialization remains mandatory *within* a process (funasr mutates shared per-instance state per call — see 2026-07-17 diarization-throughput spec). Worker processes satisfy it by construction: one pipeline per process, one task per worker at a time.

**This spec carries no throughput projections.** Every performance claim above is a measurement; every expected benefit is gated on the measurements in "Measurement gates". (Project rule; see bd memory `perf-estimates-from-code-are-unreliable`.)

## Change D — diarization worker processes

### Architecture

A `concurrent.futures.ProcessPoolExecutor` with `mp_context=multiprocessing.get_context("spawn")`, `max_workers=settings.DIARIZATION_POOL_SIZE`, and an `initializer` that builds exactly one CAM++ pipeline per worker.

- **Knob reuse (per request):** `DIARIZATION_POOL_SIZE` now means *number of diarization worker processes*. Default stays 4. `.env.example` and `config.py` comments updated to say "worker processes"; the `VLLM_OFFLINE_CONCURRENCY <= DIARIZATION_POOL_SIZE` boot warning stays (an over-admitted request now queues in the process pool instead of blocking a pool checkout — same backpressure, gentler failure mode).
- **New module `app/utils/diarization_worker.py`** — the only module a spawned worker initializes. Contents:
  - `_worker_init()`: (1) `import modelscope.pipelines.audio` **first** — this forces modelscope's task registration and kills the "Unknown task speaker-diarization" failure at its root (the bug behind `qwen3-asr-9nk`); (2) build one pipeline via the existing `_build_diarization_pipeline()` (batched SV patch, stage-timing patch, empty-cache guard all apply inside the worker; spawn inherits `os.environ`, so `DIARIZATION_STAGE_TIMINGS` and batch-size settings work unchanged); (3) store it in a module global.
  - `_worker_diarize(audio_path: str) -> list[tuple[float, float, int]]`: run `pipeline(audio_path)` and return the raw `(start_sec, end_sec, speaker_label)` triples. Nothing else crosses the process boundary — a path in, small picklable triples out.
  - Import discipline: this module must not import the app's engine/model stack (no vLLM, no funasr ASR engines). It may import `app.core.config` (dotenv + settings) and the pipeline-builder pieces of `speaker_diarizer`.
- **Parent side:** `SpeakerDiarizer.diarize()` submits to the pool and blocks on `.result()` in its executor thread (same threading model and admission math as today). Segment-object construction, speaker-id formatting (`说话人{n}`), merging, low-energy splitting, and temp-file extraction all stay in the parent. The `ThreadedEnginePool` checkout in `diarize()` is deleted; `_suppress_empty_cache` moves into the worker.
- **`torch.cuda.empty_cache` guard** becomes low-stakes: funasr's per-call `empty_cache` now syncs only the worker's own CUDA context, not vLLM's. The guard still installs in workers (latency), and the parent-side installation is removed with the parent-side pipeline.

### Failure and lifecycle

- **Boot:** `model_loader` step 5 creates the pool and forces all workers up (submit `DIARIZATION_POOL_SIZE` barrier no-op tasks, wait). **A warmup failure fails the boot loudly** — this replaces today's swallow-and-degrade behavior (`model_loader.py:547-549`) that produced the silent lazy-build stall. Closes the `qwen3-asr-9nk` failure mode for this path.
- **`BrokenProcessPool`** (worker OOM/crash mid-flight): one rebuild attempt under a parent-side lock; concurrent callers wait for the rebuild rather than each triggering one. If the rebuild fails, raise `DefaultServerErrorException` and log CRITICAL. The in-flight request that hit the break fails; it is not silently retried (retry policy belongs to callers).
- **CPU budget:** funasr's `torch.set_num_threads(4)` is now per *worker* — total CPU threads ≈ K × (1 Python + 4 BLAS). Worker count and any `ncpu` tuning are decided by measurement gate G1, not guessed here.

### Retired

- The diarization use of `ThreadedEnginePool` and its per-instance-mutex documentation (the class itself stays only if other callers exist; today diarization is its only user, so it is deleted with its tests updated).
- The "checkout is the mutex" invariant text moves to: "the worker process is the instance; exclusivity is structural."

## Change C — AsyncLLM migration

### Architecture

- **Two `AsyncLLM` instances** replace the two offline `LLM` instances: the ASR generate engine and the forced-aligner pooling engine, constructed via `AsyncLLM.from_engine_args` with the same model paths, chat-template loading, and `gpu_memory_utilization` split as today. Aligner init keeps the existing init-only lock and its documented no-self-deadlock constraint.
- **Sync-async bridge:** `Qwen3VLLMBackend` owns one dedicated asyncio event loop running in a daemon thread ("engine loop"), started at engine init. All existing call sites stay synchronous: `_run_generate`, `_decode_stream`, and `align_transcript` submit coroutines with `asyncio.run_coroutine_threadsafe(...)` and block on `.result()` in their executor thread. This keeps the blast radius inside `qwen3_vllm.py` — `engines/base.py`, the router, admission, and executor sizing are untouched.
- **`_llm_lock` is deleted for engine calls.** Each call generates a unique `request_id` (uuid) and consumes its own per-request output stream; cross-request mixing is impossible at the API level. A defensive `if len(outputs) != len(inputs): raise RuntimeError` stays in the batch path (folds in the `qwen3asr-followup-zip-assert` follow-up, done properly with a test this time).
- **Batch fan-out:** `_run_generate` currently submits a list to `LLM.generate`. Under AsyncLLM it fans out one `generate` coroutine per item and `asyncio.gather`s them, preserving input order. Continuous batching in the engine replaces client-side batching; `ASR_BATCH_SIZE` semantics upstream are unchanged (it still bounds how many segments one request submits at once).
- **Residual tokenizer guard:** a new narrow `_tokenizer_lock` protects only the direct `self._tokenizer.encode(...)` in `_decode_stream` (the "Already borrowed" hazard). `_build_chat_prompt` remains lock-free (pure Jinja, unchanged).
- **Streaming path:** `_decode_stream` performs the same per-chunk full-prompt generate as today, through the bridge. No protocol change for websocket clients.
- **Admission knobs keep their meaning** (`VLLM_OFFLINE_CONCURRENCY`, `VLLM_WS_DECODE_CONCURRENCY` are ceilings protecting memory and fairness, not engine-safety locks anymore). Executor sizing formula is unchanged.
- **Rollback:** `VLLM_USE_ASYNC_ENGINE` (default `true` once gate G2 passes). `false` keeps the legacy offline-`LLM` + `_llm_lock` path, which is retained intact for one release and then removed via a filed follow-up issue. Boot log states which engine mode is active. This mirrors the project's `VLLM_OFFLINE_CONCURRENCY=1` rollback pattern.

## Rejected alternatives

- **Bigger thread pool / tuning `DIARIZATION_POOL_SIZE` upward in-process** — measured dead end; the GIL is the wall (2026-07-18 stage timings).
- **`WORKERS>1` uvicorn replicas as *the* fix** — works, but duplicates the vLLM engine's VRAM per worker and multiplies every warm-up cost; remains available as an orthogonal ops lever, not part of this design.
- **Upgrading vLLM to 0.25.x for change C** — unnecessary; 0.19.0 has both AsyncLLM APIs (verified above). An upgrade would churn the CUDA lockfile that cannot be validated off-H100.
- **One combined AsyncLLM for generate + aligner encode** — generate-mode and pooling-mode engines are separate runners; today's two-engine split is kept.
- **Dropping CAM++ per-instance serialization** — still unsafe inside a process (funasr shared-state mutation); worker processes make it structural instead.
- **Making the whole call chain async instead of a bridge loop** — touches every engine, router, and API layer for no measured benefit; the bridge confines change C to one file.

## Invariants (unchanged unless stated)

1. An engine or pipeline instance is never visible to two concurrent requests. (Workers: structural. AsyncLLM: engine-internal batching is the engine's contract; per-request streams are ours.)
2. No request's transcript may pair with another request's segments — the adapted cross-request fidelity test is a MANDATORY merge gate on the H100.
3. `/health` never blocks behind generation or diarization (existing lease path untouched).
4. Boot fails loudly on invalid knobs and now also on diarization-pool warmup failure.
5. Tests are unittest, run green on the CPU dev box after every task; engine construction is faked off-H100 exactly as the current suite does.

## Measurement gates (blocking; all on the H100; no projections)

- **G0 — clean baseline:** current main, `DIARIZATION_POOL_SIZE=4`, `VLLM_OFFLINE_CONCURRENCY=4`, restarted server, `scripts/h100/bench.sh` levels [1,2,4,8,10,16]. Records the honest pre-change numbers (the 2026-07-18 16/16 run is not a valid baseline).
- **G1 — change D verdict:** same bench + `DIARIZATION_STAGE_TIMINGS`. Pass criteria: (a) per-call `preprocess` at n=10 is in the neighborhood of its n=1 time rather than ~n× it (GIL-escape proof); (b) zero failures incl. the mixing test; (c) per-worker VRAM measured (`nvidia-smi` deltas across worker spawn) — **production `DIARIZATION_POOL_SIZE` is chosen from this number, here, not assumed**. This is the revived Task 7.
- **G2 — change C verdict:** bench with diarization on and `--no-diarization`, async engine on vs `VLLM_USE_ASYNC_ENGINE=false`. The `--no-diarization` ceiling (1.47 req/s on the old engine) is the signature number to watch. Mixing/fidelity test green at n=10. Aligner (`word_timestamps`) e2e exercised.
- **G3 — 16-concurrent load test:** bench level [16] with both changes live, p50/p95/fails recorded. This gate's pass condition is *measurement + zero failures*; the latency SLO judgment belongs to the service owner reviewing G3's numbers — this spec deliberately does not invent an SLO.

## Testing (local, per task)

- Worker module: initializer injection point so tests fake the pipeline build (no modelscope on the test path); tests for triple marshalling, "too short" fallback (stays parent-side), and rebuild-once-then-raise on `BrokenProcessPool`.
- Bridge: submit/timeout/exception propagation across the loop thread; engine-loop shutdown on backend close.
- Batch fan-out: order preservation and the length assert (test proves it raises on a dropped output).
- Rollback flag: both modes construct against fakes; lock-test suite reworked — serialization asserts move from `_llm_lock` to the tokenizer lock and to per-request-id uniqueness.
- Suite must stay green on the CPU box (`DEVICE=cpu`, unittest discovery) after every task.

## Rollout

1. **Phase 1 (change D)** on its own branch: merge gated on G0+G1.
2. **Phase 2 (change C)** on its own branch behind `VLLM_USE_ASYNC_ENGINE`: merge gated on G2; G3 runs with both live.
3. bd: file one issue per phase plus the legacy-path-removal follow-up; `qwen3-asr-2vs` gets its closing verdict (thread-pool approach measured GIL-bound, superseded by this spec) with the 2026-07-18 stage-timing evidence attached.
