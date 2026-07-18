# Process-Pool Diarization + AsyncLLM Migration — Design

Date: 2026-07-18
Status: draft — adversarial review (fresh subagent) verdict WORKS WITH FIXES; all 12 findings folded in below
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
  - `_worker_init()`: (1) configure worker logging first (loguru/std sinks are NOT inherited from the parent — without this, `[diarization-profile]` lines and initializer tracebacks land on raw stderr; G1's method reads these lines, so they must go somewhere defined); (2) `import modelscope.pipelines.audio` — forces modelscope's task registration and kills the "Unknown task speaker-diarization" failure at its root (the bug behind `qwen3-asr-9nk`); (3) build one pipeline (batched SV patch, stage-timing patch, empty-cache guard all apply inside the worker; spawn inherits `os.environ`, so `DIARIZATION_STAGE_TIMINGS` and batch-size settings work unchanged); (4) store it in a module global; (5) rendezvous on the boot barrier (below).
  - `_worker_diarize(audio_path: str) -> list[tuple[float, float, int]]`: run `pipeline(audio_path)` and return raw `(start_sec, end_sec, speaker_label)` triples with **native Python types** (the pipeline emits lists with numpy scalars; the worker converts before pickling). Nothing else crosses the process boundary — a path in, small picklable triples out.
  - **Parent parsing changes with it**: `diarize()`'s current shape check (`isinstance(seg, list) and len(seg) == 3`, `speaker_diarizer.py:431`) is replaced by the worker contract. Getting this wrong is silent (`[]` → per-request VAD fallback with no error), so the marshalling test MUST pickle-round-trip real-shaped output including numpy scalars.
  - Import discipline: this module must not import the app's engine/model stack (no vLLM, no funasr ASR engines, no `app.services.asr.runtime` — note `speaker_diarizer.py:22` currently imports `local_pool`, whose package `__init__` pulls the router and torch stack). The pipeline-builder functions therefore **move into `diarization_worker.py`** (or the `local_pool` import is deleted from `speaker_diarizer` in the same task); `app.core.config` (dotenv + settings) is allowed.
- **Parent side:** `SpeakerDiarizer.diarize()` submits to the pool and blocks on `.result()` in its executor thread (same threading model and admission math as today). Segment-object construction, speaker-id formatting (`说话人{n}`), merging, low-energy splitting, and temp-file extraction all stay in the parent. The `ThreadedEnginePool` checkout in `diarize()` is deleted; `_suppress_empty_cache` moves into the worker.
- **`torch.cuda.empty_cache` guard** becomes low-stakes: funasr's per-call `empty_cache` now syncs only the worker's own CUDA context, not vLLM's. The guard still installs in workers (latency), and the parent-side installation is removed with the parent-side pipeline.

### Failure and lifecycle

- **Boot:** `model_loader` step 5 creates the pool and forces all workers up. Submitting N no-op tasks does NOT guarantee N workers spawn (`ProcessPoolExecutor._adjust_process_count` reuses an idle worker instead of spawning), so the fan-out is made structural: a `multiprocessing.Barrier(N+1)` passed via `initargs` that `_worker_init` and the boot code rendezvous on, with a generous boot timeout so a wedged spawn fails the boot instead of hanging it. **A warmup failure fails the boot loudly** — this replaces today's swallow-and-degrade behavior (`model_loader.py:547-549`) that produced the silent lazy-build stall. Closes the `qwen3-asr-9nk` failure mode for this path. Note: an initializer exception surfaces as `BrokenProcessPool` with no cause text — the real traceback is in the worker's log sink (see `_worker_init` step 1); the boot error message must say so.
- **`BrokenProcessPool`** (worker OOM/crash mid-flight): one rebuild attempt under a parent-side lock — only the first caller triggers it; racing callers skip it and fail their own request, and requests arriving during the rebuild window **fail fast** rather than block behind a potentially minutes-long respawn. If the rebuild fails, raise `DefaultServerErrorException` and log CRITICAL. The in-flight request that hit the break fails; it is not silently retried (retry policy belongs to callers).
- **Boot fails fast on worker death, not by timeout:** a worker that crashes in its initializer never reaches the barrier, so the parent must not sit out the full boot timeout — a watcher on the warmup probe futures (`FIRST_EXCEPTION`) aborts the barrier the moment any worker dies. The timeout remains only for genuinely wedged spawns.
- **Shutdown:** the app lifespan shutdown closes the pool explicitly (next to `shutdown_executor()`); atexit-only teardown can hang on workers still parked at an aborted boot barrier.
- **Rollback story (explicit):** Change D has no engine-swap flag — the thread-pool path is a *measured* dead end, so reverting to it is pointless. The degraded levers are `DIARIZATION_POOL_SIZE=1` (single serialized worker) or per-request `enable_speaker_diarization=false`; full revert is a branch revert, acceptable because the merge is G0+G1-gated.
- Test-only escape hatch `DIARIZATION_WORKER_FAKE=1` (canned pipeline for spawn-integration tests) must be guarded so it cannot silently fake diarization in production (refuse unless `DEVICE=cpu`). `DIARIZATION_WORKER_LOG_LEVEL` (worker stderr sink level, default INFO) is sanctioned alongside `DIARIZATION_BOOT_TIMEOUT_S`; both validated as positive/known values at boot.
- **CPU budget:** funasr's `torch.set_num_threads(4)` is now per *worker* — total CPU threads ≈ K × (1 Python + 4 BLAS). Worker count and any `ncpu` tuning are decided by measurement gate G1, not guessed here.

### Retired

- The diarization use of `ThreadedEnginePool` and its per-instance-mutex documentation (the class itself stays only if other callers exist; today diarization is its only user, so it is deleted with its tests updated).
- The "checkout is the mutex" invariant text moves to: "the worker process is the instance; exclusivity is structural."

## Change C — AsyncLLM migration

### Architecture

- **Two `AsyncLLM` instances** replace the two offline `LLM` instances: the ASR generate engine and the forced-aligner pooling engine, constructed via `AsyncLLM.from_engine_args` with the same model paths, chat-template loading, and `gpu_memory_utilization` split as today. Aligner init keeps the existing init-only lock and its documented no-self-deadlock constraint. The aligner call maps `pooling_task="token_classify"` to `PoolingParams(task="token_classify")` (`AsyncLLM.encode` takes `pooling_params`, not the offline API's kwarg).
- **CONSTRUCTION RULE (load-bearing):** both `AsyncLLM` instances MUST be constructed **on the bridge-loop thread**. `AsyncLLM.__init__` calls `asyncio.get_running_loop()` and eagerly starts its `output_handler` task on whatever loop is running (`async_llm.py:178-184`); today's boot preloads models from *inside the uvicorn lifespan loop* (`main.py` → `preload_models` → `warmup_model`), so naive construction binds the output handler to the uvicorn loop while `generate()` runs on the bridge loop — cross-loop queue corruption/hang on the first request. Every construction site (including the lazy aligner path) submits the constructor to the bridge loop; an assertion guards against construction with a foreign running loop.
- **Sync-async bridge:** `Qwen3VLLMBackend` owns one dedicated asyncio event loop running in a daemon thread ("engine loop"), started at engine init. All existing call sites stay synchronous: `_run_generate`, `_decode_stream`, and `align_transcript` submit coroutines with `asyncio.run_coroutine_threadsafe(...)` and block on `.result(timeout=...)` — the timeout sized generously above worst-case generation, because a silently dead loop thread would otherwise wedge executor threads that hold admission permits forever. A loop-thread supervisor marks the backend dead and fails all pending futures fast (→ `DefaultServerErrorException`) if the loop stops. The backend gains an explicit `close()` lifecycle hook (stops the loop, joins the thread) — it has none today and the tests need it. Bridge re-raises unwrap `EngineGenerateError.__cause__` so API error payloads keep a message (v1 wraps engine errors in a bare exception). This keeps the blast radius inside `qwen3_vllm.py` — `engines/base.py`, the router, admission, and executor sizing are untouched.
- **Known residual wall (named on purpose):** `AsyncLLM.add_request` runs input processing — tokenization AND multimodal audio feature extraction — synchronously on the bridge loop (`async_llm.py:356-367`), which also runs the output handler. One loop thread therefore serializes all per-request preprocessing. This is no worse than today (the same work ran under `_llm_lock`), but it is the **first suspect if G2 underperforms**; mitigations to evaluate only if measured (pre-computed features, pre-tokenized prompts).
- **`_llm_lock` is deleted for engine calls.** Each call generates a unique `request_id` (uuid) and consumes its own per-request output stream; cross-request mixing is impossible at the API level. A defensive `if len(outputs) != len(inputs): raise RuntimeError` stays in the batch path (folds in the `qwen3asr-followup-zip-assert` follow-up, done properly with a test this time).
- **Batch fan-out:** `_run_generate` currently submits a list to `LLM.generate`. Under AsyncLLM it fans out one `generate` coroutine per item and `asyncio.gather`s them, preserving input order. Continuous batching in the engine replaces client-side batching; `ASR_BATCH_SIZE` semantics upstream are unchanged (it still bounds how many segments one request submits at once).
- **Residual tokenizer guard:** a new narrow `_tokenizer_lock` protects the whole prefix-computation block in `_decode_stream` — **both** `self._tokenizer.encode(...)` (`:576`) **and** `self._tokenizer.decode(...)` (`:579`); decode on the same Rust fast-tokenizer object has the identical "Already borrowed" hazard. `_build_chat_prompt` remains lock-free (pure Jinja, unchanged). AsyncLLM tokenizes with its own separately-loaded tokenizer instance — no cross-object hazard.
- **Streaming path:** `_decode_stream` performs the same per-chunk full-prompt generate as today, through the bridge. No protocol change for websocket clients.
- **Admission knobs keep their meaning** (`VLLM_OFFLINE_CONCURRENCY`, `VLLM_WS_DECODE_CONCURRENCY` are ceilings protecting memory and fairness, not engine-safety locks anymore). Executor sizing formula is unchanged.
- **Rollback:** `ASR_USE_ASYNC_VLLM` (default `true` once gate G2 passes; NOT `VLLM_*`-prefixed — vLLM parses that namespace as its own via `vllm.envs`). `false` keeps the legacy offline-`LLM` + `_llm_lock` path, which is retained intact for one release and then removed via a filed follow-up issue. Boot log states which engine mode is active. This mirrors the project's `VLLM_OFFLINE_CONCURRENCY=1` rollback pattern.

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
- **G2 — change C verdict:** bench with diarization on and `--no-diarization`, async engine on vs `ASR_USE_ASYNC_VLLM=false`. The `--no-diarization` ceiling (1.47 req/s on the old engine) is the signature number to watch. Mixing/fidelity test green at n=10. Aligner (`word_timestamps`) e2e exercised. **VRAM measured across AsyncLLM bring-up of both engines** (`nvidia-smi` deltas): v1's `AsyncLLM` runs its engine core in a background *process* per instance, and the aligner's default slice equals the primary's (`_resolve_forced_aligner_gpu_memory_utilization`), so the two-async-engine memory topology must be measured, not assumed.
- **G3 — 16-concurrent load test:** bench level [16] with both changes live, p50/p95/fails recorded. This gate's pass condition is *measurement + zero failures*; the latency SLO judgment belongs to the service owner reviewing G3's numbers — this spec deliberately does not invent an SLO.

## Testing (local, per task)

- Worker module: initializer injection point so tests fake the pipeline build (no modelscope on the test path); tests for triple marshalling, "too short" fallback (stays parent-side), and rebuild-once-then-raise on `BrokenProcessPool`.
- Bridge: submit/timeout/exception propagation across the loop thread; engine-loop shutdown on backend close.
- Batch fan-out: order preservation and the length assert (test proves it raises on a dropped output).
- Rollback flag: both modes construct against fakes; lock-test suite reworked — serialization asserts move from `_llm_lock` to the tokenizer lock and to per-request-id uniqueness.
- Suite must stay green on the CPU box (`DEVICE=cpu`, unittest discovery) after every task.

## Open verifications (Phase 2 Task 0 — before any change-C code)

vLLM is not installed on the dev box; only `async_llm.py` and `llm.py` were source-verified against the `v0.19.0` tag. Before Phase 2 starts, fetch and read (same tag): `engine/arg_utils.py` (does `AsyncEngineArgs` accept the runner/pooling configuration and the engine args we pass today — `gpu_memory_utilization`, `enforce_eager`, `hf_overrides`), `v1/engine/core_client.py` (whether `make_async_mp_client` binds a loop at construction — reinforces the construction rule), and the v1 supported-pooling-task validation for `token_classify` on this model. Any mismatch is a spec amendment, not an implementation improvisation.

## Rollout

1. **Phase 1 (change D)** on its own branch: merge gated on G0+G1. **Migration note (blocking, operator):** existing deployments' `.env` files must reset `DIARIZATION_POOL_SIZE` before deploying Phase 1 — the knob's meaning changes from in-process instances to **worker processes**, and a stale value (the H100 currently carries 16) would spawn 16 model-loading CUDA processes at boot and likely OOM. Start at 4; raise only per G1's measured per-worker VRAM.
2. **Phase 2 (change C)** on its own branch behind `ASR_USE_ASYNC_VLLM`: merge gated on G2; **G3 is a Phase-2 deliverable** (tracked by its own bd issue filed during Phase 1 bookkeeping so it cannot fall between the plans).
3. bd: file one issue per phase plus the legacy-path-removal follow-up and the G3 placeholder; `qwen3-asr-2vs` gets its closing verdict (thread-pool approach measured GIL-bound, superseded by this spec) with the 2026-07-18 stage-timing evidence attached; `qwen3-asr-9nk` is dispositioned by Change D (registration import + loud boot), closed after G1 verifies both loud-failure paths.
