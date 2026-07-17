# Diarization Throughput — Design

Date: 2026-07-17
Status: awaiting approval
Scope: two changes — (1) stop funasr's per-call `torch.cuda.empty_cache()`, (2) pool the CAM++ pipeline.

## Spec

### Problem — measured, not estimated

Change A (merged, `a2064a1`) narrowed the router lock and bought correctness plus ~1.3x. Benchmarking on the production H100 then showed diarization is the binding constraint. Numbers from `scripts/h100/bench.sh` and the `ASR_STAGE_TIMINGS` log, 2-minute audio, 24 segments/request:

| | n=1 (uncontended) | n=10 | inflation |
|---|---|---|---|
| `diarization_s` | **0.883s** | **4.299s** | **4.9x** — a queue |
| `inference_s` (vLLM ASR) | 0.599s | 0.875s | 1.46x — **flat** |
| total_s | 1.633s | ~5.0s | |

Throughput at n=10: **0.78 req/s** with diarization on, **1.47 req/s** with `--no-diarization` (+88%). Wall 12.8s vs 6.8s. Amdahl: **~80% serialized** with diarization, **~52%** without.

`diarization_s` climbs monotonically (0.88 → 1.96 → 3.13 → 4.08 → 4.30, then plateaus) — the signature of a queue on `_diarization_inference_semaphore`, a `threading.BoundedSemaphore(1)` (`app/utils/speaker_diarizer.py:25`, taken at `:282`) around `pipeline(audio_path)` on one global CAM++ instance. `enable_speaker_diarization` defaults `True` (`router.py:35`), so this is the default path.

**`diarization_s` is queue-INCLUSIVE, not service time.** It is measured around `split_audio_by_speakers` (`app/services/asr/engines/base.py:192-194`), which wraps the semaphore acquire at `speaker_diarizer.py:282`. So the 4.299s is *wait + work*, not work. True service time is ~**0.883s**, and the arithmetic confirms it: at n=10 the mean queue position is 4.5, and 4.5 × 0.883 ≈ 4.0s ≈ the observed 4.299s. Ten requests × ~0.9s plus overhead ≈ the measured 12.8s wall.

This matters for expectations: **pooling removes the ~3.4s of queueing, not the 0.883s of work.** Each request still pays its diarization; it just pays it in parallel. The floor is therefore ASR inference (~0.75s × 10 ≈ 7.5s), which is what the `--no-diarization` run measured at 6.8s.

**The finding that reorders everything: `_llm_lock` is not contended.** The diarization mutex sits *upstream* and throttles requests single-file, so they arrive at the GPU already staggered and never collide. Diarization is masking the vLLM serialization. Until this is fixed, change C (AsyncLLM) would buy nothing.

### The mutex is load-bearing — do NOT delete it

The cheapest theoretical fix (drop the semaphore) is **unsafe**, verified in dependency source:

- `funasr/auto/auto_model.py:344-348` — `inference()` calls `self._reset_runtime_configs()`, then `kwargs = self.kwargs`, `kwargs.pop("cache")`, `deep_update(kwargs, cfg)`. **The shared `self.kwargs` dict is popped and deep-updated per call.**
- `funasr/auto/auto_model.py:729-748` — `_reset_runtime_configs()` rewrites instance attributes wholesale: `setattr(self, name, copy.deepcopy(base))`.

Two threads on one instance: A rebuilds `self.kwargs` while B is mid-read → nondeterministic VAD parameters. That is the same silent cross-request corruption class this service already shipped once. `modelscope/pipelines/audio/segmentation_clustering_pipeline.py:72` (`self.config.update(params)`) is a second instance of it — a no-op today only because we pass no params.

**The decisive evidence — `is_final` is written per chunk into that shared dict.** Everything above could be argued benign (`_reset_runtime_configs` writes the *same* values each call; the per-call `cache` is local; `self.config.update` is a no-op with no params). This one cannot:

```python
# funasr/auto/auto_model.py:345 — kwargs IS the shared instance dict
kwargs = self.kwargs if kwargs is None else kwargs

# funasr/models/fsmn_vad_streaming/model.py:697 — written PER CHUNK, value TOGGLES
for i in range(n):
    kwargs["is_final"] = _is_final and i == n - 1
    ...
    is_final=kwargs["is_final"],   # :706 — read back to drive VAD state
```

`is_final` **changes within a single call**, so GIL atomicity does not save it: two threads sharing the dict write *different* values to the same key while each depends on reading its own back. Thread A sees `is_final=True` mid-stream because thread B set it → VAD closes segments early → wrong speech boundaries → wrong speaker segmentation, **returned as a normal 200 response**. There is no interleaving under which both threads are correct.

*Note the race window is sub-second on a GPU, so the corruption would be **rare and unreproducible** — worse than frequent breakage, and exactly the failure mode this codebase has already been bitten by once.*

**Rejected accordingly:** "raise the semaphore to >1" and "delete the semaphore — it's just GPU inference." It is not just GPU inference; the hazard is CPU-side Python state on the singleton.

**Per-instance serialization must stay. Nothing requires *global* serialization across separate instances** — which is what pooling provides.

### Change 1 — stop funasr's per-call `torch.cuda.empty_cache()`

`funasr/auto/auto_model.py:410-417`, run after **every** funasr inference (i.e. the VAD inside every diarization call, inside our mutex):

```python
device = next(model.parameters()).device
if device.type == "cuda":
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
```

`empty_cache()` synchronizes the device and returns cached blocks to the driver. Two costs: per-request latency inside the critical section, and it flushes the caching allocator **on the same GPU vLLM is using**, forcing vLLM to re-acquire memory it had cached. The second is a cross-engine perturbation nobody has accounted for.

**Approach — a thread-local guard, not a global no-op.** `empty_cache` is resolved as `torch.cuda.empty_cache` at call time, so it cannot be patched per-module. Patching it to a global no-op would also disable it for vLLM and any future caller. Instead install a wrapper once at startup that skips only when the calling thread is inside diarization:

```python
_real_empty_cache = torch.cuda.empty_cache
_tls = threading.local()

def _guarded_empty_cache():
    if getattr(_tls, "skip_empty_cache", False):
        return
    _real_empty_cache()

torch.cuda.empty_cache = _guarded_empty_cache
```

Set `_tls.skip_empty_cache = True` for the duration of the diarization call (try/finally). Thread-local is required: diarization runs on executor threads while other threads use CUDA concurrently, so a process-global flag would race.

Install it in the same place and style as the existing diarization monkeypatching (`_enable_batched_sv`), idempotently — patching twice must not nest wrappers.

*Payoff unmeasured.* This is a latency cut of unknown size inside the 0.883s; the H100 profile (below) sizes it. Justified regardless: an allocator flush per request on a shared GPU is wrong on its own terms.

### Change 2 — pool N CAM++ pipeline instances

Replace the one global instance + global mutex with N independent instances. An instance checked out to a request is not shared, so **the pool itself is the mutex** and the per-instance serialization §2 requires is preserved with zero global contention.

**`LocalEnginePool` cannot be reused as-is.** It is `asyncio.Queue`-backed (`app/services/asr/runtime/local_pool.py:19-48`), but diarization is called **synchronously from a worker thread** (`app/services/asr/engines/base.py:193`, inside `run_sync`). It needs a `queue.Queue`-backed thread-safe twin of the same shape (lazy init under a lock, `acquire`/`release`, `warmup`). Mirror the existing pattern; do not make diarization async.

**Construction must be sequential.** modelscope `pipeline()` touches global registries, reads config files, and may download models on first run — building N concurrently is a race. Construct all N at warmup under the existing `_diarization_pipeline_lock`, each followed by `_enable_batched_sv`. Lazy child-pipeline creation inside the pipeline (`preprocess:179`, `postprocess:132`) is already neutralized because `_enable_batched_sv` pre-creates all three children.

**VRAM per instance is UNMEASURED — measure it on the H100 before fixing N. This is a blocking step, not a formality.**

> An earlier draft of this spec claimed "83 MB on disk for all four components, therefore <300 MB per instance, N=4 ≈ 1 GB." **That was false and is retracted.** The 83 MB of `iic/speech_campplus_speaker-diarization_common` is 60 MB of `asd.onnx`, 18 MB of example `.mp4`/`.wav`, a face-detection ONNX (`version-RFB-320.onnx`), and PNGs — **zero PyTorch weights for any of the four components**. The real weights live in separate dirs (`damo/speech_campplus_sv_zh-cn_16k-common`, `damo/speech_campplus-transformer_scl_zh-cn_16k-common`, `damo/speech_fsmn_vad_zh-cn-16k-common-pytorch`), and only the VAD (1.6 MB `model.pt`) is cached on the dev box — the SV and change-locator (a transformer) were never measured by anyone. The ~1 GB figure may still land in range, but that would be luck, not evidence.

What must actually be measured, on the H100:
- `nvidia-smi` delta across constructing ONE pipeline instance at warmup, with `_enable_batched_sv` applied. That is the real per-instance cost: weights + CUDA context + activations at the configured `DIARIZATION_SV_BATCH_SIZE`.
- Note each instance also builds a **throwaway duplicate SV pipeline** in `SegmentationClusteringPipeline.__init__` (`segmentation_clustering_pipeline.py:57-58`) *before* `_enable_batched_sv` replaces it. Its allocator-cached VRAM does not vanish on replacement. Unbudgeted; the nvidia-smi delta captures it.
- Then size N against what vLLM + the forced aligner have **not** pre-reserved via `gpu_memory_utilization`. vLLM reserves a fraction of the card up front, so "80GB" is not the headroom.

**Failure if skipped:** N sized on wrong arithmetic against a card where vLLM already holds most VRAM → CUDA OOM mid-diarization, in production, on the default path.

**Every consumer of the singleton must move.** `get_global_diarization_pipeline` has exactly three call sites, and all three are in scope:
- `app/utils/speaker_diarizer.py:282` — the request path; becomes pool acquire/release.
- `app/utils/model_loader.py:536-538` — startup warmup; must warm **all N**, not one.
- `tests/test_preload_models_config_repair.py:45` — patches the symbol; will break if the name disappears.

**Pool size:** `DIARIZATION_POOL_SIZE`, default **4**, env-configurable following `DIARIZATION_SV_BATCH_SIZE`. Reject `<1` at boot via the existing `_positive_int_from_env`. Sized against the measured queue depth (0.88→4.3s ≈ 5 requests deep), the **measured** per-instance VRAM (above), and the executor pool — not chosen independently.

**Invariant to state and hold:** a blocking `queue.Queue.get()` runs on an executor thread and holds that slot while waiting. This is safe only while `VLLM_OFFLINE_CONCURRENCY` (default 4) ≤ `DIARIZATION_POOL_SIZE` (default 4), which bounds waiters to zero. If an operator raises offline concurrency above the pool size, requests block executor threads waiting for a pipeline — the same starvation class change A's admission work addressed. Either enforce the relationship at boot or document it where both knobs are defined.

**Expected payoff — deliberately hedged.** The GIL caps thread-parallel gains: FSMN-VAD runs a per-frame Python state machine (`funasr/models/fsmn_vad_streaming/model.py`, ~12,000 iterations for 2 minutes of audio) that holds the GIL; the torch/numpy/scipy portions release it. So N=4 plausibly gives **2-3x diarization throughput, not 4x** — `diarization_s` at n=10 from 4.3s toward ~1.5-2s, and overall throughput from 0.78 toward ~1.2-1.3 req/s against the 1.47 ceiling that `--no-diarization` measured.

> **Do not treat that range as a forecast.** Performance predicted from code reading has been wrong four times on this system (2.7x, 3-5x, 4-5x, and 1.4-1.6x against a measured 1.29x→1.22x). It is a hypothesis. The bench decides.

### Ordering

1. **Change 1** (empty_cache guard) — independent, ~1 hour, no dependencies.
2. **Profile the 0.883s on the H100** — wrap the five stages of `SegmentationClusteringPipeline.__call__` (VAD → chunk → embed → cluster → postprocess) with `perf_counter` in the `_enable_batched_sv` style, log per stage. **Blocks nothing, but decides whether follow-on work targets VAD, change-locator, or nothing.** Prime suspect is FSMN-VAD's Python loop; clustering is verified cheap (spectral on a ~160×160 matrix; the UMAP/HDBSCAN path needs ≥2048 chunks ≈ >25 min of speech).
3. **Change 2** (pool) — the main event.
4. **Re-bench** with `scripts/h100/bench.sh`, diarization ON, same 2-minute audio. Compare against 0.78 req/s / 12.8s / 4.3s.

### Success criteria

- `diarization_s` at n=10 drops materially from 4.299s. **No target multiple is set** — the GIL ceiling is unknown until measured, and inventing a number here would repeat a documented failure mode.
- Throughput at n=10 improves from 0.78 req/s toward the 1.47 req/s that `--no-diarization` established as the ceiling.
- **`scripts/h100/test_offline_mixing.py` passes** — N pipeline instances must not mix transcripts across requests. Mandatory: this change touches the exact machinery whose serialization was proven necessary above.
- Zero `fails` in the bench. A better p95 achieved by erroring out slow requests is not a win.
- Existing suite stays green: `DEVICE=cpu .venv/bin/python -m unittest discover -s tests`
- No `git pull`/fetch/rebase during the branch.

### Non-goals

- **Change C (AsyncLLM) is out of scope** and would currently buy nothing: `inference_s` is flat at ~0.75s because the diarization mutex throttles upstream. Re-measure after this lands — with the queue gone, `_llm_lock` becomes the next ceiling and C becomes the right work.
- **Deleting the mutex** — proven unsafe above.
- **Raising `DIARIZATION_SV_BATCH_SIZE` further.** The default is **32** (`config.py:74`); prod currently overrides it to 128. At ~160 chunks per 2-minute request, 128 already collapses the work into 2 batches, so raising it further changes nothing. (At the 32 default it would be ~5 batches — so the override is doing real work and should stay.)
- Silero-VAD front-end, dropping `change_locator`, a process pool to escape the GIL, or a different diarization stack (pyannote/NeMo). All live options; all gated on the §2 profile.
- A VAD-only path for requests not needing speaker labels. Worth noting the cheapest win available needs **no code at all**: `enable_speaker_diarization=false` per request already yields +88% throughput. That is a product decision, not engineering.

### Risks

- **Correctness.** N instances must be genuinely independent. Verified: `_enable_batched_sv` monkeypatches per instance via `types.MethodType` (`speaker_diarizer.py:202`), and each instance gets its own sv/vad/change_locator children (`:127-146`). The only cross-instance shared object is the CUDA default stream — kernels serialize on-GPU, which is fine. Still, the mixing test is mandatory.
- **The GIL may cap the win below the useful threshold.** If the profile shows VAD-Python is >60% of the 0.883s, thread pooling under-delivers and a process pool (or a Silero front-end) becomes the real answer. Measure before assuming.
- **Idempotent patching.** The `empty_cache` wrapper must not stack if installed twice (reload, tests, multiple workers).
- **Thread-local correctness.** The guard must use `threading.local`, not a module flag — diarization runs on executor threads concurrently with other CUDA users.
- **Dev box cannot validate any of this.** Non-CUDA AMD APU; local tests verify structure with fakes only. Both changes need the H100 to be believed.

---

**Implementation plan:** [`docs/superpowers/plans/2026-07-17-diarization-throughput-plan.md`](../plans/2026-07-17-diarization-throughput-plan.md)
