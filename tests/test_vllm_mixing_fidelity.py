# -*- coding: utf-8 -*-
"""Regression test: `_llm_lock` defeats vLLM's cross-caller output mixing.

WHAT THIS PROVES
----------------
`Qwen3VLLMBackend._run_generate` calls `LLM.generate` on a SHARED engine and
then does `zip(outputs, audio_items)`. vLLM's `LLM.generate` returns whatever
finished on the shared engine, with no ownership filter, so two concurrent
callers can each receive the other's outputs; `zip` then silently truncates and
pairs the WRONG transcript onto the RIGHT segment. Wrong text, no exception.

The fake engine below replicates that mechanism from the real vLLM 0.19.0
source (wheel `vllm-0.19.0-cp38-abi3-manylinux_2_31_x86_64.whl`, extracted to
`vllm/entrypoints/llm.py`; verbatim excerpts in
`.superpowers/sdd/vllm-0.19.0-evidence.txt`). Each modeling choice cites the
line it comes from:

  * ONE shared engine + ONE shared `self.request_counter = Counter()`
    (llm.py:388). Modeled by `_FakeEngine.request_counter`.
  * `_add_request` draws `request_id = str(next(self.request_counter))`
    (llm.py:1976) and `_render_and_add_requests` calls it in a PER-PROMPT loop
    (llm.py:1949-1959) -> concurrent callers INTERLEAVE ids (A=0, B=1, A=2,
    B=3). This interleave is what makes the corruption produce wrong text
    rather than a merely short list. Modeled by `_FakeLLM.generate`'s add loop.
  * `_run_engine` drains the shared engine:
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
    (llm.py:1989-2028) -- appends EVERY finished output, NO filter for the
    caller's own request ids. Modeled by `_FakeLLM._run_engine`.
  * `generate` calls `_run_engine(output_type, use_tqdm=use_tqdm)` (llm.py:588)
    passing NO request ids, so filtering by ownership is structurally
    impossible. Modeled by `_FakeLLM._run_engine` taking no id argument.
  * The drain ends with `return sorted(outputs, key=lambda x: int(x.request_id))`
    (llm.py:2035). Sorting does NOT restore ownership -- with interleaved ids
    the sort puts a foreign output at index 1 of caller A's slice. Modeled
    verbatim.

WHAT THIS DOES *NOT* PROVE -- read before trusting it
-----------------------------------------------------
This exercises OUR locking against a faithful MODEL of vLLM's behavior as
documented by its source. It does NOT:
  * run real vLLM, a real LLMEngine, real CUDA, or a real model;
  * prove anything about real-hardware timing, scheduler behavior, continuous
    batching, or preemption;
  * prove vLLM 0.19.0 behaves this way at runtime -- only that its source says
    so (the H100 probe in `scripts/h100/` is what checks the real thing);
  * cover any vLLM version other than 0.19.0.

A green run here means "our lock defeats the mixing mechanism the vLLM source
describes", NOT "verified on hardware". Do not mistake this for hardware
verification.
"""
from __future__ import annotations

import itertools
import threading
import time
import unittest
from pathlib import Path

import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from app.services.asr.qwen3_vllm import Qwen3VLLMBackend, _load_chat_template

# Rendezvous timeout. When the lock serializes callers, the second caller never
# shows up, so the first must time out and proceed SOLO rather than hang.
_RENDEZVOUS_TIMEOUT = 0.25


class _Rendezvous:
    """Two-party meeting point with a timeout.

    Unlocked run: both callers arrive -> they sync -> ids interleave and the
    drains race, which is the real-world overlap we need to reproduce.
    Locked run: the second caller is blocked on `_llm_lock` and never arrives,
    so the first times out and proceeds alone. The timeout is what keeps the
    locked case from deadlocking instead of passing.
    """

    def __init__(self, parties: int, timeout: float) -> None:
        self._parties = parties
        self._timeout = timeout
        self._cv = threading.Condition()
        self._count = 0
        self._generation = 0

    def wait(self) -> bool:
        with self._cv:
            self._count += 1
            if self._count >= self._parties:
                self._count = 0
                self._generation += 1
                self._cv.notify_all()
                return True
            generation = self._generation
            deadline = time.monotonic() + self._timeout
            while self._generation == generation:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._count -= 1
                    return False
                self._cv.wait(remaining)
            return True


class _Turnstile:
    """Forces strict round-robin so ids interleave A,B,A,B deterministically.

    Models llm.py:1949-1959 + :1976 under contention: each caller takes the
    shared counter once per prompt, so with two concurrent callers the ids
    alternate. Without strict alternation the unlocked case would sometimes
    hand caller A ids 0,1 (its own, in order) and the corruption would show up
    only as a short list rather than as wrong text -- the test would be flaky
    about WHICH failure it demonstrates. On timeout the waiter proceeds anyway,
    so the locked (solo) case cannot hang.
    """

    def __init__(self, parties: int, timeout: float) -> None:
        self._parties = parties
        self._timeout = timeout
        self._cv = threading.Condition()
        self._turn = 0

    def take(self, index: int) -> None:
        with self._cv:
            deadline = time.monotonic() + self._timeout
            while self._turn % self._parties != index:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # Solo caller (the other is blocked on the lock): claim the
                    # turn and continue.
                    self._turn = index
                    return
                self._cv.wait(remaining)

    def release(self) -> None:
        with self._cv:
            self._turn += 1
            self._cv.notify_all()


class _FakeCompletion:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeRequestOutput:
    """Shape-compatible with vllm.RequestOutput for the fields we touch."""

    def __init__(self, request_id: str, text: str) -> None:
        self.request_id = request_id
        self.finished = True
        self.outputs = [_FakeCompletion(text)]


class _FakeEngine:
    """The ONE shared engine, with the ONE shared request counter (llm.py:388).

    `step()` returns finished outputs belonging to ANY caller -- that is the
    whole bug. Pending state is mutex-guarded only so concurrent drains do not
    corrupt the dict itself; the mutex deliberately does NOT restore ownership.
    """

    def __init__(self) -> None:
        self.request_counter = itertools.count()
        self._mu = threading.Lock()
        self._pending: dict[str, _FakeRequestOutput] = {}

    def add_request(self, request_id: str, output: _FakeRequestOutput) -> None:
        with self._mu:
            self._pending[request_id] = output

    def has_unfinished_requests(self) -> bool:
        with self._mu:
            return bool(self._pending)

    def step(self) -> list[_FakeRequestOutput]:
        # One request completes per step, and it goes to WHOEVER is draining.
        with self._mu:
            if not self._pending:
                return []
            request_id = next(iter(self._pending))
            return [self._pending.pop(request_id)]


class _FakeLLM:
    """Models vllm.LLM.generate over the shared engine.

    Mirrors the real control flow: per-prompt `_add_request` off the shared
    counter, then `_run_engine`'s unfiltered drain, then the sort by
    request_id.
    """

    def __init__(self, engine: _FakeEngine, turnstile: _Turnstile, rendezvous: _Rendezvous) -> None:
        self.llm_engine = engine
        self._turnstile = turnstile
        self._rendezvous = rendezvous
        self.caller_index = threading.local()

    @staticmethod
    def _marker(prompt: dict) -> str:
        # The caller's context string IS the system-turn content under the
        # native prompt format, so slice the system turn directly. The
        # traceability guarantee (WRONG TEXT on the WRONG request is provable)
        # is unchanged.
        text = prompt["prompt"]
        needle = "<|im_start|>system\n"
        start = text.index(needle) + len(needle)
        return text[start:text.index("<|im_end|>", start)].strip()

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        index = self.caller_index.value
        # --- add phase: llm.py:1949-1959 calling :1976 per prompt ---
        for prompt in prompts:
            self._turnstile.take(index)
            request_id = str(next(self.llm_engine.request_counter))
            self.llm_engine.add_request(
                request_id, _FakeRequestOutput(request_id, self._marker(prompt))
            )
            self._turnstile.release()
        # Both callers finish adding before either drains: the overlap window
        # the real bug needs. Solo callers time out and drain immediately.
        self._rendezvous.wait()
        return self._run_engine()

    def _run_engine(self):
        # Verbatim shape of llm.py:1989-2035. Note what is absent: any request
        # id. `generate` passes none (llm.py:588), so no ownership filter is
        # possible.
        outputs: list[_FakeRequestOutput] = []
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
        return sorted(outputs, key=lambda x: int(x.request_id))


class _NullLock:
    """Stands in for `_llm_lock` when proving the test fails without it.

    A no-op context manager (reusable and reentrant, unlike a one-shot
    @contextmanager instance) -- i.e. exactly a build with no lock.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeSamplingParamsCls:
    def __init__(self, **kwargs):
        pass


def _backend(locked: bool) -> Qwen3VLLMBackend:
    engine = _FakeEngine()
    turnstile = _Turnstile(2, _RENDEZVOUS_TIMEOUT)
    rendezvous = _Rendezvous(2, _RENDEZVOUS_TIMEOUT)
    backend = Qwen3VLLMBackend.__new__(Qwen3VLLMBackend)
    backend._llm = _FakeLLM(engine, turnstile, rendezvous)
    backend._sampling_params = object()
    backend._sampling_params_cls = _FakeSamplingParamsCls
    backend._tokenizer = PreTrainedTokenizerBase()
    backend._chat_template = _load_chat_template(Path(__file__).parent / "fixtures" / "qwen3_asr")
    backend._max_inference_batch_size = 4
    backend._llm_lock = threading.Lock() if locked else _NullLock()
    backend._aligner_lock = threading.Lock()
    backend._aligner_init_lock = threading.Lock()
    return backend


def _run_two_callers(backend: Qwen3VLLMBackend) -> dict[str, list]:
    """Two concurrent callers, two segments each, through the REAL _run_generate."""
    audio = np.zeros(160, dtype=np.float32)
    results: dict[str, list] = {}
    errors: list[BaseException] = []

    def call(name: str, index: int) -> None:
        backend._llm.caller_index.value = index
        try:
            results[name] = backend._run_generate(
                [(audio, f"{name}-0", None), (audio, f"{name}-1", None)]
            )
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion
            errors.append(exc)

    threads = [
        threading.Thread(target=call, args=("ALPHA", 0), daemon=True),
        threading.Thread(target=call, args=("BRAVO", 1), daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15.0)
    assert not errors, errors
    return results


class VLLMMixingFidelityTest(unittest.TestCase):
    def test_without_lock_callers_receive_each_others_transcripts(self) -> None:
        """The bug, reproduced: caller ALPHA is handed BRAVO's transcript.

        Guards the fidelity of the model itself. If this stops failing-without-
        the-lock, the fake has drifted from the real vLLM semantics and the
        companion test below is no longer proving anything.
        """
        backend = _backend(locked=False)
        results = _run_two_callers(backend)

        alpha = [t.text for t in results["ALPHA"]]
        bravo = [t.text for t in results["BRAVO"]]

        # ids interleave ALPHA=0, BRAVO=1, ALPHA=2, BRAVO=3; whichever caller
        # wins the drain collects all four, sorts them, and zip() truncates to
        # its own two segments -> index 1 is the OTHER caller's transcript.
        foreign = [t for t in alpha if t.startswith("BRAVO")] + [
            t for t in bravo if t.startswith("ALPHA")
        ]
        self.assertTrue(
            foreign,
            f"expected cross-caller mixing without the lock; got ALPHA={alpha} BRAVO={bravo}",
        )

        # Demonstrate the CORRUPTION concretely, not merely "something broke":
        # a caller's own segment is paired with text it never submitted, and
        # the other caller's transcript is silently lost to zip truncation.
        winner, stolen = ("ALPHA", alpha) if any(
            t.startswith("BRAVO") for t in alpha
        ) else ("BRAVO", bravo)
        self.assertEqual(
            len(stolen), 2, "zip should still yield one transcript per submitted segment"
        )
        self.assertTrue(
            any(not t.startswith(winner) for t in stolen),
            f"{winner} should hold a foreign transcript; got {stolen}",
        )

    def test_with_llm_lock_each_caller_gets_only_its_own_outputs(self) -> None:
        """The fix: `_llm_lock` makes the mixing mechanism unreachable."""
        backend = _backend(locked=True)
        results = _run_two_callers(backend)

        self.assertEqual(sorted(results), ["ALPHA", "BRAVO"])
        for name in ("ALPHA", "BRAVO"):
            texts = [t.text for t in results[name]]
            # Exact pairing: segment i's transcript is segment i's own marker.
            self.assertEqual(
                texts,
                [f"{name}-0", f"{name}-1"],
                f"{name} received mixed or truncated outputs: {texts}",
            )


if __name__ == "__main__":
    unittest.main()
