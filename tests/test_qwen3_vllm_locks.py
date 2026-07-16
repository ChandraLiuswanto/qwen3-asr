# -*- coding: utf-8 -*-
"""Tests for per-engine backend locks in Qwen3VLLMBackend (spec change A, items 1-2)."""
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

        # daemon=True + watchdog, matching the neighbouring lock tests: if the
        # offline and ws paths stop sharing one lock and deadlock, this must
        # fail the assertion rather than hang the suite forever.
        threads = [
            threading.Thread(target=offline if i % 2 else ws, daemon=True)
            for i in range(8)
        ]
        done = threading.Event()

        def guard() -> None:
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            done.set()

        watchdog = threading.Thread(target=guard, daemon=True)
        watchdog.start()
        self.assertTrue(done.wait(timeout=10.0), "offline/ws decode deadlocked")
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
