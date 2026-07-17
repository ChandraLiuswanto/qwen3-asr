"""Structure-only tests for the torch.cuda.empty_cache guard.

funasr calls torch.cuda.empty_cache() after every inference
(funasr/auto/auto_model.py:410-417), flushing the allocator on the GPU vLLM
shares. The guard skips it only on threads inside a diarization call.
On this CPU dev box empty_cache is a no-op, so these tests verify wiring
(idempotence, thread-locality, restore-on-exit) with a sentinel — not any
CUDA behavior.
"""

import threading
import unittest

import torch

from app.utils import speaker_diarizer as sd


class EmptyCacheGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        # Isolate each test from prior installs.
        self._original = torch.cuda.empty_cache
        self.calls = []
        torch.cuda.empty_cache = lambda: self.calls.append(True)
        sd._empty_cache_guard_installed = False

    def tearDown(self) -> None:
        torch.cuda.empty_cache = self._original
        sd._empty_cache_guard_installed = False

    def test_install_is_idempotent_no_nested_wrappers(self) -> None:
        sd._install_empty_cache_guard()
        first = torch.cuda.empty_cache
        sd._install_empty_cache_guard()
        self.assertIs(torch.cuda.empty_cache, first)  # not re-wrapped

    def test_passthrough_outside_diarization(self) -> None:
        sd._install_empty_cache_guard()
        torch.cuda.empty_cache()
        self.assertEqual(len(self.calls), 1)

    def test_skipped_inside_suppress_and_restored_after(self) -> None:
        sd._install_empty_cache_guard()
        with sd._suppress_empty_cache():
            torch.cuda.empty_cache()
        self.assertEqual(len(self.calls), 0)
        torch.cuda.empty_cache()
        self.assertEqual(len(self.calls), 1)

    def test_restored_even_on_exception(self) -> None:
        sd._install_empty_cache_guard()
        with self.assertRaises(RuntimeError):
            with sd._suppress_empty_cache():
                raise RuntimeError("boom")
        torch.cuda.empty_cache()
        self.assertEqual(len(self.calls), 1)

    def test_flag_is_thread_local(self) -> None:
        sd._install_empty_cache_guard()
        other_thread_calls = []

        def other() -> None:
            torch.cuda.empty_cache()
            other_thread_calls.append(len(self.calls))

        with sd._suppress_empty_cache():
            t = threading.Thread(target=other)
            t.start()
            t.join(timeout=5.0)
        # The other thread was NOT suppressed.
        self.assertEqual(other_thread_calls, [1])


if __name__ == "__main__":
    unittest.main()
