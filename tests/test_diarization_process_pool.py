"""Parent-side pool manager. Two layers:
- REAL spawn integration (DIARIZATION_WORKER_FAKE=1): start() must bring up
  N distinct worker processes and diarize() must return native triples.
- Fake-executor unit tests for the BrokenProcessPool rebuild-once contract.
"""

import os
import threading
import unittest
from concurrent.futures.process import BrokenProcessPool
from unittest import mock

from app.core.exceptions import DefaultServerErrorException
from app.utils.diarization_pool import DiarizationProcessPool


class SpawnIntegrationTest(unittest.TestCase):
    """Real processes, fake pipeline. Slow-ish (~seconds); still CPU-safe."""

    def setUp(self):
        self._env = mock.patch.dict(os.environ, {"DIARIZATION_WORKER_FAKE": "1"})
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_start_then_diarize_end_to_end(self):
        pool = DiarizationProcessPool(2, boot_timeout_s=120)
        pool.start()
        self.addCleanup(pool.close)
        triples = pool.diarize("/tmp/anything.wav")
        self.assertEqual(triples, [(0.0, 1.5, 0), (1.5, 3.0, 1)])
        for s, e, label in triples:
            self.assertIs(type(s), float)
            self.assertIs(type(label), int)

    def test_start_is_idempotent(self):
        pool = DiarizationProcessPool(1, boot_timeout_s=120)
        pool.start()
        self.addCleanup(pool.close)
        pool.start()  # second call must be a no-op, not a respawn

    def test_broken_pool_republishes_and_next_diarize_succeeds(self):
        # Real processes: a break fast-fails the request, then the off-thread
        # rebuild republishes a working pool and the next diarize() succeeds.
        pool = DiarizationProcessPool(1, boot_timeout_s=120)
        pool.start()
        self.addCleanup(pool.close)
        self.assertEqual(
            pool.diarize("/tmp/a.wav"), [(0.0, 1.5, 0), (1.5, 3.0, 1)]
        )

        # Swap the live executor for a broken stand-in to simulate a crash.
        live = pool._executor
        self.addCleanup(
            lambda: live.shutdown(wait=False, cancel_futures=True)
        )
        broken = mock.Mock()
        broken.submit.side_effect = BrokenProcessPool("simulated crash")
        pool._executor = broken

        with self.assertRaises(DefaultServerErrorException):
            pool.diarize("/tmp/b.wav")  # fast-fails, rebuild goes off-thread
        pool._rebuild_thread.join(timeout=120)  # real respawn + fake pipeline
        self.assertFalse(pool._rebuild_thread.is_alive())

        self.assertIsNotNone(pool._executor)
        self.assertIsNot(pool._executor, broken)
        self.assertEqual(
            pool.diarize("/tmp/c.wav"), [(0.0, 1.5, 0), (1.5, 3.0, 1)]
        )

    def test_close_terminates_real_worker_processes(self):
        # close() on a live pool must actually terminate the worker processes
        # (not just null the reference) and leave diarize() fast-failing.
        pool = DiarizationProcessPool(2, boot_timeout_s=120)
        pool.start()
        procs = list(pool._executor._processes.values())
        self.assertEqual(len(procs), 2)
        self.assertTrue(all(p.is_alive() for p in procs))

        pool.close()

        self.assertIsNone(pool._executor)
        for p in procs:
            p.join(timeout=10)
            self.assertFalse(p.is_alive())
        with self.assertRaises(DefaultServerErrorException):
            pool.diarize("/tmp/x.wav")


class PoolContractTest(unittest.TestCase):
    def test_diarize_before_start_raises_server_error(self):
        pool = DiarizationProcessPool(2, boot_timeout_s=5)
        with self.assertRaises(DefaultServerErrorException):
            pool.diarize("/tmp/x.wav")

    def test_broken_pool_rebuilds_once_and_fails_the_request(self):
        pool = DiarizationProcessPool(2, boot_timeout_s=5)
        broken_executor = mock.Mock()
        broken_executor.submit.side_effect = BrokenProcessPool("worker died")
        pool._executor = broken_executor
        with mock.patch.object(pool, "start") as restart:
            with self.assertRaises(DefaultServerErrorException):
                pool.diarize("/tmp/x.wav")
            # Rebuild now runs off-thread; join it so the assertions below are
            # not racing the background rebuild. Join is INSIDE the patch
            # context so start() is still the mock when the thread calls it.
            pool._rebuild_thread.join(timeout=5)
            self.assertFalse(pool._rebuild_thread.is_alive())
        restart.assert_called_once_with()
        broken_executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    def test_rebuild_skipped_if_another_thread_already_rebuilt(self):
        pool = DiarizationProcessPool(2, boot_timeout_s=5)
        broken_executor = mock.Mock()
        broken_executor.submit.side_effect = BrokenProcessPool("worker died")
        pool._executor = broken_executor
        replacement = mock.Mock()

        def swap(broken):
            # simulate a racing thread having already swapped the executor
            pool._executor = replacement
            return DiarizationProcessPool._rebuild(pool, broken)

        with mock.patch.object(pool, "_rebuild", side_effect=swap):
            with self.assertRaises(DefaultServerErrorException):
                pool.diarize("/tmp/x.wav")
            # Join the off-thread rebuild so the compare-and-swap in the real
            # _rebuild has actually run before we assert on the replacement.
            pool._rebuild_thread.join(timeout=5)
            self.assertFalse(pool._rebuild_thread.is_alive())
        # the raced rebuild must NOT tear down the replacement
        replacement.shutdown.assert_not_called()

    def test_diarize_fast_fails_while_rebuild_blocks_off_thread(self):
        # Prove the failing request returns immediately even while the rebuild
        # is still blocked deep inside a slow start(): if the rebuild ran
        # inline, diarize() could not have returned until start() finished.
        pool = DiarizationProcessPool(2, boot_timeout_s=300)
        broken_executor = mock.Mock()
        broken_executor.submit.side_effect = BrokenProcessPool("worker died")
        pool._executor = broken_executor

        start_entered = threading.Event()
        release_start = threading.Event()

        def blocking_start():
            start_entered.set()
            # hold the rebuild thread hostage inside start()
            if not release_start.wait(timeout=10):
                raise AssertionError("release_start never set")

        with mock.patch.object(pool, "start", side_effect=blocking_start):
            with self.assertRaises(DefaultServerErrorException):
                pool.diarize("/tmp/x.wav")
            # The request has already returned. The rebuild is on another
            # thread and is (about to be) blocked inside start().
            self.assertTrue(start_entered.wait(timeout=5))
            self.assertIsNot(
                pool._rebuild_thread, threading.current_thread()
            )
            self.assertTrue(pool._rebuild_thread.is_alive())
            # let the off-thread rebuild finish so nothing leaks
            release_start.set()
            pool._rebuild_thread.join(timeout=5)
        self.assertFalse(pool._rebuild_thread.is_alive())

    def test_rebuild_in_flight_racing_close_leaves_no_orphan(self):
        # Fix 2: a close() that lands WHILE a rebuild/start() is building the
        # replacement must end with no published executor and the fresh
        # executor torn down (not orphaned). Driven deterministically: the
        # ProcessPoolExecutor construction itself simulates close() landing
        # mid-build, then start() must observe _closing at publish.
        pool = DiarizationProcessPool(1, boot_timeout_s=5)
        fresh = mock.Mock()
        ctx = mock.MagicMock()

        def build_fresh(*args, **kwargs):
            # close() arrives while start() is between the top-guard and the
            # publish (this is the exact race Fix 2 addresses).
            pool.close()
            return fresh

        with mock.patch(
            "app.utils.diarization_pool.multiprocessing.get_context",
            return_value=ctx,
        ), mock.patch(
            "app.utils.diarization_pool.ProcessPoolExecutor",
            side_effect=build_fresh,
        ), mock.patch(
            "app.utils.diarization_pool.futures_wait",
            return_value=mock.Mock(done=[]),
        ):
            pool.start()

        # No orphan: the fresh executor was shut down and never published.
        self.assertIsNone(pool._executor)
        self.assertTrue(pool._closing)
        fresh.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    def test_worker_exception_propagates_raw(self):
        pool = DiarizationProcessPool(2, boot_timeout_s=5)
        fake_future = mock.Mock()
        fake_future.result.side_effect = AssertionError(
            "modelscope error: The effective audio duration is too short."
        )
        fake_executor = mock.Mock()
        fake_executor.submit.return_value = fake_future
        pool._executor = fake_executor
        with self.assertRaises(AssertionError):
            pool.diarize("/tmp/x.wav")


if __name__ == "__main__":
    unittest.main()
