"""Parent-side pool manager. Two layers:
- REAL spawn integration (DIARIZATION_WORKER_FAKE=1): start() must bring up
  N distinct worker processes and diarize() must return native triples.
- Fake-executor unit tests for the BrokenProcessPool rebuild-once contract.
"""

import os
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
        # the raced rebuild must NOT tear down the replacement
        replacement.shutdown.assert_not_called()

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
