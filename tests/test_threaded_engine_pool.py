"""Structure-only tests for ThreadedEnginePool.

These verify checkout semantics with fakes on a CPU dev box. They prove
NOTHING about real GPU concurrency; the H100 mixing test is the real gate.
"""

import queue
import threading
import unittest

from app.services.asr.runtime.local_pool import ThreadedEnginePool


class ThreadedEnginePoolTest(unittest.TestCase):
    def test_lazy_builds_exactly_n_instances_sequentially(self) -> None:
        build_log = []

        def factory():
            build_log.append(threading.get_ident())
            return object()

        pool = ThreadedEnginePool(3, factory)
        self.assertEqual(build_log, [])  # lazy: nothing built yet
        pool.warmup()
        self.assertEqual(len(build_log), 3)
        # Sequential under the init lock: all built on one thread.
        self.assertEqual(len(set(build_log)), 1)

    def test_warmup_is_idempotent(self) -> None:
        count = [0]

        def factory():
            count[0] += 1
            return object()

        pool = ThreadedEnginePool(2, factory)
        pool.warmup()
        pool.warmup()
        self.assertEqual(count[0], 2)

    def test_checked_out_instance_is_exclusive(self) -> None:
        pool = ThreadedEnginePool(2, object)
        a = pool.acquire()
        b = pool.acquire()
        self.assertIsNot(a, b)
        # Pool exhausted: a third acquire must block, not hand out a dup.
        result: "queue.Queue[object]" = queue.Queue()
        t = threading.Thread(target=lambda: result.put(pool.acquire()))
        t.start()
        with self.assertRaises(queue.Empty):
            result.get(timeout=0.2)  # still blocked -> exclusivity holds
        pool.release(a)
        c = result.get(timeout=2.0)
        t.join(timeout=2.0)
        self.assertIs(c, a)  # released instance is the one handed out
        pool.release(b)
        pool.release(c)

    def test_size_floor_is_one(self) -> None:
        pool = ThreadedEnginePool(0, object)
        pool.warmup()
        self.assertIsNotNone(pool.acquire())


if __name__ == "__main__":
    unittest.main()
