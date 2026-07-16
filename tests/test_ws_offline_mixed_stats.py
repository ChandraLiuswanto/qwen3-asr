# -*- coding: utf-8 -*-
"""Unit tests for the PURE statistics logic of the mixed ws+offline H100 script.

SCOPE — READ THIS FIRST
-----------------------
These tests cover `percentile` and `describe_latencies`: numbers in, numbers
out. Nothing else in that script is tested here.

WHAT THEY DO NOT COVER — and cannot:
  * Whether the websocket latencies being fed in are MEASURED correctly. That
    needs a real service.
  * Whether the ITN race exists, or whether the ITN lock fixes it. That race
    only appears when a websocket's event-loop-thread ITN call overlaps an
    offline request's executor-thread ITN call, on a real deployment.
  * Whether the ws protocol handshake, the lockstep chunking, or the mixed
    concurrency work at all.

A green run here means "the percentile arithmetic is right". It is not evidence
about the service. A local test that faked the concurrency and went green would
be worse than no test; there is deliberately none.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

_H100 = Path(__file__).resolve().parents[1] / "scripts" / "h100"
# test_ws_offline_mixed imports test_offline_mixing as a sibling module.
sys.path.insert(0, str(_H100))
_spec = importlib.util.spec_from_file_location("test_ws_offline_mixed", _H100 / "test_ws_offline_mixed.py")
assert _spec is not None and _spec.loader is not None
mixed = importlib.util.module_from_spec(_spec)
# Register before exec: @dataclass resolves its module from sys.modules.
sys.modules[_spec.name] = mixed
_spec.loader.exec_module(mixed)


class PercentileTests(unittest.TestCase):
    def test_empty_returns_none_not_zero(self):
        # "No data" must never be reportable as "zero latency".
        self.assertIsNone(mixed.percentile([], 0.5))

    def test_single_value(self):
        self.assertEqual(mixed.percentile([0.42], 0.5), 0.42)
        self.assertEqual(mixed.percentile([0.42], 0.95), 0.42)

    def test_nearest_rank_p50_of_ten(self):
        values = [float(n) for n in range(1, 11)]  # 1..10
        self.assertEqual(mixed.percentile(values, 0.50), 5.0)

    def test_nearest_rank_p95_of_twenty(self):
        values = [float(n) for n in range(1, 21)]  # 1..20
        self.assertEqual(mixed.percentile(values, 0.95), 19.0)

    def test_result_is_always_an_observed_sample_not_an_interpolation(self):
        values = [1.0, 100.0]
        self.assertIn(mixed.percentile(values, 0.5), values)
        self.assertIn(mixed.percentile(values, 0.95), values)

    def test_unsorted_input_is_sorted_internally(self):
        self.assertEqual(mixed.percentile([9.0, 1.0, 5.0], 0.5), 5.0)

    def test_p100_is_the_max_and_p0_is_the_min(self):
        values = [3.0, 1.0, 2.0]
        self.assertEqual(mixed.percentile(values, 1.0), 3.0)
        self.assertEqual(mixed.percentile(values, 0.0), 1.0)

    def test_out_of_range_fraction_raises(self):
        with self.assertRaises(ValueError):
            mixed.percentile([1.0], 1.5)
        with self.assertRaises(ValueError):
            mixed.percentile([1.0], -0.1)


class DescribeLatenciesTests(unittest.TestCase):
    def test_empty_reports_zero_count_and_no_numbers(self):
        stats = mixed.describe_latencies([])
        self.assertEqual(stats["count"], 0)
        for key in ("p50", "p95", "min", "max", "mean"):
            self.assertIsNone(stats[key], f"{key} must be None, not a fabricated value")

    def test_reports_count_min_max_mean_and_percentiles(self):
        stats = mixed.describe_latencies([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(stats["count"], 4)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 4.0)
        self.assertEqual(stats["mean"], 2.5)
        self.assertEqual(stats["p50"], 2.0)
        self.assertEqual(stats["p95"], 4.0)


class WsUrlTests(unittest.TestCase):
    def test_http_becomes_ws(self):
        self.assertEqual(mixed.ws_url("http://localhost:8000"), "ws://localhost:8000/ws/v1/asr/qwen")

    def test_https_becomes_wss(self):
        self.assertEqual(mixed.ws_url("https://asr.example.com/"), "wss://asr.example.com/ws/v1/asr/qwen")

    def test_unknown_scheme_fails_loudly_rather_than_guessing(self):
        with self.assertRaises(SystemExit):
            mixed.ws_url("localhost:8000")


if __name__ == "__main__":
    unittest.main()
