# -*- coding: utf-8 -*-
"""Unit tests for the pure classification logic of the H100 vLLM probe.

Only `classify_drain` is testable here: it takes source TEXT and returns a
verdict, so it needs no vllm install. Everything else in the probe requires a
real vllm on a CUDA box and is deliberately untested locally — a mock of
`inspect.getsource` would prove nothing about vLLM 0.19.0.
"""

import importlib.util
import unittest
from pathlib import Path

_PROBE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "h100" / "verify_vllm_root_cause.py"
_spec = importlib.util.spec_from_file_location("verify_vllm_root_cause", _PROBE_PATH)
assert _spec is not None and _spec.loader is not None
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


# Shaped after the upstream drain the hypothesis describes: collect whatever
# finished, then sort by request id. Sorting is the caveat — it must NOT be
# read as filtering.
UNFILTERED_WITH_SORT = '''
def _run_engine(self, *, use_tqdm):
    outputs = []
    while self.llm_engine.has_unfinished_requests():
        step_outputs = self.llm_engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
    outputs.sort(key=lambda x: int(x.request_id))
    return outputs
'''

FILTERED_BY_REQUEST_ID = '''
def _run_engine(self, request_ids, *, use_tqdm):
    outputs = []
    while self.llm_engine.has_unfinished_requests():
        step_outputs = self.llm_engine.step()
        for output in step_outputs:
            if output.finished and output.request_id in request_ids:
                outputs.append(output)
    return outputs
'''

FILTERED_BY_PER_CALLER_MAP = '''
def _run_engine(self, pending, *, use_tqdm):
    results = {}
    while pending:
        for output in self.llm_engine.step():
            if output.finished:
                slot = pending.pop(output.request_id)
                results[slot] = output
    return [results[i] for i in range(len(results))]
'''

UNRECOGNIZED = '''
def _run_engine(self):
    return self.some_totally_different_backend.collect_everything()
'''


class ClassifyDrainTests(unittest.TestCase):
    def test_unfiltered_drain_confirms_hypothesis(self):
        verdict, evidence, _notes = probe.classify_drain(UNFILTERED_WITH_SORT)
        self.assertEqual(verdict, probe.CONFIRMED)
        self.assertTrue(evidence)

    def test_sorting_is_reported_as_a_note_not_as_filtering(self):
        verdict, _evidence, notes = probe.classify_drain(UNFILTERED_WITH_SORT)
        # The caveat: a sort must not flip the verdict to REFUTED...
        self.assertEqual(verdict, probe.CONFIRMED)
        # ...but it must be surfaced so the reader sees the loop is not
        # literally the pseudocode.
        self.assertTrue(any("sorted by request_id" in note for note in notes))

    def test_request_id_membership_test_refutes_hypothesis(self):
        verdict, evidence, _notes = probe.classify_drain(FILTERED_BY_REQUEST_ID)
        self.assertEqual(verdict, probe.REFUTED)
        self.assertTrue(any("Ownership filter present" in line for line in evidence))

    def test_per_caller_output_map_refutes_hypothesis(self):
        verdict, _evidence, _notes = probe.classify_drain(FILTERED_BY_PER_CALLER_MAP)
        self.assertEqual(verdict, probe.REFUTED)

    def test_ownership_filter_wins_over_unfiltered_shape(self):
        # FILTERED_BY_REQUEST_ID also contains `step_outputs` and an
        # `outputs.append(output)`; the filter must still dominate.
        verdict, _evidence, _notes = probe.classify_drain(FILTERED_BY_REQUEST_ID)
        self.assertEqual(verdict, probe.REFUTED)

    def test_missing_source_is_inconclusive_not_confirmed(self):
        for source in (None, "", "   "):
            with self.subTest(source=source):
                verdict, _evidence, _notes = probe.classify_drain(source)
                self.assertEqual(verdict, probe.INCONCLUSIVE)

    def test_unrecognized_shape_is_inconclusive_not_confirmed(self):
        verdict, evidence, _notes = probe.classify_drain(UNRECOGNIZED)
        self.assertEqual(verdict, probe.INCONCLUSIVE)
        self.assertTrue(any("decide by hand" in line for line in evidence))

    def test_exit_codes_are_distinct_per_verdict(self):
        codes = probe._EXIT_CODES
        self.assertEqual(codes[probe.CONFIRMED], 0)
        self.assertEqual(codes[probe.REFUTED], 1)
        self.assertEqual(codes[probe.INCONCLUSIVE], 2)


if __name__ == "__main__":
    unittest.main()
