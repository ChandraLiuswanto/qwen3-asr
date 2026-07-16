# -*- coding: utf-8 -*-
"""Unit tests for the PURE logic of the H100 offline-mixing script.

SCOPE — READ THIS FIRST
-----------------------
These tests cover exactly three pure functions: `parse_case` (filename ->
keyword), `check_integrity` (response text -> verdict) and the
`summarize`/`is_contaminated` counting. They take strings and return verdicts.

WHAT THEY DO NOT COVER — and cannot:
  * Whether the real service mixes concurrent offline requests. That is the
    entire point of the H100 script and it requires vLLM on a CUDA box.
  * Whether `_llm_lock` fixes anything.
  * Whether the HTTP client, the concurrency, or the multipart encoding work.

A green run here means "the contamination DETECTOR does the arithmetic we
intend", not "the service is correct". Faking the concurrency to make a local
test go green is the exact failure mode this design is built to avoid; there
is deliberately no such test here.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "h100" / "test_offline_mixing.py"
_spec = importlib.util.spec_from_file_location("test_offline_mixing", _SCRIPT)
assert _spec is not None and _spec.loader is not None
mixing = importlib.util.module_from_spec(_spec)
# Register before exec: @dataclass resolves its module from sys.modules.
sys.modules[_spec.name] = mixing
_spec.loader.exec_module(mixing)


class ParseCaseTests(unittest.TestCase):
    def test_parses_label_and_keyword(self):
        case = mixing.parse_case("/audio/alpha_银行.wav")
        self.assertEqual(case.label, "alpha")
        self.assertEqual(case.keyword, "银行")
        self.assertEqual(case.path, "/audio/alpha_银行.wav")

    def test_splits_on_last_underscore_so_labels_may_contain_underscores(self):
        case = mixing.parse_case("/audio/long_form_alpha_银行.mp3")
        self.assertEqual(case.label, "long_form_alpha")
        self.assertEqual(case.keyword, "银行")

    def test_filename_without_underscore_is_rejected_not_guessed(self):
        self.assertIsNone(mixing.parse_case("/audio/alpha.wav"))

    def test_empty_label_or_keyword_is_rejected(self):
        self.assertIsNone(mixing.parse_case("/audio/_银行.wav"))
        self.assertIsNone(mixing.parse_case("/audio/alpha_.wav"))


class CheckIntegrityTests(unittest.TestCase):
    def test_own_keyword_only_is_ok(self):
        verdict, foreign = mixing.check_integrity("我去了银行", "银行", ["机场", "医院"])
        self.assertEqual(verdict, mixing.OK)
        self.assertEqual(foreign, [])

    def test_foreign_keyword_alongside_own_is_the_mixing_signature(self):
        verdict, foreign = mixing.check_integrity("我去了银行和机场", "银行", ["机场", "医院"])
        self.assertEqual(verdict, mixing.FOREIGN)
        self.assertEqual(foreign, ["机场"])

    def test_foreign_keyword_replacing_own_is_the_clearest_signature(self):
        # The shape a truncating `zip` produces: request B's transcript served
        # as request A's answer.
        verdict, foreign = mixing.check_integrity("我去了机场", "银行", ["机场", "医院"])
        self.assertEqual(verdict, mixing.BOTH)
        self.assertEqual(foreign, ["机场"])

    def test_missing_own_with_no_foreign_is_not_reported_as_mixing(self):
        # A bad transcript is not evidence of a race. It must be distinguishable.
        verdict, foreign = mixing.check_integrity("完全无关的文本", "银行", ["机场"])
        self.assertEqual(verdict, mixing.MISSING_OWN)
        self.assertEqual(foreign, [])

    def test_empty_response_is_missing_own_not_ok(self):
        verdict, _ = mixing.check_integrity("", "银行", ["机场"])
        self.assertEqual(verdict, mixing.MISSING_OWN)

    def test_own_keyword_is_never_counted_as_foreign_even_if_listed(self):
        verdict, foreign = mixing.check_integrity("我去了银行", "银行", ["银行", "机场"])
        self.assertEqual(verdict, mixing.OK)
        self.assertEqual(foreign, [])

    def test_multiple_foreign_keywords_are_all_reported(self):
        verdict, foreign = mixing.check_integrity("机场医院", "银行", ["机场", "医院"])
        self.assertEqual(verdict, mixing.BOTH)
        self.assertEqual(sorted(foreign), sorted(["机场", "医院"]))


class SummaryTests(unittest.TestCase):
    def test_contamination_is_flagged_for_foreign_and_both(self):
        self.assertTrue(mixing.is_contaminated(mixing.summarize([mixing.OK, mixing.FOREIGN])))
        self.assertTrue(mixing.is_contaminated(mixing.summarize([mixing.OK, mixing.BOTH])))

    def test_missing_own_alone_is_not_contamination(self):
        # Deliberate: MISSING_OWN is reported and fails the run, but it is NOT
        # claimed to be the mixing bug. Conflating them would let a bad audio
        # fixture masquerade as a detected race.
        counts = mixing.summarize([mixing.OK, mixing.MISSING_OWN])
        self.assertFalse(mixing.is_contaminated(counts))

    def test_all_ok_is_not_contamination(self):
        self.assertFalse(mixing.is_contaminated(mixing.summarize([mixing.OK, mixing.OK])))

    def test_summarize_counts_every_kind(self):
        counts = mixing.summarize([mixing.OK, mixing.OK, mixing.FOREIGN, mixing.MISSING_OWN])
        self.assertEqual(counts[mixing.OK], 2)
        self.assertEqual(counts[mixing.FOREIGN], 1)
        self.assertEqual(counts[mixing.MISSING_OWN], 1)
        self.assertEqual(counts[mixing.BOTH], 0)


class ValidateCasesTests(unittest.TestCase):
    """The fixture guard: overlapping keywords would produce false positives."""

    def _case(self, label, keyword):
        return mixing.AudioCase(label=label, keyword=keyword, path=f"/audio/{label}_{keyword}.wav")

    def test_duplicate_keywords_are_rejected(self):
        cases = [self._case("a", "银行"), self._case("b", "银行")]
        with self.assertRaises(SystemExit):
            mixing.validate_cases(cases, 2)

    def test_substring_keywords_are_rejected(self):
        # "银行" inside "银行卡" would look like contamination on every run.
        cases = [self._case("a", "银行"), self._case("b", "银行卡")]
        with self.assertRaises(SystemExit):
            mixing.validate_cases(cases, 2)

    def test_fewer_than_two_cases_is_rejected(self):
        with self.assertRaises(SystemExit):
            mixing.validate_cases([self._case("a", "银行")], 1)

    def test_distinct_keywords_pass(self):
        cases = [self._case("a", "银行"), self._case("b", "机场")]
        mixing.validate_cases(cases, 2)  # must not raise


if __name__ == "__main__":
    unittest.main()
