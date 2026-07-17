# -*- coding: utf-8 -*-
"""Tests for _normalize_language_name (alias mapping + validation).

Upstream Qwen3-ASR validates language against a canonical 30-name set and
raises on anything else (qwen_asr/inference/utils.py:105). Our alias layer
(zh -> Chinese, id -> Indonesian, ...) is strictly friendlier and feeds the
same canonical values; anything that does not land in the canonical set must
raise rather than be injected into the assistant prefill as a guess.
"""

import unittest

from app.services.asr.qwen3_vllm import _SUPPORTED_LANGUAGES, _normalize_language_name


class NormalizeLanguageNameTest(unittest.TestCase):
    def test_none_and_blank_pass_through_as_none(self) -> None:
        for value in (None, "", "   "):
            with self.subTest(value=repr(value)):
                self.assertIsNone(_normalize_language_name(value))

    def test_iso_aliases_map_to_canonical_names(self) -> None:
        cases = {
            "id": "Indonesian",
            "zh": "Chinese",
            "zh-CN": "Chinese",
            "en": "English",
            "yue": "Cantonese",
            "ja": "Japanese",
        }
        for alias, canonical in cases.items():
            with self.subTest(alias=alias):
                self.assertEqual(_normalize_language_name(alias), canonical)

    def test_full_names_normalize_case(self) -> None:
        self.assertEqual(_normalize_language_name("indonesian"), "Indonesian")
        self.assertEqual(_normalize_language_name("CHINESE"), "Chinese")

    def test_unsupported_language_raises(self) -> None:
        for bad in ("tl", "Klingon", "xx-YY"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as ctx:
                    _normalize_language_name(bad)
                self.assertIn(bad, str(ctx.exception))

    def test_supported_set_is_upstreams_thirty(self) -> None:
        # Pinned to qwen_asr/inference/utils.py:37 (SUPPORTED_LANGUAGES).
        self.assertEqual(len(_SUPPORTED_LANGUAGES), 30)
        self.assertIn("Indonesian", _SUPPORTED_LANGUAGES)
        self.assertIn("Cantonese", _SUPPORTED_LANGUAGES)

    def test_every_alias_lands_in_the_supported_set(self) -> None:
        from app.services.asr.qwen3_vllm import _LANGUAGE_ALIASES

        for alias, canonical in _LANGUAGE_ALIASES.items():
            with self.subTest(alias=alias):
                self.assertIn(canonical, _SUPPORTED_LANGUAGES)


if __name__ == "__main__":
    unittest.main()
