# -*- coding: utf-8 -*-
"""Unit tests for _build_chat_prompt (vLLM chat prompt construction).

Pins the prompt shape after the context wrapper was broadened from
"when resolving named entities" to "when transcribing". Pure-function tests;
no vLLM install required (qwen3_vllm has no module-level vllm import).
"""

import unittest

from app.services.asr.qwen3_vllm import _build_chat_prompt


def _system_text(prompt: str) -> str:
    """Slice the system block out of the full chat-template string."""
    start = prompt.index("<|im_start|>system\n") + len("<|im_start|>system\n")
    return prompt[start:prompt.index("<|im_end|>", start)]


class BuildChatPromptTest(unittest.TestCase):
    def test_context_and_language_yields_both_instructions_language_first(self) -> None:
        prompt = _build_chat_prompt(context="Danantara dana kelolaan", language="Indonesian")

        self.assertEqual(
            _system_text(prompt),
            "Transcribe the speech in Indonesian. "
            "Use this context when transcribing: Danantara dana kelolaan",
        )

    def test_context_without_language_uses_accurately_preamble(self) -> None:
        prompt = _build_chat_prompt(context="阿里巴巴 腾讯")

        self.assertEqual(
            _system_text(prompt),
            "Transcribe the speech accurately. "
            "Use this context when transcribing: 阿里巴巴 腾讯",
        )

    def test_empty_and_whitespace_context_emit_no_context_clause(self) -> None:
        for context in ("", "   ", "\n\t"):
            with self.subTest(context=repr(context)):
                prompt = _build_chat_prompt(context=context)

                self.assertEqual(_system_text(prompt), "Transcribe the speech accurately.")
                self.assertNotIn("Use this context", prompt)

    def test_language_alias_id_normalizes_to_indonesian(self) -> None:
        # Callers pass through _normalize_language_name; mirror that here.
        from app.services.asr.qwen3_vllm import _normalize_language_name

        prompt = _build_chat_prompt(context="x", language=_normalize_language_name("id"))

        self.assertIn("Transcribe the speech in Indonesian.", prompt)

    def test_context_with_surrounding_whitespace_is_stripped(self) -> None:
        # Pins the `{context.strip()}` interpolation itself, not just the
        # empty/whitespace guard: a mutant that drops .strip() from the
        # f-string must fail here.
        prompt = _build_chat_prompt(context="  Danantara \n", language="Indonesian")

        self.assertEqual(
            _system_text(prompt),
            "Transcribe the speech in Indonesian. "
            "Use this context when transcribing: Danantara",
        )

    def test_full_template_scaffold_pins_user_audio_and_assistant_turns(self) -> None:
        # The other tests slice out only the system block, so a mutant that
        # drops the user turn (with the <|audio_pad|> placeholder) or the
        # trailing assistant turn would survive them. Pin the ENTIRE
        # template here so the scaffold cannot silently regress.
        prompt = _build_chat_prompt(context="Danantara", language="Indonesian")

        self.assertEqual(
            prompt,
            "<|im_start|>system\n"
            "Transcribe the speech in Indonesian. "
            "Use this context when transcribing: Danantara<|im_end|>\n"
            "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
            "<|im_start|>assistant\n",
        )

    def test_old_named_entity_wording_is_gone(self) -> None:
        prompt = _build_chat_prompt(context="anything", language="Chinese")

        self.assertNotIn("resolving named entities", prompt)


if __name__ == "__main__":
    unittest.main()
