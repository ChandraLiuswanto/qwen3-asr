# -*- coding: utf-8 -*-
"""Tests for loading the model-shipped chat_template.json.

The template is a load-bearing runtime dependency: the tokenizer never loads
chat_template.json itself (it is a processor-level file), so the backend must
read it from the model snapshot at init and fail loudly if it is absent —
silently falling back to a hand-written template is the bug this change removes.
"""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from app.infrastructure import find_huggingface_snapshot_dir
from app.services.asr.qwen3_vllm import _load_chat_template

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "qwen3_asr"
VENDORED_SHA256 = "75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff"


def _find_real_snapshot() -> Path | None:
    """A real model snapshot, when one exists on this machine."""
    override = os.environ.get("QWEN3_ASR_SNAPSHOT_DIR", "").strip()
    if override and (Path(override) / "chat_template.json").is_file():
        return Path(override)
    for ref in ("Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-0.6B"):
        snapshot = find_huggingface_snapshot_dir(ref)
        if snapshot is not None and (snapshot / "chat_template.json").is_file():
            return snapshot
    return None


class LoadChatTemplateTest(unittest.TestCase):
    def test_loads_template_string_from_snapshot_dir(self) -> None:
        template = _load_chat_template(FIXTURE_DIR)
        self.assertIsInstance(template, str)
        self.assertIn("<|audio_start|><|audio_pad|><|audio_end|>", template)
        self.assertIn("add_generation_prompt", template)

    def test_missing_file_raises_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as ctx:
                _load_chat_template(Path(tmp))
        self.assertIn("chat_template.json", str(ctx.exception))

    def test_malformed_json_raises_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "chat_template.json").write_text("not json", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _load_chat_template(Path(tmp))

    def test_missing_key_raises_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "chat_template.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _load_chat_template(Path(tmp))


class VendoredTemplateGuardTest(unittest.TestCase):
    """The fixture is a copy of the model-shipped file. A stale copy would
    silently re-introduce template drift, so it is guarded twice."""

    def test_vendored_file_sha256_is_pinned(self) -> None:
        # Local-tamper guard: any edit to the vendored file fails loudly.
        # If upstream legitimately ships a new template, update the fixture
        # AND this hash AND re-verify the skeleton tests in the same commit.
        digest = hashlib.sha256(
            (FIXTURE_DIR / "chat_template.json").read_bytes()
        ).hexdigest()
        self.assertEqual(digest, VENDORED_SHA256)

    def test_vendored_template_matches_model_snapshot(self) -> None:
        # Upstream-drift guard. Skips where no model exists (the dev box);
        # EXECUTES in the container quality gate (plan Task 5), which is the
        # environment whose verdict gates the merge. Never delete the skip
        # message: a skip here means drift is unguarded on this machine.
        snapshot = _find_real_snapshot()
        if snapshot is None:
            self.skipTest(
                "no Qwen3-ASR snapshot on this machine; template drift is NOT "
                "guarded here — the container run (merge gate) executes this"
            )
        vendored = json.loads(
            (FIXTURE_DIR / "chat_template.json").read_text(encoding="utf-8")
        )["chat_template"]
        real = json.loads(
            (snapshot / "chat_template.json").read_text(encoding="utf-8")
        )["chat_template"]
        self.assertEqual(
            vendored,
            real,
            "vendored tests/fixtures/qwen3_asr/chat_template.json has drifted "
            "from the model snapshot — update the fixture and re-verify the "
            "prompt-shape tests",
        )


if __name__ == "__main__":
    unittest.main()
