# tests/test_diarization_worker.py
"""Worker-side marshalling: modelscope pipeline output (lists containing
numpy scalars) must cross the process boundary as native-typed triples.
Getting this wrong is SILENT (empty result -> per-request VAD fallback),
so types are asserted exactly, through a real pickle round trip."""

import pickle
import unittest

import numpy as np

from app.utils import diarization_worker as dw


class _FakePipeline:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, audio_path):
        self.calls.append(audio_path)
        return self.result


class WorkerDiarizeTest(unittest.TestCase):
    def tearDown(self):
        dw._pipeline = None

    def test_numpy_scalars_become_native_types(self):
        dw._pipeline = _FakePipeline(
            {"text": [[np.float64(0.0), np.float64(1.5), np.int64(0)],
                      [np.float64(1.5), np.float64(3.0), np.int64(1)]]}
        )
        triples = dw._worker_diarize("/tmp/x.wav")
        self.assertEqual(triples, [(0.0, 1.5, 0), (1.5, 3.0, 1)])
        for s, e, label in triples:
            self.assertIs(type(s), float)
            self.assertIs(type(e), float)
            self.assertIs(type(label), int)
        # The contract is "small picklable triples": prove it round-trips.
        self.assertEqual(pickle.loads(pickle.dumps(triples)), triples)

    def test_malformed_segments_are_skipped_not_fatal(self):
        dw._pipeline = _FakePipeline(
            {"text": [[0.0, 1.5, 0], ["bad", "seg"], [1.5, "x", 1], None]}
        )
        self.assertEqual(dw._worker_diarize("/tmp/x.wav"), [(0.0, 1.5, 0)])

    def test_non_dict_result_uses_text_attribute(self):
        class R:
            text = [[0.0, 2.0, 3]]
        dw._pipeline = _FakePipeline(R())
        self.assertEqual(dw._worker_diarize("/tmp/x.wav"), [(0.0, 2.0, 3)])

    def test_use_before_init_raises(self):
        with self.assertRaises(RuntimeError):
            dw._worker_diarize("/tmp/x.wav")

    def test_pipeline_exception_propagates_untouched(self):
        # Throwaway subclass — never mutate _FakePipeline itself, that would
        # poison later tests that instantiate it.
        class _Raising(_FakePipeline):
            def __call__(self, audio_path):
                raise AssertionError(
                    "modelscope error: The effective audio duration is too short."
                )

        dw._pipeline = _Raising(None)
        with self.assertRaises(AssertionError) as ctx:
            dw._worker_diarize("/tmp/x.wav")
        self.assertIn("too short", str(ctx.exception))

    def test_fake_mode_refuses_non_cpu_device(self):
        # DIARIZATION_WORKER_FAKE is a test-only escape hatch: it must never
        # silently fake diarization on a real (cuda) deployment.
        from unittest import mock
        with mock.patch.dict(
            "os.environ", {"DIARIZATION_WORKER_FAKE": "1", "DEVICE": "cuda"}
        ):
            with self.assertRaises(RuntimeError):
                dw._build_pipeline()

    def test_fake_worker_pipeline_shape(self):
        result = dw._FakeWorkerPipeline()("/tmp/x.wav")
        self.assertIn("text", result)
        self.assertEqual(len(result["text"]), 2)


if __name__ == "__main__":
    unittest.main()
