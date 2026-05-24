import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.services.asr.engines import ASRFullResult, ASRSegmentResult
from app.services.speaker.domain import (
    SpeakerEmbedding,
    SpeakerMatchStatus,
    VoiceprintCandidate,
)
from app.services.speaker.matching import SpeakerMatcher
from app.services.speaker.stores.sqlite_vec_store import SqliteVecVoiceprintStore


class VoiceprintMatchingTest(unittest.TestCase):
    def test_matches_when_top_candidate_passes_threshold_and_margin(self) -> None:
        matcher = SpeakerMatcher(threshold=0.75, margin=0.08)

        decision = matcher.decide(
            local_speaker_id="Speaker1",
            candidates=[
                VoiceprintCandidate(
                    speaker_id="spk1",
                    display_name="Alice",
                    score=0.91,
                    max_score=0.93,
                    top3_mean_score=0.863333,
                    sample_count=3,
                ),
                VoiceprintCandidate(
                    speaker_id="spk2",
                    display_name="Bob",
                    score=0.72,
                    max_score=0.74,
                    top3_mean_score=0.673333,
                    sample_count=3,
                ),
            ],
        )

        self.assertEqual(decision.status, SpeakerMatchStatus.MATCHED)
        self.assertEqual(decision.output_speaker_id, "Alice")

    def test_keeps_local_label_when_margin_is_too_small(self) -> None:
        matcher = SpeakerMatcher(threshold=0.75, margin=0.08)

        decision = matcher.decide(
            local_speaker_id="Speaker1",
            candidates=[
                VoiceprintCandidate(
                    speaker_id="spk1",
                    display_name="Alice",
                    score=0.91,
                    max_score=0.93,
                    top3_mean_score=0.863333,
                    sample_count=3,
                ),
                VoiceprintCandidate(
                    speaker_id="spk2",
                    display_name="Bob",
                    score=0.88,
                    max_score=0.9,
                    top3_mean_score=0.833333,
                    sample_count=3,
                ),
            ],
        )

        self.assertEqual(decision.status, SpeakerMatchStatus.UNKNOWN)
        self.assertEqual(decision.output_speaker_id, "Speaker1")

    def test_asr_segment_schema_is_not_extended(self) -> None:
        result = ASRFullResult(
            text="hello",
            duration=1.0,
            segments=[
                ASRSegmentResult(
                    text="hello",
                    start_time=0.0,
                    end_time=1.0,
                    speaker_id="Speaker1",
                )
            ],
        )

        segment = result.segments[0]
        self.assertEqual(segment.speaker_id, "Speaker1")
        self.assertFalse(hasattr(segment, "speaker_match"))
        self.assertFalse(hasattr(segment, "match_score"))
        self.assertFalse(hasattr(segment, "speaker_identity"))

    def test_sqlite_vec_speaker_score_uses_weighted_max_and_top3_mean(self) -> None:
        self.assertAlmostEqual(
            SqliteVecVoiceprintStore.calculate_speaker_score(
                max_score=0.8,
                top3_mean_score=0.5,
            ),
            0.71,
        )

    def test_sqlite_vec_store_groups_multiple_samples_by_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SqliteVecVoiceprintStore(
                db_path=str(Path(temp_dir) / "voiceprints.sqlite3")
            )
            store.ensure_schema()
            alice = store.create_speaker(
                display_name="Alice",
                description=None,
            )
            bob = store.create_speaker(
                display_name="Bob",
                description=None,
            )

            store.add_voiceprint(
                speaker_id=alice.id,
                embedding=self._embedding([1.0, 0.0, 0.0]),
                source_hash=None,
            )
            store.add_voiceprint(
                speaker_id=alice.id,
                embedding=self._embedding([0.95, 0.05, 0.0]),
                source_hash=None,
            )
            store.add_voiceprint(
                speaker_id=bob.id,
                embedding=self._embedding([0.0, 1.0, 0.0]),
                source_hash=None,
            )

            candidates = store.search(
                embedding=self._embedding([1.0, 0.0, 0.0]),
                limit=2,
            )

        self.assertEqual(candidates[0].display_name, "Alice")
        self.assertEqual(candidates[0].sample_count, 2)
        self.assertGreater(candidates[0].score, candidates[1].score)

    @staticmethod
    def _embedding(prefix: list[float]) -> SpeakerEmbedding:
        vector = np.zeros(192, dtype=np.float32)
        vector[: len(prefix)] = np.array(prefix, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return SpeakerEmbedding(
            vector=vector,
            provider="campplus",
            model_name="test-model",
            sample_rate=16000,
            duration_sec=3.0,
        )


if __name__ == "__main__":
    unittest.main()
