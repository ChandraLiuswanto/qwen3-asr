"""Voiceprint match decision policy."""

from __future__ import annotations

from dataclasses import dataclass

from .domain import SpeakerMatchDecision, SpeakerMatchStatus, VoiceprintCandidate


@dataclass(frozen=True)
class SpeakerMatcher:
    threshold: float
    margin: float

    def decide(
        self,
        *,
        local_speaker_id: str,
        candidates: list[VoiceprintCandidate],
    ) -> SpeakerMatchDecision:
        if not candidates:
            return SpeakerMatchDecision(
                status=SpeakerMatchStatus.UNKNOWN,
                local_speaker_id=local_speaker_id,
            )

        sorted_candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
        top1 = sorted_candidates[0]
        top2_score = sorted_candidates[1].score if len(sorted_candidates) > 1 else None

        if top1.score < self.threshold:
            return SpeakerMatchDecision(
                status=SpeakerMatchStatus.UNKNOWN,
                local_speaker_id=local_speaker_id,
                score=top1.score,
            )

        if top2_score is not None and top1.score - top2_score < self.margin:
            return SpeakerMatchDecision(
                status=SpeakerMatchStatus.UNKNOWN,
                local_speaker_id=local_speaker_id,
                score=top1.score,
            )

        return SpeakerMatchDecision(
            status=SpeakerMatchStatus.MATCHED,
            local_speaker_id=local_speaker_id,
            display_name=top1.display_name,
            score=top1.score,
        )
