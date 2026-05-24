"""Store protocols for voiceprint persistence."""

from __future__ import annotations

from typing import Protocol

from app.services.speaker.domain import (
    SpeakerEmbedding,
    VoiceprintCandidate,
    VoiceprintRecord,
    VoiceprintSpeaker,
    VoiceprintSpeakerSummary,
)


class VoiceprintStore(Protocol):
    def is_available(self) -> bool:
        ...

    def ensure_schema(self) -> None:
        ...

    def create_speaker(
        self,
        *,
        display_name: str,
        description: str | None,
    ) -> VoiceprintSpeaker:
        ...

    def add_voiceprint(
        self,
        *,
        speaker_id: str,
        embedding: SpeakerEmbedding,
        source_hash: str | None,
    ) -> VoiceprintRecord:
        ...

    def search(
        self,
        *,
        embedding: SpeakerEmbedding,
        limit: int,
    ) -> list[VoiceprintCandidate]:
        ...

    def list_speakers(self) -> list[VoiceprintSpeakerSummary]:
        ...

    def delete_speaker(self, *, speaker_id: str) -> None:
        ...
