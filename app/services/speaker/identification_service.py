"""Persistent speaker identification orchestration."""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from typing import Optional, Sequence

from app.core.config import settings
from app.services.asr.engines import ASRFullResult

from .config import (
    VOICEPRINT_AUTO_CREATE_SCHEMA,
    VOICEPRINT_MATCH_MARGIN,
    VOICEPRINT_PROVIDER,
    VOICEPRINT_TOP_K,
)
from .domain import (
    SpeakerMatchDecision,
    SpeakerMatchStatus,
    VoiceprintSpeaker,
    VoiceprintSpeakerSummary,
)
from .embedding_service import SpeakerEmbeddingService
from .matching import SpeakerMatcher
from .providers import CampplusEmbeddingProvider
from .stores import SqliteVecVoiceprintStore
from .stores.base import VoiceprintStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceprintRegistrationResult:
    speaker: VoiceprintSpeaker
    voiceprint_ids: list[str]

    @property
    def voiceprint_id(self) -> str:
        return self.voiceprint_ids[0]


@dataclass(frozen=True)
class VoiceprintSampleSource:
    audio_path: str
    source_bytes: bytes | None = None


class SpeakerIdentificationService:
    def __init__(
        self,
        *,
        embedding_service: SpeakerEmbeddingService,
        store: VoiceprintStore,
        matcher: SpeakerMatcher,
    ) -> None:
        self._embedding_service = embedding_service
        self._store = store
        self._matcher = matcher
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    def enrich_asr_result(
        self,
        *,
        audio_path: str,
        asr_result: ASRFullResult,
        timestamp_scale: float,
        task_id: Optional[str] = None,
    ) -> ASRFullResult:
        if not self._is_runtime_enabled():
            return asr_result
        if not self._store.is_available():
            logger.info("Voiceprint database is not configured; keep diarization labels")
            return asr_result

        try:
            self._ensure_schema()
            embeddings = self._embedding_service.extract_local_speaker_embeddings(
                audio_path=audio_path,
                asr_result=asr_result,
                timestamp_scale=timestamp_scale,
            )
            if not embeddings:
                return asr_result

            decisions: dict[str, SpeakerMatchDecision] = {}
            for local_speaker_id, embedding in embeddings.items():
                candidates = self._store.search(
                    embedding=embedding,
                    limit=VOICEPRINT_TOP_K,
                )
                match_decision = self._matcher.decide(
                    local_speaker_id=local_speaker_id,
                    candidates=candidates,
                )
                decisions[local_speaker_id] = match_decision

            for segment in asr_result.segments:
                if not segment.speaker_id:
                    continue
                segment_decision = decisions.get(segment.speaker_id)
                if (
                    segment_decision
                    and segment_decision.status == SpeakerMatchStatus.MATCHED
                ):
                    segment.speaker_id = segment_decision.output_speaker_id

            self._log_decisions(decisions=decisions, task_id=task_id)
            return asr_result
        except Exception as exc:
            logger.warning(
                "Voiceprint enrichment failed; keep diarization labels: %s",
                exc,
            )
            return asr_result

    def register_speaker(
        self,
        *,
        display_name: str,
        description: str | None,
        audio_path: str,
        source_bytes: bytes | None = None,
    ) -> VoiceprintRegistrationResult:
        return self.register_speaker_samples(
            display_name=display_name,
            description=description,
            samples=(VoiceprintSampleSource(audio_path, source_bytes),),
        )

    def register_speaker_samples(
        self,
        *,
        display_name: str,
        description: str | None,
        samples: Sequence[VoiceprintSampleSource],
    ) -> VoiceprintRegistrationResult:
        if not self._store.is_available():
            raise RuntimeError("VOICEPRINT_DB_PATH is not configured")
        if not samples:
            raise ValueError("at least one voiceprint sample is required")

        self._ensure_schema()
        speaker = self._store.create_speaker(
            display_name=display_name,
            description=description,
        )
        voiceprint_ids = self._add_samples_to_speaker(
            speaker_id=speaker.id,
            samples=samples,
        )
        return VoiceprintRegistrationResult(
            speaker=speaker,
            voiceprint_ids=voiceprint_ids,
        )

    def add_speaker_samples(
        self,
        *,
        speaker_id: str,
        samples: Sequence[VoiceprintSampleSource],
    ) -> list[str]:
        if not self._store.is_available():
            raise RuntimeError("VOICEPRINT_DB_PATH is not configured")
        if not samples:
            raise ValueError("at least one voiceprint sample is required")

        self._ensure_schema()
        return self._add_samples_to_speaker(
            speaker_id=speaker_id,
            samples=samples,
        )

    def _add_samples_to_speaker(
        self,
        *,
        speaker_id: str,
        samples: Sequence[VoiceprintSampleSource],
    ) -> list[str]:
        voiceprint_ids: list[str] = []
        for sample in samples:
            source_hash = (
                hashlib.sha256(sample.source_bytes).hexdigest()
                if sample.source_bytes is not None
                else None
            )
            embedding = self._embedding_service.extract_reference_embedding(
                audio_path=sample.audio_path,
            )
            record = self._store.add_voiceprint(
                speaker_id=speaker_id,
                embedding=embedding,
                source_hash=source_hash,
            )
            voiceprint_ids.append(record.id)
        return voiceprint_ids

    def list_speakers(self) -> list[VoiceprintSpeakerSummary]:
        if not self._store.is_available():
            return []
        self._ensure_schema()
        return self._store.list_speakers()

    def delete_speaker(self, *, speaker_id: str) -> None:
        if not self._store.is_available():
            raise RuntimeError("VOICEPRINT_DB_PATH is not configured")
        self._ensure_schema()
        self._store.delete_speaker(speaker_id=speaker_id)

    def _ensure_schema(self) -> None:
        if not VOICEPRINT_AUTO_CREATE_SCHEMA:
            return
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            self._store.ensure_schema()
            self._schema_ready = True

    @staticmethod
    def _is_runtime_enabled() -> bool:
        return settings.VOICEPRINT_ENABLED

    @staticmethod
    def _log_decisions(
        *,
        decisions: dict[str, SpeakerMatchDecision],
        task_id: Optional[str],
    ) -> None:
        if not decisions:
            return
        prefix = f"[{task_id}] " if task_id else ""
        for local_speaker_id, decision in decisions.items():
            logger.info(
                "%sVoiceprint decision: local=%s, output=%s, status=%s",
                prefix,
                local_speaker_id,
                decision.output_speaker_id,
                decision.status.value,
            )


_speaker_identification_service: SpeakerIdentificationService | None = None
_speaker_identification_lock = threading.Lock()


def get_speaker_identification_service() -> SpeakerIdentificationService:
    global _speaker_identification_service
    if _speaker_identification_service is not None:
        return _speaker_identification_service

    with _speaker_identification_lock:
        if _speaker_identification_service is None:
            provider_name = VOICEPRINT_PROVIDER.strip().lower()
            if provider_name != CampplusEmbeddingProvider.provider_name:
                raise RuntimeError(f"unsupported voiceprint provider: {provider_name}")
            provider = CampplusEmbeddingProvider()
            _speaker_identification_service = SpeakerIdentificationService(
                embedding_service=SpeakerEmbeddingService(provider),
                store=SqliteVecVoiceprintStore(),
                matcher=SpeakerMatcher(
                    threshold=settings.VOICEPRINT_MATCH_THRESHOLD,
                    margin=VOICEPRINT_MATCH_MARGIN,
                ),
            )
    return _speaker_identification_service
