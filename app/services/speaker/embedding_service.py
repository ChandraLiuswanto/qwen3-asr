"""Speaker embedding extraction and aggregation."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Protocol

import librosa
import numpy as np

from app.services.asr.engines import ASRFullResult

from .config import VOICEPRINT_MIN_SPEECH_SEC
from .domain import SpeakerEmbedding

logger = logging.getLogger(__name__)


class SpeakerEmbeddingProvider(Protocol):
    provider_name: str
    model_name: str
    sample_rate: int

    def extract(self, audio_data: np.ndarray) -> SpeakerEmbedding:
        ...


class SpeakerEmbeddingService:
    """Extract one centroid speaker embedding per local diarization label."""

    def __init__(self, provider: SpeakerEmbeddingProvider) -> None:
        self._provider = provider

    def extract_local_speaker_embeddings(
        self,
        *,
        audio_path: str,
        asr_result: ASRFullResult,
        timestamp_scale: float,
    ) -> dict[str, SpeakerEmbedding]:
        grouped_ranges = self._group_speaker_ranges(
            asr_result=asr_result,
            timestamp_scale=timestamp_scale,
        )
        if not grouped_ranges:
            return {}

        audio_data, loaded_sample_rate = librosa.load(
            audio_path,
            sr=self._provider.sample_rate,
        )
        sample_rate = int(loaded_sample_rate)
        if sample_rate <= 0 or audio_data.size == 0:
            return {}

        embeddings: dict[str, SpeakerEmbedding] = {}
        for speaker_id, ranges in grouped_ranges.items():
            chunks: list[np.ndarray] = []
            for start_sec, end_sec in ranges:
                start_sample = max(0, int(start_sec * sample_rate))
                end_sample = min(audio_data.shape[0], int(end_sec * sample_rate))
                if end_sample > start_sample:
                    chunks.append(audio_data[start_sample:end_sample])

            if not chunks:
                continue

            speaker_audio = np.concatenate(chunks).astype(np.float32, copy=False)
            duration_sec = float(speaker_audio.shape[0] / sample_rate)
            if duration_sec < VOICEPRINT_MIN_SPEECH_SEC:
                logger.info(
                    "Skip voiceprint matching for %s: speech %.2fs < %.2fs",
                    speaker_id,
                    duration_sec,
                    VOICEPRINT_MIN_SPEECH_SEC,
                )
                continue

            embedding = self._provider.extract(speaker_audio)
            embeddings[speaker_id] = SpeakerEmbedding(
                vector=self._normalize(embedding.vector),
                provider=embedding.provider,
                model_name=embedding.model_name,
                sample_rate=sample_rate,
                duration_sec=duration_sec,
            )

        return embeddings

    def extract_reference_embedding(self, *, audio_path: str) -> SpeakerEmbedding:
        audio_data, loaded_sample_rate = librosa.load(
            audio_path,
            sr=self._provider.sample_rate,
        )
        sample_rate = int(loaded_sample_rate)
        embedding = self._provider.extract(audio_data.astype(np.float32, copy=False))
        return SpeakerEmbedding(
            vector=self._normalize(embedding.vector),
            provider=embedding.provider,
            model_name=embedding.model_name,
            sample_rate=sample_rate,
            duration_sec=float(audio_data.shape[0] / sample_rate) if sample_rate else 0.0,
        )

    @staticmethod
    def _group_speaker_ranges(
        *,
        asr_result: ASRFullResult,
        timestamp_scale: float,
    ) -> dict[str, list[tuple[float, float]]]:
        scale = timestamp_scale if timestamp_scale > 0 else 1.0
        ranges: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for segment in asr_result.segments:
            if not segment.speaker_id:
                continue
            start_sec = max(0.0, float(segment.start_time) / scale)
            end_sec = max(start_sec, float(segment.end_time) / scale)
            if end_sec > start_sec:
                ranges[segment.speaker_id].append((start_sec, end_sec))
        return dict(ranges)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(value))
        if norm <= 0:
            return value
        return value / norm
