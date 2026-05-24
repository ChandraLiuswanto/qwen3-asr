"""Domain objects for persistent speaker identification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class SpeakerMatchStatus(str, Enum):
    MATCHED = "matched"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SpeakerEmbedding:
    vector: np.ndarray
    provider: str
    model_name: str
    sample_rate: int
    duration_sec: float


@dataclass(frozen=True)
class VoiceprintCandidate:
    speaker_id: str
    display_name: str
    score: float
    max_score: float
    top3_mean_score: float
    sample_count: int


@dataclass(frozen=True)
class SpeakerMatchDecision:
    status: SpeakerMatchStatus
    local_speaker_id: str
    display_name: Optional[str] = None
    score: Optional[float] = None

    @property
    def output_speaker_id(self) -> str:
        if self.status == SpeakerMatchStatus.MATCHED and self.display_name:
            return self.display_name
        return self.local_speaker_id


@dataclass(frozen=True)
class VoiceprintSpeaker:
    id: str
    display_name: str
    description: Optional[str] = None


@dataclass(frozen=True)
class VoiceprintRecord:
    id: str
    speaker_id: str
    display_name: str
    provider: str
    model_name: str
    duration_sec: float


@dataclass(frozen=True)
class VoiceprintSpeakerSummary:
    id: str
    display_name: str
    description: Optional[str]
    voiceprint_count: int
