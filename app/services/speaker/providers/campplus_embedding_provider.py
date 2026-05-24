"""CAM++ speaker embedding provider."""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from app.core.config import settings
from app.core.device import detect_device
from app.infrastructure.model_utils import resolve_model_path
from app.services.speaker.domain import SpeakerEmbedding

logger = logging.getLogger(__name__)


class CampplusEmbeddingProvider:
    provider_name = "campplus"
    model_name = "damo/speech_campplus_sv_zh-cn_16k-common"
    sample_rate = 16000

    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._lock = threading.Lock()

    def extract(self, audio_data: np.ndarray) -> SpeakerEmbedding:
        pipeline = self._get_pipeline()
        output = pipeline([audio_data], output_emb=True)
        raw_embedding = self._extract_embedding_from_output(output)
        return SpeakerEmbedding(
            vector=raw_embedding,
            provider=self.provider_name,
            model_name=self.model_name,
            sample_rate=self.sample_rate,
            duration_sec=float(audio_data.shape[0] / self.sample_rate),
        )

    def _get_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks

            model_path = resolve_model_path(self.model_name)
            device = detect_device(settings.DEVICE)
            logger.info(
                "Loading CAM++ speaker embedding provider: model=%s, device=%s",
                model_path,
                device,
            )
            self._pipeline = pipeline(
                task=Tasks.speaker_verification,
                model=model_path,
                device=device,
            )
            model = getattr(self._pipeline, "model", None)
            if model is not None and hasattr(model, "to"):
                self._pipeline.model = model.to(device)
            return self._pipeline

    @staticmethod
    def _extract_embedding_from_output(output: Any) -> np.ndarray:
        if isinstance(output, dict) and "embs" in output:
            embedding = np.asarray(output["embs"], dtype=np.float32)
            if embedding.ndim == 2 and embedding.shape[0] >= 1:
                return embedding[0]
            return embedding.reshape(-1)
        raise RuntimeError("CAM++ embedding output does not contain 'embs'")

