"""SQLite sqlite-vec store for speaker voiceprints."""

from __future__ import annotations

import sqlite3
import uuid
from collections import defaultdict
from pathlib import Path

import sqlite_vec

from app.core.config import settings
from app.services.speaker.domain import (
    SpeakerEmbedding,
    VoiceprintCandidate,
    VoiceprintRecord,
    VoiceprintSpeaker,
    VoiceprintSpeakerSummary,
)


class SqliteVecVoiceprintStore:
    embedding_dim = 192
    max_score_weight = 0.7
    top3_mean_score_weight = 0.3

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or settings.VOICEPRINT_DB_PATH

    def is_available(self) -> bool:
        return bool(self._db_path)

    def ensure_schema(self) -> None:
        if not self._db_path:
            return

        db_file = Path(self._db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS voiceprint_speakers (
                    id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS voiceprints (
                    vector_rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    speaker_id TEXT NOT NULL REFERENCES voiceprint_speakers(id),
                    provider TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    duration_sec REAL NOT NULL,
                    sample_rate INTEGER NOT NULL DEFAULT 16000,
                    source_hash TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS voiceprint_vectors
                USING vec0(
                    embedding float[{self.embedding_dim}]
                    distance_metric=cosine
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS voiceprints_lookup_idx
                ON voiceprints(provider, model_name, speaker_id)
                """
            )
            conn.commit()

    def create_speaker(
        self,
        *,
        display_name: str,
        description: str | None,
    ) -> VoiceprintSpeaker:
        speaker_id = str(uuid.uuid4())
        with self._connect() as conn:
            row = conn.execute(
                """
                INSERT INTO voiceprint_speakers (id, display_name, description)
                VALUES (?, ?, ?)
                RETURNING id, display_name, description
                """,
                (speaker_id, display_name, description),
            ).fetchone()
            conn.commit()

        if row is None:
            raise RuntimeError("failed to create voiceprint speaker")
        return VoiceprintSpeaker(
            id=str(row["id"]),
            display_name=str(row["display_name"]),
            description=row["description"],
        )

    def add_voiceprint(
        self,
        *,
        speaker_id: str,
        embedding: SpeakerEmbedding,
        source_hash: str | None,
    ) -> VoiceprintRecord:
        self._validate_embedding_dim(embedding)
        voiceprint_id = str(uuid.uuid4())
        vector_blob = self._serialize_vector(embedding)

        with self._connect() as conn:
            row = conn.execute(
                """
                INSERT INTO voiceprints (
                    id,
                    speaker_id,
                    provider,
                    model_name,
                    duration_sec,
                    sample_rate,
                    source_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                RETURNING
                    vector_rowid,
                    id,
                    speaker_id,
                    provider,
                    model_name,
                    duration_sec
                """,
                (
                    voiceprint_id,
                    speaker_id,
                    embedding.provider,
                    embedding.model_name,
                    embedding.duration_sec,
                    embedding.sample_rate,
                    source_hash,
                ),
            ).fetchone()
            if row is None:
                raise RuntimeError("failed to add voiceprint")

            conn.execute(
                """
                INSERT INTO voiceprint_vectors(rowid, embedding)
                VALUES (?, ?)
                """,
                (int(row["vector_rowid"]), vector_blob),
            )
            speaker_row = conn.execute(
                """
                SELECT display_name
                FROM voiceprint_speakers
                WHERE id = ?
                """,
                (speaker_id,),
            ).fetchone()
            conn.commit()

        if speaker_row is None:
            raise RuntimeError("failed to add voiceprint")
        return VoiceprintRecord(
            id=str(row["id"]),
            speaker_id=str(row["speaker_id"]),
            display_name=str(speaker_row["display_name"]),
            provider=str(row["provider"]),
            model_name=str(row["model_name"]),
            duration_sec=float(row["duration_sec"]),
        )

    def search(
        self,
        *,
        embedding: SpeakerEmbedding,
        limit: int,
    ) -> list[VoiceprintCandidate]:
        if not self._db_path:
            return []

        self._validate_embedding_dim(embedding)
        with self._connect() as conn:
            matching_sample_count = conn.execute(
                """
                SELECT count(*)
                FROM voiceprints v
                JOIN voiceprint_speakers s ON s.id = v.speaker_id
                WHERE s.status = 'active'
                  AND v.provider = ?
                  AND v.model_name = ?
                """,
                (embedding.provider, embedding.model_name),
            ).fetchone()[0]
            if matching_sample_count == 0:
                return []

            total_vector_count = conn.execute(
                "SELECT count(*) FROM voiceprint_vectors"
            ).fetchone()[0]

            rows = conn.execute(
                """
                WITH nearest AS (
                    SELECT rowid, distance
                    FROM voiceprint_vectors
                    WHERE embedding MATCH ?
                    ORDER BY distance
                    LIMIT ?
                )
                SELECT
                    s.id AS speaker_id,
                    s.display_name,
                    1.0 - nearest.distance AS sample_score
                FROM nearest
                JOIN voiceprints v ON v.vector_rowid = nearest.rowid
                JOIN voiceprint_speakers s ON s.id = v.speaker_id
                WHERE s.status = 'active'
                  AND v.provider = ?
                  AND v.model_name = ?
                """,
                (
                    self._serialize_vector(embedding),
                    int(total_vector_count),
                    embedding.provider,
                    embedding.model_name,
                ),
            ).fetchall()

        speaker_scores: dict[str, list[float]] = defaultdict(list)
        display_names: dict[str, str] = {}
        for row in rows:
            speaker_id = str(row["speaker_id"])
            display_names[speaker_id] = str(row["display_name"])
            speaker_scores[speaker_id].append(float(row["sample_score"]))

        candidates = [
            self._build_candidate(
                speaker_id=speaker_id,
                display_name=display_names[speaker_id],
                sample_scores=sample_scores,
            )
            for speaker_id, sample_scores in speaker_scores.items()
        ]
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[: max(1, limit)]

    def list_speakers(self) -> list[VoiceprintSpeakerSummary]:
        if not self._db_path:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    s.id,
                    s.display_name,
                    s.description,
                    count(v.id) AS voiceprint_count
                FROM voiceprint_speakers s
                LEFT JOIN voiceprints v ON v.speaker_id = s.id
                WHERE s.status = 'active'
                GROUP BY s.id, s.display_name, s.description, s.created_at
                ORDER BY s.created_at DESC
                """
            ).fetchall()
        return [
            VoiceprintSpeakerSummary(
                id=str(row["id"]),
                display_name=str(row["display_name"]),
                description=row["description"],
                voiceprint_count=int(row["voiceprint_count"]),
            )
            for row in rows
        ]

    def delete_speaker(self, *, speaker_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE voiceprint_speakers
                SET status = 'deleted', updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (speaker_id,),
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        if not self._db_path:
            raise RuntimeError("VOICEPRINT_DB_PATH is not configured")

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _validate_embedding_dim(self, embedding: SpeakerEmbedding) -> None:
        actual_dim = int(embedding.vector.reshape(-1).shape[0])
        if actual_dim != self.embedding_dim:
            raise ValueError(
                "voiceprint embedding dimension mismatch: "
                f"expected={self.embedding_dim}, actual={actual_dim}"
            )

    @staticmethod
    def _serialize_vector(embedding: SpeakerEmbedding) -> bytes:
        return sqlite_vec.serialize_float32(
            [float(value) for value in embedding.vector.reshape(-1).tolist()]
        )

    @classmethod
    def _build_candidate(
        cls,
        *,
        speaker_id: str,
        display_name: str,
        sample_scores: list[float],
    ) -> VoiceprintCandidate:
        sorted_scores = sorted(sample_scores, reverse=True)
        max_score = sorted_scores[0]
        top3_scores = sorted_scores[:3]
        top3_mean_score = sum(top3_scores) / len(top3_scores)
        return VoiceprintCandidate(
            speaker_id=speaker_id,
            display_name=display_name,
            score=cls.calculate_speaker_score(
                max_score=max_score,
                top3_mean_score=top3_mean_score,
            ),
            max_score=max_score,
            top3_mean_score=top3_mean_score,
            sample_count=len(sample_scores),
        )

    @classmethod
    def calculate_speaker_score(
        cls,
        *,
        max_score: float,
        top3_mean_score: float,
    ) -> float:
        return (
            max_score * cls.max_score_weight
            + top3_mean_score * cls.top3_mean_score_weight
        )
