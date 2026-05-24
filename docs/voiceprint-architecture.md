# Voiceprint Database Architecture

## Goal

`v1.0.2` adds persistent speaker identity matching without changing the ASR
response schema. The ASR engine still returns local diarization labels such as
`Speaker1` or `说话人1`. When voiceprint matching is reliable, the service
replaces the existing `speaker_id` value with the registered speaker display
name. When matching is uncertain, the local diarization label is preserved.

## Runtime Contract

Voiceprint matching is controlled by deployment configuration only.

```text
VOICEPRINT_ENABLED=true
VOICEPRINT_DB_PATH=./data/voiceprints.sqlite3
VOICEPRINT_MATCH_THRESHOLD=0.70
```

ASR requests do not include a voiceprint enable flag, and ASR responses do not
include match scores, voiceprint ids, or extra speaker identity fields.

## Storage

The project uses SQLite plus `sqlite-vec` as the only voiceprint store. Metadata
is stored in normal SQLite tables, and embeddings are stored in a `vec0` virtual
table.

```text
voiceprint_speakers
  id
  display_name
  description
  status

voiceprints
  vector_rowid
  id
  speaker_id
  provider
  model_name
  duration_sec
  sample_rate
  source_hash

voiceprint_vectors
  rowid
  embedding float[192]
```

Docker Compose mounts `./data` to `/app/data`, so the default
`./data/voiceprints.sqlite3` path is persistent in container deployments and
also works for local runs from the project root.

## Module Boundaries

```text
app/services/speaker/
  domain.py
  embedding_service.py
  identification_service.py
  matching.py
  providers/campplus_embedding_provider.py
  stores/base.py
  stores/sqlite_vec_store.py
```

- `domain.py`: framework-independent dataclasses and enums.
- `embedding_service.py`: audio loading, diarized segment aggregation, and
  embedding normalization.
- `identification_service.py`: registration and ASR enrichment orchestration.
- `matching.py`: threshold and margin based match decision.
- `campplus_embedding_provider.py`: CAM++ embedding provider adapter.
- `sqlite_vec_store.py`: SQLite metadata and sqlite-vec vector search.

## Matching Policy

A speaker may have multiple registered samples. Matching aggregates sample
similarities by persistent speaker:

```text
speaker_score = max_sample_score * 0.7 + top3_sample_mean_score * 0.3
```

The final decision is conservative:

```text
matched:
  top1_speaker_score >= VOICEPRINT_MATCH_THRESHOLD
  and top1_speaker_score - top2_speaker_score >= internal_match_margin

unknown:
  no candidate
  or score below threshold
  or top candidates are too close
```

Scores are internal diagnostics only and are not returned to ASR callers.

## API

Create a speaker and register one or more single-speaker samples:

```bash
curl -X POST 'http://localhost:8000/api/v1/voiceprint-speakers' \
  -F 'display_name=Alice' \
  -F 'file=@tests/files/voiceprint_samples/dialogue_speaker_01_reference.wav'
```

Add samples to an existing speaker:

```bash
curl -X POST 'http://localhost:8000/api/v1/voiceprint-speakers/{speaker_id}/samples' \
  -F 'file=@tests/files/voiceprint_samples/dialogue_speaker_02_reference.wav'
```

List speakers:

```bash
curl 'http://localhost:8000/api/v1/voiceprint-speakers'
```

Soft-delete a speaker:

```bash
curl -X DELETE 'http://localhost:8000/api/v1/voiceprint-speakers/{speaker_id}'
```

## Failure Behavior

Voiceprint failures do not break ASR by default. Database errors, embedding
provider errors, missing local speaker labels, or insufficient speech duration
are logged and the original diarization labels are kept.
