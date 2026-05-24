"""Voiceprint management API."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    AuthenticationException,
    InvalidParameterException,
    create_error_response,
    get_http_status_code,
)
from app.core.security import validate_token
from app.models.voiceprint import (
    VoiceprintDeleteResponse,
    VoiceprintRegistrationResponse,
    VoiceprintSampleRegistrationResponse,
    VoiceprintSpeakerItem,
    VoiceprintSpeakerListResponse,
)
from app.services.audio import get_audio_service
from app.services.audio.audio_service import AudioProcessingResult
from app.services.speaker import get_speaker_identification_service
from app.services.speaker.identification_service import VoiceprintSampleSource
from app.utils.common import generate_task_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Voiceprints"])


async def _prepare_voiceprint_samples(
    *,
    files: list[UploadFile],
    task_id: str,
) -> tuple[list[VoiceprintSampleSource], list[AudioProcessingResult]]:
    if not files:
        raise InvalidParameterException("at least one voiceprint sample is required", task_id)

    audio_service = get_audio_service()
    samples: list[VoiceprintSampleSource] = []
    prepared_audios: list[AudioProcessingResult] = []

    for file in files:
        audio_data = await file.read()
        prepared_audio = await audio_service.process_upload_file(
            audio_data=audio_data,
            filename=file.filename,
            task_id=task_id,
            sample_rate=16000,
        )
        prepared_audios.append(prepared_audio)
        samples.append(
            VoiceprintSampleSource(
                audio_path=prepared_audio.normalized_path,
                source_bytes=audio_data,
            )
        )

    return samples, prepared_audios


def _cleanup_prepared_audios(prepared_audios: list[AudioProcessingResult]) -> None:
    audio_service = get_audio_service()
    for prepared_audio in prepared_audios:
        audio_service.cleanup(
            prepared_audio.original_path,
            prepared_audio.normalized_path,
        )


@router.post(
    "/voiceprint-speakers",
    response_model=VoiceprintRegistrationResponse,
    summary="Create a speaker and register voiceprint samples",
)
async def create_voiceprint_speaker(
    request: Request,
    display_name: str = Form(..., description="Speaker display name"),
    description: Optional[str] = Form(None, description="Speaker description"),
    files: list[UploadFile] = File(
        ...,
        alias="file",
        description="One or more single-speaker reference audio files",
    ),
) -> JSONResponse:
    task_id = generate_task_id()
    prepared_audios: list[AudioProcessingResult] = []

    try:
        auth_ok, auth_content = validate_token(request, task_id)
        if not auth_ok:
            raise AuthenticationException(auth_content, task_id)

        normalized_display_name = display_name.strip()
        if not normalized_display_name:
            raise InvalidParameterException("display_name cannot be empty", task_id)

        samples, prepared_audios = await _prepare_voiceprint_samples(
            files=files,
            task_id=task_id,
        )
        result = get_speaker_identification_service().register_speaker_samples(
            display_name=normalized_display_name,
            description=description.strip() if description else None,
            samples=samples,
        )
        payload = VoiceprintRegistrationResponse(
            speaker_id=result.speaker.id,
            display_name=result.speaker.display_name,
            voiceprint_id=result.voiceprint_id,
            voiceprint_ids=result.voiceprint_ids,
            voiceprint_count=len(result.voiceprint_ids),
        ).model_dump()
        return JSONResponse(content=payload, headers={"task_id": task_id})

    except (AuthenticationException, InvalidParameterException) as exc:
        exc.task_id = task_id
        return JSONResponse(
            content=exc.to_dict(),
            headers={"task_id": task_id},
            status_code=get_http_status_code(exc.status_code),
        )
    except Exception as exc:
        logger.error("[%s] Voiceprint registration failed: %s", task_id, exc)
        return JSONResponse(
            content=create_error_response(
                error_code="DEFAULT_SERVER_ERROR",
                message=f"Voiceprint registration failed: {exc}",
                task_id=task_id,
            ),
            headers={"task_id": task_id},
            status_code=500,
        )
    finally:
        _cleanup_prepared_audios(prepared_audios)


@router.post(
    "/voiceprint-speakers/{speaker_id}/samples",
    response_model=VoiceprintSampleRegistrationResponse,
    summary="Add voiceprint samples to an existing speaker",
)
async def add_voiceprint_speaker_samples(
    request: Request,
    speaker_id: str,
    files: list[UploadFile] = File(
        ...,
        alias="file",
        description="One or more single-speaker reference audio files",
    ),
) -> JSONResponse:
    task_id = generate_task_id()
    prepared_audios: list[AudioProcessingResult] = []

    try:
        auth_ok, auth_content = validate_token(request, task_id)
        if not auth_ok:
            raise AuthenticationException(auth_content, task_id)

        samples, prepared_audios = await _prepare_voiceprint_samples(
            files=files,
            task_id=task_id,
        )
        voiceprint_ids = get_speaker_identification_service().add_speaker_samples(
            speaker_id=speaker_id,
            samples=samples,
        )
        payload = VoiceprintSampleRegistrationResponse(
            speaker_id=speaker_id,
            voiceprint_ids=voiceprint_ids,
            voiceprint_count=len(voiceprint_ids),
        ).model_dump()
        return JSONResponse(content=payload, headers={"task_id": task_id})
    except (AuthenticationException, InvalidParameterException) as exc:
        exc.task_id = task_id
        return JSONResponse(
            content=exc.to_dict(),
            headers={"task_id": task_id},
            status_code=get_http_status_code(exc.status_code),
        )
    except Exception as exc:
        logger.error("[%s] Voiceprint sample registration failed: %s", task_id, exc)
        return JSONResponse(
            content=create_error_response(
                error_code="DEFAULT_SERVER_ERROR",
                message=f"Voiceprint sample registration failed: {exc}",
                task_id=task_id,
            ),
            headers={"task_id": task_id},
            status_code=500,
        )
    finally:
        _cleanup_prepared_audios(prepared_audios)


@router.get(
    "/voiceprint-speakers",
    response_model=VoiceprintSpeakerListResponse,
    summary="List registered voiceprint speakers",
)
async def list_voiceprint_speakers(request: Request) -> JSONResponse:
    task_id = generate_task_id()
    try:
        auth_ok, auth_content = validate_token(request, task_id)
        if not auth_ok:
            raise AuthenticationException(auth_content, task_id)

        speakers = get_speaker_identification_service().list_speakers()
        payload = VoiceprintSpeakerListResponse(
            speakers=[
                VoiceprintSpeakerItem(
                    speaker_id=item.id,
                    display_name=item.display_name,
                    description=item.description,
                    voiceprint_count=item.voiceprint_count,
                )
                for item in speakers
            ]
        ).model_dump()
        return JSONResponse(content=payload, headers={"task_id": task_id})
    except AuthenticationException as exc:
        exc.task_id = task_id
        return JSONResponse(
            content=exc.to_dict(),
            headers={"task_id": task_id},
            status_code=get_http_status_code(exc.status_code),
        )
    except Exception as exc:
        logger.error("[%s] Voiceprint speaker list failed: %s", task_id, exc)
        return JSONResponse(
            content=create_error_response(
                error_code="DEFAULT_SERVER_ERROR",
                message=f"Voiceprint speaker list failed: {exc}",
                task_id=task_id,
            ),
            headers={"task_id": task_id},
            status_code=500,
        )


@router.delete(
    "/voiceprint-speakers/{speaker_id}",
    response_model=VoiceprintDeleteResponse,
    summary="Delete a voiceprint speaker",
)
async def delete_voiceprint_speaker(
    request: Request,
    speaker_id: str,
) -> JSONResponse:
    task_id = generate_task_id()
    try:
        auth_ok, auth_content = validate_token(request, task_id)
        if not auth_ok:
            raise AuthenticationException(auth_content, task_id)

        get_speaker_identification_service().delete_speaker(speaker_id=speaker_id)
        payload = VoiceprintDeleteResponse(speaker_id=speaker_id).model_dump()
        return JSONResponse(content=payload, headers={"task_id": task_id})
    except AuthenticationException as exc:
        exc.task_id = task_id
        return JSONResponse(
            content=exc.to_dict(),
            headers={"task_id": task_id},
            status_code=get_http_status_code(exc.status_code),
        )
    except Exception as exc:
        logger.error("[%s] Voiceprint speaker delete failed: %s", task_id, exc)
        return JSONResponse(
            content=create_error_response(
                error_code="DEFAULT_SERVER_ERROR",
                message=f"Voiceprint speaker delete failed: {exc}",
                task_id=task_id,
            ),
            headers={"task_id": task_id},
            status_code=500,
        )
