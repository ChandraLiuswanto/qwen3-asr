"""Voiceprint API models."""

from pydantic import BaseModel, Field


class VoiceprintRegistrationResponse(BaseModel):
    speaker_id: str = Field(..., description="Persistent speaker id")
    display_name: str = Field(..., description="Speaker display name")
    voiceprint_id: str = Field(..., description="First voiceprint sample id")
    voiceprint_ids: list[str] = Field(..., description="Voiceprint sample ids")
    voiceprint_count: int = Field(..., description="Registered sample count")
    status: int = Field(default=200, description="Status code")
    message: str = Field(default="SUCCESS", description="Response message")


class VoiceprintSampleRegistrationResponse(BaseModel):
    speaker_id: str = Field(..., description="Persistent speaker id")
    voiceprint_ids: list[str] = Field(..., description="Voiceprint sample ids")
    voiceprint_count: int = Field(..., description="Registered sample count")
    status: int = Field(default=200, description="Status code")
    message: str = Field(default="SUCCESS", description="Response message")


class VoiceprintSpeakerItem(BaseModel):
    speaker_id: str = Field(..., description="Persistent speaker id")
    display_name: str = Field(..., description="Speaker display name")
    description: str | None = Field(default=None, description="Speaker description")
    voiceprint_count: int = Field(default=0, description="Voiceprint sample count")


class VoiceprintSpeakerListResponse(BaseModel):
    speakers: list[VoiceprintSpeakerItem] = Field(default_factory=list)
    status: int = Field(default=200, description="Status code")
    message: str = Field(default="SUCCESS", description="Response message")


class VoiceprintDeleteResponse(BaseModel):
    speaker_id: str = Field(..., description="Persistent speaker id")
    status: int = Field(default=200, description="Status code")
    message: str = Field(default="SUCCESS", description="Response message")
