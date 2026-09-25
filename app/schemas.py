"""Pydantic models that define the JSON returned by the API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SpeciesPrediction(BaseModel):
    """One species and how confident the model is about it."""

    species: str = Field(description="Display name of the bird species.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence from 0 to 1.")


class IdentifyResponse(BaseModel):
    """Result of identifying the bird in an uploaded photo."""

    top_prediction: SpeciesPrediction
    predictions: list[SpeciesPrediction] = Field(
        description="Top predictions, highest confidence first."
    )
    low_confidence: bool = Field(
        description="True when the top confidence is below the configured threshold."
    )


class HealthResponse(BaseModel):
    """Service status and whether the model is ready."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(description='"ok" when the model is loaded, otherwise "degraded".')
    model_loaded: bool
    max_upload_bytes: int = Field(description="Largest accepted upload, in bytes.")


class ErrorResponse(BaseModel):
    """A user-friendly error message."""

    detail: str
