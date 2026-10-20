"""Common reusable Pydantic schemas."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for health check endpoint response."""

    status: str = Field(..., json_schema_extra={"example": "healthy"}, description="Status of the application")
    service: str = Field(..., json_schema_extra={"example": "Adaptive RAG System"}, description="Service name")
    version: str = Field(default="1.0.0", json_schema_extra={"example": "1.0.0"}, description="Application version")


class ErrorResponse(BaseModel):
    """Schema for standard API error responses."""

    error: str = Field(..., description="Error type or summary")
    detail: str = Field(..., description="Detailed description of the error")


class LivenessResponse(BaseModel):
    """Schema for liveness probe endpoint response."""

    status: str = Field(default="alive", description="Process liveness status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Application version")


class ReadinessResponse(BaseModel):
    """Schema for readiness probe endpoint response."""

    status: str = Field(..., description="Overall readiness status ('ready' or 'degraded')")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Application version")
    components: dict = Field(default_factory=dict, description="Sanitized status breakdown for subsystem dependencies")
