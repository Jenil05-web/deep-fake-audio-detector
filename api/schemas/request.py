"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel
from typing import Optional


class AnalyzeResponse(BaseModel):
    job_id:  str
    status:  str   # "done" or "error"


class ResultResponse(BaseModel):
    job_id:       str
    filename:     str
    verdict:      str        # "REAL" or "FAKE"
    confidence:   float      # 0.0 – 1.0 probability of being fake
    proba_cnn:    float
    proba_lstm:   float
    proba_bio:    float
    duration_sec: float
    inference_ms: float
    status:       str


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool