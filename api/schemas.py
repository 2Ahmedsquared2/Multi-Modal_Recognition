"""
Pydantic Response Models for the Acoustic Pattern Recognition API.

Every endpoint returns a typed, documented schema so FastAPI can
auto-generate Swagger/OpenAPI docs at /docs.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Health & Info
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    num_classes: Optional[int] = None
    classes: Optional[List[str]] = None


class LayerInfo(BaseModel):
    type: str
    input_size: int
    output_size: int
    activation: str
    parameters: int


class ArchitectureInfo(BaseModel):
    input_size: int
    hidden_sizes: List[int]
    num_classes: int
    layers: List[LayerInfo]


class ModelInfoResponse(BaseModel):
    architecture: ArchitectureInfo
    parameters: int
    class_names: List[str]
    spectrogram_shape: List[int]
    sample_rate: int
    audio_duration: float
    test_accuracy: Optional[float] = None
    macro_f1: Optional[float] = None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class WaveformSummary(BaseModel):
    duration: float
    sample_rate: int
    peak_amplitude: float


class ClassifyResponse(BaseModel):
    prediction: str
    confidence: float
    all_confidences: Dict[str, float]
    spectrogram: List[List[float]]
    waveform: List[float]
    waveform_summary: WaveformSummary


# ---------------------------------------------------------------------------
# Spectrogram
# ---------------------------------------------------------------------------

class SpectrogramResponse(BaseModel):
    spectrogram: List[List[float]]
    shape: List[int]


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

class ConfusionMatrixResponse(BaseModel):
    matrix: List[List[int]]
    class_names: List[str]


class ClassMetric(BaseModel):
    precision: float
    recall: float
    f1_score: float
    support: int


class ClassMetricsResponse(BaseModel):
    per_class: Dict[str, ClassMetric]
    macro_f1: Optional[float] = None
    test_accuracy: Optional[float] = None


class TrainingHistoryResponse(BaseModel):
    train_loss: List[float]
    val_loss: List[float]
    train_accuracy: List[float]
    val_accuracy: List[float]
    epochs: int


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------

class TSNEPoint(BaseModel):
    x: float
    y: float
    class_name: str
    class_idx: int


class TSNEResponse(BaseModel):
    points: List[TSNEPoint]
    class_names: List[str]


# ---------------------------------------------------------------------------
# What-If Tool
# ---------------------------------------------------------------------------

class WhatIfRequest(BaseModel):
    spectrogram: List[List[float]] = Field(
        ..., description="Modified 2-D spectrogram array (e.g. 64×64)"
    )


class WhatIfResponse(BaseModel):
    prediction: str
    confidence: float
    all_confidences: Dict[str, float]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
