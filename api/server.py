"""
FastAPI Server — Multi-Modal Pattern Recognition Engine

Wraps the from-scratch NumPy neural network AND a PyTorch comparison model
as a REST API.  All endpoints accept:
  - `dataset` query parameter to select the active dataset (lazy-loaded)
  - `model`   query parameter to select the inference engine

Run:
    uvicorn api.server:app --reload --port 8000
    # Then visit http://localhost:8000/docs for Swagger UI
"""

import time
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.model_service import ModelService
from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    ClassifyResponse,
    SpectrogramResponse,
    ConfusionMatrixResponse,
    ClassMetricsResponse,
    TrainingHistoryResponse,
    TSNEResponse,
    TSNEPoint,
    WhatIfRequest,
    WhatIfResponse,
    DatasetListResponse,
    DatasetSummary,
)

# ---------------------------------------------------------------------------
# Singleton model service
# ---------------------------------------------------------------------------

model_service = ModelService()

# Reusable query parameter descriptions
ENGINE_DESC = "Inference engine: 'custom' (NumPy from-scratch) or 'pytorch'"
DATASET_DESC = (
    "Dataset key, e.g. 'audio/music', 'audio/medical'. "
    "Omit to keep the currently active dataset."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_dataset(dataset: Optional[str]) -> None:
    """
    Ensure the requested dataset is loaded.
    - None → keep whatever is currently active (no-op).
    - Specified → lazy-swap to that dataset.
    Raises HTTP 404 if dataset is unknown or has no trained models.
    """
    if dataset is None:
        if model_service.active_dataset is None:
            model_service.load_default()
        return

    try:
        model_service.ensure_dataset(dataset)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


def _require_model() -> None:
    """Raise 503 if no model is loaded at all."""
    if model_service.network is None and model_service.pytorch_model is None:
        raise HTTPException(status_code=503, detail="No model loaded")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the default dataset once at startup."""
    print("\n🚀 Starting Multi-Modal Pattern Recognition API...")
    start = time.time()
    model_service.load_default()
    elapsed = time.time() - start
    print(f"⏱️  Default dataset loaded in {elapsed:.2f}s\n")
    yield
    print("\n👋 Shutting down API...")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Multi-Modal Pattern Recognition Engine",
    description=(
        "REST API for multi-modal classification.  Supports multiple datasets "
        "(audio genres, medical audio, images, etc.) with lazy loading, and "
        "two engines per dataset: NumPy from-scratch & PyTorch.  "
        "Pass `?dataset=audio/music` and `?model=custom` to select."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev servers + deployed origins
import os as _os

_cors_origins = [
    "http://localhost:5173",    # Vite dev server
    "http://localhost:3000",    # Alternative React port
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

_extra_origins = _os.environ.get("CORS_ORIGINS", "")
if _extra_origins:
    _cors_origins.extend([o.strip() for o in _extra_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================================================================
#  Datasets
# ===================================================================

@app.get(
    "/api/datasets",
    response_model=DatasetListResponse,
    tags=["Datasets"],
)
async def list_datasets():
    """List all registered datasets and their readiness status."""
    summaries = [
        DatasetSummary(**ds)
        for ds in model_service.registry.summary()
    ]
    return DatasetListResponse(
        datasets=summaries,
        active_dataset=model_service.active_dataset,
    )


# ===================================================================
#  Health & Info
# ===================================================================

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check — confirms API is up, active dataset, and loaded engines."""
    model_loaded = model_service.network is not None
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        num_classes=len(model_service.class_names) if model_loaded else None,
        classes=model_service.class_names if model_loaded else None,
        available_engines=model_service.available_engines(),
        active_dataset=model_service.active_dataset,
    )


@app.get("/api/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info(
    model: str = Query("custom", description=ENGINE_DESC),
    dataset: Optional[str] = Query(None, description=DATASET_DESC),
):
    """Architecture, parameter count, class names, and test-set metrics."""
    _resolve_dataset(dataset)
    _require_model()
    return model_service.get_model_info(engine=model)


# ===================================================================
#  Upload validation helpers (modality-aware)
# ===================================================================

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Audio file types
AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".wma"}
AUDIO_CONTENT_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave",
    "audio/mpeg", "audio/mp3",
    "audio/ogg", "audio/flac",
    "audio/mp4", "audio/x-m4a", "audio/aac",
    "audio/x-ms-wma",
    "application/octet-stream",
}

# Image file types
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}
IMAGE_CONTENT_TYPES = {
    "image/png", "image/jpeg", "image/gif",
    "image/bmp", "image/tiff", "image/webp",
    "application/octet-stream",
}


def _allowed_extensions() -> set:
    """Return allowed file extensions based on active modality."""
    if model_service.modality == "image":
        return IMAGE_EXTENSIONS
    return AUDIO_EXTENSIONS


def _allowed_content_types() -> set:
    """Return allowed content types based on active modality."""
    if model_service.modality == "image":
        return IMAGE_CONTENT_TYPES
    return AUDIO_CONTENT_TYPES


async def _validate_and_read(file: UploadFile, label: str = "file") -> bytes:
    """Validate an uploaded file for type and size, then return bytes.

    Automatically checks against audio or image types depending on
    the currently active dataset's modality.
    """
    modality = model_service.modality
    allowed_ext = _allowed_extensions()
    allowed_ct = _allowed_content_types()

    default_name = "upload.wav" if modality == "audio" else "upload.png"
    filename = file.filename or default_name
    ext = Path(filename).suffix.lower()

    if ext and ext not in allowed_ext:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{ext}' for {modality} dataset. "
                f"Accepted formats: {', '.join(sorted(allowed_ext))}"
            ),
        )

    ct = (file.content_type or "").lower()
    ct_prefix = "audio/" if modality == "audio" else "image/"
    if ct and ct not in allowed_ct and not ct.startswith(ct_prefix):
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported content type '{ct}'. Please upload "
                f"{'an audio' if modality == 'audio' else 'an image'} file "
                f"({', '.join(sorted(allowed_ext))})."
            ),
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        size_mb = len(file_bytes) / (1024 * 1024)
        limit_mb = MAX_FILE_SIZE // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum allowed size is {limit_mb} MB.",
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty — no content found.",
        )

    return file_bytes


# ===================================================================
#  Classification
# ===================================================================

@app.post("/api/classify", response_model=ClassifyResponse, tags=["Classification"])
async def classify(
    file: UploadFile = File(...),
    model: str = Query("custom", description=ENGINE_DESC),
    dataset: Optional[str] = Query(None, description=DATASET_DESC),
):
    """Upload an audio or image file → get prediction + confidence scores.

    Automatically dispatches to the correct preprocessing pipeline
    (audio spectrogram or image preprocessor) based on the active dataset's
    modality.
    """
    _resolve_dataset(dataset)
    _require_model()

    file_bytes = await _validate_and_read(file)

    try:
        result = model_service.classify_input(
            file_bytes, file.filename or "upload", engine=model
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process file: {str(e)}",
        )


@app.post(
    "/api/classify/live",
    response_model=ClassifyResponse,
    tags=["Classification"],
)
async def classify_live(
    file: UploadFile = File(...),
    model: str = Query("custom", description=ENGINE_DESC),
    dataset: Optional[str] = Query(None, description=DATASET_DESC),
):
    """Classify a live audio chunk from the browser microphone.

    Always uses the audio pipeline (microphone input is audio by definition).
    """
    _resolve_dataset(dataset)
    _require_model()

    if model_service.modality != "audio":
        raise HTTPException(
            status_code=400,
            detail="Live microphone classification is only available for audio datasets.",
        )

    file_bytes = await _validate_and_read(file, label="live audio")

    try:
        audio = model_service.load_audio_from_bytes(
            file_bytes, file.filename or "live.wav"
        )
        result = model_service.classify(audio, engine=model)
        result["modality"] = "audio"
        return result
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process audio: {str(e)}",
        )


# ===================================================================
#  Spectrogram
# ===================================================================

@app.post(
    "/api/spectrogram",
    response_model=SpectrogramResponse,
    tags=["Spectrogram"],
)
async def spectrogram(
    file: UploadFile = File(...),
    dataset: Optional[str] = Query(None, description=DATASET_DESC),
):
    """Upload an audio file → get the mel-spectrogram as a 2-D JSON array.

    Audio datasets only. For image datasets, the classify endpoint
    returns the preprocessed image in the spectrogram field.
    """
    _resolve_dataset(dataset)
    _require_model()

    if model_service.modality != "audio":
        raise HTTPException(
            status_code=400,
            detail=(
                "Spectrogram generation is only available for audio datasets. "
                "For image datasets, use /api/classify which returns the "
                "preprocessed image."
            ),
        )

    file_bytes = await _validate_and_read(file)

    try:
        audio = model_service.load_audio_from_bytes(
            file_bytes, file.filename or "upload.wav"
        )
        spec = model_service.generate_spectrogram(audio)
        return SpectrogramResponse(
            spectrogram=spec.tolist(),
            shape=list(spec.shape),
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not generate spectrogram: {str(e)}",
        )


# ===================================================================
#  Evaluation Metrics
# ===================================================================

@app.get(
    "/api/model/confusion-matrix",
    response_model=ConfusionMatrixResponse,
    tags=["Metrics"],
)
async def confusion_matrix(
    model: str = Query("custom", description=ENGINE_DESC),
    dataset: Optional[str] = Query(None, description=DATASET_DESC),
):
    """Confusion matrix computed on the held-out test set."""
    _resolve_dataset(dataset)
    result = model_service.get_confusion_matrix(engine=model)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Confusion matrix not available — model may not have been evaluated.",
        )
    return result


@app.get(
    "/api/model/class-metrics",
    response_model=ClassMetricsResponse,
    tags=["Metrics"],
)
async def class_metrics(
    model: str = Query("custom", description=ENGINE_DESC),
    dataset: Optional[str] = Query(None, description=DATASET_DESC),
):
    """Per-class precision, recall, F1-score from test-set evaluation."""
    _resolve_dataset(dataset)
    result = model_service.get_class_metrics(engine=model)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Class metrics not available.",
        )
    return result


@app.get(
    "/api/model/training-history",
    response_model=TrainingHistoryResponse,
    tags=["Metrics"],
)
async def training_history(
    model: str = Query("custom", description=ENGINE_DESC),
    dataset: Optional[str] = Query(None, description=DATASET_DESC),
):
    """Training / validation loss and accuracy curves."""
    _resolve_dataset(dataset)

    history_path = model_service.get_training_history_path(engine=model)

    if history_path is None or not history_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Training history not saved for engine '{model}' "
                f"on dataset '{model_service.active_dataset}'. "
                f"Expected: {history_path}"
            ),
        )

    try:
        data = np.load(str(history_path), allow_pickle=True)
        return TrainingHistoryResponse(
            train_loss=data["train_loss"].tolist(),
            val_loss=data["val_loss"].tolist(),
            train_accuracy=data["train_accuracy"].tolist(),
            val_accuracy=data["val_accuracy"].tolist(),
            epochs=len(data["train_loss"]),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load training history: {str(e)}",
        )


@app.get(
    "/api/model/tsne",
    response_model=TSNEResponse,
    tags=["Metrics"],
)
async def tsne(
    model: str = Query("custom", description=ENGINE_DESC),
    dataset: Optional[str] = Query(None, description=DATASET_DESC),
):
    """t-SNE 2-D embeddings of test-set features."""
    _resolve_dataset(dataset)

    tsne_path = model_service.get_tsne_path(engine=model)

    if tsne_path is None or not tsne_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"t-SNE data not pre-computed for engine '{model}' "
                f"on dataset '{model_service.active_dataset}'. "
                f"Expected: {tsne_path}"
            ),
        )

    try:
        data = np.load(str(tsne_path), allow_pickle=True)
        coords = data["coords"]
        labels = data["labels"]
        predictions = data["predictions"]
        confidences = data["confidences"]
        class_names = data["class_names"].tolist()

        n_samples = len(coords)
        n_correct = int(np.sum(labels == predictions))

        points = [
            TSNEPoint(
                x=float(coords[i, 0]),
                y=float(coords[i, 1]),
                class_name=class_names[int(labels[i])],
                class_idx=int(labels[i]),
                predicted_label=class_names[int(predictions[i])],
                predicted_idx=int(predictions[i]),
                correct=bool(labels[i] == predictions[i]),
                confidence=round(float(confidences[i]), 4),
            )
            for i in range(n_samples)
        ]

        return TSNEResponse(
            points=points,
            class_names=class_names,
            n_samples=n_samples,
            accuracy=round(n_correct / n_samples, 4) if n_samples > 0 else 0.0,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load t-SNE data: {str(e)}",
        )


# ===================================================================
#  What-If Tool
# ===================================================================

@app.post(
    "/api/what-if",
    response_model=WhatIfResponse,
    tags=["What-If"],
)
async def what_if(
    body: WhatIfRequest,
    model: str = Query("custom", description=ENGINE_DESC),
    dataset: Optional[str] = Query(None, description=DATASET_DESC),
):
    """Re-classify a modified spectrogram (What-If tool)."""
    _resolve_dataset(dataset)
    _require_model()

    try:
        spec = np.array(body.spectrogram)
        if spec.ndim != 2:
            raise ValueError("Spectrogram must be a 2-D array")

        result = model_service.classify_spectrogram(spec, engine=model)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"What-If processing failed: {str(e)}",
        )
