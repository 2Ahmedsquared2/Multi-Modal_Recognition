"""
FastAPI Server — Acoustic Pattern Recognition Engine

Wraps the from-scratch NumPy neural network as a REST API.
Serves classification, spectrogram generation, evaluation metrics,
and interactive tool endpoints for the React frontend.

Run:
    uvicorn api.server:app --reload --port 8000
    # Then visit http://localhost:8000/docs for Swagger UI
"""

import time
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
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
)

# ---------------------------------------------------------------------------
# Singleton model service
# ---------------------------------------------------------------------------

model_service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup, clean up on shutdown."""
    print("\n🚀 Starting Acoustic Pattern Recognition API...")
    start = time.time()
    model_service.load()
    elapsed = time.time() - start
    print(f"⏱️  Model loaded in {elapsed:.2f}s\n")
    yield
    print("\n👋 Shutting down API...")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Acoustic Pattern Recognition Engine",
    description=(
        "REST API for instrument classification using a from-scratch "
        "NumPy neural network.  Upload audio or a spectrogram and get "
        "predictions, confidence scores, and evaluation metrics."
    ),
    version="1.0.0",
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

# Add deployed frontend URL(s) from environment, comma-separated
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
#  Health & Info
# ===================================================================

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check — confirms the API is up and model is loaded."""
    model_loaded = model_service.network is not None
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        num_classes=len(model_service.class_names) if model_loaded else None,
        classes=model_service.class_names if model_loaded else None,
    )


@app.get("/api/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Architecture, parameter count, class names, and test-set metrics."""
    if model_service.network is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_service.get_model_info()


# ===================================================================
#  Upload validation helpers
# ===================================================================

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".wma"}
ALLOWED_CONTENT_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave",
    "audio/mpeg", "audio/mp3",
    "audio/ogg", "audio/flac",
    "audio/mp4", "audio/x-m4a", "audio/aac",
    "audio/x-ms-wma",
    "application/octet-stream",   # browsers sometimes send this
}


async def _validate_and_read(file: UploadFile, label: str = "audio") -> bytes:
    """
    Validate an uploaded audio file for type and size, then return bytes.

    Raises HTTPException with 413 (too large) or 415 (unsupported type).
    """
    # --- Extension check ---
    filename = file.filename or "upload.wav"
    ext = Path(filename).suffix.lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Accepted formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    # --- Content-type check (lenient — browsers are inconsistent) ---
    ct = (file.content_type or "").lower()
    if ct and ct not in ALLOWED_CONTENT_TYPES and not ct.startswith("audio/"):
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported content type '{ct}'. Please upload an audio file "
                f"({', '.join(sorted(ALLOWED_EXTENSIONS))})."
            ),
        )

    # --- Read + size check ---
    audio_bytes = await file.read()
    if len(audio_bytes) > MAX_FILE_SIZE:
        size_mb = len(audio_bytes) / (1024 * 1024)
        limit_mb = MAX_FILE_SIZE // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum allowed size is {limit_mb} MB.",
        )

    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty — no audio content found.",
        )

    return audio_bytes


# ===================================================================
#  Classification
# ===================================================================

@app.post("/api/classify", response_model=ClassifyResponse, tags=["Classification"])
async def classify(file: UploadFile = File(...)):
    """
    Upload an audio file → get instrument prediction + confidence scores.

    Accepts any format supported by librosa (.wav, .mp3, .flac, .ogg, etc.).
    Returns the predicted class, per-class confidence, the generated
    mel-spectrogram, and basic waveform metadata.
    """
    if model_service.network is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    audio_bytes = await _validate_and_read(file)

    try:
        audio = model_service.load_audio_from_bytes(
            audio_bytes, file.filename or "upload.wav"
        )
        result = model_service.classify(audio)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process audio file: {str(e)}",
        )


@app.post(
    "/api/classify/live",
    response_model=ClassifyResponse,
    tags=["Classification"],
)
async def classify_live(file: UploadFile = File(...)):
    """
    Classify a live audio chunk from the browser microphone.

    Same processing pipeline as /api/classify — accepts a short WAV
    blob captured via the Web Audio API.
    """
    if model_service.network is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    audio_bytes = await _validate_and_read(file, label="live audio")

    try:
        audio = model_service.load_audio_from_bytes(
            audio_bytes, file.filename or "live.wav"
        )
        result = model_service.classify(audio)
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
async def spectrogram(file: UploadFile = File(...)):
    """
    Upload an audio file → get the 64×64 mel-spectrogram as a 2-D JSON array.
    """
    if model_service.network is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    audio_bytes = await _validate_and_read(file)

    try:
        audio = model_service.load_audio_from_bytes(
            audio_bytes, file.filename or "upload.wav"
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
async def confusion_matrix():
    """Confusion matrix computed on the held-out test set at startup."""
    result = model_service.get_confusion_matrix()
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
async def class_metrics():
    """Per-class precision, recall, F1-score from test-set evaluation."""
    result = model_service.get_class_metrics()
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
async def training_history():
    """
    Training / validation loss and accuracy curves.

    Requires ``results/metrics/training_history.npz`` to have been saved
    during training.  Returns 404 if the file does not exist.
    """
    history_path = Path("results/metrics/training_history.npz")

    if not history_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Training history not saved. "
                "Re-run training to generate results/metrics/training_history.npz."
            ),
        )

    try:
        data = np.load(history_path, allow_pickle=True)
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
async def tsne():
    """
    t-SNE 2-D embeddings of test-set features.

    Requires ``results/metrics/tsne_data.npz`` to have been pre-computed.
    Returns 404 if the file does not exist.
    """
    tsne_path = Path("results/metrics/tsne_data.npz")

    if not tsne_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "t-SNE data not pre-computed. "
                "Run the visualization pipeline to generate results/metrics/tsne_data.npz."
            ),
        )

    try:
        data = np.load(tsne_path, allow_pickle=True)
        coords = data["coords"]            # (N, 2)
        labels = data["labels"]             # (N,)  — true class indices
        predictions = data["predictions"]   # (N,)  — predicted class indices
        confidences = data["confidences"]   # (N,)  — prediction confidence
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
async def what_if(body: WhatIfRequest):
    """
    Re-classify a modified spectrogram (What-If tool).

    Send a 2-D spectrogram array and receive a new prediction.
    Used by the frontend spectrogram editor to show how changes
    in frequency content affect classification.
    """
    if model_service.network is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        spec = np.array(body.spectrogram)
        if spec.ndim != 2:
            raise ValueError("Spectrogram must be a 2-D array")

        result = model_service.classify_spectrogram(spec)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"What-If processing failed: {str(e)}",
        )
