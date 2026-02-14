# Step 21: FastAPI Backend Setup

**Estimated Time:** 45 minutes
**Status:** ✅ Complete

---

## Goal
Create a FastAPI server that wraps the existing from-scratch neural network as a REST API. This is the bridge between the React frontend and the NumPy model.

## Tasks
- [x] Create `api/` directory with `__init__.py`
- [x] Create `api/server.py` — FastAPI app with CORS middleware
- [x] Create `api/model_service.py` — singleton that loads the trained model once at startup
- [x] Create `api/schemas.py` — Pydantic response models for all endpoints
- [x] Implement `GET /api/health` — returns server status
- [x] Implement `GET /api/model/info` — returns architecture, param count, accuracy, class names
- [x] Implement `POST /api/classify` — accepts any audio format, returns prediction + confidence scores + spectrogram data
- [x] Implement `POST /api/classify/live` — same as classify, for microphone input
- [x] Implement `POST /api/spectrogram` — accepts audio file, returns 64×64 spectrogram as JSON array
- [x] Implement `GET /api/model/confusion-matrix` — returns confusion matrix from test set
- [x] Implement `GET /api/model/class-metrics` — returns per-class P/R/F1
- [x] Implement `GET /api/model/training-history` — returns training curves (404 until data saved)
- [x] Implement `GET /api/model/tsne` — returns t-SNE coordinates (404 until data saved)
- [x] Implement `POST /api/what-if` — accepts modified spectrogram, returns new prediction
- [x] Add `fastapi`, `uvicorn`, `python-multipart` to `requirements.txt`
- [x] Test all endpoints with curl and Swagger docs
- [x] Verify model loads correctly from `models/model.npz`
- [x] Norm stats auto-extracted to `models/norm_stats.npz` on first startup

## Implementation Plan

### Directory Structure
```
api/
├── __init__.py
├── server.py           # FastAPI app, routes, CORS
├── model_service.py    # Loads model + spectrogram generator at startup
└── schemas.py          # Pydantic response models
```

### Key Design Decisions
1. **Singleton model** — Load model once at startup, not per-request (fast inference)
2. **CORS enabled** — Frontend runs on different port during dev (Vite :5173 → FastAPI :8000)
3. **Pydantic schemas** — Type-safe request/response models
4. **Auto-docs** — FastAPI generates Swagger UI at `/docs` for free
5. **Audio processing reuse** — Import existing `AudioLoader` and `SpectrogramGenerator` directly

### Model Service Pattern
```python
# api/model_service.py
class ModelService:
    def __init__(self):
        self.network = NeuralNetwork.load("models/model.npz")
        self.spec_gen = SpectrogramGenerator(target_shape=(64, 64))
        self.class_names = ['bass', 'brass', 'flute', 'guitar', 'keyboard',
                           'mallet', 'organ', 'reed', 'string', 'vocal']
    
    def classify(self, audio_array):
        spectrogram = self.spec_gen.audio_to_spectrogram(audio_array)
        flat = spectrogram.flatten().reshape(1, -1)
        # normalize with saved mean/std
        probs = self.network.forward(flat)
        return probs, spectrogram
```

### Classify Endpoint Response Shape
```json
{
  "prediction": "guitar",
  "confidence": 0.92,
  "all_confidences": {
    "bass": 0.03,
    "brass": 0.01,
    "guitar": 0.92,
    ...
  },
  "spectrogram": [[0.1, 0.3, ...], ...],
  "waveform_summary": {
    "duration": 2.0,
    "sample_rate": 22050,
    "peak_amplitude": 0.87
  }
}
```

## How to Run (Once Built)
```bash
source venv/bin/activate
uvicorn api.server:app --reload --port 8000

# Then visit http://localhost:8000/docs for Swagger UI
```

## Verification
- [x] `/api/health` returns 200 — `{"status": "ok", "model_loaded": true}`
- [x] `/api/model/info` returns correct architecture (4096→128→64→10, 533,322 params)
- [x] `/api/classify` with guitar.wav returns `"guitar"` at 95.89% confidence
- [x] `/api/spectrogram` returns 64×64 array
- [x] `/api/model/confusion-matrix` returns 10×10 matrix
- [x] `/api/model/class-metrics` returns per-class P/R/F1 (macro F1 = 0.9473)
- [x] `/api/what-if` re-classifies a modified spectrogram correctly
- [x] CORS headers allow requests from localhost:5173
- [x] Model loads successfully at startup (test accuracy: 94.46%)

## Dependencies to Add
```
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
```

## Next
Step 22: React + TypeScript Frontend Setup
