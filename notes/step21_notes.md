# Step 21: FastAPI Backend Setup

**Estimated Time:** 45 minutes
**Status:** ⬜ Not Started

---

## Goal
Create a FastAPI server that wraps the existing from-scratch neural network as a REST API. This is the bridge between the React frontend and the NumPy model.

## Tasks
- [ ] Create `api/` directory with `__init__.py`
- [ ] Create `api/server.py` — FastAPI app with CORS middleware
- [ ] Create `api/model_service.py` — singleton that loads the trained model once at startup
- [ ] Implement `GET /api/health` — returns server status
- [ ] Implement `GET /api/model/info` — returns architecture, param count, accuracy, class names
- [ ] Implement `POST /api/classify` — accepts audio file, returns prediction + confidence scores + spectrogram data
- [ ] Implement `POST /api/spectrogram` — accepts audio file, returns spectrogram as base64 or JSON array
- [ ] Add `fastapi`, `uvicorn`, `python-multipart` to `requirements.txt`
- [ ] Test all endpoints with curl or the auto-generated Swagger docs at `/docs`
- [ ] Verify model loads correctly from `models/model.npz`

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
- [ ] `/api/health` returns 200
- [ ] `/api/model/info` returns correct architecture info
- [ ] `/api/classify` with a .wav file returns valid prediction
- [ ] `/api/spectrogram` returns 64×64 array
- [ ] CORS headers allow requests from localhost:5173
- [ ] Model loads in <1 second at startup

## Dependencies to Add
```
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
```

## Next
Step 22: React + TypeScript Frontend Setup
