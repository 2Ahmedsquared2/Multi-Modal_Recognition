# Phase 2: Interactive Web Application

> Transform the from-scratch neural network into a polished, interactive web app for portfolio presentation

---

## Overview

**Goal:** Build a React + TypeScript frontend with a FastAPI backend that lets anyone upload audio or use their microphone, watch the classification pipeline work in real time, explore interactive visualizations, and manipulate spectrograms with a What-If tool.

**Architecture:**
```
React + TypeScript (Frontend)    ←→    FastAPI (Python Backend)    ←→    NumPy Neural Network (Existing)
     Tailwind CSS                        Serves API endpoints              From-scratch model
     Plotly / Recharts                   Audio processing                  Spectrogram generation
     Web Audio API                       Model inference                   Classification
     Canvas API                          Visualization data                Attention maps
```

**Why this matters for portfolio:**
- A live web app is 10× more impressive than a GitHub repo
- Interactive visualizations show product thinking, not just ML skill
- The What-If tool is something almost no student portfolio has
- React + TypeScript signals full-stack capability

---

## Step Breakdown

### Backend Foundation
| Step | Title | Est. Time | Description |
|------|-------|-----------|-------------|
| 21 | FastAPI Backend Setup | 45 min | Create API server wrapping existing model with endpoints for health, classify, spectrogram, and model info |
| 22 | React + TypeScript Frontend Setup | 30 min | Scaffold Vite + React + TS + Tailwind project with routing and layout shell |

### Core Features
| Step | Title | Est. Time | Description |
|------|-------|-----------|-------------|
| 23 | Audio Upload & Classification | 60 min | Full round-trip: upload audio file → backend processes → frontend displays prediction with confidence bars |
| 24 | Waveform & Spectrogram Display | 45 min | Visualize the uploaded audio as a waveform and the generated spectrogram side by side |
| 25 | Live Microphone Input | 60 min | Web Audio API microphone capture → send to backend → classify in real time |
| 26 | Processing Pipeline Animation | 45 min | Animated step-by-step visualization: raw audio → spectrogram → neural network → prediction |

### Interactive Visualizations
| Step | Title | Est. Time | Description |
|------|-------|-----------|-------------|
| 27 | Interactive Training Dashboard | 60 min | Plotly interactive charts: training/val curves, clickable confusion matrix with drill-down |
| 28 | Interactive t-SNE & Feature Explorer | 60 min | Hoverable t-SNE scatter plot, click points to see spectrogram + hear audio, class filtering |

### Advanced Features
| Step | Title | Est. Time | Description |
|------|-------|-----------|-------------|
| 29 | What-If Tool | 90 min | Canvas-based spectrogram editor: paint/erase frequency regions, re-predict in real time, before/after comparison |
| 30 | Polish & Deployment | 60 min | Responsive design, loading states, error handling, deploy frontend + backend to the web |

### Dual Engine Comparison
| Step | Title | Est. Time | Description |
|------|-------|-----------|-------------|
| 31 | PyTorch Comparison Engine | 90 min | Implement PyTorch version of the neural network, dual-model API support, frontend engine toggle to compare custom NumPy vs PyTorch performance |

---

## Estimated Total Time: ~10.5 hours

---

## Tech Stack

### Frontend
- **React 18** + **TypeScript** — Component-based UI with type safety
- **Vite** — Fast build tool, instant HMR
- **Tailwind CSS** — Utility-first styling, dark mode support
- **Plotly.js** (via react-plotly.js) — Interactive charts (training curves, confusion matrix, t-SNE)
- **Recharts** — Simpler charts (confidence bars, metrics)
- **Web Audio API** — Microphone capture and audio playback
- **Canvas API** — What-If tool spectrogram editor

### Backend
- **FastAPI** — Modern Python API framework, auto-generated docs
- **Uvicorn** — ASGI server
- **Existing codebase** — All from-scratch NumPy code stays as-is
- **librosa** — Audio processing (already installed)
- **NumPy** — Model inference (already installed)

### Deployment
- **Frontend** — Vercel (free, auto-deploys from GitHub)
- **Backend** — Railway or Render (free tier, runs Python)

---

## API Endpoints (Planned)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/health` | Health check |
| GET | `/api/model/info?model=custom` | Architecture, param count, accuracy (supports `?model=pytorch`) |
| GET | `/api/model/training-history?model=custom` | Training curves data (JSON) (supports `?model=pytorch`) |
| GET | `/api/model/confusion-matrix?model=custom` | Confusion matrix data (JSON) (supports `?model=pytorch`) |
| GET | `/api/model/tsne?model=custom` | t-SNE coordinates + metadata (JSON) (supports `?model=pytorch`) |
| GET | `/api/model/class-metrics?model=custom` | Per-class P/R/F1 (JSON) (supports `?model=pytorch`) |
| POST | `/api/classify?model=custom` | Upload audio → prediction + confidence + spectrogram (supports `?model=pytorch`) |
| POST | `/api/classify/live?model=custom` | Stream audio chunk → prediction (supports `?model=pytorch`) |
| POST | `/api/spectrogram` | Upload audio → spectrogram image data |
| POST | `/api/what-if?model=custom` | Modified spectrogram → re-prediction (supports `?model=pytorch`) |

**Note:** All model-specific endpoints support the `?model=custom` or `?model=pytorch` query parameter to select the inference engine.

---

## Build Order & Dependencies

```
Step 21 (Backend) ──→ Step 23 (Upload+Classify) ──→ Step 25 (Live Mic)
                  │                              │
Step 22 (Frontend) ─┘                            ├──→ Step 26 (Pipeline Animation)
                                                  │
                                                  ├──→ Step 27 (Training Dashboard)
                                                  │
                                                  ├──→ Step 28 (t-SNE Explorer)
                                                  │
Step 24 (Waveform Display) ───────────────────────┤
                                                  │
                                                  └──→ Step 29 (What-If Tool)
                                                            │
                                                            ├──→ Step 30 (Polish & Deploy)
                                                            │
                                                            └──→ Step 31 (PyTorch Engine)
```

Steps 21-22 must come first (backend + frontend scaffold).
Step 23 is the first end-to-end feature.
Steps 24-30 can be done in the listed order (each builds on the previous).
Step 31 can be done anytime after Step 23 (adds parallel PyTorch model).

---

## Success Criteria

- [ ] Someone can visit a URL and classify an instrument from audio
- [ ] Microphone input works in Chrome/Safari
- [ ] All visualizations are interactive (hover, click, zoom)
- [ ] What-If tool lets you modify a spectrogram and see prediction change
- [ ] Frontend includes a toggle to switch between custom NumPy and PyTorch engines
- [ ] Both engines produce consistent predictions on the same input
- [ ] Looks polished and professional on desktop
- [ ] Loads fast, handles errors gracefully
- [ ] Deployed and accessible via a public URL

---

## Notes

### What stays the same
- All existing Python code (model, preprocessing, training) is untouched
- FastAPI just wraps the existing functions as API endpoints
- The model weights are loaded from `models/model.npz` (custom) and `models/pytorch_model.pt` (PyTorch)

### What's new
- `web/` directory for the React frontend
- `api/` directory for the FastAPI backend
- `src/pytorch_model/` directory with PyTorch implementation
- New visualization data endpoints that return JSON instead of saving PNGs
- Engine selection toggle in the frontend sidebar
- Dual-model comparison capability across all features
