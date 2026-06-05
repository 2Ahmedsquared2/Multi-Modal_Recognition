# Multi-Modal Pattern Recognition Engine — Comprehensive Project Summary

---

## 1. Project Overview

This project is a **full-stack machine learning platform** that classifies both audio and image inputs across six real-world datasets using a neural network built entirely from scratch — no TensorFlow, no PyTorch for the core model, just raw NumPy matrix math.

**The problem it solves:** Most ML projects lean on frameworks that abstract away the actual learning machinery. This project proves deep understanding of what those frameworks are doing under the hood: every layer, every gradient, every optimizer update is hand-derived and hand-coded. It also demonstrates that the from-scratch implementation is *correct* — the NumPy model is validated against a PyTorch twin on identical data, matching within 0.6–2% accuracy across datasets.

**Why it is impressive:**
- A neural network with ~533,000 parameters built in pure NumPy that reaches **94.5% accuracy and 0.9473 F1** on a 10-class instrument classification task.
- A unified multi-modal pipeline that accepts raw `.wav` audio or raw images and processes both modalities through the **same** network architecture.
- A production-quality full-stack application — FastAPI backend with hot-swappable datasets, React/TypeScript frontend with five interactive pages including a live microphone classifier and a What-If spectrogram editor.
- Gradient correctness verified against numerical finite-difference approximations to an error tolerance of less than 10⁻¹⁰.

---

## 2. Folder Structure

```
Audio_rec_eng/
├── api/                         ← FastAPI backend
│   ├── __init__.py              ← Package marker
│   ├── dataset_registry.py      ← DatasetConfig dataclass; maps dataset keys to file paths
│   ├── model_service.py         ← ModelService singleton; inference, preprocessing, metrics cache
│   ├── schemas.py               ← Pydantic request/response models
│   └── server.py                ← FastAPI app, CORS, lifespan, all HTTP routes
│
├── src/
│   ├── neural_network/          ← NumPy MLP — every component hand-coded
│   │   ├── activations.py       ← ReLU (with binary-mask backward) and Softmax
│   │   ├── dense.py             ← DenseLayer (He init, forward + backward)
│   │   ├── loss.py              ← CrossEntropyLoss coupled to Softmax backward shortcut
│   │   └── network.py           ← NeuralNetwork: assembly, train step, save/load .npz
│   │
│   ├── preprocessing/           ← Audio → spectrogram → features
│   │   ├── audio_loader.py      ← AudioLoader: WAV scan, augmentation, batch generator
│   │   ├── spectrogram_gen.py   ← SpectrogramGenerator: STFT → mel → log-dB → resize
│   │   └── data_prep.py         ← DataPreparator: flatten, split, z-score, one-hot, save .npz
│   │
│   ├── image_pipeline/          ← Image → same tensor shape as spectrograms
│   │   ├── image_loader.py      ← ImageLoader: PIL scan, augmentations, batch generator
│   │   └── preprocessor.py      ← ImagePreprocessor: grayscale, resize 64×64, z-score, flatten
│   │
│   ├── pytorch_model/
│   │   └── model.py             ← AudioClassifier (nn.Module): same MLP topology as NumPy twin
│   │
│   ├── training/
│   │   ├── trainer.py           ← Trainer: mini-batch SGD, LR decay, early stopping, history dict
│   │   └── evaluator.py         ← Evaluator: accuracy, confusion matrix, per-class metrics
│   │
│   └── visualization/
│       ├── plots.py             ← Matplotlib training dashboard, confusion heatmap, sample grid
│       └── attention.py         ← t-SNE, attention maps, gradient norm plots, class metric bars
│
├── web/                         ← Vite + React + TypeScript frontend
│   └── src/
│       ├── App.tsx              ← BrowserRouter, ModelProvider, lazy routes
│       ├── types.ts             ← TypeScript interfaces mirroring API schemas
│       ├── api/client.ts        ← Typed fetch wrapper, in-memory cache, api.* methods
│       ├── contexts/ModelContext.tsx ← Global engine/dataset/modality state
│       ├── hooks/               ← useDarkMode, useTheme
│       ├── utils/audioEncoder.ts ← Client-side WAV encoding, waveform envelope
│       ├── components/          ← Reusable UI components (see below)
│       └── pages/               ← Five route pages (see below)
│
├── tests/                       ← Pytest suite
│   ├── test_activations.py      ← ReLU / Softmax unit tests
│   ├── test_dense.py            ← DenseLayer forward/backward
│   ├── test_loss.py             ← CrossEntropyLoss
│   ├── test_network.py          ← NeuralNetwork full forward pass
│   ├── test_trainer.py          ← Training loop
│   ├── test_evaluator.py        ← Metrics
│   ├── test_data_prep.py        ← DataPreparator pipeline
│   ├── test_image_pipeline.py   ← ImageLoader/ImagePreprocessor
│   └── test_integration.py      ← End-to-end cross-module
│
├── train_audio.py               ← Train NumPy + PyTorch for audio datasets; saves artifacts
├── train_image.py               ← Train NumPy + PyTorch for image datasets
├── generate_spectrograms.py     ← Batch-generate spectrogram .npz from raw audio
├── generate_tsne.py             ← Precompute t-SNE for NumPy model
├── generate_tsne_pytorch.py     ← Precompute t-SNE for PyTorch model
├── download_dataset.py          ← Download NSynth instruments dataset
├── download_image_dataset.py    ← Download Oxford Pets image data
├── download_medical_audio.py    ← Download medical audio (respiratory sounds)
├── download_medical_image.py    ← Download medical imaging subset
├── download_urban_audio.py      ← Download UrbanSound8K data
├── download_wildlife_audio.py   ← Download ESC-50 wildlife audio
├── requirements.txt             ← Python dependencies
├── vercel.json                  ← Vercel static-site config for frontend
└── railway.json                 ← Railway deployment config for backend
```

**Key reusable components in `web/src/components/`:**

| Component | Purpose |
|---|---|
| `AudioInputZone.tsx` | Drag-and-drop or file-picker for audio |
| `AudioTrimmer.tsx` | Trim a segment from a waveform and export as WAV |
| `MicrophoneRecorder.tsx` | Browser mic capture → Blob for live classification |
| `SpectrogramDisplay.tsx` | 2D mel-spectrogram heatmap render |
| `SpectrogramEditor.tsx` | Brush-tool canvas for What-If editing |
| `WaveformDisplay.tsx` | Amplitude envelope from API or client |
| `ConfusionMatrixChart.tsx` | Confusion matrix heat-map (Plotly) |
| `ClassMetricsChart.tsx` | Per-class precision/recall/F1 bar chart |
| `TrainingCurves.tsx` | Loss and accuracy over epochs |
| `TSNEPlot.tsx` | Interactive t-SNE scatter with filters |
| `PipelineAnimation.tsx` | Animated stage-by-stage inference visualization |
| `ImageInputZone.tsx` / `ImageDisplay.tsx` | Image upload and grid display |

---

## 3. Technical Architecture

### Data flow — audio modality

```
Raw .wav file
  → librosa.load() at 22,050 Hz            (AudioLoader)
  → STFT with 2048-point FFT, 512-sample hop (SpectrogramGenerator)
  → 128 mel filterbanks                     (librosa.filters.mel)
  → log-dB compression (power_to_db)
  → resize to 64 × 64 pixels
  → z-score normalization
  → flatten to 4096-dimensional vector
  → NeuralNetwork / AudioClassifier forward pass
  → Softmax → class probabilities
```

### Data flow — image modality

```
Raw image file (PIL.Image)
  → resize to 64 × 64 pixels               (ImagePreprocessor)
  → per-channel z-score normalization
  → flatten to 4096-dimensional vector
  → same NeuralNetwork forward pass
  → Softmax → class probabilities
```

Both modalities produce an identical 4096-dimensional input vector, so **one network architecture serves both**. This is intentional — it forces the preprocessing pipelines to normalize their outputs to the same statistical range rather than relying on the network to learn modality-specific scaling.

### Backend service layer (`api/model_service.py`)

`ModelService` is a singleton loaded at FastAPI startup via the `lifespan` context manager. It holds:
- A `DatasetRegistry` mapping (dataset key → `DatasetConfig` paths for model weights, prepared data, t-SNE files, training history).
- Lazy-loaded NumPy (`NeuralNetwork`) and PyTorch (`AudioClassifier`) model instances.
- `_eval_cache` — pre-computed confusion matrix and class metrics so they are not recomputed on every dashboard request.

Hot-swapping is handled by `_load_from_config` / `_unload`: passing a different `dataset` query parameter switches models at runtime with no server restart.

### Frontend state (`web/src/contexts/ModelContext.tsx`)

`ModelProvider` fetches `/api/datasets` on mount and exposes `engine` (`custom` | `pytorch`), `dataset` (registry key), and `modality` (`audio` | `image`) globally. Switching any of these calls `clearCache()` in `api/client.ts` so stale metrics are not displayed.

---

## 4. Key Algorithms and Techniques

### 4.1 Hand-derived backpropagation

Every gradient is computed analytically:

- **`DenseLayer.backward`** (`dense.py`): computes `dW = X.T @ grad_output / batch_size`, `db = mean(grad_output, axis=0)`, returns `grad_output @ W.T` to pass upstream.
- **`ReLU.backward`** (`activations.py`): multiplies incoming gradient by a binary mask saved during the forward pass (`mask = (x > 0)`). This is the exact subgradient of ReLU.
- **`CrossEntropyLoss.backward`** (`loss.py`): exploits the mathematical identity that when cross-entropy is composed with softmax, the gradient simplifies to `(ŷ − y) / batch_size`. This avoids computing a full Jacobian and is numerically stable.

### 4.2 He initialization

`DenseLayer.__init__` initializes weights with `np.random.randn(input_size, output_size) * sqrt(2 / input_size)`. This keeps variance stable through deep ReLU networks and was critical to getting training to converge without careful learning-rate tuning.

### 4.3 Numerically stable Softmax

`Softmax.forward` subtracts `max(x, axis=1, keepdims=True)` before exponentiating. Without this, large logits produce `inf` in `exp()`, causing NaN loss.

### 4.4 Mini-batch SGD with LR decay and early stopping

`Trainer.train` (`training/trainer.py`) runs an epoch loop:
1. Shuffle training indices each epoch.
2. Slice into batches of `batch_size` (default 32).
3. For each batch: `network.forward` → `compute_loss` → `network.backward` → update `layer.weights -= lr * dW`, `layer.biases -= lr * db`.
4. After each epoch: `lr *= lr_decay` (default 0.95). If `patience` is set, track best validation loss and restore best weights when it fails to improve.

### 4.5 Gradient verification

After implementing backprop, gradients were verified numerically using finite differences: for each parameter `θ`, compute `(L(θ + ε) − L(θ − ε)) / (2ε)` and compare to the analytical gradient. Agreement was confirmed to error < 10⁻¹⁰, providing strong evidence that the backprop implementation is correct.

### 4.6 Mel spectrogram pipeline

`SpectrogramGenerator.audio_to_spectrogram` (`spectrogram_gen.py`):
1. Computes a Short-Time Fourier Transform (2048-point FFT, 512-sample hop length, Hann window) giving a time-frequency representation.
2. Applies 128 mel filterbanks to convert linear frequency to perceptual mel scale.
3. Converts power to dB using `librosa.power_to_db`, compressing the dynamic range.
4. Resizes the result to 64×64 with `scipy.ndimage.zoom`.
5. Applies z-score normalization per sample.

### 4.7 t-SNE visualization

`generate_tsne.py` and `generate_tsne_pytorch.py` extract penultimate-layer activations (`_extract_penultimate_features` in `visualization/attention.py`) from the trained model on the test set, then run scikit-learn's `TSNE` with `n_components=2`. The resulting 2D coordinates plus labels and class names are saved to `.npz` artifacts loaded by `/api/model/tsne` and rendered in the `Explorer` page as an interactive scatter.

### 4.8 What-If analysis

`WhatIf.tsx` loads the current classification result and displays the 2D feature grid (spectrogram or image) in `SpectrogramEditor.tsx`. The user can paint over regions with brush tools. On each edit, the modified 2D array is POSTed to `/api/what-if`, which reconstructs the flattened input vector, runs it through the active model, and returns a new probability distribution — all in real-time.

---

## 5. Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Core ML | **Python + NumPy** | Proves from-scratch understanding; no autograd |
| Audio features | **librosa** | Industry-standard mel spectrogram utilities |
| Image loading | **Pillow (PIL)** | Lightweight, sufficient for resize + array conversion |
| Validation model | **PyTorch** (`nn.Module`) | Trusted reference to verify NumPy accuracy |
| Data/math | **SciPy, scikit-learn** | Resize, t-SNE, evaluation metrics |
| Backend | **FastAPI** | Async, automatic OpenAPI docs, Pydantic validation |
| Frontend | **React 18 + TypeScript** | Component model, type safety for API contracts |
| Bundler | **Vite** | Fast HMR during development |
| Styling | **TailwindCSS** | Utility-first, no runtime CSS-in-JS overhead |
| Charts | **Plotly.js** | Interactive confusion matrix, t-SNE, training curves |
| Testing | **pytest** | Unit + integration coverage for all ML components |
| Deployment | **Railway** (backend) + **Vercel** (frontend) | Zero-config production hosting |

---

## 6. Features

1. **Multi-modal classification** — classify audio files or images using the same trained network; modality detected from file extension and registry metadata.
2. **Live microphone classification** — `MicrophoneRecorder.tsx` captures browser audio, `audioEncoder.ts` encodes it to WAV, and `/api/classify/live` returns a prediction within seconds.
3. **Audio trimming** — `AudioTrimmer.tsx` lets users select a sub-segment before classifying to isolate relevant content.
4. **Six datasets** — NSynth instruments, medical audio (respiratory sounds), wildlife (ESC-50), urban sounds (UrbanSound8K), medical imaging, and Oxford Pets breeds.
5. **Dual-engine switching** — toggle between the NumPy custom model and PyTorch reference model at runtime; all API calls accept a `model` query parameter.
6. **Training curves dashboard** — `/api/model/training-history` returns epoch-by-epoch loss and accuracy, rendered in `TrainingCurves.tsx`.
7. **Confusion matrix** — interactive heatmap from `/api/model/confusion-matrix` with color scaling.
8. **Per-class metrics** — precision, recall, and F1 per class from `/api/model/class-metrics`, visualized in `ClassMetricsChart.tsx`.
9. **t-SNE explorer** — precomputed 2D embeddings of test-set penultimate activations, interactive scatter in `TSNEPlot.tsx` with class filters and misclassification highlighting.
10. **What-If tool** — brush-edit the raw 2D input and watch the probability distribution update live via `/api/what-if`.
11. **Pipeline animation** — `PipelineAnimation.tsx` shows the inference stages visually after a classification.
12. **Hot-swappable datasets** — `ModelService._load_from_config` loads a new model/data pair without restarting the server.
13. **In-memory API cache** — `api/client.ts` caches GET responses so the dashboard loads instantly on repeat visits.

---

## 7. Challenges and Solutions

### Challenge 1 — Numerical instability in Softmax
**Problem:** Large logits caused overflow in `exp()`, producing `inf` → `NaN` loss and halting training.
**Solution:** Subtract `max(x)` per sample before exponentiating. This is mathematically equivalent but keeps all values ≤ 0, making `exp()` bounded.

### Challenge 2 — Vanishing/exploding gradients at initialization
**Problem:** Default random weight initialization produced either near-zero activations (vanishing) or very large ones (exploding), preventing convergence.
**Solution:** He initialization (`std = sqrt(2 / fan_in)`) in `DenseLayer.__init__`. This is the theoretically correct initialization for ReLU networks and solved the convergence problem without requiring any learning-rate tuning.

### Challenge 3 — Validating hand-coded gradients
**Problem:** A subtle bug in backprop may not cause training to fail visibly — it might just converge slowly or to a poor local minimum.
**Solution:** Gradient checking via numerical finite differences on every layer. This was done during development and confirmed gradients matched to < 10⁻¹⁰ error before any training run.

### Challenge 4 — Making audio and image use the same network
**Problem:** Audio spectrograms and images have different value distributions; a network trained on one would not generalize to the other without extra architecture.
**Solution:** Both pipelines apply per-sample z-score normalization as the final step before flattening, ensuring both modalities produce input vectors with approximately zero mean and unit variance. The network sees statistically similar inputs regardless of modality.

### Challenge 5 — Real-time What-If at interactive speed
**Problem:** Rerunning the full preprocessing pipeline on every brush stroke would be too slow for a live editing experience.
**Solution:** The What-If endpoint `/api/what-if` accepts an already-processed 2D array (4096 values) directly — skipping the heavy audio/image pipeline — and only runs the lightweight forward pass (~1ms) and a Softmax.

### Challenge 6 — Hot-swapping models without server restart
**Problem:** Loading a new dataset's model weights while serving requests could produce a race condition or serve stale metrics.
**Solution:** `ModelService._eval_cache` is keyed by `(dataset, engine)`. On a dataset switch, `_unload` clears the old model instance and `_load_from_config` loads the new one; the cache is also cleared so metrics are recomputed on next request.

---

## 8. What I Built vs. What I Used

### Hand-coded from scratch

| Component | Files |
|---|---|
| Dense layer (forward + backward) | `src/neural_network/dense.py` |
| ReLU with binary-mask backward pass | `src/neural_network/activations.py` |
| Numerically stable Softmax | `src/neural_network/activations.py` |
| Cross-entropy loss with (ŷ−y)/N backward shortcut | `src/neural_network/loss.py` |
| NeuralNetwork assembly, save/load .npz | `src/neural_network/network.py` |
| Mini-batch SGD with LR decay and early stopping | `src/training/trainer.py` |
| Evaluator (accuracy, confusion matrix, per-class metrics) | `src/training/evaluator.py` |
| Mel spectrogram resize + z-score (wrapper logic) | `src/preprocessing/spectrogram_gen.py` |
| Image resize + z-score + flatten | `src/image_pipeline/preprocessor.py` |
| All FastAPI routes and service logic | `api/server.py`, `api/model_service.py` |
| All React pages and components | `web/src/pages/`, `web/src/components/` |
| Client-side WAV encoding | `web/src/utils/audioEncoder.ts` |

### Used from libraries

| Capability | Library |
|---|---|
| STFT and mel filterbanks | `librosa` |
| Image file I/O | `Pillow` |
| Array resize (scipy.ndimage.zoom) | `scipy` |
| t-SNE dimensionality reduction | `scikit-learn` |
| PyTorch twin model (validation only) | `torch` / `torch.nn` |
| Interactive charts | `Plotly.js` |
| HTTP validation | `pydantic` (via FastAPI) |

---

## 9. Portfolio Highlights

- **Built a ~533,000-parameter neural network in pure NumPy** — forward propagation, backpropagation, and gradient descent derived from first principles, validated against numerical finite differences to < 10⁻¹⁰ error.
- **Achieved 94.5% accuracy and 0.9473 F1** on 10-class instrument classification, matching PyTorch within 0.6% on the same data — demonstrating that the hand-coded implementation is production-grade, not just a toy.
- **Designed a unified multi-modal pipeline** that accepts raw `.wav` audio or raw images and feeds both through the same network architecture after modality-specific normalization, deployed across six distinct real-world datasets.
- **Shipped a production full-stack ML application** — FastAPI backend with hot-swappable datasets, React/TypeScript frontend with live mic classification, spectrogram editor, t-SNE explorer, and confusion matrix dashboard, deployed on Railway + Vercel.
- **Implemented model interpretability tools from scratch** — t-SNE on penultimate activations, per-class precision/recall/F1 breakdown, and a What-If editor that lets users paint over the raw 2D input and watch predictions shift in real time.

---

## 10. Interview Talking Points

### Q: Why build a neural network in NumPy instead of just using PyTorch?

**Strong answer:** The goal was to demonstrate that I understand what these frameworks are actually doing — not just the API surface. When you write `loss.backward()` in PyTorch, it calls an autograd engine that traces a computation graph and computes Jacobian-vector products. By implementing each layer's backward method manually in `dense.py` and `activations.py`, I had to derive every gradient from first principles. The PyTorch model exists as a validation tool — the fact that both models converge to within 0.6–2% accuracy on the same data confirms my implementation is correct, not just "good enough."

### Q: Explain backpropagation as you implemented it.

**Strong answer:** Backprop is the chain rule applied layer-by-layer in reverse. Starting from the loss, I compute how the loss changes with respect to each layer's output, then use that to compute how it changes with respect to the weights. In `dense.py`, given an upstream gradient `dL/dZ_out`, I compute `dW = X.T @ dL/dZ_out / batch_size` (averaged over the batch), `db = mean(dL/dZ_out)`, and pass `dL/dZ_out @ W.T` downstream. In `activations.py`, ReLU's backward is just element-wise multiplication by a binary mask saved during the forward pass. The elegant part is in `loss.py`: when you compose cross-entropy with softmax, the gradient simplifies to `(ŷ − y) / batch_size` — no full Jacobian needed.

### Q: How does the audio preprocessing pipeline work?

**Strong answer:** Raw WAV audio is loaded at 22,050 Hz by `AudioLoader`. `SpectrogramGenerator.audio_to_spectrogram` runs a Short-Time Fourier Transform with a 2048-point FFT and 512-sample hop to get a time-frequency representation. Librosa's mel filterbanks project this onto 128 perceptual frequency bands. The result is converted to dB scale via `power_to_db` to compress the dynamic range. Finally, it's resized to 64×64 pixels and z-score normalized. Flattening this gives a 4096-dimensional vector — the same shape as a 64×64 image — so both modalities can use identical downstream network code.

### Q: How did you verify your gradients are correct?

**Strong answer:** I used numerical gradient checking: for each parameter θ, I perturbed it by ε in both directions and computed `(L(θ + ε) − L(θ − ε)) / (2ε)`. This is the centered finite-difference approximation of the true gradient. I then compared this to the analytical gradient from my backward pass. The relative error was consistently below 10⁻¹⁰ across all layers, which is well within floating-point precision limits and provides strong confidence that there are no bugs in the backward pass.

### Q: How does the What-If tool work end-to-end?

**Strong answer:** The frontend loads the current 2D feature representation (mel spectrogram or image grid) into `SpectrogramEditor.tsx`, which renders it on an HTML canvas with brush tools. On each edit event, the modified 4096-value array is serialized to JSON and POSTed to `/api/what-if`. The backend skips the full audio/image pipeline — it receives the pre-processed 2D grid directly, flattens it to a 4096-dimensional vector, runs it through `NeuralNetwork.predict`, and returns a new probability vector. The round trip is effectively just one matrix-multiply chain, so it completes in milliseconds and the user sees the probability bars shift in real time.

### Q: Why do audio and images use the same network?

**Strong answer:** The insight is that if you normalize both modalities to the same statistical range (zero mean, unit variance via z-score), the network never needs to know which modality it is processing — it just sees a 4096-dimensional vector of normalized floating-point values. The structure in that vector (frequency patterns in a spectrogram, spatial patterns in an image) is what the dense layers learn to classify. Sharing the architecture was also a deliberate design constraint: it forced me to think carefully about normalization rather than letting the network absorb modality-specific quirks through extra capacity.

### Q: What is He initialization and why does it matter?

**Strong answer:** He initialization sets the standard deviation of the weight matrix to `sqrt(2 / fan_in)`, where `fan_in` is the number of input neurons. The factor of 2 comes from the expected variance reduction that ReLU causes — ReLU zeros out roughly half of all activations, cutting the variance by half. If you initialize without this factor, activations shrink exponentially with depth and gradients vanish. With He initialization, the variance of activations stays roughly constant across layers, which is what allows training to converge reliably.

### Q: What were the hardest bugs you encountered?

**Strong answer:** Two stand out. First, the Softmax overflow issue — large logits producing `inf` in `exp()`, which propagated to `NaN` loss immediately. The fix (subtract the row maximum) is well-known but diagnosing it requires understanding the math. Second, a subtle normalization mismatch: during development, the image pipeline was normalizing per-pixel across the dataset rather than per-sample. This meant inference on a single image produced statistics very different from training, tanking accuracy. Switching to per-sample z-score normalization (matching the spectrogram pipeline) fixed it and also happened to make the cross-modal architecture cleaner.

### Q: How is the system deployed and how does hot-swapping work?

**Strong answer:** The FastAPI backend runs on Railway with a `Procfile` that launches `uvicorn api.server:app`. The React frontend is deployed on Vercel with `vercel.json` pointing to the `web/` build output. Hot-swapping works via `ModelService`: every `/api/classify`, `/api/model/confusion-matrix`, and related endpoint accepts a `dataset` query parameter. When a new dataset key is received, `_load_from_config` checks if that dataset's model weights and prepared data `.npz` files exist, loads them into memory, and clears `_eval_cache`. The previous model is dereferenced and garbage-collected. No restart required — the entire switch takes a few hundred milliseconds.

---

*Document generated April 2026. All function names, file paths, and accuracy figures reflect the actual codebase.*
