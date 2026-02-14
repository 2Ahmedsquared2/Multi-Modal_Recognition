# Acoustic Pattern Recognition Engine

> A neural network built entirely from scratch (pure NumPy — no PyTorch, no TensorFlow) that classifies musical instruments by analyzing audio spectrograms. Achieves **94.5% test accuracy** across 10 instrument families.

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **94.5%** |
| Macro F1-Score | **0.9473** |
| Parameters | 533,322 |
| Training Time | ~13 seconds (CPU) |
| Dataset | 3,333 NSynth samples |

### Per-Class Performance

| Instrument | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Bass | 0.972 | 0.933 | 0.952 |
| Brass | 1.000 | 0.976 | 0.988 |
| Flute | 0.964 | 0.964 | 0.964 |
| Guitar | 0.885 | 0.920 | 0.902 |
| Keyboard | 0.908 | 0.920 | 0.914 |
| Mallet | 0.867 | 0.839 | 0.852 |
| Organ | 0.973 | 0.960 | 0.966 |
| Reed | 1.000 | 1.000 | 1.000 |
| String | 0.959 | 1.000 | 0.979 |
| Vocal | 0.955 | 0.955 | 0.955 |

---

## Architecture

```
Audio (.wav) → Mel-Spectrogram (64×64) → Flatten (4096) → Neural Network → Prediction
```

### Neural Network

```
Input (4096) → Dense(128) → ReLU → Dense(64) → ReLU → Dense(10) → Softmax
```

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Input | (batch, 4096) | 0 |
| Dense 1 (4096 → 128) | (batch, 128) | 524,416 |
| ReLU | (batch, 128) | 0 |
| Dense 2 (128 → 64) | (batch, 64) | 8,256 |
| ReLU | (batch, 64) | 0 |
| Dense 3 (64 → 10) | (batch, 10) | 650 |
| Softmax | (batch, 10) | 0 |
| **Total** | | **533,322** |

### Signal Processing Pipeline

1. **Load audio** — 2-second clips at 22,050 Hz sample rate
2. **STFT** — Short-Time Fourier Transform (2048-point FFT, 512-sample hop)
3. **Mel scaling** — Map to 128 mel-frequency bands (perceptually aligned)
4. **Log compression** — Convert power to decibels
5. **Resize** — Scale to 64×64 for neural network input
6. **Normalize** — Zero-mean, unit-variance standardization

---

## Quick Start

### 1. Setup

```bash
git clone <repository-url>
cd Audio_rec_eng

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python download_dataset.py
```

Downloads a subset of Google's NSynth dataset (~350 MB). Organizes 3,333 audio samples into 10 instrument classes.

### 3. Generate Spectrograms

```bash
python generate_spectrograms.py --resolution 64
```

Converts audio files to mel-spectrograms (~15 seconds for all 3,333 samples).

### 4. Train

```bash
python train.py
```

Runs the full pipeline: data loading → training → evaluation → visualization → model saving. Takes ~13 seconds on CPU.

### 5. Custom Training

```bash
# Adjust hyperparameters
python train.py --lr 0.005 --epochs 150 --batch-size 64

# Different architecture
python train.py --hidden 256 128 64

# Skip visualizations for faster iteration
python train.py --no-viz
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | 0.01 | Initial learning rate |
| `--lr-decay` | 0.95 | LR decay per epoch |
| `--batch-size` | 32 | Mini-batch size |
| `--epochs` | 100 | Max training epochs |
| `--patience` | 10 | Early stopping patience |
| `--hidden` | 128 64 | Hidden layer sizes |
| `--resolution` | 64 | Spectrogram NxN resolution |
| `--seed` | 42 | Random seed |
| `--no-viz` | off | Skip visualization generation |

---

## Visualizations

The training pipeline generates 8 publication-quality visualizations to `results/figures/`:

| Visualization | What It Shows |
|--------------|---------------|
| Training History | Loss and accuracy curves over epochs (train + validation) |
| Sample Predictions | 3×3 grid of spectrograms with predicted vs. true labels |
| Confusion Matrix | Heatmap of classification errors across all class pairs |
| Normalized Confusion Matrix | Row-normalized (recall %) version |
| Attention Maps | Which spectrogram regions the network focuses on per class |
| t-SNE Features | 2D scatter plot of learned representations colored by class |
| Gradient Flow | Layer-wise gradient magnitudes (verifies healthy training) |
| Per-Class Metrics | Grouped bar chart of precision, recall, and F1 per class |

---

## What's Built From Scratch

Every component below is implemented using only NumPy — no ML frameworks:

- **Dense (fully connected) layer** — forward pass, backward pass, He weight initialization
- **ReLU activation** — forward and backward with binary gradient mask
- **Softmax activation** — numerically stable with max-subtraction trick
- **Cross-entropy loss** — combined Softmax+CE backward for efficient gradient computation
- **Backpropagation** — full chain-rule gradient flow through all layers
- **Mini-batch SGD optimizer** — with learning rate decay
- **Training loop** — shuffling, batching, early stopping, history tracking
- **Evaluation** — confusion matrix, per-class precision/recall/F1, top confused pairs

**External libraries used only for:**
- Audio I/O and spectrogram generation (librosa)
- Plotting (matplotlib, seaborn)
- t-SNE visualization (scikit-learn)
- Data handling (numpy)

---

## Dataset

**NSynth** (Neural Audio Synthesis) by Google Magenta — a large-scale dataset of annotated musical notes.

| Class | Training | Validation | Test | Total |
|-------|---------|-----------|------|-------|
| Bass | 350 | 75 | 75 | 500 |
| Brass | 188 | 40 | 41 | 269 |
| Flute | 126 | 26 | 28 | 180 |
| Guitar | 350 | 75 | 75 | 500 |
| Keyboard | 350 | 75 | 75 | 500 |
| Mallet | 141 | 30 | 31 | 202 |
| Organ | 350 | 75 | 75 | 500 |
| Reed | 164 | 35 | 36 | 235 |
| String | 214 | 45 | 47 | 306 |
| Vocal | 97 | 22 | 22 | 141 |
| **Total** | **2,330** | **498** | **505** | **3,333** |

Split: 70% train / 15% validation / 15% test (stratified to maintain class distribution).

---

## Project Structure

```
Audio_rec_eng/
├── train.py                        # Main training pipeline (single command)
├── download_dataset.py             # NSynth dataset downloader
├── generate_spectrograms.py        # Spectrogram generation utility
├── requirements.txt                # Python dependencies
│
├── src/
│   ├── preprocessing/
│   │   ├── audio_loader.py         # Audio loading with augmentation
│   │   ├── spectrogram_gen.py      # Mel-spectrogram generation (STFT pipeline)
│   │   └── data_prep.py            # Normalization, splitting, batching
│   │
│   ├── neural_network/
│   │   ├── dense.py                # Dense layer (forward + backward)
│   │   ├── activations.py          # ReLU and Softmax
│   │   ├── loss.py                 # Cross-entropy loss
│   │   └── network.py              # Full network assembly
│   │
│   ├── training/
│   │   ├── trainer.py              # Training loop (SGD, LR decay, early stopping)
│   │   └── evaluator.py            # Evaluation metrics and reports
│   │
│   └── visualization/
│       ├── plots.py                # Training curves, predictions, confusion matrix
│       └── attention.py            # Attention maps, t-SNE, gradient flow
│
├── data/
│   ├── raw/organized/              # Audio files by instrument class
│   ├── spectrograms/               # Pre-generated spectrograms (.npz)
│   └── prepared/                   # Normalized, split data (.npz)
│
├── models/                         # Saved model weights (.npz)
├── results/figures/                # Generated visualizations (.png)
│
├── docs/
│   ├── hub.md                      # Project navigation hub
│   ├── math_derivations.md         # Backpropagation math (LaTeX)
│   ├── architecture.md             # Design decisions and rationale
│   ├── context.md                  # Project goals and approach
│   └── steps.md                    # 16-step implementation guide
│
└── notes/                          # Step-by-step implementation notes
    ├── step01_notes.md             # through
    └── step15_notes.md             # step 15
```

---

## Technical Documentation

- **[Math Derivations](docs/math_derivations.md)** — Complete backpropagation equations, gradient derivations, and numerical verification
- **[Architecture Decisions](docs/architecture.md)** — Why every design choice was made (network shape, activations, loss function, initialization, etc.)
- **[Project Hub](docs/hub.md)** — Central navigation for all docs and progress tracking

---

## Requirements

- Python 3.10+
- NumPy ≥ 1.24
- Librosa ≥ 0.10
- Matplotlib ≥ 3.7
- Seaborn
- Scikit-learn ≥ 1.3 (t-SNE visualization only)

Full list in `requirements.txt`.

---

## Key Technical Highlights

**Signal Processing:** Implemented a Short-Time Fourier Transform pipeline with mel-frequency scaling for perceptual alignment. The mel-scale transformation maps linear frequency to a logarithmic scale matching human auditory perception, enabling the network to learn timbral features that distinguish instrument families.

**From-Scratch Backpropagation:** Derived and implemented the full gradient computation chain, including the combined Softmax + Cross-Entropy backward pass that simplifies to `(predicted - true) / batch_size`. Verified against numerical finite-difference gradients with error < 10⁻¹⁰.

**Gradient Health Analysis:** Automated gradient flow visualization confirms no vanishing or exploding gradients across all layers — validating He initialization and architecture choices.

**Interpretability:** Attention maps computed via input-gradient analysis reveal which spectrogram regions (time-frequency bins) the network relies on for each instrument class, providing insight into learned representations.
