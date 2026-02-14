# 🎵 Acoustic Pattern Recognition Engine

> Building a neural network from scratch (pure NumPy) to classify musical instruments using spectrogram analysis

## Overview

This project implements an end-to-end acoustic pattern recognition pipeline that classifies musical instruments by analyzing audio spectrograms. Unlike typical ML projects that rely on frameworks like TensorFlow or PyTorch, this neural network is built entirely from scratch using only NumPy, demonstrating deep understanding of both signal processing and machine learning fundamentals.

## Key Features

- **Custom Neural Network**: Forward propagation, backpropagation, and gradient descent implemented from scratch
- **Signal Processing Pipeline**: Audio waveforms → Mel-spectrograms → Neural network input
- **Advanced Visualizations**: Training dashboards, attention heatmaps, t-SNE clustering, gradient flow analysis
- **Mathematical Rigor**: Gradient checking validates backpropagation implementation

## Technical Approach

### Signal Processing
- Convert audio files to mel-spectrograms using librosa
- Spectrograms provide visual representation of frequency content over time
- Output: 128×128 grayscale images representing audio characteristics

### Neural Network Architecture
```
Input (16,384) → Dense(512) → ReLU → Dense(256) → ReLU → Dense(10) → Softmax
```

- **Loss Function**: Cross-entropy
- **Optimizer**: Stochastic Gradient Descent (SGD) with momentum
- **Implementation**: Pure NumPy (no ML frameworks)

### Dataset
Training on **NSynth** (Google's Neural Audio Synthesis dataset) - a subset of ~10,000-20,000 samples across multiple instrument families.

## Project Status

🚧 **In Development**

See [docs/hub.md](docs/hub.md) for complete project navigation and progress tracking.

## Requirements

- Python 3.10+
- NumPy, Librosa, Matplotlib (see `requirements.txt`)

## Installation

```bash
# Clone repository
git clone <repository-url>
cd Audio_rec_eng

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
Audio_rec_eng/
├── data/              # Audio files and spectrograms
├── src/               # Source code
│   ├── preprocessing/ # Audio loading, spectrogram generation
│   ├── neural_network/# Network implementation
│   ├── training/      # Training and evaluation
│   └── visualization/ # Plots and dashboards
├── models/            # Saved model checkpoints
├── results/           # Generated visualizations
├── docs/              # Documentation and planning
└── notes/             # Step-by-step implementation notes
```

## Philosophy

This project prioritizes:
1. **Understanding** ML fundamentals over using frameworks
2. **Visual impact** and professional presentation
3. **Technical sophistication** in portfolio materials
4. **Completeness** of implementation

---

**Note**: This project is designed to demonstrate deep technical understanding for portfolio/academic purposes. For production systems, established frameworks (PyTorch, TensorFlow) are recommended.
