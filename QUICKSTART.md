# 🚀 Quick Start Guide

## Step-by-Step Setup

### 1. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 2. Download Dataset (~10-30 minutes)
```bash
python download_dataset.py
```

This will:
- Download NSynth test set (~4GB)
- Extract and organize by instrument class
- Create `data/raw/organized/` with 10 instrument folders
- Target: 500 samples per instrument

### 3. Test Audio Loader
```bash
python src/preprocessing/audio_loader.py
```

Expected output:
```
📂 Scanning dataset: data/raw/organized
✓ Found 10 classes: bass, brass, flute, guitar, keyboard...
✓ Found ~5000 audio files total
```

### 4. Next Steps
- Step 3: Generate spectrograms from audio
- Step 4: Prepare training data
- Step 5+: Build neural network

## What Was Implemented (Step 2)

### ✅ Download Script (`download_dataset.py`)
- Downloads NSynth dataset from Magenta
- Organizes by instrument family
- Creates symlinks to save disk space
- Configurable samples per class

### ✅ Audio Loader (`src/preprocessing/audio_loader.py`)
- Loads .wav files with librosa
- **Data Augmentation**:
  - Pitch shifting (±2 semitones)
  - Time stretching (0.8x-1.2x)
  - Background noise
- Train/val/test splitting (70%/15%/15%)
- Batch loading for training

## Troubleshooting

### Download fails?
Manually download from: https://magenta.tensorflow.org/datasets/nsynth#files

### Low disk space?
- Use `--samples-per-instrument 200` for fewer samples
- The test set is smaller than the train set

### Audio loading errors?
- Check that `data/raw/organized/` exists
- Verify .wav files are present
- Run with `--help` for options

## Configuration Options

### Download Script
```bash
python download_dataset.py --help

Options:
  --data-dir DIR                Directory for data (default: data/raw)
  --num-instruments N           Number of classes (default: 10)
  --samples-per-instrument N    Samples per class (default: 500)
```

### Audio Loader
```python
loader = AudioLoader(
    data_dir="data/raw/organized",  # Dataset location
    sample_rate=22050,               # Hz
    duration=2.0,                    # seconds
    augment=True,                    # Enable augmentation
    augment_prob=0.5                 # Probability per augmentation
)
```

## Project Status
- [x] Step 1: Project Setup
- [x] Step 2: Audio Data Loading
- [ ] Step 3: Spectrogram Generation
- [ ] Step 4-16: Neural network & training

---

**Next**: Generate spectrograms from loaded audio (Step 3)
