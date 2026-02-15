# Getting Started Guide

Quick setup guide to get the Multi-Modal Pattern Recognition Engine running on your machine.

---

## Prerequisites

- **Python 3.10+** — Check with `python --version`
- **Node.js 18+** — Check with `node --version` (for web app only)
- **Git** — Check with `git --version`
- **8GB RAM minimum** (16GB recommended for image datasets)
- **~5GB disk space** (for datasets and models)

---

## Option 1: Web Application (Recommended)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd Audio_rec_eng
```

### Step 2: Python Backend Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python -c "import numpy, librosa, fastapi; print('✓ Dependencies installed')"
```

### Step 3: Start Backend Server

```bash
cd api
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

**Keep this terminal open!**

### Step 4: Frontend Setup (New Terminal)

```bash
# Navigate to web folder
cd web

# Install Node dependencies (this may take a few minutes)
npm install

# Verify installation
npm list react react-dom
```

### Step 5: Start Frontend Dev Server

```bash
npm run dev
```

You should see:
```
  VITE v7.x.x  ready in XXX ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### Step 6: Open in Browser

Navigate to **http://localhost:5173** in your browser.

You should see the home page with dataset/modality selectors.

### Step 7: Try It Out!

1. **Select a Dataset** — Choose "Instrument Families" (audio/music) for the pre-trained demo
2. **Go to Classify** — Click "Classify" in the sidebar
3. **Upload Audio** — Drag-drop a `.wav` file or use the microphone
4. **View Results** — See predictions, confidence scores, and visualizations

---

## Option 2: Command-Line Training Only

If you only want to train models without the web interface:

### Step 1: Setup

```bash
git clone <repository-url>
cd Audio_rec_eng

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Download Dataset

```bash
# Download NSynth instrument dataset (~350 MB)
python download_dataset.py
```

### Step 3: Generate Spectrograms

```bash
# Convert audio to spectrograms (~15 seconds)
python generate_spectrograms.py --resolution 64
```

### Step 4: Train Model

```bash
# Train NumPy model (~13 seconds on CPU)
python train.py

# Or with PyTorch comparison
python train.py --compare-pytorch
```

### Step 5: View Results

Training outputs are saved to:
- **Models:** `models/model.npz`, `models/pytorch_model.pt`
- **Figures:** `results/figures/*.png` (8 visualizations)
- **Metrics:** `results/metrics/*.npz` (training history, t-SNE data)

Open the PNG files to view:
- Training curves (loss, accuracy)
- Confusion matrix
- Sample predictions
- Attention maps
- t-SNE embeddings
- Gradient flow
- Per-class metrics

---

## Option 3: Custom Dataset Training

### Train on Different Datasets

```bash
# Medical audio (requires download first)
python train.py --dataset audio/medical --epochs 50

# Wildlife audio
python train.py --dataset audio/wildlife --epochs 50

# Urban sounds
python train.py --dataset audio/urban --epochs 50

# Medical images (DermaMNIST)
python train.py --dataset image/medical --epochs 50

# Pet breeds (Oxford-IIIT)
python train.py --dataset image/wildlife --epochs 50
```

### Adjust Hyperparameters

```bash
python train.py \
  --dataset audio/music \
  --lr 0.005 \
  --lr-decay 0.95 \
  --batch-size 64 \
  --epochs 150 \
  --hidden 256 128 64 \
  --compare-pytorch
```

---

## Troubleshooting

### Backend Issues

**Problem:** `ModuleNotFoundError: No module named 'fastapi'`  
**Solution:** Make sure virtual environment is activated: `source venv/bin/activate`

**Problem:** `Address already in use`  
**Solution:** Change port: `uvicorn server:app --port 8001`

**Problem:** `Model file not found`  
**Solution:** Train a model first: `python train.py --dataset audio/music`

### Frontend Issues

**Problem:** `npm: command not found`  
**Solution:** Install Node.js from [nodejs.org](https://nodejs.org)

**Problem:** `ECONNREFUSED` or `Network Error`  
**Solution:** Make sure backend is running on port 8000

**Problem:** `Module not found` errors  
**Solution:** Delete `node_modules` and reinstall: `rm -rf node_modules && npm install`

### Training Issues

**Problem:** `FileNotFoundError` for dataset  
**Solution:** Run `python download_dataset.py` first

**Problem:** `Out of memory`  
**Solution:** Reduce batch size: `python train.py --batch-size 16`

**Problem:** Very slow training  
**Solution:** This is normal for NumPy (CPU-only). Use `--compare-pytorch` to see GPU speedup.

### Audio Issues

**Problem:** Microphone not working  
**Solution:** Grant browser microphone permissions (check browser settings)

**Problem:** Audio upload fails  
**Solution:** Ensure file is `.wav` format, 16-bit PCM, mono or stereo

**Problem:** No waveform displayed  
**Solution:** Check browser console for errors, may need HTTPS for some features

---

## Verification Checklist

After setup, verify everything works:

- [ ] Backend accessible at http://localhost:8000/docs (FastAPI Swagger UI)
- [ ] Frontend accessible at http://localhost:5173
- [ ] Home page loads with dataset selector
- [ ] Classify page accepts audio upload
- [ ] Dashboard shows training metrics (if model trained)
- [ ] Explorer shows t-SNE plot (if model trained)
- [ ] What-If tool loads spectrogram editor
- [ ] Dark mode toggle works (top-right)
- [ ] Sidebar navigation works on all pages

---

## Next Steps

Once everything is running:

1. **Explore the Demo** — Use pre-trained instrument model (audio/music)
2. **Train Your Own** — Try different datasets and hyperparameters
3. **Experiment with What-If** — Edit spectrograms/images to see how predictions change
4. **Compare Engines** — Train with `--compare-pytorch` to validate NumPy implementation
5. **Read the Docs** — Check `docs/hub.md` for technical deep-dives
6. **Extend the Project** — Add new datasets, architectures, or visualizations

---

## Quick Commands Reference

```bash
# Backend
cd api && uvicorn server:app --reload

# Frontend
cd web && npm run dev

# Train NumPy model
python train.py --dataset audio/music

# Train with PyTorch comparison
python train.py --dataset audio/music --compare-pytorch

# Train all datasets
for dataset in audio/music audio/medical audio/wildlife audio/urban image/medical image/wildlife; do
  python train.py --dataset $dataset --compare-pytorch
done

# Run tests
pytest tests/ -v

# Check code style
flake8 src/ api/ --max-line-length=100
```

---

## Resources

- **[README.md](README.md)** — Project overview, features, results
- **[docs/hub.md](docs/hub.md)** — Documentation hub, all step notes
- **[docs/math_derivations.md](docs/math_derivations.md)** — Backpropagation math
- **[docs/architecture.md](docs/architecture.md)** — Design decisions
- **[CHANGELOG.md](CHANGELOG.md)** — Version history, what changed

---

**Need Help?** Check the [troubleshooting section](#troubleshooting) above or open an issue on GitHub.

**Ready to Deploy?** See `docs/deployment.md` (if available) for production setup.
