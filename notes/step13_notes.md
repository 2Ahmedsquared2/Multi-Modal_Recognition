# Step 13: Main Training Script

**Estimated Time:** 20 minutes
**Status:** ✅ Completed

---

## Completed
- ✅ Created `train.py` — full end-to-end pipeline at project root
- ✅ Argparse CLI with all configurable hyperparameters
- ✅ 3-tier data loading fallback (prepared → spectrograms → raw audio)
- ✅ Builds NeuralNetwork, trains with Trainer, evaluates with Evaluator
- ✅ Generates all 8 visualizations (training curves, predictions, confusion matrices, attention, t-SNE, gradient flow, per-class metrics)
- ✅ Saves trained model to `models/model.npz`
- ✅ Added `save()` and `load()` methods to NeuralNetwork class
- ✅ Full pipeline runs in ~13 seconds on CPU
- ✅ **94.5% test accuracy, 0.9473 macro F1** on first run!

## Implementation Details

### Pipeline Stages
1. **Data Loading** — 3-tier fallback:
   - **Fastest:** Load pre-prepared data from `data/prepared/prepared_data.npz`
   - **Medium:** Load spectrograms from `data/spectrograms/spectrograms.npz` → run DataPreparator
   - **Full rebuild:** Load raw audio → generate spectrograms → prepare data
2. **Training** — Build NeuralNetwork + Trainer, run training loop
3. **Evaluation** — Run Evaluator on test set, print classification report + confusion matrix
4. **Visualization** — Generate all 8 figures to `results/figures/`
5. **Save Model** — Save weights to `models/model.npz`

### CLI Arguments
```bash
python train.py                           # Default settings
python train.py --lr 0.005 --epochs 150   # Custom hyperparameters
python train.py --hidden 256 128 64       # Deeper network
python train.py --no-viz                  # Skip visualizations
python train.py --prepared-data data/prepared/prepared_data.npz  # Explicit path
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | 0.01 | Initial learning rate |
| `--lr-decay` | 0.95 | LR decay per epoch |
| `--batch-size` | 32 | Mini-batch size |
| `--epochs` | 100 | Max epochs |
| `--patience` | 10 | Early stopping patience |
| `--hidden` | 128 64 | Hidden layer sizes |
| `--resolution` | 64 | Spectrogram NxN |
| `--seed` | 42 | Random seed |
| `--no-viz` | off | Skip visualizations |

### Model Save/Load (added to NeuralNetwork)
```python
# Save after training
net.save("models")  # → models/model.npz

# Load for inference
net = NeuralNetwork.load("models/model.npz")
predictions = net.predict(X_test)
```

### Files Modified
- **Created:** `train.py` (project root)
- **Modified:** `src/neural_network/network.py` (added `save()` and `load()` methods)

## Training Results (First Run)

### Configuration
- Architecture: 4096 → 128 → 64 → 10 (533,322 parameters)
- Learning rate: 0.01 (decay: 0.95/epoch)
- Batch size: 32
- Early stopping: patience=10
- Seed: 42

### Results
```
Epochs trained:      88 (early stopped at 78 best val loss)
Training accuracy:   99.8%
Validation accuracy: 91.8%
Test accuracy:       94.5%
Macro F1:            0.9473
Total pipeline time: 12.7s
```

### Per-Class Test Performance
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| bass | 0.972 | 0.933 | 0.952 | 75 |
| brass | 1.000 | 0.976 | 0.988 | 41 |
| flute | 0.964 | 0.964 | 0.964 | 28 |
| guitar | 0.885 | 0.920 | 0.902 | 75 |
| keyboard | 0.908 | 0.920 | 0.914 | 75 |
| mallet | 0.867 | 0.839 | 0.852 | 31 |
| organ | 0.973 | 0.960 | 0.966 | 75 |
| reed | 1.000 | 1.000 | 1.000 | 36 |
| string | 0.959 | 1.000 | 0.979 | 47 |
| vocal | 0.955 | 0.955 | 0.955 | 22 |

### Top Misclassifications
- bass → guitar: 4 times
- guitar → keyboard: 4 times
- keyboard → guitar: 3 times

### Generated Visualizations (8 files, ~1.3 MB total)
| File | Size |
|------|------|
| training_history.png | 157 KB |
| sample_predictions.png | 205 KB |
| confusion_matrix.png | 93 KB |
| confusion_matrix_normalized.png | 110 KB |
| attention_maps.png | 366 KB |
| tsne_features.png | 176 KB |
| gradient_flow.png | 97 KB |
| per_class_metrics.png | 76 KB |

## Performance Review

**PERFORMANCE REVIEW:**

**What Changed:**
- Created `train.py` wiring together all 12 previous steps
- Added `save()`/`load()` to NeuralNetwork (~70 lines)
- No new algorithms — pure orchestration

**Performance Impact:**

1. **Pipeline Speed**: ⚡ **12.7 seconds total**
   - Data loading (cached): <1s
   - Training (88 epochs): ~8s (~0.09s/epoch)
   - Evaluation: <0.1s
   - Visualizations (8 plots): ~3s
   - Model saving: <0.1s

2. **Model Quality**: 📈 **EXCEEDED EXPECTATIONS**
   - Test accuracy: 94.5% (target was 70-85%)
   - Macro F1: 0.9473 (excellent balance across classes)
   - Reed: perfect 100% recall + precision
   - Only weak spot: mallet (F1=0.852) — smallest class (31 test samples)

3. **Disk Usage**: 💾 **MINIMAL**
   - Model: 3.9 MB
   - Figures: 1.3 MB
   - Prepared data (cached): 34.8 MB

4. **Usability**: ✅ **EXCELLENT**
   - Single command to run everything: `python train.py`
   - Argparse for experimentation without code changes
   - 3-tier fallback handles any data state gracefully
   - `--no-viz` flag for fast iteration

**Bottom Line**: One command, 13 seconds, 94.5% accuracy, 8 publication-quality visualizations, saved model. The pipeline is production-ready.

## Next Steps
1. ✅ ~~Main training script~~ (DONE!)
2. **Next:** Step 14: Documentation (README, math derivations, architecture docs)
3. Then: Step 15: Testing & Debugging, Step 16: Portfolio Materials
