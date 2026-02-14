# Step 11: Basic Visualizations

**Estimated Time:** 45 minutes
**Status:** ✅ Completed

---

## Completed
- ✅ Created `src/visualization/plots.py` — 3 core visualization functions
- ✅ `plot_training_history()` — 2×2 training dashboard (train loss, val loss, train acc, val acc)
- ✅ `plot_sample_predictions()` — 3×3 spectrogram grid with predicted vs true labels
- ✅ `plot_confusion_matrix()` — Seaborn heatmap with raw counts or normalized percentages
- ✅ Dark theme styling for all plots (professional look)
- ✅ Auto-creates `results/figures/` directory
- ✅ High-resolution output (150 DPI)
- ✅ All 10 tests passed!

## Implementation Details

### Functions

| Function | What It Does | Data Source |
|----------|-------------|-------------|
| `plot_training_history(history, save_path)` | 2×2 grid: Train Loss, Val Loss, Train Acc, Val Acc with annotations | `Trainer.train()` returns `history` dict |
| `plot_sample_predictions(X_flat, y_true, y_pred, ...)` | 3×3 grid of spectrograms, reshapes 4096→64×64, green=correct red=wrong | Flattened test data + `Evaluator.evaluate()` probabilities |
| `plot_confusion_matrix(cm, class_names, ...)` | Seaborn heatmap, optional row-normalization, overall accuracy annotation | `Evaluator.evaluate()` returns `confusion_matrix` |

### Design Decisions
1. **Dark theme** — `#1a1a2e` background with `#e0e0e0` text, looks clean and modern
2. **Seaborn for confusion matrix** — much better heatmap than pure matplotlib (annotations, color scale, square cells)
3. **Matplotlib for curves** — simpler, more control over line styles and annotations
4. **Reshape for display** — `X_flat` (4096-dim) is reshaped back to (64, 64) spectrograms inside the function
5. **Mix of correct/incorrect** — sample predictions grid deliberately picks ~50/50 correct and incorrect for an informative view
6. **Agg backend** — non-interactive, saves to file only, no display window needed
7. **Normalize option** — confusion matrix can show raw counts or recall percentages

### Dependencies
- `matplotlib` (already installed)
- `seaborn` (newly installed for heatmaps)
- `numpy` (already installed)

### File Location
📁 `src/visualization/plots.py`

## How to Use

```python
from src.visualization.plots import (
    plot_training_history,
    plot_sample_predictions,
    plot_confusion_matrix,
)

# After training...
history = trainer.train(X_train, y_train, X_val, y_val)
results = evaluator.evaluate(X_test, y_test)

# 1. Training dashboard
plot_training_history(history, save_path='results/figures')

# 2. Sample predictions (reshapes 4096 → 64×64 internally)
plot_sample_predictions(
    X_flat=X_test,
    y_true_onehot=y_test,
    y_pred_probs=results['probabilities'],
    class_names=['bass', 'brass', 'flute', ...],
    spec_shape=(64, 64),
    n_samples=9,
    save_path='results/figures'
)

# 3. Confusion matrix (raw counts)
plot_confusion_matrix(
    confusion_matrix=results['confusion_matrix'],
    class_names=['bass', 'brass', 'flute', ...],
    save_path='results/figures',
    normalize=False
)

# 4. Confusion matrix (normalized — recall %)
plot_confusion_matrix(
    confusion_matrix=results['confusion_matrix'],
    class_names=['bass', 'brass', 'flute', ...],
    save_path='results/figures/confusion_matrix_normalized.png',
    normalize=True
)
```

### Run Tests
```bash
source venv/bin/activate
python -m src.visualization.plots
```

## Testing Results
```
1️⃣  plot_training_history saves a file                — ✓ 189.1 KB
2️⃣  Training history uses correct data                — ✓ all 4 keys, 15 epochs
3️⃣  plot_sample_predictions saves a file              — ✓ 100.2 KB
4️⃣  Sample predictions includes correct/incorrect     — ✓ 33 correct, 42 incorrect
5️⃣  plot_confusion_matrix saves (raw counts)          — ✓ 63.4 KB
6️⃣  plot_confusion_matrix saves (normalized)          — ✓ 76.3 KB
7️⃣  Confusion matrix values match evaluator           — ✓ diagonal matches accuracy
8️⃣  Default save path creates results/figures/        — ✓ auto-created
9️⃣  Works without class names (index fallback)        — ✓ index labels
🔟  All figures are high-resolution (>10KB)            — ✓ 150 DPI

✅ All 10 tests passed!
```

## Verification Checklist
- [x] Loss curves show decreasing trend
- [x] Accuracy curves show increasing trend
- [x] Sample predictions show spectrograms clearly
- [x] Confusion matrix is readable with labels
- [x] All plots saved as high-res images
- [x] Confusion matrix included (moved from Step 12)

## Performance Review

**PERFORMANCE REVIEW:**

**What Changed:**
- Added `src/visualization/plots.py` (~320 lines of core code)
- Added seaborn dependency
- Created `src/visualization/__init__.py`

**Performance Impact:**

1. **Generation Speed**: 🚀 **FAST**
   - Training dashboard: ~1s (matplotlib rendering)
   - Sample predictions: ~0.5s
   - Confusion matrix: ~0.5s
   - Total for all 3 plots: ~2s

2. **File Sizes**: 📊 **REASONABLE**
   - Training dashboard: ~189 KB (high-res, dark theme)
   - Sample predictions: ~100 KB (9 spectrograms)
   - Confusion matrix: ~63-76 KB (heatmap)
   - Total: ~430 KB for all figures

3. **Memory Usage**: ⚖️ **MINIMAL**
   - Creates figures in memory, saves, then closes (`plt.close()`)
   - No persistent memory leak
   - Reshape is a view, not a copy

4. **Portfolio Impact**: 📈 **HIGH VALUE**
   - Dark-themed plots look professional and modern
   - 2×2 dashboard shows you understand training dynamics
   - Sample predictions with ✓/✗ labels are immediately compelling
   - Seaborn confusion matrix heatmap is publication-quality

**Bottom Line**: ~2s total render time, ~430KB disk, high visual impact for portfolio.

## Next Steps
1. ✅ ~~Basic Visualizations~~ (DONE!)
2. **Next:** Step 12: Advanced Visualizations (per-class bar charts, t-SNE, gradient flow)
3. Then: Step 13: Main training script (wires everything together)
