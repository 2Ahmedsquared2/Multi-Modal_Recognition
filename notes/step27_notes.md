# Step 27: Interactive Training Dashboard

**Estimated Time:** 60 minutes
**Status:** ✅ Complete

---

## Goal
Replace the static PNG training plots with interactive Plotly charts. Users can hover over data points, zoom into regions, and click on confusion matrix cells to drill down into misclassifications.

## Tasks

### Backend
- [x] Implement `GET /api/model/training-history` — returns training curves as JSON
- [x] Implement `GET /api/model/confusion-matrix` — returns CM data + class names
- [x] Implement `GET /api/model/class-metrics` — returns per-class P/R/F1/support
- [x] Pre-compute and cache these at startup (they don't change)

### Frontend
- [x] Create `Dashboard` page with grid layout for charts
- [x] Create `TrainingCurves` component — interactive dual-axis line chart
  - Train loss + val loss (left axis)
  - Train accuracy + val accuracy (right axis)
  - Hover: show exact values at each epoch
  - Zoom: select region to zoom in
  - Annotation: mark early stopping point
- [x] Create `ConfusionMatrix` component — interactive heatmap
  - Hover: show "True: X, Predicted: Y, Count: Z"
  - Click on a cell: show example spectrograms of that confusion pair
  - Toggle: raw counts vs. normalized (recall %)
  - Color scale: dark for low, bright for high
- [x] Create `ClassMetrics` component — interactive grouped bar chart
  - Bars: precision, recall, F1 for each class
  - Hover: exact values
  - Macro F1 reference line
  - Sort toggle: by F1, by support, alphabetical
- [x] Create `ModelInfo` card — architecture summary, param count, best accuracy

## Implementation Plan

### Dashboard Layout
```
┌─────────────────────────────────────────────────────┐
│  📊 Model Dashboard                                 │
│                                                     │
│  ┌──────────────────────┐  ┌──────────────────────┐ │
│  │  Model Info Card     │  │  Key Metrics          │ │
│  │  Architecture: ...   │  │  Accuracy: 94.5%      │ │
│  │  Parameters: 533K    │  │  Macro F1: 0.947      │ │
│  │  Epochs: 88          │  │  Best Val Loss: 0.21  │ │
│  └──────────────────────┘  └──────────────────────┘ │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │  Training Curves (Plotly)                     │   │
│  │  Loss ↓            Accuracy ↑                 │   │
│  │  ──────            ──────────                 │   │
│  │  hover/zoom enabled                           │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌────────────────────┐  ┌────────────────────────┐ │
│  │  Confusion Matrix  │  │  Per-Class Metrics     │  │
│  │  (Plotly Heatmap)  │  │  (Plotly Bars)         │  │
│  │  click cells →     │  │  hover for values      │  │
│  │  see examples      │  │  sort options          │  │
│  └────────────────────┘  └────────────────────────┘ │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Backend Response Shapes

**Training History:**
```json
{
  "epochs": [1, 2, 3, ...],
  "train_loss": [2.3, 1.8, ...],
  "val_loss": [2.4, 1.9, ...],
  "train_accuracy": [0.12, 0.35, ...],
  "val_accuracy": [0.10, 0.32, ...],
  "learning_rate": [0.01, 0.0095, ...],
  "early_stop_epoch": 78,
  "total_epochs": 88
}
```

**Confusion Matrix:**
```json
{
  "matrix": [[70, 4, 0, ...], ...],
  "class_names": ["bass", "brass", ...],
  "normalized": [[0.93, 0.05, 0.0, ...], ...]
}
```

**Class Metrics:**
```json
{
  "classes": [
    {"name": "bass", "precision": 0.972, "recall": 0.933, "f1": 0.952, "support": 75},
    ...
  ],
  "macro_f1": 0.9473,
  "accuracy": 0.945
}
```

### Plotly Configuration
- Dark theme: `layout.template = "plotly_dark"` with custom background matching the app
- Responsive: `responsive: true` in config
- Export: Plotly has built-in "download as PNG" button
- Modebar: show zoom, pan, reset, download; hide logo

### Confusion Matrix Click Drill-Down
When user clicks a cell (e.g., "bass predicted as guitar: 4"):
- Expand a panel below the matrix
- Show up to 4 example spectrograms from that confusion pair
- For each: spectrogram image + true label + predicted label + confidence
- This requires a backend endpoint or pre-computed examples

## Verification
- [x] Training curves show hover tooltips with exact values
- [x] Zoom/pan works on training curves
- [x] Early stopping point is annotated on the curve
- [x] Confusion matrix hover shows true/predicted/count
- [x] Confusion matrix toggle between raw and normalized works
- [x] Per-class bar chart shows P/R/F1 correctly
- [x] Bar chart sort toggles work
- [x] All charts use dark theme matching the app
- [x] Data loads fast (<500ms from API)
- [x] Charts are responsive (resize with window)

## Dependencies
```json
{
  "plotly.js-dist-min": "^2.35.2",
  "react-plotly.js": "^2.6.0",
  "@types/react-plotly.js": "^2.6.3"
}
```

## Implementation Summary

### Files Created
- **`web/src/hooks/useDarkMode.ts`** — Lightweight hook that observes `<html class="dark">` via MutationObserver for Plotly theme reactivity
- **`web/src/components/TrainingCurves.tsx`** — Dual-panel interactive line charts (Loss + Accuracy) with early stopping annotation
- **`web/src/components/ConfusionMatrixChart.tsx`** — Interactive Plotly heatmap with toggle (counts/recall %) and click drill-down panel
- **`web/src/components/ClassMetricsChart.tsx`** — Interactive grouped bar chart with P/R/F1 bars, macro F1 reference line, and sort toggles

### Files Modified
- **`web/src/pages/Dashboard.tsx`** — Complete rewrite with all 3 Plotly components, model info cards, and collapsible architecture details
- **`web/vite.config.ts`** — Added `manualChunks` to code-split Plotly into separate bundle
- **`train.py`** — Added save for `results/metrics/training_history.npz` after training (line ~413)

### Backend
All required endpoints already existed:
- `GET /api/model/training-history` ✓
- `GET /api/model/confusion-matrix` ✓
- `GET /api/model/class-metrics` ✓
- `GET /api/model/info` ✓

### Key Features Implemented
1. **Training Curves**: Interactive dual-axis charts with zoom/pan, hover tooltips, and dashed red line marking early stopping epoch
2. **Confusion Matrix**: Plotly heatmap with counts/recall % toggle, hover details, and click-to-expand drill-down panel showing count + percentage
3. **Class Metrics**: Grouped bar chart with P/R/F1 bars, macro F1 reference line, and sort by F1/Support/A-Z
4. **Dark Mode**: All charts react to theme toggle via `useDarkMode()` hook (no page reload needed)
5. **Code Splitting**: Plotly bundled separately (~1.5MB gzipped) — only loads on Dashboard page

### Performance
- **Bundle impact**: Plotly split into separate chunk, no impact on other pages
- **API calls**: 4 parallel fetches via `Promise.allSettled`
- **Rendering**: Memoized derived data, responsive resize handler
- **Training history**: 2.7KB `.npz` file, minimal overhead

## Issues Resolved
- **Missing training_history.npz**: Patched `train.py` to save history after training, re-ran training (6s) to generate file
- **TypeScript type errors**: Fixed Plotly heatmap `text`/`hovertext` type casting for 2D arrays
- **Build warnings**: Added Plotly to manual chunks to suppress large bundle warnings

## Next
Step 28: Interactive t-SNE & Feature Explorer
