# Step 27: Interactive Training Dashboard

**Estimated Time:** 60 minutes
**Status:** ⬜ Not Started

---

## Goal
Replace the static PNG training plots with interactive Plotly charts. Users can hover over data points, zoom into regions, and click on confusion matrix cells to drill down into misclassifications.

## Tasks

### Backend
- [ ] Implement `GET /api/model/training-history` — returns training curves as JSON
- [ ] Implement `GET /api/model/confusion-matrix` — returns CM data + class names
- [ ] Implement `GET /api/model/class-metrics` — returns per-class P/R/F1/support
- [ ] Pre-compute and cache these at startup (they don't change)

### Frontend
- [ ] Create `Dashboard` page with grid layout for charts
- [ ] Create `TrainingCurves` component — interactive dual-axis line chart
  - Train loss + val loss (left axis)
  - Train accuracy + val accuracy (right axis)
  - Hover: show exact values at each epoch
  - Zoom: select region to zoom in
  - Annotation: mark early stopping point
- [ ] Create `ConfusionMatrix` component — interactive heatmap
  - Hover: show "True: X, Predicted: Y, Count: Z"
  - Click on a cell: show example spectrograms of that confusion pair
  - Toggle: raw counts vs. normalized (recall %)
  - Color scale: dark for low, bright for high
- [ ] Create `ClassMetrics` component — interactive grouped bar chart
  - Bars: precision, recall, F1 for each class
  - Hover: exact values
  - Macro F1 reference line
  - Sort toggle: by F1, by support, alphabetical
- [ ] Create `ModelInfo` card — architecture summary, param count, best accuracy

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
- [ ] Training curves show hover tooltips with exact values
- [ ] Zoom/pan works on training curves
- [ ] Early stopping point is annotated on the curve
- [ ] Confusion matrix hover shows true/predicted/count
- [ ] Confusion matrix toggle between raw and normalized works
- [ ] Per-class bar chart shows P/R/F1 correctly
- [ ] Bar chart sort toggles work
- [ ] All charts use dark theme matching the app
- [ ] Data loads fast (<500ms from API)
- [ ] Charts are responsive (resize with window)

## Dependencies
```json
{
  "plotly.js": "^2",
  "react-plotly.js": "^2"
}
```

## Next
Step 28: Interactive t-SNE & Feature Explorer
