# Step 28: Interactive t-SNE & Feature Explorer

**Estimated Time:** 60 minutes
**Status:** ✅ Complete
**Actual Time:** ~55 minutes

---

## Goal
Build an interactive t-SNE scatter plot where users can hover over points to see the spectrogram, click to hear the audio, filter by class, and explore how the neural network organizes instruments in its learned feature space.

## Tasks

### Backend
- [x] Implement `GET /api/model/tsne` — returns t-SNE coordinates + metadata for all test samples
- [x] Pre-compute t-SNE at startup (or cache to disk) — it's slow to compute on the fly
- [x] Each point includes: x, y, true_label, predicted_label, correct, confidence ~~spectrogram_thumbnail, sample_id~~
- [x] ~~Implement `GET /api/model/tsne/sample/{id}` — returns full spectrogram + audio path for a specific point~~ (deferred to What-If tool)

### Frontend
- [x] Create `Explorer` page with the t-SNE scatter as the main feature
- [x] Create `TSNEPlot` component using Plotly scatter
  - Each point colored by instrument class (10 distinct colors)
  - Hover: show instrument name, predicted label, confidence
  - Click: open detail panel with labels + confidence
  - Incorrect predictions highlighted (red outline + diamond marker)
- [x] Create class filter panel
  - Checkboxes for each instrument class
  - Toggle all on/off
  - Click a class name to isolate just that class
- [x] Create detail panel (appears on click)
  - ~~Full spectrogram display~~ (coordinates shown instead)
  - ~~Audio playback~~ (deferred to What-If tool)
  - True vs. predicted label
  - Confidence breakdown
  - ~~"Why did the model think X?" → link to What-If tool~~ (can be added in Step 29)

## Implementation Summary

### What Was Built

#### Backend Changes

1. **`src/visualization/attention.py`** — Enhanced `visualize_tsne()`
   - Now runs forward pass on test data to get predictions + confidences
   - Saves raw coordinates to `results/metrics/tsne_data.npz` alongside PNG
   - Includes: coords (N×2), labels (N), predictions (N), confidences (N), class_names

2. **`api/schemas.py`** — Enhanced `TSNEPoint` schema
   - Added: `predicted_label`, `predicted_idx`, `correct`, `confidence`
   - `TSNEResponse` now includes: `n_samples`, `accuracy`

3. **`api/server.py`** — Enhanced `/api/model/tsne` endpoint
   - Reads pre-computed npz file
   - Serves richer per-point metadata
   - Returns overall accuracy from test set

4. **`generate_tsne.py`** — New standalone script
   - Loads trained model + test data
   - Generates t-SNE data (~17 seconds one-time cost)
   - Output: 505 samples, 10 classes, 94.5% accuracy, 28 errors

#### Frontend Changes

5. **`web/src/types.ts`** — Updated TypeScript types
   - `TSNEPoint` now matches backend schema
   - `TSNEData` includes n_samples and accuracy

6. **`web/src/components/TSNEPlot.tsx`** — New Plotly component
   - One scatter trace per class (10 distinct colors)
   - Circle markers for correct predictions
   - Diamond markers for incorrect predictions
   - Red outline on misclassifications
   - White highlight ring on selected point
   - Pan/zoom/scroll support
   - Hover tooltip: class name + predicted label + confidence

7. **`web/src/pages/Explorer.tsx`** — Complete rewrite
   - **Summary cards**: total samples, accuracy, error count
   - **Interactive Plotly scatter**: 505 points, responsive, clickable
   - **Class filter sidebar**:
     - Checkboxes to toggle each class on/off
     - "All / None" quick buttons
     - Per-class "only" isolate button (on hover)
     - Per-class sample counts
   - **"Show Errors Only" toggle**: isolates 28 misclassified samples
   - **Detail panel** (on click):
     - True vs predicted label
     - Correct/Wrong badge
     - Confidence bar (green for correct, red for wrong)
     - Coordinates display
     - Explanation text for misclassifications

### Color Palette (Actual Implementation)
```typescript
bass:     #38bdf8  (sky-400)
brass:    #fb7185  (rose-400)
flute:    #fbbf24  (amber-400)
guitar:   #fb923c  (orange-400)
keyboard: #94a3b8  (slate-400)
mallet:   #a78bfa  (violet-400)
organ:    #ef4444  (red-500)
reed:     #2dd4bf  (teal-400)
string:   #34d399  (emerald-400)
vocal:    #818cf8  (indigo-400)
```

### Data Flow

```
generate_tsne.py
    ↓
results/metrics/tsne_data.npz (505 samples, cached)
    ↓
GET /api/model/tsne (reads npz, instant)
    ↓
Explorer.tsx → TSNEPlot (Plotly renders)
```

### Explorer Page Layout
```
┌─────────────────────────────────────────────────────┐
│  🔍 Feature Space Explorer                          │
│                                                     │
│  ┌────────────┐  ┌──────────────────────────────┐   │
│  │ Filters    │  │                              │   │
│  │            │  │      t-SNE Scatter Plot       │   │
│  │ ☑ bass    │  │        (Plotly)               │   │
│  │ ☑ brass   │  │                              │   │
│  │ ☑ flute   │  │     ● ●   ● ●               │   │
│  │ ☑ guitar  │  │   ●     ●     ●  ●          │   │
│  │ ☑ keyboard│  │      ●●    ●                 │   │
│  │ ☑ mallet  │  │   ●      ●   ●              │   │
│  │ ☑ organ   │  │     ●  ●      ●             │   │
│  │ ☑ reed    │  │                              │   │
│  │ ☑ string  │  │                              │   │
│  │ ☑ vocal   │  │                              │   │
│  │            │  │                              │   │
│  │ [Show All] │  │  ● correct  ◆ incorrect     │   │
│  │ [Errors]   │  │                              │   │
│  └────────────┘  └──────────────────────────────┘   │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │  Selected Sample Detail (appears on click)    │   │
│  │  ┌────────────┐                               │   │
│  │  │ spectrogram│  True: guitar                 │   │
│  │  │  [64×64]   │  Predicted: guitar ✓          │   │
│  │  │            │  Confidence: 92.3%            │   │
│  │  └────────────┘  [▶ Play Audio]               │   │
│  │                   [Open in What-If →]          │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Backend t-SNE Response
```json
{
  "points": [
    {
      "id": 0,
      "x": -12.34,
      "y": 5.67,
      "true_label": "guitar",
      "predicted_label": "guitar",
      "correct": true,
      "confidence": 0.923,
      "true_label_index": 3,
      "spectrogram_thumbnail": "base64..."
    },
    ...
  ],
  "class_names": ["bass", "brass", ...],
  "n_samples": 505,
  "accuracy": 0.945
}
```

### Color Scheme (10 classes)
```
bass:     #FF6B6B (red)
brass:    #FFA500 (orange)
flute:    #FFD93D (yellow)
guitar:   #6BCB77 (green)
keyboard: #4D96FF (blue)
mallet:   #9B59B6 (purple)
organ:    #E91E8C (pink)
reed:     #00CEC9 (teal)
string:   #FD7272 (coral)
vocal:    #A29BFE (lavender)
```

### Plotly Scatter Configuration
```typescript
const traces = classNames.map((cls, i) => ({
  x: points.filter(p => p.true_label === cls).map(p => p.x),
  y: points.filter(p => p.true_label === cls).map(p => p.y),
  mode: 'markers',
  type: 'scatter',
  name: cls,
  marker: { size: 8, color: colors[i], opacity: 0.8 },
  hovertemplate: '%{customdata.label}<br>Conf: %{customdata.conf:.1%}<extra></extra>',
}));
```

### Interaction Details
- **Hover**: Plotly native hover shows class name + confidence
- **Click**: `plotly_click` event → set selected sample ID → show detail panel
- **Filter**: Toggle traces on/off via custom checkboxes (Plotly's legend click is too small)
- **"Show Errors" button**: Isolates only misclassified points

### Performance Consideration
- t-SNE computation on 505 test samples takes ~2-5 seconds
- Pre-compute at server startup and cache in memory
- Or pre-compute once and save to `results/tsne_data.json`
- Frontend just fetches JSON — instant render

## Verification
- [x] t-SNE scatter renders with correct colors per class
- [x] Hover shows class name, predicted label, and confidence
- [x] Click opens detail panel ~~with spectrogram~~ with labels + confidence
- [ ] ~~Audio playback works from detail panel~~ (deferred to What-If tool)
- [x] Class filters toggle visibility of points
- [x] "Show Errors" highlights misclassified samples
- [x] Plot is responsive (resizes with window)
- [x] Points load fast (<1s from cached data)
- [x] Dark theme matches rest of the app

## Performance Notes

- **t-SNE computation**: ~17 seconds (one-time, cached to disk)
- **API response**: Instant (~50KB npz file read)
- **Frontend render**: Near-instant (505 points, Plotly is fast)
- **Plotly bundle**: Already code-split in `vite.config.ts`
- **No startup cost**: Backend just reads a pre-computed file

## What's Missing (Optional Enhancements)

These were in the original spec but simplified for MVP:

1. **Spectrogram thumbnails in hover**: Would add ~1MB to the npz file and slow down initial load. Can add later if needed.

2. **Audio playback from detail panel**: Requires storing test audio paths in the npz or a separate lookup table. Better suited for the What-If tool (Step 29).

3. **Per-sample endpoint** (`/api/model/tsne/sample/{id}`): Not needed since we send all data upfront. Can add if we want lazy-loading for larger datasets.

4. **Link to What-If tool**: Can be added in Step 29 once that page exists.

## Usage

1. **Generate t-SNE data** (if not already done):
   ```bash
   python generate_tsne.py
   ```

2. **Start backend**:
   ```bash
   cd api && uvicorn api.server:app --reload
   ```

3. **Start frontend**:
   ```bash
   cd web && npm run dev
   ```

4. **Navigate to `/explorer`** and interact:
   - Hover over points to see details
   - Click to open detail panel
   - Use class filters to isolate specific instruments
   - Toggle "Show Errors Only" to focus on misclassifications
   - Pan/zoom the scatter plot

## Issues Encountered

None. Implementation went smoothly.

## Next
Step 29: What-If Tool
