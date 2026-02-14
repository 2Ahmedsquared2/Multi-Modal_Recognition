# Step 28: Interactive t-SNE & Feature Explorer

**Estimated Time:** 60 minutes
**Status:** ⬜ Not Started

---

## Goal
Build an interactive t-SNE scatter plot where users can hover over points to see the spectrogram, click to hear the audio, filter by class, and explore how the neural network organizes instruments in its learned feature space.

## Tasks

### Backend
- [ ] Implement `GET /api/model/tsne` — returns t-SNE coordinates + metadata for all test samples
- [ ] Pre-compute t-SNE at startup (or cache to disk) — it's slow to compute on the fly
- [ ] Each point includes: x, y, true_label, predicted_label, correct, confidence, spectrogram_thumbnail, sample_id
- [ ] Implement `GET /api/model/tsne/sample/{id}` — returns full spectrogram + audio path for a specific point

### Frontend
- [ ] Create `Explorer` page with the t-SNE scatter as the main feature
- [ ] Create `TSNEPlot` component using Plotly scatter
  - Each point colored by instrument class (10 distinct colors)
  - Hover: show instrument name, confidence, tiny spectrogram thumbnail
  - Click: open detail panel with full spectrogram + play audio
  - Incorrect predictions highlighted (red ring or different marker)
- [ ] Create class filter panel
  - Checkboxes for each instrument class
  - Toggle all on/off
  - Click a class name to isolate just that class
- [ ] Create detail panel (appears on click)
  - Full spectrogram display
  - Audio playback
  - True vs. predicted label
  - Confidence breakdown
  - "Why did the model think X?" → link to What-If tool

## Implementation Plan

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
- [ ] t-SNE scatter renders with correct colors per class
- [ ] Hover shows class name and confidence
- [ ] Click opens detail panel with spectrogram
- [ ] Audio playback works from detail panel
- [ ] Class filters toggle visibility of points
- [ ] "Show Errors" highlights misclassified samples
- [ ] Plot is responsive (resizes with window)
- [ ] Points load fast (<1s from cached data)
- [ ] Dark theme matches rest of the app

## Next
Step 29: What-If Tool
