# Step 29: What-If Tool

**Estimated Time:** 90 minutes
**Status:** ⬜ Not Started

---

## Goal
Build a canvas-based spectrogram editor where users can paint, erase, or zero out frequency regions on a spectrogram and instantly see how the model's prediction changes. This is the most unique and memorable feature of the entire app.

## Tasks

### Backend
- [ ] Implement `POST /api/what-if` — accepts a modified 64×64 spectrogram, returns new prediction
  - Input: 2D array (64×64) of spectrogram values
  - Processing: flatten, normalize (same mean/std as training), forward pass
  - Output: prediction, confidence, all class scores
- [ ] Fast response time target: <100ms (it's just a forward pass)

### Frontend
- [ ] Create `WhatIf` page with the spectrogram editor as centerpiece
- [ ] Create `SpectrogramEditor` component using HTML Canvas
  - Displays a spectrogram as a colored grid (magma colormap)
  - Brush tools: paint (increase energy), erase (decrease energy), zero-out (set to 0)
  - Brush size: small, medium, large (adjustable slider)
  - Region selection: draw rectangle to zero out entire frequency band
- [ ] Live prediction panel that updates as user edits
  - Shows current prediction + confidence bars
  - Updates in real-time (debounced, ~200ms after last edit)
  - Highlights what changed from original prediction
- [ ] Before/After comparison view
  - Original spectrogram + prediction on the left
  - Modified spectrogram + new prediction on the right
  - Visual diff: highlight changed regions
- [ ] Preset experiments (one-click demos):
  - "Remove low frequencies" (zero out bottom 1/3)
  - "Remove high frequencies" (zero out top 1/3)
  - "Keep only mid-range" (zero out top and bottom)
  - "Add noise" (random values in a region)
  - "Reset to original"
- [ ] Load spectrogram from: upload new audio, select from test set, or start from the Classify page result

## Implementation Plan

### Page Layout
```
┌──────────────────────────────────────────────────────────┐
│  🔬 What-If Tool — Explore What the Model Sees           │
│                                                          │
│  Source: [Upload Audio ▾]  [From Test Set ▾]  [Classify] │
│                                                          │
│  ┌─────────────────────┐  ┌─────────────────────┐       │
│  │  Original           │  │  Modified (editable) │       │
│  │  ┌───────────────┐  │  │  ┌───────────────┐  │       │
│  │  │               │  │  │  │    (canvas)    │  │       │
│  │  │  spectrogram  │  │  │  │  draw/erase    │  │       │
│  │  │  (read-only)  │  │  │  │  here          │  │       │
│  │  │               │  │  │  │               │  │       │
│  │  └───────────────┘  │  │  └───────────────┘  │       │
│  │                     │  │                     │       │
│  │  🎸 Guitar  92.3%   │  │  🎹 Keyboard  67.1%│       │
│  │  [confidence bars]  │  │  [confidence bars]  │       │
│  └─────────────────────┘  └─────────────────────┘       │
│                                                          │
│  Tools: [🖌 Paint] [🧹 Erase] [⬛ Zero] [📐 Region]     │
│  Brush: ○ Small  ● Medium  ◉ Large                      │
│                                                          │
│  Presets: [🔇 Remove Low Freq] [🔇 Remove High Freq]     │
│           [🎯 Keep Mid-Range]  [🔄 Reset]                │
│                                                          │
│  Insight: "Removing high frequencies shifted the         │
│  prediction from Guitar (92%) to Keyboard (67%).         │
│  This suggests the model uses high-frequency             │
│  harmonics to distinguish these instruments."            │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Canvas Spectrogram Editor

**Rendering:**
- 64×64 spectrogram → scale up to ~512×512 px canvas (8× zoom)
- Each spectrogram cell = 8×8 pixel block on canvas
- Color mapping: value 0.0 = dark/black, value 1.0 = bright/white (magma colormap)

**Interaction:**
```typescript
// Mouse/touch events on canvas
canvas.onMouseDown → start painting
canvas.onMouseMove → if painting, update cells under brush
canvas.onMouseUp   → stop painting, trigger re-prediction

// Brush applies to spectrogram grid coordinates
const gridX = Math.floor(canvasX / cellSize);  // 0-63
const gridY = Math.floor(canvasY / cellSize);  // 0-63
```

**Brush Tools:**
| Tool | Action | Cursor |
|------|--------|--------|
| Paint | Set value to 1.0 (max energy) | Circle with + |
| Erase | Set value to 0.0 (no energy) | Circle with - |
| Zero | Same as erase but for regions | Rectangle drag |
| Region Select | Select rectangle → apply operation | Dashed rectangle |

**Brush Size:**
| Size | Radius (grid cells) | Pixel area |
|------|---------------------|-----------|
| Small | 1 cell | 8×8 px |
| Medium | 3 cells | 24×24 px |
| Large | 6 cells | 48×48 px |

### Real-Time Re-Prediction

```typescript
// Debounced prediction update
const debouncedPredict = useMemo(
  () => debounce(async (spectrogram: number[][]) => {
    const result = await api.whatIf(spectrogram);
    setModifiedPrediction(result);
  }, 200),
  []
);

// Called on every canvas edit
const handleCanvasEdit = (x: number, y: number, value: number) => {
  const newSpec = [...modifiedSpectrogram];
  newSpec[y][x] = value;
  setModifiedSpectrogram(newSpec);
  debouncedPredict(newSpec);
};
```

### Insight Generation (Frontend Logic)
After each re-prediction, compare original vs. modified:
```typescript
if (originalPrediction !== modifiedPrediction) {
  setInsight(`Removing those frequencies shifted the prediction from 
    ${original} (${origConf}%) to ${modified} (${modConf}%). 
    This suggests the model relies on that frequency range to identify ${original}.`);
} else if (modifiedConfidence < originalConfidence - 0.1) {
  setInsight(`The prediction is still ${original}, but confidence dropped from 
    ${origConf}% to ${modConf}%. Those frequencies contribute to the model's certainty.`);
} else {
  setInsight(`The prediction is unchanged. Those frequencies don't significantly 
    affect the model's decision.`);
}
```

### Colormap Implementation (Magma)
```typescript
// Simplified magma colormap (5-point interpolation)
function magmaColor(value: number): [number, number, number] {
  const stops = [
    [0.0, [0, 0, 4]],        // near-black
    [0.25, [81, 18, 124]],   // deep purple
    [0.5, [183, 55, 121]],   // magenta
    [0.75, [252, 137, 97]],  // orange
    [1.0, [252, 253, 191]],  // pale yellow
  ];
  // Interpolate between nearest stops
  // ...
}
```

## Verification
- [ ] Canvas renders spectrogram correctly (colors match values)
- [ ] Paint tool increases values (bright spots appear)
- [ ] Erase tool decreases values (dark spots appear)
- [ ] Region zero-out works (select rectangle → all zeros)
- [ ] Prediction updates within ~300ms of last edit
- [ ] Before/after comparison shows original vs. modified
- [ ] Presets work correctly (remove low/high freq, reset)
- [ ] Insight text updates with meaningful explanations
- [ ] Reset returns to exact original spectrogram
- [ ] Can load spectrogram from different sources (upload, test set, classify result)
- [ ] No canvas performance issues with rapid painting

## Why This Feature Matters
- Almost no student portfolio has anything like this
- It demonstrates *interpretability* — you don't just classify, you explain
- It's interactive and playful — people will spend time with it
- It shows product thinking: "What would a user want to explore?"
- In an interview: "I built a tool that lets you manipulate what the model sees and watch its mind change in real time"

## Next
Step 30: Polish & Deployment
