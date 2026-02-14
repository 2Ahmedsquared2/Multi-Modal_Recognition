# Step 24: Waveform & Spectrogram Display

**Estimated Time:** 45 minutes
**Status:** ⬜ Not Started

---

## Goal
After classification, show the user the audio waveform and the generated mel-spectrogram side by side. This visualizes "what the model sees" and makes the signal processing pipeline tangible.

## Tasks

### Backend
- [ ] Include waveform data in classify response (downsampled for display, ~500 points)
- [ ] Include spectrogram as 2D array in classify response (64×64 values)
- [ ] Add `GET /api/spectrogram` standalone endpoint for spectrogram-only generation

### Frontend
- [ ] Create `WaveformDisplay` component — plots the raw audio waveform
- [ ] Create `SpectrogramDisplay` component — renders the 64×64 spectrogram as a heatmap
- [ ] Integrate both into the Classify page results section
- [ ] Add axis labels (Time, Frequency, Amplitude)
- [ ] Color scale for spectrogram (viridis or magma colormap)
- [ ] Audio playback button — user can listen to their uploaded file

## Implementation Plan

### Layout (Added to Classify Results)
```
┌────────────────────────────────────────────────────┐
│  🎸 Guitar — 92.3% confidence                     │
│  [confidence bars...]                              │
│                                                    │
│  ┌──────────────────┐  ┌──────────────────┐        │
│  │   📊 Waveform    │  │ 🔥 Spectrogram   │        │
│  │                  │  │                  │        │
│  │  ∿∿∿∿∿∿∿∿∿∿∿∿  │  │  [heatmap grid]  │        │
│  │                  │  │                  │        │
│  │  Time →          │  │  Time →          │        │
│  │  ▶ Play Audio    │  │  ↑ Frequency     │        │
│  └──────────────────┘  └──────────────────┘        │
│                                                    │
│  "What the model sees: your audio transformed      │
│   into a mel-spectrogram for classification"       │
│                                                    │
└────────────────────────────────────────────────────┘
```

### Waveform Component
- Use Recharts `AreaChart` or Canvas for the waveform
- Downsample to ~500 points (backend does this to keep response small)
- Show amplitude on Y-axis, time on X-axis
- Light blue fill under the curve on dark background

### Spectrogram Component
- Use Plotly heatmap (`Plotly.Heatmap`) for interactivity (hover shows value)
- Or Canvas for faster rendering
- Colormap: magma (dark → orange → white) matches audio visualization convention
- Y-axis: Mel frequency bands (low → high)
- X-axis: Time frames
- Hover tooltip: "Frequency: X Hz, Time: Y ms, Energy: Z"

### Audio Playback
- Create an `<audio>` element with the uploaded file as source
- Play/pause button with waveform scrubber
- Shows current playback position on the waveform chart
- Uses `URL.createObjectURL(file)` — no backend needed for playback

### Backend Response Update
```json
{
  "prediction": "guitar",
  "confidence": 0.92,
  "all_confidences": { ... },
  "spectrogram": [[0.1, 0.3, ...], ...],
  "waveform": [0.01, -0.03, 0.05, ...],
  "audio_info": {
    "duration_seconds": 2.0,
    "sample_rate": 22050,
    "n_samples": 44100
  }
}
```

## Verification
- [ ] Waveform renders correctly for uploaded audio
- [ ] Spectrogram renders as a clear heatmap
- [ ] Axis labels are present and readable
- [ ] Audio playback works (play/pause)
- [ ] Hover on spectrogram shows frequency/time/energy
- [ ] Both visualizations update when a new file is uploaded
- [ ] Responsive layout — stacks vertically on narrow screens

## Design Notes
- These two panels together tell the story: "raw sound → what the model sees"
- The spectrogram should look visually similar to the dark-themed spectrograms from Phase 1
- Add a small caption explaining what a mel-spectrogram is (one sentence, not a paragraph)

## Next
Step 25: Live Microphone Input
