# Step 24: Waveform & Spectrogram Display

**Estimated Time:** 45 minutes
**Status:** ✅ Complete

---

## Goal
After classification, show the user the audio waveform and the generated mel-spectrogram side by side. This visualizes "what the model sees" and makes the signal processing pipeline tangible.

## Tasks

### Backend
- [x] Include waveform data in classify response (downsampled for display, ~500 points)
- [x] Include spectrogram as 2D array in classify response (64×64 values)
- [x] `POST /api/spectrogram` standalone endpoint already exists from Step 21

### Frontend
- [x] Create `WaveformDisplay` component — plots the raw audio waveform
- [x] Create `SpectrogramDisplay` component — renders the 64×64 spectrogram as a heatmap
- [x] Integrate both into the Classify page results section
- [x] Add axis labels (Time, Frequency, Amplitude)
- [x] Color scale for spectrogram (magma colormap)
- [x] Audio playback — user can preview segment via AudioTrimmer before classifying

## What Was Built

### Backend — `api/model_service.py`
- Added `_downsample_waveform()` static method:
  - Uses min/max envelope bucketing to preserve peaks
  - Reduces full waveform (44,100 samples) to ~500 points
  - Returns pairs of (min, max) per bucket for accurate envelope shape
- `classify()` now returns a `waveform` field alongside `spectrogram`

### Backend — `api/schemas.py`
- Added `waveform: List[float]` to `ClassifyResponse`

### Frontend — `web/src/types.ts`
- Added `waveform: number[]` to `ClassifyResult` interface

### Frontend — `web/src/components/WaveformDisplay.tsx`
- Canvas-based waveform renderer (no library dependency)
- Indigo gradient fill under/over the center line
- Stroke line for the waveform envelope
- Time axis labels (0s, midpoint, end)
- HiDPI-aware (devicePixelRatio scaling)

### Frontend — `web/src/components/SpectrogramDisplay.tsx`
- Canvas-based heatmap renderer (no library dependency)
- Magma colormap (12-stop dark → purple → orange → yellow-white)
- Frequency axis flipped so low freq is at bottom
- Hover tooltip showing time, frequency bin, and energy value
- Colored tooltip labels (indigo=time, amber=freq, emerald=energy)
- Italic caption: "What the model sees"

### Frontend — `web/src/pages/Classify.tsx`
- Waveform + Spectrogram shown in Phase 3 (results) in a responsive 2-column grid
- Visualizations appear alongside prediction card and probability bars
- Audio playback handled in Phase 2 (AudioTrimmer) — user previews segment before classifying

## Files Changed
- `api/model_service.py` — `_downsample_waveform()` + updated `classify()` return
- `api/schemas.py` — added `waveform` field
- `web/src/types.ts` — added `waveform` field
- `web/src/components/WaveformDisplay.tsx` — **new file**
- `web/src/components/SpectrogramDisplay.tsx` — **new file**
- `web/src/pages/Classify.tsx` — integrated visualizations into results phase

## Verification
- [x] Waveform renders correctly for classified audio segment
- [x] Spectrogram renders as a clear heatmap (magma colormap)
- [x] Axis labels are present and readable
- [x] Audio preview works via AudioTrimmer (play selected segment)
- [x] Hover on spectrogram shows frequency/time/energy tooltip
- [x] Both visualizations update when a new classification is made
- [x] Responsive layout — stacks vertically on narrow screens

## Design Notes
- Both components are pure Canvas (zero extra dependencies)
- Waveform uses min/max envelope so peaks aren't lost in downsampling
- Magma colormap chosen to match audio visualization conventions
- Playback is client-side only — no round-trip to server

## Next
Step 25: Live Microphone Input
