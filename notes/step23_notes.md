# Step 23: Audio Upload & Classification

**Estimated Time:** 60 minutes
**Status:** ⬜ Not Started

---

## Goal
Build the first end-to-end feature: user uploads an audio file, backend processes it, frontend displays the predicted instrument with confidence scores. This is the core demo moment.

## Tasks

### Backend
- [ ] Implement `POST /api/classify` endpoint fully
  - Accept `.wav` file upload via multipart form
  - Load audio with librosa (resample to 22050 Hz, trim/pad to 2 seconds)
  - Generate mel-spectrogram (64×64)
  - Flatten, normalize (using saved mean/std from training)
  - Run through neural network → probabilities
  - Return prediction, confidence, all class scores, spectrogram data
- [ ] Add input validation (file type, file size limit)
- [ ] Handle errors gracefully (invalid audio, corrupt file)

### Frontend
- [ ] Create `Classify` page with drag-and-drop upload zone
- [ ] File input accepting `.wav`, `.mp3`, `.ogg`, `.flac`
- [ ] Upload progress indicator
- [ ] Display results:
  - Predicted instrument name (large, prominent)
  - Confidence percentage
  - Horizontal bar chart of all 10 class confidences (sorted)
  - Color coding: top prediction highlighted, others dimmed
- [ ] "Try another" button to reset
- [ ] Loading state while backend processes

## Implementation Plan

### Upload Component
```
┌─────────────────────────────────────────┐
│                                         │
│     🎵 Drop an audio file here          │
│        or click to browse               │
│                                         │
│     Supports: .wav .mp3 .ogg .flac      │
│                                         │
└─────────────────────────────────────────┘
```

### Results Display
```
┌─────────────────────────────────────────┐
│                                         │
│     🎸 Guitar                           │
│     92.3% confidence                    │
│                                         │
│     guitar    ████████████████████ 92%  │
│     bass      ███                  3%   │
│     keyboard  ██                   2%   │
│     string    █                    1%   │
│     ...                                 │
│                                         │
│     [Try Another]                       │
│                                         │
└─────────────────────────────────────────┘
```

### Backend Normalization
Important: the backend must apply the same normalization as training:
```python
# Load mean/std from prepared data
prepared = np.load("data/prepared/prepared_data.npz")
self.mean = prepared['mean']
self.std = prepared['std']

# During inference:
flat = spectrogram.flatten().reshape(1, -1)
flat_normalized = (flat - self.mean) / self.std
probs = self.network.forward(flat_normalized)
```

### Error Handling
- File too large (>10MB) → 413 with message
- Invalid format → 415 with supported formats list
- Processing error → 500 with user-friendly message
- No audio content → 400 with explanation

## Verification
- [ ] Upload a .wav file → correct prediction returned
- [ ] Confidence bars render correctly (sorted, colored)
- [ ] Drag-and-drop works
- [ ] Click-to-browse works
- [ ] Error shown for non-audio files
- [ ] Loading spinner appears during processing
- [ ] "Try another" resets the UI cleanly
- [ ] Backend handles concurrent requests

## UX Considerations
- The upload zone should be large and inviting (not a tiny button)
- Results should animate in (fade/slide)
- Confidence bars should animate from 0 to final value
- The predicted instrument should be unmistakably prominent
- Keep the page simple — this is the "hero" feature

## Next
Step 24: Waveform & Spectrogram Display
