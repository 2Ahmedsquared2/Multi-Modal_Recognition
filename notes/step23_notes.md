# Step 23: Audio Upload & Classification

**Estimated Time:** 60 minutes
**Status:** ✅ Complete

---

## Goal
Build the first end-to-end feature: user uploads an audio file, backend processes it, frontend displays the predicted instrument with confidence scores. This is the core demo moment.

## Tasks

### Backend
- [x] Implement `POST /api/classify` endpoint fully
  - Accept `.wav` file upload via multipart form
  - Load audio with librosa (resample to 22050 Hz, trim/pad to 2 seconds)
  - Generate mel-spectrogram (64×64)
  - Flatten, normalize (using saved mean/std from training)
  - Run through neural network → probabilities
  - Return prediction, confidence, all class scores, spectrogram data
- [x] Add input validation (file type, file size limit)
- [x] Handle errors gracefully (invalid audio, corrupt file)

### Frontend
- [x] Create `Classify` page with drag-and-drop upload zone
- [x] File input accepting `.wav`, `.mp3`, `.ogg`, `.flac`
- [x] Display results:
  - Predicted instrument name (large, prominent)
  - Confidence percentage
  - Horizontal bar chart of all 10 class confidences (sorted)
  - Color coding: top prediction highlighted (indigo), others dimmed (slate)
- [x] "Try another" button to reset
- [x] Loading state while backend processes
- [x] Audio trimmer for long files (pick which 2s segment to classify)

## What Was Built

### Backend — `api/server.py`
- `_validate_and_read()` helper shared across upload endpoints:
  - Extension allowlist: `.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`, `.aac`, `.wma`
  - Content-type check (lenient — any `audio/*` passes)
  - **50 MB** file size limit → 413 (raised from 10 MB to support longer songs)
  - Empty file check → 400
  - Unsupported format → 415 with list of accepted types
- Applied to `/api/classify`, `/api/classify/live`, and `/api/spectrogram`

### Frontend — 3-Phase Classify Flow
The Classify page now has three distinct phases:

**Phase 1 — Upload:** Drag-and-drop zone + click-to-browse file input. Selecting a file moves to Phase 2.

**Phase 2 — Trim:** The `AudioTrimmer` component:
- Decodes full audio in-browser via Web Audio API
- Draws entire waveform on canvas
- Draggable 2-second selection window (indigo highlight with grip handles)
- Click or drag anywhere to reposition the selection
- "Preview Segment" plays just the selected 2 seconds
- "Classify This Segment" extracts + encodes to WAV in-browser, sends to backend
- Short files (≤2s) skip the trimmer — show simple Classify button
- "Change file" link to go back to upload

**Phase 3 — Results:**
- Prediction card with instrument name + confidence %
- All-probabilities breakdown with sorted, color-coded horizontal bars
- Waveform + Spectrogram visualizations (from Step 24)
- "Try Different Segment" — goes back to trimmer with same file
- "Upload New File" — resets everything

### Audio Trimmer — `web/src/components/AudioTrimmer.tsx`
- Decodes audio file using Web Audio API's `decodeAudioData()`
- Full waveform rendered on canvas (min/max envelope, ~1000 points)
- Draggable 2-second selection with indigo highlight + grip handles
- Playback of selected segment only (via `AudioBufferSourceNode`)
- Playback position indicator (rose line on waveform)
- Time labels in `m:ss` format
- Loading state while decoding, error state if decode fails

### WAV Encoder — `web/src/utils/audioEncoder.ts`
- `extractSegmentAsWav()` — extracts a time range from AudioBuffer, mixes to mono, encodes as 16-bit PCM WAV blob
- `getWaveformEnvelope()` — downsamples AudioBuffer to min/max envelope for display
- Pure TypeScript, zero dependencies

### Bug Fix — `web/src/api/client.ts`
- FormData field name corrected from `"audio"` to `"file"` (matching FastAPI parameter name)
- Error handler now extracts FastAPI's `detail` message for user-friendly errors

### CSS — `web/src/index.css`
- Added `animate-fade-in` (300ms) and `animate-fade-in-up` (400ms) utility classes

## Files Changed
- `api/server.py` — validation helper + 50MB limit + applied to 3 endpoints
- `web/src/api/client.ts` — field name fix + better error extraction
- `web/src/pages/Classify.tsx` — full rewrite: 3-phase flow (upload → trim → results)
- `web/src/components/AudioTrimmer.tsx` — **new file**
- `web/src/utils/audioEncoder.ts` — **new file**
- `web/src/index.css` — animation keyframes

## Verification
- [x] Upload a .wav file → correct prediction returned
- [x] Confidence bars render correctly (sorted, colored)
- [x] Drag-and-drop works
- [x] Click-to-browse works
- [x] Error shown for non-audio files (415 with message)
- [x] Loading spinner appears during processing
- [x] "Try another" resets the UI cleanly
- [x] Long files show trimmer with draggable selection
- [x] Selected segment plays back correctly
- [x] WAV encoding produces valid audio for backend
- [ ] Backend handles concurrent requests (needs live testing)

## UX Considerations
- [x] The upload zone is large and inviting (not a tiny button)
- [x] Results animate in (fade/slide-up)
- [x] Confidence bars animate from 0 to final value
- [x] The predicted instrument is unmistakably prominent
- [x] Long files are handled gracefully (trimmer, not error)
- [x] User can preview before classifying

## Next
Step 24: Waveform & Spectrogram Display
