# Step 25: Live Microphone Input

**Estimated Time:** 60 minutes
**Status:** ✅ Complete

---

## Goal
Let users record audio directly from their microphone in the browser, send it to the backend for classification, and display results. This is the "wow" moment — speak or play an instrument and watch the system identify it live.

## Tasks

### Frontend
- [x] Create `MicrophoneRecorder` component using Web Audio API
- [x] Request microphone permission (`navigator.mediaDevices.getUserMedia`)
- [x] Show live audio level meter while recording (visual feedback)
- [x] Record for exactly 2 seconds (matches model's expected duration)
- [x] Convert recorded audio to WAV format (client-side)
- [x] Send WAV blob to `POST /api/classify/live` endpoint
- [x] Display results using the same results UI from Step 23
- [x] Add record button with visual states: idle → requesting → recording → processing → error
- [x] Handle permission denied gracefully (show message, fall back to upload)

### Backend
- [x] Ensure `/api/classify` handles WAV blobs from the mic (already worked)
- [x] `POST /api/classify/live` endpoint (already existed from Step 21)

## Implementation Plan

### Recording Flow
```
[🎙️ Record]  →  Recording (2s countdown)  →  Processing...  →  Results
   idle             ● ● ● pulsing              spinner           same UI
                    "Recording... 1.4s"                          as upload
```

### Web Audio API Approach
```
getUserMedia → MediaRecorder → Blob (WAV) → FormData → POST /api/classify
```

1. `getUserMedia({ audio: true })` — get mic stream
2. `MediaRecorder` — record the stream
3. On stop → collect chunks → create WAV Blob
4. Send to backend as if it were a file upload

### Live Audio Level Meter
While recording, show a real-time volume bar:
```
🎙️ Recording...  ████████░░░░░░░░  1.2s / 2.0s
```
- Uses `AnalyserNode` from Web Audio API
- `getByteFrequencyData()` → average amplitude → bar width
- Updates at ~30fps via `requestAnimationFrame`

### WAV Encoding
The MediaRecorder may output webm/opus. Need to convert to WAV:
- Option A: Use `AudioContext.decodeAudioData()` + manual WAV header
- Option B: Use a small library like `audiobuffer-to-wav`
- Option C: Send raw PCM and let backend handle conversion

Recommend Option A (no extra dependency, keeps it lean).

### Button States
```typescript
type RecordingState = 'idle' | 'requesting' | 'recording' | 'processing' | 'done' | 'error';
```

| State | Button | Visual |
|-------|--------|--------|
| idle | "🎙️ Record" | Blue, inviting |
| requesting | "Requesting mic..." | Gray, pulsing |
| recording | "⏹ Stop (1.4s)" | Red, pulsing ring animation |
| processing | "Processing..." | Spinner |
| done | "🎙️ Record Again" | Green → Blue |
| error | "Mic unavailable" | Red text, show upload fallback |

### Microphone Permission Handling
```typescript
try {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  // proceed with recording
} catch (err) {
  if (err.name === 'NotAllowedError') {
    // Show: "Microphone access denied. You can still upload files."
  } else {
    // Show: "Microphone not available on this device."
  }
}
```

## UI Layout (Classify Page Updated)
```
┌─────────────────────────────────────────┐
│                                         │
│  ┌─────────────┐   ┌─────────────┐     │
│  │ 📁 Upload   │   │ 🎙️ Record   │     │
│  │   a file    │   │   live      │     │
│  └─────────────┘   └─────────────┘     │
│                                         │
│  ─── OR drag & drop anywhere ───        │
│                                         │
└─────────────────────────────────────────┘
```

Two equal tabs/options at the top of the Classify page. Both lead to the same results view.

## Verification
- [ ] Mic permission request appears on first click
- [ ] Live volume meter shows activity while recording
- [ ] 2-second countdown visible and accurate
- [ ] Recorded audio classifies correctly (test with YouTube instrument sounds)
- [ ] Results display identically to file upload results
- [ ] Permission denied shows helpful fallback message
- [ ] Works in Chrome and Safari
- [ ] No audio glitches or memory leaks from repeated recordings

## Browser Compatibility Notes
- `getUserMedia` works in Chrome, Firefox, Safari, Edge
- Safari may require HTTPS (even localhost should work during dev)
- Mobile browsers: should work but mic quality varies
- `MediaRecorder` API: Chrome/Firefox fully supported, Safari since iOS 14.3

## Next
Step 26: Processing Pipeline Animation
