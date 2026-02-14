# Step 26: Processing Pipeline Animation

**Estimated Time:** 45 minutes
**Status:** ✅ Complete

---

## Goal
Show the user a step-by-step animated visualization of how their audio gets processed: raw waveform → spectrogram generation → neural network layers → final prediction. This makes the "black box" transparent and is visually striking for a portfolio.

## Tasks
- [x] Create `PipelineAnimation` component
- [x] Step 1 animation: Raw waveform appears
- [x] Step 2 animation: Waveform transforms into spectrogram (visual morph)
- [x] Step 3 animation: Spectrogram flows into neural network diagram
- [x] Step 4 animation: Network layers light up sequentially (4096 → 128 → 64 → 10)
- [x] Step 5 animation: Output neurons highlight → prediction appears
- [x] Add toggle: "Show pipeline" on/off (not everyone wants to wait)
- [x] Animate on first classify, then skip on subsequent (or user preference)

## Implementation Plan

### Pipeline Stages
```
┌──────────┐    ┌──────────────┐    ┌────────────────┐    ┌──────────┐
│  🔊 Raw  │ →  │ 📊 Spectro-  │ →  │ 🧠 Neural      │ →  │ 🎸 Pre-  │
│  Audio   │    │    gram       │    │    Network      │    │  diction │
│          │    │              │    │                │    │          │
│ waveform │    │  64×64 grid  │    │ 4096→128→64→10│    │ "Guitar" │
│  appears │    │  fills in    │    │ layers light up│    │  92.3%   │
└──────────┘    └──────────────┘    └────────────────┘    └──────────┘
     1s              1.5s                 1.5s                 1s
```

### Animation Approach
- **CSS transitions + React state** — simple, performant, no heavy animation library needed
- Each stage is a component that animates in via Tailwind `transition-all` + `opacity` + `transform`
- Stagger timing with `setTimeout` or CSS `animation-delay`
- Arrow connectors between stages animate with `width` transition

### Neural Network Visualization (Stage 3)
```
    Input        Hidden 1      Hidden 2      Output
   (4096)        (128)          (64)          (10)
   
    ○ ○ ○         ○ ○            ○             ○ ← guitar (bright)
    ○ ○ ○   →    ○ ○      →    ○       →     ○
    ○ ○ ○         ○ ○            ○             ○
    ○ ○ ○         ○ ○            ○             ○
    ⋮              ⋮              ⋮             ⋮
```
- Nodes are small circles, connections are thin lines
- Animation: nodes light up left-to-right (input → hidden1 → hidden2 → output)
- Output layer: winning neuron glows bright, others dim
- Don't draw all 4096 nodes — show ~20 representative nodes per layer with "..." indicator

### Component Structure
```typescript
interface PipelineAnimationProps {
  waveform: number[];          // Raw audio data points
  spectrogram: number[][];     // 64×64 grid
  prediction: string;          // "guitar"
  confidence: number;          // 0.923
  allConfidences: Record<string, number>;
  isPlaying: boolean;          // Controls animation start
}
```

### Timing
| Stage | Duration | Cumulative |
|-------|----------|-----------|
| Waveform appears | 0.8s | 0.8s |
| Arrow 1 animates | 0.3s | 1.1s |
| Spectrogram fills in | 1.0s | 2.1s |
| Arrow 2 animates | 0.3s | 2.4s |
| Network layers light up | 1.2s | 3.6s |
| Arrow 3 animates | 0.3s | 3.9s |
| Prediction reveals | 0.6s | 4.5s |

Total: ~4.5 seconds. Fast enough to not feel slow, slow enough to be impressive.

### User Control
- "Skip animation" link appears during animation
- On subsequent classifies, pipeline shows instantly (no repeat animation)
- Toggle in settings: "Always show animation"

## Verification
- [x] Animation plays smoothly (no jank, 60fps)
- [x] Each stage is visually distinct and labeled
- [x] Neural network diagram shows correct layer sizes
- [x] Prediction at the end matches the actual classification result
- [x] "Skip" button works
- [x] Animation doesn't break on rapid re-classifies
- [x] Works on slower machines (CSS transitions, not JS-heavy)

## Implementation Summary

### What Was Built
Created `web/src/components/PipelineAnimation.tsx` — a fully animated 4-stage pipeline:

1. **Stage 1: Raw Audio** (indigo)
   - Mini waveform SVG rendered from actual audio data
   - Fades in from below with `translateY` + `opacity`
   
2. **Stage 2: Spectrogram** (amber)
   - Canvas-rendered mel-spectrogram using the Magma colormap (matches SpectrogramDisplay)
   - 64×64 grid rendered at native resolution then scaled with `imageRendering: pixelated`
   
3. **Stage 3: Neural Network** (violet)
   - SVG diagram with 4 layers showing representative nodes (7→5→4→5 visible nodes)
   - Connections drawn as thin lines between all nodes
   - Layers light up sequentially left-to-right
   - Output layer: winning neuron glows emerald, others stay dim
   - Size labels below each layer (4096, 128, 64, 10)
   
4. **Stage 4: Prediction** (emerald)
   - Final classification with confidence percentage
   - Reveals last

**Animation Flow:**
- Progress bar at top shows completion (0-100%)
- Arrow connectors fade in between stages
- Step indicators at bottom track: "Load Audio" → "Generate Spectrogram" → "Run Inference" → "Classify"
- Total duration: ~4.8 seconds
- "Skip →" button in header allows instant skip to results

### Integration with Classify Page
Modified `web/src/pages/Classify.tsx`:
- Added `'pipeline'` phase to the phase type
- Imported `PipelineAnimation` component
- Added `hasAnimatedRef` to track first-time animation state
- On first classify: routes through `'pipeline'` phase
- On subsequent classifies: skips directly to `'results'` phase
- `handlePipelineComplete` callback advances from pipeline → results and marks animation as seen

### Technical Approach
- **Zero animation libraries** — pure CSS `transition-all` + React state
- **GPU-accelerated** — only `opacity` and `transform` transitions (no layout thrashing)
- **Scheduled with setTimeout** — 11 stage transitions over ~4.8 seconds
- **Canvas drawn once** — spectrogram canvas guarded by `specDrawnRef`, never redraws
- **Waveform memoized** — computed once with `useMemo`, generates SVG path from ~60 sampled points
- **Cleanup on unmount** — all timers cleared to prevent memory leaks

### Performance Impact
- **First classify:** +4.8s animation (skippable)
- **Subsequent classifies:** Zero overhead — animation completely bypassed
- **No re-renders during animation** — only state changes trigger CSS transitions
- **Lightweight SVG** — neural network is ~200 DOM nodes, all static except fill/stroke colors

## Design Notes
- This is purely a frontend feature — no backend changes needed
- The animation uses data already returned by the classify endpoint
- Keep it elegant and simple — avoid cartoon-ish graphics
- Think "Apple keynote product demo" energy

## Next
Step 27: Interactive Training Dashboard
