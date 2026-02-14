# Step 26: Processing Pipeline Animation

**Estimated Time:** 45 minutes
**Status:** ⬜ Not Started

---

## Goal
Show the user a step-by-step animated visualization of how their audio gets processed: raw waveform → spectrogram generation → neural network layers → final prediction. This makes the "black box" transparent and is visually striking for a portfolio.

## Tasks
- [ ] Create `PipelineAnimation` component
- [ ] Step 1 animation: Raw waveform appears
- [ ] Step 2 animation: Waveform transforms into spectrogram (visual morph)
- [ ] Step 3 animation: Spectrogram flows into neural network diagram
- [ ] Step 4 animation: Network layers light up sequentially (4096 → 128 → 64 → 10)
- [ ] Step 5 animation: Output neurons highlight → prediction appears
- [ ] Add toggle: "Show pipeline" on/off (not everyone wants to wait)
- [ ] Animate on first classify, then skip on subsequent (or user preference)

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
- [ ] Animation plays smoothly (no jank, 60fps)
- [ ] Each stage is visually distinct and labeled
- [ ] Neural network diagram shows correct layer sizes
- [ ] Prediction at the end matches the actual classification result
- [ ] "Skip" button works
- [ ] Animation doesn't break on rapid re-classifies
- [ ] Works on slower machines (CSS transitions, not JS-heavy)

## Design Notes
- This is purely a frontend feature — no backend changes needed
- The animation uses data already returned by the classify endpoint
- Keep it elegant and simple — avoid cartoon-ish graphics
- Think "Apple keynote product demo" energy

## Next
Step 27: Interactive Training Dashboard
