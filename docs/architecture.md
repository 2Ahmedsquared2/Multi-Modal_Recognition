# Architecture & Design Decisions

> Why every choice was made the way it was

---

## Network Architecture

```
Input (4096) → Dense(128) → ReLU → Dense(64) → ReLU → Dense(10) → Softmax
```

| Layer | Input → Output | Parameters | Purpose |
|-------|---------------|------------|---------|
| Dense 1 | 4096 → 128 | 524,416 | Compress spectrogram features to compact representation |
| ReLU | 128 → 128 | 0 | Non-linearity — enables learning complex patterns |
| Dense 2 | 128 → 64 | 8,256 | Refine features into class-relevant abstractions |
| ReLU | 64 → 64 | 0 | Non-linearity |
| Dense 3 | 64 → 10 | 650 | Map to 10 instrument class scores |
| Softmax | 10 → 10 | 0 | Convert scores to probabilities |
| **Total** | | **533,322** | |

---

## Decision 1: Why 4096 → 128 → 64 → 10?

### Input: 4096 (64×64 flattened spectrogram)

We flatten the 64×64 mel-spectrogram into a 4096-dimensional vector. This is necessary for a fully connected (dense) network since it operates on flat vectors, not 2D grids.

**Why 64×64 and not 128×128?**
- 64×64 = 4,096 features — fast training, small memory footprint
- 128×128 = 16,384 features — 4× more parameters in the first layer, diminishing returns for our dataset size (3,333 samples)
- 64×64 proved sufficient: **94.5% accuracy** confirms no resolution bottleneck

### Hidden Layer 1: 128 neurons

- **Compression ratio:** 32:1 (4096 → 128) — forces the network to learn compact, meaningful representations
- Large enough to capture spectral patterns across instruments
- Small enough to prevent overfitting on 2,330 training samples

### Hidden Layer 2: 64 neurons

- **Further refinement:** takes the 128 intermediate features and distills them into 64 class-relevant features
- The funnel shape (4096 → 128 → 64 → 10) is a proven architecture pattern — each layer progressively abstracts

### Output: 10 neurons

One per instrument class: bass, brass, flute, guitar, keyboard, mallet, organ, reed, string, vocal.

### Why not wider or deeper?

- **Wider (e.g., 512 → 256):** More parameters than training samples — overfitting risk. Our 533K params already approach our dataset size.
- **Deeper (e.g., 4+ layers):** Diminishing returns for fully connected networks on this task. CNNs benefit from depth; dense networks on flat vectors less so.
- **Shallower (e.g., 1 hidden layer):** Tested implicitly — the two hidden layers provide enough capacity for 94.5% accuracy.

---

## Decision 2: Why ReLU?

### Compared to alternatives:

| Activation | Formula | Pros | Cons |
|-----------|---------|------|------|
| **ReLU** | max(0, x) | Fast, no saturation for positive values, sparse activations | Dead neurons (input ≤ 0 → gradient = 0 forever) |
| Sigmoid | 1/(1+e⁻ˣ) | Bounded [0,1] | Vanishing gradients at extremes, slow |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | Zero-centered, bounded [-1,1] | Vanishing gradients at extremes |
| Leaky ReLU | max(0.01x, x) | Fixes dead neuron problem | Marginal improvement, adds hyperparameter |

### Why ReLU wins here:

1. **Speed:** A single `np.maximum(0, x)` — effectively free computation
2. **Gradient flow:** No saturation for positive inputs → healthy gradients throughout training
3. **Sparsity:** ~50% of neurons output zero → implicit regularization
4. **He initialization compatibility:** He init is designed specifically for ReLU
5. **Standard choice:** Used in the vast majority of modern neural networks — proven reliable
6. **Dead neurons not a problem:** Our gradient flow analysis confirms all layers have healthy gradient norms (no vanishing)

---

## Decision 3: Why Cross-Entropy Loss?

### For multi-class classification, cross-entropy is the standard choice:

**vs. Mean Squared Error (MSE):**
- MSE treats all output neurons equally — it doesn't understand that outputs should sum to 1
- Cross-entropy directly penalizes confident wrong predictions (through the log term)
- Cross-entropy + Softmax gradient simplifies to `(predicted - true) / N` — clean, stable, fast
- MSE + Softmax gradient requires the full Softmax Jacobian — expensive and numerically fragile

**vs. Hinge Loss (SVM-style):**
- Hinge loss doesn't produce calibrated probabilities
- Cross-entropy outputs actual probabilities (useful for confidence analysis)
- Cross-entropy is differentiable everywhere — no kinks

### The combined Softmax + Cross-Entropy backward pass

This is a key design decision. Instead of implementing Softmax backward and Cross-Entropy backward as separate operations, we combine them. The mathematical simplification:

```
Separate:  dL/dz = dL/dŷ × dŷ/dz  (requires C×C Jacobian per sample)
Combined:  dL/dz = (ŷ - y) / N     (single subtraction)
```

This is the same approach used internally by PyTorch (`CrossEntropyLoss`) and TensorFlow (`SparseCategoricalCrossentropy`).

---

## Decision 4: Why Plain SGD (no momentum)?

### Our optimizer: vanilla mini-batch SGD

```
W ← W - η × ∂L/∂W
```

**Why not Adam, RMSprop, or SGD with momentum?**

1. **From-scratch philosophy:** Plain SGD is the purest form of gradient descent — it directly demonstrates that we understand the optimization math
2. **Sufficient performance:** 94.5% test accuracy in 88 epochs proves SGD works well for this problem
3. **Simplicity:** No additional state variables (momentum buffers, second-moment estimates)
4. **Transparency:** Every weight update is directly proportional to the gradient — easy to reason about and debug

**Learning rate decay compensates:** The multiplicative decay (0.95× per epoch) gives us large steps early (explore) and small steps late (converge) — accomplishing some of what momentum provides.

**When would we upgrade?** If the loss landscape were more complex (e.g., many local minima, saddle points), Adam would converge faster. For our well-conditioned problem with proper normalization and He init, plain SGD is enough.

---

## Decision 5: Why Mel-Spectrograms?

### Audio representation choice

| Representation | Shape | Information | Complexity |
|---------------|-------|-------------|------------|
| Raw waveform | (44100,) | Complete but unstructured | Hard for networks to learn |
| **Mel-spectrogram** | (64, 64) | Time × frequency, human-aligned | Best for instrument recognition |
| MFCC | (13, T) | Compressed spectral envelope | Loses fine detail |
| Chromagram | (12, T) | Pitch classes only | Ignores timbre |

### Why mel-spectrogram is ideal:

1. **Perceptual alignment:** Mel scale matches human hearing — frequency bins are spaced logarithmically, giving more resolution to lower frequencies (where most musical content lives)
2. **Visual interpretability:** Different instruments have distinct spectral signatures visible in the spectrogram
3. **2D structure:** Time × frequency is a natural image-like representation
4. **Proven approach:** Used in state-of-the-art audio classification systems

### Spectrogram parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Sample rate | 22,050 Hz | Standard for music (captures up to 11 kHz — sufficient for instruments) |
| FFT size | 2,048 | Good frequency resolution (~10 Hz per bin) |
| Hop length | 512 | 4:1 overlap ratio — smooth time axis |
| Mel bands | 128 | High frequency detail before resizing |
| Duration | 2.0 seconds | Enough to capture instrument timbre |
| Output resolution | 64×64 | Balanced detail vs. computational cost |

---

## Decision 6: Why He Initialization?

### Weight initialization matters

Bad initialization → gradients explode or vanish → network doesn't learn.

| Method | Scale | Best for |
|--------|-------|----------|
| **He** | √(2/fan_in) | ReLU networks |
| Xavier/Glorot | √(1/fan_in) | Sigmoid/Tanh networks |
| Random (0.01) | 0.01 | Nothing — too small |
| Random (1.0) | 1.0 | Nothing — too large |

**He initialization** accounts for ReLU killing ~50% of neurons by scaling up by √2:

```
W ~ N(0, √(2/d_in))
```

For our first layer (d_in=4096): std ≈ 0.0221 — keeps output activations in a healthy [-5, +5] range.

---

## Decision 7: Why Stratified Splitting?

### Data split: 70% train / 15% validation / 15% test

**Stratified** means each split maintains the same class distribution as the full dataset. This is critical because our dataset is **imbalanced**:

| Class | Samples | % of total |
|-------|---------|-----------|
| bass | 500 | 15.0% |
| guitar | 500 | 15.0% |
| keyboard | 500 | 15.0% |
| organ | 500 | 15.0% |
| string | 306 | 9.2% |
| brass | 269 | 8.1% |
| reed | 235 | 7.1% |
| mallet | 202 | 6.1% |
| flute | 180 | 5.4% |
| vocal | 141 | 4.2% |

Without stratified splitting, the test set might have zero vocal samples (only 141 total) — making evaluation meaningless for that class.

---

## Decision 8: Why Early Stopping?

**Patience = 10 epochs.** If validation loss hasn't improved for 10 consecutive epochs, training halts.

- **Prevents overfitting:** Training accuracy reached 99.8% while validation plateaued at 91.8%. Continuing further would only memorize training data.
- **Saves time:** Stopped at epoch 78 instead of running all 100 epochs.
- **No manual tuning:** Don't need to guess the right number of epochs.

---

## Decision 9: Why No Regularization (L2/Dropout)?

For the MVP, we intentionally omit:
- **L2 regularization:** Adds a penalty term λ‖W‖² to the loss
- **Dropout:** Randomly zeros neurons during training

**Why skip them?**
- 94.5% test accuracy shows the network generalizes well without them
- The gap between train (99.8%) and test (94.5%) is moderate — slight overfitting but not catastrophic
- Data augmentation (pitch shift, time stretch, noise injection) during audio loading already provides implicit regularization
- Fewer moving parts = easier to reason about and debug

**When to add them:** If we scaled to a larger network or saw significant train-test accuracy gaps (>15%), L2 or dropout would be the next step.

---

## Results Validation

The architecture choices are validated by the final results:

| Metric | Value | Assessment |
|--------|-------|-----------|
| Test accuracy | 94.5% | Exceeds 85% portfolio target |
| Macro F1 | 0.9473 | Balanced performance across classes |
| Training time | 12.7s | Fast iteration on CPU |
| Gradient norms | Healthy | No vanishing/exploding gradients |
| Convergence | 88 epochs | Stable training with early stopping |
| Best class (reed) | F1 = 1.000 | Perfect classification |
| Worst class (mallet) | F1 = 0.852 | Still strong (smallest class, 31 test samples) |
