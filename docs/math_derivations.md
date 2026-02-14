# Mathematical Derivations

> Complete backpropagation math for the Acoustic Pattern Recognition Engine

---

## Network Architecture

```
Input x ∈ ℝ^(N×4096)     (N = batch size, 4096 = flattened 64×64 spectrogram)

Layer 1:  z₁ = x W₁ + b₁          W₁ ∈ ℝ^(4096×128),  b₁ ∈ ℝ^(1×128)
          a₁ = ReLU(z₁)

Layer 2:  z₂ = a₁ W₂ + b₂         W₂ ∈ ℝ^(128×64),    b₂ ∈ ℝ^(1×64)
          a₂ = ReLU(z₂)

Layer 3:  z₃ = a₂ W₃ + b₃         W₃ ∈ ℝ^(64×10),     b₃ ∈ ℝ^(1×10)
          ŷ  = Softmax(z₃)         ŷ ∈ ℝ^(N×10) — predicted probabilities
```

Total trainable parameters: **533,322**
- Layer 1: 4096 × 128 + 128 = 524,416
- Layer 2: 128 × 64 + 64 = 8,256
- Layer 3: 64 × 10 + 10 = 650

---

## 1. Forward Pass

### 1.1 Dense Layer

Each dense layer computes a linear transformation:

\[
z = x W + b
\]

Where:
- \( x \in \mathbb{R}^{N \times d_{in}} \) — input batch
- \( W \in \mathbb{R}^{d_{in} \times d_{out}} \) — weight matrix
- \( b \in \mathbb{R}^{1 \times d_{out}} \) — bias vector (broadcast across batch)
- \( z \in \mathbb{R}^{N \times d_{out}} \) — pre-activation output

### 1.2 ReLU Activation

Applied element-wise after hidden dense layers:

\[
\text{ReLU}(z) = \max(0, z)
\]

- Passes positive values unchanged
- Blocks negative values (sets to 0)
- Introduces non-linearity so the network can learn complex decision boundaries

### 1.3 Softmax Activation

Applied at the output layer to convert raw logits to a probability distribution:

\[
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}
\]

Where \( C = 10 \) (number of instrument classes).

**Numerical stability trick:** To prevent overflow from large exponentials, we subtract the row-wise maximum before exponentiation:

\[
\text{Softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^{C} e^{z_j - \max(z)}}
\]

This does not change the result (the constant cancels in numerator and denominator) but keeps all exponents in a numerically safe range.

**Properties:**
- All outputs are positive: \( \hat{y}_i > 0 \)
- Outputs sum to 1: \( \sum_i \hat{y}_i = 1 \)
- Preserves ordering: highest logit → highest probability

---

## 2. Loss Function

### Cross-Entropy Loss

Measures the divergence between the predicted probability distribution \( \hat{y} \) and the true label distribution \( y \) (one-hot encoded):

\[
\mathcal{L} = -\frac{1}{N} \sum_{n=1}^{N} \sum_{c=1}^{C} y_{n,c} \log(\hat{y}_{n,c})
\]

Since \( y \) is one-hot (only one class is 1, rest are 0), this simplifies to:

\[
\mathcal{L} = -\frac{1}{N} \sum_{n=1}^{N} \log(\hat{y}_{n, k_n})
\]

Where \( k_n \) is the true class for sample \( n \).

**Numerical stability:** We clip predictions before taking the log:

\[
\hat{y}_{\text{clipped}} = \text{clip}(\hat{y}, 10^{-15}, 1 - 10^{-15})
\]

This prevents \( \log(0) = -\infty \).

**Intuition:**
- Perfect prediction (\( \hat{y}_{k} = 1.0 \)) → loss = \( -\log(1) = 0 \)
- Terrible prediction (\( \hat{y}_{k} \approx 0 \)) → loss = \( -\log(0) \to \infty \)
- Uniform prediction (\( \hat{y}_{k} = 1/C \)) → loss = \( \log(C) \approx 2.303 \) for 10 classes

---

## 3. Backward Pass (Backpropagation)

Backpropagation computes gradients of the loss with respect to every parameter by applying the chain rule layer by layer, working backward from the output.

### 3.1 Combined Softmax + Cross-Entropy Gradient

The gradient of the loss with respect to the **pre-softmax logits** \( z_3 \) has a remarkably clean form:

\[
\frac{\partial \mathcal{L}}{\partial z_3} = \frac{\hat{y} - y}{N}
\]

Where:
- \( \hat{y} \) is the softmax output (predicted probabilities)
- \( y \) is the one-hot true labels
- \( N \) is the batch size

**Why this is elegant:** Computing the Softmax Jacobian separately would require an \( N \times C \times C \) tensor. The combined gradient collapses to a simple subtraction divided by batch size.

**Derivation sketch:**

For a single sample with true class \( k \):

\[
\frac{\partial \mathcal{L}}{\partial z_{3,j}} = \hat{y}_j - y_j = 
\begin{cases}
\hat{y}_j - 1 & \text{if } j = k \text{ (true class: negative, pushes probability up)} \\
\hat{y}_j & \text{if } j \neq k \text{ (other classes: positive, pushes probabilities down)}
\end{cases}
\]

### 3.2 Dense Layer Backward

Given gradient flowing back from the next layer, \( \frac{\partial \mathcal{L}}{\partial z} \) (which we'll call \( \delta \)):

**Weight gradient** (used by optimizer to update weights):
\[
\frac{\partial \mathcal{L}}{\partial W} = x^T \delta
\]

**Bias gradient** (used by optimizer to update biases):
\[
\frac{\partial \mathcal{L}}{\partial b} = \sum_{n=1}^{N} \delta_n = \text{sum}(\delta, \text{axis}=0)
\]

**Input gradient** (passed to the previous layer):
\[
\frac{\partial \mathcal{L}}{\partial x} = \delta W^T
\]

### 3.3 ReLU Backward

ReLU's derivative is a binary mask:

\[
\frac{\partial \text{ReLU}(z)}{\partial z} = 
\begin{cases}
1 & \text{if } z > 0 \\
0 & \text{if } z \leq 0
\end{cases}
\]

So the gradient passes through where the input was positive and is blocked where it was zero or negative:

\[
\frac{\partial \mathcal{L}}{\partial z} = \delta \odot \mathbb{1}[z > 0]
\]

Where \( \odot \) is element-wise multiplication and \( \mathbb{1}[\cdot] \) is the indicator function.

---

## 4. Full Backward Pass Chain

Putting it all together, the complete backward pass flows in reverse through the network:

```
Step 1: δ₃ = (ŷ - y) / N                            ← Combined Softmax + CE gradient

Step 2: ∂L/∂W₃ = a₂ᵀ δ₃                             ← Layer 3 weight gradient
        ∂L/∂b₃ = sum(δ₃, axis=0)                    ← Layer 3 bias gradient
        δ₂_pre = δ₃ W₃ᵀ                             ← Gradient passed to Layer 2

Step 3: δ₂ = δ₂_pre ⊙ 𝟙[z₂ > 0]                    ← ReLU backward (Layer 2)

Step 4: ∂L/∂W₂ = a₁ᵀ δ₂                             ← Layer 2 weight gradient
        ∂L/∂b₂ = sum(δ₂, axis=0)                    ← Layer 2 bias gradient
        δ₁_pre = δ₂ W₂ᵀ                             ← Gradient passed to Layer 1

Step 5: δ₁ = δ₁_pre ⊙ 𝟙[z₁ > 0]                    ← ReLU backward (Layer 1)

Step 6: ∂L/∂W₁ = xᵀ δ₁                              ← Layer 1 weight gradient
        ∂L/∂b₁ = sum(δ₁, axis=0)                    ← Layer 1 bias gradient
```

After this, every Dense layer has its `grad_weights` and `grad_biases` populated, ready for the optimizer.

---

## 5. Weight Update (SGD)

Stochastic Gradient Descent updates each parameter in the direction that reduces the loss:

\[
W \leftarrow W - \eta \cdot \frac{\partial \mathcal{L}}{\partial W}
\]

\[
b \leftarrow b - \eta \cdot \frac{\partial \mathcal{L}}{\partial b}
\]

Where \( \eta \) is the learning rate.

### Learning Rate Decay

The learning rate is reduced by a multiplicative factor after each epoch to allow fine-grained convergence in later stages:

\[
\eta_{t+1} = \eta_t \cdot \gamma
\]

Where \( \gamma = 0.95 \) (decay factor). After \( t \) epochs:

\[
\eta_t = \eta_0 \cdot \gamma^t = 0.01 \cdot 0.95^t
\]

---

## 6. Weight Initialization (He)

Weights are initialized using He initialization, which keeps the variance of activations stable through ReLU layers:

\[
W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{d_{in}}}\right)
\]

Where \( d_{in} \) is the number of input features to that layer.

**Why He and not Xavier?** Xavier initialization assumes linear activations. ReLU kills ~50% of neurons (those with negative input), so He scales by \( \sqrt{2/d_{in}} \) instead of \( \sqrt{1/d_{in}} \) to compensate.

Biases are initialized to zero — this is standard and does not break symmetry because the weights are already random.

---

## 7. Gradient Verification

The analytical gradients derived above are verified against numerical (finite-difference) gradients:

\[
\frac{\partial \mathcal{L}}{\partial w_{ij}} \approx \frac{\mathcal{L}(w_{ij} + \epsilon) - \mathcal{L}(w_{ij} - \epsilon)}{2\epsilon}
\]

With \( \epsilon = 10^{-5} \), the maximum difference between analytical and numerical gradients is **1.19 × 10⁻¹⁰** — effectively machine precision. This confirms the backpropagation math is correct.

---

## 8. Data Normalization

Before training, the flattened spectrograms are standardized using training-set statistics:

\[
x_{\text{normalized}} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}
\]

Where \( \mu_{\text{train}} \) and \( \sigma_{\text{train}} \) are the per-feature mean and standard deviation computed **only from the training set** (to prevent data leakage).

**Zero-variance handling:** 302 of the 4096 features have near-zero variance (std ≤ 10⁻⁶) in the training set. These correspond to pixel positions in the spectrogram that are always near-zero (e.g., very high frequency bins). Instead of dividing by a tiny number (which amplifies noise), these features are zeroed out entirely.

---

## Summary of Shapes

| Quantity | Shape | Description |
|----------|-------|-------------|
| \( x \) | (N, 4096) | Input (flattened spectrogram) |
| \( W_1 \) | (4096, 128) | Layer 1 weights |
| \( z_1, a_1 \) | (N, 128) | Layer 1 pre/post activation |
| \( W_2 \) | (128, 64) | Layer 2 weights |
| \( z_2, a_2 \) | (N, 64) | Layer 2 pre/post activation |
| \( W_3 \) | (64, 10) | Layer 3 weights |
| \( z_3 \) | (N, 10) | Output logits |
| \( \hat{y} \) | (N, 10) | Softmax probabilities |
| \( y \) | (N, 10) | One-hot true labels |
| \( \mathcal{L} \) | scalar | Cross-entropy loss |
