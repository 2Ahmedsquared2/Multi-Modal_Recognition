"""
Advanced Visualization Module — Step 12: Advanced Visualizations
Gradient-based attention maps, t-SNE feature visualization, gradient flow
analysis, and per-class metrics bar charts.

Functions:
    compute_attention_map      — Gradient of prediction w.r.t. input (which pixels matter)
    plot_attention_maps        — Grid of spectrograms with attention overlays
    visualize_tsne             — t-SNE scatter plot of penultimate layer features
    plot_gradient_norms        — Gradient magnitudes at each layer (health check)
    plot_per_class_metrics     — Grouped bar chart of precision/recall/F1 per class
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# ---------------------------------------------------------------------------
# Style Configuration (matches plots.py dark theme)
# ---------------------------------------------------------------------------

STYLE_CONFIG = {
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0',
    'xtick.color': '#e0e0e0',
    'ytick.color': '#e0e0e0',
    'grid.color': '#2a2a4a',
    'grid.alpha': 0.5,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.titlesize': 16,
}


def _apply_style():
    """Apply the dark theme style to matplotlib"""
    plt.rcParams.update(STYLE_CONFIG)


def _resolve_save_path(save_path: str, default_filename: str) -> str:
    """Resolve save path, creating directories as needed"""
    if save_path is None:
        save_dir = Path('results/figures')
    else:
        save_path = Path(save_path)
        if save_path.suffix:
            save_dir = save_path.parent
            default_filename = save_path.name
        else:
            save_dir = save_path

    save_dir.mkdir(parents=True, exist_ok=True)
    return str(save_dir / default_filename)


# ---------------------------------------------------------------------------
# 1. Gradient-Based Attention Maps
# ---------------------------------------------------------------------------

def compute_attention_map(network, x_single: np.ndarray,
                          target_class: int = None) -> np.ndarray:
    """
    Compute gradient of the predicted class score w.r.t. the input.
    Shows which input features (spectrogram pixels) matter most.

    Uses a manual forward → backward pass with a one-hot gradient at the
    output to isolate the target class's gradient signal.

    Args:
        network:      NeuralNetwork instance
        x_single:     Single input vector, shape (D,) or (1, D)
        target_class: Class to compute gradient for. If None, uses argmax prediction.

    Returns:
        Attention map (absolute gradient), shape (D,), same as input.
    """
    # Ensure shape is (1, D)
    if x_single.ndim == 1:
        x_single = x_single.reshape(1, -1)

    # Forward pass through all internal layers (not softmax)
    # We want gradient of the logit, not the probability
    out = x_single
    for layer in network.layers:
        out = layer.forward(out)

    # Also run softmax to get predictions (for choosing target class)
    probs = network.softmax.forward(out)

    if target_class is None:
        target_class = np.argmax(probs, axis=1)[0]

    # Create one-hot gradient at the output logits (pre-softmax)
    # We backprop from the logit of the target class
    grad = np.zeros_like(out)
    grad[0, target_class] = 1.0

    # Backward through layers in reverse (skip softmax — we use logit gradient)
    for layer in reversed(network.layers):
        grad = layer.backward(grad)

    # The resulting grad is dlogit/dx — take absolute value as attention
    attention = np.abs(grad.squeeze())

    return attention


def plot_attention_maps(network, X_flat: np.ndarray,
                        y_true_onehot: np.ndarray,
                        class_names: list = None,
                        spec_shape: tuple = (64, 64),
                        n_samples: int = 6,
                        save_path: str = None) -> None:
    """
    Plot spectrograms with attention map overlays (side by side).

    For each sample: shows the original spectrogram, the attention map,
    and the overlay.

    Args:
        network:       NeuralNetwork instance
        X_flat:        Flattened feature vectors, shape (N, D)
        y_true_onehot: One-hot true labels, shape (N, C)
        class_names:   Class name strings
        spec_shape:    (H, W) to reshape flattened vectors
        n_samples:     Number of samples to show
        save_path:     File path to save
    """
    _apply_style()

    n_total = len(X_flat)
    n_samples = min(n_samples, n_total)

    # Pick diverse samples (one per class if possible)
    true_classes = np.argmax(y_true_onehot, axis=1)
    unique_classes = np.unique(true_classes)
    rng = np.random.default_rng(42)

    selected = []
    for cls in unique_classes:
        cls_indices = np.where(true_classes == cls)[0]
        selected.append(rng.choice(cls_indices))
        if len(selected) >= n_samples:
            break

    # Fill remaining if needed
    while len(selected) < n_samples:
        idx = rng.integers(0, n_total)
        if idx not in selected:
            selected.append(idx)

    # 3 columns per sample: spectrogram, attention, overlay
    n_rows = n_samples
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    fig.suptitle('Gradient-Based Attention Maps', fontweight='bold',
                 fontsize=16, y=1.01)

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Column headers
    for j, title in enumerate(['Spectrogram', 'Attention Map', 'Overlay']):
        axes[0, j].set_title(title, fontsize=12, fontweight='bold', pad=10)

    for i, sample_idx in enumerate(selected):
        x_flat = X_flat[sample_idx]
        true_cls = true_classes[sample_idx]
        true_name = class_names[true_cls] if class_names else str(true_cls)

        # Compute attention
        attention = compute_attention_map(network, x_flat, target_class=true_cls)

        # Reshape to 2D
        spec = x_flat.reshape(spec_shape)
        attn_2d = attention.reshape(spec_shape)

        # Normalize attention to [0, 1]
        attn_max = attn_2d.max()
        if attn_max > 0:
            attn_norm = attn_2d / attn_max
        else:
            attn_norm = attn_2d

        # Column 0: Original spectrogram
        ax = axes[i, 0]
        ax.imshow(spec, aspect='auto', origin='lower', cmap='magma')
        ax.set_ylabel(f'{true_name}', fontsize=10, fontweight='bold')
        if i == n_rows - 1:
            ax.set_xlabel('Time')

        # Column 1: Attention heatmap
        ax = axes[i, 1]
        ax.imshow(attn_norm, aspect='auto', origin='lower', cmap='hot')
        if i == n_rows - 1:
            ax.set_xlabel('Time')

        # Column 2: Overlay (spectrogram + attention)
        ax = axes[i, 2]
        ax.imshow(spec, aspect='auto', origin='lower', cmap='magma')
        ax.imshow(attn_norm, aspect='auto', origin='lower', cmap='Reds',
                  alpha=0.5)
        if i == n_rows - 1:
            ax.set_xlabel('Time')

    plt.tight_layout()

    save_path = _resolve_save_path(save_path, 'attention_maps.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   ✓ Attention maps saved → {save_path}")


# ---------------------------------------------------------------------------
# 2. t-SNE Visualization
# ---------------------------------------------------------------------------

def _extract_penultimate_features(network, X: np.ndarray) -> np.ndarray:
    """
    Run forward pass and extract activations from the penultimate layer.

    For default architecture Dense→ReLU→Dense→ReLU→Dense→Softmax,
    this extracts the output of the second ReLU (64-dim features).

    Args:
        network: NeuralNetwork instance
        X:       Input array, shape (N, D)

    Returns:
        Penultimate layer activations, shape (N, hidden_sizes[-1])
    """
    out = X
    # Forward through all layers except the last Dense (output layer)
    # network.layers = [Dense, ReLU, Dense, ReLU, ..., Dense]
    # We want output after the second-to-last element (the last ReLU)
    for layer in network.layers[:-1]:
        out = layer.forward(out)
    return out


def visualize_tsne(network, X: np.ndarray, y_onehot: np.ndarray,
                   class_names: list = None,
                   perplexity: float = 30.0,
                   save_path: str = None) -> None:
    """
    t-SNE scatter plot of penultimate layer activations.

    Extracts hidden features from the network, reduces to 2D using sklearn's
    t-SNE, and plots each sample as a dot colored by class.

    Also saves the raw t-SNE coordinates, labels, predictions, and confidences
    to ``results/metrics/tsne_data.npz`` for the API to serve.

    Args:
        network:    Trained NeuralNetwork instance
        X:          Input features, shape (N, D)
        y_onehot:   One-hot true labels, shape (N, C)
        class_names: List of class name strings
        perplexity:  t-SNE perplexity (default 30.0)
        save_path:   File path to save
    """
    from sklearn.manifold import TSNE

    _apply_style()

    # Extract penultimate layer features
    features = _extract_penultimate_features(network, X)
    true_classes = np.argmax(y_onehot, axis=1)
    num_classes = y_onehot.shape[1]

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # Run forward pass to get predictions + confidences
    probs = network.forward(X)
    pred_classes = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    # Adjust perplexity if dataset is small
    effective_perplexity = min(perplexity, len(X) / 4)

    # Run t-SNE
    print(f"   Running t-SNE on {len(X)} samples ({features.shape[1]}-dim → 2D)...")
    tsne = TSNE(n_components=2, perplexity=effective_perplexity,
                random_state=42, max_iter=1000)
    embedded = tsne.fit_transform(features)

    # ── Save raw data for API ──────────────────────────────────────────
    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    npz_path = metrics_dir / "tsne_data.npz"
    np.savez_compressed(
        npz_path,
        coords=embedded,
        labels=true_classes,
        predictions=pred_classes,
        confidences=confidences,
        class_names=np.array(class_names),
    )
    print(f"   ✓ t-SNE data saved → {npz_path}")

    # ── Plot ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('t-SNE: Learned Feature Space', fontweight='bold', fontsize=16)

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for cls in range(num_classes):
        mask = true_classes == cls
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                   c=[colors[cls]], label=class_names[cls],
                   s=40, alpha=0.7, edgecolors='white', linewidths=0.3)

    ax.legend(loc='best', fontsize=9, framealpha=0.8,
              facecolor='#1a1a2e', edgecolor='#e0e0e0')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.grid(True, linestyle='--', alpha=0.3)

    # Remove tick numbers (t-SNE axes are meaningless)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.tight_layout()

    save_path = _resolve_save_path(save_path, 'tsne_features.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   ✓ t-SNE visualization saved → {save_path}")


# ---------------------------------------------------------------------------
# 3. Gradient Flow / Gradient Norms
# ---------------------------------------------------------------------------

def plot_gradient_norms(network, X_batch: np.ndarray, y_batch: np.ndarray,
                        save_path: str = None) -> None:
    """
    Plot gradient magnitudes at each layer after a forward-backward pass.

    Shows whether gradients are healthy (reasonable norms), vanishing
    (norms → 0 in early layers), or exploding (norms → ∞).

    Args:
        network:  NeuralNetwork instance
        X_batch:  Input batch, shape (batch_size, D)
        y_batch:  One-hot labels, shape (batch_size, C)
        save_path: File path to save
    """
    _apply_style()

    # Forward + backward to populate gradients
    probs = network.forward(X_batch)
    network.compute_loss(probs, y_batch)
    network.backward()

    # Collect gradient stats per Dense layer
    layer_names = []
    grad_weight_norms = []
    grad_bias_norms = []
    grad_weight_means = []
    grad_weight_maxes = []

    for i, layer in enumerate(network.get_trainable_layers()):
        layer_names.append(f'Dense {i+1}\n({layer.input_size}→{layer.output_size})')

        gw = layer.grad_weights
        gb = layer.grad_biases

        grad_weight_norms.append(np.linalg.norm(gw))
        grad_bias_norms.append(np.linalg.norm(gb))
        grad_weight_means.append(np.mean(np.abs(gw)))
        grad_weight_maxes.append(np.max(np.abs(gw)))

    n_layers = len(layer_names)
    x_pos = np.arange(n_layers)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Gradient Flow Analysis', fontweight='bold', fontsize=16, y=1.02)

    # --- Left: L2 Norms ---
    ax = axes[0]
    bar_width = 0.35
    bars1 = ax.bar(x_pos - bar_width / 2, grad_weight_norms, bar_width,
                   label='Weight Gradients', color='#e94560', edgecolor='white',
                   linewidth=0.5)
    bars2 = ax.bar(x_pos + bar_width / 2, grad_bias_norms, bar_width,
                   label='Bias Gradients', color='#0f3460', edgecolor='white',
                   linewidth=0.5)

    ax.set_title('Gradient L2 Norms per Layer')
    ax.set_xlabel('Layer')
    ax.set_ylabel('L2 Norm')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(layer_names, fontsize=9)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Annotate values
    for bar, val in zip(bars1, grad_weight_norms):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=8,
                color='#e94560', fontweight='bold')
    for bar, val in zip(bars2, grad_bias_norms):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=8,
                color='#0f3460', fontweight='bold')

    # --- Right: Mean and Max absolute gradient ---
    ax = axes[1]
    bars3 = ax.bar(x_pos - bar_width / 2, grad_weight_means, bar_width,
                   label='Mean |grad|', color='#53d769', edgecolor='white',
                   linewidth=0.5)
    bars4 = ax.bar(x_pos + bar_width / 2, grad_weight_maxes, bar_width,
                   label='Max |grad|', color='#ff9f43', edgecolor='white',
                   linewidth=0.5)

    ax.set_title('Weight Gradient Statistics per Layer')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Absolute Gradient Value')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(layer_names, fontsize=9)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    for bar, val in zip(bars3, grad_weight_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=8,
                color='#53d769', fontweight='bold')
    for bar, val in zip(bars4, grad_weight_maxes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=8,
                color='#ff9f43', fontweight='bold')

    plt.tight_layout()

    save_path = _resolve_save_path(save_path, 'gradient_flow.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   ✓ Gradient flow saved → {save_path}")


# ---------------------------------------------------------------------------
# 4. Per-Class Metrics Bar Chart
# ---------------------------------------------------------------------------

def plot_per_class_metrics(per_class: list, save_path: str = None) -> None:
    """
    Grouped bar chart showing precision, recall, and F1 for each class.

    Args:
        per_class: List of dicts from Evaluator.evaluate()['per_class'],
                   each with 'class_name', 'precision', 'recall', 'f1'
        save_path: File path to save
    """
    _apply_style()

    class_names = [m['class_name'] for m in per_class]
    precisions = [m['precision'] for m in per_class]
    recalls = [m['recall'] for m in per_class]
    f1s = [m['f1'] for m in per_class]

    n_classes = len(class_names)
    x_pos = np.arange(n_classes)
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_p = ax.bar(x_pos - bar_width, precisions, bar_width,
                    label='Precision', color='#e94560', edgecolor='white',
                    linewidth=0.5)
    bars_r = ax.bar(x_pos, recalls, bar_width,
                    label='Recall', color='#0f3460', edgecolor='white',
                    linewidth=0.5)
    bars_f = ax.bar(x_pos + bar_width, f1s, bar_width,
                    label='F1-Score', color='#53d769', edgecolor='white',
                    linewidth=0.5)

    ax.set_title('Per-Class Metrics', fontweight='bold', fontsize=14, pad=15)
    ax.set_xlabel('Instrument Class', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Annotate F1 values on top of bars
    for bar, val in zip(bars_f, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8,
                color='#53d769', fontweight='bold')

    # Macro F1 line
    macro_f1 = np.mean(f1s)
    ax.axhline(y=macro_f1, color='#ff9f43', linestyle='--', linewidth=1.5,
               alpha=0.8, label=f'Macro F1: {macro_f1:.3f}')
    ax.legend(loc='upper right')

    plt.tight_layout()

    save_path = _resolve_save_path(save_path, 'per_class_metrics.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   ✓ Per-class metrics saved → {save_path}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_attention():
    """Test all advanced visualization functions with synthetic data"""

    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    from src.neural_network.network import NeuralNetwork
    from src.training.trainer import Trainer
    from src.training.evaluator import Evaluator

    print("=" * 60)
    print("Testing Advanced Visualization — attention.py")
    print("=" * 60)

    passed = 0
    total = 0

    # --- Setup: Create and train a small network ---
    np.random.seed(42)
    input_size = 64
    num_classes = 5
    class_names = ['bass', 'brass', 'flute', 'guitar', 'keyboard']
    spec_shape = (8, 8)

    n_train = 300
    n_val = 75
    n_test = 75

    X_train = np.random.randn(n_train, input_size)
    y_train_idx = np.random.randint(num_classes, size=n_train)
    for i in range(n_train):
        X_train[i, :10] += y_train_idx[i] * 0.5
    y_train = np.zeros((n_train, num_classes))
    for i in range(n_train):
        y_train[i, y_train_idx[i]] = 1.0

    X_val = np.random.randn(n_val, input_size)
    y_val_idx = np.random.randint(num_classes, size=n_val)
    for i in range(n_val):
        X_val[i, :10] += y_val_idx[i] * 0.5
    y_val = np.zeros((n_val, num_classes))
    for i in range(n_val):
        y_val[i, y_val_idx[i]] = 1.0

    X_test = np.random.randn(n_test, input_size)
    y_test_idx = np.random.randint(num_classes, size=n_test)
    for i in range(n_test):
        X_test[i, :10] += y_test_idx[i] * 0.5
    y_test = np.zeros((n_test, num_classes))
    for i in range(n_test):
        y_test[i, y_test_idx[i]] = 1.0

    # Train
    net = NeuralNetwork(input_size=input_size, hidden_sizes=[32, 16],
                        num_classes=num_classes, seed=42)
    trainer = Trainer(net, learning_rate=0.1, lr_decay=0.95,
                      batch_size=32, epochs=15, seed=42)

    print("\n--- Training small network (15 epochs) ---")
    history = trainer.train(X_train, y_train, X_val, y_val)

    # Evaluate
    evaluator = Evaluator(net, class_names=class_names)
    results = evaluator.evaluate(X_test, y_test)

    test_save_dir = 'results/figures/test'

    # ---- Test 1: compute_attention_map returns correct shape ----
    print("\n1️⃣  compute_attention_map returns correct shape")

    total += 1
    try:
        attn = compute_attention_map(net, X_test[0])
        if attn.shape == (input_size,) and not np.any(np.isnan(attn)):
            print(f"   ✓ Shape: {attn.shape}, no NaN, range: [{attn.min():.4f}, {attn.max():.4f}]")
            passed += 1
        else:
            print(f"   ✗ FAILED: shape={attn.shape}, has NaN={np.any(np.isnan(attn))}")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")

    # ---- Test 2: Attention map is non-trivial (not all zeros) ----
    print("\n2️⃣  Attention map is non-trivial")

    total += 1
    if attn.max() > 0:
        print(f"   ✓ Non-zero attention values (max={attn.max():.4f})")
        passed += 1
    else:
        print(f"   ✗ FAILED: all zeros")

    # ---- Test 3: plot_attention_maps saves a file ----
    print("\n3️⃣  plot_attention_maps saves a file")

    total += 1
    try:
        save_file = os.path.join(test_save_dir, 'attention_maps.png')
        plot_attention_maps(
            network=net,
            X_flat=X_test,
            y_true_onehot=y_test,
            class_names=class_names,
            spec_shape=spec_shape,
            n_samples=4,
            save_path=test_save_dir
        )
        if os.path.exists(save_file) and os.path.getsize(save_file) > 0:
            size_kb = os.path.getsize(save_file) / 1024
            print(f"   ✓ Saved: {save_file} ({size_kb:.1f} KB)")
            passed += 1
        else:
            print(f"   ✗ FAILED: file not created")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")

    # ---- Test 4: _extract_penultimate_features returns correct shape ----
    print("\n4️⃣  _extract_penultimate_features returns correct shape")

    total += 1
    try:
        features = _extract_penultimate_features(net, X_test)
        expected_dim = net.hidden_sizes[-1]  # 16
        if features.shape == (n_test, expected_dim):
            print(f"   ✓ Shape: {features.shape} (penultimate dim = {expected_dim})")
            passed += 1
        else:
            print(f"   ✗ FAILED: shape={features.shape}, expected ({n_test}, {expected_dim})")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")

    # ---- Test 5: visualize_tsne saves a file ----
    print("\n5️⃣  visualize_tsne saves a file")

    total += 1
    try:
        save_file = os.path.join(test_save_dir, 'tsne_features.png')
        visualize_tsne(
            network=net,
            X=X_test,
            y_onehot=y_test,
            class_names=class_names,
            perplexity=10.0,  # Low for small test data
            save_path=test_save_dir
        )
        if os.path.exists(save_file) and os.path.getsize(save_file) > 0:
            size_kb = os.path.getsize(save_file) / 1024
            print(f"   ✓ Saved: {save_file} ({size_kb:.1f} KB)")
            passed += 1
        else:
            print(f"   ✗ FAILED: file not created")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")

    # ---- Test 6: plot_gradient_norms saves a file ----
    print("\n6️⃣  plot_gradient_norms saves a file")

    total += 1
    try:
        save_file = os.path.join(test_save_dir, 'gradient_flow.png')
        plot_gradient_norms(
            network=net,
            X_batch=X_test[:32],
            y_batch=y_test[:32],
            save_path=test_save_dir
        )
        if os.path.exists(save_file) and os.path.getsize(save_file) > 0:
            size_kb = os.path.getsize(save_file) / 1024
            print(f"   ✓ Saved: {save_file} ({size_kb:.1f} KB)")
            passed += 1
        else:
            print(f"   ✗ FAILED: file not created")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")

    # ---- Test 7: Gradient norms are reasonable (not vanishing/exploding) ----
    print("\n7️⃣  Gradient norms are reasonable")

    total += 1
    all_reasonable = True
    for i, layer in enumerate(net.get_trainable_layers()):
        norm = np.linalg.norm(layer.grad_weights)
        if norm == 0 or np.isinf(norm) or np.isnan(norm):
            all_reasonable = False
            break

    if all_reasonable:
        norms = [np.linalg.norm(l.grad_weights) for l in net.get_trainable_layers()]
        print(f"   ✓ Layer norms: {[f'{n:.2f}' for n in norms]} (no vanishing/exploding)")
        passed += 1
    else:
        print(f"   ✗ FAILED: zero, inf, or NaN gradient norms")

    # ---- Test 8: plot_per_class_metrics saves a file ----
    print("\n8️⃣  plot_per_class_metrics saves a file")

    total += 1
    try:
        save_file = os.path.join(test_save_dir, 'per_class_metrics.png')
        plot_per_class_metrics(
            per_class=results['per_class'],
            save_path=test_save_dir
        )
        if os.path.exists(save_file) and os.path.getsize(save_file) > 0:
            size_kb = os.path.getsize(save_file) / 1024
            print(f"   ✓ Saved: {save_file} ({size_kb:.1f} KB)")
            passed += 1
        else:
            print(f"   ✗ FAILED: file not created")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")

    # ---- Test 9: All generated files are high-res ----
    print("\n9️⃣  All figures are high-resolution (>10KB)")

    total += 1
    all_files = [
        os.path.join(test_save_dir, 'attention_maps.png'),
        os.path.join(test_save_dir, 'tsne_features.png'),
        os.path.join(test_save_dir, 'gradient_flow.png'),
        os.path.join(test_save_dir, 'per_class_metrics.png'),
    ]
    all_hires = True
    for f in all_files:
        if os.path.exists(f):
            size_kb = os.path.getsize(f) / 1024
            if size_kb < 10:
                print(f"   ✗ {f} only {size_kb:.1f} KB")
                all_hires = False
        else:
            print(f"   ✗ {f} missing")
            all_hires = False

    if all_hires:
        print(f"   ✓ All {len(all_files)} figures are high-res (150 DPI)")
        passed += 1

    # ---- Test 10: Network still works after attention/gradient passes ----
    print("\n🔟  Network still functional after visualization passes")

    total += 1
    try:
        probs = net.forward(X_test[:5])
        preds = net.predict(X_test[:5])
        if probs.shape == (5, num_classes) and preds.shape == (5,):
            print(f"   ✓ Forward and predict still work after visualization")
            passed += 1
        else:
            print(f"   ✗ FAILED: unexpected shapes")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    if passed == total:
        print(f"✅ All {total} tests passed!")
    else:
        print(f"❌ {passed}/{total} tests passed")
    print("=" * 60)

    # List generated files
    print(f"\n📁 Generated test figures:")
    for f in sorted(all_files):
        if os.path.exists(f):
            size_kb = os.path.getsize(f) / 1024
            print(f"   {f} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    test_attention()
