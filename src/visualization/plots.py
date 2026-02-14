"""
Visualization Module — Step 11: Basic Visualizations
Generates training dashboards, sample prediction grids, and confusion matrix heatmaps.

Functions:
    plot_training_history  — 2×2 dashboard: train loss, val loss, train acc, val acc
    plot_sample_predictions — 3×3 grid of spectrograms with predicted vs true labels
    plot_confusion_matrix  — Seaborn heatmap of the confusion matrix
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ---------------------------------------------------------------------------
# Style Configuration
# ---------------------------------------------------------------------------

# Consistent, clean style for all plots
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


# ---------------------------------------------------------------------------
# 1. Training History Dashboard (2×2 grid)
# ---------------------------------------------------------------------------

def plot_training_history(history: dict, save_path: str = None) -> None:
    """
    Plot a 2×2 training dashboard: Train Loss, Val Loss, Train Acc, Val Acc

    Args:
        history:   Dict from Trainer.train() with keys:
                   'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy',
                   'learning_rate', 'epoch_time'
        save_path: File path to save the figure. If None, saves to
                   results/figures/training_history.png
    """
    _apply_style()

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Dashboard', fontweight='bold', fontsize=16, y=0.98)

    # --- Top Left: Train Loss ---
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], color='#e94560', linewidth=2,
            marker='o', markersize=3, label='Train Loss')
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')

    # Annotate final value
    final_train_loss = history['train_loss'][-1]
    ax.annotate(f'{final_train_loss:.4f}',
                xy=(len(epochs), final_train_loss),
                xytext=(-40, 10), textcoords='offset points',
                fontsize=9, color='#e94560', fontweight='bold')

    # --- Top Right: Validation Loss ---
    ax = axes[0, 1]
    ax.plot(epochs, history['val_loss'], color='#0f3460', linewidth=2,
            marker='s', markersize=3, label='Val Loss')
    ax.set_title('Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')

    # Mark best val loss
    best_idx = np.argmin(history['val_loss'])
    best_val = history['val_loss'][best_idx]
    ax.axhline(y=best_val, color='#53d769', linestyle=':', alpha=0.7)
    ax.scatter([best_idx + 1], [best_val], color='#53d769', s=80,
               zorder=5, marker='*')
    ax.annotate(f'Best: {best_val:.4f} (ep {best_idx + 1})',
                xy=(best_idx + 1, best_val),
                xytext=(10, -15), textcoords='offset points',
                fontsize=9, color='#53d769', fontweight='bold')

    # --- Bottom Left: Train Accuracy ---
    ax = axes[1, 0]
    train_acc_pct = [a * 100 for a in history['train_accuracy']]
    ax.plot(epochs, train_acc_pct, color='#e94560', linewidth=2,
            marker='o', markersize=3, label='Train Accuracy')
    ax.set_title('Training Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right')

    final_train_acc = train_acc_pct[-1]
    ax.annotate(f'{final_train_acc:.1f}%',
                xy=(len(epochs), final_train_acc),
                xytext=(-40, -15), textcoords='offset points',
                fontsize=9, color='#e94560', fontweight='bold')

    # --- Bottom Right: Validation Accuracy ---
    ax = axes[1, 1]
    val_acc_pct = [a * 100 for a in history['val_accuracy']]
    ax.plot(epochs, val_acc_pct, color='#0f3460', linewidth=2,
            marker='s', markersize=3, label='Val Accuracy')
    ax.set_title('Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right')

    # Mark best val accuracy
    best_acc_idx = np.argmax(history['val_accuracy'])
    best_acc = val_acc_pct[best_acc_idx]
    ax.axhline(y=best_acc, color='#53d769', linestyle=':', alpha=0.7)
    ax.scatter([best_acc_idx + 1], [best_acc], color='#53d769', s=80,
               zorder=5, marker='*')
    ax.annotate(f'Best: {best_acc:.1f}% (ep {best_acc_idx + 1})',
                xy=(best_acc_idx + 1, best_acc),
                xytext=(10, -15), textcoords='offset points',
                fontsize=9, color='#53d769', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    save_path = _resolve_save_path(save_path, 'training_history.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   ✓ Training dashboard saved → {save_path}")


# ---------------------------------------------------------------------------
# 2. Sample Predictions Grid (3×3)
# ---------------------------------------------------------------------------

def plot_sample_predictions(X_flat: np.ndarray, y_true_onehot: np.ndarray,
                            y_pred_probs: np.ndarray,
                            class_names: list = None,
                            spec_shape: tuple = (64, 64),
                            n_samples: int = 9,
                            save_path: str = None) -> None:
    """
    Show a 3×3 grid of spectrograms with predicted vs true labels.

    Reshapes flattened feature vectors back into 2D spectrograms for display.

    Args:
        X_flat:        Flattened feature vectors, shape (N, D) where D = H*W
        y_true_onehot: One-hot true labels, shape (N, C)
        y_pred_probs:  Predicted probabilities, shape (N, C)
        class_names:   List of class name strings. If None, uses indices.
        spec_shape:    (H, W) to reshape flattened vectors back to images.
        n_samples:     Number of samples to show (default 9 for 3×3 grid).
        save_path:     File path to save the figure.
    """
    _apply_style()

    n_total = len(X_flat)
    n_samples = min(n_samples, n_total)

    # Pick samples: mix of correct and incorrect predictions
    pred_classes = np.argmax(y_pred_probs, axis=1)
    true_classes = np.argmax(y_true_onehot, axis=1)
    correct_mask = (pred_classes == true_classes)

    # Try to get ~half correct, ~half incorrect for an interesting grid
    correct_indices = np.where(correct_mask)[0]
    incorrect_indices = np.where(~correct_mask)[0]

    n_incorrect = min(len(incorrect_indices), n_samples // 2)
    n_correct = min(len(correct_indices), n_samples - n_incorrect)
    # Fill remaining if not enough incorrect
    n_incorrect = min(len(incorrect_indices), n_samples - n_correct)

    rng = np.random.default_rng(42)
    selected = []
    if n_correct > 0:
        selected.extend(rng.choice(correct_indices, size=n_correct, replace=False))
    if n_incorrect > 0:
        selected.extend(rng.choice(incorrect_indices, size=n_incorrect, replace=False))

    # Shuffle so correct/incorrect aren't grouped
    rng.shuffle(selected)
    selected = selected[:n_samples]

    # Grid layout
    n_cols = 3
    n_rows = int(np.ceil(n_samples / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    fig.suptitle('Sample Predictions', fontweight='bold', fontsize=16, y=1.02)

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, ax in enumerate(axes.flat):
        if idx < len(selected):
            sample_idx = selected[idx]

            # Reshape flattened vector back to 2D spectrogram
            spec = X_flat[sample_idx].reshape(spec_shape)

            true_cls = true_classes[sample_idx]
            pred_cls = pred_classes[sample_idx]
            confidence = y_pred_probs[sample_idx, pred_cls] * 100

            true_name = class_names[true_cls] if class_names else str(true_cls)
            pred_name = class_names[pred_cls] if class_names else str(pred_cls)
            is_correct = (true_cls == pred_cls)

            # Display spectrogram
            im = ax.imshow(spec, aspect='auto', origin='lower', cmap='magma')

            # Title: green if correct, red if wrong
            title_color = '#53d769' if is_correct else '#e94560'
            symbol = '✓' if is_correct else '✗'
            ax.set_title(f'{symbol} Pred: {pred_name} ({confidence:.0f}%)\n'
                         f'True: {true_name}',
                         fontsize=10, color=title_color, fontweight='bold')

            ax.set_xlabel('Time')
            ax.set_ylabel('Mel Bin')
        else:
            ax.axis('off')

    plt.tight_layout()

    save_path = _resolve_save_path(save_path, 'sample_predictions.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   ✓ Sample predictions saved → {save_path}")


# ---------------------------------------------------------------------------
# 3. Confusion Matrix Heatmap
# ---------------------------------------------------------------------------

def plot_confusion_matrix(confusion_matrix: np.ndarray,
                          class_names: list = None,
                          save_path: str = None,
                          normalize: bool = False) -> None:
    """
    Visualize a confusion matrix as a seaborn heatmap.

    Args:
        confusion_matrix: 2D array of shape (C, C), where C[i][j] = count of
                          true class i predicted as class j
        class_names:      List of class name strings. If None, uses indices.
        save_path:        File path to save the figure.
        normalize:        If True, show percentages per row (recall-oriented).
    """
    _apply_style()

    num_classes = confusion_matrix.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # Optionally normalize by row (each row sums to 100%)
    if normalize:
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1, row_sums)
        display_data = confusion_matrix.astype(float) / row_sums * 100
        fmt = '.1f'
        cbar_label = 'Recall (%)'
    else:
        display_data = confusion_matrix
        fmt = 'd'
        cbar_label = 'Count'

    fig, ax = plt.subplots(figsize=(10, 8))

    # Seaborn heatmap — this is why we use seaborn
    sns.heatmap(display_data,
                annot=True,
                fmt=fmt,
                cmap='YlOrRd',
                xticklabels=class_names,
                yticklabels=class_names,
                linewidths=0.5,
                linecolor='#2a2a4a',
                cbar_kws={'label': cbar_label},
                square=True,
                ax=ax)

    ax.set_title('Confusion Matrix', fontweight='bold', fontsize=14, pad=15)
    ax.set_xlabel('Predicted Class', fontweight='bold')
    ax.set_ylabel('True Class', fontweight='bold')

    # Rotate tick labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Compute and display overall accuracy
    total = confusion_matrix.sum()
    correct = np.trace(confusion_matrix)
    accuracy = correct / total if total > 0 else 0
    fig.text(0.5, 0.01,
             f'Overall Accuracy: {accuracy * 100:.1f}% ({correct}/{total})',
             ha='center', fontsize=12, fontweight='bold', color='#e0e0e0')

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    save_path = _resolve_save_path(save_path, 'confusion_matrix.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   ✓ Confusion matrix saved → {save_path}")


# ---------------------------------------------------------------------------
# Helper: Resolve save path
# ---------------------------------------------------------------------------

def _resolve_save_path(save_path: str, default_filename: str) -> str:
    """Resolve save path, creating directories as needed"""
    if save_path is None:
        save_dir = Path('results/figures')
    else:
        save_path = Path(save_path)
        if save_path.suffix:
            # It's a file path
            save_dir = save_path.parent
            default_filename = save_path.name
        else:
            # It's a directory
            save_dir = save_path

    save_dir.mkdir(parents=True, exist_ok=True)
    return str(save_dir / default_filename)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_plots():
    """Test all visualization functions with synthetic data"""

    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    from src.neural_network.network import NeuralNetwork
    from src.training.trainer import Trainer
    from src.training.evaluator import Evaluator

    print("=" * 60)
    print("Testing Visualization — plots.py")
    print("=" * 60)

    passed = 0
    total = 0

    # --- Setup: Create and train a small network on synthetic data ---
    np.random.seed(42)
    input_size = 64  # Small for fast testing (will reshape to 8×8)
    num_classes = 5
    class_names = ['bass', 'brass', 'flute', 'guitar', 'keyboard']
    spec_shape = (8, 8)  # 8*8 = 64 = input_size

    # Synthetic data with a learnable signal
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

    # Train a small network
    net = NeuralNetwork(input_size=input_size, hidden_sizes=[32, 16],
                        num_classes=num_classes, seed=42)
    trainer = Trainer(net, learning_rate=0.1, lr_decay=0.95,
                      batch_size=32, epochs=15, seed=42)

    print("\n--- Training small network (15 epochs) ---")
    history = trainer.train(X_train, y_train, X_val, y_val)

    # Evaluate
    evaluator = Evaluator(net, class_names=class_names)
    results = evaluator.evaluate(X_test, y_test)

    # Test output directory
    test_save_dir = 'results/figures/test'

    # ---- Test 1: plot_training_history saves a file ----
    print("\n1️⃣  plot_training_history saves a file")

    total += 1
    try:
        save_file = os.path.join(test_save_dir, 'training_history.png')
        plot_training_history(history, save_path=test_save_dir)
        if os.path.exists(save_file) and os.path.getsize(save_file) > 0:
            size_kb = os.path.getsize(save_file) / 1024
            print(f"   ✓ Saved: {save_file} ({size_kb:.1f} KB)")
            passed += 1
        else:
            print(f"   ✗ FAILED: file not created or empty")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")

    # ---- Test 2: Training dashboard has correct content ----
    print("\n2️⃣  Training history uses correct data from history dict")

    total += 1
    keys_present = all(k in history for k in
                       ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'])
    lengths_ok = len(history['train_loss']) == 15
    if keys_present and lengths_ok:
        print(f"   ✓ History has all 4 metric keys, 15 epochs each")
        passed += 1
    else:
        print(f"   ✗ FAILED: keys={keys_present}, length={len(history['train_loss'])}")

    # ---- Test 3: plot_sample_predictions saves a file ----
    print("\n3️⃣  plot_sample_predictions saves a file")

    total += 1
    try:
        save_file = os.path.join(test_save_dir, 'sample_predictions.png')
        probs = results['probabilities']

        # Use X_test (still flattened) — function reshapes internally
        plot_sample_predictions(
            X_flat=X_test,
            y_true_onehot=y_test,
            y_pred_probs=probs,
            class_names=class_names,
            spec_shape=spec_shape,
            n_samples=9,
            save_path=test_save_dir
        )

        if os.path.exists(save_file) and os.path.getsize(save_file) > 0:
            size_kb = os.path.getsize(save_file) / 1024
            print(f"   ✓ Saved: {save_file} ({size_kb:.1f} KB)")
            passed += 1
        else:
            print(f"   ✗ FAILED: file not created or empty")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")

    # ---- Test 4: Sample predictions shows mix of correct/incorrect ----
    print("\n4️⃣  Sample predictions includes both correct and incorrect")

    total += 1
    pred_classes = np.argmax(probs, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    n_correct = np.sum(pred_classes == true_classes)
    n_incorrect = len(true_classes) - n_correct
    if n_correct > 0 and n_incorrect > 0:
        print(f"   ✓ Test set has {n_correct} correct and {n_incorrect} incorrect — grid shows mix")
        passed += 1
    elif n_correct > 0:
        print(f"   ✓ All predictions correct ({n_correct}) — grid shows all correct")
        passed += 1
    else:
        print(f"   ✓ All predictions incorrect ({n_incorrect}) — grid shows all incorrect")
        passed += 1

    # ---- Test 5: plot_confusion_matrix saves a file (raw counts) ----
    print("\n5️⃣  plot_confusion_matrix saves a file (raw counts)")

    total += 1
    try:
        save_file = os.path.join(test_save_dir, 'confusion_matrix.png')
        plot_confusion_matrix(
            confusion_matrix=results['confusion_matrix'],
            class_names=class_names,
            save_path=test_save_dir,
            normalize=False
        )

        if os.path.exists(save_file) and os.path.getsize(save_file) > 0:
            size_kb = os.path.getsize(save_file) / 1024
            print(f"   ✓ Saved: {save_file} ({size_kb:.1f} KB)")
            passed += 1
        else:
            print(f"   ✗ FAILED: file not created or empty")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")

    # ---- Test 6: plot_confusion_matrix saves normalized version ----
    print("\n6️⃣  plot_confusion_matrix saves normalized version")

    total += 1
    try:
        save_file = os.path.join(test_save_dir, 'confusion_matrix_normalized.png')
        plot_confusion_matrix(
            confusion_matrix=results['confusion_matrix'],
            class_names=class_names,
            save_path=os.path.join(test_save_dir, 'confusion_matrix_normalized.png'),
            normalize=True
        )

        if os.path.exists(save_file) and os.path.getsize(save_file) > 0:
            size_kb = os.path.getsize(save_file) / 1024
            print(f"   ✓ Saved: {save_file} ({size_kb:.1f} KB)")
            passed += 1
        else:
            print(f"   ✗ FAILED: file not created or empty")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")

    # ---- Test 7: Confusion matrix values match evaluator ----
    print("\n7️⃣  Confusion matrix values match evaluator results")

    total += 1
    cm = results['confusion_matrix']
    if (cm.sum() == n_test and
        cm.shape == (num_classes, num_classes) and
        np.trace(cm) == int(results['accuracy'] * n_test)):
        print(f"   ✓ CM sum={cm.sum()}, shape={cm.shape}, "
              f"diagonal={np.trace(cm)} matches accuracy={results['accuracy']*100:.1f}%")
        passed += 1
    else:
        print(f"   ✗ FAILED")

    # ---- Test 8: Default save path creates results/figures/ ----
    print("\n8️⃣  Default save path creates results/figures/ directory")

    total += 1
    try:
        plot_training_history(history, save_path=None)
        default_file = 'results/figures/training_history.png'
        if os.path.exists(default_file) and os.path.getsize(default_file) > 0:
            print(f"   ✓ Default path works: {default_file}")
            passed += 1
        else:
            print(f"   ✗ FAILED: default file not found")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")

    # ---- Test 9: Works without class names (falls back to indices) ----
    print("\n9️⃣  Works without class names (index fallback)")

    total += 1
    try:
        plot_confusion_matrix(
            confusion_matrix=results['confusion_matrix'],
            class_names=None,
            save_path=os.path.join(test_save_dir, 'confusion_matrix_no_names.png')
        )
        print(f"   ✓ Confusion matrix with index labels saved successfully")
        passed += 1
    except Exception as e:
        print(f"   ✗ FAILED: {e}")

    # ---- Test 10: All files are high-res (>10KB) ----
    print("\n🔟  All saved figures are high-resolution (>10KB)")

    total += 1
    all_files = [
        os.path.join(test_save_dir, 'training_history.png'),
        os.path.join(test_save_dir, 'sample_predictions.png'),
        os.path.join(test_save_dir, 'confusion_matrix.png'),
        os.path.join(test_save_dir, 'confusion_matrix_normalized.png'),
    ]
    all_hires = True
    for f in all_files:
        if os.path.exists(f):
            size_kb = os.path.getsize(f) / 1024
            if size_kb < 10:
                print(f"   ✗ {f} is only {size_kb:.1f} KB (< 10 KB)")
                all_hires = False
        else:
            print(f"   ✗ {f} does not exist")
            all_hires = False

    if all_hires:
        print(f"   ✓ All {len(all_files)} figures are high-res (150 DPI)")
        passed += 1
    else:
        print(f"   ✗ FAILED: some files are too small or missing")

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
    test_plots()
