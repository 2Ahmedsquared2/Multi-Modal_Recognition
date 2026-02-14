#!/usr/bin/env python3
"""
Main Training Script — Acoustic Pattern Recognition Engine

End-to-end pipeline: data loading → training → evaluation → visualizations → model saving.
Wires together all components from Steps 1–12 and runs on real data.

Usage:
    # Default settings (recommended)
    python train.py

    # Custom hyperparameters
    python train.py --lr 0.005 --epochs 150 --batch-size 64 --hidden 256 128

    # Skip visualizations (training only)
    python train.py --no-viz

    # Use pre-prepared data (fastest)
    python train.py --prepared-data data/prepared/prepared_data.npz
"""

import argparse
import time
import sys
import os
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Imports from project modules
# ---------------------------------------------------------------------------

from src.preprocessing.audio_loader import AudioLoader
from src.preprocessing.spectrogram_gen import SpectrogramGenerator
from src.preprocessing.data_prep import DataPreparator
from src.neural_network.network import NeuralNetwork
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.visualization.plots import (
    plot_training_history,
    plot_sample_predictions,
    plot_confusion_matrix,
)
from src.visualization.attention import (
    plot_attention_maps,
    visualize_tsne,
    plot_gradient_norms,
    plot_per_class_metrics,
)


# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Acoustic Pattern Recognition Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                          # Default settings
  python train.py --lr 0.005 --epochs 150  # Custom hyperparameters
  python train.py --hidden 256 128 64      # Deeper network
  python train.py --no-viz                 # Skip visualizations
        """,
    )

    # Data paths
    parser.add_argument("--audio-dir", type=str, default="data/raw/organized",
                        help="Directory with organized audio files (default: data/raw/organized)")
    parser.add_argument("--spectrogram-path", type=str, default="data/spectrograms/spectrograms.npz",
                        help="Path to pre-generated spectrograms (default: data/spectrograms/spectrograms.npz)")
    parser.add_argument("--prepared-data", type=str, default=None,
                        help="Path to pre-prepared data file. If provided, skips spectrogram "
                             "generation and data prep (fastest startup)")

    # Architecture
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64],
                        help="Hidden layer sizes (default: 128 64)")
    parser.add_argument("--resolution", type=int, default=64,
                        help="Spectrogram resolution NxN (default: 64)")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Initial learning rate (default: 0.01)")
    parser.add_argument("--lr-decay", type=float, default=0.95,
                        help="LR decay factor per epoch (default: 0.95)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Mini-batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum training epochs (default: 100)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (default: 10, 0 to disable)")

    # Output
    parser.add_argument("--output-dir", type=str, default="results/figures",
                        help="Directory for saved visualizations (default: results/figures)")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Directory for saved model weights (default: models)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization generation")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline Stages
# ---------------------------------------------------------------------------

def stage_load_data(args) -> dict:
    """
    Stage 1: Load and prepare data
    
    Three paths (fastest → slowest):
    1. Load pre-prepared data from .npz (--prepared-data)
    2. Load pre-generated spectrograms → run data prep
    3. Load raw audio → generate spectrograms → run data prep
    """
    print("\n" + "=" * 70)
    print("STAGE 1: DATA LOADING")
    print("=" * 70)

    prep = DataPreparator(
        spectrogram_path=args.spectrogram_path,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed,
    )

    # --- Path 1: Pre-prepared data exists (fastest) ---
    if args.prepared_data and Path(args.prepared_data).exists():
        print(f"\n⚡ Loading pre-prepared data (fastest path)")
        data = prep.load_prepared_data(args.prepared_data)
        return data

    # Check for default prepared data location
    default_prepared = Path("data/prepared/prepared_data.npz")
    if args.prepared_data is None and default_prepared.exists():
        print(f"\n⚡ Found cached prepared data at {default_prepared}")
        data = prep.load_prepared_data(str(default_prepared))
        return data

    # --- Path 2: Pre-generated spectrograms exist ---
    spectrogram_file = Path(args.spectrogram_path)
    if spectrogram_file.exists():
        print(f"\n📊 Found pre-generated spectrograms at {spectrogram_file}")
        data = prep.prepare()
        prep.save_prepared_data("data/prepared")
        return data

    # --- Path 3: Generate everything from scratch ---
    print(f"\n🔧 No cached data found. Generating from scratch...")

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        print(f"\n❌ Audio directory not found: {audio_dir}")
        print(f"   Please ensure your audio data is organized at: {audio_dir}")
        print(f"   Expected structure: {audio_dir}/<class_name>/<audio_files>.wav")
        print(f"\n   If you haven't downloaded the dataset yet, run:")
        print(f"     python download_dataset.py")
        sys.exit(1)

    # Step A: Load audio files
    print(f"\n📂 Loading audio files from: {audio_dir}")
    audio_loader = AudioLoader(
        data_dir=str(audio_dir),
        sample_rate=22050,
        duration=2.0,
        augment=False,  # No augmentation for pre-generation
    )

    # Step B: Generate spectrograms
    print(f"\n🎨 Generating {args.resolution}x{args.resolution} spectrograms...")
    spec_gen = SpectrogramGenerator(
        target_shape=(args.resolution, args.resolution),
        sample_rate=22050,
    )
    spec_gen.generate_and_save_dataset(
        audio_loader=audio_loader,
        output_dir="data/spectrograms",
    )

    # Step C: Run data preparation pipeline
    prep = DataPreparator(
        spectrogram_path="data/spectrograms/spectrograms.npz",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed,
    )
    data = prep.prepare()
    prep.save_prepared_data("data/prepared")

    return data


def stage_train(args, data: dict) -> tuple:
    """
    Stage 2: Build network and train
    
    Returns:
        (network, history, trainer) tuple
    """
    print("\n" + "=" * 70)
    print("STAGE 2: TRAINING")
    print("=" * 70)

    input_size = data['input_dim']
    num_classes = data['num_classes']

    # Build network
    print(f"\n🧠 Building network...")
    net = NeuralNetwork(
        input_size=input_size,
        hidden_sizes=args.hidden,
        num_classes=num_classes,
        seed=args.seed,
    )
    net.summary()

    # Create trainer
    patience = args.patience if args.patience > 0 else None
    trainer = Trainer(
        network=net,
        learning_rate=args.lr,
        lr_decay=args.lr_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=patience,
        seed=args.seed,
    )

    # Train
    print(f"\n🚀 Starting training...")
    history = trainer.train(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_val'],
        y_val=data['y_val'],
    )

    return net, history, trainer


def stage_evaluate(args, net, data: dict) -> dict:
    """
    Stage 3: Evaluate on test set
    
    Returns:
        Evaluation results dict
    """
    print("\n" + "=" * 70)
    print("STAGE 3: EVALUATION")
    print("=" * 70)

    evaluator = Evaluator(
        network=net,
        class_names=data['class_names'],
    )

    results = evaluator.evaluate(data['X_test'], data['y_test'])

    print()
    evaluator.print_report()
    evaluator.print_confusion_matrix()

    return results


def stage_visualize(args, net, history, data, results):
    """
    Stage 4: Generate all visualizations
    """
    print("\n" + "=" * 70)
    print("STAGE 4: GENERATING VISUALIZATIONS")
    print("=" * 70)

    save_dir = args.output_dir
    class_names = data['class_names']
    resolution = int(np.sqrt(data['input_dim']))
    spec_shape = (resolution, resolution)

    print(f"\n📊 Saving visualizations to: {save_dir}/")

    # 1. Training history dashboard
    print("\n1/7 Training dashboard...")
    plot_training_history(history, save_path=save_dir)

    # 2. Sample predictions grid
    print("2/7 Sample predictions...")
    plot_sample_predictions(
        X_flat=data['X_test'],
        y_true_onehot=data['y_test'],
        y_pred_probs=results['probabilities'],
        class_names=class_names,
        spec_shape=spec_shape,
        n_samples=9,
        save_path=save_dir,
    )

    # 3. Confusion matrix (raw counts)
    print("3/7 Confusion matrix (raw)...")
    plot_confusion_matrix(
        confusion_matrix=results['confusion_matrix'],
        class_names=class_names,
        save_path=save_dir,
        normalize=False,
    )

    # 4. Confusion matrix (normalized — recall %)
    print("4/7 Confusion matrix (normalized)...")
    plot_confusion_matrix(
        confusion_matrix=results['confusion_matrix'],
        class_names=class_names,
        save_path=str(Path(save_dir) / "confusion_matrix_normalized.png"),
        normalize=True,
    )

    # 5. Attention maps
    print("5/7 Attention maps...")
    plot_attention_maps(
        network=net,
        X_flat=data['X_test'],
        y_true_onehot=data['y_test'],
        class_names=class_names,
        spec_shape=spec_shape,
        n_samples=6,
        save_path=save_dir,
    )

    # 6. t-SNE feature visualization
    print("6/7 t-SNE visualization...")
    visualize_tsne(
        network=net,
        X=data['X_test'],
        y_onehot=data['y_test'],
        class_names=class_names,
        perplexity=30.0,
        save_path=save_dir,
    )

    # 7. Gradient flow analysis
    print("7/7 Gradient flow + per-class metrics...")
    plot_gradient_norms(
        network=net,
        X_batch=data['X_test'][:args.batch_size],
        y_batch=data['y_test'][:args.batch_size],
        save_path=save_dir,
    )

    # 8. Per-class metrics bar chart
    plot_per_class_metrics(
        per_class=results['per_class'],
        save_path=save_dir,
    )

    # List all generated files
    figures_dir = Path(save_dir)
    if figures_dir.exists():
        print(f"\n📁 Generated figures:")
        total_size = 0
        for f in sorted(figures_dir.glob("*.png")):
            size_kb = f.stat().st_size / 1024
            total_size += size_kb
            print(f"   {f.name:<40} {size_kb:>8.1f} KB")
        print(f"   {'TOTAL':<40} {total_size:>8.1f} KB")


def stage_save_model(args, net):
    """
    Stage 5: Save trained model weights
    """
    print("\n" + "=" * 70)
    print("STAGE 5: SAVING MODEL")
    print("=" * 70)

    net.save(args.model_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 70)
    print("🎯 ACOUSTIC PATTERN RECOGNITION ENGINE")
    print("   End-to-end training pipeline")
    print("=" * 70)
    print(f"\n⚙️  Configuration:")
    print(f"   Architecture:    {args.resolution}² → {' → '.join(map(str, args.hidden))} → 10")
    print(f"   Learning rate:   {args.lr} (decay: {args.lr_decay}/epoch)")
    print(f"   Batch size:      {args.batch_size}")
    print(f"   Max epochs:      {args.epochs} (patience: {args.patience})")
    print(f"   Seed:            {args.seed}")
    print(f"   Visualizations:  {'OFF' if args.no_viz else 'ON'}")

    pipeline_start = time.time()

    # --- Stage 1: Data ---
    data = stage_load_data(args)

    # --- Stage 2: Training ---
    net, history, trainer = stage_train(args, data)

    # --- Save training history for the web dashboard ---
    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        metrics_dir / "training_history.npz",
        train_loss=np.array(history['train_loss']),
        val_loss=np.array(history['val_loss']),
        train_accuracy=np.array(history['train_accuracy']),
        val_accuracy=np.array(history['val_accuracy']),
    )
    print(f"\n💾 Saved training history to: {metrics_dir / 'training_history.npz'}")

    # --- Stage 3: Evaluation ---
    results = stage_evaluate(args, net, data)

    # --- Stage 4: Visualizations ---
    if not args.no_viz:
        stage_visualize(args, net, history, data, results)
    else:
        print("\n⏭️  Skipping visualizations (--no-viz)")

    # --- Stage 5: Save Model ---
    stage_save_model(args, net)

    # --- Final Summary ---
    pipeline_time = time.time() - pipeline_start
    final_train_acc = history['train_accuracy'][-1] * 100
    final_val_acc = history['val_accuracy'][-1] * 100
    test_acc = results['accuracy'] * 100
    epochs_trained = len(history['train_loss'])

    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n📊 Results Summary:")
    print(f"   Epochs trained:     {epochs_trained}")
    print(f"   Training accuracy:  {final_train_acc:.1f}%")
    print(f"   Validation accuracy:{final_val_acc:.1f}%")
    print(f"   Test accuracy:      {test_acc:.1f}%")
    print(f"   Macro F1:           {results['macro_f1']:.4f}")
    print(f"   Parameters:         {net.count_parameters():,}")
    print(f"   Total pipeline time:{pipeline_time:.1f}s")
    print(f"\n📁 Outputs:")
    print(f"   Model:         {args.model_dir}/model.npz")
    if not args.no_viz:
        print(f"   Figures:        {args.output_dir}/")
    print(f"\n🎯 Done!")


if __name__ == "__main__":
    main()
