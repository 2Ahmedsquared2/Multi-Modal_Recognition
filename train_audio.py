#!/usr/bin/env python3
"""
Train audio models (NumPy from-scratch + PyTorch) for any registered audio dataset.

End-to-end pipeline:
  1. Check for prepared data → if missing, run full audio pipeline
     (AudioLoader → SpectrogramGenerator → DataPreparator)
  2. Train custom NumPy model (using Trainer from src/training/)
  3. Train PyTorch model (using AudioClassifier)
  4. Generate t-SNE embeddings for both engines
  5. Save all artifacts to registry-defined paths

Artifacts saved per dataset:
  - models/audio/<dataset>/model.npz                     (NumPy weights)
  - models/audio/<dataset>/pytorch_model.pt              (PyTorch weights)
  - models/audio/<dataset>/norm_stats.npz                (mean/std for inference)
  - data/prepared/audio/<dataset>/prepared_data.npz      (train/val/test splits)
  - results/metrics/audio/<dataset>/training_history.npz
  - results/metrics/audio/<dataset>/pytorch_training_history.npz
  - results/metrics/audio/<dataset>/tsne_data.npz
  - results/metrics/audio/<dataset>/pytorch_tsne_data.npz

Usage:
    python train_audio.py --dataset audio/medical
    python train_audio.py --dataset audio/wildlife
    python train_audio.py --dataset audio/urban
    python train_audio.py --dataset audio/medical --epochs 150 --lr 0.005
    python train_audio.py --dataset audio/medical --skip-pytorch
"""

import argparse
import time
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.manifold import TSNE

from src.preprocessing.audio_loader import AudioLoader
from src.preprocessing.spectrogram_gen import SpectrogramGenerator
from src.preprocessing.data_prep import DataPreparator
from src.neural_network.network import NeuralNetwork
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.pytorch_model.model import AudioClassifier
from api.dataset_registry import DatasetRegistry


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train audio classification models (NumPy + PyTorch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_audio.py --dataset audio/medical
  python train_audio.py --dataset audio/wildlife
  python train_audio.py --dataset audio/medical --epochs 150 --lr 0.005
  python train_audio.py --dataset audio/medical --hidden 256 128
  python train_audio.py --dataset audio/medical --skip-pytorch
        """,
    )

    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset key from registry (e.g. audio/medical, audio/wildlife, audio/urban)",
    )

    # Architecture
    parser.add_argument(
        "--hidden", type=int, nargs="+", default=None,
        help="Hidden layer sizes (default: from registry config)",
    )
    parser.add_argument(
        "--resolution", type=int, default=None,
        help="Spectrogram resolution NxN (default: from registry config)",
    )

    # Training
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument("--lr-decay", type=float, default=0.95, help="LR decay per epoch (default: 0.95)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs (default: 100)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Flags
    parser.add_argument("--skip-pytorch", action="store_true", help="Skip PyTorch training")
    parser.add_argument("--skip-tsne", action="store_true", help="Skip t-SNE generation")
    parser.add_argument("--force-preprocess", action="store_true", help="Re-preprocess even if cached data exists")

    return parser.parse_args()


# ── Stage 1: Data Loading & Preprocessing ─────────────────────────────────

def stage_preprocess(args, config) -> dict:
    """
    Load or generate prepared data for the audio dataset.

    Path 1 (fastest): Load pre-prepared data from disk
    Path 2: Run full pipeline: scan audio → generate spectrograms → prepare
    """
    print("\n" + "=" * 70)
    print("STAGE 1: DATA LOADING & PREPROCESSING")
    print("=" * 70)

    prepared_path = Path(config.prepared_data_path)
    target_shape = tuple(config.target_shape)
    sample_rate = config.sample_rate
    duration = config.duration

    if args.resolution:
        target_shape = (args.resolution, args.resolution)

    # --- Fast path: prepared data exists ---
    if prepared_path.exists() and not args.force_preprocess:
        print(f"\n⚡ Found cached prepared data: {prepared_path}")
        prep = DataPreparator(spectrogram_path="")
        data = prep.load_prepared_data(str(prepared_path))
        return data

    # --- Full pipeline ---
    # Determine audio directory from dataset key
    # e.g. "audio/medical" → "data/audio/medical"
    modality, subset = config.key.split("/")
    audio_dir = Path("data") / modality / subset

    if not audio_dir.exists():
        print(f"\n❌ Audio directory not found: {audio_dir}")
        print(f"   Expected structure: {audio_dir}/<class_name>/<audio_files>.wav")
        print(f"\n   Download the dataset first:")
        if subset == "medical":
            print(f"     python download_medical_audio.py")
        elif subset == "wildlife":
            print(f"     python download_wildlife_audio.py")
        elif subset == "urban":
            print(f"     python download_urban_audio.py")
        else:
            print(f"     (download script for {subset} not yet available)")
        sys.exit(1)

    # Step A: Scan audio files
    print(f"\n📂 Scanning audio from: {audio_dir}")
    loader = AudioLoader(
        data_dir=str(audio_dir),
        sample_rate=sample_rate,
        duration=duration,
        augment=False,
    )

    # Step B: Generate spectrograms
    print(f"\n🎨 Generating {target_shape[0]}×{target_shape[1]} spectrograms...")
    spec_gen = SpectrogramGenerator(
        sample_rate=sample_rate,
        target_shape=target_shape,
    )

    spectrograms_dir = Path("data") / "preprocessed" / modality / subset
    spec_gen.generate_and_save_dataset(
        audio_loader=loader,
        output_dir=str(spectrograms_dir),
    )

    # Step C: Prepare (flatten, split, normalize, one-hot)
    print(f"\n🔧 Running data preparation...")
    prep = DataPreparator(
        spectrogram_path=str(spectrograms_dir / "spectrograms.npz"),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed,
    )
    data = prep.prepare()

    # Save prepared data for next time
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    prep.save_prepared_data(str(prepared_path.parent))

    # Save normalization stats separately (used by API for inference)
    norm_path = Path(config.norm_stats_path)
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(norm_path),
        mean=data["mean"],
        std=data["std"],
        class_names=np.array(data["class_names"]),
    )
    print(f"💾 Saved norm stats: {norm_path}")

    return data


# ── Stage 2: Custom NumPy Training ────────────────────────────────────────

def stage_train_custom(args, config, data: dict) -> tuple:
    """Train the custom from-scratch NumPy neural network."""
    print("\n" + "=" * 70)
    print("STAGE 2: CUSTOM NUMPY MODEL TRAINING")
    print("=" * 70)

    input_size = data["input_dim"]
    num_classes = data["num_classes"]
    hidden = args.hidden if args.hidden else list(config.hidden_sizes)

    # Build network
    print(f"\n🧠 Building network: {input_size} → {' → '.join(map(str, hidden))} → {num_classes}")
    net = NeuralNetwork(
        input_size=input_size,
        hidden_sizes=hidden,
        num_classes=num_classes,
        seed=args.seed,
    )
    net.summary()

    # Train
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

    print(f"\n🚀 Starting training...")
    history = trainer.train(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
    )

    # Evaluate
    evaluator = Evaluator(network=net, class_names=data["class_names"])
    results = evaluator.evaluate(data["X_test"], data["y_test"])
    evaluator.print_report()

    # Save model
    model_path = Path(config.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    net.save(str(model_path))

    # Save training history
    history_path = Path(config.training_history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(history_path),
        train_loss=np.array(history["train_loss"]),
        val_loss=np.array(history["val_loss"]),
        train_accuracy=np.array(history["train_accuracy"]),
        val_accuracy=np.array(history["val_accuracy"]),
    )
    print(f"📊 Saved training history: {history_path}")

    return net, history, results


# ── Stage 3: PyTorch Training ─────────────────────────────────────────────

def stage_train_pytorch(args, config, data: dict) -> tuple:
    """Train the PyTorch comparison model."""
    print("\n" + "=" * 70)
    print("STAGE 3: PYTORCH MODEL TRAINING")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    input_size = data["input_dim"]
    num_classes = data["num_classes"]
    hidden = args.hidden if args.hidden else list(config.hidden_sizes)

    # Convert one-hot to class indices
    y_train = np.argmax(data["y_train"], axis=1)
    y_val = np.argmax(data["y_val"], axis=1)
    y_test = np.argmax(data["y_test"], axis=1)

    # DataLoaders
    train_ds = TensorDataset(torch.FloatTensor(data["X_train"]), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(data["X_val"]), torch.LongTensor(y_val))
    test_ds = TensorDataset(torch.FloatTensor(data["X_test"]), torch.LongTensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Build model
    print(f"\n🔥 Building PyTorch model: {input_size} → {' → '.join(map(str, hidden))} → {num_classes}")
    model = AudioClassifier(
        input_size=input_size,
        hidden_sizes=hidden,
        num_classes=num_classes,
    ).to(device)
    model.summary()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    # Training loop
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    model_save_path = config.pytorch_model_path

    print(f"\n{'=' * 70}")
    print(f"Training: {len(train_ds)} samples, Validation: {len(val_ds)} samples")
    print(f"{'=' * 70}")

    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        model.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(X_b)
            train_correct += (logits.argmax(dim=1) == y_b).sum().item()
            train_total += len(X_b)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b)
                loss = criterion(logits, y_b)
                val_loss_sum += loss.item() * len(X_b)
                val_correct += (logits.argmax(dim=1) == y_b).sum().item()
                val_total += len(X_b)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch:>3}/{args.epochs} | "
            f"Train: {train_loss:.4f} / {train_acc*100:>5.1f}% | "
            f"Val: {val_loss:.4f} / {val_acc*100:>5.1f}% | "
            f"LR: {current_lr:.6f} | {epoch_time:.2f}s"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
            model.save(model_save_path)
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\n⏹️  Early stopping at epoch {epoch} (patience {args.patience})")
            break

    total_time = time.time() - total_start

    # Test accuracy with best model
    best_model = AudioClassifier.load(model_save_path).to(device)
    best_model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            test_correct += (best_model.predict(X_b) == y_b).sum().item()
            test_total += len(X_b)
    test_acc = test_correct / test_total

    print(f"\n{'=' * 70}")
    print(f"PyTorch Training Complete — {total_time:.1f}s")
    print(f"  Best val loss:   {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"  Test accuracy:   {test_acc*100:.1f}%")
    print(f"{'=' * 70}")

    # Save history
    history_path = Path(config.pytorch_history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(history_path),
        train_loss=np.array(history["train_loss"]),
        val_loss=np.array(history["val_loss"]),
        train_accuracy=np.array(history["train_accuracy"]),
        val_accuracy=np.array(history["val_accuracy"]),
    )
    print(f"📊 Saved PyTorch training history: {history_path}")

    return best_model, history


# ── Stage 4: t-SNE Generation ─────────────────────────────────────────────

def stage_generate_tsne(config, data: dict, net=None, pt_model=None):
    """Generate t-SNE embeddings for both engines."""
    print("\n" + "=" * 70)
    print("STAGE 4: t-SNE EMBEDDING GENERATION")
    print("=" * 70)

    X_test = data["X_test"]
    y_test = data["y_test"]
    class_names = data["class_names"]
    true_labels = np.argmax(y_test, axis=1)

    def _compute_and_save(features, probs, save_path, engine_name):
        """Run t-SNE and save results."""
        print(f"\n   Computing t-SNE for {engine_name} ({features.shape[1]}-D → 2-D)...")
        tsne = TSNE(
            n_components=2,
            perplexity=min(30.0, len(features) - 1),
            random_state=42,
            max_iter=1000,
            learning_rate="auto",
            init="pca",
        )
        coords = tsne.fit_transform(features)

        predictions = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)

        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(save_p),
            coords=coords,
            labels=true_labels,
            predictions=predictions,
            confidences=confidences,
            class_names=np.array(class_names),
        )
        print(f"   💾 Saved: {save_p}")

    # Custom NumPy model
    if net is not None:
        # Extract penultimate layer features
        x = X_test.copy()
        for layer in net.layers[:-1]:
            x = layer.forward(x)
        custom_features = x

        probs = net.forward(X_test)
        _compute_and_save(custom_features, probs, config.tsne_path, "Custom NumPy")

    # PyTorch model
    if pt_model is not None:
        pt_model.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(X_test)
            # network[:penultimate] = all layers except final Linear
            n_modules = len(list(pt_model.network))
            penultimate_idx = n_modules - 1
            pt_features = pt_model.network[:penultimate_idx](x_t).numpy()
            probs = pt_model.predict_proba(x_t).numpy()

        _compute_and_save(pt_features, probs, config.pytorch_tsne_path, "PyTorch")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Lookup dataset config
    registry = DatasetRegistry()
    try:
        config = registry.get(args.dataset)
    except KeyError as e:
        print(f"❌ {e}")
        sys.exit(1)

    if config.modality != "audio":
        print(f"❌ Dataset '{args.dataset}' is modality='{config.modality}', not 'audio'.")
        print("   Use train_image.py for image datasets.")
        sys.exit(1)

    hidden = args.hidden if args.hidden else list(config.hidden_sizes)
    resolution = args.resolution if args.resolution else config.target_shape[0]

    print("=" * 70)
    print(f"🔊 AUDIO MODEL TRAINING — {config.name}")
    print("=" * 70)
    print(f"\n⚙️  Configuration:")
    print(f"   Dataset:        {config.key} ({config.name})")
    print(f"   Classes:        {config.num_classes} ({', '.join(config.class_names)})")
    print(f"   Sample rate:    {config.sample_rate} Hz")
    print(f"   Duration:       {config.duration}s")
    print(f"   Spectrogram:    {resolution}×{resolution}")
    print(f"   Architecture:   {resolution**2} → {' → '.join(map(str, hidden))} → {config.num_classes}")
    print(f"   Learning rate:  {args.lr} (decay: {args.lr_decay}/epoch)")
    print(f"   Batch size:     {args.batch_size}")
    print(f"   Max epochs:     {args.epochs} (patience: {args.patience})")
    print(f"   Engines:        NumPy{' + PyTorch' if not args.skip_pytorch else ''}")

    pipeline_start = time.time()

    # Stage 1: Data
    data = stage_preprocess(args, config)

    # Stage 2: Custom NumPy
    net, custom_history, custom_results = stage_train_custom(args, config, data)

    # Stage 3: PyTorch
    pt_model = None
    if not args.skip_pytorch:
        pt_model, pt_history = stage_train_pytorch(args, config, data)

    # Stage 4: t-SNE
    if not args.skip_tsne:
        stage_generate_tsne(config, data, net=net, pt_model=pt_model)

    # Final summary
    pipeline_time = time.time() - pipeline_start
    print("\n" + "=" * 70)
    print("✅ AUDIO TRAINING PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n📊 Results:")
    print(f"   Custom NumPy  — Test accuracy: {custom_results['accuracy']*100:.1f}%, "
          f"Macro F1: {custom_results['macro_f1']:.4f}")
    if pt_model:
        print(f"   PyTorch       — see above for test accuracy")
    print(f"\n   Total pipeline time: {pipeline_time:.1f}s")
    print(f"\n📁 Artifacts saved:")
    print(f"   Model (NumPy):    {config.model_path}")
    if not args.skip_pytorch:
        print(f"   Model (PyTorch):  {config.pytorch_model_path}")
    print(f"   Norm stats:       {config.norm_stats_path}")
    print(f"   Prepared data:    {config.prepared_data_path}")
    print(f"   Training history: {config.training_history_path}")
    if not args.skip_tsne:
        print(f"   t-SNE (NumPy):    {config.tsne_path}")
        if not args.skip_pytorch:
            print(f"   t-SNE (PyTorch):  {config.pytorch_tsne_path}")
    print(f"\n🎯 Done! The dataset is now ready to use in the web app.")


if __name__ == "__main__":
    main()
