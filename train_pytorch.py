#!/usr/bin/env python3
"""
Train the PyTorch model on the same prepared dataset used by the custom NN.

Uses identical hyperparameters for a fair comparison:
  - SGD optimizer, lr=0.01
  - LR decay: ×0.95 per epoch
  - Batch size: 32
  - Early stopping with patience

Saves:
  - models/pytorch_model.pt              (model weights)
  - results/metrics/pytorch_training_history.npz  (loss/accuracy curves)

Usage:
    python train_pytorch.py
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from src.pytorch_model.model import AudioClassifier

# ── Hyperparameters (match custom NN training) ──────────────────────────

DATA_PATH = "data/prepared/prepared_data.npz"
MODEL_SAVE_PATH = "models/pytorch_model.pt"
HISTORY_SAVE_PATH = "results/metrics/pytorch_training_history.npz"

LEARNING_RATE = 0.01
LR_DECAY = 0.95
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 10
SEED = 42


def main():
    # ── Seed everything ──────────────────────────────────────────────
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load prepared data ───────────────────────────────────────────
    print("Loading data...")
    data = np.load(DATA_PATH, allow_pickle=True)

    X_train = data["X_train"]
    y_train_onehot = data["y_train"]
    X_val = data["X_val"]
    y_val_onehot = data["y_val"]
    X_test = data["X_test"]
    y_test_onehot = data["y_test"]
    class_names = data["class_names"].tolist()

    # Convert one-hot to class indices (PyTorch CrossEntropyLoss expects this)
    y_train = np.argmax(y_train_onehot, axis=1)
    y_val = np.argmax(y_val_onehot, axis=1)
    y_test = np.argmax(y_test_onehot, axis=1)

    print(f"Train: {len(X_train)} samples")
    print(f"Val:   {len(X_val)} samples")
    print(f"Test:  {len(X_test)} samples")
    print(f"Classes ({len(class_names)}): {class_names}")

    # ── Create DataLoaders ───────────────────────────────────────────
    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val),
    )
    test_ds = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ── Build model ──────────────────────────────────────────────────
    input_size = X_train.shape[1]
    num_classes = len(class_names)

    model = AudioClassifier(
        input_size=input_size,
        hidden_sizes=[128, 64],
        num_classes=num_classes,
    ).to(device)

    model.summary()

    # ── Optimizer & scheduler (matching custom NN) ───────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)

    # ── Training loop ────────────────────────────────────────────────
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    print("\n" + "=" * 80)
    print(f"Training Started — {EPOCHS} epochs, batch_size={BATCH_SIZE}, "
          f"lr={LEARNING_RATE}, decay={LR_DECAY}")
    print(f"Training: {len(X_train)} samples ({len(train_loader)} batches/epoch) | "
          f"Validation: {len(X_val)} samples")
    print(f"Early stopping: patience={PATIENCE}")
    print("=" * 80)

    total_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # ── Train phase ──
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(X_batch)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += len(X_batch)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # ── Validation phase ──
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                val_loss_sum += loss.item() * len(X_batch)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += len(X_batch)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        # ── LR decay ──
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # ── Record ──
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch:>3}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:>5.1f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:>5.1f}% | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )

        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            model.save(MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping at epoch {epoch} — "
                  f"val loss hasn't improved for {PATIENCE} epochs")
            print(f"   Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
            break

    total_time = time.time() - total_start
    final_epoch = len(history["train_loss"])

    print(f"\n{'=' * 80}")
    print("Training Complete!")
    print(f"{'=' * 80}")
    print(f"  Total time:        {total_time:.1f}s ({total_time / final_epoch:.2f}s/epoch)")
    print(f"  Epochs trained:    {final_epoch}")
    print(f"  Best val loss:     {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"  Final train acc:   {history['train_accuracy'][-1] * 100:.1f}%")
    print(f"  Final val acc:     {history['val_accuracy'][-1] * 100:.1f}%")

    # ── Reload best model for test evaluation ────────────────────────
    print("\nLoading best model for test evaluation...")
    best_model = AudioClassifier.load(MODEL_SAVE_PATH).to(device)
    best_model.eval()

    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = best_model.predict(X_batch)
            test_correct += (preds == y_batch).sum().item()
            test_total += len(X_batch)

    test_acc = test_correct / test_total
    print(f"  Test accuracy:     {test_acc * 100:.1f}%")
    print(f"{'=' * 80}")

    # ── Save training history ────────────────────────────────────────
    Path(HISTORY_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        HISTORY_SAVE_PATH,
        train_loss=np.array(history["train_loss"]),
        val_loss=np.array(history["val_loss"]),
        train_accuracy=np.array(history["train_accuracy"]),
        val_accuracy=np.array(history["val_accuracy"]),
    )
    print(f"\n📊 Training history saved to: {HISTORY_SAVE_PATH}")
    print("✅ Done!")


if __name__ == "__main__":
    main()
