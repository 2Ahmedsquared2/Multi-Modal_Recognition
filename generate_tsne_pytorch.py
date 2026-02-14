#!/usr/bin/env python3
"""
Generate t-SNE data for the PyTorch model's Feature Explorer.

Loads the trained PyTorch model and test data, extracts penultimate-layer
features, computes t-SNE embeddings, and saves everything to
results/metrics/pytorch_tsne_data.npz (consumed by /api/model/tsne?model=pytorch).

Usage:
    python generate_tsne_pytorch.py
"""

import numpy as np
import torch
from sklearn.manifold import TSNE
from pathlib import Path

from src.pytorch_model.model import AudioClassifier

MODEL_PATH = "models/pytorch_model.pt"
DATA_PATH = "data/prepared/prepared_data.npz"
SAVE_PATH = "results/metrics/pytorch_tsne_data.npz"


def extract_penultimate_features(model: AudioClassifier, X: np.ndarray) -> np.ndarray:
    """
    Extract features from the penultimate layer (after second ReLU, before final Linear).

    Architecture: Linear→ReLU→Linear→ReLU→Linear
    Penultimate = output of network[:4] (first 4 modules: Lin, ReLU, Lin, ReLU)
    """
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(X)
        # network[:4] = Linear(4096,128) → ReLU → Linear(128,64) → ReLU
        features = model.network[:4](x)
        return features.numpy()


def main():
    print("Loading PyTorch model...")
    model = AudioClassifier.load(MODEL_PATH)

    print("Loading test data...")
    data = np.load(DATA_PATH, allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"]
    class_names = data["class_names"].tolist()

    true_labels = np.argmax(y_test, axis=1)

    print(f"Test set: {len(X_test)} samples, {len(class_names)} classes")
    print(f"Classes: {class_names}")

    # ── Get predictions + confidences ──────────────────────────────────
    print("Running inference...")
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(torch.FloatTensor(X_test)).numpy()

    predictions = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    accuracy = np.mean(predictions == true_labels)
    print(f"Test accuracy: {accuracy:.2%}")

    # ── Extract penultimate features ───────────────────────────────────
    print("Extracting penultimate-layer features...")
    features = extract_penultimate_features(model, X_test)
    print(f"Feature shape: {features.shape}")

    # ── Compute t-SNE ──────────────────────────────────────────────────
    print("Computing t-SNE (perplexity=30)...")
    tsne = TSNE(
        n_components=2,
        perplexity=30.0,
        random_state=42,
        max_iter=1000,
        learning_rate="auto",
        init="pca",
    )
    coords = tsne.fit_transform(features)
    print(f"t-SNE shape: {coords.shape}")

    # ── Save ───────────────────────────────────────────────────────────
    Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SAVE_PATH,
        coords=coords,
        labels=true_labels,
        predictions=predictions,
        confidences=confidences,
        class_names=np.array(class_names),
    )

    print(f"\n✅ Done! PyTorch t-SNE data saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
