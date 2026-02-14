#!/usr/bin/env python3
"""
Generate t-SNE data for the Feature Explorer page.

Loads the trained model and test data, computes t-SNE embeddings of the
penultimate layer features, and saves coordinates + predictions + confidences
to results/metrics/tsne_data.npz (consumed by the /api/model/tsne endpoint).

Usage:
    python generate_tsne.py
"""

import numpy as np
from src.neural_network.network import NeuralNetwork
from src.visualization.attention import visualize_tsne

MODEL_PATH = "models/model.npz"
DATA_PATH = "data/prepared/prepared_data.npz"


def main():
    print("Loading model...")
    net = NeuralNetwork.load(MODEL_PATH)

    print("Loading test data...")
    data = np.load(DATA_PATH, allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"]
    class_names = data["class_names"].tolist()

    print(f"Test set: {len(X_test)} samples, {len(class_names)} classes")
    print(f"Classes: {class_names}")

    print("\nComputing t-SNE...")
    visualize_tsne(
        network=net,
        X=X_test,
        y_onehot=y_test,
        class_names=class_names,
        perplexity=30.0,
        save_path="results/figures",
    )

    print("\n✅ Done! t-SNE data saved to results/metrics/tsne_data.npz")


if __name__ == "__main__":
    main()
