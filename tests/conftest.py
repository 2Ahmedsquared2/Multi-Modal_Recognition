"""
Shared pytest fixtures for the Acoustic Pattern Recognition Engine test suite.

All fixtures use synthetic data so tests run without the real dataset.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Seeded random number generator for reproducibility."""
    return np.random.RandomState(42)


@pytest.fixture
def small_batch(rng):
    """Small batch of synthetic input data (32 samples, 64 features, 5 classes)."""
    n, d, c = 32, 64, 5
    X = rng.randn(n, d)
    y = np.zeros((n, c))
    for i in range(n):
        y[i, rng.randint(c)] = 1.0
    return X, y, n, d, c


@pytest.fixture
def real_sized_batch(rng):
    """Realistic batch matching the actual network dimensions (32, 4096, 10)."""
    n, d, c = 32, 4096, 10
    X = rng.randn(n, d)
    y = np.zeros((n, c))
    for i in range(n):
        y[i, rng.randint(c)] = 1.0
    return X, y, n, d, c


@pytest.fixture
def synthetic_dataset(rng):
    """
    Larger synthetic dataset for trainer/evaluator tests.
    200 train, 50 val, 50 test  |  64 features, 5 classes
    A small class-dependent signal is injected so the network can learn.
    """
    d, c = 64, 5
    def _make(n, seed_offset=0):
        r = np.random.RandomState(42 + seed_offset)
        X = r.randn(n, d)
        y_idx = r.randint(c, size=n)
        # inject learnable signal in first 10 features
        for i in range(n):
            X[i, :10] += y_idx[i] * 0.5
        y = np.zeros((n, c))
        for i in range(n):
            y[i, y_idx[i]] = 1.0
        return X, y

    X_train, y_train = _make(200, seed_offset=0)
    X_val, y_val = _make(50, seed_offset=1)
    X_test, y_test = _make(50, seed_offset=2)

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'input_size': d, 'num_classes': c,
    }


# ---------------------------------------------------------------------------
# Network fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_network():
    """Small NeuralNetwork for fast testing (64 → 32 → 16 → 5)."""
    from src.neural_network.network import NeuralNetwork
    return NeuralNetwork(input_size=64, hidden_sizes=[32, 16],
                         num_classes=5, seed=42)


@pytest.fixture
def default_network():
    """Default-sized NeuralNetwork (4096 → 128 → 64 → 10)."""
    from src.neural_network.network import NeuralNetwork
    return NeuralNetwork(input_size=4096, hidden_sizes=[128, 64],
                         num_classes=10, seed=42)
