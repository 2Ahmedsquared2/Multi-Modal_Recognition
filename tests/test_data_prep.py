"""Tests for the DataPreparator class (using synthetic data — no file dependencies)."""

import numpy as np
import pytest
from src.preprocessing.data_prep import DataPreparator


class TestDataPreparator:
    """Tests for individual DataPreparator methods using synthetic inputs."""

    # ----- Flatten -----

    def test_flatten(self):
        """Flattening (N, 64, 64) should give (N, 4096)."""
        prep = DataPreparator()
        specs = np.random.randn(100, 64, 64)
        flat = prep.flatten(specs)

        assert flat.shape == (100, 4096)

    def test_flatten_preserves_data(self):
        """Flattened values must match original values (just reshaped)."""
        prep = DataPreparator()
        specs = np.random.randn(5, 8, 8)
        flat = prep.flatten(specs)

        np.testing.assert_array_equal(flat[0], specs[0].ravel())

    # ----- One-hot encoding -----

    def test_one_hot_encode_shape(self):
        """One-hot output must be (N, num_classes)."""
        prep = DataPreparator()
        labels = np.array([0, 1, 2, 3, 4])
        one_hot = prep.one_hot_encode(labels, num_classes=5)

        assert one_hot.shape == (5, 5)

    def test_one_hot_encode_values(self):
        """Each row should have exactly one 1.0 at the correct index."""
        prep = DataPreparator()
        labels = np.array([0, 3, 2])
        one_hot = prep.one_hot_encode(labels, num_classes=5)

        np.testing.assert_allclose(one_hot.sum(axis=1), 1.0)
        assert one_hot[0, 0] == 1.0
        assert one_hot[1, 3] == 1.0
        assert one_hot[2, 2] == 1.0

    # ----- Normalization -----

    def test_compute_normalization_shapes(self):
        """Mean and std must have shape (D,)."""
        prep = DataPreparator()
        X = np.random.randn(100, 64)
        mean, std = prep.compute_normalization(X)

        assert mean.shape == (64,)
        assert std.shape == (64,)

    def test_normalize_training_set(self):
        """After normalization, training set mean ≈ 0, std ≈ 1."""
        prep = DataPreparator()
        X = np.random.randn(200, 64) * 5 + 3  # shifted and scaled
        mean, std = prep.compute_normalization(X)
        X_norm = prep.normalize(X, mean, std)

        np.testing.assert_allclose(X_norm.mean(axis=0), 0.0, atol=1e-10)
        # std should be ~1 for features with variance
        nonzero = std > 1e-6
        np.testing.assert_allclose(X_norm[:, nonzero].std(axis=0), 1.0, atol=1e-10)

    def test_normalize_zero_variance_feature(self):
        """Features with zero variance should be zeroed, not blown up."""
        prep = DataPreparator()
        X = np.random.randn(50, 10)
        X[:, 5] = 3.0  # constant feature

        mean, std = prep.compute_normalization(X)
        X_norm = prep.normalize(X, mean, std)

        # The constant column should be all zeros (not enormous numbers)
        assert np.all(X_norm[:, 5] == 0.0)

    # ----- Stratified split -----

    def test_stratified_split_sizes(self):
        """Split sizes should match the requested ratios."""
        prep = DataPreparator(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        X = np.random.randn(100, 16)
        y = np.repeat(np.arange(5), 20)  # 20 per class

        X_tr, y_tr, X_v, y_v, X_te, y_te = prep.stratified_split(X, y)

        total = len(X_tr) + len(X_v) + len(X_te)
        assert total == 100

    def test_stratified_split_preserves_labels(self):
        """All original labels should appear in the union of the splits."""
        prep = DataPreparator(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        X = np.random.randn(100, 16)
        y = np.repeat(np.arange(5), 20)

        X_tr, y_tr, X_v, y_v, X_te, y_te = prep.stratified_split(X, y)

        all_labels = np.concatenate([y_tr, y_v, y_te])
        assert set(all_labels) == set(range(5))

    # ----- Batch generator -----

    def test_batch_generator_covers_all_samples(self):
        """Batch generator should yield all samples exactly once per epoch."""
        prep = DataPreparator()
        X = np.random.randn(100, 16)
        y = np.random.randn(100, 5)

        total = 0
        for Xb, yb in prep.batch_generator(X, y, batch_size=32, shuffle=False):
            total += len(Xb)

        assert total == 100

    def test_batch_generator_last_batch_smaller(self):
        """If N not divisible by batch_size, last batch should be smaller."""
        prep = DataPreparator()
        X = np.random.randn(100, 16)
        y = np.random.randn(100, 5)

        sizes = [len(Xb) for Xb, _ in prep.batch_generator(X, y, batch_size=32, shuffle=False)]

        assert sizes[-1] == 100 - 32 * 3  # 100 mod 32 = 4
        assert all(s == 32 for s in sizes[:-1])
