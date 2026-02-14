"""Tests for the NeuralNetwork class."""

import numpy as np
import pytest
from src.neural_network.network import NeuralNetwork
from src.neural_network.dense import DenseLayer


class TestNeuralNetwork:
    """Tests for the NeuralNetwork class."""

    # ----- Construction -----

    def test_default_architecture(self, default_network):
        """Default network has 3 Dense layers with correct weight shapes."""
        net = default_network
        assert len(net.dense_layers) == 3
        assert net.dense_layers[0].weights.shape == (4096, 128)
        assert net.dense_layers[1].weights.shape == (128, 64)
        assert net.dense_layers[2].weights.shape == (64, 10)

    def test_custom_architecture(self):
        """Custom hidden_sizes should produce the right number of layers."""
        net = NeuralNetwork(input_size=100, hidden_sizes=[256, 128, 64],
                            num_classes=5, seed=0)
        assert len(net.dense_layers) == 4
        assert net.dense_layers[0].weights.shape == (100, 256)
        assert net.dense_layers[3].weights.shape == (64, 5)

    def test_parameter_count(self, default_network):
        """Total parameter count must match manual calculation."""
        expected = (4096 * 128 + 128) + (128 * 64 + 64) + (64 * 10 + 10)
        assert default_network.count_parameters() == expected

    def test_seed_reproducibility(self):
        """Same seed must produce identical networks."""
        a = NeuralNetwork(seed=42)
        b = NeuralNetwork(seed=42)
        for la, lb in zip(a.dense_layers, b.dense_layers):
            np.testing.assert_array_equal(la.weights, lb.weights)

    # ----- Forward pass -----

    def test_forward_output_shape(self, default_network, rng):
        """Forward output shape must be (batch, num_classes)."""
        x = rng.randn(32, 4096)
        probs = default_network.forward(x)
        assert probs.shape == (32, 10)

    def test_forward_valid_probabilities(self, default_network, rng):
        """Forward output must be valid probabilities (positive, sum to 1)."""
        x = rng.randn(32, 4096)
        probs = default_network.forward(x)

        assert probs.min() > 0.0
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-7)

    # ----- Loss -----

    def test_compute_loss_returns_positive_scalar(self, default_network, real_sized_batch):
        """compute_loss should return a positive float."""
        X, y, *_ = real_sized_batch
        probs = default_network.forward(X)
        loss = default_network.compute_loss(probs, y)

        assert isinstance(loss, float)
        assert loss > 0
        assert not np.isnan(loss)

    # ----- Backward -----

    def test_backward_populates_gradients(self, default_network, real_sized_batch):
        """After backward, every Dense layer must have non-None, NaN-free gradients."""
        X, y, *_ = real_sized_batch
        probs = default_network.forward(X)
        default_network.compute_loss(probs, y)
        default_network.backward()

        for layer in default_network.dense_layers:
            assert layer.grad_weights is not None
            assert layer.grad_biases is not None
            assert not np.any(np.isnan(layer.grad_weights))
            assert not np.any(np.isnan(layer.grad_biases))

    # ----- Predict -----

    def test_predict_returns_class_indices(self, default_network, rng):
        """predict() should return integer class indices in [0, num_classes)."""
        x = rng.randn(32, 4096)
        preds = default_network.predict(x)

        assert preds.shape == (32,)
        assert preds.min() >= 0
        assert preds.max() <= 9

    # ----- Training step -----

    def test_single_training_step_decreases_loss(self, small_network, small_batch):
        """A single SGD step should reduce loss on the same batch."""
        X, y, *_ = small_batch
        net = small_network

        probs = net.forward(X)
        loss_before = net.compute_loss(probs, y)
        net.backward()

        lr = 0.01
        for layer in net.get_trainable_layers():
            layer.weights -= lr * layer.grad_weights
            layer.biases -= lr * layer.grad_biases

        probs_after = net.forward(X)
        loss_after = net.compute_loss(probs_after, y)

        assert loss_after < loss_before

    # ----- get_trainable_layers -----

    def test_get_trainable_layers(self, default_network):
        """Should return exactly the Dense layers."""
        trainable = default_network.get_trainable_layers()
        assert len(trainable) == 3
        assert all(isinstance(l, DenseLayer) for l in trainable)

    # ----- Summary -----

    def test_summary_runs(self, default_network, capsys):
        """summary() should print without raising."""
        default_network.summary()
        captured = capsys.readouterr()
        assert "Neural Network Summary" in captured.out

    # ----- No NaN anywhere -----

    def test_full_pipeline_no_nan(self, small_network, small_batch):
        """Full forward→loss→backward→update cycle should be NaN-free."""
        X, y, *_ = small_batch
        net = small_network

        probs = net.forward(X)
        loss = net.compute_loss(probs, y)
        net.backward()

        for layer in net.get_trainable_layers():
            layer.weights -= 0.01 * layer.grad_weights
            layer.biases -= 0.01 * layer.grad_biases

        assert not np.isnan(loss)
        for layer in net.dense_layers:
            assert not np.any(np.isnan(layer.weights))
            assert not np.any(np.isinf(layer.weights))
