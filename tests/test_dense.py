"""Tests for the Dense (fully connected) layer."""

import numpy as np
import pytest
from src.neural_network.dense import DenseLayer


class TestDenseLayer:
    """Tests for the DenseLayer class."""

    # ----- Forward pass -----

    def test_forward_output_shape(self):
        """Output shape must be (batch, output_size)."""
        layer = DenseLayer(4096, 128, seed=42)
        x = np.random.randn(32, 4096)
        out = layer.forward(x)

        assert out.shape == (32, 128)

    def test_forward_matches_manual(self):
        """Forward output must equal X @ W + b."""
        layer = DenseLayer(3, 2, seed=0)
        x = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
        out = layer.forward(x)
        expected = x @ layer.weights + layer.biases

        np.testing.assert_allclose(out, expected)

    # ----- Initialization -----

    def test_he_initialization_scale(self):
        """Weight std should approximate sqrt(2 / fan_in)."""
        layer = DenseLayer(4096, 128, seed=42)
        expected_std = np.sqrt(2.0 / 4096)

        assert abs(layer.weights.std() - expected_std) < 0.005

    def test_biases_initialized_to_zero(self):
        """Biases must start at zero."""
        layer = DenseLayer(4096, 128, seed=42)

        assert np.all(layer.biases == 0.0)
        assert layer.biases.shape == (1, 128)

    def test_seed_reproducibility(self):
        """Same seed must produce identical weights."""
        a = DenseLayer(100, 50, seed=123)
        b = DenseLayer(100, 50, seed=123)

        np.testing.assert_array_equal(a.weights, b.weights)
        np.testing.assert_array_equal(a.biases, b.biases)

    # ----- Backward pass -----

    def test_backward_gradient_shapes(self):
        """Backward must produce correctly shaped gradients."""
        layer = DenseLayer(4096, 128, seed=42)
        x = np.random.randn(32, 4096)
        layer.forward(x)

        grad_output = np.random.randn(32, 128)
        grad_input = layer.backward(grad_output)

        assert layer.grad_weights.shape == (4096, 128)
        assert layer.grad_biases.shape == (1, 128)
        assert grad_input.shape == (32, 4096)

    def test_backward_matches_manual(self):
        """Analytical gradients must match manual computation."""
        layer = DenseLayer(3, 2, seed=0)
        x = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
        layer.forward(x)

        grad_out = np.array([[0.1, 0.2],
                              [0.3, 0.4]])
        grad_in = layer.backward(grad_out)

        np.testing.assert_allclose(layer.grad_weights, x.T @ grad_out)
        np.testing.assert_allclose(layer.grad_biases, grad_out.sum(axis=0, keepdims=True))
        np.testing.assert_allclose(grad_in, grad_out @ layer.weights.T)

    def test_numerical_gradient_check(self):
        """Analytical gradients must match finite-difference approximation."""
        np.random.seed(42)
        layer = DenseLayer(5, 3, seed=42)
        x = np.random.randn(4, 5)

        # Analytical
        layer.forward(x)
        layer.backward(np.ones((4, 3)))

        # Numerical (finite differences)
        eps = 1e-5
        numerical = np.zeros_like(layer.weights)
        for i in range(layer.weights.shape[0]):
            for j in range(layer.weights.shape[1]):
                layer.weights[i, j] += eps
                loss_plus = layer.forward(x).sum()
                layer.weights[i, j] -= 2 * eps
                loss_minus = layer.forward(x).sum()
                layer.weights[i, j] += eps  # restore
                numerical[i, j] = (loss_plus - loss_minus) / (2 * eps)

        # Recompute analytical after restoring
        layer.forward(x)
        layer.backward(np.ones((4, 3)))

        np.testing.assert_allclose(layer.grad_weights, numerical, atol=1e-5)

    # ----- Robustness -----

    def test_no_nan_inf(self, rng):
        """Full forward-backward should produce no NaN or Inf."""
        layer = DenseLayer(4096, 128, seed=42)
        x = rng.randn(32, 4096)
        out = layer.forward(x)
        grad = layer.backward(rng.randn(32, 128))

        for arr in [out, grad, layer.grad_weights, layer.grad_biases]:
            assert not np.any(np.isnan(arr))
            assert not np.any(np.isinf(arr))

    def test_chaining_two_layers(self, rng):
        """Two Dense layers should chain forward and backward correctly."""
        l1 = DenseLayer(4096, 128, seed=1)
        l2 = DenseLayer(128, 10, seed=2)

        x = rng.randn(32, 4096)
        h = l1.forward(x)
        out = l2.forward(h)

        grad_h = l2.backward(rng.randn(32, 10))
        grad_x = l1.backward(grad_h)

        assert h.shape == (32, 128)
        assert out.shape == (32, 10)
        assert grad_h.shape == (32, 128)
        assert grad_x.shape == (32, 4096)
