"""Tests for ReLU and Softmax activation functions."""

import numpy as np
import pytest
from src.neural_network.activations import ReLU, Softmax


# =====================================================================
# ReLU
# =====================================================================

class TestReLU:
    """Tests for the ReLU activation function."""

    def test_forward_zeros_negatives(self):
        """Negative inputs become 0; positives pass through unchanged."""
        relu = ReLU()
        x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0],
                       [3.0, -0.5, 0.1, -3.0, 0.0]])
        out = relu.forward(x)

        expected = np.array([[0.0, 0.0, 0.0, 1.0, 2.0],
                              [3.0, 0.0, 0.1, 0.0, 0.0]])
        np.testing.assert_array_equal(out, expected)

    def test_forward_preserves_shape(self, rng):
        """Output shape must equal input shape for batch data."""
        relu = ReLU()
        x = rng.randn(32, 4096)
        out = relu.forward(x)

        assert out.shape == (32, 4096)
        assert out.min() >= 0.0

    def test_backward_gradient_mask(self):
        """Gradient passes where input > 0, blocked where <= 0."""
        relu = ReLU()
        x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0],
                       [3.0, -0.5, 0.1, -3.0, 0.0]])
        relu.forward(x)

        grad_output = np.ones_like(x)
        grad_input = relu.backward(grad_output)

        expected = np.array([[0.0, 0.0, 0.0, 1.0, 1.0],
                              [1.0, 0.0, 1.0, 0.0, 0.0]])
        np.testing.assert_array_equal(grad_input, expected)

    def test_backward_scales_gradient(self, rng):
        """Backward should element-wise multiply by the mask, not just return the mask."""
        relu = ReLU()
        x = np.array([[1.0, -1.0], [0.5, -0.5]])
        relu.forward(x)

        grad_output = np.array([[3.0, 7.0], [2.0, 5.0]])
        grad_input = relu.backward(grad_output)

        expected = np.array([[3.0, 0.0], [2.0, 0.0]])
        np.testing.assert_array_equal(grad_input, expected)


# =====================================================================
# Softmax
# =====================================================================

class TestSoftmax:
    """Tests for the Softmax activation function."""

    def test_forward_probabilities_sum_to_one(self):
        """Each row of softmax output should sum to 1.0."""
        softmax = Softmax()
        logits = np.array([[2.0, 1.0, 0.1],
                            [1.0, 2.0, 3.0]])
        probs = softmax.forward(logits)

        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-7)

    def test_forward_all_positive(self):
        """All softmax outputs must be strictly positive."""
        softmax = Softmax()
        logits = np.array([[2.0, 1.0, 0.1],
                            [1.0, 2.0, 3.0]])
        probs = softmax.forward(logits)

        assert probs.min() > 0.0

    def test_forward_preserves_ordering(self):
        """Highest logit should produce highest probability."""
        softmax = Softmax()
        logits = np.array([[2.0, 1.0, 0.1],
                            [1.0, 2.0, 3.0]])
        probs = softmax.forward(logits)

        assert np.argmax(probs[0]) == np.argmax(logits[0])
        assert np.argmax(probs[1]) == np.argmax(logits[1])

    def test_numerical_stability_extreme_values(self):
        """No NaN/Inf with very large or very small logits."""
        softmax = Softmax()
        logits = np.array([[1000.0, 1001.0, 999.0],
                            [-1000.0, -999.0, -1001.0]])
        probs = softmax.forward(logits)

        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-7)

    def test_batch_shape(self, rng):
        """Batch softmax should return correct shape with valid probabilities."""
        softmax = Softmax()
        logits = rng.randn(32, 10)
        probs = softmax.forward(logits)

        assert probs.shape == (32, 10)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-7)
        assert probs.min() > 0.0

    def test_caches_output(self):
        """Softmax must cache its output for use in the loss backward pass."""
        softmax = Softmax()
        logits = np.array([[1.0, 2.0, 3.0]])
        probs = softmax.forward(logits)

        assert softmax.output is not None
        np.testing.assert_array_equal(softmax.output, probs)
