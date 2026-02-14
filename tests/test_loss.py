"""Tests for the Cross-Entropy Loss function."""

import numpy as np
import pytest
from src.neural_network.activations import Softmax
from src.neural_network.loss import CrossEntropyLoss


class TestCrossEntropyLoss:
    """Tests for the CrossEntropyLoss class."""

    def _make(self):
        """Helper: create a fresh Softmax + Loss pair."""
        softmax = Softmax()
        loss_fn = CrossEntropyLoss(softmax)
        return softmax, loss_fn

    # ----- Forward -----

    def test_perfect_prediction_loss_near_zero(self):
        """Confident correct predictions should give loss close to 0."""
        softmax, loss_fn = self._make()
        logits = np.array([[10.0, 0.0, 0.0],
                            [0.0, 10.0, 0.0],
                            [0.0, 0.0, 10.0]])
        y_true = np.eye(3)

        probs = softmax.forward(logits)
        loss = loss_fn.forward(probs, y_true)

        assert loss < 0.01

    def test_wrong_prediction_loss_high(self):
        """Confident wrong predictions should give high loss."""
        softmax, loss_fn = self._make()
        logits = np.array([[0.0, 0.0, 10.0],
                            [10.0, 0.0, 0.0],
                            [0.0, 10.0, 0.0]])
        y_true = np.eye(3)

        probs = softmax.forward(logits)
        loss = loss_fn.forward(probs, y_true)

        assert loss > 5.0

    def test_loss_is_non_negative(self, rng):
        """Cross-entropy loss must always be >= 0."""
        softmax, loss_fn = self._make()
        logits = rng.randn(32, 10)
        y_true = np.zeros((32, 10))
        for i in range(32):
            y_true[i, rng.randint(10)] = 1.0

        probs = softmax.forward(logits)
        loss = loss_fn.forward(probs, y_true)

        assert loss >= 0.0

    def test_uniform_prediction_equals_log_classes(self):
        """Uniform predictions should give loss = log(num_classes)."""
        softmax, loss_fn = self._make()
        logits = np.array([[1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0]])
        y_true = np.array([[1, 0, 0],
                            [0, 1, 0]], dtype=np.float64)

        probs = softmax.forward(logits)
        loss = loss_fn.forward(probs, y_true)

        np.testing.assert_allclose(loss, np.log(3), atol=0.01)

    # ----- Backward -----

    def test_backward_gradient_shape(self, rng):
        """Backward gradient must have shape (batch_size, num_classes)."""
        softmax, loss_fn = self._make()
        logits = rng.randn(32, 10)
        y_true = np.zeros((32, 10))
        for i in range(32):
            y_true[i, rng.randint(10)] = 1.0

        probs = softmax.forward(logits)
        loss_fn.forward(probs, y_true)
        grad = loss_fn.backward()

        assert grad.shape == (32, 10)

    def test_backward_equals_pred_minus_true(self):
        """Gradient must equal (softmax_output - y_true) / batch_size."""
        softmax, loss_fn = self._make()
        logits = np.array([[2.0, 1.0, 0.1],
                            [0.5, 2.5, 1.0]])
        y_true = np.array([[1, 0, 0],
                            [0, 1, 0]], dtype=np.float64)

        probs = softmax.forward(logits)
        loss_fn.forward(probs, y_true)
        grad = loss_fn.backward()

        expected = (probs - y_true) / 2.0
        np.testing.assert_allclose(grad, expected)

    def test_gradient_direction(self):
        """True class gradient < 0 (push up), other classes > 0 (push down)."""
        softmax, loss_fn = self._make()
        logits = np.array([[2.0, 1.0, 0.1]])
        y_true = np.array([[1, 0, 0]], dtype=np.float64)

        probs = softmax.forward(logits)
        loss_fn.forward(probs, y_true)
        grad = loss_fn.backward()

        assert grad[0, 0] < 0   # true class — push probability up
        assert grad[0, 1] > 0   # other class — push probability down
        assert grad[0, 2] > 0

    def test_gradient_rows_sum_near_zero(self, rng):
        """Each gradient row should sum to ~0 (pred sums to 1, true sums to 1)."""
        softmax, loss_fn = self._make()
        logits = rng.randn(32, 10)
        y_true = np.zeros((32, 10))
        for i in range(32):
            y_true[i, rng.randint(10)] = 1.0

        probs = softmax.forward(logits)
        loss_fn.forward(probs, y_true)
        grad = loss_fn.backward()

        np.testing.assert_allclose(grad.sum(axis=1), 0.0, atol=1e-10)

    # ----- Stability -----

    def test_numerical_stability_extreme_logits(self):
        """No NaN/Inf with extreme logit values (±1000)."""
        softmax, loss_fn = self._make()
        logits = np.array([[1000.0, -1000.0, -1000.0],
                            [-1000.0, -1000.0, 1000.0]])
        y_true = np.array([[1, 0, 0],
                            [0, 0, 1]], dtype=np.float64)

        probs = softmax.forward(logits)
        loss = loss_fn.forward(probs, y_true)
        grad = loss_fn.backward()

        assert not np.isnan(loss)
        assert not np.any(np.isnan(grad))
