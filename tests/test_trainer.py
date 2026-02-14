"""Tests for the Trainer class."""

import numpy as np
import pytest
from src.neural_network.network import NeuralNetwork
from src.training.trainer import Trainer


class TestTrainer:
    """Tests for the Trainer training loop."""

    @pytest.fixture
    def trained(self, synthetic_dataset):
        """Train a small network for 5 epochs and return (trainer, history)."""
        ds = synthetic_dataset
        net = NeuralNetwork(input_size=ds['input_size'],
                            hidden_sizes=[32, 16],
                            num_classes=ds['num_classes'], seed=42)
        trainer = Trainer(net, learning_rate=0.1, lr_decay=0.95,
                          batch_size=32, epochs=5, seed=42)
        history = trainer.train(ds['X_train'], ds['y_train'],
                                ds['X_val'], ds['y_val'])
        return trainer, history

    # ----- Initialization -----

    def test_init_params(self, small_network):
        """Trainer should store all hyperparameters correctly."""
        t = Trainer(small_network, learning_rate=0.01, lr_decay=0.9,
                    batch_size=64, epochs=50, patience=5)

        assert t.learning_rate == 0.01
        assert t.lr_decay == 0.9
        assert t.batch_size == 64
        assert t.epochs == 50
        assert t.patience == 5

    # ----- History structure -----

    def test_history_keys(self, trained):
        """History must contain all 6 expected keys."""
        _, history = trained
        expected = {'train_loss', 'train_accuracy', 'val_loss',
                    'val_accuracy', 'learning_rate', 'epoch_time'}
        assert set(history.keys()) == expected

    def test_history_lengths(self, trained):
        """Each history list should have exactly epochs entries."""
        _, history = trained
        for values in history.values():
            assert len(values) == 5

    # ----- Learning dynamics -----

    def test_training_loss_decreases(self, trained):
        """Training loss should decrease from first to last epoch."""
        _, history = trained
        assert history['train_loss'][-1] < history['train_loss'][0]

    def test_training_accuracy_increases(self, trained):
        """Training accuracy should increase from first to last epoch."""
        _, history = trained
        assert history['train_accuracy'][-1] > history['train_accuracy'][0]

    def test_learning_rate_decays(self, trained):
        """Learning rate should decrease monotonically."""
        _, history = trained
        lrs = history['learning_rate']
        assert all(lrs[i] > lrs[i + 1] for i in range(len(lrs) - 1))

    # ----- Validation -----

    def test_validation_metrics_valid(self, trained):
        """Validation loss > 0 and accuracy in [0, 1]."""
        _, history = trained
        assert all(v > 0 for v in history['val_loss'])
        assert all(0.0 <= a <= 1.0 for a in history['val_accuracy'])

    # ----- Early stopping -----

    def test_early_stopping_triggers(self, synthetic_dataset):
        """With tight patience, training should stop before max epochs."""
        ds = synthetic_dataset
        net = NeuralNetwork(input_size=ds['input_size'],
                            hidden_sizes=[32, 16],
                            num_classes=ds['num_classes'], seed=42)
        trainer = Trainer(net, learning_rate=0.1, lr_decay=0.95,
                          batch_size=32, epochs=100, patience=3, seed=42)
        history = trainer.train(ds['X_train'], ds['y_train'],
                                ds['X_val'], ds['y_val'])

        assert len(history['train_loss']) < 100

    # ----- No NaN -----

    def test_no_nan_in_history(self, trained):
        """No NaN should appear in any history values."""
        _, history = trained
        for values in history.values():
            assert not any(np.isnan(v) for v in values)

    # ----- Evaluate helper -----

    def test_evaluate_returns_valid(self, trained, synthetic_dataset):
        """_evaluate() should return valid (loss, accuracy) tuple."""
        trainer, _ = trained
        ds = synthetic_dataset
        loss, acc = trainer._evaluate(ds['X_val'], ds['y_val'])

        assert isinstance(loss, float)
        assert not np.isnan(loss)
        assert 0.0 <= acc <= 1.0
