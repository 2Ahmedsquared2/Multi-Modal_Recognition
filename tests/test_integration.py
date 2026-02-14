"""
End-to-end integration test.

Builds a network, trains it on synthetic data, evaluates,
and verifies the entire pipeline hangs together.
"""

import numpy as np
import pytest
from src.neural_network.network import NeuralNetwork
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator


class TestEndToEnd:
    """Integration test that exercises the full pipeline on synthetic data."""

    def test_full_pipeline(self, synthetic_dataset):
        """
        Train → evaluate → verify accuracy above random chance.

        This is the single most important test: if this passes,
        all components (dense, activations, loss, trainer, evaluator)
        are wired correctly.
        """
        ds = synthetic_dataset
        num_classes = ds['num_classes']

        # 1. Build network
        net = NeuralNetwork(
            input_size=ds['input_size'],
            hidden_sizes=[32, 16],
            num_classes=num_classes,
            seed=42,
        )

        # 2. Train
        trainer = Trainer(
            net,
            learning_rate=0.1,
            lr_decay=0.98,
            batch_size=32,
            epochs=30,
            seed=42,
        )
        history = trainer.train(
            ds['X_train'], ds['y_train'],
            ds['X_val'], ds['y_val'],
        )

        # 3. Evaluate
        evaluator = Evaluator(net, class_names=[f'cls_{i}' for i in range(num_classes)])
        results = evaluator.evaluate(ds['X_test'], ds['y_test'])

        # 4. Assertions
        random_chance = 1.0 / num_classes  # 0.2 for 5 classes

        # Training loss should have decreased
        assert history['train_loss'][-1] < history['train_loss'][0]

        # Test accuracy should beat random chance
        assert results['accuracy'] > random_chance, (
            f"accuracy {results['accuracy']:.2f} not above "
            f"random chance {random_chance:.2f}"
        )

        # Confusion matrix should sum to number of test samples
        assert results['confusion_matrix'].sum() == results['num_samples']

        # No NaN anywhere
        assert not np.isnan(results['loss'])
        assert not np.any(np.isnan(results['probabilities']))

    def test_save_and_load_model(self, synthetic_dataset, tmp_path):
        """Save a trained model, load it back, and verify predictions match."""
        ds = synthetic_dataset

        # Train
        net = NeuralNetwork(input_size=ds['input_size'],
                            hidden_sizes=[32, 16],
                            num_classes=ds['num_classes'], seed=42)
        trainer = Trainer(net, learning_rate=0.1, lr_decay=0.95,
                          batch_size=32, epochs=5, seed=42)
        trainer.train(ds['X_train'], ds['y_train'],
                      ds['X_val'], ds['y_val'])

        # Predict before save
        preds_before = net.predict(ds['X_test'])

        # Save
        save_path = net.save(str(tmp_path))

        # Load into a new network
        net2 = NeuralNetwork.load(save_path)
        preds_after = net2.predict(ds['X_test'])

        # Predictions must be identical
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_different_architectures_all_work(self, synthetic_dataset):
        """Multiple architectures should all train without errors."""
        ds = synthetic_dataset
        architectures = [
            [32],           # single hidden layer
            [32, 16],       # two hidden layers (default)
            [64, 32, 16],   # three hidden layers
        ]

        for hidden in architectures:
            net = NeuralNetwork(input_size=ds['input_size'],
                                hidden_sizes=hidden,
                                num_classes=ds['num_classes'], seed=42)
            trainer = Trainer(net, learning_rate=0.1, lr_decay=0.95,
                              batch_size=32, epochs=3, seed=42)
            history = trainer.train(ds['X_train'], ds['y_train'],
                                    ds['X_val'], ds['y_val'])

            # Should complete without NaN
            assert not any(np.isnan(v) for v in history['train_loss']), \
                f"NaN in training with hidden={hidden}"
