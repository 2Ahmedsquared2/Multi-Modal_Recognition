"""Tests for the Evaluator class."""

import numpy as np
import pytest
from src.neural_network.network import NeuralNetwork
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator


class TestEvaluator:
    """Tests for the Evaluator evaluation suite."""

    @pytest.fixture
    def eval_results(self, synthetic_dataset):
        """Train a small network, evaluate on test set, return (evaluator, results)."""
        ds = synthetic_dataset
        net = NeuralNetwork(input_size=ds['input_size'],
                            hidden_sizes=[32, 16],
                            num_classes=ds['num_classes'], seed=42)
        trainer = Trainer(net, learning_rate=0.1, lr_decay=0.98,
                          batch_size=32, epochs=15, seed=42)
        trainer.train(ds['X_train'], ds['y_train'],
                      ds['X_val'], ds['y_val'])

        names = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']
        evaluator = Evaluator(net, class_names=names)
        results = evaluator.evaluate(ds['X_test'], ds['y_test'])
        return evaluator, results

    # ----- Results dict -----

    def test_results_keys(self, eval_results):
        """Results dict should contain all expected keys."""
        _, results = eval_results
        expected = {'loss', 'accuracy', 'confusion_matrix', 'per_class',
                    'macro_f1', 'top_confused', 'predictions', 'true_labels',
                    'probabilities', 'num_samples'}
        assert set(results.keys()) == expected

    # ----- Accuracy -----

    def test_accuracy_valid_range(self, eval_results):
        """Accuracy should be in [0, 1]."""
        _, results = eval_results
        assert 0.0 <= results['accuracy'] <= 1.0

    # ----- Confusion matrix -----

    def test_confusion_matrix_shape(self, eval_results):
        """Confusion matrix should be (num_classes, num_classes)."""
        _, results = eval_results
        assert results['confusion_matrix'].shape == (5, 5)

    def test_confusion_matrix_sum(self, eval_results):
        """Confusion matrix entries should sum to number of test samples."""
        _, results = eval_results
        assert results['confusion_matrix'].sum() == results['num_samples']

    def test_confusion_matrix_diagonal_matches_accuracy(self, eval_results):
        """Diagonal of CM should equal accuracy * num_samples."""
        _, results = eval_results
        correct = np.trace(results['confusion_matrix'])
        expected = int(results['accuracy'] * results['num_samples'])
        assert correct == expected

    # ----- Per-class metrics -----

    def test_per_class_structure(self, eval_results):
        """Each class should have precision, recall, F1, support."""
        _, results = eval_results
        for m in results['per_class']:
            assert 'precision' in m
            assert 'recall' in m
            assert 'f1' in m
            assert 'support' in m

    def test_per_class_ranges(self, eval_results):
        """Precision, recall, F1 should all be in [0, 1]."""
        _, results = eval_results
        for m in results['per_class']:
            assert 0.0 <= m['precision'] <= 1.0
            assert 0.0 <= m['recall'] <= 1.0
            assert 0.0 <= m['f1'] <= 1.0

    def test_support_sums_to_total(self, eval_results):
        """Sum of per-class support should equal total test samples."""
        _, results = eval_results
        total_support = sum(m['support'] for m in results['per_class'])
        assert total_support == results['num_samples']

    # ----- Macro F1 -----

    def test_macro_f1_is_average(self, eval_results):
        """Macro F1 should be the unweighted mean of per-class F1 scores."""
        _, results = eval_results
        manual = np.mean([m['f1'] for m in results['per_class']])
        np.testing.assert_allclose(results['macro_f1'], manual)

    # ----- Printing -----

    def test_print_report_runs(self, eval_results, capsys):
        """print_report() should run without error."""
        evaluator, _ = eval_results
        evaluator.print_report()
        captured = capsys.readouterr()
        assert "Classification Report" in captured.out

    def test_print_confusion_matrix_runs(self, eval_results, capsys):
        """print_confusion_matrix() should run without error."""
        evaluator, _ = eval_results
        evaluator.print_confusion_matrix()
        captured = capsys.readouterr()
        assert "Confusion Matrix" in captured.out

    # ----- Top confused -----

    def test_top_confused_pairs(self, eval_results):
        """top_confused should be a list of (true, pred, count) tuples."""
        _, results = eval_results
        tc = results['top_confused']
        assert isinstance(tc, list)
        assert len(tc) <= 5
        for item in tc:
            assert len(item) == 3
            assert isinstance(item[2], int)
