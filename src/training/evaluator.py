"""
Evaluation Module
Implements test set evaluation, confusion matrix, per-class metrics
(precision, recall, F1-score), and a formatted classification report.

This is the "report card" for the trained neural network.
"""

import numpy as np


class Evaluator:
    """
    Evaluates a trained NeuralNetwork on a dataset
    
    Features:
    - Overall accuracy
    - Confusion matrix (true vs predicted)
    - Per-class: precision, recall, F1-score, accuracy, support
    - Macro-averaged F1-score
    - Formatted classification report (print_report)
    - Results dictionary for visualization (Steps 11-12)
    """
    
    def __init__(self, network, class_names: list = None):
        """
        Args:
            network:     Trained NeuralNetwork instance
            class_names: List of class name strings (e.g., ['bass', 'brass', ...])
                         If None, uses integer indices
        """
        self.network = network
        self.class_names = class_names
        self.num_classes = network.num_classes
        
        # Results — populated by evaluate()
        self.results = None
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Run full evaluation on a dataset
        
        Args:
            X: Feature array, shape (N, D)
            y: One-hot encoded true labels, shape (N, C)
            
        Returns:
            Results dictionary with all metrics
        """
        # Forward pass (no backward, no weight updates)
        probs = self.network.forward(X)
        loss = self.network.compute_loss(probs, y)
        
        # Convert to class indices
        pred_classes = np.argmax(probs, axis=1)
        true_classes = np.argmax(y, axis=1)
        
        # Overall accuracy
        accuracy = np.mean(pred_classes == true_classes)
        
        # Confusion matrix
        confusion = self._confusion_matrix(true_classes, pred_classes)
        
        # Per-class metrics
        per_class = self._per_class_metrics(confusion)
        
        # Macro-averaged F1
        macro_f1 = np.mean([m['f1'] for m in per_class])
        
        # Top confused pairs (most common mistakes)
        top_confused = self._top_confused_pairs(confusion, top_k=5)
        
        # Package results
        self.results = {
            'loss': loss,
            'accuracy': accuracy,
            'confusion_matrix': confusion,
            'per_class': per_class,
            'macro_f1': macro_f1,
            'top_confused': top_confused,
            'predictions': pred_classes,
            'true_labels': true_classes,
            'probabilities': probs,
            'num_samples': len(X),
        }
        
        return self.results
    
    def _confusion_matrix(self, true: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """
        Build confusion matrix where C[i][j] = count of true class i predicted as class j
        
        Args:
            true: True class indices, shape (N,)
            pred: Predicted class indices, shape (N,)
            
        Returns:
            Confusion matrix of shape (num_classes, num_classes)
        """
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for t, p in zip(true, pred):
            cm[t, p] += 1
        return cm
    
    def _per_class_metrics(self, confusion: np.ndarray) -> list:
        """
        Compute precision, recall, F1-score, accuracy, and support per class
        
        Args:
            confusion: Confusion matrix of shape (C, C)
            
        Returns:
            List of dicts, one per class
        """
        metrics = []
        
        for c in range(self.num_classes):
            tp = confusion[c, c]                       # True positives
            fp = confusion[:, c].sum() - tp            # False positives (column sum - TP)
            fn = confusion[c, :].sum() - tp            # False negatives (row sum - TP)
            support = confusion[c, :].sum()            # Total samples in this class
            
            # Precision: TP / (TP + FP) — "when you predict this class, how often are you right?"
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            # Recall: TP / (TP + FN) — "of all actual samples of this class, how many did you find?"
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # F1: harmonic mean of precision and recall
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Per-class accuracy: correct / total for this class
            class_accuracy = tp / support if support > 0 else 0.0
            
            class_name = self.class_names[c] if self.class_names else str(c)
            
            metrics.append({
                'class_name': class_name,
                'class_idx': c,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': class_accuracy,
                'support': int(support),
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
            })
        
        return metrics
    
    def _top_confused_pairs(self, confusion: np.ndarray, top_k: int = 5) -> list:
        """
        Find the most common misclassification pairs
        
        Args:
            confusion: Confusion matrix
            top_k: Number of top confused pairs to return
            
        Returns:
            List of (true_class, pred_class, count) tuples
        """
        pairs = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j and confusion[i, j] > 0:
                    true_name = self.class_names[i] if self.class_names else str(i)
                    pred_name = self.class_names[j] if self.class_names else str(j)
                    pairs.append((true_name, pred_name, int(confusion[i, j])))
        
        # Sort by count descending
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:top_k]
    
    def print_report(self) -> None:
        """
        Print a formatted classification report
        """
        if self.results is None:
            print("No results yet. Run evaluate() first.")
            return
        
        r = self.results
        
        print("=" * 70)
        print("Classification Report")
        print("=" * 70)
        
        # Header
        print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Accuracy':>10} {'Support':>10}")
        print("-" * 70)
        
        # Per-class rows
        for m in r['per_class']:
            print(f"{m['class_name']:<15} {m['precision']:>10.3f} {m['recall']:>10.3f} "
                  f"{m['f1']:>10.3f} {m['accuracy'] * 100:>9.1f}% {m['support']:>10}")
        
        print("-" * 70)
        
        # Summary
        total_support = sum(m['support'] for m in r['per_class'])
        print(f"{'Overall':<15} {'':>10} {'':>10} {r['macro_f1']:>10.3f} "
              f"{r['accuracy'] * 100:>9.1f}% {total_support:>10}")
        
        print(f"\n  Loss:       {r['loss']:.4f}")
        print(f"  Accuracy:   {r['accuracy'] * 100:.1f}% ({int(r['accuracy'] * r['num_samples'])}/{r['num_samples']})")
        print(f"  Macro F1:   {r['macro_f1']:.4f}")
        
        # Top confused pairs
        if r['top_confused']:
            print(f"\n  Top Misclassifications:")
            for true_name, pred_name, count in r['top_confused']:
                print(f"    {true_name} → {pred_name}: {count} times")
        
        print("=" * 70)
    
    def print_confusion_matrix(self) -> None:
        """
        Print a formatted confusion matrix
        """
        if self.results is None:
            print("No results yet. Run evaluate() first.")
            return
        
        cm = self.results['confusion_matrix']
        names = self.class_names if self.class_names else [str(i) for i in range(self.num_classes)]
        
        # Abbreviate long names
        short_names = [n[:7] for n in names]
        
        print("\nConfusion Matrix (rows=true, cols=predicted):")
        print("-" * (10 + 7 * self.num_classes))
        
        # Header row
        header = f"{'':>9} " + " ".join(f"{n:>6}" for n in short_names)
        print(header)
        print("-" * (10 + 7 * self.num_classes))
        
        # Data rows
        for i in range(self.num_classes):
            row = f"{short_names[i]:>9} " + " ".join(f"{cm[i, j]:>6}" for j in range(self.num_classes))
            print(row)
        
        print("-" * (10 + 7 * self.num_classes))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_evaluator():
    """Test Evaluator with synthetic data to verify mechanics"""
    
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.neural_network.network import NeuralNetwork
    from src.training.trainer import Trainer
    
    print("=" * 60)
    print("Testing Evaluator")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    # Create and train a small network on synthetic data
    np.random.seed(42)
    input_size = 64
    num_classes = 5
    class_names = ['bass', 'brass', 'flute', 'guitar', 'keyboard']
    
    # Make data slightly learnable (not pure random)
    n_train = 300
    n_test = 100
    
    X_train = np.random.randn(n_train, input_size)
    y_train_idx = np.random.randint(num_classes, size=n_train)
    # Add a small signal: shift mean based on class
    for i in range(n_train):
        X_train[i, :10] += y_train_idx[i] * 0.5
    y_train = np.zeros((n_train, num_classes))
    for i in range(n_train):
        y_train[i, y_train_idx[i]] = 1.0
    
    X_test = np.random.randn(n_test, input_size)
    y_test_idx = np.random.randint(num_classes, size=n_test)
    for i in range(n_test):
        X_test[i, :10] += y_test_idx[i] * 0.5
    y_test = np.zeros((n_test, num_classes))
    for i in range(n_test):
        y_test[i, y_test_idx[i]] = 1.0
    
    # Train a small network
    net = NeuralNetwork(input_size=input_size, hidden_sizes=[32, 16],
                        num_classes=num_classes, seed=42)
    trainer = Trainer(net, learning_rate=0.1, lr_decay=0.98,
                      batch_size=32, epochs=20, seed=42)
    
    print("\n--- Training small network (20 epochs) ---")
    trainer.train(X_train, y_train, X_train[:50], y_train[:50])
    
    # ---- Test 1: Evaluator initializes correctly ----
    print("\n1️⃣  Evaluator initializes correctly")
    
    evaluator = Evaluator(net, class_names=class_names)
    
    total += 1
    if evaluator.num_classes == 5 and evaluator.class_names == class_names:
        print(f"   ✓ num_classes={evaluator.num_classes}, class_names set")
        passed += 1
    else:
        print(f"   ✗ FAILED")
    
    # ---- Test 2: evaluate() returns results dict ----
    print("\n2️⃣  evaluate() returns results dict")
    
    results = evaluator.evaluate(X_test, y_test)
    
    total += 1
    expected_keys = ['loss', 'accuracy', 'confusion_matrix', 'per_class',
                     'macro_f1', 'top_confused', 'predictions', 'true_labels',
                     'probabilities', 'num_samples']
    if all(k in results for k in expected_keys):
        print(f"   ✓ All {len(expected_keys)} keys present in results")
        passed += 1
    else:
        missing = [k for k in expected_keys if k not in results]
        print(f"   ✗ FAILED: missing keys: {missing}")
    
    # ---- Test 3: Accuracy is valid ----
    print("\n3️⃣  Accuracy is valid (0-1 range)")
    
    total += 1
    if 0.0 <= results['accuracy'] <= 1.0:
        print(f"   ✓ Accuracy: {results['accuracy'] * 100:.1f}%")
        passed += 1
    else:
        print(f"   ✗ FAILED: accuracy = {results['accuracy']}")
    
    # ---- Test 4: Confusion matrix shape and values ----
    print("\n4️⃣  Confusion matrix shape and values")
    
    cm = results['confusion_matrix']
    
    total += 1
    if (cm.shape == (5, 5) and
        cm.sum() == n_test and
        cm.min() >= 0):
        print(f"   ✓ Shape: {cm.shape}, total entries sum to {cm.sum()} (= n_test)")
        passed += 1
    else:
        print(f"   ✗ FAILED: shape={cm.shape}, sum={cm.sum()}")
    
    # ---- Test 5: Confusion matrix diagonal = correct predictions ----
    print("\n5️⃣  Confusion matrix diagonal = correct predictions")
    
    total += 1
    correct_from_cm = np.trace(cm)
    correct_from_acc = int(results['accuracy'] * n_test)
    if correct_from_cm == correct_from_acc:
        print(f"   ✓ Diagonal sum: {correct_from_cm} = accuracy * n_test")
        passed += 1
    else:
        print(f"   ✗ FAILED: diagonal={correct_from_cm}, expected={correct_from_acc}")
    
    # ---- Test 6: Per-class metrics structure ----
    print("\n6️⃣  Per-class metrics structure")
    
    total += 1
    pc = results['per_class']
    if (len(pc) == 5 and
        all('precision' in m and 'recall' in m and 'f1' in m and
            'support' in m and 'class_name' in m for m in pc)):
        print(f"   ✓ {len(pc)} classes, all have precision/recall/F1/support")
        passed += 1
    else:
        print(f"   ✗ FAILED")
    
    # ---- Test 7: Precision and recall are valid ----
    print("\n7️⃣  Precision and recall are in [0, 1] range")
    
    total += 1
    all_valid = all(
        0.0 <= m['precision'] <= 1.0 and
        0.0 <= m['recall'] <= 1.0 and
        0.0 <= m['f1'] <= 1.0
        for m in pc
    )
    if all_valid:
        print(f"   ✓ All precision/recall/F1 values in valid range")
        for m in pc:
            print(f"     {m['class_name']}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
        passed += 1
    else:
        print(f"   ✗ FAILED")
    
    # ---- Test 8: Support sums to n_test ----
    print("\n8️⃣  Support sums to n_test")
    
    total += 1
    total_support = sum(m['support'] for m in pc)
    if total_support == n_test:
        print(f"   ✓ Total support: {total_support} = n_test ({n_test})")
        passed += 1
    else:
        print(f"   ✗ FAILED: total support = {total_support}, expected {n_test}")
    
    # ---- Test 9: Macro F1 is average of per-class F1 ----
    print("\n9️⃣  Macro F1 = average of per-class F1 scores")
    
    total += 1
    manual_macro_f1 = np.mean([m['f1'] for m in pc])
    if np.isclose(results['macro_f1'], manual_macro_f1):
        print(f"   ✓ Macro F1: {results['macro_f1']:.4f} (matches manual average)")
        passed += 1
    else:
        print(f"   ✗ FAILED: {results['macro_f1']} vs {manual_macro_f1}")
    
    # ---- Test 10: print_report runs without error ----
    print("\n🔟  print_report() runs without error")
    
    total += 1
    try:
        evaluator.print_report()
        print(f"   ✓ Report printed successfully")
        passed += 1
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
    
    # ---- Test 11: print_confusion_matrix runs without error ----
    print("\n1️⃣1️⃣ print_confusion_matrix() runs without error")
    
    total += 1
    try:
        evaluator.print_confusion_matrix()
        print(f"   ✓ Confusion matrix printed successfully")
        passed += 1
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
    
    # ---- Test 12: Top confused pairs ----
    print("\n1️⃣2️⃣ Top confused pairs")
    
    total += 1
    tc = results['top_confused']
    if isinstance(tc, list) and len(tc) <= 5:
        print(f"   ✓ {len(tc)} top confused pairs:")
        for true_name, pred_name, count in tc:
            print(f"     {true_name} → {pred_name}: {count} times")
        passed += 1
    else:
        print(f"   ✗ FAILED")
    
    # ---- Summary ----
    print(f"\n{'=' * 60}")
    if passed == total:
        print(f"✅ All {total} tests passed!")
    else:
        print(f"❌ {passed}/{total} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    test_evaluator()
