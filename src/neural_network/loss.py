"""
Loss Function Module
Implements Cross-Entropy Loss from scratch using NumPy

Cross-Entropy Loss measures how far the network's predicted probability
distribution is from the true label distribution. Lower = better.

The backward pass computes the combined Softmax + Cross-Entropy gradient,
which simplifies to (predicted - actual) / batch_size.
"""

import numpy as np


class CrossEntropyLoss:
    """
    Cross-Entropy Loss with combined Softmax backward pass
    
    Forward:  L = -mean( sum( y_true * log(y_pred) ) )
    Backward: dL/dz = (y_pred - y_true) / batch_size
    
    The backward gradient is computed with respect to the Softmax INPUT (logits),
    not the Softmax output. This is because the combined Softmax + CE gradient
    simplifies beautifully and avoids computing the full Softmax Jacobian.
    """
    
    def __init__(self, softmax):
        """
        Args:
            softmax: A Softmax instance from activations.py
                     Used to read cached softmax.output during backward
        """
        self.softmax = softmax
        self.y_true = None  # Cache true labels for backward pass
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute cross-entropy loss
        
        Args:
            y_pred: Predicted probabilities from Softmax, shape (batch_size, num_classes)
            y_true: One-hot encoded true labels, shape (batch_size, num_classes)
            
        Returns:
            Scalar loss value (float)
        """
        self.y_true = y_true
        
        # Clip predictions to prevent log(0) = -inf
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Cross-entropy: -sum(y_true * log(y_pred)) per sample, then average over batch
        # Since y_true is one-hot, only the true class contributes per sample
        sample_losses = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
        
        # Average over the batch
        loss = np.mean(sample_losses)
        
        return float(loss)
    
    def backward(self) -> np.ndarray:
        """
        Compute gradient of loss with respect to Softmax INPUT (logits)
        
        Combined Softmax + Cross-Entropy gradient:
            dL/dz = (softmax_output - y_true) / batch_size
        
        This is the gradient that flows back into the last Dense layer.
        
        Returns:
            Gradient array of shape (batch_size, num_classes)
        """
        batch_size = self.y_true.shape[0]
        
        # Combined softmax + cross-entropy gradient
        # Uses the cached softmax output from the forward pass
        grad = (self.softmax.output - self.y_true) / batch_size
        
        return grad


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cross_entropy_loss():
    """Test CrossEntropyLoss with known inputs and expected outputs"""
    
    # Import Softmax for integration testing
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.neural_network.activations import Softmax
    
    print("=" * 60)
    print("Testing Cross-Entropy Loss")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    # ---- Test 1: Perfect prediction → loss near 0 ----
    print("\n1️⃣  Perfect prediction → loss near 0")
    
    softmax = Softmax()
    loss_fn = CrossEntropyLoss(softmax)
    
    # Simulate: network is very confident and correct
    # High logit for the true class
    logits = np.array([[10.0, 0.0, 0.0],
                        [0.0, 10.0, 0.0],
                        [0.0, 0.0, 10.0]])
    
    y_true = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]], dtype=np.float64)
    
    probs = softmax.forward(logits)
    loss = loss_fn.forward(probs, y_true)
    
    total += 1
    if loss < 0.01:
        print(f"   ✓ Loss = {loss:.6f} (near zero for confident correct predictions)")
        passed += 1
    else:
        print(f"   ✗ FAILED: loss = {loss:.6f}, expected < 0.01")
    
    # ---- Test 2: Terrible prediction → loss is high ----
    print("\n2️⃣  Wrong prediction → loss is high")
    
    softmax2 = Softmax()
    loss_fn2 = CrossEntropyLoss(softmax2)
    
    # Network is very confident but WRONG
    logits_wrong = np.array([[0.0, 0.0, 10.0],   # predicts class 2, true is 0
                              [10.0, 0.0, 0.0],   # predicts class 0, true is 1
                              [0.0, 10.0, 0.0]])   # predicts class 1, true is 2
    
    probs_wrong = softmax2.forward(logits_wrong)
    loss_wrong = loss_fn2.forward(probs_wrong, y_true)
    
    total += 1
    if loss_wrong > 5.0:
        print(f"   ✓ Loss = {loss_wrong:.6f} (high for confident wrong predictions)")
        passed += 1
    else:
        print(f"   ✗ FAILED: loss = {loss_wrong:.6f}, expected > 5.0")
    
    # ---- Test 3: Loss is always non-negative ----
    print("\n3️⃣  Loss is always non-negative")
    
    softmax3 = Softmax()
    loss_fn3 = CrossEntropyLoss(softmax3)
    
    random_logits = np.random.randn(32, 10)
    random_labels = np.zeros((32, 10))
    for i in range(32):
        random_labels[i, np.random.randint(10)] = 1.0
    
    probs3 = softmax3.forward(random_logits)
    loss3 = loss_fn3.forward(probs3, random_labels)
    
    total += 1
    if loss3 >= 0.0:
        print(f"   ✓ Loss = {loss3:.6f} (non-negative)")
        passed += 1
    else:
        print(f"   ✗ FAILED: loss = {loss3:.6f}, expected >= 0")
    
    # ---- Test 4: Uniform predictions → loss = log(num_classes) ----
    print("\n4️⃣  Uniform predictions → loss ≈ log(num_classes)")
    
    softmax4 = Softmax()
    loss_fn4 = CrossEntropyLoss(softmax4)
    
    # All logits equal → softmax gives uniform 1/3
    uniform_logits = np.array([[1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0]])
    
    y_true4 = np.array([[1, 0, 0],
                          [0, 1, 0]], dtype=np.float64)
    
    probs4 = softmax4.forward(uniform_logits)
    loss4 = loss_fn4.forward(probs4, y_true4)
    expected_loss = np.log(3)  # = 1.0986...
    
    total += 1
    if np.abs(loss4 - expected_loss) < 0.01:
        print(f"   ✓ Loss = {loss4:.6f} ≈ log(3) = {expected_loss:.6f}")
        passed += 1
    else:
        print(f"   ✗ FAILED: loss = {loss4:.6f}, expected ≈ {expected_loss:.6f}")
    
    # ---- Test 5: Backward — gradient shape ----
    print("\n5️⃣  Backward — gradient shape correct")
    
    grad = loss_fn3.backward()
    
    total += 1
    if grad.shape == (32, 10):
        print(f"   ✓ Gradient shape: {grad.shape} (matches batch_size × num_classes)")
        passed += 1
    else:
        print(f"   ✗ FAILED: shape = {grad.shape}, expected (32, 10)")
    
    # ---- Test 6: Backward — gradient = (pred - true) / N ----
    print("\n6️⃣  Backward — gradient = (predicted - true) / batch_size")
    
    softmax6 = Softmax()
    loss_fn6 = CrossEntropyLoss(softmax6)
    
    logits6 = np.array([[2.0, 1.0, 0.1],
                          [0.5, 2.5, 1.0]])
    
    y_true6 = np.array([[1, 0, 0],
                          [0, 1, 0]], dtype=np.float64)
    
    probs6 = softmax6.forward(logits6)
    _ = loss_fn6.forward(probs6, y_true6)
    grad6 = loss_fn6.backward()
    
    expected_grad = (probs6 - y_true6) / 2.0  # batch_size = 2
    
    total += 1
    if np.allclose(grad6, expected_grad):
        print(f"   ✓ Gradient matches (predicted - true) / batch_size exactly")
        passed += 1
    else:
        print(f"   ✗ FAILED")
        print(f"     Expected: {expected_grad}")
        print(f"     Got:      {grad6}")
    
    # ---- Test 7: Gradient points in the right direction ----
    print("\n7️⃣  Gradient direction — negative for true class, positive for others")
    
    # For the true class: pred < 1.0, so (pred - 1) < 0 → gradient is negative (push up)
    # For other classes: pred > 0, so (pred - 0) > 0 → gradient is positive (push down)
    
    total += 1
    row0_true_class_grad = grad6[0, 0]   # True class is 0 for row 0
    row0_other_class_grad = grad6[0, 1]  # Other class for row 0
    
    if row0_true_class_grad < 0 and row0_other_class_grad > 0:
        print(f"   ✓ True class grad: {row0_true_class_grad:.6f} (negative → push probability up)")
        print(f"     Other class grad: {row0_other_class_grad:.6f} (positive → push probability down)")
        passed += 1
    else:
        print(f"   ✗ FAILED: true_grad={row0_true_class_grad}, other_grad={row0_other_class_grad}")
    
    # ---- Test 8: Full pipeline with real-sized data ----
    print("\n8️⃣  Full pipeline with real-sized data (32 samples, 10 classes)")
    
    softmax8 = Softmax()
    loss_fn8 = CrossEntropyLoss(softmax8)
    
    batch_logits = np.random.randn(32, 10)
    batch_labels = np.zeros((32, 10))
    for i in range(32):
        batch_labels[i, np.random.randint(10)] = 1.0
    
    batch_probs = softmax8.forward(batch_logits)
    batch_loss = loss_fn8.forward(batch_probs, batch_labels)
    batch_grad = loss_fn8.backward()
    
    total += 1
    checks = [
        batch_loss > 0,                              # Loss is positive
        not np.isnan(batch_loss),                     # No NaN
        batch_grad.shape == (32, 10),                 # Correct shape
        not np.any(np.isnan(batch_grad)),             # No NaN in gradient
        np.allclose(batch_grad.sum(axis=1), 0.0),    # Rows sum to ~0 (pred sums to 1, true sums to 1)
    ]
    
    if all(checks):
        print(f"   ✓ Loss: {batch_loss:.4f}, gradient shape: {batch_grad.shape}")
        print(f"     Gradient row sums ≈ 0: {np.allclose(batch_grad.sum(axis=1), 0.0)}")
        print(f"     No NaN/Inf: ✓")
        passed += 1
    else:
        print(f"   ✗ FAILED: checks = {checks}")
    
    # ---- Test 9: Numerical stability — no NaN with near-zero predictions ----
    print("\n9️⃣  Numerical stability — extreme logits don't produce NaN")
    
    softmax9 = Softmax()
    loss_fn9 = CrossEntropyLoss(softmax9)
    
    extreme_logits = np.array([[1000.0, -1000.0, -1000.0],
                                [-1000.0, -1000.0, 1000.0]])
    
    y_true9 = np.array([[1, 0, 0],
                          [0, 0, 1]], dtype=np.float64)
    
    probs9 = softmax9.forward(extreme_logits)
    loss9 = loss_fn9.forward(probs9, y_true9)
    grad9 = loss_fn9.backward()
    
    total += 1
    if not np.isnan(loss9) and not np.any(np.isnan(grad9)):
        print(f"   ✓ Loss = {loss9:.6f}, no NaN with extreme logits (±1000)")
        passed += 1
    else:
        print(f"   ✗ FAILED: NaN detected")
    
    # ---- Summary ----
    print(f"\n{'=' * 60}")
    if passed == total:
        print(f"✅ All {total} tests passed!")
    else:
        print(f"❌ {passed}/{total} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    test_cross_entropy_loss()
