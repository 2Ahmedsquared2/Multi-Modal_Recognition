"""
Activation Functions Module
Implements ReLU and Softmax activation functions from scratch using NumPy

ReLU:    Used after hidden layers — lets positive values through, blocks negatives
Softmax: Used at output layer — converts raw scores to probabilities summing to 1.0
"""

import numpy as np


class ReLU:
    """
    Rectified Linear Unit activation function
    
    Forward:  output = max(0, x)
    Backward: gradient passes through where input was positive, blocked where ≤ 0
    """
    
    def __init__(self):
        self.input = None  # Cache input for backward pass
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: apply ReLU element-wise
        
        Args:
            x: Input array of any shape (typically batch_size, features)
            
        Returns:
            Array with same shape, negative values replaced with 0
        """
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: pass gradient through where input was positive
        
        Args:
            grad_output: Gradient flowing back from the next layer
            
        Returns:
            Gradient to pass to the previous layer
        """
        # ReLU derivative: 1 where input > 0, 0 where input <= 0
        mask = (self.input > 0).astype(np.float64)
        return grad_output * mask


class Softmax:
    """
    Softmax activation function (output layer only)
    
    Forward: converts raw scores (logits) to probabilities summing to 1.0
    
    Backward is NOT implemented here — it's handled together with
    cross-entropy loss in Step 6, where the combined gradient simplifies
    to just (predicted - actual). This is both faster and more numerically stable.
    """
    
    def __init__(self):
        self.output = None  # Cache output for use in loss backward
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: convert logits to probabilities
        
        Uses the max-subtraction trick for numerical stability:
        exp(x - max(x)) prevents overflow from large exponentials
        
        Args:
            x: Raw scores of shape (batch_size, num_classes)
            
        Returns:
            Probabilities of shape (batch_size, num_classes), each row sums to 1.0
        """
        # Subtract max per row for numerical stability (doesn't change the result)
        shifted = x - np.max(x, axis=1, keepdims=True)
        
        # Exponentiate
        exp_vals = np.exp(shifted)
        
        # Normalize: divide each by its row sum
        self.output = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        
        return self.output


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_activations():
    """Test ReLU and Softmax with known inputs and expected outputs"""
    
    print("=" * 60)
    print("Testing Activation Functions")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    # ---- ReLU Forward ----
    print("\n1️⃣  ReLU Forward")
    relu = ReLU()
    
    x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0],
                   [3.0, -0.5, 0.1, -3.0, 0.0]])
    
    expected = np.array([[0.0, 0.0, 0.0, 1.0, 2.0],
                          [3.0, 0.0, 0.1, 0.0, 0.0]])
    
    out = relu.forward(x)
    total += 1
    if np.allclose(out, expected):
        print(f"   ✓ Correct: negatives → 0, positives unchanged")
        passed += 1
    else:
        print(f"   ✗ FAILED")
        print(f"     Expected: {expected}")
        print(f"     Got:      {out}")
    
    # ---- ReLU Backward ----
    print("\n2️⃣  ReLU Backward")
    
    grad_output = np.ones_like(x)  # Pretend gradient of 1 everywhere
    grad_input = relu.backward(grad_output)
    
    # Gradient should be 1 where input > 0, 0 elsewhere
    expected_grad = np.array([[0.0, 0.0, 0.0, 1.0, 1.0],
                               [1.0, 0.0, 1.0, 0.0, 0.0]])
    
    total += 1
    if np.allclose(grad_input, expected_grad):
        print(f"   ✓ Correct: gradient passes where input > 0, blocked where ≤ 0")
        passed += 1
    else:
        print(f"   ✗ FAILED")
        print(f"     Expected: {expected_grad}")
        print(f"     Got:      {grad_input}")
    
    # ---- ReLU with real-sized data ----
    print("\n3️⃣  ReLU with batch data (like training)")
    
    batch = np.random.randn(32, 4096)  # 32 samples, 4096 features
    out = relu.forward(batch)
    
    total += 1
    if out.shape == (32, 4096) and out.min() >= 0.0:
        print(f"   ✓ Shape preserved: {out.shape}, no negatives (min={out.min():.4f})")
        passed += 1
    else:
        print(f"   ✗ FAILED: shape={out.shape}, min={out.min():.4f}")
    
    # ---- Softmax Forward ----
    print("\n4️⃣  Softmax Forward")
    softmax = Softmax()
    
    logits = np.array([[2.0, 1.0, 0.1],
                        [1.0, 2.0, 3.0]])
    
    probs = softmax.forward(logits)
    
    total += 1
    row_sums = probs.sum(axis=1)
    if np.allclose(row_sums, 1.0) and probs.min() > 0:
        print(f"   ✓ All probabilities positive, rows sum to 1.0")
        print(f"     Row 0: {probs[0]} (sum={row_sums[0]:.6f})")
        print(f"     Row 1: {probs[1]} (sum={row_sums[1]:.6f})")
        passed += 1
    else:
        print(f"   ✗ FAILED: sums={row_sums}")
    
    # ---- Softmax highest prob matches highest logit ----
    print("\n5️⃣  Softmax preserves ordering")
    
    total += 1
    if np.argmax(probs[0]) == np.argmax(logits[0]) and \
       np.argmax(probs[1]) == np.argmax(logits[1]):
        print(f"   ✓ Highest logit → highest probability")
        passed += 1
    else:
        print(f"   ✗ FAILED: ordering not preserved")
    
    # ---- Softmax numerical stability ----
    print("\n6️⃣  Softmax numerical stability (large values)")
    
    large_logits = np.array([[1000.0, 1001.0, 999.0],
                              [-1000.0, -999.0, -1001.0]])
    
    probs_large = softmax.forward(large_logits)
    
    total += 1
    if not np.any(np.isnan(probs_large)) and not np.any(np.isinf(probs_large)):
        print(f"   ✓ No NaN or Inf with extreme values")
        print(f"     Row 0: {probs_large[0]}")
        print(f"     Row 1: {probs_large[1]}")
        passed += 1
    else:
        print(f"   ✗ FAILED: contains NaN or Inf")
    
    # ---- Softmax with real-sized data ----
    print("\n7️⃣  Softmax with batch data (10 classes, like our network)")
    
    batch_logits = np.random.randn(32, 10)  # 32 samples, 10 classes
    batch_probs = softmax.forward(batch_logits)
    
    total += 1
    sums_ok = np.allclose(batch_probs.sum(axis=1), 1.0)
    positive = batch_probs.min() > 0
    if sums_ok and positive and batch_probs.shape == (32, 10):
        print(f"   ✓ Shape: {batch_probs.shape}, all rows sum to 1.0, all positive")
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
    test_activations()
