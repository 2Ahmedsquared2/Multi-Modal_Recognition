"""
Dense (Fully Connected) Layer Module
Implements a Dense layer from scratch using NumPy

A Dense layer performs: output = input @ weights + bias
Each input feature is connected to every output feature through a learnable weight.

Forward:  output = X @ W + b
Backward: computes gradients for weights, biases, and input (to pass to previous layer)
"""

import numpy as np


class DenseLayer:
    """
    Fully Connected (Dense) Layer
    
    Forward:  output = X @ W + b
    Backward: computes dW, db (stored internally for optimizer), returns dX (for previous layer)
    
    Weight initialization: He initialization — scales by sqrt(2 / fan_in)
    to keep signal magnitudes healthy through ReLU-activated networks.
    """
    
    def __init__(self, input_size: int, output_size: int, seed: int = None):
        """
        Args:
            input_size:  Number of input features (e.g., 4096 for flattened spectrogram)
            output_size: Number of output features (e.g., 128 for hidden layer)
            seed:        Optional random seed for reproducible initialization
        """
        self.input_size = input_size
        self.output_size = output_size
        
        # Reproducible initialization if seed is provided
        rng = np.random.RandomState(seed) if seed is not None else np.random
        
        # He initialization: scale by sqrt(2 / fan_in)
        # Keeps variance stable through ReLU layers
        scale = np.sqrt(2.0 / input_size)
        self.weights = rng.randn(input_size, output_size) * scale
        
        # Biases initialized to zero (standard practice)
        self.biases = np.zeros((1, output_size))
        
        # Cache for backward pass
        self.input = None
        
        # Gradients — computed during backward, read by optimizer
        self.grad_weights = None
        self.grad_biases = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: output = X @ W + b
        
        Args:
            x: Input array of shape (batch_size, input_size)
            
        Returns:
            Output array of shape (batch_size, output_size)
        """
        self.input = x
        return x @ self.weights + self.biases
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: compute gradients and pass gradient to previous layer
        
        Computes:
            grad_weights = X^T @ grad_output        → stored in self.grad_weights
            grad_biases  = sum(grad_output, axis=0)  → stored in self.grad_biases
            grad_input   = grad_output @ W^T          → returned for previous layer
        
        Args:
            grad_output: Gradient flowing back from the next layer, shape (batch_size, output_size)
            
        Returns:
            Gradient to pass to the previous layer, shape (batch_size, input_size)
        """
        # Gradient w.r.t. weights: X^T @ grad_output
        self.grad_weights = self.input.T @ grad_output
        
        # Gradient w.r.t. biases: sum across batch dimension
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        
        # Gradient w.r.t. input: pass back to previous layer
        grad_input = grad_output @ self.weights.T
        
        return grad_input


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dense_layer():
    """Test DenseLayer with known inputs and expected behaviors"""
    
    print("=" * 60)
    print("Testing Dense Layer")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    # ---- Test 1: Output shape ----
    print("\n1️⃣  Forward — output shape correct")
    
    layer = DenseLayer(input_size=4096, output_size=128, seed=42)
    x = np.random.randn(32, 4096)
    out = layer.forward(x)
    
    total += 1
    if out.shape == (32, 128):
        print(f"   ✓ Input (32, 4096) → Output (32, 128)")
        passed += 1
    else:
        print(f"   ✗ FAILED: got shape {out.shape}, expected (32, 128)")
    
    # ---- Test 2: Forward is X @ W + b ----
    print("\n2️⃣  Forward — manual verification with small layer")
    
    small = DenseLayer(input_size=3, output_size=2, seed=0)
    x_small = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]])
    
    out_small = small.forward(x_small)
    expected = x_small @ small.weights + small.biases
    
    total += 1
    if np.allclose(out_small, expected):
        print(f"   ✓ Output matches X @ W + b exactly")
        print(f"     Row 0: {out_small[0]}")
        print(f"     Row 1: {out_small[1]}")
        passed += 1
    else:
        print(f"   ✗ FAILED")
    
    # ---- Test 3: He initialization scale ----
    print("\n3️⃣  He initialization — weight scale is correct")
    
    layer_he = DenseLayer(input_size=4096, output_size=128, seed=42)
    weight_std = layer_he.weights.std()
    expected_std = np.sqrt(2.0 / 4096)  # ~0.0221
    
    total += 1
    # Allow some tolerance since it's random
    if abs(weight_std - expected_std) < 0.005:
        print(f"   ✓ Weight std: {weight_std:.6f} ≈ sqrt(2/4096) = {expected_std:.6f}")
        passed += 1
    else:
        print(f"   ✗ FAILED: std = {weight_std:.6f}, expected ≈ {expected_std:.6f}")
    
    # ---- Test 4: Biases initialized to zero ----
    print("\n4️⃣  Biases initialized to zero")
    
    total += 1
    if np.all(layer_he.biases == 0.0):
        print(f"   ✓ All biases = 0.0, shape: {layer_he.biases.shape}")
        passed += 1
    else:
        print(f"   ✗ FAILED: biases not zero")
    
    # ---- Test 5: Backward — gradient shapes ----
    print("\n5️⃣  Backward — gradient shapes correct")
    
    layer5 = DenseLayer(input_size=4096, output_size=128, seed=42)
    x5 = np.random.randn(32, 4096)
    out5 = layer5.forward(x5)
    
    # Fake gradient from next layer
    grad_output = np.random.randn(32, 128)
    grad_input = layer5.backward(grad_output)
    
    total += 1
    shapes_ok = (
        layer5.grad_weights.shape == (4096, 128) and
        layer5.grad_biases.shape == (1, 128) and
        grad_input.shape == (32, 4096)
    )
    if shapes_ok:
        print(f"   ✓ grad_weights: {layer5.grad_weights.shape}")
        print(f"     grad_biases:  {layer5.grad_biases.shape}")
        print(f"     grad_input:   {grad_input.shape}")
        passed += 1
    else:
        print(f"   ✗ FAILED")
    
    # ---- Test 6: Backward — manual gradient verification ----
    print("\n6️⃣  Backward — manual gradient verification (small layer)")
    
    small2 = DenseLayer(input_size=3, output_size=2, seed=0)
    x_s = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
    
    out_s = small2.forward(x_s)
    grad_out_s = np.array([[0.1, 0.2],
                             [0.3, 0.4]])
    
    grad_in_s = small2.backward(grad_out_s)
    
    expected_gw = x_s.T @ grad_out_s
    expected_gb = np.sum(grad_out_s, axis=0, keepdims=True)
    expected_gi = grad_out_s @ small2.weights.T
    
    total += 1
    if (np.allclose(small2.grad_weights, expected_gw) and
        np.allclose(small2.grad_biases, expected_gb) and
        np.allclose(grad_in_s, expected_gi)):
        print(f"   ✓ grad_weights = X^T @ grad_output — exact match")
        print(f"     grad_biases  = sum(grad_output)   — exact match")
        print(f"     grad_input   = grad_output @ W^T  — exact match")
        passed += 1
    else:
        print(f"   ✗ FAILED")
    
    # ---- Test 7: Numerical gradient check ----
    print("\n7️⃣  Numerical gradient check (finite differences)")
    
    np.random.seed(42)
    layer7 = DenseLayer(input_size=5, output_size=3, seed=42)
    x7 = np.random.randn(4, 5)
    
    # Forward + backward to get analytical gradients
    out7 = layer7.forward(x7)
    # Use sum of output as a simple scalar loss for gradient checking
    grad_out7 = np.ones_like(out7)
    layer7.backward(grad_out7)
    
    # Numerical gradient for weights using finite differences
    eps = 1e-5
    numerical_grad_w = np.zeros_like(layer7.weights)
    
    for i in range(layer7.weights.shape[0]):
        for j in range(layer7.weights.shape[1]):
            # f(w + eps)
            layer7.weights[i, j] += eps
            out_plus = layer7.forward(x7)
            loss_plus = np.sum(out_plus)
            
            # f(w - eps)
            layer7.weights[i, j] -= 2 * eps
            out_minus = layer7.forward(x7)
            loss_minus = np.sum(out_minus)
            
            # Restore
            layer7.weights[i, j] += eps
            
            numerical_grad_w[i, j] = (loss_plus - loss_minus) / (2 * eps)
    
    # Re-run forward to restore cache
    layer7.forward(x7)
    layer7.backward(grad_out7)
    
    total += 1
    if np.allclose(layer7.grad_weights, numerical_grad_w, atol=1e-5):
        max_diff = np.max(np.abs(layer7.grad_weights - numerical_grad_w))
        print(f"   ✓ Analytical gradients match numerical (max diff: {max_diff:.2e})")
        passed += 1
    else:
        max_diff = np.max(np.abs(layer7.grad_weights - numerical_grad_w))
        print(f"   ✗ FAILED: max difference = {max_diff:.6f}")
    
    # ---- Test 8: Seed reproducibility ----
    print("\n8️⃣  Seed reproducibility")
    
    layer_a = DenseLayer(input_size=100, output_size=50, seed=123)
    layer_b = DenseLayer(input_size=100, output_size=50, seed=123)
    
    total += 1
    if np.array_equal(layer_a.weights, layer_b.weights):
        print(f"   ✓ Same seed → identical weights")
        passed += 1
    else:
        print(f"   ✗ FAILED: weights differ with same seed")
    
    # ---- Test 9: No NaN/Inf with real-sized data ----
    print("\n9️⃣  Full pipeline — no NaN/Inf with real-sized data")
    
    layer9 = DenseLayer(input_size=4096, output_size=128, seed=42)
    x9 = np.random.randn(32, 4096)
    
    out9 = layer9.forward(x9)
    grad9 = layer9.backward(np.random.randn(32, 128))
    
    total += 1
    checks = [
        not np.any(np.isnan(out9)),
        not np.any(np.isinf(out9)),
        not np.any(np.isnan(grad9)),
        not np.any(np.isinf(grad9)),
        not np.any(np.isnan(layer9.grad_weights)),
        not np.any(np.isnan(layer9.grad_biases)),
    ]
    if all(checks):
        print(f"   ✓ No NaN/Inf in output, gradients, or stored grads")
        print(f"     Output range: [{out9.min():.4f}, {out9.max():.4f}]")
        print(f"     Weight grad range: [{layer9.grad_weights.min():.4f}, {layer9.grad_weights.max():.4f}]")
        passed += 1
    else:
        print(f"   ✗ FAILED: NaN or Inf detected")
    
    # ---- Test 10: Chaining two layers (like our network) ----
    print("\n🔟  Chaining two Dense layers (simulates network)")
    
    layer_1 = DenseLayer(input_size=4096, output_size=128, seed=1)
    layer_2 = DenseLayer(input_size=128, output_size=10, seed=2)
    
    x10 = np.random.randn(32, 4096)
    
    # Forward through both
    h = layer_1.forward(x10)       # (32, 4096) → (32, 128)
    out10 = layer_2.forward(h)     # (32, 128)  → (32, 10)
    
    # Backward through both (fake loss gradient)
    grad_from_loss = np.random.randn(32, 10)
    grad_h = layer_2.backward(grad_from_loss)   # (32, 10)  → (32, 128)
    grad_x = layer_1.backward(grad_h)           # (32, 128) → (32, 4096)
    
    total += 1
    chain_ok = (
        h.shape == (32, 128) and
        out10.shape == (32, 10) and
        grad_h.shape == (32, 128) and
        grad_x.shape == (32, 4096) and
        layer_1.grad_weights.shape == (4096, 128) and
        layer_2.grad_weights.shape == (128, 10)
    )
    if chain_ok:
        print(f"   ✓ Forward:  (32, 4096) → (32, 128) → (32, 10)")
        print(f"     Backward: (32, 10) → (32, 128) → (32, 4096)")
        print(f"     Both layers have correct grad_weights shapes")
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
    test_dense_layer()
