"""
Neural Network Class
Assembles Dense layers, ReLU activations, Softmax output, and Cross-Entropy loss
into a complete feedforward neural network for instrument classification.

Architecture (default):
    Input (batch, 4096)
    → Dense(4096, 128) → ReLU
    → Dense(128, 64)   → ReLU
    → Dense(64, 10)    → Softmax
    Output (batch, 10) probabilities

Configurable via hidden_sizes parameter.
"""

import numpy as np
from pathlib import Path
from src.neural_network.dense import DenseLayer
from src.neural_network.activations import ReLU, Softmax
from src.neural_network.loss import CrossEntropyLoss


class NeuralNetwork:
    """
    Feedforward Neural Network for multi-class classification
    
    Chains Dense → ReLU pairs for hidden layers, followed by Dense → Softmax
    for the output layer. Includes integrated cross-entropy loss.
    
    The network computes forward/backward passes but does NOT update its own
    weights — that's the training loop's job (Step 9).
    """
    
    def __init__(self, input_size: int = 4096, hidden_sizes: list = None,
                 num_classes: int = 10, seed: int = None):
        """
        Args:
            input_size:   Number of input features (e.g., 4096 for flattened 64×64 spectrogram)
            hidden_sizes: List of hidden layer sizes (default: [128, 64])
            num_classes:  Number of output classes (e.g., 10 instruments)
            seed:         Optional random seed for reproducible weight initialization
        """
        if hidden_sizes is None:
            hidden_sizes = [128, 64]
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        
        # Build the layer sequence
        self.layers = []       # All layers in forward order (Dense, ReLU, Dense, ReLU, ..., Dense)
        self.dense_layers = [] # Just the Dense layers (for optimizer access)
        
        # Determine layer sizes: [input_size, hidden1, hidden2, ..., num_classes]
        all_sizes = [input_size] + hidden_sizes + [num_classes]
        
        # Use incrementing seeds for each layer if seed is provided
        for i in range(len(all_sizes) - 1):
            layer_seed = (seed + i) if seed is not None else None
            
            # Dense layer
            dense = DenseLayer(all_sizes[i], all_sizes[i + 1], seed=layer_seed)
            self.layers.append(dense)
            self.dense_layers.append(dense)
            
            # Add ReLU after every hidden layer (NOT after the last Dense)
            if i < len(all_sizes) - 2:
                self.layers.append(ReLU())
        
        # Output activation
        self.softmax = Softmax()
        
        # Loss function (coupled with softmax for combined backward)
        self.loss_fn = CrossEntropyLoss(self.softmax)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through all layers + softmax
        
        Args:
            x: Input array of shape (batch_size, input_size)
            
        Returns:
            Probabilities of shape (batch_size, num_classes)
        """
        # Pass through all layers (Dense → ReLU → Dense → ReLU → ... → Dense)
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        
        # Final softmax activation
        out = self.softmax.forward(out)
        
        return out
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute cross-entropy loss
        
        Args:
            y_pred: Predicted probabilities from forward(), shape (batch_size, num_classes)
            y_true: One-hot encoded true labels, shape (batch_size, num_classes)
            
        Returns:
            Scalar loss value
        """
        return self.loss_fn.forward(y_pred, y_true)
    
    def backward(self) -> None:
        """
        Backward pass through all layers in reverse order
        
        After this call, every Dense layer has its grad_weights and grad_biases
        populated, ready for the optimizer to read and update.
        """
        # Start with the combined softmax + cross-entropy gradient
        grad = self.loss_fn.backward()
        
        # Pass gradient backward through all layers in reverse
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class labels (argmax of forward output)
        
        Args:
            x: Input array of shape (batch_size, input_size)
            
        Returns:
            Predicted class indices of shape (batch_size,)
        """
        probs = self.forward(x)
        return np.argmax(probs, axis=1)
    
    def get_trainable_layers(self) -> list:
        """
        Return list of Dense layers for the optimizer to update
        
        Each layer has:
            - layer.weights, layer.biases (parameters to update)
            - layer.grad_weights, layer.grad_biases (gradients from backward)
            
        Returns:
            List of DenseLayer instances
        """
        return self.dense_layers
    
    def count_parameters(self) -> int:
        """
        Count total trainable parameters (weights + biases)
        
        Returns:
            Total number of trainable parameters
        """
        total = 0
        for layer in self.dense_layers:
            total += layer.weights.size + layer.biases.size
        return total
    
    def summary(self) -> None:
        """
        Print a summary of the network architecture and parameter counts
        """
        print("=" * 65)
        print("Neural Network Summary")
        print("=" * 65)
        print(f"{'Layer':<30} {'Output Shape':<18} {'Parameters':>10}")
        print("-" * 65)
        
        # Input
        print(f"{'Input':<30} {'(batch, ' + str(self.input_size) + ')':<18} {'0':>10}")
        
        total_params = 0
        dense_idx = 0
        
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                dense_idx += 1
                w_params = layer.weights.size
                b_params = layer.biases.size
                layer_params = w_params + b_params
                total_params += layer_params
                
                in_s = layer.input_size
                out_s = layer.output_size
                name = f"Dense_{dense_idx} ({in_s} → {out_s})"
                shape = f"(batch, {out_s})"
                print(f"{name:<30} {shape:<18} {layer_params:>10,}")
                
            elif isinstance(layer, ReLU):
                print(f"{'ReLU':<30} {'(same)':<18} {'0':>10}")
        
        # Softmax
        print(f"{'Softmax':<30} {'(batch, ' + str(self.num_classes) + ')':<18} {'0':>10}")
        
        print("-" * 65)
        print(f"{'Total trainable parameters:':<48} {total_params:>10,}")
        print("=" * 65)
    
    def save(self, save_path: str) -> str:
        """
        Save trained model weights and architecture config to disk
        
        Args:
            save_path: Directory or file path to save the model.
                       If directory, saves as 'model.npz' inside it.
                       
        Returns:
            Path to the saved file
        """
        save_path = Path(save_path)
        
        if not save_path.suffix:
            # It's a directory — save as model.npz inside it
            save_path.mkdir(parents=True, exist_ok=True)
            save_path = save_path / "model.npz"
        else:
            save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Collect weights and biases from all Dense layers
        save_dict = {
            'input_size': self.input_size,
            'hidden_sizes': np.array(self.hidden_sizes),
            'num_classes': self.num_classes,
        }
        
        for i, layer in enumerate(self.dense_layers):
            save_dict[f'weights_{i}'] = layer.weights
            save_dict[f'biases_{i}'] = layer.biases
        
        np.savez_compressed(save_path, **save_dict)
        
        file_size_mb = save_path.stat().st_size / (1024 * 1024)
        print(f"💾 Model saved to: {save_path} ({file_size_mb:.2f} MB)")
        
        return str(save_path)
    
    @classmethod
    def load(cls, load_path: str, seed: int = None) -> 'NeuralNetwork':
        """
        Load a saved model from disk
        
        Args:
            load_path: Path to the saved .npz file
            seed:      Not used for loading (architecture is rebuilt, weights overwritten)
            
        Returns:
            NeuralNetwork instance with loaded weights
        """
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        data = np.load(load_path, allow_pickle=True)
        
        input_size = int(data['input_size'])
        hidden_sizes = data['hidden_sizes'].tolist()
        num_classes = int(data['num_classes'])
        
        # Rebuild architecture
        net = cls(input_size=input_size, hidden_sizes=hidden_sizes,
                  num_classes=num_classes, seed=0)
        
        # Overwrite weights with saved values
        for i, layer in enumerate(net.dense_layers):
            layer.weights = data[f'weights_{i}']
            layer.biases = data[f'biases_{i}']
        
        print(f"📂 Model loaded from: {load_path}")
        print(f"   Architecture: {input_size} → {' → '.join(map(str, hidden_sizes))} → {num_classes}")
        print(f"   Parameters: {net.count_parameters():,}")
        
        return net


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_neural_network():
    """Test NeuralNetwork with known inputs and expected behaviors"""
    
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    print("=" * 60)
    print("Testing Neural Network")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    # ---- Test 1: Default architecture builds correctly ----
    print("\n1️⃣  Default architecture (4096 → 128 → 64 → 10)")
    
    net = NeuralNetwork(input_size=4096, hidden_sizes=[128, 64], num_classes=10, seed=42)
    
    total += 1
    if (len(net.dense_layers) == 3 and
        net.dense_layers[0].weights.shape == (4096, 128) and
        net.dense_layers[1].weights.shape == (128, 64) and
        net.dense_layers[2].weights.shape == (64, 10)):
        print(f"   ✓ 3 Dense layers with correct shapes")
        print(f"     Dense 1: (4096, 128)")
        print(f"     Dense 2: (128, 64)")
        print(f"     Dense 3: (64, 10)")
        passed += 1
    else:
        print(f"   ✗ FAILED")
    
    # ---- Test 2: Custom architecture ----
    print("\n2️⃣  Custom architecture (100 → 256 → 128 → 64 → 5)")
    
    custom_net = NeuralNetwork(input_size=100, hidden_sizes=[256, 128, 64], num_classes=5, seed=0)
    
    total += 1
    if (len(custom_net.dense_layers) == 4 and
        custom_net.dense_layers[0].weights.shape == (100, 256) and
        custom_net.dense_layers[1].weights.shape == (256, 128) and
        custom_net.dense_layers[2].weights.shape == (128, 64) and
        custom_net.dense_layers[3].weights.shape == (64, 5)):
        print(f"   ✓ 4 Dense layers with correct shapes")
        passed += 1
    else:
        print(f"   ✗ FAILED")
    
    # ---- Test 3: Forward output shape and probabilities ----
    print("\n3️⃣  Forward — output shape and valid probabilities")
    
    x = np.random.randn(32, 4096)
    probs = net.forward(x)
    
    total += 1
    row_sums = probs.sum(axis=1)
    if (probs.shape == (32, 10) and
        np.allclose(row_sums, 1.0) and
        probs.min() > 0):
        print(f"   ✓ Shape: {probs.shape}, all rows sum to 1.0, all positive")
        passed += 1
    else:
        print(f"   ✗ FAILED: shape={probs.shape}, sums={row_sums}")
    
    # ---- Test 4: Compute loss ----
    print("\n4️⃣  Compute loss — returns valid scalar")
    
    y_true = np.zeros((32, 10))
    for i in range(32):
        y_true[i, np.random.randint(10)] = 1.0
    
    loss = net.compute_loss(probs, y_true)
    
    total += 1
    if isinstance(loss, float) and loss > 0 and not np.isnan(loss):
        print(f"   ✓ Loss = {loss:.4f} (positive scalar, no NaN)")
        passed += 1
    else:
        print(f"   ✗ FAILED: loss = {loss}")
    
    # ---- Test 5: Backward populates all gradients ----
    print("\n5️⃣  Backward — all Dense layers have gradients")
    
    net.backward()
    
    total += 1
    all_grads_ok = True
    for i, layer in enumerate(net.dense_layers):
        if layer.grad_weights is None or layer.grad_biases is None:
            all_grads_ok = False
            break
        if np.any(np.isnan(layer.grad_weights)) or np.any(np.isnan(layer.grad_biases)):
            all_grads_ok = False
            break
    
    if all_grads_ok:
        print(f"   ✓ All 3 Dense layers have valid gradients")
        for i, layer in enumerate(net.dense_layers):
            print(f"     Dense {i+1}: grad_weights {layer.grad_weights.shape}, grad_biases {layer.grad_biases.shape}")
        passed += 1
    else:
        print(f"   ✗ FAILED: missing or NaN gradients")
    
    # ---- Test 6: Predict returns class indices ----
    print("\n6️⃣  Predict — returns class indices")
    
    predictions = net.predict(x)
    
    total += 1
    if (predictions.shape == (32,) and
        predictions.min() >= 0 and
        predictions.max() <= 9 and
        predictions.dtype in [np.int64, np.intp]):
        print(f"   ✓ Shape: {predictions.shape}, range: [{predictions.min()}, {predictions.max()}]")
        print(f"     First 10 predictions: {predictions[:10]}")
        passed += 1
    else:
        print(f"   ✗ FAILED: shape={predictions.shape}, dtype={predictions.dtype}")
    
    # ---- Test 7: get_trainable_layers ----
    print("\n7️⃣  get_trainable_layers — returns Dense layers")
    
    trainable = net.get_trainable_layers()
    
    total += 1
    if (len(trainable) == 3 and
        all(isinstance(l, DenseLayer) for l in trainable)):
        print(f"   ✓ {len(trainable)} trainable layers, all DenseLayer instances")
        passed += 1
    else:
        print(f"   ✗ FAILED")
    
    # ---- Test 8: Parameter count ----
    print("\n8️⃣  Parameter count")
    
    param_count = net.count_parameters()
    # Dense 1: 4096*128 + 128 = 524,416
    # Dense 2: 128*64 + 64 = 8,256
    # Dense 3: 64*10 + 10 = 650
    # Total: 533,322
    expected_params = (4096 * 128 + 128) + (128 * 64 + 64) + (64 * 10 + 10)
    
    total += 1
    if param_count == expected_params:
        print(f"   ✓ Total parameters: {param_count:,} (matches expected {expected_params:,})")
        passed += 1
    else:
        print(f"   ✗ FAILED: got {param_count}, expected {expected_params}")
    
    # ---- Test 9: Summary prints without error ----
    print("\n9️⃣  Summary output")
    
    total += 1
    try:
        net.summary()
        print(f"   ✓ Summary printed successfully")
        passed += 1
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
    
    # ---- Test 10: Seed reproducibility ----
    print("\n🔟  Seed reproducibility")
    
    net_a = NeuralNetwork(input_size=4096, hidden_sizes=[128, 64], num_classes=10, seed=42)
    net_b = NeuralNetwork(input_size=4096, hidden_sizes=[128, 64], num_classes=10, seed=42)
    
    total += 1
    weights_match = all(
        np.array_equal(a.weights, b.weights)
        for a, b in zip(net_a.dense_layers, net_b.dense_layers)
    )
    if weights_match:
        print(f"   ✓ Same seed → identical weights across all layers")
        passed += 1
    else:
        print(f"   ✗ FAILED: weights differ")
    
    # ---- Test 11: Full forward-backward-update cycle (simulated) ----
    print("\n1️⃣1️⃣ Full training step simulation")
    
    net11 = NeuralNetwork(input_size=4096, hidden_sizes=[128, 64], num_classes=10, seed=42)
    x11 = np.random.randn(32, 4096)
    y11 = np.zeros((32, 10))
    for i in range(32):
        y11[i, np.random.randint(10)] = 1.0
    
    # Forward
    probs11 = net11.forward(x11)
    loss_before = net11.compute_loss(probs11, y11)
    
    # Backward
    net11.backward()
    
    # Simulate weight update (SGD with lr=0.01)
    lr = 0.01
    for layer in net11.get_trainable_layers():
        layer.weights -= lr * layer.grad_weights
        layer.biases -= lr * layer.grad_biases
    
    # Forward again — loss should decrease after one step
    probs11_after = net11.forward(x11)
    loss_after = net11.compute_loss(probs11_after, y11)
    
    total += 1
    if loss_after < loss_before:
        print(f"   ✓ Loss decreased: {loss_before:.4f} → {loss_after:.4f} (Δ = {loss_before - loss_after:.4f})")
        passed += 1
    else:
        print(f"   ✗ FAILED: loss didn't decrease ({loss_before:.4f} → {loss_after:.4f})")
    
    # ---- Test 12: No NaN/Inf throughout ----
    print("\n1️⃣2️⃣ No NaN/Inf throughout entire pipeline")
    
    total += 1
    all_clean = True
    for layer in net11.dense_layers:
        if (np.any(np.isnan(layer.weights)) or np.any(np.isinf(layer.weights)) or
            np.any(np.isnan(layer.biases)) or np.any(np.isinf(layer.biases)) or
            np.any(np.isnan(layer.grad_weights)) or np.any(np.isinf(layer.grad_weights)) or
            np.any(np.isnan(layer.grad_biases)) or np.any(np.isinf(layer.grad_biases))):
            all_clean = False
            break
    
    if all_clean and not np.isnan(loss_after):
        print(f"   ✓ All weights, biases, gradients, and loss are clean")
        passed += 1
    else:
        print(f"   ✗ FAILED: NaN or Inf detected")
    
    # ---- Summary ----
    print(f"\n{'=' * 60}")
    if passed == total:
        print(f"✅ All {total} tests passed!")
    else:
        print(f"❌ {passed}/{total} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    test_neural_network()
