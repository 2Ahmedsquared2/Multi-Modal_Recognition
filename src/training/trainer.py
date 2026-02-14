"""
Training Loop Module
Implements the full training pipeline: SGD optimizer, learning rate decay,
epoch logging, early stopping, and history tracking.

The Trainer class takes a NeuralNetwork and prepared data, runs the
training loop, and returns a history dictionary for visualization.
"""

import numpy as np
import time


class Trainer:
    """
    Trains a NeuralNetwork using mini-batch Stochastic Gradient Descent (SGD)
    
    Features:
    - Mini-batch SGD with configurable learning rate
    - Learning rate decay (multiplicative per epoch)
    - Early stopping based on validation loss (optional)
    - Per-epoch logging with loss, accuracy, and learning rate
    - History tracking for plotting loss/accuracy curves
    """
    
    def __init__(self, network, learning_rate: float = 0.01,
                 lr_decay: float = 0.95, batch_size: int = 32,
                 epochs: int = 100, patience: int = None, seed: int = 42):
        """
        Args:
            network:       NeuralNetwork instance to train
            learning_rate: Initial learning rate for SGD (default: 0.01)
            lr_decay:      Multiply LR by this factor each epoch (default: 0.95)
            batch_size:    Samples per mini-batch (default: 32)
            epochs:        Maximum number of training epochs (default: 100)
            patience:      Early stopping patience — stop if val loss hasn't improved
                           for this many epochs. None = no early stopping (default: None)
            seed:          Random seed for shuffling (default: 42)
        """
        self.network = network
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        
        # History — populated during training
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
            'epoch_time': [],
        }
    
    def _shuffle_data(self, X: np.ndarray, y: np.ndarray,
                      rng: np.random.RandomState) -> tuple:
        """Shuffle X and y together, preserving correspondence"""
        indices = np.arange(len(X))
        rng.shuffle(indices)
        return X[indices], y[indices]
    
    def _compute_accuracy(self, y_pred_probs: np.ndarray,
                          y_true_onehot: np.ndarray) -> float:
        """
        Compute classification accuracy
        
        Args:
            y_pred_probs: Predicted probabilities, shape (N, C)
            y_true_onehot: One-hot true labels, shape (N, C)
            
        Returns:
            Accuracy as a float between 0.0 and 1.0
        """
        pred_classes = np.argmax(y_pred_probs, axis=1)
        true_classes = np.argmax(y_true_onehot, axis=1)
        return np.mean(pred_classes == true_classes)
    
    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Evaluate the network on a dataset (forward pass only, no backward)
        
        Args:
            X: Feature array, shape (N, D)
            y: One-hot labels, shape (N, C)
            
        Returns:
            (loss, accuracy) tuple
        """
        probs = self.network.forward(X)
        loss = self.network.compute_loss(probs, y)
        accuracy = self._compute_accuracy(probs, y)
        return loss, accuracy
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> dict:
        """
        Run the full training loop
        
        Args:
            X_train: Training features, shape (N_train, D)
            y_train: Training labels (one-hot), shape (N_train, C)
            X_val:   Validation features, shape (N_val, D)
            y_val:   Validation labels (one-hot), shape (N_val, C)
            
        Returns:
            History dictionary with per-epoch metrics
        """
        rng = np.random.RandomState(self.seed)
        n_train = len(X_train)
        n_batches = int(np.ceil(n_train / self.batch_size))
        
        # Reset history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
            'epoch_time': [],
        }
        
        # Early stopping state
        best_val_loss = float('inf')
        best_epoch = 0
        epochs_without_improvement = 0
        
        # Reset learning rate
        self.learning_rate = self.initial_lr
        
        print("=" * 80)
        print(f"Training Started — {self.epochs} epochs, batch_size={self.batch_size}, "
              f"lr={self.learning_rate}, decay={self.lr_decay}")
        print(f"Training: {n_train} samples ({n_batches} batches/epoch) | "
              f"Validation: {len(X_val)} samples")
        if self.patience is not None:
            print(f"Early stopping: patience={self.patience}")
        print("=" * 80)
        
        total_start = time.time()
        
        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            
            # --- Training phase ---
            X_shuffled, y_shuffled = self._shuffle_data(X_train, y_train, rng)
            
            epoch_losses = []
            epoch_correct = 0
            epoch_total = 0
            
            for batch_start in range(0, n_train, self.batch_size):
                batch_end = min(batch_start + self.batch_size, n_train)
                X_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]
                
                # Forward pass
                probs = self.network.forward(X_batch)
                loss = self.network.compute_loss(probs, y_batch)
                
                # Backward pass
                self.network.backward()
                
                # Update weights (SGD)
                for layer in self.network.get_trainable_layers():
                    layer.weights -= self.learning_rate * layer.grad_weights
                    layer.biases -= self.learning_rate * layer.grad_biases
                
                # Track batch metrics
                epoch_losses.append(loss * len(X_batch))  # Weight by batch size
                pred_classes = np.argmax(probs, axis=1)
                true_classes = np.argmax(y_batch, axis=1)
                epoch_correct += np.sum(pred_classes == true_classes)
                epoch_total += len(X_batch)
            
            # Compute epoch training metrics
            train_loss = sum(epoch_losses) / epoch_total
            train_accuracy = epoch_correct / epoch_total
            
            # --- Validation phase (no gradient updates) ---
            val_loss, val_accuracy = self._evaluate(X_val, y_val)
            
            # --- Learning rate decay ---
            self.learning_rate *= self.lr_decay
            
            # --- Record history ---
            epoch_time = time.time() - epoch_start
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_accuracy)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            self.history['learning_rate'].append(self.learning_rate / self.lr_decay)  # LR used this epoch
            self.history['epoch_time'].append(epoch_time)
            
            # --- Logging ---
            print(f"Epoch {epoch:>3}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy * 100:>5.1f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy * 100:>5.1f}% | "
                  f"LR: {self.history['learning_rate'][-1]:.6f} | "
                  f"Time: {epoch_time:.2f}s")
            
            # --- Early stopping check ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if self.patience is not None and epochs_without_improvement >= self.patience:
                print(f"\n⏹️  Early stopping at epoch {epoch} — "
                      f"val loss hasn't improved for {self.patience} epochs")
                print(f"   Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
                break
        
        # --- Training complete ---
        total_time = time.time() - total_start
        final_epoch = len(self.history['train_loss'])
        
        print(f"\n{'=' * 80}")
        print(f"Training Complete!")
        print(f"{'=' * 80}")
        print(f"  Total time:        {total_time:.1f}s ({total_time / final_epoch:.2f}s/epoch)")
        print(f"  Epochs trained:    {final_epoch}")
        print(f"  Best val loss:     {best_val_loss:.4f} (epoch {best_epoch})")
        print(f"  Final train acc:   {self.history['train_accuracy'][-1] * 100:.1f}%")
        print(f"  Final val acc:     {self.history['val_accuracy'][-1] * 100:.1f}%")
        print(f"{'=' * 80}")
        
        return self.history


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_trainer():
    """Test Trainer with synthetic data to verify mechanics"""
    
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.neural_network.network import NeuralNetwork
    
    print("=" * 60)
    print("Testing Trainer")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    # Create synthetic data (small, fast)
    np.random.seed(42)
    n_train = 200
    n_val = 50
    input_size = 64  # Small for fast testing
    num_classes = 5
    
    X_train = np.random.randn(n_train, input_size)
    y_train = np.zeros((n_train, num_classes))
    for i in range(n_train):
        y_train[i, np.random.randint(num_classes)] = 1.0
    
    X_val = np.random.randn(n_val, input_size)
    y_val = np.zeros((n_val, num_classes))
    for i in range(n_val):
        y_val[i, np.random.randint(num_classes)] = 1.0
    
    # ---- Test 1: Trainer initializes correctly ----
    print("\n1️⃣  Trainer initializes correctly")
    
    net = NeuralNetwork(input_size=input_size, hidden_sizes=[32, 16],
                        num_classes=num_classes, seed=42)
    trainer = Trainer(net, learning_rate=0.1, lr_decay=0.95,
                      batch_size=32, epochs=5, seed=42)
    
    total += 1
    if (trainer.learning_rate == 0.1 and trainer.lr_decay == 0.95 and
        trainer.batch_size == 32 and trainer.epochs == 5):
        print(f"   ✓ All parameters set correctly")
        passed += 1
    else:
        print(f"   ✗ FAILED")
    
    # ---- Test 2: 5-epoch training completes ----
    print("\n2️⃣  5-epoch training completes without error")
    
    total += 1
    try:
        history = trainer.train(X_train, y_train, X_val, y_val)
        print(f"   ✓ Training completed successfully")
        passed += 1
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        history = None
    
    # ---- Test 3: History has correct structure ----
    print("\n3️⃣  History has correct structure")
    
    total += 1
    if history is not None:
        keys_ok = all(k in history for k in ['train_loss', 'train_accuracy',
                                                'val_loss', 'val_accuracy',
                                                'learning_rate', 'epoch_time'])
        lengths_ok = all(len(history[k]) == 5 for k in history)
        
        if keys_ok and lengths_ok:
            print(f"   ✓ All 6 keys present, each has 5 entries (one per epoch)")
            passed += 1
        else:
            print(f"   ✗ FAILED: keys={keys_ok}, lengths={[len(history[k]) for k in history]}")
    else:
        print(f"   ✗ FAILED: no history returned")
    
    # ---- Test 4: Training loss decreases ----
    print("\n4️⃣  Training loss decreases over epochs")
    
    total += 1
    if history is not None:
        first_loss = history['train_loss'][0]
        last_loss = history['train_loss'][-1]
        if last_loss < first_loss:
            print(f"   ✓ Loss: {first_loss:.4f} → {last_loss:.4f} (decreased)")
            passed += 1
        else:
            print(f"   ✗ FAILED: {first_loss:.4f} → {last_loss:.4f} (did not decrease)")
    else:
        print(f"   ✗ FAILED: no history")
    
    # ---- Test 5: Training accuracy increases ----
    print("\n5️⃣  Training accuracy increases over epochs")
    
    total += 1
    if history is not None:
        first_acc = history['train_accuracy'][0]
        last_acc = history['train_accuracy'][-1]
        if last_acc > first_acc:
            print(f"   ✓ Accuracy: {first_acc * 100:.1f}% → {last_acc * 100:.1f}% (increased)")
            passed += 1
        else:
            print(f"   ✗ FAILED: {first_acc * 100:.1f}% → {last_acc * 100:.1f}%")
    else:
        print(f"   ✗ FAILED: no history")
    
    # ---- Test 6: Learning rate decays ----
    print("\n6️⃣  Learning rate decays each epoch")
    
    total += 1
    if history is not None:
        lrs = history['learning_rate']
        all_decreasing = all(lrs[i] > lrs[i + 1] for i in range(len(lrs) - 1))
        expected_last = 0.1 * (0.95 ** 4)  # After 5 epochs, LR used in epoch 5
        if all_decreasing and abs(lrs[-1] - expected_last) < 1e-6:
            print(f"   ✓ LR: {lrs[0]:.6f} → {lrs[-1]:.6f} (decaying by {trainer.lr_decay}x)")
            passed += 1
        else:
            print(f"   ✗ FAILED: lrs={lrs}, expected last≈{expected_last:.6f}")
    else:
        print(f"   ✗ FAILED: no history")
    
    # ---- Test 7: Validation metrics are computed ----
    print("\n7️⃣  Validation metrics are computed each epoch")
    
    total += 1
    if history is not None:
        val_losses = history['val_loss']
        val_accs = history['val_accuracy']
        all_valid = (all(v > 0 for v in val_losses) and
                     all(0.0 <= a <= 1.0 for a in val_accs))
        if all_valid:
            print(f"   ✓ Val loss range: [{min(val_losses):.4f}, {max(val_losses):.4f}]")
            print(f"     Val acc range:  [{min(val_accs) * 100:.1f}%, {max(val_accs) * 100:.1f}%]")
            passed += 1
        else:
            print(f"   ✗ FAILED")
    else:
        print(f"   ✗ FAILED: no history")
    
    # ---- Test 8: Early stopping triggers ----
    print("\n8️⃣  Early stopping triggers correctly")
    
    net2 = NeuralNetwork(input_size=input_size, hidden_sizes=[32, 16],
                         num_classes=num_classes, seed=42)
    trainer2 = Trainer(net2, learning_rate=0.1, lr_decay=0.95,
                       batch_size=32, epochs=100, patience=3, seed=42)
    
    history2 = trainer2.train(X_train, y_train, X_val, y_val)
    
    total += 1
    epochs_trained = len(history2['train_loss'])
    if epochs_trained < 100:
        print(f"   ✓ Early stopping triggered at epoch {epochs_trained} (< 100)")
        passed += 1
    else:
        print(f"   ✗ FAILED: trained all 100 epochs (early stopping didn't trigger)")
    
    # ---- Test 9: No NaN in history ----
    print("\n9️⃣  No NaN in any history values")
    
    total += 1
    all_clean = True
    for key, values in history2.items():
        if any(np.isnan(v) for v in values):
            all_clean = False
            break
    
    if all_clean:
        print(f"   ✓ All history values clean (no NaN)")
        passed += 1
    else:
        print(f"   ✗ FAILED: NaN found in history")
    
    # ---- Test 10: Evaluate helper works ----
    print("\n🔟  Evaluate helper returns valid results")
    
    total += 1
    loss, acc = trainer2._evaluate(X_val, y_val)
    if isinstance(loss, float) and 0.0 <= acc <= 1.0 and not np.isnan(loss):
        print(f"   ✓ Val loss: {loss:.4f}, Val acc: {acc * 100:.1f}%")
        passed += 1
    else:
        print(f"   ✗ FAILED: loss={loss}, acc={acc}")
    
    # ---- Summary ----
    print(f"\n{'=' * 60}")
    if passed == total:
        print(f"✅ All {total} tests passed!")
    else:
        print(f"❌ {passed}/{total} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    test_trainer()
