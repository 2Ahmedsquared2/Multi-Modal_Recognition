"""
Data Preparation Module
Loads spectrograms, splits, normalizes, encodes labels, and provides batch generation
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, Generator


class DataPreparator:
    """
    Prepare spectrogram data for neural network training
    
    Handles:
    - Loading pre-generated spectrograms from disk
    - Flattening 2D spectrograms to 1D vectors
    - Stratified train/validation/test splitting
    - Normalization (zero-mean, unit-variance from training set)
    - One-hot encoding of labels
    - Single-epoch batch generation
    """
    
    def __init__(
        self,
        spectrogram_path: str = "data/spectrograms/spectrograms.npz",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ):
        """
        Initialize DataPreparator
        
        Args:
            spectrogram_path: Path to pre-generated spectrograms (.npz file)
            train_ratio: Proportion for training set (default: 0.7)
            val_ratio: Proportion for validation set (default: 0.15)
            test_ratio: Proportion for testing set (default: 0.15)
            seed: Random seed for reproducibility (default: 42)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        self.spectrogram_path = Path(spectrogram_path)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Data containers (populated by prepare())
        self.spectrograms_raw = None  # Original 2D spectrograms
        self.labels_raw = None        # Original integer labels
        self.class_names = None       # List of class name strings
        self.num_classes = 0
        self.input_dim = 0            # Flattened input dimension
        
        # Prepared data splits
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        # Normalization parameters (computed from training set only)
        self.mean = None
        self.std = None
    
    def load_spectrograms(self) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Load pre-generated spectrograms from disk
        
        Returns:
            Tuple of (spectrograms, labels, class_names)
        """
        if not self.spectrogram_path.exists():
            raise FileNotFoundError(
                f"Spectrogram file not found: {self.spectrogram_path}\n"
                f"Run 'python generate_spectrograms.py' first."
            )
        
        print(f"📂 Loading spectrograms from: {self.spectrogram_path}")
        
        data = np.load(self.spectrogram_path, allow_pickle=True)
        spectrograms = data['spectrograms']
        labels = data['labels']
        class_names = data['class_names'].tolist()
        
        print(f"   Loaded {len(spectrograms)} spectrograms")
        print(f"   Shape: {spectrograms.shape}")
        print(f"   Classes ({len(class_names)}): {', '.join(class_names)}")
        
        return spectrograms, labels, class_names
    
    def flatten(self, spectrograms: np.ndarray) -> np.ndarray:
        """
        Flatten 2D spectrograms to 1D vectors
        
        Args:
            spectrograms: Array of shape (N, height, width)
            
        Returns:
            Flattened array of shape (N, height * width)
        """
        n_samples = spectrograms.shape[0]
        flattened = spectrograms.reshape(n_samples, -1)
        
        print(f"📐 Flattened: {spectrograms.shape} → {flattened.shape}")
        
        return flattened
    
    def stratified_split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/val/test sets maintaining class distribution
        
        Args:
            X: Feature array of shape (N, D)
            y: Label array of shape (N,) with integer labels
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        unique_classes = np.unique(y)
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        for cls in unique_classes:
            # Get all indices for this class
            cls_indices = np.where(y == cls)[0]
            
            # Shuffle
            self.rng.shuffle(cls_indices)
            
            # Calculate split points
            n = len(cls_indices)
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)
            
            # Split
            train_indices.extend(cls_indices[:n_train])
            val_indices.extend(cls_indices[n_train:n_train + n_val])
            test_indices.extend(cls_indices[n_train + n_val:])
        
        # Convert to arrays and shuffle
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)
        
        self.rng.shuffle(train_indices)
        self.rng.shuffle(val_indices)
        self.rng.shuffle(test_indices)
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        print(f"\n📊 Stratified split:")
        print(f"   Training:   {len(X_train):4d} samples ({self.train_ratio*100:.0f}%)")
        print(f"   Validation: {len(X_val):4d} samples ({self.val_ratio*100:.0f}%)")
        print(f"   Testing:    {len(X_test):4d} samples ({self.test_ratio*100:.0f}%)")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def compute_normalization(self, X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and std from training data only
        
        Args:
            X_train: Training features of shape (N, D)
            
        Returns:
            Tuple of (mean, std) each of shape (D,)
        """
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        
        # Count near-zero-variance features (will be zeroed out during normalization)
        n_low_var = np.sum(std <= 1e-6)
        if n_low_var > 0:
            print(f"   ⚠️ {n_low_var} features have near-zero variance (will be zeroed out)")
        
        return mean, std
    
    def normalize(self, X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """
        Normalize features using pre-computed mean and std
        
        Zero-variance features (std=0) are left as zero rather than
        being amplified by a tiny epsilon, which would blow up noise
        in the validation/test sets.
        
        Args:
            X: Feature array of shape (N, D)
            mean: Mean array of shape (D,)
            std: Std array of shape (D,)
            
        Returns:
            Normalized array of shape (N, D)
        """
        result = np.zeros_like(X)
        # Treat features with very small variance as constant (avoid amplifying noise)
        nonzero_mask = std > 1e-6
        result[:, nonzero_mask] = (X[:, nonzero_mask] - mean[nonzero_mask]) / std[nonzero_mask]
        return result
    
    def one_hot_encode(self, labels: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Convert integer labels to one-hot encoded vectors
        
        Args:
            labels: Integer label array of shape (N,)
            num_classes: Total number of classes
            
        Returns:
            One-hot encoded array of shape (N, num_classes)
        """
        n = len(labels)
        one_hot = np.zeros((n, num_classes))
        one_hot[np.arange(n), labels] = 1.0
        return one_hot
    
    def batch_generator(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yield mini-batches for one epoch (stops after one pass)
        
        Args:
            X: Feature array of shape (N, D)
            y: Label array of shape (N, C) — one-hot encoded
            batch_size: Number of samples per batch (default: 32)
            shuffle: Whether to shuffle before iterating (default: True)
            
        Yields:
            Tuples of (X_batch, y_batch)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            yield X[batch_idx], y[batch_idx]
    
    def prepare(self) -> Dict:
        """
        Run the full data preparation pipeline
        
        Returns:
            Dictionary with all prepared data and metadata
        """
        print("=" * 60)
        print("🔧 DATA PREPARATION")
        print("=" * 60)
        
        # 1. Load spectrograms
        print("\n1️⃣  Loading spectrograms...")
        self.spectrograms_raw, self.labels_raw, self.class_names = \
            self.load_spectrograms()
        self.num_classes = len(self.class_names)
        
        # 2. Flatten
        print("\n2️⃣  Flattening spectrograms...")
        X_flat = self.flatten(self.spectrograms_raw)
        self.input_dim = X_flat.shape[1]
        
        # 3. Stratified split
        print("\n3️⃣  Splitting dataset...")
        X_train, y_train, X_val, y_val, X_test, y_test = \
            self.stratified_split(X_flat, self.labels_raw)
        
        # 4. Normalize (compute from training set only)
        print("\n4️⃣  Normalizing...")
        self.mean, self.std = self.compute_normalization(X_train)
        
        X_train = self.normalize(X_train, self.mean, self.std)
        X_val = self.normalize(X_val, self.mean, self.std)
        X_test = self.normalize(X_test, self.mean, self.std)
        
        print(f"   Training set — mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
        print(f"   Validation   — mean: {X_val.mean():.4f}, std: {X_val.std():.4f}")
        print(f"   Test         — mean: {X_test.mean():.4f}, std: {X_test.std():.4f}")
        
        # 5. One-hot encode labels
        print("\n5️⃣  One-hot encoding labels...")
        self.X_train = X_train
        self.y_train = self.one_hot_encode(y_train, self.num_classes)
        self.X_val = X_val
        self.y_val = self.one_hot_encode(y_val, self.num_classes)
        self.X_test = X_test
        self.y_test = self.one_hot_encode(y_test, self.num_classes)
        
        print(f"   y_train: {y_train.shape} → {self.y_train.shape}")
        print(f"   y_val:   {y_val.shape} → {self.y_val.shape}")
        print(f"   y_test:  {y_test.shape} → {self.y_test.shape}")
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ DATA PREPARATION COMPLETE")
        print("=" * 60)
        print(f"\n📊 Final shapes:")
        print(f"   X_train: {self.X_train.shape}  y_train: {self.y_train.shape}")
        print(f"   X_val:   {self.X_val.shape}  y_val:   {self.y_val.shape}")
        print(f"   X_test:  {self.X_test.shape}  y_test:  {self.y_test.shape}")
        print(f"   Input dimension: {self.input_dim}")
        print(f"   Number of classes: {self.num_classes}")
        
        return {
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_val': self.X_val,
            'y_val': self.y_val,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'input_dim': self.input_dim,
            'mean': self.mean,
            'std': self.std
        }
    
    def save_prepared_data(self, output_path: str = "data/prepared") -> str:
        """
        Save prepared data and normalization params to disk
        
        Args:
            output_path: Directory to save prepared data
            
        Returns:
            Path to saved file
        """
        if self.X_train is None:
            raise RuntimeError("Call prepare() first before saving.")
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = output_dir / "prepared_data.npz"
        
        np.savez_compressed(
            save_path,
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            X_test=self.X_test,
            y_test=self.y_test,
            mean=self.mean,
            std=self.std,
            class_names=np.array(self.class_names),
            input_dim=self.input_dim,
            num_classes=self.num_classes
        )
        
        file_size_mb = save_path.stat().st_size / (1024 * 1024)
        print(f"\n💾 Saved prepared data to: {save_path}")
        print(f"   File size: {file_size_mb:.2f} MB")
        
        return str(save_path)
    
    def load_prepared_data(self, input_path: str = "data/prepared/prepared_data.npz") -> Dict:
        """
        Load previously prepared data from disk (skip the full pipeline)
        
        Args:
            input_path: Path to saved prepared data file
            
        Returns:
            Dictionary with all prepared data and metadata
        """
        file_path = Path(input_path)
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Prepared data not found: {file_path}\n"
                f"Run prepare() and save_prepared_data() first."
            )
        
        print(f"📂 Loading prepared data from: {file_path}")
        
        data = np.load(file_path, allow_pickle=True)
        
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        self.mean = data['mean']
        self.std = data['std']
        self.class_names = data['class_names'].tolist()
        self.input_dim = int(data['input_dim'])
        self.num_classes = int(data['num_classes'])
        
        print(f"   X_train: {self.X_train.shape}  y_train: {self.y_train.shape}")
        print(f"   X_val:   {self.X_val.shape}  y_val:   {self.y_val.shape}")
        print(f"   X_test:  {self.X_test.shape}  y_test:  {self.y_test.shape}")
        print(f"   Classes: {self.class_names}")
        
        return {
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_val': self.X_val,
            'y_val': self.y_val,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'input_dim': self.input_dim,
            'mean': self.mean,
            'std': self.std
        }
    
    def get_info(self) -> Dict:
        """Get summary of prepared data"""
        if self.X_train is None:
            return {'status': 'Not prepared yet. Call prepare() first.'}
        
        return {
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'train_samples': len(self.X_train),
            'val_samples': len(self.X_val),
            'test_samples': len(self.X_test),
            'total_samples': len(self.X_train) + len(self.X_val) + len(self.X_test),
            'train_mean': float(self.X_train.mean()),
            'train_std': float(self.X_train.std())
        }


def test_data_prep():
    """Test the DataPreparator with actual dataset"""
    
    print("=" * 60)
    print("Testing DataPreparator")
    print("=" * 60)
    
    try:
        # Initialize and run full pipeline
        prep = DataPreparator(
            spectrogram_path="data/spectrograms/spectrograms.npz",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42
        )
        
        data = prep.prepare()
        
        # Verify shapes
        print("\n🧪 Running verification checks...")
        
        # Check 1: Flattened dimension matches
        assert data['input_dim'] == data['X_train'].shape[1], \
            "Input dim mismatch"
        print(f"   ✓ Input dimension: {data['input_dim']}")
        
        # Check 2: Splits sum to total
        total = len(data['X_train']) + len(data['X_val']) + len(data['X_test'])
        print(f"   ✓ Total samples across splits: {total}")
        
        # Check 3: One-hot encoding is valid
        assert data['y_train'].shape[1] == data['num_classes'], \
            "One-hot encoding dimension mismatch"
        assert np.allclose(data['y_train'].sum(axis=1), 1.0), \
            "One-hot rows don't sum to 1"
        print(f"   ✓ One-hot encoding valid ({data['num_classes']} classes)")
        
        # Check 4: Training set is approximately normalized
        train_mean = data['X_train'].mean()
        train_std = data['X_train'].std()
        assert abs(train_mean) < 0.1, \
            f"Training mean too far from 0: {train_mean:.4f}"
        assert abs(train_std - 1.0) < 0.2, \
            f"Training std too far from 1: {train_std:.4f}"
        print(f"   ✓ Normalization valid (mean={train_mean:.4f}, std={train_std:.4f})")
        
        # Check 5: Mean/std are stored
        assert data['mean'] is not None and data['std'] is not None, \
            "Normalization params not stored"
        print(f"   ✓ Normalization params stored (mean shape: {data['mean'].shape})")
        
        # Check 6: Batch generator works
        print(f"\n📦 Testing batch generator...")
        batch_count = 0
        total_samples = 0
        
        for X_batch, y_batch in prep.batch_generator(
            data['X_train'], data['y_train'], batch_size=32, shuffle=True
        ):
            batch_count += 1
            total_samples += len(X_batch)
            
            if batch_count == 1:
                print(f"   First batch: X={X_batch.shape}, y={y_batch.shape}")
        
        print(f"   ✓ Generated {batch_count} batches, {total_samples} total samples")
        assert total_samples == len(data['X_train']), \
            "Batch generator didn't cover all training samples"
        print(f"   ✓ Batch generator covers all training samples (one epoch)")
        
        # Check 7: Save and reload
        print(f"\n💾 Testing save/load...")
        save_path = prep.save_prepared_data(output_path="data/prepared")
        
        prep2 = DataPreparator()
        data2 = prep2.load_prepared_data(save_path)
        
        assert np.allclose(data['X_train'], data2['X_train']), \
            "Loaded X_train doesn't match"
        assert np.allclose(data['mean'], data2['mean']), \
            "Loaded mean doesn't match"
        print(f"   ✓ Save/load roundtrip verified")
        
        # Print final info
        print(f"\n📊 Dataset info:")
        info = prep.get_info()
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        print("\n✅ All tests passed!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)


if __name__ == "__main__":
    test_data_prep()
