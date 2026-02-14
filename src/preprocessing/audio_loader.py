"""
Audio Data Loading Module
Loads audio files, applies augmentation, and prepares data for neural network
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class AudioLoader:
    """
    Audio data loader with augmentation support
    
    Handles:
    - Audio file loading and validation
    - Data augmentation (pitch shift, time stretch, noise)
    - Batch generation for training
    - Train/validation/test splits
    """
    
    def __init__(
        self,
        data_dir: str = "data/raw/organized",
        sample_rate: int = 22050,
        duration: float = 2.0,
        augment: bool = True,
        augment_prob: float = 0.5
    ):
        """
        Initialize AudioLoader
        
        Args:
            data_dir: Directory containing organized audio files (one folder per class)
            sample_rate: Target sample rate in Hz (default: 22050)
            duration: Audio duration in seconds (default: 2.0)
            augment: Whether to apply data augmentation (default: True)
            augment_prob: Probability of applying each augmentation (default: 0.5)
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.augment = augment
        self.augment_prob = augment_prob
        
        # Scan dataset
        self.class_names = []
        self.class_to_idx = {}
        self.file_paths = []
        self.labels = []
        
        self._scan_dataset()
    
    def _scan_dataset(self):
        """Scan directory structure and build file list"""
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        print(f"📂 Scanning dataset: {self.data_dir}")
        
        # Get all class directories
        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        if len(class_dirs) == 0:
            raise ValueError(f"No class directories found in {self.data_dir}")
        
        self.class_names = [d.name for d in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        print(f"✓ Found {len(self.class_names)} classes: {', '.join(self.class_names)}")
        
        # Scan audio files in each class
        for class_dir in class_dirs:
            class_name = class_dir.name
            class_idx = self.class_to_idx[class_name]
            
            # Find all .wav files
            audio_files = list(class_dir.glob("*.wav"))
            
            for audio_file in audio_files:
                self.file_paths.append(audio_file)
                self.labels.append(class_idx)
        
        print(f"✓ Found {len(self.file_paths)} audio files total")
        
        # Print class distribution
        print(f"\n📊 Class distribution:")
        for class_name in self.class_names:
            class_idx = self.class_to_idx[class_name]
            count = self.labels.count(class_idx)
            print(f"   {class_name:15s}: {count:4d} files")
    
    def load_audio(self, file_path: Path) -> np.ndarray:
        """
        Load audio file and normalize
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio waveform as numpy array
        """
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            # Ensure consistent length
            if len(audio) < self.n_samples:
                # Pad with zeros if too short
                audio = np.pad(audio, (0, self.n_samples - len(audio)), mode='constant')
            else:
                # Trim if too long
                audio = audio[:self.n_samples]
            
            return audio
            
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file {file_path}: {e}") from e
    
    def pitch_shift(self, audio: np.ndarray, n_steps: Optional[int] = None) -> np.ndarray:
        """
        Shift audio pitch
        
        Args:
            audio: Input audio waveform
            n_steps: Number of semitones to shift (default: random -2 to +2)
            
        Returns:
            Pitch-shifted audio
        """
        if n_steps is None:
            n_steps = np.random.randint(-2, 3)  # -2 to +2 semitones
        
        if n_steps == 0:
            return audio
        
        try:
            return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
        except Exception:
            return audio
    
    def time_stretch(self, audio: np.ndarray, rate: Optional[float] = None) -> np.ndarray:
        """
        Stretch/compress audio in time
        
        Args:
            audio: Input audio waveform
            rate: Stretch factor (default: random 0.8 to 1.2)
                  rate < 1.0 = slower (stretched)
                  rate > 1.0 = faster (compressed)
            
        Returns:
            Time-stretched audio
        """
        if rate is None:
            rate = np.random.uniform(0.8, 1.2)
        
        if rate == 1.0:
            return audio
        
        try:
            stretched = librosa.effects.time_stretch(audio, rate=rate)
            
            # Ensure consistent length
            if len(stretched) < self.n_samples:
                stretched = np.pad(stretched, (0, self.n_samples - len(stretched)), mode='constant')
            else:
                stretched = stretched[:self.n_samples]
            
            return stretched
        except Exception:
            return audio
    
    def add_noise(self, audio: np.ndarray, noise_level: Optional[float] = None) -> np.ndarray:
        """
        Add Gaussian noise to audio
        
        Args:
            audio: Input audio waveform
            noise_level: Noise standard deviation (default: random 0.001 to 0.005)
            
        Returns:
            Audio with added noise
        """
        if noise_level is None:
            noise_level = np.random.uniform(0.001, 0.005)
        
        noise = np.random.normal(0, noise_level, len(audio))
        return audio + noise
    
    def augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to audio
        
        Args:
            audio: Input audio waveform
            
        Returns:
            Augmented audio
        """
        if not self.augment:
            return audio
        
        # Randomly apply each augmentation
        if np.random.random() < self.augment_prob:
            audio = self.pitch_shift(audio)
        
        if np.random.random() < self.augment_prob:
            audio = self.time_stretch(audio)
        
        if np.random.random() < self.augment_prob:
            audio = self.add_noise(audio)
        
        return audio
    
    def load_batch(
        self,
        indices: List[int],
        augment: Optional[bool] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a batch of audio files
        
        Args:
            indices: List of file indices to load
            augment: Override augmentation setting (default: use self.augment)
            
        Returns:
            Tuple of (audio_batch, labels_batch)
            - audio_batch: shape (batch_size, n_samples)
            - labels_batch: shape (batch_size,)
        """
        if augment is None:
            augment = self.augment
        
        batch_audio = []
        batch_labels = []
        
        for idx in indices:
            # Load audio
            audio = self.load_audio(self.file_paths[idx])
            
            # Apply augmentation
            if augment:
                audio = self.augment_audio(audio)
            
            batch_audio.append(audio)
            batch_labels.append(self.labels[idx])
        
        return np.array(batch_audio), np.array(batch_labels)
    
    def train_val_test_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Split dataset into train/validation/test sets
        
        Args:
            train_ratio: Proportion for training (default: 0.7)
            val_ratio: Proportion for validation (default: 0.15)
            test_ratio: Proportion for testing (default: 0.15)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        # Use a local RNG to avoid polluting global random state
        rng = np.random.default_rng(seed)
        
        # Stratified split (maintain class distribution)
        train_indices = []
        val_indices = []
        test_indices = []
        
        for class_idx in range(len(self.class_names)):
            # Get all indices for this class
            class_indices = [i for i, label in enumerate(self.labels) if label == class_idx]
            
            # Shuffle
            rng.shuffle(class_indices)
            
            # Calculate split points
            n_samples = len(class_indices)
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            
            # Split
            train_indices.extend(class_indices[:n_train])
            val_indices.extend(class_indices[n_train:n_train + n_val])
            test_indices.extend(class_indices[n_train + n_val:])
        
        # Shuffle again
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        rng.shuffle(test_indices)
        
        print(f"\n📊 Dataset split:")
        print(f"   Training:   {len(train_indices):4d} samples ({train_ratio*100:.1f}%)")
        print(f"   Validation: {len(val_indices):4d} samples ({val_ratio*100:.1f}%)")
        print(f"   Testing:    {len(test_indices):4d} samples ({test_ratio*100:.1f}%)")
        
        return train_indices, val_indices, test_indices
    
    def get_batch_generator(
        self,
        indices: List[int],
        batch_size: int = 32,
        shuffle: bool = True,
        augment: Optional[bool] = None
    ):
        """
        Create batch generator for training
        
        Args:
            indices: List of indices to use
            batch_size: Number of samples per batch
            shuffle: Shuffle indices each epoch
            augment: Override augmentation setting
            
        Yields:
            Tuples of (audio_batch, labels_batch)
        """
        indices = list(indices)
        
        while True:
            if shuffle:
                np.random.shuffle(indices)
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                yield self.load_batch(batch_indices, augment=augment)
    
    def get_info(self) -> Dict:
        """Get dataset information"""
        return {
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'num_files': len(self.file_paths),
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'n_samples': self.n_samples,
            'augmentation': self.augment
        }


def test_loader():
    """Test the AudioLoader with sample data"""
    
    print("=" * 60)
    print("Testing AudioLoader")
    print("=" * 60)
    
    try:
        # Initialize loader
        loader = AudioLoader(
            data_dir="data/raw/organized",
            sample_rate=22050,
            duration=2.0,
            augment=True,
            augment_prob=0.5
        )
        
        # Print info
        info = loader.get_info()
        print(f"\n📊 Dataset Info:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Test loading a few files
        print(f"\n🔊 Testing audio loading...")
        for i in range(min(3, len(loader.file_paths))):
            audio = loader.load_audio(loader.file_paths[i])
            print(f"   File {i}: shape={audio.shape}, min={audio.min():.3f}, max={audio.max():.3f}")
        
        # Test augmentation
        if len(loader.file_paths) > 0:
            print(f"\n🎵 Testing augmentation...")
            audio = loader.load_audio(loader.file_paths[0])
            
            aug_audio = loader.augment_audio(audio.copy())
            print(f"   Original: shape={audio.shape}, mean={audio.mean():.3f}")
            print(f"   Augmented: shape={aug_audio.shape}, mean={aug_audio.mean():.3f}")
        
        # Test batch loading
        print(f"\n📦 Testing batch loading...")
        batch_audio, batch_labels = loader.load_batch([0, 1, 2])
        print(f"   Batch audio shape: {batch_audio.shape}")
        print(f"   Batch labels shape: {batch_labels.shape}")
        print(f"   Batch labels: {batch_labels}")
        
        # Test train/val/test split
        train_idx, val_idx, test_idx = loader.train_val_test_split()
        
        print("\n✅ All tests passed!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\n💡 Please run the download script first:")
        print("   python download_dataset.py")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)


if __name__ == "__main__":
    test_loader()
