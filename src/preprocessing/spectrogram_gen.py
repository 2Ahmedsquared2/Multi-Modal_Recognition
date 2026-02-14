"""
Spectrogram Generation Module
Converts audio waveforms to mel-spectrograms for neural network input
"""

import numpy as np
import librosa
import librosa.display
from pathlib import Path
from typing import Tuple, Optional, Dict
from tqdm import tqdm
from scipy.ndimage import zoom
import warnings
warnings.filterwarnings('ignore')


class SpectrogramGenerator:
    """
    Generate mel-spectrograms from audio waveforms
    
    Handles:
    - Short-Time Fourier Transform (STFT)
    - Mel-frequency scaling (perceptual alignment)
    - Normalization and preprocessing
    - Batch spectrogram generation
    - Caching for efficient training
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        target_shape: Tuple[int, int] = (64, 64),
        normalize: bool = True
    ):
        """
        Initialize SpectrogramGenerator
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 22050)
            n_fft: FFT window size (default: 2048)
            hop_length: Number of samples between successive frames (default: 512)
            n_mels: Number of mel bands (default: 128)
            target_shape: Output spectrogram shape (height, width) (default: (64, 64))
            normalize: Normalize spectrograms to [0, 1] range (default: True)
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.target_shape = target_shape
        self.normalize = normalize
        
        print(f"🎼 SpectrogramGenerator initialized:")
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   FFT size: {n_fft}")
        print(f"   Hop length: {hop_length}")
        print(f"   Mel bands: {n_mels}")
        print(f"   Target shape: {target_shape}")
    
    def audio_to_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio waveform to mel-spectrogram
        
        Args:
            audio: Audio waveform (1D numpy array)
            
        Returns:
            Mel-spectrogram (2D numpy array) with shape target_shape
        """
        try:
            # Generate mel-spectrogram
            # This applies STFT + mel-frequency scaling
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=0,
                fmax=self.sample_rate // 2
            )
            
            # Convert to log scale (dB)
            # Human hearing is logarithmic, not linear
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Resize to target shape
            height_ratio = self.target_shape[0] / mel_spec_db.shape[0]
            width_ratio = self.target_shape[1] / mel_spec_db.shape[1]
            
            mel_spec_resized = zoom(mel_spec_db, (height_ratio, width_ratio), order=1)
            
            # Normalize to [0, 1] if requested
            if self.normalize:
                mel_spec_resized = self._normalize_spectrogram(mel_spec_resized)
            
            return mel_spec_resized
            
        except Exception as e:
            print(f"❌ Error generating spectrogram: {e}")
            # Return zeros if generation fails
            return np.zeros(self.target_shape)
    
    def _normalize_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Normalize spectrogram to [0, 1] range
        
        Args:
            spectrogram: Input spectrogram
            
        Returns:
            Normalized spectrogram
        """
        min_val = spectrogram.min()
        max_val = spectrogram.max()
        
        if max_val - min_val > 0:
            return (spectrogram - min_val) / (max_val - min_val)
        else:
            # Constant-value spectrogram (e.g. silent audio) — return zeros
            # to stay in the expected [0, 1] range
            return np.zeros_like(spectrogram)
    
    def batch_audio_to_spectrograms(
        self,
        audio_batch: np.ndarray
    ) -> np.ndarray:
        """
        Convert batch of audio waveforms to spectrograms
        
        Args:
            audio_batch: Batch of audio waveforms, shape (batch_size, n_samples)
            
        Returns:
            Batch of spectrograms, shape (batch_size, height, width)
        """
        spectrograms = []
        
        for audio in audio_batch:
            spec = self.audio_to_spectrogram(audio)
            spectrograms.append(spec)
        
        return np.array(spectrograms)
    
    def generate_and_save_dataset(
        self,
        audio_loader,
        output_dir: str = "data/spectrograms",
        save_format: str = "npz"
    ):
        """
        Pre-generate spectrograms for entire dataset and save to disk
        
        Args:
            audio_loader: AudioLoader instance with dataset
            output_dir: Directory to save spectrograms
            save_format: Format to save ('npz' or 'npy')
            
        Returns:
            Dictionary with paths and metadata
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎨 Generating spectrograms for {len(audio_loader.file_paths)} audio files...")
        print(f"   Output directory: {output_path}")
        print(f"   Target shape: {self.target_shape}")
        
        all_spectrograms = []
        all_labels = []
        
        # Generate spectrograms with progress bar
        for idx in tqdm(range(len(audio_loader.file_paths)), desc="Generating"):
            # Load audio (no augmentation for pre-generated)
            audio = audio_loader.load_audio(audio_loader.file_paths[idx])
            
            # Generate spectrogram
            spec = self.audio_to_spectrogram(audio)
            
            all_spectrograms.append(spec)
            all_labels.append(audio_loader.labels[idx])
        
        # Convert to numpy arrays
        all_spectrograms = np.array(all_spectrograms)
        all_labels = np.array(all_labels)
        
        # Save to disk
        if save_format == "npz":
            save_path = output_path / "spectrograms.npz"
            np.savez_compressed(
                save_path,
                spectrograms=all_spectrograms,
                labels=all_labels,
                class_names=audio_loader.class_names,
                target_shape=self.target_shape
            )
            print(f"✅ Saved {len(all_spectrograms)} spectrograms to: {save_path}")
            print(f"   File size: {save_path.stat().st_size / (1024*1024):.2f} MB")
        else:
            spec_path = output_path / "spectrograms.npy"
            labels_path = output_path / "labels.npy"
            np.save(spec_path, all_spectrograms)
            np.save(labels_path, all_labels)
            print(f"✅ Saved spectrograms to: {spec_path}")
            print(f"✅ Saved labels to: {labels_path}")
        
        # Print statistics
        print(f"\n📊 Spectrogram dataset statistics:")
        print(f"   Total samples: {len(all_spectrograms)}")
        print(f"   Spectrogram shape: {all_spectrograms.shape}")
        print(f"   Labels shape: {all_labels.shape}")
        print(f"   Value range: [{all_spectrograms.min():.3f}, {all_spectrograms.max():.3f}]")
        print(f"   Memory size: {all_spectrograms.nbytes / (1024*1024):.2f} MB")
        
        return {
            'output_path': str(output_path),
            'num_samples': len(all_spectrograms),
            'shape': all_spectrograms.shape,
            'save_format': save_format
        }
    
    def load_pregenerated_dataset(
        self,
        data_dir: str = "data/spectrograms",
        filename: str = "spectrograms.npz"
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Load pre-generated spectrogram dataset from disk
        
        Args:
            data_dir: Directory containing saved spectrograms
            filename: Name of the saved file
            
        Returns:
            Tuple of (spectrograms, labels, class_names)
        """
        file_path = Path(data_dir) / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Spectrogram file not found: {file_path}")
        
        print(f"📂 Loading pre-generated spectrograms from: {file_path}")
        
        data = np.load(file_path)
        spectrograms = data['spectrograms']
        labels = data['labels']
        class_names = data['class_names'].tolist()
        
        print(f"✅ Loaded {len(spectrograms)} spectrograms")
        print(f"   Shape: {spectrograms.shape}")
        print(f"   Classes: {class_names}")
        
        return spectrograms, labels, class_names
    
    def get_config(self) -> Dict:
        """Get generator configuration"""
        return {
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'n_mels': self.n_mels,
            'target_shape': self.target_shape,
            'normalize': self.normalize
        }


def test_spectrogram_generator():
    """Test the SpectrogramGenerator with sample data"""
    
    print("=" * 60)
    print("Testing SpectrogramGenerator")
    print("=" * 60)
    
    try:
        # Import AudioLoader
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from audio_loader import AudioLoader
        
        # Initialize audio loader
        print("\n1️⃣ Initializing AudioLoader...")
        audio_loader = AudioLoader(
            data_dir="data/raw/organized",
            sample_rate=22050,
            duration=2.0,
            augment=False  # No augmentation for testing
        )
        
        # Test with 64x64 (fast, MVP)
        print("\n2️⃣ Testing with 64x64 spectrograms...")
        spec_gen_64 = SpectrogramGenerator(
            sample_rate=22050,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            target_shape=(64, 64),
            normalize=True
        )
        
        # Load a sample audio file
        print("\n3️⃣ Loading sample audio...")
        audio = audio_loader.load_audio(audio_loader.file_paths[0])
        print(f"   Audio shape: {audio.shape}")
        print(f"   Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
        
        # Generate spectrogram
        print("\n4️⃣ Generating spectrogram...")
        spec = spec_gen_64.audio_to_spectrogram(audio)
        print(f"   Spectrogram shape: {spec.shape}")
        print(f"   Spectrogram range: [{spec.min():.3f}, {spec.max():.3f}]")
        
        # Test batch generation
        print("\n5️⃣ Testing batch generation...")
        audio_batch, _ = audio_loader.load_batch([0, 1, 2], augment=False)
        spec_batch = spec_gen_64.batch_audio_to_spectrograms(audio_batch)
        print(f"   Batch audio shape: {audio_batch.shape}")
        print(f"   Batch spectrogram shape: {spec_batch.shape}")
        
        # Test with 128x128 (higher quality)
        print("\n6️⃣ Testing with 128x128 spectrograms...")
        spec_gen_128 = SpectrogramGenerator(
            sample_rate=22050,
            target_shape=(128, 128)
        )
        spec_128 = spec_gen_128.audio_to_spectrogram(audio)
        print(f"   Spectrogram shape: {spec_128.shape}")
        
        # Test pre-generation (small subset)
        print("\n7️⃣ Testing pre-generation (first 10 samples)...")
        # Create a temporary small loader for testing
        test_loader = AudioLoader(
            data_dir="data/raw/organized",
            sample_rate=22050,
            duration=2.0,
            augment=False
        )
        # Limit to first 10 files for quick test
        test_loader.file_paths = test_loader.file_paths[:10]
        test_loader.labels = test_loader.labels[:10]
        
        result = spec_gen_64.generate_and_save_dataset(
            audio_loader=test_loader,
            output_dir="data/spectrograms_test",
            save_format="npz"
        )
        
        print(f"\n✅ Pre-generation test complete!")
        print(f"   Generated: {result['num_samples']} spectrograms")
        
        # Test loading pre-generated
        print("\n8️⃣ Testing loading pre-generated spectrograms...")
        specs, labels, classes = spec_gen_64.load_pregenerated_dataset(
            data_dir="data/spectrograms_test",
            filename="spectrograms.npz"
        )
        print(f"   Loaded shape: {specs.shape}")
        print(f"   Labels: {labels}")
        
        print("\n✅ All tests passed!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\n💡 Make sure the audio dataset is downloaded.")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)


if __name__ == "__main__":
    test_spectrogram_generator()
