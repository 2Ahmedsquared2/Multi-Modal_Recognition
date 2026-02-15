"""
Image Preprocessor — converts raw images to fixed-size numpy arrays
for neural network input.

Parallel to SpectrogramGenerator for audio:
    SpectrogramGenerator:  audio waveform → mel-spectrogram → 2D array
    ImagePreprocessor:     image file     → resize + grayscale → 2D array

Both produce a (target_shape) 2D numpy array normalized to [0, 1],
which gets flattened and fed into the same Dense neural network.
"""

import io
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List

from PIL import Image
from tqdm import tqdm


class ImagePreprocessor:
    """
    Preprocess images for neural network classification.

    Pipeline:
        1. Load image from file path or bytes
        2. Convert to grayscale (single channel)
        3. Resize to target_shape
        4. Normalize pixel values to [0, 1]

    The output matches the shape and value range of SpectrogramGenerator
    so both modalities can share the same network architecture.

    Additionally handles dataset generation and caching to `.npz` files,
    mirroring SpectrogramGenerator.generate_and_save_dataset().
    """

    def __init__(
        self,
        target_shape: Tuple[int, int] = (64, 64),
        normalize: bool = True,
    ):
        """
        Args:
            target_shape: Output image size (height, width). Default (64, 64).
            normalize:    Normalize to [0, 1] range. Default True.
        """
        self.target_shape = target_shape
        self.normalize = normalize

        print(f"🖼️  ImagePreprocessor initialized:")
        print(f"   Target shape: {target_shape}")
        print(f"   Normalize: {normalize}")

    # ------------------------------------------------------------------
    # Core conversion
    # ------------------------------------------------------------------

    def image_to_array(self, image: Image.Image) -> np.ndarray:
        """
        Convert a PIL Image to a preprocessed 2D numpy array.

        Args:
            image: PIL Image (any mode — RGB, RGBA, L, etc.)

        Returns:
            2D numpy array of shape target_shape, values in [0, 1].
        """
        # Convert to grayscale
        gray = image.convert("L")

        # Resize to target shape using high-quality resampling
        resized = gray.resize(
            (self.target_shape[1], self.target_shape[0]),  # PIL uses (width, height)
            Image.Resampling.LANCZOS,
        )

        # Convert to numpy
        arr = np.array(resized, dtype=np.float64)

        # Normalize to [0, 1]
        if self.normalize:
            arr = arr / 255.0

        return arr

    # ------------------------------------------------------------------
    # Single-file loaders
    # ------------------------------------------------------------------

    def load_from_path(self, path: str) -> np.ndarray:
        """
        Load an image from a file path and preprocess it.

        Args:
            path: Path to the image file.

        Returns:
            2D numpy array of shape target_shape.
        """
        img = Image.open(path)
        return self.image_to_array(img)

    def load_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """
        Load an image from raw bytes and preprocess it.

        Args:
            image_bytes: Raw image file bytes (PNG, JPEG, etc.)

        Returns:
            2D numpy array of shape target_shape.
        """
        img = Image.open(io.BytesIO(image_bytes))
        return self.image_to_array(img)

    # ------------------------------------------------------------------
    # Batch loading
    # ------------------------------------------------------------------

    def batch_load(self, paths: list) -> np.ndarray:
        """
        Load and preprocess a batch of images from file paths.

        Args:
            paths: List of image file paths.

        Returns:
            3D numpy array of shape (batch_size, *target_shape).
        """
        arrays = [self.load_from_path(p) for p in paths]
        return np.array(arrays)

    def batch_images_to_arrays(self, images: List[Image.Image]) -> np.ndarray:
        """
        Convert a batch of PIL Images to preprocessed numpy arrays.

        Args:
            images: List of PIL Image objects.

        Returns:
            3D numpy array of shape (batch_size, *target_shape).
        """
        arrays = [self.image_to_array(img) for img in images]
        return np.array(arrays)

    # ------------------------------------------------------------------
    # Dataset generation & caching (mirrors SpectrogramGenerator)
    # ------------------------------------------------------------------

    def generate_and_save_dataset(
        self,
        image_loader,
        output_dir: str = "data/spectrograms",
        save_format: str = "npz",
    ) -> Dict:
        """
        Pre-generate preprocessed image arrays for entire dataset and save to disk.

        This mirrors SpectrogramGenerator.generate_and_save_dataset() — the
        output .npz has the same keys (spectrograms, labels, class_names,
        target_shape) so DataPreparator can consume it without changes.

        Args:
            image_loader: ImageLoader instance with scanned dataset.
            output_dir: Directory to save preprocessed images.
            save_format: Format to save ('npz' or 'npy').

        Returns:
            Dictionary with output path, num_samples, shape, save_format.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n🖼️  Generating preprocessed images for {len(image_loader.file_paths)} files...")
        print(f"   Output directory: {output_path}")
        print(f"   Target shape: {self.target_shape}")

        all_arrays: List[np.ndarray] = []
        all_labels: List[int] = []
        skipped = 0

        for idx in tqdm(range(len(image_loader.file_paths)), desc="Processing images"):
            try:
                img = image_loader.load_image(image_loader.file_paths[idx])
                arr = self.image_to_array(img)
                all_arrays.append(arr)
                all_labels.append(image_loader.labels[idx])
            except Exception as e:
                print(f"   ⚠️  Skipping {image_loader.file_paths[idx]}: {e}")
                skipped += 1

        # Convert to numpy arrays
        all_arrays_np = np.array(all_arrays)
        all_labels_np = np.array(all_labels)

        # Save to disk — use the SAME key name "spectrograms" so
        # DataPreparator.load_spectrograms() works for both modalities.
        if save_format == "npz":
            save_path = output_path / "spectrograms.npz"
            np.savez_compressed(
                save_path,
                spectrograms=all_arrays_np,
                labels=all_labels_np,
                class_names=image_loader.class_names,
                target_shape=self.target_shape,
            )
            print(f"✅ Saved {len(all_arrays_np)} preprocessed images to: {save_path}")
            print(f"   File size: {save_path.stat().st_size / (1024*1024):.2f} MB")
        else:
            spec_path = output_path / "spectrograms.npy"
            labels_path = output_path / "labels.npy"
            np.save(spec_path, all_arrays_np)
            np.save(labels_path, all_labels_np)
            print(f"✅ Saved preprocessed images to: {spec_path}")
            print(f"✅ Saved labels to: {labels_path}")

        if skipped > 0:
            print(f"   ⚠️  Skipped {skipped} files due to errors")

        # Print statistics
        print(f"\n📊 Image dataset statistics:")
        print(f"   Total samples: {len(all_arrays_np)}")
        print(f"   Array shape: {all_arrays_np.shape}")
        print(f"   Labels shape: {all_labels_np.shape}")
        print(f"   Value range: [{all_arrays_np.min():.3f}, {all_arrays_np.max():.3f}]")
        print(f"   Memory size: {all_arrays_np.nbytes / (1024*1024):.2f} MB")

        return {
            "output_path": str(output_path),
            "num_samples": len(all_arrays_np),
            "shape": all_arrays_np.shape,
            "save_format": save_format,
            "skipped": skipped,
        }

    def load_pregenerated_dataset(
        self,
        data_dir: str = "data/spectrograms",
        filename: str = "spectrograms.npz",
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Load pre-generated preprocessed image dataset from disk.

        Args:
            data_dir: Directory containing saved arrays.
            filename: Name of the saved file.

        Returns:
            Tuple of (image_arrays, labels, class_names).
        """
        file_path = Path(data_dir) / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Preprocessed image file not found: {file_path}")

        print(f"📂 Loading pre-generated image arrays from: {file_path}")

        data = np.load(file_path, allow_pickle=True)
        arrays = data["spectrograms"]
        labels = data["labels"]
        class_names = data["class_names"].tolist()

        print(f"✅ Loaded {len(arrays)} preprocessed images")
        print(f"   Shape: {arrays.shape}")
        print(f"   Classes: {class_names}")

        return arrays, labels, class_names

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        """Return preprocessor configuration."""
        return {
            "target_shape": self.target_shape,
            "normalize": self.normalize,
        }
