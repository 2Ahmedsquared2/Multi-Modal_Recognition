"""
Image Data Loading Module
Loads image files, applies augmentation, and prepares data for neural network.

Parallel to AudioLoader for audio:
    AudioLoader:  scans audio dirs → loads .wav → augment → batches
    ImageLoader:  scans image dirs → loads .png/.jpg → augment → batches

Both produce file_paths + labels ready for the preprocessing step.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional

from PIL import Image, ImageEnhance, ImageFilter
import warnings

warnings.filterwarnings("ignore")

# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


class ImageLoader:
    """
    Image data loader with augmentation support.

    Handles:
    - Directory scanning for image files organized by class
    - Image loading and validation
    - Data augmentation (flip, rotation, brightness, contrast, noise)
    - Batch generation for training
    - Stratified train/validation/test splits
    """

    def __init__(
        self,
        data_dir: str = "data/image",
        augment: bool = True,
        augment_prob: float = 0.5,
    ):
        """
        Initialize ImageLoader.

        Args:
            data_dir: Directory containing organized images (one folder per class).
            augment: Whether to apply data augmentation (default: True).
            augment_prob: Probability of applying each augmentation (default: 0.5).
        """
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.augment_prob = augment_prob
        self.rng = np.random.default_rng(42)

        # Populated by _scan_dataset
        self.class_names: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.file_paths: List[Path] = []
        self.labels: List[int] = []

        self._scan_dataset()

    # ------------------------------------------------------------------
    # Dataset scanning
    # ------------------------------------------------------------------

    def _scan_dataset(self) -> None:
        """Scan directory structure and build file list."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        print(f"📂 Scanning image dataset: {self.data_dir}")

        # Get all class directories (sorted for deterministic ordering)
        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

        if len(class_dirs) == 0:
            raise ValueError(f"No class directories found in {self.data_dir}")

        self.class_names = [d.name for d in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        print(f"✓ Found {len(self.class_names)} classes: {', '.join(self.class_names)}")

        # Scan image files in each class directory
        for class_dir in class_dirs:
            class_name = class_dir.name
            class_idx = self.class_to_idx[class_name]

            image_files = sorted(
                f
                for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            )

            for image_file in image_files:
                self.file_paths.append(image_file)
                self.labels.append(class_idx)

        print(f"✓ Found {len(self.file_paths)} image files total")

        # Print class distribution
        print(f"\n📊 Class distribution:")
        for class_name in self.class_names:
            class_idx = self.class_to_idx[class_name]
            count = self.labels.count(class_idx)
            print(f"   {class_name:15s}: {count:4d} files")

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def load_image(self, file_path: Path) -> Image.Image:
        """
        Load an image file as a PIL Image.

        Args:
            file_path: Path to image file.

        Returns:
            PIL Image object.
        """
        try:
            img = Image.open(file_path)
            # Convert to RGB to handle palette, grayscale, RGBA consistently
            img = img.convert("RGB")
            return img
        except Exception as e:
            raise RuntimeError(f"Failed to load image file {file_path}: {e}") from e

    # ------------------------------------------------------------------
    # Augmentations
    # ------------------------------------------------------------------

    def horizontal_flip(self, img: Image.Image) -> Image.Image:
        """Flip image horizontally."""
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    def random_rotation(
        self, img: Image.Image, max_degrees: float = 15.0
    ) -> Image.Image:
        """
        Rotate image by a random angle.

        Args:
            img: Input PIL Image.
            max_degrees: Maximum rotation angle in degrees (default: 15).

        Returns:
            Rotated image.
        """
        angle = self.rng.uniform(-max_degrees, max_degrees)
        return img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)

    def adjust_brightness(
        self, img: Image.Image, low: float = 0.8, high: float = 1.2
    ) -> Image.Image:
        """
        Randomly adjust brightness.

        Args:
            img: Input PIL Image.
            low: Minimum brightness factor.
            high: Maximum brightness factor.

        Returns:
            Brightness-adjusted image.
        """
        factor = self.rng.uniform(low, high)
        return ImageEnhance.Brightness(img).enhance(factor)

    def adjust_contrast(
        self, img: Image.Image, low: float = 0.8, high: float = 1.2
    ) -> Image.Image:
        """
        Randomly adjust contrast.

        Args:
            img: Input PIL Image.
            low: Minimum contrast factor.
            high: Maximum contrast factor.

        Returns:
            Contrast-adjusted image.
        """
        factor = self.rng.uniform(low, high)
        return ImageEnhance.Contrast(img).enhance(factor)

    def add_gaussian_noise(
        self, img: Image.Image, noise_level: Optional[float] = None
    ) -> Image.Image:
        """
        Add Gaussian noise to an image.

        Args:
            img: Input PIL Image.
            noise_level: Noise standard deviation (0–255 scale).
                         Default: random between 2 and 10.

        Returns:
            Noisy image.
        """
        if noise_level is None:
            noise_level = self.rng.uniform(2.0, 10.0)

        arr = np.array(img, dtype=np.float64)
        noise = self.rng.normal(0, noise_level, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    def augment_image(self, img: Image.Image) -> Image.Image:
        """
        Apply random augmentations to an image.

        Args:
            img: Input PIL Image.

        Returns:
            Augmented PIL Image.
        """
        if not self.augment:
            return img

        if self.rng.random() < self.augment_prob:
            img = self.horizontal_flip(img)

        if self.rng.random() < self.augment_prob:
            img = self.random_rotation(img)

        if self.rng.random() < self.augment_prob:
            img = self.adjust_brightness(img)

        if self.rng.random() < self.augment_prob:
            img = self.adjust_contrast(img)

        if self.rng.random() < self.augment_prob:
            img = self.add_gaussian_noise(img)

        return img

    # ------------------------------------------------------------------
    # Batch loading
    # ------------------------------------------------------------------

    def load_batch(
        self, indices: List[int], augment: Optional[bool] = None
    ) -> Tuple[List[Image.Image], np.ndarray]:
        """
        Load a batch of images.

        Args:
            indices: List of file indices to load.
            augment: Override augmentation setting (default: use self.augment).

        Returns:
            Tuple of (images_list, labels_array).
        """
        if augment is None:
            augment = self.augment

        batch_images: List[Image.Image] = []
        batch_labels: List[int] = []

        for idx in indices:
            img = self.load_image(self.file_paths[idx])

            if augment:
                img = self.augment_image(img)

            batch_images.append(img)
            batch_labels.append(self.labels[idx])

        return batch_images, np.array(batch_labels)

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def train_val_test_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Split dataset into train/validation/test sets (stratified).

        Args:
            train_ratio: Proportion for training (default: 0.7).
            val_ratio: Proportion for validation (default: 0.15).
            test_ratio: Proportion for testing (default: 0.15).
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_indices, val_indices, test_indices).
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
            "Ratios must sum to 1.0"
        )

        rng = np.random.default_rng(seed)

        train_indices: List[int] = []
        val_indices: List[int] = []
        test_indices: List[int] = []

        for class_idx in range(len(self.class_names)):
            class_indices = [
                i for i, label in enumerate(self.labels) if label == class_idx
            ]
            rng.shuffle(class_indices)

            n_samples = len(class_indices)
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)

            train_indices.extend(class_indices[:n_train])
            val_indices.extend(class_indices[n_train : n_train + n_val])
            test_indices.extend(class_indices[n_train + n_val :])

        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        rng.shuffle(test_indices)

        print(f"\n📊 Dataset split:")
        print(f"   Training:   {len(train_indices):4d} samples ({train_ratio*100:.1f}%)")
        print(f"   Validation: {len(val_indices):4d} samples ({val_ratio*100:.1f}%)")
        print(f"   Testing:    {len(test_indices):4d} samples ({test_ratio*100:.1f}%)")

        return train_indices, val_indices, test_indices

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_info(self) -> Dict:
        """Get dataset information."""
        return {
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "num_files": len(self.file_paths),
            "augmentation": self.augment,
        }
