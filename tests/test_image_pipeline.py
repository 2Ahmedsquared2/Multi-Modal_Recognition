"""Tests for the Image Pipeline (ImagePreprocessor + ImageLoader) — using synthetic data."""

import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path
from PIL import Image

from src.image_pipeline.preprocessor import ImagePreprocessor
from src.image_pipeline.image_loader import ImageLoader


# ---------------------------------------------------------------------------
# Helpers: create synthetic image files on disk
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_image_dataset(tmp_path):
    """
    Create a tiny synthetic image dataset on disk:

        tmp_path/
          class_a/  (10 images)
          class_b/  (10 images)
          class_c/  (10 images)

    Each image is 32x32 with class-dependent pixel values so that
    the preprocessing pipeline produces distinguishable arrays.
    """
    rng = np.random.default_rng(42)

    for cls_idx, cls_name in enumerate(["class_a", "class_b", "class_c"]):
        cls_dir = tmp_path / cls_name
        cls_dir.mkdir()

        for i in range(10):
            # Create a small image with class-dependent base color
            base_value = 50 + cls_idx * 80  # ~50, ~130, ~210
            pixels = rng.integers(
                max(0, base_value - 20),
                min(255, base_value + 20),
                size=(32, 32, 3),
                dtype=np.uint8,
            )
            img = Image.fromarray(pixels, "RGB")
            img.save(cls_dir / f"img_{i:03d}.png")

    return tmp_path


@pytest.fixture
def single_pil_image():
    """A single 48x48 RGB PIL Image for unit-testing ImagePreprocessor."""
    rng = np.random.default_rng(99)
    pixels = rng.integers(0, 255, size=(48, 48, 3), dtype=np.uint8)
    return Image.fromarray(pixels, "RGB")


# ===========================================================================
# ImagePreprocessor tests
# ===========================================================================

class TestImagePreprocessor:
    """Tests for ImagePreprocessor methods using synthetic inputs."""

    # ----- image_to_array -----

    def test_image_to_array_output_shape(self, single_pil_image):
        """Output should match the configured target_shape."""
        proc = ImagePreprocessor(target_shape=(64, 64))
        arr = proc.image_to_array(single_pil_image)
        assert arr.shape == (64, 64)

    def test_image_to_array_custom_shape(self, single_pil_image):
        """Non-square target shapes should work."""
        proc = ImagePreprocessor(target_shape=(32, 48))
        arr = proc.image_to_array(single_pil_image)
        assert arr.shape == (32, 48)

    def test_image_to_array_normalized_range(self, single_pil_image):
        """Normalized output should be in [0, 1]."""
        proc = ImagePreprocessor(target_shape=(64, 64), normalize=True)
        arr = proc.image_to_array(single_pil_image)
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_image_to_array_unnormalized_range(self, single_pil_image):
        """With normalize=False, values should be in [0, 255] range."""
        proc = ImagePreprocessor(target_shape=(64, 64), normalize=False)
        arr = proc.image_to_array(single_pil_image)
        assert arr.max() > 1.0  # at least some pixel above 1
        assert arr.max() <= 255.0

    def test_image_to_array_dtype(self, single_pil_image):
        """Output should be float64."""
        proc = ImagePreprocessor(target_shape=(64, 64))
        arr = proc.image_to_array(single_pil_image)
        assert arr.dtype == np.float64

    def test_image_to_array_grayscale_input(self):
        """Grayscale input images should be handled correctly."""
        proc = ImagePreprocessor(target_shape=(16, 16))
        gray_img = Image.fromarray(
            np.random.randint(0, 255, (20, 20), dtype=np.uint8), "L"
        )
        arr = proc.image_to_array(gray_img)
        assert arr.shape == (16, 16)

    def test_image_to_array_rgba_input(self):
        """RGBA input images should be handled correctly."""
        proc = ImagePreprocessor(target_shape=(16, 16))
        rgba = Image.fromarray(
            np.random.randint(0, 255, (20, 20, 4), dtype=np.uint8), "RGBA"
        )
        arr = proc.image_to_array(rgba)
        assert arr.shape == (16, 16)

    # ----- load_from_path -----

    def test_load_from_path(self, synthetic_image_dataset):
        """Loading from a valid path should return correct shape."""
        proc = ImagePreprocessor(target_shape=(64, 64))
        img_path = str(synthetic_image_dataset / "class_a" / "img_000.png")
        arr = proc.load_from_path(img_path)
        assert arr.shape == (64, 64)

    # ----- load_from_bytes -----

    def test_load_from_bytes(self, single_pil_image):
        """Loading from bytes should return correct shape."""
        proc = ImagePreprocessor(target_shape=(64, 64))
        import io

        buf = io.BytesIO()
        single_pil_image.save(buf, format="PNG")
        arr = proc.load_from_bytes(buf.getvalue())
        assert arr.shape == (64, 64)

    # ----- batch_load -----

    def test_batch_load(self, synthetic_image_dataset):
        """Batch load should return (N, H, W) array."""
        proc = ImagePreprocessor(target_shape=(32, 32))
        paths = [
            str(synthetic_image_dataset / "class_a" / f"img_{i:03d}.png")
            for i in range(5)
        ]
        batch = proc.batch_load(paths)
        assert batch.shape == (5, 32, 32)

    # ----- batch_images_to_arrays -----

    def test_batch_images_to_arrays(self, single_pil_image):
        """Batch convert PIL images to arrays."""
        proc = ImagePreprocessor(target_shape=(16, 16))
        images = [single_pil_image, single_pil_image, single_pil_image]
        batch = proc.batch_images_to_arrays(images)
        assert batch.shape == (3, 16, 16)

    # ----- generate_and_save_dataset / load_pregenerated_dataset -----

    def test_generate_and_save_dataset(self, synthetic_image_dataset, tmp_path):
        """Dataset generation should create a valid .npz file."""
        loader = ImageLoader(data_dir=str(synthetic_image_dataset), augment=False)
        proc = ImagePreprocessor(target_shape=(16, 16))

        output_dir = str(tmp_path / "output")
        result = proc.generate_and_save_dataset(
            image_loader=loader,
            output_dir=output_dir,
            save_format="npz",
        )

        assert result["num_samples"] == 30  # 10 per class × 3 classes
        assert result["shape"] == (30, 16, 16)
        assert Path(output_dir, "spectrograms.npz").exists()

    def test_load_pregenerated_dataset(self, synthetic_image_dataset, tmp_path):
        """Round-trip: generate → save → load should preserve data."""
        loader = ImageLoader(data_dir=str(synthetic_image_dataset), augment=False)
        proc = ImagePreprocessor(target_shape=(16, 16))

        output_dir = str(tmp_path / "output")
        proc.generate_and_save_dataset(
            image_loader=loader, output_dir=output_dir
        )

        arrays, labels, class_names = proc.load_pregenerated_dataset(
            data_dir=output_dir
        )

        assert arrays.shape == (30, 16, 16)
        assert labels.shape == (30,)
        assert class_names == ["class_a", "class_b", "class_c"]

    def test_npz_compatible_with_data_preparator(
        self, synthetic_image_dataset, tmp_path
    ):
        """
        The generated .npz should be consumable by DataPreparator
        (same keys: spectrograms, labels, class_names).
        """
        loader = ImageLoader(data_dir=str(synthetic_image_dataset), augment=False)
        proc = ImagePreprocessor(target_shape=(16, 16))

        output_dir = str(tmp_path / "output")
        proc.generate_and_save_dataset(
            image_loader=loader, output_dir=output_dir
        )

        # Load the raw .npz and check it has the keys DataPreparator expects
        npz_path = Path(output_dir) / "spectrograms.npz"
        data = np.load(str(npz_path), allow_pickle=True)

        assert "spectrograms" in data
        assert "labels" in data
        assert "class_names" in data

    # ----- get_config -----

    def test_get_config(self):
        """Config dict should include target_shape and normalize."""
        proc = ImagePreprocessor(target_shape=(128, 128), normalize=False)
        config = proc.get_config()
        assert config["target_shape"] == (128, 128)
        assert config["normalize"] is False


# ===========================================================================
# ImageLoader tests
# ===========================================================================

class TestImageLoader:
    """Tests for ImageLoader methods using synthetic image dataset."""

    # ----- Scanning -----

    def test_scan_finds_all_classes(self, synthetic_image_dataset):
        """Should discover all class directories."""
        loader = ImageLoader(data_dir=str(synthetic_image_dataset), augment=False)
        assert loader.class_names == ["class_a", "class_b", "class_c"]

    def test_scan_finds_all_files(self, synthetic_image_dataset):
        """Should discover all image files."""
        loader = ImageLoader(data_dir=str(synthetic_image_dataset), augment=False)
        assert len(loader.file_paths) == 30  # 10 × 3

    def test_scan_labels_correct(self, synthetic_image_dataset):
        """Labels should correspond to class directories."""
        loader = ImageLoader(data_dir=str(synthetic_image_dataset), augment=False)

        # First 10 are class_a (idx=0), next 10 class_b (idx=1), etc.
        assert all(l == 0 for l in loader.labels[:10])
        assert all(l == 1 for l in loader.labels[10:20])
        assert all(l == 2 for l in loader.labels[20:30])

    def test_scan_missing_dir_raises(self, tmp_path):
        """Should raise FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError):
            ImageLoader(data_dir=str(tmp_path / "nonexistent"))

    def test_scan_empty_dir_raises(self, tmp_path):
        """Should raise ValueError when no class directories exist."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError):
            ImageLoader(data_dir=str(empty_dir))

    # ----- Loading -----

    def test_load_image_returns_rgb(self, synthetic_image_dataset):
        """Loaded images should be RGB mode."""
        loader = ImageLoader(data_dir=str(synthetic_image_dataset), augment=False)
        img = loader.load_image(loader.file_paths[0])
        assert img.mode == "RGB"

    # ----- Augmentation -----

    def test_augment_image_returns_pil(self, synthetic_image_dataset):
        """Augmented result should still be a PIL Image."""
        loader = ImageLoader(data_dir=str(synthetic_image_dataset), augment=True)
        img = loader.load_image(loader.file_paths[0])
        augmented = loader.augment_image(img)
        assert isinstance(augmented, Image.Image)

    def test_augment_disabled(self, synthetic_image_dataset):
        """With augment=False, augment_image should return identical image."""
        loader = ImageLoader(data_dir=str(synthetic_image_dataset), augment=False)
        img = loader.load_image(loader.file_paths[0])
        result = loader.augment_image(img)

        np.testing.assert_array_equal(np.array(img), np.array(result))

    def test_horizontal_flip(self, synthetic_image_dataset):
        """Horizontal flip should mirror pixels."""
        loader = ImageLoader(data_dir=str(synthetic_image_dataset))
        img = loader.load_image(loader.file_paths[0])
        flipped = loader.horizontal_flip(img)

        original_arr = np.array(img)
        flipped_arr = np.array(flipped)

        np.testing.assert_array_equal(
            original_arr[:, ::-1, :], flipped_arr
        )

    # ----- Batch loading -----

    def test_load_batch(self, synthetic_image_dataset):
        """Batch load should return correct number of images and labels."""
        loader = ImageLoader(data_dir=str(synthetic_image_dataset), augment=False)
        images, labels = loader.load_batch([0, 1, 2, 10, 20])

        assert len(images) == 5
        assert labels.shape == (5,)
        assert all(isinstance(img, Image.Image) for img in images)

    # ----- Splitting -----

    def test_train_val_test_split_sizes(self, synthetic_image_dataset):
        """Split sizes should sum to total dataset size."""
        loader = ImageLoader(data_dir=str(synthetic_image_dataset), augment=False)
        train_idx, val_idx, test_idx = loader.train_val_test_split()

        total = len(train_idx) + len(val_idx) + len(test_idx)
        assert total == 30

    def test_train_val_test_split_no_overlap(self, synthetic_image_dataset):
        """Splits should have no overlapping indices."""
        loader = ImageLoader(data_dir=str(synthetic_image_dataset), augment=False)
        train_idx, val_idx, test_idx = loader.train_val_test_split()

        all_indices = set(train_idx) | set(val_idx) | set(test_idx)
        assert len(all_indices) == len(train_idx) + len(val_idx) + len(test_idx)

    def test_train_val_test_split_ratios_error(self, synthetic_image_dataset):
        """Ratios not summing to 1.0 should raise."""
        loader = ImageLoader(data_dir=str(synthetic_image_dataset), augment=False)
        with pytest.raises(AssertionError):
            loader.train_val_test_split(
                train_ratio=0.5, val_ratio=0.3, test_ratio=0.3
            )

    # ----- Info -----

    def test_get_info(self, synthetic_image_dataset):
        """get_info should return expected keys."""
        loader = ImageLoader(data_dir=str(synthetic_image_dataset), augment=False)
        info = loader.get_info()

        assert info["num_classes"] == 3
        assert info["num_files"] == 30
        assert info["augmentation"] is False


# ===========================================================================
# End-to-end integration test
# ===========================================================================

class TestImagePipelineIntegration:
    """End-to-end: ImageLoader → ImagePreprocessor → .npz → DataPreparator."""

    def test_full_pipeline(self, synthetic_image_dataset, tmp_path):
        """
        Full pipeline round-trip:
        1. Scan dataset with ImageLoader
        2. Generate .npz with ImagePreprocessor
        3. Load with DataPreparator
        4. Flatten, split, normalize, one-hot encode
        """
        from src.preprocessing.data_prep import DataPreparator

        # 1. Scan
        loader = ImageLoader(data_dir=str(synthetic_image_dataset), augment=False)
        assert len(loader.file_paths) == 30

        # 2. Generate .npz
        proc = ImagePreprocessor(target_shape=(16, 16))
        output_dir = str(tmp_path / "spectrograms")
        proc.generate_and_save_dataset(image_loader=loader, output_dir=output_dir)

        # 3. Load via DataPreparator
        prep = DataPreparator(
            spectrogram_path=str(Path(output_dir) / "spectrograms.npz"),
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        data = prep.prepare()

        # 4. Verify
        assert data["num_classes"] == 3
        assert data["input_dim"] == 16 * 16  # 256
        assert data["X_train"].shape[1] == 256
        assert data["y_train"].shape[1] == 3

        total = len(data["X_train"]) + len(data["X_val"]) + len(data["X_test"])
        assert total == 30

        # One-hot rows sum to 1
        np.testing.assert_allclose(data["y_train"].sum(axis=1), 1.0)
        np.testing.assert_allclose(data["y_val"].sum(axis=1), 1.0)
        np.testing.assert_allclose(data["y_test"].sum(axis=1), 1.0)
