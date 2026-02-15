#!/usr/bin/env python3
"""
Download and organize the DermaMNIST medical image dataset.

Source:  MedMNIST v2 — https://medmnist.com/
Paper:   Yang et al., "MedMNIST v2", Scientific Data (2023)
License: Creative Commons Attribution 4.0

DermaMNIST contains 10,015 dermatoscopic images of skin lesions across 7 classes:

    actinic_keratoses      — pre-cancerous scaly patches
    basal_cell_carcinoma   — most common skin cancer
    benign_keratosis       — non-cancerous growth (seborrheic keratosis, etc.)
    dermatofibroma         — benign fibrous skin nodule
    melanoma               — most dangerous skin cancer
    melanocytic_nevi       — common moles
    vascular_lesion        — angiomas, angiokeratomas, etc.

Original images are 28×28 RGB. The training pipeline upscales to 64×64.

Data is downloaded as a single .npz from Zenodo (no extra pip dependency).

Usage:
    python download_medical_image.py                               # defaults
    python download_medical_image.py --output data/image/medical   # custom output
"""

import argparse
import os
import sys
import shutil
import urllib.request
import numpy as np
from pathlib import Path

from PIL import Image
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────

DERMAMNIST_URL = (
    "https://zenodo.org/records/10519652/files/dermamnist.npz"
)
DERMAMNIST_SIZE_MB = 29  # approximate

# Label index → class name
DERMAMNIST_CLASSES = {
    0: "actinic_keratoses",
    1: "basal_cell_carcinoma",
    2: "benign_keratosis",
    3: "dermatofibroma",
    4: "melanoma",
    5: "melanocytic_nevi",
    6: "vascular_lesion",
}

DEFAULT_OUTPUT = "data/image/medical"


# ── Helpers ───────────────────────────────────────────────────────────────

class DownloadProgressBar(tqdm):
    """tqdm-based progress bar for urllib downloads."""

    def update_to(self, blocks=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download_file(url: str, dest: str) -> str:
    """Download a file with a progress bar. Returns path to downloaded file."""
    print(f"\n📥 Downloading DermaMNIST dataset...")
    print(f"   URL:  {url}")
    print(f"   Dest: {dest}")
    print(f"   Estimated size: ~{DERMAMNIST_SIZE_MB} MB\n")

    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc="Downloading"
    ) as pbar:
        urllib.request.urlretrieve(url, dest, reporthook=pbar.update_to)

    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"✅ Downloaded: {size_mb:.1f} MB")
    return dest


def extract_to_class_dirs(
    npz_path: str,
    output_dir: str,
    class_map: dict[int, str],
) -> dict[str, int]:
    """
    Load the MedMNIST .npz, merge all splits (train+val+test),
    and save individual images as PNGs organized by class directory.

    Args:
        npz_path:   Path to dermamnist.npz
        output_dir: Target directory (e.g. data/image/medical)
        class_map:  Dict mapping label index → class name

    Returns:
        Dict of {class_name: image_count}
    """
    output_path = Path(output_dir)
    counts: dict[str, int] = {name: 0 for name in class_map.values()}

    print(f"\n📂 Extracting images to: {output_path}")

    data = np.load(npz_path)

    # Merge all splits
    all_images = np.concatenate([
        data["train_images"],
        data["val_images"],
        data["test_images"],
    ], axis=0)
    all_labels = np.concatenate([
        data["train_labels"],
        data["val_labels"],
        data["test_labels"],
    ], axis=0).flatten()

    print(f"   Total images: {len(all_images)} ({all_images.shape[1]}×{all_images.shape[2]} RGB)")
    print(f"   Classes: {len(class_map)}")

    for idx in tqdm(range(len(all_images)), desc="Saving PNGs"):
        label = int(all_labels[idx])
        class_name = class_map[label]

        class_dir = output_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        # Save as PNG
        img_array = all_images[idx]  # shape: (28, 28, 3)
        img = Image.fromarray(img_array)
        img.save(class_dir / f"{class_name}_{idx:05d}.png")

        counts[class_name] += 1

    return counts


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download DermaMNIST skin lesion images for medical image classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DermaMNIST classes (7 skin lesion types):
  0: actinic_keratoses      — pre-cancerous scaly patches
  1: basal_cell_carcinoma   — most common skin cancer
  2: benign_keratosis       — non-cancerous growths
  3: dermatofibroma         — benign fibrous nodule
  4: melanoma               — most dangerous skin cancer
  5: melanocytic_nevi       — common moles
  6: vascular_lesion        — angiomas, etc.

Examples:
  python download_medical_image.py                        # 7 classes, ~10,015 images
  python download_medical_image.py --output data/image/medical
        """,
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--keep-npz", action="store_true",
        help="Keep the downloaded .npz file after extraction",
    )

    args = parser.parse_args()

    class_names = sorted(DERMAMNIST_CLASSES.values())

    print("=" * 65)
    print("🏥 Medical Image Dataset Downloader  (DermaMNIST)")
    print("=" * 65)
    print(f"   Output:     {args.output}")
    print(f"   Classes:    {len(DERMAMNIST_CLASSES)}")
    for name in class_names:
        print(f"               - {name}")
    print(f"   Expected:   ~10,015 images total (28×28 RGB)")
    print()

    # Check if output already has data
    output_path = Path(args.output)
    if output_path.exists():
        existing = [d.name for d in output_path.iterdir() if d.is_dir()]
        if existing:
            print(f"⚠️  Output directory already contains: {', '.join(sorted(existing))}")
            response = input("   Overwrite? [y/N] ").strip().lower()
            if response != "y":
                print("Aborted.")
                sys.exit(0)
            shutil.rmtree(output_path)

    # Download (or reuse cache)
    cache_dir = Path("data/.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / "dermamnist.npz"

    if npz_path.exists():
        size_mb = npz_path.stat().st_size / (1024 * 1024)
        print(f"📦 Found cached archive: {npz_path} ({size_mb:.1f} MB)")
    else:
        try:
            download_file(DERMAMNIST_URL, str(npz_path))
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            print("\nYou can manually download from:")
            print(f"   {DERMAMNIST_URL}")
            print(f"   Save to: {npz_path}")
            sys.exit(1)

    # Extract to class directories
    counts = extract_to_class_dirs(
        npz_path=str(npz_path),
        output_dir=args.output,
        class_map=DERMAMNIST_CLASSES,
    )

    # Clean up npz
    if not args.keep_npz and npz_path.exists():
        size_mb = npz_path.stat().st_size / (1024 * 1024)
        print(f"\n🗑️  Removing cached .npz ({size_mb:.0f} MB)...")
        npz_path.unlink()

    # Summary
    total = sum(counts.values())
    print(f"\n{'=' * 65}")
    print(f"✅ Dataset ready: {total} images across {len(counts)} classes")
    print(f"{'=' * 65}")
    print(f"\n📊 Class distribution:")
    for cls in sorted(counts.keys()):
        print(f"   {cls:25s}: {counts[cls]:5d} images")

    print(f"\n📁 Directory structure:")
    print(f"   {args.output}/")
    for cls in sorted(counts.keys()):
        print(f"   ├── {cls}/")

    print(f"\n🎯 Next step: Train the model:")
    print(f"   python train_image.py --dataset image/medical")


if __name__ == "__main__":
    main()
