#!/usr/bin/env python3
"""
Download and organize the Oxford-IIIT Pet Dataset for image classification.

Downloads real pet-breed images and organizes them into the expected
directory structure for the image preprocessing pipeline.

Source:  https://www.robots.ox.ac.uk/~vgg/data/pets/
License: Creative Commons Attribution-ShareAlike 4.0

Default selection (8 visually diverse breeds):
  Cats: abyssinian, bengal, persian, siamese
  Dogs: beagle, german_shorthaired, pug, samoyed

Each breed has ~200 high-quality real photographs.

Usage:
    python download_image_dataset.py                        # 8 default breeds
    python download_image_dataset.py --all-breeds           # All 37 breeds
    python download_image_dataset.py --output data/image/wildlife
"""

import argparse
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────

IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
IMAGES_TAR_SIZE_MB = 791  # approximate

# 8 visually diverse breeds (4 cats + 4 dogs)
DEFAULT_BREEDS = [
    "abyssinian",           # cat — sleek, golden
    "bengal",               # cat — spotted/marbled
    "persian",              # cat — flat face, long fur
    "siamese",              # cat — pointed coloring
    "beagle",               # dog — classic, tri-color
    "german_shorthaired",   # dog — spotted liver/white
    "pug",                  # dog — distinctive flat face
    "samoyed",              # dog — white, fluffy
]

DEFAULT_OUTPUT = "data/image/wildlife"

# Regex to parse breed from Oxford-IIIT Pet filenames
# e.g. "Abyssinian_100.jpg" → breed = "abyssinian"
# e.g. "german_shorthaired_12.jpg" → breed = "german_shorthaired"
FILENAME_RE = re.compile(r"^(.+?)_\d+\.jpg$", re.IGNORECASE)


# ── Helpers ───────────────────────────────────────────────────────────────

class DownloadProgressBar(tqdm):
    """tqdm-based progress bar for urllib downloads."""

    def update_to(self, blocks=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download_file(url: str, dest: str) -> str:
    """Download a file with a progress bar. Returns path to downloaded file."""
    print(f"\n📥 Downloading: {url}")
    print(f"   Destination: {dest}")
    print(f"   Estimated size: ~{IMAGES_TAR_SIZE_MB} MB\n")

    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc="Downloading"
    ) as pbar:
        urllib.request.urlretrieve(url, dest, reporthook=pbar.update_to)

    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"✅ Downloaded: {size_mb:.1f} MB")
    return dest


def parse_breed(filename: str) -> str | None:
    """Extract breed name from an Oxford-IIIT Pet filename."""
    match = FILENAME_RE.match(filename)
    if match:
        return match.group(1).lower()
    return None


def extract_and_organize(
    tar_path: str,
    output_dir: str,
    selected_breeds: list[str] | None,
) -> dict[str, int]:
    """
    Extract images from tar.gz and organize into class directories.

    Args:
        tar_path: Path to images.tar.gz
        output_dir: Target directory (e.g. data/image/wildlife)
        selected_breeds: List of breed names to keep (None = all)

    Returns:
        Dict of {breed_name: count}
    """
    output_path = Path(output_dir)
    breed_set = set(selected_breeds) if selected_breeds else None
    counts: dict[str, int] = {}

    print(f"\n📂 Extracting images to: {output_path}")

    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        jpg_members = [
            m for m in members
            if m.name.endswith(".jpg") and not m.name.startswith("._")
        ]

        print(f"   Total images in archive: {len(jpg_members)}")

        for member in tqdm(jpg_members, desc="Organizing"):
            filename = os.path.basename(member.name)
            breed = parse_breed(filename)

            if breed is None:
                continue

            # Filter to selected breeds
            if breed_set and breed not in breed_set:
                continue

            # Create breed directory and extract
            breed_dir = output_path / breed
            breed_dir.mkdir(parents=True, exist_ok=True)

            # Extract the file
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue

            dest_path = breed_dir / filename.lower()
            with open(dest_path, "wb") as f:
                f.write(fileobj.read())

            counts[breed] = counts.get(breed, 0) + 1

    return counts


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download Oxford-IIIT Pet images for image classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_image_dataset.py                  # 8 default breeds (~1,600 images)
  python download_image_dataset.py --all-breeds     # All 37 breeds (~7,400 images)
  python download_image_dataset.py --output data/image/pets
        """,
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--all-breeds", action="store_true",
        help="Download all 37 breeds instead of the default 8",
    )
    parser.add_argument(
        "--breeds", type=str, nargs="+", default=None,
        help="Custom list of breed names to download",
    )
    parser.add_argument(
        "--keep-tar", action="store_true",
        help="Keep the downloaded tar.gz file after extraction",
    )

    args = parser.parse_args()

    # Determine which breeds to download
    if args.all_breeds:
        selected_breeds = None  # None = all
        breed_desc = "all 37 breeds"
    elif args.breeds:
        selected_breeds = [b.lower() for b in args.breeds]
        breed_desc = f"{len(selected_breeds)} custom breeds"
    else:
        selected_breeds = DEFAULT_BREEDS
        breed_desc = f"{len(DEFAULT_BREEDS)} default breeds"

    print("=" * 65)
    print("🐾 Oxford-IIIT Pet Dataset Downloader")
    print("=" * 65)
    print(f"   Output:  {args.output}")
    print(f"   Breeds:  {breed_desc}")
    if selected_breeds:
        for b in selected_breeds:
            print(f"            - {b}")
    print()

    # Check if output directory already has data
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

    # Download to a temp directory
    cache_dir = Path("data/.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    tar_path = cache_dir / "oxford_pets_images.tar.gz"

    if tar_path.exists():
        size_mb = tar_path.stat().st_size / (1024 * 1024)
        print(f"📦 Found cached archive: {tar_path} ({size_mb:.1f} MB)")
    else:
        try:
            download_file(IMAGES_URL, str(tar_path))
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            print("\nIf the Oxford server is unavailable, you can manually download from:")
            print(f"   {IMAGES_URL}")
            print(f"   Save to: {tar_path}")
            sys.exit(1)

    # Extract and organize
    counts = extract_and_organize(
        tar_path=str(tar_path),
        output_dir=args.output,
        selected_breeds=selected_breeds,
    )

    # Clean up tarball
    if not args.keep_tar and tar_path.exists():
        print(f"\n🗑️  Removing cached archive ({tar_path.stat().st_size / (1024*1024):.0f} MB)...")
        tar_path.unlink()

    # Summary
    total_images = sum(counts.values())
    print(f"\n{'=' * 65}")
    print(f"✅ Dataset ready: {total_images} images across {len(counts)} breeds")
    print(f"{'=' * 65}")
    print(f"\n📊 Class distribution:")
    for breed in sorted(counts.keys()):
        print(f"   {breed:25s}: {counts[breed]:4d} images")

    print(f"\n📁 Directory structure:")
    print(f"   {args.output}/")
    for breed in sorted(counts.keys()):
        print(f"   ├── {breed}/")
    print(f"\n🎯 Next step: Run the training pipeline:")
    print(f"   python train_image.py --dataset image/wildlife")


if __name__ == "__main__":
    main()
