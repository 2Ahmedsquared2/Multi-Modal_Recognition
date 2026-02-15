#!/usr/bin/env python3
"""
Download and organize medical/body-sound audio from the ESC-50 dataset.

Source:  https://github.com/karolpiczak/ESC-50
License: Creative Commons Attribution-NonCommercial 3.0

ESC-50 contains 2,000 five-second environmental audio clips across 50 classes.
We extract 5 health/body-related classes (40 clips each, 200 total):

    breathing     — normal respiratory sounds
    coughing      — cough events
    crying_baby   — infant cry (neonatal/pediatric)
    sneezing      — sneeze events
    snoring       — sleep-related respiratory sounds

Each clip is 5 seconds at 44,100 Hz (our pipeline resamples to 22,050 Hz / 2 s).

Usage:
    python download_medical_audio.py                               # defaults
    python download_medical_audio.py --output data/audio/medical   # custom output
    python download_medical_audio.py --keep-zip                    # keep archive
"""

import argparse
import csv
import io
import os
import shutil
import sys
import zipfile
import urllib.request
from pathlib import Path

from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────

ESC50_ZIP_URL = (
    "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
)
ESC50_ZIP_SIZE_MB = 640  # approximate

# Body/health-related categories to extract from ESC-50
MEDICAL_CATEGORIES = [
    "breathing",
    "coughing",
    "crying_baby",
    "sneezing",
    "snoring",
]

DEFAULT_OUTPUT = "data/audio/medical"


# ── Helpers ───────────────────────────────────────────────────────────────

class DownloadProgressBar(tqdm):
    """tqdm-based progress bar for urllib downloads."""

    def update_to(self, blocks=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download_file(url: str, dest: str) -> str:
    """Download a file with a progress bar. Returns path to downloaded file."""
    print(f"\n📥 Downloading ESC-50 dataset...")
    print(f"   URL:  {url}")
    print(f"   Dest: {dest}")
    print(f"   Estimated size: ~{ESC50_ZIP_SIZE_MB} MB\n")

    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc="Downloading"
    ) as pbar:
        urllib.request.urlretrieve(url, dest, reporthook=pbar.update_to)

    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"✅ Downloaded: {size_mb:.1f} MB")
    return dest


def extract_medical_audio(
    zip_path: str,
    output_dir: str,
    categories: list[str],
) -> dict[str, int]:
    """
    Open the ESC-50 ZIP, read the metadata CSV, and extract only the audio
    files belonging to the requested categories.

    Args:
        zip_path:   Path to ESC-50-master.zip
        output_dir: Target directory (e.g. data/audio/medical)
        categories: List of ESC-50 category names to keep

    Returns:
        Dict of {category: file_count}
    """
    output_path = Path(output_dir)
    cat_set = set(categories)
    counts: dict[str, int] = {c: 0 for c in categories}

    print(f"\n📂 Extracting medical audio to: {output_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # ── 1. Read metadata CSV ──────────────────────────────────────
        csv_name = None
        for name in zf.namelist():
            if name.endswith("meta/esc50.csv"):
                csv_name = name
                break

        if csv_name is None:
            print("❌ Could not find meta/esc50.csv inside the ZIP archive.")
            sys.exit(1)

        csv_bytes = zf.read(csv_name).decode("utf-8")
        reader = csv.DictReader(io.StringIO(csv_bytes))

        # Build mapping: filename → category  (only for our categories)
        file_to_cat: dict[str, str] = {}
        for row in reader:
            cat = row["category"]
            if cat in cat_set:
                file_to_cat[row["filename"]] = cat

        print(f"   Found {len(file_to_cat)} audio files across {len(cat_set)} categories")

        # ── 2. Extract matching audio files ───────────────────────────
        audio_prefix = None
        for name in zf.namelist():
            if "/audio/" in name and name.endswith(".wav"):
                # e.g. "ESC-50-master/audio/1-100032-A-0.wav"
                audio_prefix = name.rsplit("/audio/", 1)[0] + "/audio/"
                break

        if audio_prefix is None:
            print("❌ Could not locate audio/ folder inside the ZIP.")
            sys.exit(1)

        for fname, cat in tqdm(file_to_cat.items(), desc="Organizing"):
            zip_member = audio_prefix + fname
            try:
                audio_bytes = zf.read(zip_member)
            except KeyError:
                print(f"   ⚠️  Missing in archive: {zip_member}")
                continue

            dest_dir = output_path / cat
            dest_dir.mkdir(parents=True, exist_ok=True)

            dest_file = dest_dir / fname
            with open(dest_file, "wb") as f:
                f.write(audio_bytes)

            counts[cat] += 1

    return counts


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download ESC-50 medical/body-sound audio clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_medical_audio.py                        # 5 classes, 200 clips
  python download_medical_audio.py --output data/audio/medical
  python download_medical_audio.py --keep-zip             # keep cached archive
        """,
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--keep-zip", action="store_true",
        help="Keep the downloaded ZIP file after extraction",
    )

    args = parser.parse_args()

    print("=" * 65)
    print("🏥 Medical Audio Dataset Downloader  (ESC-50 subset)")
    print("=" * 65)
    print(f"   Output:     {args.output}")
    print(f"   Categories: {len(MEDICAL_CATEGORIES)}")
    for cat in MEDICAL_CATEGORIES:
        print(f"               - {cat}")
    print(f"   Expected:   ~40 clips per category, ~200 total")
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
    zip_path = cache_dir / "ESC-50-master.zip"

    if zip_path.exists():
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"📦 Found cached archive: {zip_path} ({size_mb:.1f} MB)")
    else:
        try:
            download_file(ESC50_ZIP_URL, str(zip_path))
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            print("\nYou can manually download from:")
            print(f"   {ESC50_ZIP_URL}")
            print(f"   Save to: {zip_path}")
            sys.exit(1)

    # Extract medical categories
    counts = extract_medical_audio(
        zip_path=str(zip_path),
        output_dir=args.output,
        categories=MEDICAL_CATEGORIES,
    )

    # Clean up ZIP
    if not args.keep_zip and zip_path.exists():
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"\n🗑️  Removing cached archive ({size_mb:.0f} MB)...")
        zip_path.unlink()

    # Summary
    total = sum(counts.values())
    print(f"\n{'=' * 65}")
    print(f"✅ Dataset ready: {total} audio clips across {len(counts)} categories")
    print(f"{'=' * 65}")
    print(f"\n📊 Class distribution:")
    for cat in sorted(counts.keys()):
        print(f"   {cat:20s}: {counts[cat]:4d} clips")

    print(f"\n📁 Directory structure:")
    print(f"   {args.output}/")
    for cat in sorted(counts.keys()):
        print(f"   ├── {cat}/")

    print(f"\n🎯 Next step: Train the model:")
    print(f"   python train_audio.py --dataset audio/medical")


if __name__ == "__main__":
    main()
