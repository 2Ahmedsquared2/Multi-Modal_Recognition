#!/usr/bin/env python3
"""
Download and organize wildlife/animal audio from the ESC-50 dataset.

Source:  https://github.com/karolpiczak/ESC-50
License: Creative Commons Attribution-NonCommercial 3.0

ESC-50 contains 2,000 five-second environmental audio clips across 50 classes.
We extract all 10 animal-related classes (40 clips each, 400 total) and remap
them into 5 balanced wildlife categories (80 clips each):

    birds       — crow + rooster
    domestic    — dog + cat
    wild_fauna  — frog + insects (flying)
    large_farm  — cow + pig
    small_farm  — sheep + hen

Each clip is 5 seconds at 44,100 Hz (our pipeline resamples to 22,050 Hz / 2 s).

Usage:
    python download_wildlife_audio.py                                # defaults
    python download_wildlife_audio.py --output data/audio/wildlife   # custom output
    python download_wildlife_audio.py --keep-zip                     # keep archive
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

# Mapping: ESC-50 original category → our target wildlife class
# 10 source categories → 5 balanced target classes (2 each → 80 clips per class)
ESC50_TO_WILDLIFE = {
    "crow":             "birds",
    "rooster":          "birds",
    "dog":              "domestic",
    "cat":              "domestic",
    "frog":             "wild_fauna",
    "insects":          "wild_fauna",
    "cow":              "large_farm",
    "pig":              "large_farm",
    "sheep":            "small_farm",
    "hen":              "small_farm",
}

# Sorted list of our target classes
WILDLIFE_CLASSES = sorted(set(ESC50_TO_WILDLIFE.values()))

DEFAULT_OUTPUT = "data/audio/wildlife"


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


def extract_wildlife_audio(
    zip_path: str,
    output_dir: str,
    category_map: dict[str, str],
) -> dict[str, int]:
    """
    Open the ESC-50 ZIP, read the metadata CSV, and extract animal audio
    files, renaming them into the mapped wildlife classes.

    Args:
        zip_path:     Path to ESC-50-master.zip
        output_dir:   Target directory (e.g. data/audio/wildlife)
        category_map: Dict of {esc50_category: target_class}

    Returns:
        Dict of {target_class: file_count}
    """
    output_path = Path(output_dir)
    source_cats = set(category_map.keys())
    target_classes = sorted(set(category_map.values()))
    counts: dict[str, int] = {c: 0 for c in target_classes}

    print(f"\n📂 Extracting wildlife audio to: {output_path}")
    print(f"   Source categories: {', '.join(sorted(source_cats))}")
    print(f"   Target classes:    {', '.join(target_classes)}")

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

        # Build mapping: filename → (original_cat, target_class)
        file_to_class: dict[str, tuple[str, str]] = {}
        for row in reader:
            esc_cat = row["category"]
            if esc_cat in source_cats:
                target = category_map[esc_cat]
                file_to_class[row["filename"]] = (esc_cat, target)

        print(f"   Found {len(file_to_class)} audio files across {len(source_cats)} ESC-50 categories")

        # ── 2. Extract matching audio files ───────────────────────────
        audio_prefix = None
        for name in zf.namelist():
            if "/audio/" in name and name.endswith(".wav"):
                audio_prefix = name.rsplit("/audio/", 1)[0] + "/audio/"
                break

        if audio_prefix is None:
            print("❌ Could not locate audio/ folder inside the ZIP.")
            sys.exit(1)

        for fname, (esc_cat, target_class) in tqdm(
            file_to_class.items(), desc="Organizing"
        ):
            zip_member = audio_prefix + fname
            try:
                audio_bytes = zf.read(zip_member)
            except KeyError:
                print(f"   ⚠️  Missing in archive: {zip_member}")
                continue

            dest_dir = output_path / target_class
            dest_dir.mkdir(parents=True, exist_ok=True)

            dest_file = dest_dir / fname
            with open(dest_file, "wb") as f:
                f.write(audio_bytes)

            counts[target_class] += 1

    return counts


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download ESC-50 wildlife/animal audio clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Remapping (ESC-50 → wildlife class):
  crow + rooster           → birds       (80 clips)
  dog + cat                → domestic    (80 clips)
  frog + insects (flying)  → wild_fauna  (80 clips)
  cow + pig                → large_farm  (80 clips)
  sheep + hen              → small_farm  (80 clips)

Examples:
  python download_wildlife_audio.py                          # 5 classes, 400 clips
  python download_wildlife_audio.py --output data/audio/wildlife
  python download_wildlife_audio.py --keep-zip               # keep cached archive
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
    print("🐾 Wildlife Audio Dataset Downloader  (ESC-50 animal subset)")
    print("=" * 65)
    print(f"   Output:     {args.output}")
    print(f"   Classes:    {len(WILDLIFE_CLASSES)}")
    print(f"\n   Remapping:")
    # Group source categories by target class for display
    target_to_sources: dict[str, list[str]] = {}
    for src, tgt in ESC50_TO_WILDLIFE.items():
        target_to_sources.setdefault(tgt, []).append(src)
    for cls in WILDLIFE_CLASSES:
        sources = ", ".join(sorted(target_to_sources[cls]))
        print(f"     {cls:15s} ← {sources}")
    print(f"\n   Expected:   ~80 clips per class, ~400 total")
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

    # Download (or reuse cache from medical download)
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

    # Extract wildlife categories (with remapping)
    counts = extract_wildlife_audio(
        zip_path=str(zip_path),
        output_dir=args.output,
        category_map=ESC50_TO_WILDLIFE,
    )

    # Clean up ZIP
    if not args.keep_zip and zip_path.exists():
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"\n🗑️  Removing cached archive ({size_mb:.0f} MB)...")
        zip_path.unlink()

    # Summary
    total = sum(counts.values())
    print(f"\n{'=' * 65}")
    print(f"✅ Dataset ready: {total} audio clips across {len(counts)} classes")
    print(f"{'=' * 65}")
    print(f"\n📊 Class distribution:")
    for cls in sorted(counts.keys()):
        sources = ", ".join(sorted(target_to_sources[cls]))
        print(f"   {cls:15s}: {counts[cls]:4d} clips  (from: {sources})")

    print(f"\n📁 Directory structure:")
    print(f"   {args.output}/")
    for cls in sorted(counts.keys()):
        print(f"   ├── {cls}/")

    print(f"\n🎯 Next step: Train the model:")
    print(f"   python train_audio.py --dataset audio/wildlife")


if __name__ == "__main__":
    main()
