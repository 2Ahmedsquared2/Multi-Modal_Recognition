#!/usr/bin/env python3
"""
NSynth Dataset Download Script
Downloads a subset of the NSynth dataset for instrument classification
"""

import os
import json
import urllib.request
import tarfile
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_nsynth_subset(data_dir='data/raw', num_instruments=10, samples_per_instrument=500):
    """
    Download NSynth dataset subset
    
    Args:
        data_dir: Directory to store downloaded data
        num_instruments: Number of instrument classes to include (default: 10)
        samples_per_instrument: Max samples per instrument class (default: 500)
    """
    
    print("=" * 60)
    print("NSynth Dataset Downloader")
    print("=" * 60)
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # NSynth dataset URLs (using the smaller "test" split for faster download)
    # Test split is ~4GB, Train split is ~30GB
    nsynth_test_url = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz"
    
    print(f"\n📦 Downloading NSynth Test Set (~4GB)")
    print(f"   This will take 10-30 minutes depending on your connection...")
    print(f"   Downloading to: {data_path.absolute()}")
    
    tar_path = data_path / "nsynth-test.tar.gz"
    
    # Download if not already downloaded
    if tar_path.exists():
        print(f"\n✓ Archive already downloaded: {tar_path}")
    else:
        print(f"\n⬇️  Downloading from: {nsynth_test_url}")
        try:
            download_url(nsynth_test_url, tar_path)
            print(f"\n✓ Download complete!")
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            print("\nAlternative: Manually download from:")
            print("https://magenta.tensorflow.org/datasets/nsynth#files")
            return
    
    # Extract archive
    extract_path = data_path / "nsynth-test"
    
    if extract_path.exists():
        print(f"\n✓ Dataset already extracted: {extract_path}")
    else:
        print(f"\n📂 Extracting archive...")
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(data_path)
            print(f"✓ Extraction complete!")
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return
    
    # Organize by instrument families
    print(f"\n🎵 Organizing audio files by instrument class...")
    
    json_path = extract_path / "examples.json"
    audio_dir = extract_path / "audio"
    
    if not json_path.exists():
        print(f"❌ Metadata file not found: {json_path}")
        return
    
    # Load metadata
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    # NSynth instrument families (all available)
    all_instrument_families = {
        0: "bass",
        1: "brass", 
        2: "flute",
        3: "guitar",
        4: "keyboard",
        5: "mallet",
        6: "organ",
        7: "reed",
        8: "string",
        9: "synth_lead",
        10: "vocal"
    }
    
    # Limit to requested number of instrument classes
    instrument_families = dict(list(all_instrument_families.items())[:num_instruments])
    
    # Create organized directory structure
    organized_dir = data_path / "organized"
    organized_dir.mkdir(exist_ok=True)
    
    instrument_counts = {name: 0 for name in instrument_families.values()}
    
    print(f"\n📊 Selecting up to {samples_per_instrument} samples from {num_instruments} instrument classes...")
    print(f"   Classes: {', '.join(instrument_families.values())}")
    
    for filename, info in tqdm(metadata.items(), desc="Processing files"):
        instrument_family_id = info['instrument_family']
        
        if instrument_family_id in instrument_families:
            instrument_name = instrument_families[instrument_family_id]
            
            # Limit samples per instrument
            if instrument_counts[instrument_name] >= samples_per_instrument:
                continue
            
            # Create instrument directory
            instrument_dir = organized_dir / instrument_name
            instrument_dir.mkdir(exist_ok=True)
            
            # Copy/symlink audio file
            src_file = audio_dir / f"{filename}.wav"
            dst_file = instrument_dir / f"{filename}.wav"
            
            if src_file.exists() and not dst_file.exists():
                # Create symlink to save disk space (use absolute path)
                try:
                    dst_file.symlink_to(src_file.absolute())
                    instrument_counts[instrument_name] += 1
                except Exception:
                    # If symlink fails, copy the file
                    import shutil
                    shutil.copy2(src_file, dst_file)
                    instrument_counts[instrument_name] += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ Dataset Download Complete!")
    print("=" * 60)
    print(f"\n📁 Organized dataset location: {organized_dir.absolute()}")
    print(f"\n📊 Samples per instrument class:")
    
    total_samples = 0
    for instrument, count in sorted(instrument_counts.items()):
        if count > 0:
            print(f"   {instrument:15s}: {count:4d} samples")
            total_samples += count
    
    print(f"\n   {'TOTAL':15s}: {total_samples:4d} samples")
    print(f"\n💡 Next Step: Run the audio loader to create spectrograms")
    print(f"   python -c \"from src.preprocessing.audio_loader import test_loader; test_loader()\"")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download NSynth dataset subset')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                        help='Directory to store data (default: data/raw)')
    parser.add_argument('--num-instruments', type=int, default=10,
                        help='Number of instrument classes (default: 10)')
    parser.add_argument('--samples-per-instrument', type=int, default=500,
                        help='Max samples per instrument (default: 500)')
    
    args = parser.parse_args()
    
    download_nsynth_subset(
        data_dir=args.data_dir,
        num_instruments=args.num_instruments,
        samples_per_instrument=args.samples_per_instrument
    )
