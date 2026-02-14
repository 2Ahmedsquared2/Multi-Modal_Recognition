"""
Generate Spectrograms Script
Pre-generates spectrograms for the entire dataset for efficient training
"""

import argparse
from src.preprocessing.audio_loader import AudioLoader
from src.preprocessing.spectrogram_gen import SpectrogramGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate spectrograms from audio dataset")
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw/organized',
        help='Directory containing organized audio files (default: data/raw/organized)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/spectrograms',
        help='Directory to save spectrograms (default: data/spectrograms)'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=64,
        choices=[64, 128],
        help='Spectrogram resolution (64x64 or 128x128) (default: 64)'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=22050,
        help='Audio sample rate in Hz (default: 22050)'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=2.0,
        help='Audio duration in seconds (default: 2.0)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎨 SPECTROGRAM GENERATION")
    print("=" * 70)
    
    # Initialize AudioLoader
    print("\n1️⃣ Loading audio dataset...")
    audio_loader = AudioLoader(
        data_dir=args.data_dir,
        sample_rate=args.sample_rate,
        duration=args.duration,
        augment=False  # No augmentation for pre-generated spectrograms
    )
    
    # Initialize SpectrogramGenerator
    print(f"\n2️⃣ Initializing spectrogram generator ({args.resolution}x{args.resolution})...")
    spec_generator = SpectrogramGenerator(
        sample_rate=args.sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        target_shape=(args.resolution, args.resolution),
        normalize=True
    )
    
    # Generate and save spectrograms
    print("\n3️⃣ Generating spectrograms...")
    result = spec_generator.generate_and_save_dataset(
        audio_loader=audio_loader,
        output_dir=args.output_dir,
        save_format='npz'
    )
    
    print("\n" + "=" * 70)
    print("✅ GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Output location: {result['output_path']}")
    print(f"📊 Total samples: {result['num_samples']}")
    print(f"🖼️  Spectrogram shape: {result['shape']}")
    print(f"\n💡 Next steps:")
    print(f"   1. Move to Step 4: Data Preparation")
    print(f"   2. Begin implementing neural network (Steps 5-8)")
    print(f"   3. Use these pre-generated spectrograms for training")
    print("\n🎯 For training with augmentation:")
    print(f"   - Load audio with AudioLoader (augment=True)")
    print(f"   - Generate spectrograms on-the-fly during training")
    print(f"   - This gives you both speed AND variety!")
    print("=" * 70)


if __name__ == "__main__":
    main()
