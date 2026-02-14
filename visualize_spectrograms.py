"""
Visualize Sample Spectrograms
Quick script to view generated spectrograms and verify quality
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_spectrograms(
    data_path: str = "data/spectrograms/spectrograms.npz",
    num_samples: int = 6,
    save_path: str = "results/sample_spectrograms.png"
):
    """
    Visualize sample spectrograms from dataset
    
    Args:
        data_path: Path to NPZ file with spectrograms
        num_samples: Number of samples to display
        save_path: Where to save the visualization
    """
    print("=" * 60)
    print("📊 Visualizing Sample Spectrograms")
    print("=" * 60)
    
    # Load data
    print(f"\n📂 Loading spectrograms from: {data_path}")
    data = np.load(data_path)
    spectrograms = data['spectrograms']
    labels = data['labels']
    class_names = data['class_names'].tolist()
    
    print(f"✓ Loaded {len(spectrograms)} spectrograms")
    print(f"✓ Classes: {class_names}")
    
    # Select samples (one from each class if possible)
    print(f"\n🎨 Selecting {num_samples} samples...")
    selected_indices = []
    selected_labels = []
    
    for class_idx in range(min(num_samples, len(class_names))):
        # Find first sample of this class
        class_samples = np.where(labels == class_idx)[0]
        if len(class_samples) > 0:
            selected_indices.append(class_samples[0])
            selected_labels.append(class_idx)
    
    # Create figure
    print(f"\n📈 Creating visualization...")
    num_rows = (len(selected_indices) + 2) // 3
    num_cols = min(3, len(selected_indices))
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))
    if len(selected_indices) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each spectrogram
    for idx, (spec_idx, label_idx) in enumerate(zip(selected_indices, selected_labels)):
        ax = axes[idx]
        
        # Display spectrogram
        im = ax.imshow(
            spectrograms[spec_idx],
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='nearest'
        )
        
        # Add title and labels
        ax.set_title(f"{class_names[label_idx].upper()}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency (Mel Scale)")
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Intensity')
    
    # Hide unused subplots
    for idx in range(len(selected_indices), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to: {save_path}")
    
    # Show statistics
    print(f"\n📊 Spectrogram Statistics:")
    print(f"   Shape: {spectrograms[0].shape}")
    print(f"   Value range: [{spectrograms.min():.3f}, {spectrograms.max():.3f}]")
    print(f"   Mean: {spectrograms.mean():.3f}")
    print(f"   Std: {spectrograms.std():.3f}")
    
    print("\n" + "=" * 60)
    print("✅ Visualization complete!")
    print("=" * 60)
    
    return fig


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize sample spectrograms")
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/spectrograms/spectrograms.npz',
        help='Path to NPZ file with spectrograms'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=6,
        help='Number of samples to display (default: 6)'
    )
    parser.add_argument(
        '--save-path',
        type=str,
        default='results/sample_spectrograms.png',
        help='Where to save the visualization'
    )
    
    args = parser.parse_args()
    
    visualize_spectrograms(
        data_path=args.data_path,
        num_samples=args.num_samples,
        save_path=args.save_path
    )
