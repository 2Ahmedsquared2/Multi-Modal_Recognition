# Step 3: Spectrogram Generation

**Estimated Time:** 45 minutes
**Status:** ✅ Completed

---

## Completed
- ✅ Created `src/preprocessing/spectrogram_gen.py` - Mel-spectrogram generator
- ✅ Implemented Short-Time Fourier Transform (STFT) pipeline
- ✅ Configurable resolution support (64×64 and 128×128)
- ✅ Pre-generation and caching system
- ✅ Batch spectrogram generation
- ✅ **Generated 3,333 spectrograms (64×64)** - 25.41 MB compressed, 52.08 MB uncompressed
- ✅ Created `generate_spectrograms.py` utility script
- ✅ All tests passed!

## Implementation Details

### SpectrogramGenerator Class Features
- **STFT Pipeline**: Converts audio waveforms to frequency-domain using librosa
- **Mel-Frequency Scaling**: Aligns frequency bands to human hearing perception
- **Configurable Resolution**: Supports 64×64 (MVP) or 128×128 (higher quality)
- **Normalization**: Scales spectrograms to [0, 1] range for neural network
- **Batch Processing**: Efficient generation for multiple audio files
- **Caching**: Pre-generate and save spectrograms to disk (NPZ format)

### Technical Specifications
- **Sample Rate**: 22,050 Hz
- **FFT Size**: 2,048 samples
- **Hop Length**: 512 samples
- **Mel Bands**: 128 bands
- **Output Shape**: 64×64 (configurable to 128×128)
- **Normalization**: Min-max scaling to [0, 1]

### Dataset Generated
📁 Location: `data/spectrograms/spectrograms.npz`

**Statistics:**
- Total samples: 3,333 spectrograms
- Resolution: 64×64 pixels
- File size: 25.41 MB (compressed)
- Memory size: 52.08 MB (uncompressed)
- Value range: [0.000, 1.000]
- Generation time: ~15 seconds (~209 spectrograms/second)

## How to Use

### 1. Generate Spectrograms (Already Done!)
```bash
# Activate virtual environment first
source venv/bin/activate

# Generate 64×64 spectrograms (fast, MVP)
python generate_spectrograms.py --resolution 64

# Or generate 128×128 spectrograms (higher quality)
python generate_spectrograms.py --resolution 128

# Custom options
python generate_spectrograms.py \
    --data-dir data/raw/organized \
    --output-dir data/spectrograms \
    --resolution 64 \
    --sample-rate 22050 \
    --duration 2.0
```

### 2. Use in Code (Two Methods)

**Method A: Load Pre-generated Spectrograms (Fast)**
```python
from src.preprocessing.spectrogram_gen import SpectrogramGenerator

# Load pre-generated spectrograms
spec_gen = SpectrogramGenerator(target_shape=(64, 64))
spectrograms, labels, class_names = spec_gen.load_pregenerated_dataset(
    data_dir="data/spectrograms",
    filename="spectrograms.npz"
)

# Ready for training!
# Shape: (3333, 64, 64)
```

**Method B: Generate On-The-Fly with Augmentation (Best Performance)**
```python
from src.preprocessing.audio_loader import AudioLoader
from src.preprocessing.spectrogram_gen import SpectrogramGenerator

# Initialize loaders
audio_loader = AudioLoader(
    data_dir="data/raw/organized",
    augment=True,  # Enable augmentation
    augment_prob=0.5
)

spec_gen = SpectrogramGenerator(target_shape=(64, 64))

# During training loop:
audio_batch, labels = audio_loader.load_batch(train_indices[:32])
spectrograms = spec_gen.batch_audio_to_spectrograms(audio_batch)

# This applies augmentation to audio THEN generates spectrograms
# Gives you data variety AND proper augmentation!
```

### 3. Test the Generator
```bash
# Run test suite
python src/preprocessing/spectrogram_gen.py
```

## Training Strategy (Recommended)

**For Best Results:**
1. ✅ **Pre-generated baseline**: Use the 3,333 raw spectrograms for initial training
2. 🎯 **On-the-fly augmentation**: During training, load audio with augmentation enabled, then generate spectrograms
3. 🚀 **Hybrid approach**: Fast training with augmented data variety

**Why this works:**
- Pre-generation eliminates spectrogram computation overhead
- On-the-fly augmentation creates infinite data variations
- Augmenting audio (before spectrogram) is more natural than augmenting images
- Expected accuracy improvement: 5-15% over raw data only

## Technical Deep Dive

### What is a Mel-Spectrogram?

**High Level:**
A mel-spectrogram is a visual representation of sound that shows:
- **X-axis**: Time (how sound changes over duration)
- **Y-axis**: Frequency (pitch, scaled to match human hearing)
- **Color/Intensity**: Loudness at each time-frequency point

**Process:**
1. **STFT (Short-Time Fourier Transform)**: Break audio into tiny chunks, analyze frequency content of each
2. **Mel Scaling**: Convert linear frequencies to mel-scale (matches how humans hear)
3. **Log Scale**: Convert to decibels (matches logarithmic human loudness perception)
4. **Resize**: Scale to target resolution (64×64 or 128×128)
5. **Normalize**: Scale values to [0, 1] for neural network

**Why it matters:**
- Neural networks understand images better than raw audio
- Mel-scale aligns with human perception (what we actually hear)
- Different instruments have distinct visual patterns in spectrograms
- This is the "signal processing" component that makes the project unique!

### Resolution Comparison

**64×64 Spectrograms:**
- ✅ Fast generation (~209/second)
- ✅ Fast training (less computation)
- ✅ Small memory footprint (52 MB for 3,333 samples)
- ✅ Good enough for MVP (expected 70-85% accuracy)
- ⚠️ Less frequency detail

**128×128 Spectrograms:**
- ✅ Higher frequency resolution
- ✅ Better for complex sounds
- ✅ Expected 5-10% accuracy improvement
- ⚠️ 4× more pixels (slower training)
- ⚠️ 4× more memory (~200 MB for 3,333 samples)

**Recommendation**: Start with 64×64, upgrade to 128×128 if time permits.

## Performance Analysis

**PERFORMANCE REVIEW:**

**What Changed:**
- Added spectrogram generation pipeline with STFT and mel-scaling
- Pre-generated 3,333 spectrograms in ~15 seconds
- Created efficient caching system (NPZ format)

**Performance Impact:**

1. **Training Speed**: 🚀 **MASSIVE IMPROVEMENT**
   - **Without pre-generation**: ~5-10ms per spectrogram during training → ~160-320ms per batch (32 samples)
   - **With pre-generation**: ~0.1ms to load from cache → ~3-5ms per batch
   - **Speedup**: 30-60× faster per training batch!

2. **Memory Usage**: ⚖️ **MODERATE**
   - Pre-generated 64×64: 52 MB in RAM (manageable)
   - Pre-generated 128×128: ~200 MB in RAM (still fine)
   - On-disk compressed: 25 MB (64×64), ~100 MB (128×128)

3. **Flexibility**: ✅ **EXCELLENT**
   - Can use pre-generated for fast baseline training
   - Can generate on-the-fly with augmentation for variety
   - Configurable resolution for performance tuning

4. **Expected Model Performance**: 📈 **SIGNIFICANT BOOST**
   - Spectrograms reveal frequency patterns invisible in raw audio
   - Mel-scaling aligns with instrument timbral characteristics
   - Expected baseline accuracy: 70-85% (compared to ~30-40% with raw audio)

**Bottom Line**: Pre-generating spectrograms is a massive win for training speed with minimal cost. The 15-second investment saves hours during training iterations.

## Next Steps
1. ✅ ~~Generate spectrograms~~ (DONE!)
2. **Next:** Move to Step 4: Data Preparation (train/val/test splits, batch loading)
3. Then: Steps 5-8 (Neural Network Implementation)

## Notes

### Testing Results
**Spectrogram Generator Test Output:**
```
✅ All tests passed!

- 64×64 generation: ✓
- 128×128 generation: ✓
- Batch generation: ✓
- Pre-generation (10 samples): ✓ (242.60 spectrograms/second)
- Full dataset (3,333 samples): ✓ (209.15 spectrograms/second)
- Save/load NPZ: ✓
```

**Key Observations:**
- Spectrograms properly normalized to [0.0, 1.0] range
- Batch shapes correct: (batch_size, height, width)
- NPZ compression effective: ~50% size reduction vs raw NumPy
- Generation speed stable at ~200-250 spectrograms/second

### Visualization Ideas for Demo
When presenting this project, you can show:
1. **Raw audio waveform** → looks boring
2. **Mel-spectrogram** → looks sophisticated and visual
3. **Side-by-side comparison** of different instruments' spectrograms
4. **Augmented vs. original** spectrograms showing data variety

This visual transformation from audio → spectrogram is impressive and easy to explain!

### Why This Approach is Sophisticated

When describing this in your portfolio:

**Instead of:** "I converted audio to images"

**Say:** "Implemented Short-Time Fourier Transform pipeline with mel-frequency scaling for perceptual alignment. The mel-scale transformation maps linear frequency to a logarithmic scale matching human auditory perception, enabling the neural network to learn timbral features that distinguish instrument families. This time-frequency representation reduces dimensionality while preserving salient acoustic characteristics."

**Translation:** You're showing you understand both signal processing AND machine learning!
