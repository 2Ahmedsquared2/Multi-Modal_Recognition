# Acoustic Pattern Recognition Engine - Project Context

## Project Overview
Building a neural network FROM SCRATCH (no TensorFlow/PyTorch) that classifies musical instruments by analyzing audio spectrograms. This demonstrates deep understanding of both signal processing and machine learning fundamentals.

## Goal
Create a portfolio piece for USC IYA transfer application that looks technically sophisticated and unique. Must be completable in one focused day (Saturday).

## Technical Approach

### Part 1: Signal Processing Pipeline
- Convert audio files (mp3/wav) to mel-spectrograms using librosa
- Spectrograms are visual representations of sound: frequency (y-axis) over time (x-axis)
- Output: 128x128 grayscale images representing audio
- This is the "signal processing" component that makes the project unique

### Part 2: Neural Network (Built from Scratch)
- Implement forward propagation, backpropagation, and gradient descent using ONLY NumPy
- No ML frameworks allowed - we're proving we understand the math
- Architecture: Input (128x128 flattened) → Dense Layer 1 (512 neurons) → Dense Layer 2 (256 neurons) → Output (10 classes)
- Loss function: Cross-entropy
- Optimizer: SGD with momentum

### Part 3: Visualization & Polish
- Real-time training dashboard showing loss curves, spectrograms, predictions
- Gradient flow analysis (prove math is correct)
- Confusion matrix
- t-SNE visualization of learned features
- Attention heatmaps (which parts of spectrogram matter most)

## Dataset
**Option A (Preferred):** NSynth Dataset
- Google's instrument dataset
- ~300,000 musical notes from 1,000+ instruments
- We'll use a subset: 10 instrument classes
- Download: https://magenta.tensorflow.org/datasets/nsynth

**Option B (Backup):** UrbanSound8K
- Smaller, easier to download
- City sounds: car horn, dog bark, drilling, etc.
- 8,732 labeled sound excerpts
- Download: https://urbansounddataset.weebly.com/urbansound8k.html

**For speed: Start with Option B, can upgrade to NSynth later**

## Technology Stack

### Core Libraries
```python
numpy          # Matrix operations, neural network math
librosa        # Audio processing, spectrogram generation
soundfile      # Audio file loading
matplotlib     # Static visualizations
plotly         # Interactive visualizations
pandas         # Data handling
tqdm           # Progress bars
```

### Optional (for advanced features)
```python
dash           # Real-time web dashboard
scikit-learn   # t-SNE, confusion matrix utilities (NOT for neural network)
scipy          # Signal processing utilities
```

## File Structure
```
acoustic-recognition/
├── data/
│   ├── raw/              # Original audio files
│   ├── spectrograms/     # Generated spectrogram images
│   └── processed/        # Train/test splits
├── src/
│   ├── preprocessing/
│   │   ├── audio_loader.py       # Load audio files
│   │   └── spectrogram_gen.py    # Generate spectrograms
│   ├── neural_network/
│   │   ├── layers.py             # Dense layer implementation
│   │   ├── activations.py        # ReLU, Softmax, etc.
│   │   ├── loss.py               # Cross-entropy loss
│   │   ├── optimizers.py         # SGD with momentum
│   │   └── network.py            # Full network class
│   ├── training/
│   │   ├── trainer.py            # Training loop
│   │   └── evaluator.py          # Testing/validation
│   └── visualization/
│       ├── plots.py              # Loss curves, confusion matrix
│       ├── dashboard.py          # Real-time dashboard (optional)
│       └── attention.py          # Attention heatmaps
├── notebooks/
│   └── exploration.ipynb         # Quick testing/prototyping
├── models/
│   └── checkpoints/              # Saved model weights
├── results/
│   ├── figures/                  # Generated plots
│   └── metrics/                  # Performance logs
├── docs/
│   ├── math_derivations.md       # Backprop equations
│   └── architecture.md           # Network design decisions
├── train.py                      # Main training script
├── test.py                       # Evaluation script
├── requirements.txt
└── README.md
```

## Mathematical Foundation

### Forward Propagation
```
Input: X (batch_size, 16384)  # Flattened 128x128 spectrogram
Layer 1: Z1 = X @ W1 + b1
         A1 = ReLU(Z1)
Layer 2: Z2 = A1 @ W2 + b2
         A2 = ReLU(Z2)
Output:  Z3 = A2 @ W3 + b3
         A3 = Softmax(Z3)  # Probabilities for 10 classes
```

### Backpropagation
```
Loss: L = -Σ(y * log(ŷ))  # Cross-entropy

dL/dW3 = A2.T @ (ŷ - y)
dL/dW2 = A1.T @ (dL/dZ2)
dL/dW1 = X.T @ (dL/dZ1)

Where dL/dZ follows chain rule through activation functions
```

### Gradient Descent
```
W = W - learning_rate * dL/dW
With momentum: v = β*v + dL/dW
               W = W - learning_rate * v
```

## Success Criteria

### Minimum Viable Product (MVP)
- [ ] Load audio files and generate spectrograms
- [ ] Neural network trains without errors
- [ ] Achieves >70% accuracy on test set
- [ ] Can make predictions on new audio
- [ ] Has basic visualizations (loss curve, confusion matrix)

### Portfolio-Ready Version
- [ ] Clean, documented code
- [ ] Training dashboard with real-time updates
- [ ] Achieves >85% accuracy
- [ ] Gradient checking validates math
- [ ] Professional visualizations (t-SNE, attention maps)
- [ ] README with equations and architecture diagrams
- [ ] Screenshots/video demonstrating system

### Stretch Goals (if time permits)
- [ ] Real-time audio classification from microphone
- [ ] Adversarial audio examples
- [ ] Ensemble of multiple networks
- [ ] Learning rate finder
- [ ] Custom activation function experiments

## Key Terminology for Portfolio

**Instead of saying:** "I built a neural network for audio"
**Say:** "Architected end-to-end acoustic pattern recognition pipeline with custom backpropagation engine"

**Instead of saying:** "I used spectrograms"
**Say:** "Implemented Short-Time Fourier Transform pipeline with mel-frequency scaling for perceptual alignment"

**Instead of saying:** "I trained it on instrument sounds"
**Say:** "Developed multi-class classification system for timbral feature extraction across instrument families"

**Instead of saying:** "I coded gradient descent"
**Say:** "Derived and implemented momentum-based stochastic gradient descent with adaptive learning rate scheduling"

## Common Pitfalls to Avoid

1. **Vanishing/Exploding Gradients**
   - Use proper weight initialization (Xavier/He)
   - Monitor gradient norms during training
   - Clip gradients if necessary

2. **Memory Issues**
   - Process audio in batches, don't load all at once
   - Use smaller spectrogram resolution if needed (64x64 instead of 128x128)
   - Clear memory between training epochs

3. **Overfitting**
   - Implement L2 regularization
   - Use dropout (optional)
   - Monitor train vs validation loss

4. **Slow Training**
   - Vectorize operations (no Python loops)
   - Use larger batch sizes (32-64)
   - Consider data augmentation (pitch shift, time stretch)

## Timeline (One Day Build)

**Hour 1-2:** Data preprocessing
- Download dataset
- Generate spectrograms for all audio files
- Create train/test split

**Hour 3-5:** Neural network core
- Implement layers, activations, loss
- Build forward and backward propagation
- Test with small batch to verify math

**Hour 6-8:** Training loop
- Implement mini-batch SGD
- Add momentum optimizer
- Train full model

**Hour 9-10:** Visualizations
- Loss curves
- Confusion matrix
- Basic predictions

**Hour 11-12:** Polish
- Screenshots
- Documentation
- Video recording

## Questions Cursor Should Ask

If anything is unclear, Cursor should ask:
- Which dataset am I using? (NSynth or UrbanSound8K)
- What spectrogram resolution? (128x128 recommended)
- How many training samples? (Start with 1000-5000 per class)
- What batch size? (32-64 recommended)
- How many epochs? (20-50 depending on dataset size)

## Project Philosophy

This is NOT about building the most accurate model. This is about:
1. **Demonstrating understanding** of ML fundamentals
2. **Looking technically sophisticated** in portfolio
3. **Being unique** compared to other applicants
4. **Actually being completable** in one day

We're optimizing for visual impact and technical clarity, not state-of-the-art performance.