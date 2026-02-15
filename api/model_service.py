"""
Model Service — Manages model lifecycle with lazy dataset loading.

Supports:
  - Multiple datasets via DatasetRegistry (audio/music, audio/medical, …)
  - Multiple modalities: audio (spectrogram pipeline) and image (image pipeline)
  - Lazy loading: only ONE dataset's models live in memory at a time
  - Two engines per dataset: "custom" (NumPy from-scratch) and "pytorch"
  - Persistent evaluation cache that survives dataset swaps

Usage:
    svc = ModelService()
    svc.load_default()                  # Called once at startup
    svc.ensure_dataset("audio/music")   # Lazy swap (no-op if already active)
    result = svc.classify_input(file_bytes, filename, engine="custom")
"""

import numpy as np
import torch
import librosa
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.neural_network.network import NeuralNetwork
from src.pytorch_model.model import AudioClassifier
from src.preprocessing.spectrogram_gen import SpectrogramGenerator
from src.image_pipeline.preprocessor import ImagePreprocessor
from api.dataset_registry import (
    DatasetConfig,
    DatasetRegistry,
    DEFAULT_DATASET,
)

# Valid engine identifiers
VALID_ENGINES = {"custom", "pytorch"}


class ModelService:
    """
    Singleton service for model loading, inference, and metrics.

    Uses lazy dataset loading — at most one dataset's models are in memory.
    When a request targets a different dataset, the current models are
    unloaded and the new dataset's models are loaded on demand.

    Evaluation caches persist across swaps (they're small; models are not).
    """

    _instance: Optional["ModelService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Dataset registry (all known datasets)
        self.registry = DatasetRegistry()

        # Active dataset tracking
        self._active_key: Optional[str] = None
        self._active_config: Optional[DatasetConfig] = None

        # Currently loaded models (only for the active dataset)
        self.network: Optional[NeuralNetwork] = None
        self.pytorch_model: Optional[AudioClassifier] = None
        self.spec_gen: Optional[SpectrogramGenerator] = None       # audio modality
        self.img_preprocessor: Optional[ImagePreprocessor] = None  # image modality
        self.class_names: List[str] = []
        self.num_classes: int = 0
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.sample_rate: int = 22050
        self.duration: float = 2.0
        self.target_shape: Tuple[int, int] = (64, 64)

        # Persistent evaluation cache: {dataset_key: {engine: {…metrics}}}
        # Survives dataset swaps — only model weights are unloaded.
        self._eval_cache: Dict[str, Dict[str, Dict]] = {}

        self._initialized = True

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def active_dataset(self) -> Optional[str]:
        """Key of the currently loaded dataset, or None."""
        return self._active_key

    @property
    def active_config(self) -> Optional[DatasetConfig]:
        """Config of the currently loaded dataset."""
        return self._active_config

    @property
    def modality(self) -> str:
        """Modality of the currently loaded dataset ('audio' or 'image')."""
        if self._active_config is not None:
            return self._active_config.modality
        return "audio"

    # ------------------------------------------------------------------
    # Dataset lifecycle (lazy loading)
    # ------------------------------------------------------------------

    def load_default(self) -> None:
        """Load the default dataset at API startup."""
        self.ensure_dataset(DEFAULT_DATASET)

    def ensure_dataset(self, dataset_key: str) -> None:
        """
        Ensure the requested dataset is the active one.

        - If it's already loaded → no-op.
        - If a different dataset is loaded → unload it, load the new one.
        - If no models exist on disk for this dataset → raise ValueError.
        """
        if dataset_key == self._active_key:
            return

        config = self.registry.get(dataset_key)

        if not self.registry.is_ready(dataset_key):
            raise ValueError(
                f"Dataset '{dataset_key}' ({config.name}) has no trained "
                f"models on disk. Train a model first."
            )

        if self._active_key is not None:
            print(f"\n🔄 Swapping dataset: {self._active_key} → {dataset_key}")

        self._unload()
        self._load_from_config(config)

    # ------------------------------------------------------------------
    # Internal: load / unload
    # ------------------------------------------------------------------

    def _load_from_config(self, config: DatasetConfig) -> None:
        """Load models, norm stats, and spectrogram generator for a config."""
        print(f"\n📦 Loading dataset: {config.name} ({config.key})")

        # 1. Custom NumPy model ------------------------------------------------
        model_path = Path(config.model_path)
        if model_path.exists():
            self.network = NeuralNetwork.load(str(model_path))
            print(f"   🧠 Custom model: {config.model_path}")
        else:
            self.network = None
            print(f"   ⚠️  Custom model not found: {config.model_path}")

        # 2. PyTorch model -----------------------------------------------------
        pt_path = Path(config.pytorch_model_path)
        if pt_path.exists():
            self.pytorch_model = AudioClassifier.load(str(pt_path))
            self.pytorch_model.eval()
            print(f"   🔥 PyTorch model: {config.pytorch_model_path}")
        else:
            self.pytorch_model = None

        # 3. Normalization stats -----------------------------------------------
        norm_path = Path(config.norm_stats_path)
        if norm_path.exists():
            stats = np.load(str(norm_path), allow_pickle=True)
            self.mean = stats["mean"]
            self.std = stats["std"]
            if "class_names" in stats:
                self.class_names = stats["class_names"].tolist()
            else:
                self.class_names = list(config.class_names)
            print(f"   📊 Norm stats: {config.norm_stats_path}")
        else:
            # Try extracting from prepared data
            prep_path = Path(config.prepared_data_path)
            if prep_path.exists():
                print(f"   📊 Extracting norm stats from: {config.prepared_data_path}")
                data = np.load(str(prep_path), allow_pickle=True)
                self.mean = data["mean"]
                self.std = data["std"]
                self.class_names = (
                    data["class_names"].tolist()
                    if "class_names" in data
                    else list(config.class_names)
                )
                # Save for next time
                norm_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    str(norm_path),
                    mean=self.mean,
                    std=self.std,
                    class_names=np.array(self.class_names),
                )
                print(f"   💾 Saved norm stats: {config.norm_stats_path}")
            else:
                self.mean = None
                self.std = None
                self.class_names = list(config.class_names)
                print("   ⚠️  No norm stats or prepared data found")

        self.num_classes = len(self.class_names)

        # 4. Preprocessing params from config ----------------------------------
        self.sample_rate = config.sample_rate
        self.duration = config.duration
        self.target_shape = config.target_shape

        # 5. Modality-specific preprocessor ------------------------------------
        if config.modality == "audio":
            self.spec_gen = SpectrogramGenerator(
                sample_rate=self.sample_rate,
                target_shape=self.target_shape,
            )
            self.img_preprocessor = None
        else:
            self.spec_gen = None
            self.img_preprocessor = ImagePreprocessor(
                target_shape=self.target_shape,
            )

        # 6. Evaluate on test set (cached persistently) ------------------------
        if config.key not in self._eval_cache:
            self._eval_cache[config.key] = {
                "custom": {},
                "pytorch": {},
            }

        cache = self._eval_cache[config.key]
        prep_path = Path(config.prepared_data_path)

        if self.network and not cache["custom"] and prep_path.exists():
            self._run_evaluation(str(prep_path), "custom", config.key)
        if self.pytorch_model and not cache["pytorch"] and prep_path.exists():
            self._run_evaluation(str(prep_path), "pytorch", config.key)

        # 7. Mark active -------------------------------------------------------
        self._active_key = config.key
        self._active_config = config

        # 8. Summary -----------------------------------------------------------
        print(f"\n✅ Dataset ready: {config.name}")
        print(f"   Classes ({self.num_classes}): {self.class_names}")
        if self.network:
            print(f"   Custom parameters: {self.network.count_parameters():,}")
        if self.pytorch_model:
            print(f"   PyTorch parameters: {self.pytorch_model.count_parameters():,}")

    def _unload(self) -> None:
        """Release current models from memory. Eval cache is NOT cleared."""
        self.network = None
        self.pytorch_model = None
        self.spec_gen = None
        self.img_preprocessor = None
        self.mean = None
        self.std = None
        self.class_names = []
        self.num_classes = 0
        self._active_key = None
        self._active_config = None

    # ------------------------------------------------------------------
    # Evaluation (run once per dataset+engine, cached persistently)
    # ------------------------------------------------------------------

    def _run_evaluation(
        self, prepared_data_path: str, engine: str, dataset_key: str
    ) -> None:
        """Run forward pass on test set and cache confusion matrix + metrics."""
        try:
            data = np.load(prepared_data_path, allow_pickle=True)
            X_test = data["X_test"]
            y_test = data["y_test"]
            true_classes = np.argmax(y_test, axis=1)

            # Forward pass
            if engine == "custom":
                probs = self.network.forward(X_test)
            else:
                self.pytorch_model.eval()
                with torch.no_grad():
                    probs = self.pytorch_model.predict_proba(
                        torch.FloatTensor(X_test)
                    ).numpy()

            pred_classes = np.argmax(probs, axis=1)
            test_accuracy = float(np.mean(pred_classes == true_classes))

            # Confusion matrix
            cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
            for t, p in zip(true_classes, pred_classes):
                cm[t, p] += 1

            # Per-class metrics
            per_class: Dict[str, Dict] = {}
            for i, name in enumerate(self.class_names):
                tp = cm[i, i]
                fp = int(cm[:, i].sum()) - tp
                fn = int(cm[i, :].sum()) - tp
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )
                per_class[name] = {
                    "precision": round(float(precision), 4),
                    "recall": round(float(recall), 4),
                    "f1_score": round(float(f1), 4),
                    "support": int(cm[i, :].sum()),
                }

            f1_scores = [m["f1_score"] for m in per_class.values()]
            macro_f1 = round(float(np.mean(f1_scores)), 4)

            self._eval_cache[dataset_key][engine] = {
                "confusion_matrix": cm,
                "per_class_metrics": per_class,
                "test_accuracy": test_accuracy,
                "macro_f1": macro_f1,
            }

            print(
                f"   📈 [{engine}] test acc = {test_accuracy:.2%}, "
                f"macro F1 = {macro_f1:.4f}"
            )
        except Exception as e:
            print(f"   ⚠️  [{engine}] evaluation failed: {e}")

    # ------------------------------------------------------------------
    # Engine availability
    # ------------------------------------------------------------------

    def available_engines(self) -> List[str]:
        """Return list of loaded engines for the active dataset."""
        engines = []
        if self.network is not None:
            engines.append("custom")
        if self.pytorch_model is not None:
            engines.append("pytorch")
        return engines

    def _validate_engine(self, engine: str) -> str:
        """Validate and default the engine parameter."""
        if engine not in VALID_ENGINES:
            engine = "custom"
        if engine == "pytorch" and self.pytorch_model is None:
            engine = "custom"
        return engine

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify(self, audio: np.ndarray, engine: str = "custom") -> Dict:
        """
        Classify a raw audio waveform.

        Returns dict with prediction, confidence, all_confidences,
        spectrogram, waveform, and waveform_summary.
        """
        engine = self._validate_engine(engine)
        audio = self._pad_or_trim(audio)

        spectrogram = self.spec_gen.audio_to_spectrogram(audio)

        flat = spectrogram.flatten().reshape(1, -1)
        flat_normalized = self._normalize(flat)

        probs = self._forward(flat_normalized, engine)[0]
        pred_idx = int(np.argmax(probs))

        waveform_display = self._downsample_waveform(audio, target_points=500)

        return {
            "prediction": self.class_names[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
            "all_confidences": {
                name: round(float(probs[i]), 4)
                for i, name in enumerate(self.class_names)
            },
            "spectrogram": spectrogram.tolist(),
            "waveform": waveform_display,
            "waveform_summary": {
                "duration": round(float(len(audio) / self.sample_rate), 4),
                "sample_rate": self.sample_rate,
                "peak_amplitude": round(float(np.max(np.abs(audio))), 4),
            },
        }

    def generate_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Generate a mel-spectrogram without classification."""
        audio = self._pad_or_trim(audio)
        return self.spec_gen.audio_to_spectrogram(audio)

    def classify_spectrogram(
        self, spectrogram: np.ndarray, engine: str = "custom"
    ) -> Dict:
        """Classify a raw 2-D spectrogram (used by the What-If tool)."""
        engine = self._validate_engine(engine)
        flat = spectrogram.flatten().reshape(1, -1)
        flat_normalized = self._normalize(flat)
        probs = self._forward(flat_normalized, engine)[0]
        pred_idx = int(np.argmax(probs))

        return {
            "prediction": self.class_names[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
            "all_confidences": {
                name: round(float(probs[i]), 4)
                for i, name in enumerate(self.class_names)
            },
        }

    # ------------------------------------------------------------------
    # Image inference
    # ------------------------------------------------------------------

    def classify_image(
        self, image_bytes: bytes, engine: str = "custom"
    ) -> Dict:
        """
        Classify an uploaded image.

        Pipeline: bytes → PIL → grayscale → resize → flatten → normalize → forward.

        Returns dict with prediction, confidence, all_confidences,
        input_image (the preprocessed 2D array), and modality="image".
        """
        engine = self._validate_engine(engine)

        # Preprocess image to 2D array (same shape as spectrogram)
        image_array = self.img_preprocessor.load_from_bytes(image_bytes)

        flat = image_array.flatten().reshape(1, -1)
        flat_normalized = self._normalize(flat)

        probs = self._forward(flat_normalized, engine)[0]
        pred_idx = int(np.argmax(probs))

        return {
            "prediction": self.class_names[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
            "all_confidences": {
                name: round(float(probs[i]), 4)
                for i, name in enumerate(self.class_names)
            },
            "input_image": image_array.tolist(),
            "modality": "image",
        }

    # ------------------------------------------------------------------
    # Unified dispatcher (modality-aware)
    # ------------------------------------------------------------------

    def classify_input(
        self, file_bytes: bytes, filename: str, engine: str = "custom"
    ) -> Dict:
        """
        Classify an uploaded file, dispatching to the correct modality pipeline.

        - Audio dataset → audio waveform → spectrogram → classify
        - Image dataset → image → preprocess → classify

        Returns a dict compatible with ClassifyResponse (modality-specific
        fields are included only when relevant).
        """
        if self.modality == "image":
            result = self.classify_image(file_bytes, engine=engine)
            # Map input_image → spectrogram field for unified response shape
            result["spectrogram"] = result.pop("input_image")
            return result
        else:
            audio = self.load_audio_from_bytes(file_bytes, filename)
            result = self.classify(audio, engine=engine)
            result["modality"] = "audio"
            return result

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, X: np.ndarray, engine: str) -> np.ndarray:
        """Run forward pass on the selected engine, return probabilities."""
        if engine == "pytorch":
            self.pytorch_model.eval()
            with torch.no_grad():
                return self.pytorch_model.predict_proba(
                    torch.FloatTensor(X)
                ).numpy()
        else:
            return self.network.forward(X)

    # ------------------------------------------------------------------
    # Info / cached metrics
    # ------------------------------------------------------------------

    def get_model_info(self, engine: str = "custom") -> Dict:
        """Architecture, param count, class names, and test-set metrics."""
        engine = self._validate_engine(engine)

        if engine == "pytorch" and self.pytorch_model is not None:
            model = self.pytorch_model
            sizes = (
                [model.input_size]
                + model.hidden_sizes
                + [model.num_classes]
            )
            param_count = model.count_parameters()
            framework = "PyTorch"
        else:
            model = self.network
            sizes = (
                [model.input_size]
                + model.hidden_sizes
                + [model.num_classes]
            )
            param_count = model.count_parameters()
            framework = "NumPy (from scratch)"

        layers = []
        for i in range(len(sizes) - 1):
            layers.append(
                {
                    "type": "Dense" if engine == "custom" else "Linear",
                    "input_size": sizes[i],
                    "output_size": sizes[i + 1],
                    "activation": "ReLU" if i < len(sizes) - 2 else "Softmax",
                    "parameters": sizes[i] * sizes[i + 1] + sizes[i + 1],
                }
            )

        cache = self._eval_cache.get(self._active_key, {}).get(engine, {})

        info: Dict = {
            "architecture": {
                "input_size": sizes[0],
                "hidden_sizes": sizes[1:-1],
                "num_classes": sizes[-1],
                "layers": layers,
            },
            "parameters": param_count,
            "class_names": self.class_names,
            "spectrogram_shape": list(self.target_shape),
            "sample_rate": self.sample_rate,
            "audio_duration": self.duration,
            "engine": engine,
            "framework": framework,
            "dataset": self._active_key,
        }

        if "test_accuracy" in cache:
            info["test_accuracy"] = round(cache["test_accuracy"], 4)
        if "macro_f1" in cache:
            info["macro_f1"] = cache["macro_f1"]

        return info

    def get_confusion_matrix(self, engine: str = "custom") -> Optional[Dict]:
        engine = self._validate_engine(engine)
        cache = self._eval_cache.get(self._active_key, {}).get(engine, {})
        cm = cache.get("confusion_matrix")
        if cm is None:
            return None
        return {
            "matrix": cm.tolist(),
            "class_names": self.class_names,
        }

    def get_class_metrics(self, engine: str = "custom") -> Optional[Dict]:
        engine = self._validate_engine(engine)
        cache = self._eval_cache.get(self._active_key, {}).get(engine, {})
        if "per_class_metrics" not in cache:
            return None
        return {
            "per_class": cache["per_class_metrics"],
            "macro_f1": cache.get("macro_f1"),
            "test_accuracy": cache.get("test_accuracy"),
        }

    def get_training_history_path(self, engine: str = "custom") -> Optional[Path]:
        """Return the path to the training history file for the active dataset."""
        if self._active_config is None:
            return None
        engine = self._validate_engine(engine)
        if engine == "pytorch":
            return Path(self._active_config.pytorch_history_path)
        return Path(self._active_config.training_history_path)

    def get_tsne_path(self, engine: str = "custom") -> Optional[Path]:
        """Return the path to the t-SNE data file for the active dataset."""
        if self._active_config is None:
            return None
        engine = self._validate_engine(engine)
        if engine == "pytorch":
            return Path(self._active_config.pytorch_tsne_path)
        return Path(self._active_config.tsne_path)

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    def load_audio_from_bytes(
        self, audio_bytes: bytes, filename: str = "upload.wav"
    ) -> np.ndarray:
        """Load audio from uploaded file bytes via a temp file + librosa."""
        suffix = Path(filename).suffix or ".wav"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            audio, _ = librosa.load(
                tmp_path, sr=self.sample_rate, duration=self.duration
            )
            return audio
        finally:
            os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pad_or_trim(self, audio: np.ndarray) -> np.ndarray:
        """Ensure audio is exactly self.duration seconds long."""
        n_samples = int(self.sample_rate * self.duration)
        if len(audio) < n_samples:
            audio = np.pad(audio, (0, n_samples - len(audio)), mode="constant")
        else:
            audio = audio[:n_samples]
        return audio

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Apply saved zero-mean / unit-variance normalization."""
        if self.mean is None or self.std is None:
            return X
        result = np.zeros_like(X)
        nonzero_mask = self.std > 1e-6
        result[:, nonzero_mask] = (
            (X[:, nonzero_mask] - self.mean[nonzero_mask]) / self.std[nonzero_mask]
        )
        return result

    @staticmethod
    def _downsample_waveform(
        audio: np.ndarray, target_points: int = 500
    ) -> List[float]:
        """Downsample a waveform to ~target_points for lightweight display."""
        n = len(audio)
        if n <= target_points:
            return [round(float(x), 5) for x in audio]

        n_buckets = target_points // 2
        bucket_size = n / n_buckets
        envelope: List[float] = []

        for i in range(n_buckets):
            start = int(i * bucket_size)
            end = int((i + 1) * bucket_size)
            chunk = audio[start:end]
            envelope.append(round(float(np.min(chunk)), 5))
            envelope.append(round(float(np.max(chunk)), 5))

        return envelope
