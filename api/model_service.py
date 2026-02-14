"""
Model Service — Singleton that loads trained models and provides inference.

Supports two engines:
  - "custom"  : from-scratch NumPy neural network  (models/model.npz)
  - "pytorch" : PyTorch neural network             (models/pytorch_model.pt)

Both share the same normalization stats, spectrogram generator, and audio
processing pipeline. Evaluation metrics are cached per engine.
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

# Valid engine identifiers
VALID_ENGINES = {"custom", "pytorch"}


class ModelService:
    """
    Singleton service that loads both the custom and PyTorch models,
    normalization stats, and spectrogram generator once at startup.
    All inference and metric queries go through this instance.
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

        # Custom (NumPy) model
        self.network: Optional[NeuralNetwork] = None

        # PyTorch model
        self.pytorch_model: Optional[AudioClassifier] = None

        # Shared resources
        self.spec_gen: Optional[SpectrogramGenerator] = None
        self.class_names: List[str] = []
        self.num_classes: int = 0
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.sample_rate: int = 22050
        self.duration: float = 2.0
        self.target_shape: Tuple[int, int] = (64, 64)

        # Cached evaluation results — per engine
        self._cache: Dict[str, Dict] = {
            "custom": {
                "confusion_matrix": None,
                "per_class_metrics": None,
                "test_accuracy": None,
                "macro_f1": None,
            },
            "pytorch": {
                "confusion_matrix": None,
                "per_class_metrics": None,
                "test_accuracy": None,
                "macro_f1": None,
            },
        }

        self._initialized = True

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        model_path: str = "models/model.npz",
        pytorch_model_path: str = "models/pytorch_model.pt",
        norm_stats_path: str = "models/norm_stats.npz",
        prepared_data_path: str = "data/prepared/prepared_data.npz",
    ) -> None:
        """
        Load both models, normalization stats, and run evaluation.
        """

        # 1. Load the custom NumPy network -----------------------------------
        self.network = NeuralNetwork.load(model_path)

        # 2. Load the PyTorch network ----------------------------------------
        pt_path = Path(pytorch_model_path)
        if pt_path.exists():
            self.pytorch_model = AudioClassifier.load(pytorch_model_path)
            self.pytorch_model.eval()
            print(f"🔥 PyTorch model loaded from: {pytorch_model_path}")
        else:
            print(f"⚠️  PyTorch model not found at {pytorch_model_path} — skipping")

        # 3. Load / extract normalization stats -------------------------------
        norm_path = Path(norm_stats_path)
        if norm_path.exists():
            stats = np.load(norm_stats_path, allow_pickle=True)
            self.mean = stats["mean"]
            self.std = stats["std"]
            self.class_names = stats["class_names"].tolist()
            self.num_classes = len(self.class_names)
            print(f"📊 Loaded norm stats from: {norm_stats_path}")
        else:
            prep_path = Path(prepared_data_path)
            if not prep_path.exists():
                raise FileNotFoundError(
                    f"Neither {norm_stats_path} nor {prepared_data_path} found. "
                    "Cannot load normalization stats."
                )

            print(f"📊 Extracting norm stats from: {prepared_data_path}")
            data = np.load(prepared_data_path, allow_pickle=True)
            self.mean = data["mean"]
            self.std = data["std"]
            self.class_names = data["class_names"].tolist()
            self.num_classes = len(self.class_names)

            norm_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                norm_stats_path,
                mean=self.mean,
                std=self.std,
                class_names=np.array(self.class_names),
            )
            print(f"💾 Saved norm stats to: {norm_stats_path}")

        # 4. Spectrogram generator --------------------------------------------
        self.spec_gen = SpectrogramGenerator(
            sample_rate=self.sample_rate,
            target_shape=self.target_shape,
        )

        # 5. Evaluate both models on test set ---------------------------------
        self._run_evaluation(prepared_data_path, "custom")
        if self.pytorch_model is not None:
            self._run_evaluation(prepared_data_path, "pytorch")

        print(f"\n✅ ModelService ready!")
        print(f"   Classes: {self.class_names}")
        print(f"   Custom parameters: {self.network.count_parameters():,}")
        if self.pytorch_model:
            print(f"   PyTorch parameters: {self.pytorch_model.count_parameters():,}")

    # ------------------------------------------------------------------
    # Evaluation (run once at startup per engine)
    # ------------------------------------------------------------------

    def _run_evaluation(self, prepared_data_path: str, engine: str) -> None:
        """Run forward pass on test set and cache confusion matrix + metrics."""
        try:
            prep_path = Path(prepared_data_path)
            if not prep_path.exists():
                print(f"⚠️  No prepared data — skipping {engine} evaluation cache")
                return

            data = np.load(prepared_data_path, allow_pickle=True)
            X_test = data["X_test"]
            y_test = data["y_test"]
            true_classes = np.argmax(y_test, axis=1)

            # Forward pass with the appropriate engine
            if engine == "custom":
                probs = self.network.forward(X_test)
            else:
                self.pytorch_model.eval()
                with torch.no_grad():
                    probs = self.pytorch_model.predict_proba(
                        torch.FloatTensor(X_test)
                    ).numpy()

            pred_classes = np.argmax(probs, axis=1)

            # Accuracy
            cache = self._cache[engine]
            cache["test_accuracy"] = float(np.mean(pred_classes == true_classes))

            # Confusion matrix
            cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
            for t, p in zip(true_classes, pred_classes):
                cm[t, p] += 1
            cache["confusion_matrix"] = cm

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
            cache["per_class_metrics"] = per_class

            f1_scores = [m["f1_score"] for m in per_class.values()]
            cache["macro_f1"] = round(float(np.mean(f1_scores)), 4)

            print(
                f"📈 [{engine}] Evaluation cached: test acc = "
                f"{cache['test_accuracy']:.2%}, macro F1 = {cache['macro_f1']:.4f}"
            )
        except Exception as e:
            print(f"⚠️  Could not cache {engine} evaluation: {e}")

    # ------------------------------------------------------------------
    # Engine availability
    # ------------------------------------------------------------------

    def available_engines(self) -> List[str]:
        """Return list of loaded engines."""
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

        Args:
            audio: 1-D numpy array at self.sample_rate Hz
            engine: "custom" or "pytorch"

        Returns:
            dict with prediction, confidence, all_confidences,
            spectrogram (2-D list), waveform (downsampled), and waveform_summary.
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
        """
        Classify a raw 2-D spectrogram (used by the What-If tool).
        """
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

    def _forward(self, X: np.ndarray, engine: str) -> np.ndarray:
        """Run forward pass on the selected engine, return probabilities as numpy."""
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
        """Architecture, param count, and (optionally) test-set metrics."""
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

        cache = self._cache[engine]

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
        }

        if cache["test_accuracy"] is not None:
            info["test_accuracy"] = round(cache["test_accuracy"], 4)
        if cache["macro_f1"] is not None:
            info["macro_f1"] = cache["macro_f1"]

        return info

    def get_confusion_matrix(self, engine: str = "custom") -> Optional[Dict]:
        engine = self._validate_engine(engine)
        cm = self._cache[engine]["confusion_matrix"]
        if cm is None:
            return None
        return {
            "matrix": cm.tolist(),
            "class_names": self.class_names,
        }

    def get_class_metrics(self, engine: str = "custom") -> Optional[Dict]:
        engine = self._validate_engine(engine)
        cache = self._cache[engine]
        if cache["per_class_metrics"] is None:
            return None
        return {
            "per_class": cache["per_class_metrics"],
            "macro_f1": cache["macro_f1"],
            "test_accuracy": cache["test_accuracy"],
        }

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
        """
        Downsample a waveform to ~target_points for lightweight frontend display.
        """
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
