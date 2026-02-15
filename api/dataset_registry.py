"""
Dataset Registry — Configuration and discovery for all supported datasets.

Each dataset is described by a DatasetConfig that specifies:
  - Identity (key, modality, name, description)
  - File paths (models, norm stats, prepared data, training results)
  - Preprocessing parameters (sample rate, duration, target shape)
  - Network architecture defaults (input size, hidden layer sizes)

The DatasetRegistry holds all configs and reports which datasets
have trained models available on disk.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class DatasetConfig:
    """Full configuration for a single dataset."""

    # --- Identity ---
    key: str                          # e.g. "audio/music", "image/medical"
    modality: str                     # "audio" or "image"
    name: str                         # Display name: "Music Genre"
    description: str                  # One-liner for UI tooltips

    # --- Class labels ---
    class_names: List[str] = field(default_factory=list)

    # --- File paths (relative to project root) ---
    model_path: str = ""
    pytorch_model_path: str = ""
    norm_stats_path: str = ""
    prepared_data_path: str = ""
    training_history_path: str = ""
    pytorch_history_path: str = ""
    tsne_path: str = ""
    pytorch_tsne_path: str = ""

    # --- Preprocessing (audio-specific; ignored for images) ---
    sample_rate: int = 22050
    duration: float = 2.0
    target_shape: Tuple[int, int] = (64, 64)

    # --- Network architecture ---
    input_size: int = 4096            # product of target_shape by default
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 64])

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


# ──────────────────────────────────────────────────────────────────────────
#  Pre-defined Dataset Configs
# ──────────────────────────────────────────────────────────────────────────

MUSIC_GENRE = DatasetConfig(
    key="audio/music",
    modality="audio",
    name="Instrument Families",
    description="10-class instrument family classification (NSynth-style)",
    class_names=[
        "bass", "brass", "flute", "guitar", "keyboard",
        "mallet", "organ", "reed", "string", "vocal",
    ],
    # Backward-compatible paths from Phase 2
    model_path="models/model.npz",
    pytorch_model_path="models/pytorch_model.pt",
    norm_stats_path="models/norm_stats.npz",
    prepared_data_path="data/prepared/prepared_data.npz",
    training_history_path="results/metrics/training_history.npz",
    pytorch_history_path="results/metrics/pytorch_training_history.npz",
    tsne_path="results/metrics/tsne_data.npz",
    pytorch_tsne_path="results/metrics/pytorch_tsne_data.npz",
    sample_rate=22050,
    duration=2.0,
    target_shape=(64, 64),
    input_size=4096,
    hidden_sizes=[128, 64],
)

MEDICAL_AUDIO = DatasetConfig(
    key="audio/medical",
    modality="audio",
    name="Medical Audio",
    description="Human body sound classification (breathing, cough, sneeze, snoring, crying)",
    class_names=["breathing", "coughing", "crying_baby", "sneezing", "snoring"],
    model_path="models/audio/medical/model.npz",
    pytorch_model_path="models/audio/medical/pytorch_model.pt",
    norm_stats_path="models/audio/medical/norm_stats.npz",
    prepared_data_path="data/prepared/audio/medical/prepared_data.npz",
    training_history_path="results/metrics/audio/medical/training_history.npz",
    pytorch_history_path="results/metrics/audio/medical/pytorch_training_history.npz",
    tsne_path="results/metrics/audio/medical/tsne_data.npz",
    pytorch_tsne_path="results/metrics/audio/medical/pytorch_tsne_data.npz",
    input_size=4096,
    hidden_sizes=[128, 64],
)

WILDLIFE_AUDIO = DatasetConfig(
    key="audio/wildlife",
    modality="audio",
    name="Wildlife Audio",
    description="Animal sound classification (birds, domestic, wild fauna, farm animals)",
    class_names=["birds", "domestic", "large_farm", "small_farm", "wild_fauna"],
    model_path="models/audio/wildlife/model.npz",
    pytorch_model_path="models/audio/wildlife/pytorch_model.pt",
    norm_stats_path="models/audio/wildlife/norm_stats.npz",
    prepared_data_path="data/prepared/audio/wildlife/prepared_data.npz",
    training_history_path="results/metrics/audio/wildlife/training_history.npz",
    pytorch_history_path="results/metrics/audio/wildlife/pytorch_training_history.npz",
    tsne_path="results/metrics/audio/wildlife/tsne_data.npz",
    pytorch_tsne_path="results/metrics/audio/wildlife/pytorch_tsne_data.npz",
    input_size=4096,
    hidden_sizes=[128, 64],
)

URBAN_SOUNDS = DatasetConfig(
    key="audio/urban",
    modality="audio",
    name="Urban Sounds",
    description="Urban/exterior sound classification (vehicles, sirens, tools, etc.)",
    class_names=[
        "airplane", "car_horn", "chainsaw", "church_bells", "engine",
        "fireworks", "hand_saw", "helicopter", "siren", "train",
    ],
    model_path="models/audio/urban/model.npz",
    pytorch_model_path="models/audio/urban/pytorch_model.pt",
    norm_stats_path="models/audio/urban/norm_stats.npz",
    prepared_data_path="data/prepared/audio/urban/prepared_data.npz",
    training_history_path="results/metrics/audio/urban/training_history.npz",
    pytorch_history_path="results/metrics/audio/urban/pytorch_training_history.npz",
    tsne_path="results/metrics/audio/urban/tsne_data.npz",
    pytorch_tsne_path="results/metrics/audio/urban/pytorch_tsne_data.npz",
    input_size=4096,
    hidden_sizes=[128, 64],
)

MEDICAL_IMAGE = DatasetConfig(
    key="image/medical",
    modality="image",
    name="Medical Imaging",
    description="Skin lesion classification (DermaMNIST — 7 dermatoscopic classes)",
    class_names=[
        "actinic_keratoses", "basal_cell_carcinoma", "benign_keratosis",
        "dermatofibroma", "melanoma", "melanocytic_nevi", "vascular_lesion",
    ],
    model_path="models/image/medical/model.npz",
    pytorch_model_path="models/image/medical/pytorch_model.pt",
    norm_stats_path="models/image/medical/norm_stats.npz",
    prepared_data_path="data/prepared/image/medical/prepared_data.npz",
    training_history_path="results/metrics/image/medical/training_history.npz",
    pytorch_history_path="results/metrics/image/medical/pytorch_training_history.npz",
    tsne_path="results/metrics/image/medical/tsne_data.npz",
    pytorch_tsne_path="results/metrics/image/medical/pytorch_tsne_data.npz",
    sample_rate=0,       # N/A for images
    duration=0.0,        # N/A for images
    target_shape=(64, 64),
    input_size=4096,
    hidden_sizes=[128, 64],
)

WILDLIFE_IMAGE = DatasetConfig(
    key="image/wildlife",
    modality="image",
    name="Pet Breed Classification",
    description="8-breed pet classification from Oxford-IIIT Pet photographs",
    class_names=[
        "abyssinian", "beagle", "bengal", "german_shorthaired",
        "persian", "pug", "samoyed", "siamese",
    ],
    model_path="models/image/wildlife/model.npz",
    pytorch_model_path="models/image/wildlife/pytorch_model.pt",
    norm_stats_path="models/image/wildlife/norm_stats.npz",
    prepared_data_path="data/prepared/image/wildlife/prepared_data.npz",
    training_history_path="results/metrics/image/wildlife/training_history.npz",
    pytorch_history_path="results/metrics/image/wildlife/pytorch_training_history.npz",
    tsne_path="results/metrics/image/wildlife/tsne_data.npz",
    pytorch_tsne_path="results/metrics/image/wildlife/pytorch_tsne_data.npz",
    sample_rate=0,
    duration=0.0,
    target_shape=(64, 64),
    input_size=4096,
    hidden_sizes=[128, 64],
)


# ──────────────────────────────────────────────────────────────────────────
#  Default dataset loaded at startup
# ──────────────────────────────────────────────────────────────────────────

DEFAULT_DATASET = "audio/music"


# ──────────────────────────────────────────────────────────────────────────
#  Registry
# ──────────────────────────────────────────────────────────────────────────

class DatasetRegistry:
    """
    Central registry of all known datasets.

    - Holds DatasetConfig objects keyed by their `key` field.
    - Reports which datasets have trained models on disk ("ready").
    - Provides lookup, listing, and summary helpers for the API layer.
    """

    def __init__(self) -> None:
        self._datasets: Dict[str, DatasetConfig] = {}
        self._register_defaults()

    # ── Registration ──────────────────────────────────────────────────────

    def _register_defaults(self) -> None:
        """Register all built-in dataset configs."""
        for config in [
            MUSIC_GENRE,
            MEDICAL_AUDIO,
            WILDLIFE_AUDIO,
            URBAN_SOUNDS,
            MEDICAL_IMAGE,
            WILDLIFE_IMAGE,
        ]:
            self.register(config)

    def register(self, config: DatasetConfig) -> None:
        """Add or overwrite a dataset config in the registry."""
        self._datasets[config.key] = config

    # ── Lookup ────────────────────────────────────────────────────────────

    def get(self, key: str) -> DatasetConfig:
        """Return config for a dataset key, or raise KeyError."""
        if key not in self._datasets:
            available = ", ".join(sorted(self._datasets.keys()))
            raise KeyError(
                f"Unknown dataset '{key}'. Registered datasets: {available}"
            )
        return self._datasets[key]

    def has(self, key: str) -> bool:
        """Check whether a dataset key is registered."""
        return key in self._datasets

    # ── Availability ──────────────────────────────────────────────────────

    def is_ready(self, key: str) -> bool:
        """True if at least one model file exists on disk for this dataset."""
        config = self._datasets.get(key)
        if config is None:
            return False
        return (
            Path(config.model_path).exists()
            or Path(config.pytorch_model_path).exists()
        )

    # ── Listing ───────────────────────────────────────────────────────────

    def list_all(self) -> List[DatasetConfig]:
        """All registered datasets (whether ready or not)."""
        return list(self._datasets.values())

    def list_available(self) -> List[DatasetConfig]:
        """Only datasets that have at least one trained model on disk."""
        return [c for c in self._datasets.values() if self.is_ready(c.key)]

    # ── Serialization (for API responses) ─────────────────────────────────

    def summary(self) -> List[Dict]:
        """Return a JSON-friendly list of all datasets with readiness status."""
        result = []
        for config in self._datasets.values():
            result.append({
                "key": config.key,
                "modality": config.modality,
                "name": config.name,
                "description": config.description,
                "num_classes": config.num_classes,
                "class_names": config.class_names,
                "ready": self.is_ready(config.key),
            })
        return result
