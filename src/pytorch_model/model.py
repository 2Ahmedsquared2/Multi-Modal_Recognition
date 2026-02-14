"""
PyTorch Audio Classifier
Mirrors the from-scratch NumPy neural network architecture using PyTorch.

Architecture (default):
    Input (batch, 4096)
    → Linear(4096, 128) → ReLU
    → Linear(128, 64)   → ReLU
    → Linear(64, 10)
    Output: raw logits (batch, 10)

Uses the same He (Kaiming) initialization and zero biases as the custom model.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Optional


class AudioClassifier(nn.Module):
    """
    Feedforward Neural Network for multi-class audio classification.

    Identical architecture to the from-scratch NumPy model, but built
    with PyTorch's nn.Module for comparison.
    """

    def __init__(
        self,
        input_size: int = 4096,
        hidden_sizes: Optional[List[int]] = None,
        num_classes: int = 10,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes

        # Build the layer sequence (same as custom NN)
        layers: list[nn.Module] = []
        all_sizes = [input_size] + hidden_sizes + [num_classes]

        for i in range(len(all_sizes) - 1):
            layers.append(nn.Linear(all_sizes[i], all_sizes[i + 1]))
            if i < len(all_sizes) - 2:
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

        # Initialize weights to match the custom model:
        # He (Kaiming) normal initialization + zero biases
        self._init_weights()

    def _init_weights(self):
        """He initialization (Kaiming normal) for Linear layers, zero biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass — returns raw logits (no softmax).

        Use with nn.CrossEntropyLoss during training (it applies LogSoftmax internally).
        For inference probabilities, call predict_proba().
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities (for inference)."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices."""
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)

    def count_parameters(self) -> int:
        """Total trainable parameters (weights + biases)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> None:
        """Print architecture summary matching the custom NN format."""
        print("=" * 65)
        print("PyTorch Neural Network Summary")
        print("=" * 65)
        print(f"{'Layer':<30} {'Output Shape':<18} {'Parameters':>10}")
        print("-" * 65)
        print(f"{'Input':<30} {'(batch, ' + str(self.input_size) + ')':<18} {'0':>10}")

        total_params = 0
        linear_idx = 0

        for module in self.network:
            if isinstance(module, nn.Linear):
                linear_idx += 1
                params = module.weight.numel() + module.bias.numel()
                total_params += params
                name = f"Linear_{linear_idx} ({module.in_features} → {module.out_features})"
                shape = f"(batch, {module.out_features})"
                print(f"{name:<30} {shape:<18} {params:>10,}")
            elif isinstance(module, nn.ReLU):
                print(f"{'ReLU':<30} {'(same)':<18} {'0':>10}")

        print(f"{'Softmax (inference)':<30} {'(batch, ' + str(self.num_classes) + ')':<18} {'0':>10}")
        print("-" * 65)
        print(f"{'Total trainable parameters:':<48} {total_params:>10,}")
        print("=" * 65)

    def save(self, save_path: str) -> str:
        """
        Save model state dict + architecture config.

        Args:
            save_path: Directory or .pt file path

        Returns:
            Path to saved file
        """
        save_path = Path(save_path)

        if not save_path.suffix:
            save_path.mkdir(parents=True, exist_ok=True)
            save_path = save_path / "pytorch_model.pt"
        else:
            save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "input_size": self.input_size,
                "hidden_sizes": self.hidden_sizes,
                "num_classes": self.num_classes,
                "state_dict": self.state_dict(),
            },
            save_path,
        )

        file_size_mb = save_path.stat().st_size / (1024 * 1024)
        print(f"💾 PyTorch model saved to: {save_path} ({file_size_mb:.2f} MB)")
        return str(save_path)

    @classmethod
    def load(cls, load_path: str) -> "AudioClassifier":
        """
        Load a saved PyTorch model.

        Args:
            load_path: Path to the .pt file

        Returns:
            AudioClassifier instance with loaded weights
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"PyTorch model not found: {load_path}")

        checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)

        model = cls(
            input_size=checkpoint["input_size"],
            hidden_sizes=checkpoint["hidden_sizes"],
            num_classes=checkpoint["num_classes"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        print(f"📂 PyTorch model loaded from: {load_path}")
        print(
            f"   Architecture: {model.input_size} → "
            f"{' → '.join(map(str, model.hidden_sizes))} → {model.num_classes}"
        )
        print(f"   Parameters: {model.count_parameters():,}")
        return model
