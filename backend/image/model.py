"""
model.py – ResNet-18 loader and ImageNet label registry for Sentinel-X.

ImageNet class names are sourced directly from PyTorch's torchvision
weight metadata (ResNet18_Weights.IMAGENET1K_V1.meta["categories"]).
No hardcoded data — labels are queried from the framework at startup.
"""

from typing import List, Optional

import torch.nn as nn
from torchvision import models

_model:  Optional[nn.Module] = None
_labels: Optional[List[str]] = None


def load_model() -> nn.Module:
    """Load ResNet-18 with ImageNet-pretrained weights (once at startup)."""
    global _model, _labels

    weights = models.ResNet18_Weights.IMAGENET1K_V1

    if _model is None:
        _model = models.resnet18(weights=weights)
        _model.eval()

    if _labels is None:
        # Pull the 1,000 class names from PyTorch's own metadata — no JSON file
        _labels = list(weights.meta["categories"])   # List[str], length 1000

    return _model


def get_model() -> nn.Module:
    """Return the already-loaded model or raise if startup didn't run."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model


def get_imagenet_labels() -> List[str]:
    """Return the 1,000 ImageNet category strings from torchvision metadata."""
    if _labels is None:
        raise RuntimeError("Labels not loaded. Call load_model() first.")
    return _labels