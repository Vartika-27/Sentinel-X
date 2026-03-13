import torch
import torch.nn as nn
from torchvision import models

_model = None


def load_model() -> nn.Module:
    global _model
    if _model is None:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        _model = models.resnet18(weights=weights)
        _model.eval()
    return _model


def get_model() -> nn.Module:
    global _model
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model