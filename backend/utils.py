from io import BytesIO

import torch
from PIL import Image
from torchvision import transforms

# Standard ImageNet preprocessing pipeline
_preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Decode raw image bytes, resize to 224x224, and apply
    ImageNet normalisation. Returns a (1, 3, 224, 224) tensor.
    """
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = _preprocess(image)          # (3, 224, 224)
    return tensor.unsqueeze(0)           # (1, 3, 224, 224)


def run_inference(model: torch.nn.Module, tensor: torch.Tensor) -> dict:
    """
    Run a forward pass and return the top-1 prediction.

    Returns:
        {
            "class_index": int,
            "confidence": float,   # softmax probability of top-1 class
        }
    """
    with torch.no_grad():
        logits = model(tensor)                          # (1, 1000)
        probabilities = torch.softmax(logits, dim=1)    # (1, 1000)
        confidence, class_index = probabilities.max(dim=1)

    return {
        "class_index": int(class_index.item()),
        "confidence": round(float(confidence.item()), 6),
    }