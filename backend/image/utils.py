"""
utils.py – Image preprocessing and inference utilities for Sentinel-X.
"""

import base64
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
    tensor = _preprocess(image)       # (3, 224, 224)
    return tensor.unsqueeze(0)        # (1, 3, 224, 224)


def run_inference(model: torch.nn.Module, tensor: torch.Tensor) -> dict:
    """
    Run a forward pass and return the top-1 prediction.

    Returns:
        {
            "class_index": int,
            "label":       str,   # human-readable ImageNet class name
            "confidence":  float, # softmax probability of top-1 class
        }
    """
    from .model import get_imagenet_labels
    labels = get_imagenet_labels()

    with torch.no_grad():
        logits = model(tensor)                          # (1, 1000)
        probabilities = torch.softmax(logits, dim=1)    # (1, 1000)
        confidence, class_index = probabilities.max(dim=1)

    idx = int(class_index.item())
    return {
        "class_index": idx,
        "label": labels[idx] if labels else f"Class #{idx}",
        "confidence": round(float(confidence.item()), 6),
    }


def run_top5_inference(model: torch.nn.Module, tensor: torch.Tensor) -> dict:
    """
    Run a forward pass and return top-1 + top-5 predictions.

    Returns:
        {
            "class_index": int,
            "label":       str,
            "confidence":  float,
            "top5": [
                {"class_index": int, "label": str, "confidence": float},
                ...  # 5 entries, sorted descending by confidence
            ]
        }
    """
    from .model import get_imagenet_labels
    labels = get_imagenet_labels()

    with torch.no_grad():
        logits = model(tensor)                              # (1, 1000)
        probabilities = torch.softmax(logits, dim=1)        # (1, 1000)

    # top-5
    top5_conf, top5_idx = probabilities.topk(5, dim=1)     # (1, 5) each
    top5_conf = top5_conf.squeeze(0).tolist()
    top5_idx  = top5_idx.squeeze(0).tolist()

    top5 = [
        {
            "class_index": int(idx),
            "label": labels[int(idx)] if labels else f"Class #{idx}",
            "confidence": round(float(conf), 6),
        }
        for idx, conf in zip(top5_idx, top5_conf)
    ]

    return {
        "class_index": top5[0]["class_index"],
        "label":       top5[0]["label"],
        "confidence":  top5[0]["confidence"],
        "top5":        top5,
    }


def tensor_to_perturbation_b64(
    original: torch.Tensor,
    adversarial: torch.Tensor,
    amplify: float = 10.0,
) -> str:
    """
    Compute the amplified perturbation noise between original and adversarial
    tensors, convert to a PNG image, and return it as a Base64-encoded string
    suitable for embedding as a data URI in HTML.

    Args:
        original:    Clean image tensor (1, 3, 224, 224) in [0, 1]
        adversarial: Perturbed image tensor (1, 3, 224, 224) in [0, 1]
        amplify:     Contrast amplification factor for the noise to make it
                     visually apparent (default: 10x)

    Returns:
        Base64-encoded PNG string (no "data:image/png;base64," prefix)
    """
    with torch.no_grad():
        diff = (adversarial - original).abs()   # (1, 3, 224, 224) in [0,1]
        diff_amplified = (diff * amplify).clamp(0.0, 1.0)
        diff_uint8 = (diff_amplified.squeeze(0) * 255).byte()   # (3, 224, 224)

    # Convert to PIL Image
    np_array = diff_uint8.permute(1, 2, 0).numpy()   # (H, W, C)
    pil_img = Image.fromarray(np_array, mode="RGB")

    # Encode to base64
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def tensor_to_b64_png(tensor: torch.Tensor) -> str:
    """
    Convert a normalised image tensor (1, 3, 224, 224) back to a
    Base64-encoded PNG for inline display.

    Denormalises using ImageNet mean/std before converting.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    with torch.no_grad():
        img = tensor * std + mean
        img = img.clamp(0.0, 1.0)
        img_uint8 = (img.squeeze(0) * 255).byte()  # (3, 224, 224)

    np_array = img_uint8.permute(1, 2, 0).numpy()
    pil_img = Image.fromarray(np_array, mode="RGB")

    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")