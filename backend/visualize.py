"""
visualize.py – Visualization utilities for adversarial attack demos.

Provides:
    visualize_attack(original_tensor, adversarial_tensor)
        Renders three side-by-side images:
            Original | Adversarial | Perturbation (amplified difference)
"""

import matplotlib.pyplot as plt
import torch


def _tensor_to_image(tensor: torch.Tensor) -> "np.ndarray":
    """
    Convert a (1, 3, H, W) float32 tensor to a (H, W, 3) NumPy array
    clipped to [0, 1] for safe matplotlib rendering.
    """
    return tensor.squeeze().permute(1, 2, 0).detach().cpu().clamp(0.0, 1.0).numpy()


def visualize_attack(
    original_tensor: torch.Tensor,
    adversarial_tensor: torch.Tensor,
    *,
    amplify_perturbation: float = 10.0,
    figsize: tuple[float, float] = (12.0, 4.0),
    title_fontsize: int = 13,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Display original, adversarial, and perturbation images side-by-side.

    Parameters
    ----------
    original_tensor       : Tensor – clean image,       shape (1, 3, 224, 224)
    adversarial_tensor    : Tensor – perturbed image,   shape (1, 3, 224, 224)
    amplify_perturbation  : float  – scale factor for the difference image so
                                     the (often tiny) perturbation is visible.
                                     Default 10×.
    figsize               : tuple  – overall figure size in inches.
    title_fontsize        : int    – font size for subplot titles.
    save_path             : str | None – if given, save the figure to this path
                                         before returning.

    Returns
    -------
    matplotlib.figure.Figure  – the rendered figure (caller can show / save).
    """
    # ── convert tensors → numpy arrays ────────────────────────────────────
    original_img     = _tensor_to_image(original_tensor)
    adversarial_img  = _tensor_to_image(adversarial_tensor)

    # Raw difference (may be very small); amplify for visibility
    perturbation_raw = (adversarial_tensor - original_tensor).detach().cpu()
    perturbation_img = (
        (perturbation_raw * amplify_perturbation)
        .squeeze()
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
        .numpy()
    )

    # ── layout ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    panels = [
        (axes[0], original_img,    "Original"),
        (axes[1], adversarial_img, "Adversarial"),
        (axes[2], perturbation_img, f"Perturbation (×{amplify_perturbation:g})"),
    ]

    for ax, image, title in panels:
        ax.imshow(image)
        ax.set_title(title, fontsize=title_fontsize, pad=8)
        ax.axis("off")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig