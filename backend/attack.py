"""
attack.py – Adversarial attack utilities.

Implements:
  • FGSM (Fast Gradient Sign Method)  — Goodfellow et al., 2014
  • PGD  (Projected Gradient Descent) — Madry et al., 2017

Both attacks are:
  - Untargeted (maximize loss w.r.t. model's own prediction)
  - Designed for tensors in range [0,1]
  - Safe for models in eval() mode
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# FGSM
# ─────────────────────────────────────────────────────────────

def fgsm_attack(
    model: nn.Module,
    image_tensor: torch.Tensor,
    epsilon: float = 0.03,
) -> torch.Tensor:
    """
    Fast Gradient Sign Method (single-step attack)
    """

    # create tensor with gradient tracking
    perturbed = image_tensor.clone().detach().requires_grad_(True)

    # forward pass
    logits = model(perturbed)

    # pseudo-label (untargeted attack)
    predicted_label = logits.argmax(dim=1)

    # loss
    loss = F.cross_entropy(logits, predicted_label)

    # compute gradients
    model.zero_grad()
    loss.backward()

    # sign of gradient
    gradient_sign = perturbed.grad.sign()

    # apply perturbation
    adversarial = perturbed + epsilon * gradient_sign

    # keep pixels valid
    adversarial = adversarial.clamp(0.0, 1.0).detach()

    return adversarial


# ─────────────────────────────────────────────────────────────
# PGD
# ─────────────────────────────────────────────────────────────

def pgd_attack(
    model: nn.Module,
    image_tensor: torch.Tensor,
    epsilon: float = 0.03,
    alpha: float = 0.005,
    steps: int = 10,
) -> torch.Tensor:
    """
    Projected Gradient Descent Attack (iterative FGSM)

    Includes random initialization inside epsilon-ball
    to produce stronger adversarial examples.
    """

    # keep original image fixed
    original = image_tensor.clone().detach()

    # random start inside epsilon-ball
    x_adv = image_tensor.clone().detach() + torch.empty_like(image_tensor).uniform_(-epsilon, epsilon)
    x_adv = x_adv.clamp(0.0, 1.0)

    for _ in range(steps):

        # enable gradient tracking
        x_adv = x_adv.requires_grad_(True)

        # forward pass
        logits = model(x_adv)

        # pseudo-label (untargeted)
        predicted_label = logits.argmax(dim=1)

        # compute loss
        loss = F.cross_entropy(logits, predicted_label)

        # compute gradient
        model.zero_grad()
        loss.backward()

        gradient_sign = x_adv.grad.sign()

        with torch.no_grad():

            # gradient ascent step
            x_adv = x_adv + alpha * gradient_sign

            # project perturbation into epsilon-ball
            perturbation = (x_adv - original).clamp(-epsilon, epsilon)
            x_adv = original + perturbation

            # keep pixels valid
            x_adv = x_adv.clamp(0.0, 1.0)

    return x_adv.detach()