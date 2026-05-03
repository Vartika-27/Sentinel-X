"""
network/model.py

Fully connected neural network for binary classification in a
network intrusion detection system.
"""

import torch
import torch.nn as nn


class NetworkMLP(nn.Module):
    """
    Multi-Layer Perceptron for network intrusion detection.

    Performs binary classification on tabular network traffic features,
    predicting whether a connection is benign (0) or malicious (1).

    Architecture:
        Input  → FC(64) → ReLU → Dropout(0.3)
               → FC(32) → ReLU
               → FC(2)  → logits

    Args:
        input_size (int): Number of input features. Must be a positive integer.

    Example:
        >>> model = NetworkMLP(input_size=78)
        >>> x = torch.randn(32, 78)       # batch of 32 samples
        >>> logits = model(x)             # shape: (32, 2)
        >>> probs = torch.softmax(logits, dim=1)
    """

    def __init__(self, input_size: int) -> None:
        super().__init__()

        if input_size <= 0:
            raise ValueError(f"input_size must be a positive integer, got {input_size}.")

        self.network = nn.Sequential(
            # Hidden layer 1
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            # Hidden layer 2
            nn.Linear(64, 32),
            nn.ReLU(),

            # Output layer — raw logits for CrossEntropyLoss
            nn.Linear(32, 2),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Apply He (Kaiming) uniform initialisation to all Linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Raw logits of shape (batch_size, 2).
                          Pass through torch.softmax / torch.argmax for
                          probabilities / predicted class labels.
        """
        return self.network(x)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the predicted class index (0 = benign, 1 = malicious).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Predicted labels of shape (batch_size,).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return class probabilities via softmax.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Probability tensor of shape (batch_size, 2).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
        return torch.softmax(logits, dim=1)