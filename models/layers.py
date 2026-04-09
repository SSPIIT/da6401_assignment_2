"""Reusable custom layers."""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout implemented from scratch without nn.Dropout or F.dropout.

    Uses inverted dropout scaling: divides by (1-p) during training to keep
    expected activation magnitudes consistent between train and eval modes.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability. Must be in [0, 1).
        """
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor of any shape.

        Returns:
            Output tensor with dropout applied during training.
        """
        if not self.training or self.p == 0.0:
            return x
        # Generate binary mask: 1 with probability (1-p), 0 with probability p
        keep_prob = 1.0 - self.p
        mask = torch.bernoulli(torch.full_like(x, keep_prob))
        # Inverted dropout: scale by 1/(1-p) so expected value stays the same
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"p={self.p}"