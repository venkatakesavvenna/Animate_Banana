from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn


class VisionEncoderBase(ABC, nn.Module):
    """Abstract base for all swappable vision encoders.

    Subclasses wrap a concrete HF or open_clip vision model and expose
    embed_dim (output feature dimension) and preprocessor_config (image
    normalization metadata) so the adapter layer and data pipeline can be
    sized correctly without hard-coding architecture constants.
    """

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Output feature dimension of the vision backbone."""

    @property
    @abstractmethod
    def preprocessor_config(self) -> dict:
        """Image preprocessing metadata.

        Must contain at minimum:
          image_mean  — per-channel normalization means (list of 3 floats)
          image_std   — per-channel normalization stds  (list of 3 floats)
          image_size  — int or (height, width) tuple
        """
