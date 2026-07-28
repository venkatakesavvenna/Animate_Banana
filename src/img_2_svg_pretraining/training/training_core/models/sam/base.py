from abc import ABC, abstractmethod

import torch.nn as nn


class SAMModelBase(ABC, nn.Module):
    @property
    @abstractmethod
    def prompt_embed_dim(self) -> int:
        """Prompt-encoder embedding width."""

    @abstractmethod
    def forward(self, images, masks, vlm_seg_hidden_states, resize_list, label_list, mask_counts=None):
        """Decode masks and optionally compute losses."""
