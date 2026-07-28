from abc import ABC, abstractmethod

import torch.nn as nn


class VLMBase(ABC, nn.Module):
    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Hidden-state width consumed by the VLM-to-SAM projector."""

    @abstractmethod
    def resize_token_embeddings(self, vocab_size: int):
        """Resize input/output embeddings after tokenizer extension."""

    @abstractmethod
    def forward(self, **kwargs):
        """Run the VLM forward pass."""

    @abstractmethod
    def generate(self, **kwargs):
        """Run the VLM generation path."""
