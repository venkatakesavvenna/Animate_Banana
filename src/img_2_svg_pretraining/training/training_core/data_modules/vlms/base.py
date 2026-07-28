from abc import ABC, abstractmethod

from img_2_svg_pretraining.training.training_core.registry.utils import DataModule


class VLMDataModuleBase(ABC):
    @property
    @abstractmethod
    def family_name(self) -> str:
        """Registry-facing decoder family name."""

    @abstractmethod
    def get_data_module(self, *args, **kwargs) -> DataModule:
        """Return the configured datamodule for this decoder family."""
