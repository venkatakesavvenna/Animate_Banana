from abc import ABC, abstractmethod


class SAMDataModuleBase(ABC):
    @abstractmethod
    def format_source(self, image, list_boxes, is_mask=False, image_size=1024) -> dict:
        """Prepare SAM-specific sample inputs."""

    @abstractmethod
    def get_collator(self):
        """Return the SAM-specific collator."""
