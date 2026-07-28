"""No-op SAM data module — registered as 'none' for VLM-only training.

When sam_version is 'none', no SAM head is loaded and no mask supervision is
provided.  The format_source method returns a structurally valid dict so that
dataset adapters (doclaynet, docsyn, etc.) can call it without branching.  The
collator is a no-op that leaves the batch unchanged; VLMOnly.forward pops the
absent SAM keys with pop(..., None) so no KeyErrors propagate.
"""
from img_2_svg_pretraining.training.training_core.data_modules.sam.base import SAMDataModuleBase
from img_2_svg_pretraining.training.training_core.registry.registry import SAMDataModuleRegistry


def _noop_sam_collator(instances, batch):
    return batch


class NoopSAMDataModule(SAMDataModuleBase):
    """SAM data module that produces no mask supervision."""

    def format_source(self, image, list_boxes, is_mask=False, image_size=1024):
        orig_size = image.size if hasattr(image, "size") else (image_size, image_size)
        return {
            "masks": [],
            "image": None,
            "resize": (image_size, image_size),
            "orig_image_size": orig_size,
            "original_image": image,
        }

    def get_collator(self):
        return _noop_sam_collator


@SAMDataModuleRegistry.register_sam_data_module("none")
def _get_noop_sam_data_module(**_kwargs):
    return NoopSAMDataModule()
