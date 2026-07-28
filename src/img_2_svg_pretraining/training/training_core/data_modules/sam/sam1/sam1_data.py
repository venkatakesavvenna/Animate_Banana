from img_2_svg_pretraining.training.training_core.data_modules.sam.base import SAMDataModuleBase
from img_2_svg_pretraining.training.training_core.data_modules.sam.sam_data import format_source_sam, sam_collator
from img_2_svg_pretraining.training.training_core.registry.registry import SAMDataModuleRegistry


class Sam1DataModule(SAMDataModuleBase):
    version = "sam1"

    def format_source(self, image, list_boxes, is_mask=False, image_size=1024) -> dict:
        return format_source_sam(
            original_image=image,
            list_boxes=list_boxes,
            is_mask=is_mask,
            image_size=image_size,
        )

    def get_collator(self):
        return sam_collator


@SAMDataModuleRegistry.register_sam_data_module("sam1")
def get_sam1_data_module(**_kwargs):
    return Sam1DataModule()
