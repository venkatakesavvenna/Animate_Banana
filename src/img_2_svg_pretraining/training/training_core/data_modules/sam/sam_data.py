# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Callable, Dict, Optional, Sequence, List, Tuple, Any
from collections.abc import Sequence
from PIL import Image
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize  # type: ignore
from torchvision.transforms.functional import to_pil_image
import cv2


# imported from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/sam.py
def preprocess(x: torch.Tensor, img_size=1024) -> torch.Tensor:
    """
    Gets input in x: (C,H,W)
    Normalize pixel values and pad to a square input."""
    pixel_mean: List[float] = [123.675, 116.28, 103.53]
    pixel_std: List[float] = [58.395, 57.12, 57.375]

    # Convert to tensors with shape (C,1,1)
    mean = torch.tensor(pixel_mean, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    std = torch.tensor(pixel_std, dtype=x.dtype, device=x.device).view(-1, 1, 1)

    # Normalize colors
    x = (x - mean) / std

    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x
    

# imported from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/utils/transforms.py
class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


def get_mask_from_points(original_image, dual_points):
    """
    input: 
    - For bounding boxes: (x1, y1, x2, y2) or [x1, y1, x2, y2]
    - For polygons: list/array of (x, y) coordinates [[x1,y1], [x2,y2], ...] or [x1,y1,x2,y2,...]
    """
    width, height = original_image.size[:2] # width, height
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Check if input is a bounding box (4 values) or polygon
    if isinstance(dual_points, (list, tuple)):
        if len(dual_points) == 4 and not isinstance(dual_points[0], (list, tuple, np.ndarray)):
            # Bounding box case: (x1, y1, x2, y2)
            x1, y1, x2, y2 = dual_points
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            mask[y1:y2, x1:x2] = 1
        else:
            # Polygon case
            points = np.array(dual_points)
            if points.ndim == 1:
                # Flatten list like [x1, y1, x2, y2, x3, y3, ...]
                points = points.reshape(-1, 2)
            points = points.astype(np.int32)
            cv2.fillPoly(mask, [points], 1)
    elif isinstance(dual_points, np.ndarray):
        # Numpy array input
        if dual_points.size == 4 and dual_points.ndim == 1:
            # Bounding box case
            x1, y1, x2, y2 = dual_points
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            mask[y1:y2, x1:x2] = 1
        else:
            # Polygon case
            points = dual_points
            if points.ndim == 1:
                points = points.reshape(-1, 2)
            points = points.astype(np.int32)
            cv2.fillPoly(mask, [points], 1)
    
    return mask


# gets called insde DataLoader
def format_source_sam(original_image: Image.Image, list_boxes, is_mask = False, image_size=1024):
    """
    List_boxes must be a list/tuple of x1,y1, h,w --> we get x2 = x1+h, y2 = y1+w
    if is_masks = True, then we consider list_boxes as a list of masks
    """
    if isinstance(original_image, Image.Image):
        original_image = original_image.convert("RGB")
        orig_np = np.array(original_image) # H, W, C
    else:
        orig_np = np.array(original_image)


    if is_mask:
        mask_layouts = list_boxes
        # further processing is required, this is just a placeholder as most of our current dataset is boxes only.
    else:
        mask_layouts = []
        for index in range(len(list_boxes)):
            box_or_polygon = list_boxes[index]
            
            # Check if it's a simple 4-value bounding box or a polygon
            if isinstance(box_or_polygon, (list, tuple, np.ndarray)):
                box_array = np.array(box_or_polygon)
                # Check if it's a 4-element bounding box (not nested)
                if box_array.size == 4 and box_array.ndim == 1:
                    # Bounding box in x1,y1,h,w format - convert to x1,y1,x2,y2
                    x1, y1, h, w = box_or_polygon
                    x2, y2 = x1 + h, y1 + w
                    mask = get_mask_from_points(original_image, (x1, y1, x2, y2))
                else:
                    # Polygon or nested coordinates - pass directly
                    mask = get_mask_from_points(original_image, box_or_polygon)
            else:
                # Fallback for unexpected format
                mask = get_mask_from_points(original_image, box_or_polygon)
            
            mask_layouts.append(mask)
    
    masks = [torch.from_numpy(mask).to(dtype=torch.float32) for mask in mask_layouts]

    # # ************ Why????  We are not even using this label********************
    # ignore_label=255
    # label = torch.ones(masks[0].shape[0], masks[0].shape[1]) * ignore_label 

    transform = ResizeLongestSide(target_length=image_size)
    # np.array(PIL_image) --> (H,W,C)
    image_sam = transform.apply_image(np.array(original_image)) # preprocess for sam.. output is (H,W,C)
    resize = image_sam.shape[:2]

    # ************ Why????********************
    image_sam = preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous(), img_size=image_size)
    return {
        "masks": masks,
        # "label": label, # get overwritten
        "image": image_sam, # C,H,W
        "resize": resize, # (H_new,W_new) -> Resized
        "orig_image_size": original_image.size[:2], # (W,H) -> Original
        "original_image": original_image # PIL Image
    }


def _normalize_size_list(
    size_list: torch.Tensor | Sequence[Tuple[int, int]] | None,
) -> list[tuple[int, int]] | None:
    if size_list is None:
        return None
    if isinstance(size_list, torch.Tensor):
        return [tuple(int(value) for value in row.tolist()) for row in size_list]
    return [tuple(int(value) for value in row) for row in size_list]


def pack_mask_batch(
    mask_batches: Sequence[Sequence[torch.Tensor] | None],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert per-sample variable-length mask lists into a single padded tensor.

    The padded `[B, M, H, W]` format keeps sample boundaries intact under
    `torch.nn.DataParallel`, which scatters tensors by the batch dimension.
    """
    batch_size = len(mask_batches)
    mask_counts = torch.tensor(
        [len(sample_masks) if sample_masks is not None else 0 for sample_masks in mask_batches],
        dtype=torch.long,
    )

    max_masks = int(mask_counts.max().item()) if batch_size > 0 else 0
    max_height = 1
    max_width = 1
    for sample_masks in mask_batches:
        if not sample_masks:
            continue
        for mask in sample_masks:
            max_height = max(max_height, int(mask.shape[-2]))
            max_width = max(max_width, int(mask.shape[-1]))

    packed_masks = torch.zeros(
        (batch_size, max_masks, max_height, max_width),
        dtype=torch.float32,
    )

    for batch_idx, sample_masks in enumerate(mask_batches):
        if not sample_masks:
            continue
        for mask_idx, mask in enumerate(sample_masks):
            mask = mask.to(dtype=torch.float32)
            height, width = mask.shape[-2:]
            packed_masks[batch_idx, mask_idx, :height, :width] = mask

    return packed_masks, mask_counts


def unpack_mask_batch(
    masks: torch.Tensor | Sequence[torch.Tensor] | Sequence[Sequence[torch.Tensor]] | None,
    mask_counts: torch.Tensor | Sequence[int] | None = None,
    image_size_list: torch.Tensor | Sequence[Tuple[int, int]] | None = None,
) -> list[torch.Tensor]:
    """
    Recover per-sample `[N_i, H_i, W_i]` tensors from a padded mask batch.
    """
    if masks is None:
        return []

    normalized_sizes = _normalize_size_list(image_size_list)

    if isinstance(masks, torch.Tensor):
        if mask_counts is None:
            counts = [int(masks.shape[1])] * int(masks.shape[0])
        elif isinstance(mask_counts, torch.Tensor):
            counts = [int(value) for value in mask_counts.tolist()]
        else:
            counts = [int(value) for value in mask_counts]

        unpacked_masks: list[torch.Tensor] = []
        for batch_idx, count in enumerate(counts):
            sample_masks = masks[batch_idx, :count]
            if normalized_sizes is not None:
                orig_w, orig_h = normalized_sizes[batch_idx]
                sample_masks = sample_masks[..., :orig_h, :orig_w]
            unpacked_masks.append(sample_masks.contiguous())
        return unpacked_masks

    unpacked_masks = []
    for batch_idx, sample_masks in enumerate(masks):
        if isinstance(sample_masks, torch.Tensor):
            tensor_masks = sample_masks
        else:
            sample_masks = list(sample_masks)
            if sample_masks:
                tensor_masks = torch.stack(
                    [mask.to(dtype=torch.float32) for mask in sample_masks],
                    dim=0,
                )
            else:
                if normalized_sizes is not None:
                    orig_w, orig_h = normalized_sizes[batch_idx]
                    tensor_masks = torch.zeros((0, orig_h, orig_w), dtype=torch.float32)
                else:
                    tensor_masks = torch.zeros((0, 1, 1), dtype=torch.float32)
        if normalized_sizes is not None and tensor_masks.numel() > 0:
            orig_w, orig_h = normalized_sizes[batch_idx]
            tensor_masks = tensor_masks[..., :orig_h, :orig_w]
        unpacked_masks.append(tensor_masks.contiguous())
    return unpacked_masks


# gets called inside Data Collator
def sam_collator(instances: Sequence[Dict], batch):
    """
    the instances are list of each get_item output by Qwen DataLoader by passing it through format_source_sam.
    we now have to add corresponding batched version to the dict batch, for sam_specific additions.
    """
  
    image_list, original_image_list, label_list, resize_list, mask_list = [], [], [], [], []
    for inst in instances:
        image_list.append(inst.get("image"))
        
        original_image = inst.get("original_image")
        original_image_list.append(np.array(original_image))
        
        mask_list.append(inst.get("masks") or [])
        
        orig_w, orig_h = inst.get("orig_image_size")
        resize_h, resize_w = inst.get("resize")
        label_list.append((int(orig_w), int(orig_h)))
        resize_list.append((int(resize_h), int(resize_w)))
    image_list = torch.stack(image_list,dim=0) # (batch, C, H, W)
    packed_masks, mask_counts = pack_mask_batch(mask_list)

    batch["masks"] = packed_masks
    batch["mask_counts"] = mask_counts
    batch["images"] = image_list
    batch["orig_image_size_list"] = torch.tensor(label_list, dtype=torch.long)
    batch["resize_list"] = torch.tensor(resize_list, dtype=torch.long)
    batch["debug_original_image_list"] = original_image_list
    return batch
