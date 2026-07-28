from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class VisionTowerOutput:
    last_hidden_state: torch.Tensor


def get_module_float_dtype(module: torch.nn.Module) -> torch.dtype | None:
    """Return the first floating-point dtype exposed by a module."""
    for tensor in list(module.parameters()) + list(module.buffers()):
        if torch.is_floating_point(tensor):
            return tensor.dtype
    return None


def cast_tensor_to_module_dtype(
    tensor: torch.Tensor,
    module: torch.nn.Module,
) -> torch.Tensor:
    """Cast a floating-point tensor to match a module's dtype when needed."""
    if not torch.is_floating_point(tensor):
        return tensor
    target_dtype = get_module_float_dtype(module)
    if target_dtype is None or tensor.dtype == target_dtype:
        return tensor
    return tensor.to(dtype=target_dtype)


def normalize_encoder_output(output) -> torch.Tensor:
    """Normalize HF / custom vision outputs to a `[B, seq, dim]` tensor."""
    if hasattr(output, "last_hidden_state"):
        output = output.last_hidden_state
    elif isinstance(output, (tuple, list)):
        output = output[0]

    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Unsupported vision encoder output type: {type(output)!r}")

    if output.ndim == 2:
        output = output.unsqueeze(1)
    if output.ndim != 3:
        raise ValueError(f"Expected a 3-D vision output, got shape {tuple(output.shape)}")
    return output


def resize_token_sequence(
    tokens: torch.Tensor,
    *,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Resize a single `[seq, dim]` token sequence to the requested 2-D grid."""
    if tokens.ndim != 2:
        raise ValueError(f"Expected `[seq, dim]`, got shape {tuple(tokens.shape)}")

    target_seq = target_h * target_w
    seq_len, hidden = tokens.shape
    if seq_len == target_seq:
        return tokens

    if seq_len == 1:
        return tokens.expand(target_seq, hidden)

    grid_tokens = tokens
    if _is_square(seq_len - 1):
        grid_tokens = tokens[1:]
        seq_len = grid_tokens.shape[0]

    if not _is_square(seq_len):
        raise ValueError(
            f"Cannot resize token sequence of length {tokens.shape[0]} to "
            f"{target_h}x{target_w}; sequence is not a square grid"
        )

    src_side = int(math.isqrt(seq_len))
    grid = grid_tokens.transpose(0, 1).reshape(1, hidden, src_side, src_side)
    resized = F.interpolate(grid.float(), size=(target_h, target_w), mode="bilinear", align_corners=False)
    resized = resized.to(dtype=tokens.dtype)
    return resized.reshape(hidden, target_seq).transpose(0, 1).contiguous()


def qwen_patch_tokens_to_images(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    patch_size: int,
    temporal_patch_size: int,
    in_channels: int,
) -> torch.Tensor:
    """Reconstruct square BCHW images from Qwen patchified vision inputs."""
    if pixel_values.ndim != 2:
        raise ValueError(f"Expected Qwen patchified pixel_values to be rank 2, got {tuple(pixel_values.shape)}")

    split_sizes = [int(value) for value in grid_thw.prod(dim=-1).detach().cpu().tolist()]
    chunks = torch.split(pixel_values, split_sizes, dim=0)
    images = []

    for chunk, thw in zip(chunks, grid_thw):
        t, grid_h, grid_w = [int(v) for v in thw.detach().cpu().tolist()]
        patches = chunk.reshape(t, grid_h, grid_w, in_channels, temporal_patch_size, patch_size, patch_size)
        # Images are represented as duplicated temporal patches; average out that axis
        # and collapse any extra temporal slices to one 2-D image.
        patches = patches.mean(dim=4)
        if t > 1:
            patches = patches.mean(dim=0)
        else:
            patches = patches[0]
        image = patches.permute(2, 0, 3, 1, 4).reshape(
            in_channels,
            grid_h * patch_size,
            grid_w * patch_size,
        )
        images.append(image)

    return torch.stack(images, dim=0)


def qwen_images_to_patch_tokens(
    pixel_values: torch.Tensor,
    *,
    patch_size: int,
    temporal_patch_size: int,
    in_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patchify BCHW images into Qwen visual-tower inputs plus `grid_thw`."""
    if pixel_values.ndim == 3:
        pixel_values = pixel_values.unsqueeze(0)
    if pixel_values.ndim != 4:
        raise ValueError(f"Expected BCHW images, got shape {tuple(pixel_values.shape)}")

    batch, channels, height, width = pixel_values.shape
    if channels != in_channels:
        raise ValueError(f"Expected {in_channels} input channels, got {channels}")
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Image size {height}x{width} is not divisible by Qwen patch size {patch_size}"
        )

    grid_h = height // patch_size
    grid_w = width // patch_size
    video = pixel_values.unsqueeze(2).repeat(1, 1, temporal_patch_size, 1, 1)
    patches = video.reshape(
        batch,
        channels,
        1,
        temporal_patch_size,
        grid_h,
        patch_size,
        grid_w,
        patch_size,
    )
    patches = patches.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    patch_tokens = patches.reshape(batch * grid_h * grid_w, channels * temporal_patch_size * patch_size * patch_size)
    grid_thw = torch.tensor(
        [[1, grid_h, grid_w]] * batch,
        device=pixel_values.device,
        dtype=torch.long,
    )
    return patch_tokens, grid_thw


def _is_square(value: int) -> bool:
    if value <= 0:
        return False
    root = math.isqrt(value)
    return root * root == value
