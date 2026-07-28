import sys
sys.path.insert(0, 'src')
from omegaconf import OmegaConf
from img_2_svg_pretraining.training.training_core.inference.factory import load_inference_model
import torch

cfg = OmegaConf.load('/code/src/img_2_svg_pretraining/training/configs/mn_molmo_pretrain.yaml')
CHECKPOINT = '/fsxvision_new/venkat.kesav/img_2_svg_pretraining/outputs/molmo_pt_mn/checkpoints/final'
model, _ = load_inference_model(checkpoint_path=CHECKPOINT, model_name_or_path=cfg.base_model, sam_checkpoint=None, device='cpu', vlm_family=cfg.vlm_family, model_family_name=cfg.model_family_name, sam_version=cfg.sam_version)

# Monkey-patch to strip empty DynamicCache
original_prepare = model.backbone.molmo.prepare_inputs_for_generation
def _safe_prepare(input_ids, past_key_values=None, **kwargs):
    if past_key_values is not None:
        try:
            if len(past_key_values) == 0:
                past_key_values = None
        except Exception:
            if hasattr(past_key_values, "get_seq_length") and past_key_values.get_seq_length() == 0:
                past_key_values = None
    return original_prepare(input_ids, past_key_values=past_key_values, **kwargs)

import types
model.backbone.molmo.prepare_inputs_for_generation = types.MethodType(_safe_prepare, model.backbone.molmo)

input_ids = torch.tensor([[1, 2, 3]])
attn_mask = torch.tensor([[1, 1, 1]])

# First try use_cache=False (should work)
print("Running use_cache=False...")
out1 = model.backbone.generate(input_ids=input_ids, attention_mask=attn_mask, max_new_tokens=2, use_cache=False)
print("use_cache=False done.", out1.shape)

# Now try use_cache=True (the real test)
print("Running use_cache=True...")
out2 = model.backbone.generate(input_ids=input_ids, attention_mask=attn_mask, max_new_tokens=2, use_cache=True)
print("use_cache=True done.", out2.shape)
