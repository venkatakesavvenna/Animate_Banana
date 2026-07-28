from transformers import AutoModelForCausalLM, AutoConfig
import inspect

model = AutoModelForCausalLM.from_pretrained("/fsxvision_new/anirudh.srinivasan/hf_cache/hub/models--allenai--Molmo-7B-O-0924/snapshots/7a8c4bf80c839c243a6908c6ebbb0f1ee576d7ca", trust_remote_code=True, device_map="cpu", torch_dtype="auto")
sig = inspect.signature(model.forward)
print("Forward signature:", sig.parameters.keys())

