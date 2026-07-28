import sys
sys.path.insert(0, "/fsxvision_new/anirudh.srinivasan/hf_cache/hub/models--allenai--Molmo-7B-O-0924/snapshots/7a8c4bf80c839c243a6908c6ebbb0f1ee576d7ca")
from modeling_molmo import MolmoForCausalLM
import inspect

class Dummy:
    def __init__(self):
        self.forward = MolmoForCausalLM.forward

sig = inspect.signature(Dummy().forward)
print("Keys:", list(sig.parameters.keys()))
