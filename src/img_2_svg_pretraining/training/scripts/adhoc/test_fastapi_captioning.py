import os
from img_2_svg_pretraining.training.training_core.inference.fastapi import load_model, inference_single

BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE_PATH = "/fsxvision_new/raghuveer.r/Layout-Bench/sroie/shard_000/000000189/pages/sroie_000000189_page_1.png"
DEVICE = "cuda:0"

if __name__ == "__main__":
    print("Loading base Qwen model as a generic captioning model (no SAM)...")
    model = load_model(
        checkpoint_path=None,
        base_model=BASE_MODEL,
        sam_checkpoint=None,
        device=DEVICE,
        sam_version="none",
        attn_implementation="sdpa",
    )
    
    print("Running inference...")
    result = inference_single(
        image=IMAGE_PATH,
        prompt="What is the title of the document?",
        model=model,
        device=DEVICE,
        base_model=BASE_MODEL,
        sam_version="none",
    )
    
    print("\n--- Inference Result ---")
    print(f"Generated text: {result['text']}")
    print(f"Number of detections: {len(result['detections'])}")
