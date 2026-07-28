import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple
import img_2_svg_pretraining.training.training_core.inference.fastapi as fastapi

# Mocking data structures
@dataclass
class MockProcessor:
    tokenizer = "mock_tokenizer"

@dataclass
class MockDataModule:
    processor = MockProcessor()

@dataclass
class MockDataArgsLayout:
    extract_from_labels_fn = lambda self, x: ["mock_layout"]

@dataclass
class MockDataArgsCaptioning:
    pass  # No extract_from_labels_fn

class MockModelLayout:
    def generate(self, **kwargs) -> Tuple[List[str], List[torch.Tensor]]:
        # Returns mocked generated text and fake mask
        return ["Mock layout output text"], [torch.zeros((1, 10, 10))]

class MockModelCaptioning:
    def generate(self, **kwargs) -> Tuple[List[str], None]:
        # Captioning model does not return masks
        return ["Mock captioning output text"], None

# Mocking external utilities used in run_inference
fastapi.move_to_device = lambda batch, device: batch
fastapi.prepare_generate_batch = lambda batch, tokenizer: batch
fastapi.get_layouts = lambda preds, extract_fn: [["mock_layout"]]
fastapi.to_numpy_mask = lambda x: x.numpy()
fastapi.mask_to_bbox = lambda x: [0, 0, 10, 10]

def test_layout():
    print("Testing Layout Model...")
    model = MockModelLayout()
    data_args = MockDataArgsLayout()
    data_module = MockDataModule()
    
    batch = {"input_ids": torch.tensor([[1, 2, 3]])}
    preds, detections = fastapi.run_inference(model, batch, data_module, data_args, "cpu")
    print(f"Preds: {preds}")
    print(f"Detections: {detections}\n")

def test_captioning():
    print("Testing Captioning Model...")
    model = MockModelCaptioning()
    data_args = MockDataArgsCaptioning()
    data_module = MockDataModule()
    
    batch = {"input_ids": torch.tensor([[1, 2, 3]])}
    preds, detections = fastapi.run_inference(model, batch, data_module, data_args, "cpu")
    print(f"Preds: {preds}")
    print(f"Detections: {detections}\n")

if __name__ == "__main__":
    test_layout()
    test_captioning()
