# Registry System

This module provides two registries for managing datasets and model-specific data modules in the DocGrounding project.

## DatasetRegistry

Manages dataset factory functions that return `DataArguments` objects containing dataset configuration and processing logic.

### Registration

```python
@DatasetRegistry.register_dataset("publaynet")
def get_publaynet_dataargs(debug_path="", seed=42, sam_image_size=1024, 
                           data_path=DATASET_PATH, get_source_fn=format_source):
    ds = load_dataset(data_path)
    data_args = DataArguments(
        dataset_path=DATASET_PATH,
        ds=ds["train"],
        seed=seed,
        get_source=get_source_fn,
        get_source_kwargs={"sam_image_size": sam_image_size, "debug_path": debug_path}
    )
    return data_args
```

**Key Arguments:**
- `get_source_fn`: Function that transforms raw dataset items into model-specific format (e.g., Qwen conversations + SAM masks)
- `sam_image_size`: Resolution for SAM mask generation (default 1024)
- `debug_path`: If provided, saves visualization of dataset inputs for debugging

### Usage

```python
# Import the file to auto-register the dataset
from img_2_svg_pretraining.training.training_core.datasets import publaynet

# Now retrieve from registry
data_args = DatasetRegistry.get_dataset(
    "publaynet", 
    debug_path="/path/to/debug", 
    seed=42, 
    sam_image_size=1024
)
```

## DataModuleRegistry

Manages model-specific data modules (processor, tokenizer, Dataset class, Collator class).

### Registration

```python
@DataModuleRegistry.register_module("qwen3vl")
def get_qwen_module(data_args, change_tokenizer_fn):
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer, seg_token_idx = change_tokenizer_fn(processor.tokenizer)
    
    return DataModule(
        model_name="qwen3vl",
        model_path=model_path,
        processor=processor,
        seg_token_idx=seg_token_idx,
        Dataset=QwenDataset,
        Collator=QwenCollator
    )
```

**Key Arguments:**
- `data_args`: DataArguments from DatasetRegistry containing dataset and `get_source` function
- `change_tokenizer_fn`: Callback to add custom tokens (e.g., `[SEG]`) and return modified tokenizer + token index

### Usage

```python
# Import the file to auto-register the module
from img_2_svg_pretraining.training.training_core.data_modules.qwen import qwen_data

# Now retrieve from registry
qwen_module = DataModuleRegistry.get_module(
    "qwen3vl",
    data_args=data_args,
    change_tokenizer_fn=add_new_tokens
)
```

## Key Benefits

- **Decoupling**: Separates dataset loading from model-specific processing
- **Extensibility**: Easy to add new datasets and models via decorators
- **Consistency**: Standardized interface across different datasets and models
- **Flexibility**: Pass custom arguments at registration and retrieval time

## Architecture

1. **DatasetRegistry** returns `DataArguments` containing raw dataset and a `get_source` function
2. **DataModuleRegistry** returns `DataModule` with model-specific processing components
3. The `get_source` function transforms raw dataset items into model-specific formats (e.g., Qwen conversation format, SAM masks)

This two-level approach allows mixing any registered dataset with any registered model.
