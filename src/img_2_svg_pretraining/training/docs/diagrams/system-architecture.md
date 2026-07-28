# System Architecture

This diagram shows the main training and inference building blocks in the training stack and how data moves between them. The vision encoder is now a swappable registry component, decoupled from the VLM-internal backbone path.

![System architecture](./system-architecture.svg)

```mermaid
flowchart TD
    subgraph Inputs
        IMG[Document image]
        ANN[Dataset annotations or masks]
        PROMPT[Prompt template]
    end

    subgraph DatasetLayer[Dataset and data module layer]
        DS[Registered dataset adapter]
        VDM[VLM family data module]
        SDM[SAM version data module]
        COL[Composite batch collator]
    end

    subgraph EncoderRegistry[Vision encoder registry]
        VER[VisionEncoderRegistry]
        ENC_NATIVE[native - VLM-internal]
        ENC_EXT[ExtractedVisionEncoder]
        ENC_SWAP[clip / siglip / siglip2 / metaclip / metaclip2 / openvision / openvision2]
        ENC_MOLMO[MolmoVisionBackboneAdapter]
        VER --> ENC_NATIVE
        VER --> ENC_EXT
        VER --> ENC_SWAP
        VER --> ENC_MOLMO
    end

    subgraph Builders[Builder layer]
        B_EXTRACT[extract_vision_encoder]
        B_SWAP[swap_vision_encoder]
        B_BUILD[build_vlm_sam]
    end

    subgraph ModelLayer[Composite model]
        PROC[VLM processor plus tokenizer]
        VE[Active vision encoder]
        VA[vision_adapter nn.Linear - optional dim bridge]
        LLM[LLM decoder]
        PROJ[SEG hidden-state projection]
        SI[SAM image encoder]
        SP[SAM prompt encoder]
        SD[SAM mask decoder]
    end

    subgraph OutputLayer[Outputs]
        TXT[Layout token sequence]
        MSK[Predicted masks]
        BOX[Bounding boxes]
        MET[Validation metrics and visualizations]
    end

    IMG --> DS
    ANN --> DS
    PROMPT --> DS
    DS --> VDM
    DS --> SDM
    VDM --> COL
    SDM --> COL
    COL --> PROC
    PROC --> VE
    VER --> VE
    VE --> VA
    VA --> LLM
    LLM --> TXT
    LLM --> PROJ
    COL --> SI
    PROJ --> SP
    SI --> SD
    SP --> SD
    SD --> MSK
    MSK --> BOX
    TXT --> MET
    BOX --> MET

    B_EXTRACT -.->|detaches and saves| ENC_EXT
    B_SWAP -.->|replaces VE, inserts VA if needed| VE
    B_BUILD -.->|constructs VLMSam + optional swap| ModelLayer
```

## Notes

- `VisionEncoderRegistry` is a distinct sixth registry alongside `DatasetRegistry`, `DataModuleRegistry`, `VLMModelRegistry`, `SAMDataModuleRegistry`, and `SAMModelRegistry`.
- The `native` encoder kind means the VLM-internal vision backbone is used unchanged (no swap).
- `ExtractedVisionEncoder` wraps an `nn.Module` detached from a trained VLM and saved to disk via `extract_vision_encoder`. It is reloaded from `weights.pt` + `preprocessor.json`.
- `MolmoVisionBackboneAdapter` handles cross-encoder swaps for Molmo decoders: it wraps the new encoder alongside the original Molmo `image_projector` and routes through `_molmo_patchify_images` for non-native encoders.
- `vision_adapter` (a `nn.Linear`) is inserted automatically by `swap_vision_encoder` when the new encoder's `embed_dim` differs from the VLM-internal projector's `in_features`. The adapter is stored as `vlm_wrapper.vision_adapter` for optimizer-group routing.
- The VLM-to-SAM projector `text_hidden_fcs_layout` is not affected by encoder swaps because it operates in LLM hidden-state space.

## Source References

- composite model: [training_core/models/vlm_sam.py](../training_core/models/vlm_sam.py)
- VLM wrappers: [training_core/models/vlms/qwen/qwen_model.py](../training_core/models/vlms/qwen/qwen_model.py), [training_core/models/vlms/gemma/gemma_model.py](../training_core/models/vlms/gemma/gemma_model.py), [training_core/models/vlms/molmo/molmo_model.py](../training_core/models/vlms/molmo/molmo_model.py)
- SAM wrapper: [training_core/models/sam/sam1/sam1_model.py](../training_core/models/sam/sam1/sam1_model.py)
- VLM data modules: [training_core/data_modules/vlms/qwen/qwen_data.py](../training_core/data_modules/vlms/qwen/qwen_data.py), [training_core/data_modules/vlms/gemma/gemma_data.py](../training_core/data_modules/vlms/gemma/gemma_data.py), [training_core/data_modules/vlms/molmo/molmo_data.py](../training_core/data_modules/vlms/molmo/molmo_data.py)
- SAM preprocessing: [training_core/data_modules/sam/sam1/sam1_data.py](../training_core/data_modules/sam/sam1/sam1_data.py)
- trainer and metrics: [training_core/train/custom_trainer.py](../training_core/train/custom_trainer.py), [training_core/validation/compute_metrics.py](../training_core/validation/compute_metrics.py)
- vision encoder registry: [training_core/registry/registry.py](../training_core/registry/registry.py)
- vision encoder base and families: [training_core/vision_encoders/](../training_core/vision_encoders/)
- extracted encoder: [training_core/vision_encoders/extracted/extracted_encoder.py](../training_core/vision_encoders/extracted/extracted_encoder.py)
- builders: [training_core/builders/extract_vision_encoder.py](../training_core/builders/extract_vision_encoder.py), [training_core/builders/swap_vision_encoder.py](../training_core/builders/swap_vision_encoder.py), [training_core/builders/build_vlm_sam.py](../training_core/builders/build_vlm_sam.py)
- encoder swap matrix: [training_core/matrix/encoder_swap_matrix.py](../training_core/matrix/encoder_swap_matrix.py)
