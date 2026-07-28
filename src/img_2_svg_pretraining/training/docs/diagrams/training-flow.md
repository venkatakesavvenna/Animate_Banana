# Training Flow

This diagram focuses on a single training step, from dataset selection to checkpoint output. It includes the optional vision encoder swap branch introduced in v0.3.x and the encoder×decoder matrix runner path added in v0.4.0.

![Training flow](./training-flow.svg)

```mermaid
flowchart TD
    A[run_pramana.yaml] --> B[train.py]
    B --> C[Resolve datasets.train entries]
    C --> D[DatasetRegistry.get_dataset]
    D --> E[Dataset adapter formats VLM conversation and SAM supervision]
    E --> F[DataModuleRegistry.get_module vlm_family]
    E --> G[SAMDataModuleRegistry.get_sam_data_module sam_version]
    F --> H[LazySupervisedDataset plus VLM collator]
    G --> I[SAM collator and preprocessing]
    H --> J[torchrun and HF Trainer batch]
    I --> J

    B --> ENCCHECK{vision_encoder\nconfig key present?}
    ENCCHECK -->|yes| ENCLOAD[VisionEncoderRegistry.get_encoder\nvision_encoder + vision_encoder_checkpoint]
    ENCLOAD --> DIMCHECK{embed_dim\nmatch?}
    DIMCHECK -->|no| VA[insert vision_adapter\nnn.Linear new_dim to orig_dim]
    DIMCHECK -->|yes| SWAP[swap_vision_encoder\nno adapter needed]
    VA --> SWAP
    SWAP --> VLM[VLM backbone with swapped encoder]
    ENCCHECK -->|no| VLM_NATIVE[VLM backbone with native encoder]
    VLM --> J
    VLM_NATIVE --> J

    J --> K[VLMSam.forward]

    subgraph Losses
        L1[VLM CE loss]
        L2[SAM BCE loss]
        L3[SAM Dice loss]
    end

    K --> L1
    K --> L2
    K --> L3
    L1 --> M[Weighted total loss]
    L2 --> M
    L3 --> M

    M --> N[Backprop and optimizer step]
    N --> O[Periodic validation]
    O --> P[CustomTrainer prediction_step]
    P --> Q[Mask to bbox conversion plus mAP]
    Q --> R[Validation images and metrics]
    N --> S[Checkpoint save]

    %% Matrix runner path - separate from main training flow
    subgraph MatrixRunner[Encoder x decoder matrix runner - separate path]
        MR_SPEC[MatrixRunSpec\ndecoder x encoder x mode]
        MR_YAML[Per-job YAML config generation]
        MR_GPU[8-GPU parallel execution]
        MR_WANDB[W&B logging\nimg-2-svg-pretraining-encoder-swap-matrix]
        MR_CSV[CSV summary\noutputs/encoder_swap_matrix/]
        MR_SPEC --> MR_YAML --> MR_GPU --> MR_WANDB
        MR_GPU --> MR_CSV
    end
```

## Notes

- Multiple training datasets can be listed in `datasets.train`; `build_multi_dataset(...)` resolves them before concatenation.
- The active `vlm_family` owns tokenizer extension, chat formatting, and family-specific positional handling, while `sam_version` owns mask/image preprocessing.
- The optional `vision_encoder` config key triggers an encoder swap via `swap_vision_encoder` after composite construction. When dimensions differ, a `vision_adapter` (`nn.Linear`) is inserted between the new encoder and the VLM-internal projector. The adapter is stored as `vlm_wrapper.vision_adapter` for separate optimizer-group routing.
- For Molmo decoders, the swap goes through `MolmoVisionBackboneAdapter`, which wraps the new encoder alongside the original `image_projector` and routes through `_molmo_patchify_images`.
- Validation uses the custom trainer path so mask predictions can be turned into visualization artifacts and detection metrics.
- The final export is saved to `<output_dir>/final` after `trainer.save_model(...)`.
- The matrix runner (`scripts/run_encoder_swap_matrix.py`) is a separate parallel execution path that drives the 4 × 10 × 2 = 80 job matrix. It does not go through the main `train.py` loop; each job generates its own YAML config and runs independently on an assigned GPU.

Relevant files:

- [training_core/train/train.py](../training_core/train/train.py)
- [training_core/train/train_utils.py](../training_core/train/train_utils.py)
- [training_core/train/custom_trainer.py](../training_core/train/custom_trainer.py)
- [training_core/validation/compute_metrics.py](../training_core/validation/compute_metrics.py)
- [training_core/builders/swap_vision_encoder.py](../training_core/builders/swap_vision_encoder.py)
- [training_core/builders/build_vlm_sam.py](../training_core/builders/build_vlm_sam.py)
- [training_core/matrix/encoder_swap_matrix.py](../training_core/matrix/encoder_swap_matrix.py)
- [scripts/run_encoder_swap_matrix.py](../scripts/run_encoder_swap_matrix.py)
