# Runtime Topology

These diagrams show how the repository is intended to run in single-node, multi-node, and encoder-swap matrix environments.

## Single-Node Topology

![Single-node topology](./runtime-single-node.svg)

```mermaid
flowchart LR
    Host[GPU host] --> Docker[Docker container]
    Docker --> Code["/code mount"]
    Docker --> Env["/environments/docgrounding_env"]
    Docker --> Cache[HF cache mount]
    Docker --> Launch[scripts/launch_qwen.sh]
    Launch --> Torchrun[torchrun nproc_per_node=N]
    Torchrun --> Train[img_2_svg_pretraining.training.training_core.train.train]
    Train --> Logs["/code/logs"]
    Train --> Outputs["/code/outputs"]
```

## Multi-Node Topology

![Multi-node topology](./runtime-multi-node.svg)

```mermaid
flowchart TD
    Slurm[Slurm allocation] --> Head[Head node]
    Slurm --> Workers[Worker nodes]
    Head --> Build[Build or load Docker image]
    Build --> Tar[Save image tarball]
    Tar --> Dist[Distribute image to all nodes]
    Dist --> Containers[One training container per node]
    Containers --> Env[Shared env mounts plus data mounts]
    Env --> Torchrun[torchrun with nnodes and node_rank]
    Torchrun --> NCCL[NCCL or EFA distributed backend]
    NCCL --> Train[img_2_svg_pretraining.training.training_core.train.train]
```

## Encoder×Decoder Swap Matrix Runner

This topology covers the 80-job encoder×decoder matrix runner introduced in v0.4.0. It runs as a standalone parallel script rather than through `torchrun` or `train.py`.

```mermaid
flowchart TD
    Script[scripts/run_encoder_swap_matrix.py] --> Specs[MatrixRunSpec enumeration\n4 decoders x 10 encoders x 2 modes = 80 jobs]
    Specs --> YAMLGen[Per-job YAML config generation\noutputs/encoder_swap_matrix/configs/]
    YAMLGen --> Sched[8-GPU parallel scheduler\none job per GPU, queue-driven]
    Sched --> GPU0[GPU 0]
    Sched --> GPU1[GPU 1]
    Sched --> GPU2[GPU 2]
    Sched --> GPUn[GPU N]
    GPU0 --> JobRun[Per-job: build_vlm_sam + optional swap + forward smoke]
    GPU1 --> JobRun
    GPU2 --> JobRun
    GPUn --> JobRun
    JobRun --> WandB[W&B run logged\nimg-2-svg-pretraining-encoder-swap-matrix]
    JobRun --> CSV[CSV summary\noutputs/encoder_swap_matrix/results.csv]
    JobRun --> Report[encoder_swap_matrix_report.md]
```

Subset runs are supported via `--encoders` and `--decoders` flags:

```bash
# Full 80-job matrix with W&B logging
python scripts/run_encoder_swap_matrix.py --report-to wandb

# Subset: only clip and siglip encoders against molmo7b-d decoder
python scripts/run_encoder_swap_matrix.py --encoders clip,siglip --decoders molmo7b-d
```

## Operational Notes

- Single-node training is the most direct supported workflow in this repo snapshot.
- The multi-node launcher is cluster-specific and should be treated as an infrastructure template.
- Both single-node and multi-node paths assume `/code` and `/environments` style mounts and write outputs back to the mounted workspace.
- The encoder swap matrix runner requires `open_clip_torch` (version 3.3.0 or later) to be installed in the active environment for `openvision` and `openvision2` encoder jobs. All other encoder kinds work without it. Install with:
  `pip install open_clip_torch>=3.3.0`
- Extracted encoders (`extracted` kind) require pre-existing `extracted_encoders/<encoder_name>/` directories on disk, created beforehand via the `extract_vision_encoder` builder:
  `python -m img_2_svg_pretraining.training.training_core.builders.extract_vision_encoder --vlm_family molmovlm --vlm_checkpoint <path> --encoder_name extracted-molmo7bd --output_dir extracted_encoders/`

Relevant files:

- [scripts/launch_qwen.sh](../scripts/launch_qwen.sh)
- [scripts/mn_launch_qwen.sh](../scripts/mn_launch_qwen.sh)
- [scripts/run_encoder_swap_matrix.py](../scripts/run_encoder_swap_matrix.py)
- [training_core/matrix/encoder_swap_matrix.py](../training_core/matrix/encoder_swap_matrix.py)
- [docker/init_multinode_docker.sh](../docker/init_multinode_docker.sh)
