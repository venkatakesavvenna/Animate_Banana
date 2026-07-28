<div align="center">
  <img src="assets/logo.svg" alt="img_2_svg_pretraining training logo" width="100"/>

  <div>
    <h1 style="margin: 0 0 8px 0;">
      <span style="color: #1f77b4;">img_2_svg_pretraining</span>
      <span style="color: #817d7d;">/</span>
      <span style="color: #ff8c00;">training</span>
    </h1>

  **Modular multimodal document-understanding training stack**

  [![Quickstart](https://img.shields.io/badge/quickstart-guide-blue)](QUICKSTART.md)
  [![Docs](https://img.shields.io/badge/docs-full%20reference-informational)](docs/architecture.md)
  [![Changelog](https://img.shields.io/badge/changelog-latest-green)](CHANGELOG.md)
  [![CI](https://img.shields.io/badge/CI-GitHub%20Actions-lightgrey)](.github/workflows)
  [![Tests](https://img.shields.io/badge/tests-pytest-yellow)](tests)
</div>

---
<div align="left">

This training stack trains a composite **VLM + SAM** model for layout-grounding and document-understanding workloads: a registered vision-language model backbone (Qwen2.5-VL, Gemma3, or Molmo) predicts structure tokens, and a segmentation head turns those cues into masks and bounding boxes. Vision encoders are swappable through a registry, so any supported decoder can be paired with any supported encoder.

## Get started

| I want to... | Go to |
|---|---|
| Launch my first training run | **[QUICKSTART.md](QUICKSTART.md)** |
| Read the full architecture, registries, and config reference | [docs/architecture.md](docs/architecture.md) |
| See what changed recently | [CHANGELOG.md](CHANGELOG.md) |
| Add a dataset, VLM family, or vision encoder | [docs/architecture.md § Extending The Repo](docs/architecture.md#extending-the-repo) |
| Run the test suite | [docs/architecture.md § Tests](docs/architecture.md#tests) |

## At a glance

- **Decoders:** Qwen2.5, Gemma3, Olmo (D/O/1B-MoE variants)
- **Vision encoders:** 10 registered kinds — CLIP, SigLIP, SigLIP2, MetaCLIP (v1/v2), OpenVision, and more — freely swappable per decoder
- **Datasets:** 47 registered keys across layout, PixMo, and public/academic benchmarks
- **Training:** single-node and multi-node via `torchrun` + DeepSpeed ZeRO-3
- **Monitoring:** always-on throughput/MFU metrics, opt-in `torch.profiler` + Nsight integration

This is a code under development — see [Known Gaps And Caveats](docs/architecture.md#known-gaps-and-caveats) in the full docs for the current rough edges.
</div>
