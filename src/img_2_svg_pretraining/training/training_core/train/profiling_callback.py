"""Opt-in PyTorch Profiler integration and NVTX range markers for Nsight Systems.

Enable by adding a ``profiling:`` block to the training config:

```yaml
profiling:
  enabled: true
  wait_steps: 1      # steps to skip before warmup
  warmup_steps: 1    # profiler warmup steps
  active_steps: 3    # steps to actively profile and record
  with_stack: false  # include Python stack traces (adds overhead, better flame graphs)
  nvtx: true         # register NVTX ranges on encoder and backbone submodules
  output_dir: null   # defaults to {logging_dir}/profiler_traces
```

Trace output:
  TensorBoard: open ``{output_dir}/`` with TensorBoard → PyTorch Profiler plugin
  Nsight:      launch with ``nsys profile --trace cuda,nvtx,osrt torchrun ...``

The ``training_step`` NVTX range (in CustomTrainer) is always registered and is a
no-op outside a profiler — it costs nothing without Nsight or torch.profiler.
"""

import logging
import os
import sys
from typing import List, Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    _logger.addHandler(_h)
    _logger.propagate = False

_nvtx = getattr(torch.cuda, "nvtx", None)


class ProfilingCallback(TrainerCallback):
    """Profiles training steps with torch.profiler and adds NVTX range markers.

    torch.profiler trace:
      Profiler runs only on rank 0 (process_index == 0) so only one trace file
      is written. Open the output directory with TensorBoard's PyTorch Profiler
      plugin: ``tensorboard --logdir {output_dir}``.

    NVTX markers (all ranks):
      NVTX hooks are registered on all ranks so the full multi-GPU GPU timeline
      is annotated in Nsight Systems. Ranges registered:
        ``training_step``   — from CustomTrainer.training_step (always-on in that file)
        ``backbone_forward``— around VLM backbone forward
        ``encoder_forward`` — around vision encoder forward (auto-discovered)

    Nsight Systems launch example:
      nsys profile \\
        --trace cuda,nvtx,osrt \\
        --cuda-memory-usage true \\
        --output ./nsys_profile \\
        torchrun --nproc_per_node 8 -m img_2_svg_pretraining.training.training_core.train.train
    """

    def __init__(
        self,
        wait_steps: int = 1,
        warmup_steps: int = 1,
        active_steps: int = 3,
        output_dir: str = "./profiler_traces",
        with_stack: bool = False,
        nvtx: bool = True,
    ):
        self._wait = wait_steps
        self._warmup = warmup_steps
        self._active = active_steps
        self._output_dir = output_dir
        self._with_stack = with_stack
        self._nvtx_enabled = nvtx
        self._prof = None
        self._total_steps = wait_steps + warmup_steps + active_steps
        self._step_count = 0
        self._hook_handles: List = []

    # ------------------------------------------------------------------
    # Training lifecycle
    # ------------------------------------------------------------------

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Optional[torch.nn.Module] = None,
        **kwargs,
    ):
        # NVTX hooks: register on ALL ranks so the full GPU timeline is annotated
        if self._nvtx_enabled and model is not None and _nvtx is not None:
            self._register_nvtx_hooks(model)

        # torch.profiler: rank 0 only (one trace file is sufficient for analysis)
        if args.process_index != 0:
            return

        from torch.profiler import (
            ProfilerActivity,
            profile,
            schedule,
            tensorboard_trace_handler,
        )

        os.makedirs(self._output_dir, exist_ok=True)
        self._prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=self._wait,
                warmup=self._warmup,
                active=self._active,
                repeat=1,
            ),
            on_trace_ready=tensorboard_trace_handler(self._output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=self._with_stack,
        )
        self._prof.start()
        _logger.info(
            "[Profiling] Profiler started (wait=%d, warmup=%d, active=%d steps). "
            "Trace will be written to: %s",
            self._wait,
            self._warmup,
            self._active,
            self._output_dir,
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self._prof is None or args.process_index != 0:
            return

        self._prof.step()
        self._step_count += 1

        if self._step_count >= self._total_steps:
            self._prof.stop()
            self._prof = None
            _logger.info(
                "[Profiling] Complete after %d steps. "
                "Open trace with: tensorboard --logdir %s",
                self._step_count,
                self._output_dir,
            )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self._prof is not None and args.process_index == 0:
            self._prof.stop()
            self._prof = None
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    # ------------------------------------------------------------------
    # NVTX registration
    # ------------------------------------------------------------------

    def _register_nvtx_hooks(self, model: torch.nn.Module):
        def make_hooks(label: str):
            def pre(module, args):
                _nvtx.range_push(label)

            def post(module, args, output):
                _nvtx.range_pop()

            return pre, post

        count = 0

        # Full backbone (captures LLM + connector + encoder in one range)
        backbone = getattr(model, "backbone", None)
        if backbone is not None:
            pre, post = make_hooks("backbone_forward")
            self._hook_handles.append(backbone.register_forward_pre_hook(pre))
            self._hook_handles.append(backbone.register_forward_hook(post))
            count += 1

        # Vision encoder sub-range (nested inside backbone_forward)
        from img_2_svg_pretraining.training.training_core.train.training_efficiency_callback import TrainingEfficiencyCallback

        enc = TrainingEfficiencyCallback._find_encoder(model)
        if enc is not None:
            pre, post = make_hooks("encoder_forward")
            self._hook_handles.append(enc.register_forward_pre_hook(pre))
            self._hook_handles.append(enc.register_forward_hook(post))
            count += 1
            _logger.info(
                "[Profiling] NVTX encoder hook on %s (rank %d).",
                enc.__class__.__name__,
                torch.distributed.get_rank()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 0,
            )

        if count:
            _logger.info("[Profiling] %d NVTX module hooks registered.", count)
