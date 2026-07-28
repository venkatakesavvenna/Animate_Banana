"""Training efficiency: step time, token throughput, encoder fraction, data stall, MFU."""

import logging
import sys
import time
from collections import deque
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

try:
    import wandb as _wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False


def _count_params_full(module: torch.nn.Module) -> int:
    """Count parameters, using ds_numel for ZeRO-3 sharded params."""
    total = 0
    for p in module.parameters():
        total += p.ds_numel if hasattr(p, "ds_numel") else p.numel()
    return total


class TrainingEfficiencyCallback(TrainerCallback):
    """Logs step timing, token throughput, encoder fraction, data stall, and MFU.

    Metrics reported under the ``efficiency/`` W&B namespace:
      step_time_ms                 – rolling-avg wall-clock ms per optimizer step
      steps_per_sec
      samples_per_sec              – samples/s across all GPUs
      data_stall_ms                – wall time between last step end and this step begin
                                     (≈ dataloader fetch + any post-step overhead)
      text_tokens_per_sec          – input_ids tokens/s across all GPUs
      text_tokens_per_sec_per_gpu
      active_tokens_per_sec        – non-ignored label tokens/s across all GPUs
      active_tokens_per_sec_per_gpu
      visual_tokens_per_sec        – encoder output tokens/s across all GPUs
      encoder_time_ms              – total wall ms spent inside encoder forward this step
                                     (includes recompute from gradient checkpointing)
      encoder_time_fraction        – encoder_time_ms / step_time_ms
      peak_gpu_mem_gb              – max GPU memory this step (max across all ranks)
      mfu                          – Model FLOP Utilization (0–1); requires known GPU
      mfu_pct                      – MFU as a percentage

    ``step_token_stats`` is a plain dict shared with ``CustomTrainer.training_step``
    (text + active tokens) and the encoder forward hook (visual tokens).
    """

    # BF16 tensor-core peak TFLOPS; keyed on substrings of torch.cuda.get_device_name()
    _GPU_PEAK_TFLOPS = {
        "H200": 1979,
        "H100": 989, "H800": 989,
        "A100": 312, "A800": 312,
        "A6000": 309, "A30": 330, "A10G": 125, "A10 ": 125,
        "V100": 130,
        "4090": 165, "4080": 97, "3090": 71,
    }

    def __init__(
        self,
        warmup_steps: int = 5,
        log_every: int = 1,
        rolling_window: int = 50,
        hardware_peak_tflops: Optional[float] = None,
    ):
        self._warmup = warmup_steps
        self._log_every = log_every
        self._times: deque = deque(maxlen=rolling_window)
        self._step_start: Optional[float] = None
        self._step_end_clock: Optional[float] = None  # for data stall measurement
        self._data_stall_ms: Optional[float] = None

        # Encoder timing state
        self._enc_module: Optional[torch.nn.Module] = None
        self._enc_hook_handles: List = []
        self._enc_pre_time: Optional[float] = None
        self._enc_time_this_step: float = 0.0
        self._enc_hook_count: int = 0  # reset each step; skip visual count on recompute

        # MFU state (populated in on_train_begin)
        self._n_llm_params: Optional[int] = None
        self._n_enc_params: Optional[int] = None
        self._hardware_peak_tflops: Optional[float] = hardware_peak_tflops

        # Written by CustomTrainer.training_step (text/active/samples) + encoder hook (visual)
        self.step_token_stats: dict = {"text": 0, "active": 0, "samples": 0, "visual": 0}

    # ------------------------------------------------------------------
    # Training lifecycle hooks
    # ------------------------------------------------------------------

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Optional[torch.nn.Module] = None,
        **kwargs,
    ):
        if model is None:
            return

        # Vision encoder: find, hook, and count its parameters
        enc = self._find_encoder(model)
        if enc is not None:
            self._enc_module = enc
            self._enc_hook_handles = [
                enc.register_forward_pre_hook(self._enc_pre_hook),
                enc.register_forward_hook(self._enc_post_hook),
            ]
            _logger.info("[Efficiency] Encoder hook registered on %s.", enc.__class__.__name__)
        else:
            _logger.warning(
                "[Efficiency] Vision encoder not found by attribute walk; "
                "encoder_time_fraction will not be reported."
            )

        # Parameter counts for MFU (use ds_numel for ZeRO-3 sharded params)
        total_params = _count_params_full(model)
        enc_params = _count_params_full(enc) if enc is not None else 0
        self._n_enc_params = enc_params
        self._n_llm_params = total_params - enc_params
        _logger.info(
            "[Efficiency] Params — LLM: %.1fM  Encoder: %.1fM  Total: %.1fM",
            self._n_llm_params / 1e6,
            self._n_enc_params / 1e6,
            total_params / 1e6,
        )

        if self._hardware_peak_tflops is None:
            self._hardware_peak_tflops = self._detect_hardware_peak()
        if self._hardware_peak_tflops:
            _logger.info("[Efficiency] Hardware peak: %.0f TFLOPS (BF16)", self._hardware_peak_tflops)
        else:
            _logger.info("[Efficiency] Hardware peak unknown; MFU will not be reported.")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        for h in self._enc_hook_handles:
            h.remove()
        self._enc_hook_handles = []

    # ------------------------------------------------------------------
    # Per-step hooks
    # ------------------------------------------------------------------

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        now = time.perf_counter()
        if self._step_end_clock is not None:
            self._data_stall_ms = (now - self._step_end_clock) * 1000
        else:
            self._data_stall_ms = None

        self._step_start = now
        self.step_token_stats = {"text": 0, "active": 0, "samples": 0, "visual": 0}
        self._enc_time_this_step = 0.0
        self._enc_hook_count = 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._step_end_clock = time.perf_counter()

        if self._step_start is None:
            return

        step_time = self._step_end_clock - self._step_start
        step = state.global_step

        # Peak GPU memory — all_reduce BEFORE any rank-specific early return
        peak_mem_gb = 0.0
        if torch.cuda.is_available():
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                t = torch.tensor(peak_mem_gb, device="cuda")
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
                peak_mem_gb = t.item()

        if step <= self._warmup:
            return

        if args.process_index != 0:
            return

        self._times.append(step_time)

        if step % self._log_every != 0:
            return

        avg_time = sum(self._times) / len(self._times)
        world_size = args.world_size
        stats = self.step_token_stats

        metrics: dict = {
            "efficiency/step_time_ms": avg_time * 1000,
            "efficiency/steps_per_sec": 1.0 / avg_time,
            "efficiency/samples_per_sec": stats["samples"] * world_size / avg_time,
            "efficiency/peak_gpu_mem_gb": peak_mem_gb,
        }

        if self._data_stall_ms is not None:
            metrics["efficiency/data_stall_ms"] = self._data_stall_ms

        if stats["text"] > 0:
            metrics["efficiency/text_tokens_per_sec"] = stats["text"] * world_size / avg_time
            metrics["efficiency/text_tokens_per_sec_per_gpu"] = stats["text"] / avg_time

        if stats["active"] > 0:
            metrics["efficiency/active_tokens_per_sec"] = stats["active"] * world_size / avg_time
            metrics["efficiency/active_tokens_per_sec_per_gpu"] = stats["active"] / avg_time

        if stats["visual"] > 0:
            metrics["efficiency/visual_tokens_per_sec"] = stats["visual"] * world_size / avg_time

        if self._enc_module is not None and self._enc_time_this_step > 0:
            metrics["efficiency/encoder_time_ms"] = self._enc_time_this_step * 1000
            metrics["efficiency/encoder_time_fraction"] = self._enc_time_this_step / step_time

        mfu = self._compute_mfu(stats, avg_time)
        if mfu is not None:
            metrics["efficiency/mfu"] = mfu
            metrics["efficiency/mfu_pct"] = mfu * 100

        if _WANDB_OK and _wandb.run is not None:
            _wandb.log(metrics, step=step)

        enc_frac_str = (
            f"enc={self._enc_time_this_step / step_time:.1%}"
            if self._enc_module is not None and self._enc_time_this_step > 0
            else "enc=?"
        )
        mfu_str = f"MFU={mfu:.1%}" if mfu is not None else ""
        _logger.info(
            "[Efficiency] step=%d  %.0f ms/step  %.2f steps/s"
            "  %.1f k-text/s/gpu  %.1f k-active/s/gpu  %s  %.2f GB peak  %s",
            step,
            avg_time * 1000,
            1.0 / avg_time,
            stats["text"] / avg_time / 1000 if stats["text"] > 0 else 0.0,
            stats["active"] / avg_time / 1000 if stats["active"] > 0 else 0.0,
            enc_frac_str,
            peak_mem_gb,
            mfu_str,
        )

    # ------------------------------------------------------------------
    # Encoder forward hooks
    # ------------------------------------------------------------------

    def _enc_pre_hook(self, module, args):
        self._enc_pre_time = time.perf_counter()

    def _enc_post_hook(self, module, args, output):
        if self._enc_pre_time is not None:
            self._enc_time_this_step += time.perf_counter() - self._enc_pre_time
            self._enc_pre_time = None

        # Count visual tokens only on the first call per step (forward pass);
        # skip subsequent calls which are gradient-checkpointing recomputes.
        if self._enc_hook_count == 0:
            try:
                self.step_token_stats["visual"] += self._count_visual_tokens(output)
            except Exception:
                pass
        self._enc_hook_count += 1

    @staticmethod
    def _count_visual_tokens(output) -> int:
        """Infer total visual token count from encoder output tensor/object."""
        if isinstance(output, torch.Tensor):
            if output.ndim == 3:       # (B, N_patches, D)
                return output.shape[0] * output.shape[1]
            if output.ndim == 2:       # (B*N_patches, D) packed
                return output.shape[0]
        hs = getattr(output, "last_hidden_state", None)
        if hs is not None and isinstance(hs, torch.Tensor) and hs.ndim == 3:
            return hs.shape[0] * hs.shape[1]
        return 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_encoder(model: torch.nn.Module) -> Optional[torch.nn.Module]:
        """Walk common attribute paths for the vision encoder across VLM families."""
        for attr_path in (
            "backbone.vision_encoder",       # after swap_vision_encoder
            "backbone.vision_backbone",      # Molmo native
            "backbone.visual",               # Qwen2VL
            "backbone.vision_tower",         # Gemma3 / LLaVA-style
            "backbone.model.vision_tower",
        ):
            obj = model
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                backbone = getattr(model, "backbone", None)
                if isinstance(obj, torch.nn.Module) and obj is not backbone:
                    return obj
            except AttributeError:
                continue

        # Fallback: first direct child of backbone whose name looks vision-related
        backbone = getattr(model, "backbone", None)
        if backbone is None:
            return None
        keywords = ("vision", "visual", "encoder", "clip", "vit", "siglip")
        for name, child in backbone.named_children():
            if any(k in name.lower() for k in keywords):
                return child
        return None

    def _detect_hardware_peak(self) -> Optional[float]:
        if not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_name(0)
        for key, tflops in self._GPU_PEAK_TFLOPS.items():
            if key in name:
                return float(tflops)
        return None

    def _compute_mfu(self, stats: dict, avg_time: float) -> Optional[float]:
        """
        MFU = observed_TFLOPS_per_GPU / hardware_peak_TFLOPS_per_GPU.

        Uses the standard 6ND approximation:
          training_FLOPs ≈ 6 × N_params × T_tokens
          (2 for multiply-add, ×3 for forward + backward + optimizer ≈ 3× forward)

        LLM processes text tokens; encoder processes visual tokens.
        Visual token count comes from the encoder output hook.
        """
        if self._hardware_peak_tflops is None or self._n_llm_params is None:
            return None

        T_text = stats["text"]
        T_visual = stats["visual"]

        flops = 6 * (self._n_llm_params * T_text + self._n_enc_params * T_visual)
        if flops == 0:
            return None

        observed_tflops_per_gpu = flops / avg_time / 1e12
        return observed_tflops_per_gpu / self._hardware_peak_tflops
