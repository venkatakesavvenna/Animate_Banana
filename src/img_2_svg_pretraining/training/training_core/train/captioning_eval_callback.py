"""captioning_eval_callback — generation-based captioning evaluation callback.

Runs model.generate() on a fixed subset of the validation set every N eval
steps, computes BLEU-4 via sacrebleu, and logs examples as a W&B table.

Config block (in the top-level YAML):

  captioning_eval:
    eval_steps: 500          # run every N global steps (default: every eval)
    num_samples: 32          # how many val samples to generate on
    max_new_tokens: 256      # generation budget
    dataset_key: pixmo_cap   # optional: restrict to a named val dataset

Usage — add to run_training after trainer creation:
  if captioning_cfg := cfg.get("captioning_eval"):
      from img_2_svg_pretraining.training.training_core.train.captioning_eval_callback import GenerativeEvalCallback
      trainer.add_callback(GenerativeEvalCallback(
          val_dataset=val_dataset,
          processor=qwen_data_module.processor,
          cfg=captioning_cfg,
      ))

  # For git + artifact logging at run start:
  from img_2_svg_pretraining.training.training_core.train.captioning_eval_callback import RunMetadataCallback
  trainer.add_callback(RunMetadataCallback(artifact_paths=[config_path, script_path]))
"""
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Subset
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

import sys

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    _logger.addHandler(handler)
    _logger.propagate = False

try:
    import sacrebleu as _sacrebleu
    _SACREBLEU_AVAILABLE = True
except ImportError:
    _SACREBLEU_AVAILABLE = False
    _logger.warning("sacrebleu not installed — BLEU-4 will not be computed. pip install sacrebleu")

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


@dataclass
class CaptioningEvalConfig:
    eval_steps: Optional[int] = None
    num_samples: int = 32
    max_new_tokens: int = 256
    dataset_key: Optional[str] = None
    temperature: float = 1.0
    do_sample: bool = False


def _parse_cfg(cfg) -> CaptioningEvalConfig:
    """Accept OmegaConf DictConfig, plain dict, or dataclass."""
    if isinstance(cfg, CaptioningEvalConfig):
        return cfg
    d = dict(cfg)
    return CaptioningEvalConfig(
        eval_steps=d.get("eval_steps"),
        num_samples=int(d.get("num_samples", 32)),
        max_new_tokens=int(d.get("max_new_tokens", 256)),
        dataset_key=d.get("dataset_key"),
        temperature=float(d.get("temperature", 1.0)),
        do_sample=bool(d.get("do_sample", False)),
    )


def _make_eval_collator(base_collate_fn):
    """Wrap a base collate_fn to also pass through reference texts and raw images.

    Dataset items are expected to have a 'messages' field in HF format
    (role: user/assistant, content: list of typed items). This is the format
    produced by GenericVisionLanguageDataset for all VLM data modules.
    """
    if base_collate_fn is None:
        return None

    def eval_collate(instances):
        batch = base_collate_fn(instances)

        reference_texts, raw_images = [], []
        for inst in instances:
            ref, img = "", None
            for msg in inst.get("messages", []):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "assistant" and not ref:
                    if isinstance(content, list):
                        ref = " ".join(
                            c.get("text", "") for c in content
                            if isinstance(c, dict) and c.get("type") == "text"
                        )
                    else:
                        ref = str(content)
                elif role == "user" and img is None:
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "image":
                                raw = item.get("image")
                                if hasattr(raw, "convert"):  # PIL Image
                                    img = raw
                                break
            reference_texts.append(ref)
            raw_images.append(img)

        batch["reference_texts"] = reference_texts
        batch["raw_images"] = raw_images
        return batch

    return eval_collate


def _truncate_and_left_pad_for_generation(
    batch: Dict[str, Any], pad_token_id: int, ignore_index: int = -100
) -> Dict[str, Any]:
    """Strip the baked-in ground-truth answer out of input_ids and left-pad.

    The base collate_fn (MolmoCollator, Qwen's DataCollatorForSupervisedDataset,
    ...) is the TRAINING collator: it tokenizes the full conversation — prompt
    plus the assistant's ground-truth answer — into one input_ids sequence, and
    marks the prompt prefix as ignore_index in labels so only the answer
    contributes to the loss. Reusing that collator's input_ids as-is for
    generation means the model is handed a prompt that already ends with the
    correct answer and is asked to continue AFTER it — it isn't generating the
    answer, it's extrapolating past a complete example. That's what produces
    degenerate continuations (hallucinated repeats of chat-template role
    markers like "Assistant: ...", or looping filler) instead of a real caption.

    Since labels marks the prompt as a contiguous ignore_index prefix followed
    by the real answer token ids, the position of the first non-ignored label
    IS the prompt length for that sample — no per-family tokenization hook
    needed. We derive everything from labels rather than attention_mask:
    Molmo's collator (MolmoCollator -> merge_molmo_features -> the "none"
    SAM collator) never produces an attention_mask key at all, so a version
    of this function that bails out when attention_mask is absent is a
    silent no-op for Molmo — it never truncates or left-pads anything, and
    the ground-truth answer keeps leaking into generate() regardless of this
    function existing. labels is always present (it's how the loss is
    computed), so we use it as the single source of truth for both the
    prompt boundary (first non-ignored label) and the total real-content
    length (one past the last non-ignored label — labels is right-padded
    with ignore_index the same way input_ids is right-padded with
    pad_token_id, so this is also where real content ends).

    We always freshly construct the output attention_mask (rather than
    reusing/truncating any input one) because generate() needs a correct
    mask to compute position_ids for left-padded content — without one,
    Molmo's manual decode loop falls back to no position offsetting at all,
    which is only harmless under right-padding (real content already starts
    at position 0) and actively wrong once the content is left-padded.

    Batched decoder-only generation requires every sample's real content to
    end at the same column so that slicing generated continuations at a
    single input_seq_len works for the whole batch; that only holds under
    left padding, which is why we left-pad here regardless of what padding
    convention (if any) the base collate_fn used.
    """
    if "input_ids" not in batch:
        return batch

    ids_t = batch["input_ids"]
    batch_size, seq_len = ids_t.shape[0], ids_t.shape[1]
    labels = batch.get("labels")
    attn = batch.get("attention_mask")

    if labels is not None:
        real_mask = labels != ignore_index
        has_answer = real_mask.any(dim=1)
        prompt_lens = real_mask.float().argmax(dim=1)
        # one past the LAST non-ignored label == where real content (prompt+answer) ends
        last_real_from_end = real_mask.flip(dims=[1]).float().argmax(dim=1)
        content_lens = seq_len - last_real_from_end
        fallback_lens = attn.sum(dim=1) if attn is not None else torch.full(
            (batch_size,), seq_len, dtype=torch.long, device=ids_t.device
        )
        real_lens = torch.where(has_answer, prompt_lens, fallback_lens)
    elif attn is not None:
        real_lens = attn.sum(dim=1)
    else:
        real_lens = torch.full((batch_size,), seq_len, dtype=torch.long, device=ids_t.device)
    real_lens = real_lens.tolist()

    new_max_len = max(int(n) for n in real_lens) if real_lens else seq_len

    def shift_row(row, n, pad_val):
        new_row = row.new_full((new_max_len,), pad_val)
        if n == 0:
            return new_row
        new_row[new_max_len - n:] = row[:n]
        return new_row

    out = dict(batch)
    out["input_ids"] = torch.stack(
        [shift_row(ids_t[i], int(real_lens[i]), pad_token_id) for i in range(batch_size)], dim=0
    )

    new_attn = torch.zeros((batch_size, new_max_len), dtype=torch.long, device=ids_t.device)
    for i, n in enumerate(real_lens):
        if n > 0:
            new_attn[i, new_max_len - int(n):] = 1
    out["attention_mask"] = new_attn

    # image_input_idx is NOT sequence-aligned like input_ids/attention_mask —
    # its shape is [B, num_crops, patches_per_crop] (e.g. [B, 6, 144]), and
    # each entry is an ABSOLUTE TEXT POSITION (an index into input_ids) where
    # that patch's embedding should be scattered, or -1 for a padding slot.
    # Truncating/left-padding it like a sequence tensor would silently slice
    # along the crops dimension instead of the text dimension. We keep its
    # shape untouched and only shift the *values* of valid (>=0) entries by
    # each sample's own left-pad offset, since the text positions they point
    # to moved by exactly that amount.
    if "image_input_idx" in batch:
        idx = batch["image_input_idx"].clone()
        for i, n in enumerate(real_lens):
            offset = new_max_len - int(n)
            if offset == 0:
                continue
            valid = idx[i] >= 0
            idx[i][valid] += offset
        out["image_input_idx"] = idx
    return out


def _resize_for_wandb(pil_image, max_px: int = 512):
    """Downscale a PIL image so its longest edge ≤ max_px (preserves aspect)."""
    w, h = pil_image.size
    if max(w, h) <= max_px:
        return pil_image
    scale = max_px / max(w, h)
    return pil_image.resize((int(w * scale), int(h * scale)))


class RunMetadataCallback(TrainerCallback):
    """Log git commit/branch and upload launch files to W&B at training start.

    Args:
        artifact_paths: Files to upload (e.g. YAML config, SLURM launch script).
    """

    def __init__(self, artifact_paths: Optional[List[str]] = None):
        self._artifact_paths = artifact_paths or []

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if args.process_index != 0:
            return
        if not (_WANDB_AVAILABLE and _wandb.run is not None):
            return

        # Git info — prefer env vars injected by the SLURM launch script (captured
        # on the host where git is available); fall back to subprocess inside container.
        git_info = {}
        for env_key, cfg_key in [("GIT_COMMIT", "git_commit"), ("GIT_BRANCH", "git_branch")]:
            val = os.environ.get(env_key, "").strip()
            if val and val != "unknown":
                git_info[cfg_key] = val
        if not git_info:
            import subprocess
            for cfg_key, cmd in [
                ("git_commit", ["git", "rev-parse", "HEAD"]),
                ("git_branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            ]:
                try:
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=10,
                        cwd=os.path.dirname(os.path.abspath(__file__)),
                    )
                    if result.returncode == 0:
                        git_info[cfg_key] = result.stdout.strip()
                except Exception:
                    pass
        dirty = os.environ.get("GIT_DIRTY", "").strip()
        if dirty:
            git_info["git_dirty_files"] = int(dirty)
        if git_info:
            _wandb.run.config.update(git_info, allow_val_change=True)
            _logger.info("[RunMetadata] git: %s", git_info)

        # Upload launch files as a W&B artifact
        valid_paths = [p for p in self._artifact_paths if os.path.isfile(p)]
        if valid_paths:
            try:
                artifact = _wandb.Artifact(name="launch-config", type="config")
                for p in valid_paths:
                    artifact.add_file(p)
                _wandb.run.log_artifact(artifact)
                _logger.info("[RunMetadata] uploaded artifacts: %s", valid_paths)
            except Exception as exc:
                _logger.warning("[RunMetadata] artifact upload failed: %s", exc)


class GenerativeEvalCallback(TrainerCallback):
    """Run generate() on a fixed val subset and log BLEU-4 + W&B table.

    Args:
        val_dataset:  The validation dataset (after collation / split).
        processor:    The VLM processor (tokenizer + image processor).
        cfg:          captioning_eval config block from YAML, or CaptioningEvalConfig.
        collate_fn:   Optional collate function; falls back to trainer's data_collator.
    """

    def __init__(
        self,
        val_dataset,
        processor,
        cfg,
        collate_fn=None,
    ):
        self._cfg = _parse_cfg(cfg)
        self._val_dataset = val_dataset
        self._processor = processor
        self._collate_fn = collate_fn
        self._fixed_indices: Optional[List[int]] = None
        self._last_eval_step = -1

    def _get_subset(self, seed: int = 42) -> Subset:
        if self._fixed_indices is None:
            n = min(self._cfg.num_samples, len(self._val_dataset))
            rng = random.Random(seed)
            self._fixed_indices = rng.sample(range(len(self._val_dataset)), n)
        return Subset(self._val_dataset, self._fixed_indices)

    def _should_run(self, state: TrainerState) -> bool:
        step = state.global_step
        if step == self._last_eval_step:
            return False
        if self._cfg.eval_steps is None:
            return True
        return step % self._cfg.eval_steps == 0


    @torch.no_grad()
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if model is None or not self._should_run(state):
            return

        # ------------------------------------------------------------------
        # NCCL barrier: ALL ranks must enter and exit together.
        #
        # Without this, rank 0 stays inside generate() while ranks 1…N have
        # already moved on to the next distributed collective (_ALLGATHER_BASE
        # from DeepSpeed ZeRO gather), causing a 30-minute NCCL timeout and
        # SIGABRT.  The barrier ensures every rank waits here before rank 0
        # starts generating, and waits again until rank 0 finishes.
        # ------------------------------------------------------------------
        import torch.distributed as dist
        is_distributed = dist.is_available() and dist.is_initialized()

        if is_distributed:
            dist.barrier()

        # ------------------------------------------------------------------
        # CRITICAL FIX FOR ZeRO-3:
        # We CANNOT skip generation on non-zero ranks if the model is sharded
        # (e.g. FSDP / DeepSpeed ZeRO-3). If rank 0 calls model.generate(), it
        # triggers an AllGather. If ranks 1..N are stuck in a barrier, rank 0
        # hangs forever. Therefore, ALL ranks must participate in generation.
        # We just restrict the final logging / metric calculation to rank 0.
        # ------------------------------------------------------------------

        self._last_eval_step = state.global_step
        if args.process_index == 0:
            _logger.info(
                "[CaptioningEval] step=%d — generating on %d samples across %d ranks... (This may take a few minutes)",
                state.global_step, self._cfg.num_samples, dist.get_world_size() if is_distributed else 1
            )

        subset = self._get_subset()
        # Wrap the collator to also extract reference texts and raw images
        eval_collate_fn = _make_eval_collator(self._collate_fn)

        sampler = None
        if is_distributed:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(subset, shuffle=False)

        loader = DataLoader(
            subset,
            batch_size=4,
            shuffle=False,
            sampler=sampler,
            collate_fn=eval_collate_fn,
            num_workers=0,
        )

        local_references: List[str] = []
        local_hypotheses: List[str] = []
        local_table_rows: List[Dict] = []

        model.eval()
        device = next(model.parameters()).device

        # ------------------------------------------------------------------
        # The base collate_fn bakes the ground-truth answer into input_ids
        # (training-style) and always right-pads. Neither is usable directly
        # for generation: see _truncate_and_left_pad_for_generation's
        # docstring for why. We still flip padding_side (some processor
        # internals may consult it) and convert each batch explicitly below.
        # ------------------------------------------------------------------
        tokenizer = self._processor.tokenizer
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        pad_id_for_shift = tokenizer.pad_token_id
        if pad_id_for_shift is None:
            pad_id_for_shift = tokenizer.eos_token_id or 0

        for batch in loader:
            batch = _truncate_and_left_pad_for_generation(batch, pad_id_for_shift)
            batch_size = len(batch.get("input_ids", [[]])) if "input_ids" in batch else 0

            # Reference texts and raw images from the eval collator wrapper
            batch_refs = batch.get("reference_texts", [""] * batch_size)
            batch_imgs = batch.get("raw_images", [None] * batch_size)

            # Build generation inputs — pass keys accepted by the model.
            # Includes Molmo-specific image keys (molmo_images, image_input_idx,
            # image_masks) alongside the standard Qwen/Gemma keys.
            _GENERATE_KEYS = (
                "input_ids", "attention_mask",
                "pixel_values", "image_grid_thw",
                "pixel_values_videos", "video_grid_thw",
                # Molmo
                "molmo_images", "image_input_idx", "image_masks",
            )
            gen_inputs = {}
            for key in _GENERATE_KEYS:
                if key in batch and batch[key] is not None:
                    val = batch[key]
                    if isinstance(val, torch.Tensor):
                        gen_inputs[key] = val.to(device)
                    else:
                        gen_inputs[key] = val

            try:
                # Route through the model's own generate() so that VLMSam and
                # VLMOnly both handle key filtering correctly (VLMOnly strips
                # SAM keys; VLMSam runs the SAM head after generation).
                result = model.generate(
                    **gen_inputs,
                    max_new_tokens=self._cfg.max_new_tokens,
                    do_sample=self._cfg.do_sample,
                    temperature=self._cfg.temperature if self._cfg.do_sample else 1.0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,  # Molmo's manual decode loop needs this
                    # explicitly to stop early; HF generate() on Qwen/Gemma reads it
                    # from generation_config anyway, so passing it is a harmless no-op there.
                    use_cache=True,  # Restored KV Cache for $O(N)$ generation speedup
                    synced_gpus=is_distributed, # Critical for ZeRO-3 to prevent straggler hangs
                )
                # model.generate() may return (preds_list, masks) tuple (VLMSam/VLMOnly)
                # or a raw tensor (HF generate on backbone).
                if isinstance(result, tuple):
                    # VLMSam / VLMOnly — already decoded strings
                    decoded = list(result[0])
                    for ref, hyp, img in zip(batch_refs, decoded, batch_imgs):
                        local_references.append(ref.strip())
                        local_hypotheses.append(hyp.strip())
                        thumb = _resize_for_wandb(img) if img is not None else None
                        local_table_rows.append({
                            "reference": ref.strip(),
                            "prediction": hyp.strip(),
                            "image": thumb,
                        })
                    continue  # skip the tensor path below

                generated_ids = result  # raw tensor from backbone.generate()

            except Exception as exc:
                _logger.warning("[CaptioningEval] generate() failed on rank %d: %s", args.process_index, exc)
                continue

            input_len = gen_inputs.get("input_ids", torch.tensor([])).shape[-1]
            new_tokens = generated_ids[:, input_len:]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for ref, hyp, img in zip(batch_refs, decoded, batch_imgs):
                local_references.append(ref.strip())
                local_hypotheses.append(hyp.strip())
                thumb = _resize_for_wandb(img) if img is not None else None
                local_table_rows.append({
                    "reference": ref.strip(),
                    "prediction": hyp.strip(),
                    "image": thumb,
                })

        # Restore original padding side
        tokenizer.padding_side = original_padding_side
        model.train()

        # CRITICAL: barrier before gather to ensure all ranks have finished
        # generate() before any rank enters the all_gather_object collective.
        # Without this, a rank that exits generate() early (e.g. after catching
        # an exception) can reach all_gather_object while others are still inside
        # generate() doing ZeRO-3 AllGather operations, causing a deadlock.
        if is_distributed:
            dist.barrier()

        # Gather results from all ranks
        references: List[str] = []
        hypotheses: List[str] = []
        table_rows: List[Dict] = []

        if is_distributed:
            gathered_refs = [None for _ in range(dist.get_world_size())]
            gathered_hyps = [None for _ in range(dist.get_world_size())]
            gathered_rows = [None for _ in range(dist.get_world_size())]

            dist.all_gather_object(gathered_refs, local_references)
            dist.all_gather_object(gathered_hyps, local_hypotheses)
            dist.all_gather_object(gathered_rows, local_table_rows)

            if args.process_index == 0:
                for r_list in gathered_refs:
                    references.extend(r_list)
                for h_list in gathered_hyps:
                    hypotheses.extend(h_list)
                for tr_list in gathered_rows:
                    table_rows.extend(tr_list)
        else:
            references = local_references
            hypotheses = local_hypotheses
            table_rows = local_table_rows

        if args.process_index != 0:
            return

        if not hypotheses:
            _logger.warning("[CaptioningEval] No outputs — skipping logging.")
            return

        n_with_img = sum(1 for r in table_rows if r.get("image") is not None)
        _logger.info("[CaptioningEval] images captured: %d/%d", n_with_img, len(table_rows))

        # Compute BLEU-4
        bleu4 = None
        if _SACREBLEU_AVAILABLE and references:
            try:
                bleu = _sacrebleu.corpus_bleu(hypotheses, [references])
                bleu4 = bleu.score
            except Exception as exc:
                _logger.warning("[CaptioningEval] BLEU computation failed: %s", exc)

        _logger.info(
            "[CaptioningEval] step=%d  BLEU-4=%.2f  samples=%d",
            state.global_step,
            bleu4 if bleu4 is not None else float("nan"),
            len(hypotheses),
        )

        # Log to W&B
        if _WANDB_AVAILABLE and _wandb.run is not None:
            log_dict = {"captioning_eval/step": state.global_step}
            if bleu4 is not None:
                log_dict["captioning_eval/bleu4"] = bleu4

            # Build W&B table with image, reference, and prediction columns
            table = _wandb.Table(columns=["step", "image", "reference", "prediction"])
            for row in table_rows:
                pil_img = row.get("image")
                if pil_img is not None:
                    try:
                        pil_img = _resize_for_wandb(pil_img)
                        if pil_img.mode not in ("RGB", "L", "RGBA"):
                            pil_img = pil_img.convert("RGB")
                        wb_img = _wandb.Image(pil_img)
                    except Exception as exc:
                        _logger.warning("[CaptioningEval] wandb.Image failed: %s", exc)
                        wb_img = None
                else:
                    wb_img = None
                table.add_data(state.global_step, wb_img, row["reference"], row["prediction"])
            log_dict["captioning_eval/examples"] = table
            # No explicit step= here: generation takes minutes, so by the time
            # this fires W&B's internal step counter has already advanced past
            # state.global_step (other periodic logging ticks forward during the
            # delay). Passing step=state.global_step then violates W&B's
            # monotonic-step rule and the ENTIRE payload — bleu4 and the examples
            # table — gets silently dropped with only a console warning. The real
            # step is already recorded as the captioning_eval/step field above.
            _wandb.log(log_dict)
        else:
            for row in table_rows[:5]:
                _logger.info("  REF: %s", row["reference"][:120])
                _logger.info("  HYP: %s", row["prediction"][:120])
