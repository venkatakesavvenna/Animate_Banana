"""pixmo_cap_local — locally downloaded PixMo-Cap dataset.

Registry key: pixmo_cap_local

Data layout expected at data_path:
  {data_path}/
    {shard}/          e.g. 00000, 00001, …, 00074
      {shard}_{id}.jpg  (or .png)
      {shard}_{id}.json
        {
          "id": "00000_0000001",
          "image_url": "…",
          "caption": "…",
          "transcripts": ["…", "…"]
        }

Images that failed to download have a .json but no image file; those are silently skipped.

Modes (controlled via dataset_kwargs.mode):
  "captions"              — use caption as answer
  "transcripts"           — use first transcript as answer
  "transcript_and_caption" — caption + newline + first transcript

Manifest cache:
  Scanning 707K flat files on FSx takes ~40 min per run. On first load the
  loader bakes a manifest (image paths + captions) to an HF Arrow dataset at
  manifest_cache_path and reuses it on subsequent loads (<2s startup).
  Default cache: /fsxvision_new/anirudh.srinivasan/DATASETS/pixmo_cap_manifest/
  Override via dataset_kwargs.manifest_cache_path, or set to "" to disable.
"""
import json
import logging
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

from datasets import Dataset, load_from_disk
from PIL import Image

from img_2_svg_pretraining.training.training_core.registry.registry import DatasetRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import AnnotationSpec, DataArguments

_logger = logging.getLogger(__name__)

DATA_PATH = "/fsxvision_new/pratyush.jena/Datasets/pixmo-cap-images/extracted_dataset"
DEFAULT_MANIFEST_CACHE = "/fsxvision_new/anirudh.srinivasan/DATASETS/pixmo_cap_manifest"
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# ~0.6% of PixMo-Cap captions carry a leftover LLM meta-commentary preamble
# from however the dataset's voice-transcript-to-caption cleanup pipeline
# worked (e.g. "Here is the cleaned-up caption for the image:\n\n---\n\n<real
# caption>"). Left in, the model learns to reproduce that templated
# preamble/divider structure as part of "how a caption looks" and
# over-generalizes it onto unrelated images — including appending spurious
# restated "Assistant: ..." continuations after an otherwise-correct caption.
_CAPTION_PREAMBLE_RE = re.compile(
    r"^\s*(here(?:'s| is)|below is|this (?:is a|caption)|following is|"
    r"sure[,!]?\s*here|certainly[,!]?\s*here|i(?:'ll| will) (?:describe|provide))"
    r".{0,400}?\n\s*-{3,}\s*\n+",
    re.IGNORECASE | re.DOTALL,
)


def _strip_caption_preamble(caption: str) -> str:
    match = _CAPTION_PREAMBLE_RE.match(caption)
    return caption[match.end():].strip() if match else caption


# ---------------------------------------------------------------------------
# Shard scanner
# ---------------------------------------------------------------------------

def _scan_shards(
    data_path: str,
    shards: Optional[List[str]] = None,
    sample_limit: Optional[int] = None,
) -> List[Dict]:
    """Walk shard directories and return list of {id, image_path, caption, transcripts}.

    Slow on FSx (~40 min for all 75 shards) — use load_or_build_manifest() instead.
    """
    root = Path(data_path)
    if not root.exists():
        raise FileNotFoundError(
            f"pixmo_cap_local: data_path not found: {data_path}\n"
            "Set dataset_kwargs.data_path to the extracted_dataset directory."
        )

    shard_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if shards is not None:
        shard_set = set(shards)
        shard_dirs = [d for d in shard_dirs if d.name in shard_set]

    records = []
    for shard_dir in shard_dirs:
        for json_path in sorted(shard_dir.glob("*.json")):
            stem = json_path.stem  # e.g. "00000_0000001"
            img_path = None
            for ext in _IMAGE_EXTS:
                candidate = shard_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_path = str(candidate)
                    break
            if img_path is None:
                continue

            try:
                meta = json.loads(json_path.read_text())
            except Exception:
                continue

            records.append({
                "id": stem,
                "image_path": img_path,
                "caption": meta.get("caption", ""),
                "transcripts": meta.get("transcripts", []),
                "image_url": meta.get("image_url", ""),
            })

            if sample_limit is not None and len(records) >= sample_limit:
                return records

    return records


# ---------------------------------------------------------------------------
# Manifest cache
# ---------------------------------------------------------------------------

def build_manifest(
    data_path: str = DATA_PATH,
    cache_path: str = DEFAULT_MANIFEST_CACHE,
) -> Dataset:
    """Scan all shards and save a manifest Arrow dataset to cache_path.

    Run once; subsequent loads use load_manifest() which is near-instant.
    """
    _logger.info("pixmo_cap_local: building manifest — scanning %s …", data_path)
    import time
    t0 = time.time()
    records = _scan_shards(data_path)
    elapsed = time.time() - t0
    _logger.info(
        "pixmo_cap_local: scan complete — %d records in %.1fs, saving to %s",
        len(records), elapsed, cache_path,
    )
    if not records:
        ds = Dataset.from_dict({})
    else:
        ds = Dataset.from_dict({k: [r[k] for r in records] for k in records[0].keys()})
    ds.save_to_disk(cache_path)
    _logger.info("pixmo_cap_local: manifest saved (%d rows).", len(ds))
    return ds


def load_manifest(cache_path: str = DEFAULT_MANIFEST_CACHE) -> Dataset:
    """Load a pre-built manifest. Raises if not found."""
    return load_from_disk(cache_path)


def load_or_build_manifest(
    data_path: str = DATA_PATH,
    cache_path: str = DEFAULT_MANIFEST_CACHE,
    wait_timeout: int = 7200,
) -> Dataset:
    """Load manifest from cache if available; build and cache otherwise.

    Distributed-safe: in a torchrun run, only LOCAL_RANK 0 scans and builds;
    other ranks wait for dataset_info.json to appear (up to wait_timeout secs).
    Also handles an external pre-build process (e.g. a background preflight).
    """
    import os
    import time

    cache = Path(cache_path)
    info_file = cache / "dataset_info.json"

    # Fast path: already built.
    if info_file.exists():
        _logger.info("pixmo_cap_local: loading manifest from %s", cache_path)
        t0 = time.time()
        ds = load_from_disk(cache_path)
        _logger.info("pixmo_cap_local: loaded %d rows in %.1fs", len(ds), time.time() - t0)
        return ds

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))

    # Non-zero rank OR directory already exists (another process is building): wait.
    if local_rank != 0 or cache.exists():
        if local_rank != 0:
            _logger.info(
                "pixmo_cap_local: rank %d waiting for rank-0 to build manifest …", local_rank
            )
        else:
            _logger.info(
                "pixmo_cap_local: manifest build in progress — waiting (max %ds) …", wait_timeout
            )
        t0 = time.time()
        while not info_file.exists() and (time.time() - t0) < wait_timeout:
            time.sleep(15)
        if not info_file.exists():
            raise TimeoutError(
                f"pixmo_cap_local: manifest not ready after {wait_timeout}s at {cache_path}"
            )
        ds = load_from_disk(cache_path)
        _logger.info("pixmo_cap_local: manifest ready (%d rows)", len(ds))
        return ds

    # Rank 0, no directory yet: build.
    return build_manifest(data_path, cache_path)


# ---------------------------------------------------------------------------
# get_source
# ---------------------------------------------------------------------------

def get_source(
    source: Dict,
    get_sam_source_fn: Callable,
    sam_image_size: int,
    debug_path: str = "",
    localization_mode=None,
    mode: str = "captions",
):
    img_path = source["image_path"]
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot open image {img_path}: {exc}") from exc

    caption = _strip_caption_preamble(source.get("caption") or "")
    transcripts = source.get("transcripts") or []
    first_transcript = transcripts[0] if transcripts else ""

    if mode == "transcripts":
        answer = first_transcript
    elif mode == "transcript_and_caption":
        answer = caption + ("\n\n" + first_transcript if first_transcript else "")
    else:
        answer = caption

    source_dict = {
        "image": image,
        "conversations": [
            {"from": "human", "value": "<image>\nDescribe this image in detail."},
            {"from": "gpt", "value": answer},
        ],
    }
    sam_specific_dict = get_sam_source_fn(image, [], is_mask=False, image_size=sam_image_size)
    return source_dict, sam_specific_dict


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------

@DatasetRegistry.register_dataset("pixmo_cap_local")
def get_pixmo_cap_local_dataargs(
    get_sam_source_fn: Callable,
    seed: int = 42,
    sam_image_size: int = 1024,
    train: bool = True,
    streaming: bool = False,
    sample_limit: Optional[int] = None,
    localization_mode=None,
    debug_path: str = "",
    data_path: str = DATA_PATH,
    mode: str = "captions",
    shards: Optional[List[str]] = None,
    manifest_cache_path: str = DEFAULT_MANIFEST_CACHE,
):
    """Load locally downloaded PixMo-Cap data from sharded directories.

    Args:
        data_path:            Root of the extracted_dataset directory.
        mode:                 "captions" | "transcripts" | "transcript_and_caption"
        shards:               Optional list of shard names to restrict (bypasses manifest).
        sample_limit:         Max number of examples (bypasses manifest).
        manifest_cache_path:  Arrow cache dir. Set to "" to disable and always scan.
    """
    # Shard subset or sample_limit → bypass cache (only feasible for small subsets)
    use_cache = manifest_cache_path and shards is None and sample_limit is None
    if use_cache:
        ds = load_or_build_manifest(data_path, manifest_cache_path)
    else:
        reason = "shard subset" if shards else "sample_limit"
        if shards is None and sample_limit is None:
            reason = "cache disabled"
        _logger.info("pixmo_cap_local: scanning shards (%s — bypassing manifest cache)", reason)
        records = _scan_shards(data_path, shards=shards, sample_limit=sample_limit)
        if not records:
            raise RuntimeError(
                f"pixmo_cap_local: no valid (image, json) pairs found under {data_path}."
            )
        ds = Dataset.from_dict({k: [r[k] for r in records] for k in records[0].keys()})

    ds = ds.shuffle(seed=seed)
    _logger.info("pixmo_cap_local: dataset ready — %d examples", len(ds))

    return DataArguments(
        dataset_path=data_path,
        ds=ds,
        seed=seed,
        get_source=get_source,
        get_source_kwargs={
            "get_sam_source_fn": get_sam_source_fn,
            "sam_image_size": sam_image_size,
            "mode": mode,
        },
        annotation_spec=AnnotationSpec(has_localization=False),
    )
