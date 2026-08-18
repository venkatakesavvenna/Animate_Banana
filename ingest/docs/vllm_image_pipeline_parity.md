# The image pipeline: what actually happens to your pixels

**Status:** investigation only — no code changed as of this writing.

**Environment investigated:** `/environments/gemma4_new`, vLLM **0.25.1**, transformers **5.14.1**,
4× **H100 80GB HBM3**, model `google/gemma-4-31B-it`
(HF cache `/hf_cache/hub/models--google--gemma-4-31B-it`, snapshot `842da3794eaa0b77d5f08bae87a17459d91ff475`).

**Companion doc:** [`vllm_serve_offline_parity.md`](./vllm_serve_offline_parity.md) covers the
*text* side (usage-context scheduler defaults, sampling-parameter merging). This doc covers
images and multi-image prompts. Read that one first — one of its findings (Divergence 1)
hits images considerably harder than it hits text, and this doc explains why.

**Questions this answers:**

1. Does HF tokenize images at their native/highest resolution, and does vLLM silently
   downscale them behind our back?
2. Is `vllm serve` (online) better than offline `AsyncLLM` for images, the way it is for text?
3. If not resolution loss, what *is* the real image-quality lever?

**Short answers:**

1. **No.** HF does not use native resolution. Gemma 4's *own* image processor resizes every
   image — up or down — to a fixed patch budget. vLLM adds no resizing of its own.
2. **No.** Past the image-decode step, online and offline are provably identical for images.
   There is no hidden server-side quality advantage.
3. **`max_soft_tokens`.** It is a 4× resolution lever that is set to its default on both sides,
   and it is worth far more than anything in the online/offline split.

All vLLM paths below are relative to:

```
/fsxvision_new/srihari.bandarupalli/environments/gemma4_new/lib/python3.12/site-packages/vllm/
```

and transformers paths to the sibling `transformers/` directory.

---

## Table of contents

- [1. The false assumption: "HF uses native resolution"](#1-the-false-assumption-hf-uses-native-resolution)
  - [1.1 The resize function](#11-the-resize-function)
  - [1.2 Measured behaviour](#12-measured-behaviour)
  - [1.3 Why it is built this way](#13-why-it-is-built-this-way)
- [2. What vLLM adds on top: nothing (for resolution)](#2-what-vllm-adds-on-top-nothing-for-resolution)
- [3. Proving online == offline for images](#3-proving-online--offline-for-images)
  - [3.1 Where the two paths converge](#31-where-the-two-paths-converge)
  - [3.2 Chat-template rendering is byte-identical](#32-chat-template-rendering-is-byte-identical)
  - [3.3 `max_soft_tokens` resolution is identical](#33-max_soft_tokens-resolution-is-identical)
  - [3.4 Token-budget accounting is identical](#34-token-budget-accounting-is-identical)
- [4. The one genuine image divergence: our own decode](#4-the-one-genuine-image-divergence-our-own-decode)
  - [4.1 Measured comparison](#41-measured-comparison)
  - [4.2 A correction to the previous doc](#42-a-correction-to-the-previous-doc)
  - [4.3 Scope: how much does this matter?](#43-scope-how-much-does-this-matter)
- [5. The real lever: `max_soft_tokens`](#5-the-real-lever-max_soft_tokens)
  - [5.1 What the settings buy](#51-what-the-settings-buy)
  - [5.2 How to set it](#52-how-to-set-it)
  - [5.3 The interaction with `max_num_batched_tokens`](#53-the-interaction-with-max_num_batched_tokens)
- [6. Multi-image specifics](#6-multi-image-specifics)
- [7. Video, briefly](#7-video-briefly)
- [8. Blockers in the current config](#8-blockers-in-the-current-config)
- [9. Recommended changes](#9-recommended-changes)
- [Appendix A: the image pipeline end to end](#appendix-a-the-image-pipeline-end-to-end)
- [Appendix B: reproduce every check in this doc](#appendix-b-reproduce-every-check-in-this-doc)

---

## 1. The false assumption: "HF uses native resolution"

The intuition that HF preprocessing preserves full resolution and that vLLM must be
degrading it somewhere is a reasonable guess, and it is wrong for this model family.
The resizing is in the model's own processor config, shipped with the weights.

From `/hf_cache/hub/models--google--gemma-4-31B-it/snapshots/842da.../processor_config.json`:

```json
"image_processor": {
  "do_convert_rgb": true,
  "do_normalize": false,
  "do_rescale": true,
  "do_resize": true,
  "image_processor_type": "Gemma4ImageProcessor",
  "image_seq_length": 280,
  "max_soft_tokens": 280,
  "patch_size": 16,
  "pooling_kernel_size": 3,
  "resample": 3,
  "rescale_factor": 0.00392156862745098
}
```

`do_resize: true`. There is no `size` key — this processor does not use the conventional
`{"height": …, "width": …}` mechanism. It computes a target size from a *token budget*.

Note also what is **not** happening: `do_normalize: false`, `image_mean: [0,0,0]`,
`image_std: [1,1,1]`. Gemma 4 only rescales to `[0, 1]`; there is no ImageNet-style
mean/std normalization to get wrong.

### 1.1 The resize function

`transformers/models/gemma4/image_processing_gemma4.py:33`:

```python
def get_aspect_ratio_preserving_size(
    height: int, width: int, patch_size: int, max_patches: int, pooling_kernel_size: int,
) -> tuple[int, int]:
    """
    Image is resized to preserve aspect ratio so it fits within the patch budget.
    Target dimensions are the largest that:
    1) Produce at most `max_patches` patches when patchified with `patch_size`
    2) Have height and width divisible by `pooling_kernel_size * patch_size`
    """
    total_px = height * width
    target_px = max_patches * (patch_size**2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width
    side_mult = pooling_kernel_size * patch_size
    target_height = int(math.floor(ideal_height / side_mult)) * side_mult
    target_width = int(math.floor(ideal_width / side_mult)) * side_mult
    ...
```

The critical detail is `factor = math.sqrt(target_px / total_px)` applied
**unconditionally**. There is no `min(factor, 1.0)`. If the image is smaller than the
budget, `factor > 1` and the image is **upscaled**.

The budget itself, from `_preprocess` (`image_processing_gemma4.py:200`):

```python
max_patches = max_soft_tokens * pooling_kernel_size**2
```

With the shipped defaults: `max_patches = 280 × 3² = 2520` patches, and
`target_px = 2520 × 16² = 645,120` pixels — roughly an 803×803 square.

Both dimensions are floored to a multiple of `pooling_kernel_size × patch_size = 48`.

### 1.2 Measured behaviour

Running `get_aspect_ratio_preserving_size` directly against real input sizes
(script in [Appendix B](#appendix-b-reproduce-every-check-in-this-doc)):

**`max_soft_tokens = 280` (the default) — budget 645,120 px**

| input | pixels | → resized | pixels | scale | soft tokens |
|---|---:|---|---:|---:|---:|
| 4096×3072 | 12,582,912 | 912×672 | 612,864 | **0.223×** | 266 |
| 2048×1536 | 3,145,728 | 912×672 | 612,864 | 0.445× | 266 |
| 1024×768 | 786,432 | 912×672 | 612,864 | 0.891× | 266 |
| 800×600 | 480,000 | 912×672 | 612,864 | **1.140× ↑** | 266 |
| 512×384 | 196,608 | 912×672 | 612,864 | **1.781× ↑** | 266 |
| 224×224 | 50,176 | 768×768 | 589,824 | **3.429× ↑** | 256 |

Read the `scale` column carefully. Every 4:3 image, whether it started at 12.6 megapixels
or 0.2 megapixels, ends up at exactly 912×672. Aspect ratio is preserved; absolute
resolution is not a property of your input at all. It is a property of `max_soft_tokens`.

This means two things people usually get wrong:

- **Feeding higher-resolution source images changes nothing** once you are above the
  budget. A 4K scan and a 1024×768 export produce the identical 912×672 tensor.
- **Small images are not "cheap."** A 224×224 thumbnail is upscaled 3.4× and still costs
  256 soft tokens. There is no token savings from small inputs.

### 1.3 Why it is built this way

This is the Siglip2 "NaFlex" style of vision encoding: rather than squashing every image
to a fixed square, patchify at a native-ish aspect ratio and pad the patch sequence to a
fixed length. You can see the lineage in the file — `convert_image_to_patches` carries a
`# Copied from transformers.models.siglip2...` marker, and `pad_along_first_dim` pads the
patch axis to `max_patches` with position IDs of `-1` for padding.

The consequence relevant to us: the vision tower has a **fixed compute cost per image**
regardless of input size, and `max_soft_tokens` is the dial that sets that cost.

---

## 2. What vLLM adds on top: nothing (for resolution)

The server-side image loader is `ImageMediaIO` in `multimodal/media/image.py:21`. Its
entire load path (`load_bytes`, line 73):

```python
def load_bytes(self, data: bytes) -> MediaWithBytes[Image.Image]:
    try:
        image = Image.open(BytesIO(data))
        w, h = image.size
        max_pixels = envs.VLLM_MAX_IMAGE_PIXELS
        if max_pixels > 0 and w * h > max_pixels:
            raise ValueError(
                f"Image dimensions {w}x{h} ({w * h} pixels) exceed "
                f"the maximum of {max_pixels} pixels. Set "
                f"VLLM_MAX_IMAGE_PIXELS to increase this limit."
            )
        image = normalize_image(image)
        image.load()
        image = self._convert_image_mode(image)
    except (OSError, Image.UnidentifiedImageError) as e:
        raise ValueError(f"Failed to load image: {e}") from e
    return MediaWithBytes(image, data)
```

Three operations, and **none of them is a resize**:

1. **`VLLM_MAX_IMAGE_PIXELS` check** — `envs.py:82` gives a default of `178_956_970`
   (~179 megapixels). This is a decompression-bomb guard inherited from Pillow's own
   limit. It **raises** on oversized input; it never downscales. At ~179 MP it will
   essentially never fire on real photographs. If it ever does, you get a hard request
   failure, not silent quality loss — which is what you want.
2. **`normalize_image`** — `multimodal/image.py:21`, just `ImageOps.exif_transpose`.
3. **`_convert_image_mode`** — mode coercion to RGB, discussed in [§4](#4-the-one-genuine-image-divergence-our-own-decode).

So the answer to "does vLLM resize behind our back" is a clean no. Every pixel that
reaches the processor is the pixel Pillow decoded. All resizing is the HF processor's,
i.e. the model's own, i.e. identical no matter how you drive vLLM.

---

## 3. Proving online == offline for images

### 3.1 Where the two paths converge

The two backends differ only in how a request is *constructed*. They converge before any
image math happens:

```
ONLINE  (backend="online")                 OFFLINE (backend="async"/"batch")
────────────────────────────               ─────────────────────────────────
online_worker._build_chat_payload          vllm_config.get_prompt_with_image
  → HTTP POST /v1/chat/completions            → load_cv2_pil()  ← PIL objects
  → server parses OpenAI content              → apply_chat_template(tokenize=False)
  → ImageMediaIO.load_file(file://…)          → {"prompt": str,
  → renderer.render_chat()                       "multi_modal_data": {"image": [PIL…]}}
  → apply_chat_template                            │
        │                                          │
        └──────────────┬───────────────────────────┘
                       ▼
        v1/engine/input_processor.py
                       ▼
        MultiModalProcessor.apply(prompt, mm_data, hf_processor_mm_kwargs)
                       ▼
        Gemma4Processor  →  Gemma4ImageProcessor  →  resize / patchify / pad
                       ▼
                  same tensors
```

Everything below the junction is shared code. For the two paths to produce different
pixels, they would have to hand different things *into* the junction. There are exactly
three things handed in — the prompt string, the PIL images, and
`hf_processor_mm_kwargs` — so those are the three things to check.

### 3.2 Chat-template rendering is byte-identical

The prompt string is the first candidate. Our offline code builds content items carrying
the PIL object inline (`{"type": "image", "image": <PIL>}`), whereas vLLM's server parses
OpenAI-format content into bare placeholders (`{"type": "image"}`) and keeps the images
in a sidecar list. If Gemma 4's Jinja template inspected the `image` key, these would
diverge.

Tested with two images plus text, using the model's real `chat_template.jinja`:

```
OURS   : '<bos><|turn>user\n<|image|><|image|>Describe.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>'
SERVER : '<bos><|turn>user\n<|image|><|image|>Describe.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>'
IDENTICAL: True
```

Byte-identical, including placeholder count and ordering. The template only branches on
`content.type`, never on payload. **Not a divergence.**

Two follow-on points this also settles:

- The `<bos>` in the rendered string is *not* doubled later.
  `gemma4_mm.py:189` overrides `get_default_tok_params` to force
  `add_special_tokens=False` whenever the tokenizer has a chat template:

  ```python
  tokenizer = self.ctx.get_tokenizer()
  has_chat_template = getattr(tokenizer, "chat_template", None) is not None
  params = super().get_default_tok_params()
  if has_chat_template:
      params = params.with_kwargs(add_special_tokens=False)
  return params
  ```

  This is model-specific code on the shared path, so it protects both backends equally.
  (This matters because `multimodal/processing/context.py` otherwise defaults multimodal
  tokenization to `add_special_tokens=True` — Gemma 4 opts out.)
- The trailing `<|channel>thought\n<channel|>` is the thinking channel opening, and it is
  present in both. It is not evidence of a template mismatch.

### 3.3 `max_soft_tokens` resolution is identical

`hf_processor_mm_kwargs` is the second candidate, and the one that could actually change
resolution. vLLM resolves it in `gemma4_mm.py:95`:

```python
def _get_max_soft_tokens(merged_kwargs):
    """Return configured image max_soft_tokens and whether it is top-level."""
    val = merged_kwargs.get("max_soft_tokens")
    if val is not None:
        return val, True
    images_kwargs = merged_kwargs.get("images_kwargs")
    if isinstance(images_kwargs, Mapping):
        return images_kwargs.get("max_soft_tokens"), False
    return None, False
```

`merged_kwargs` comes from `self.ctx.get_merged_mm_kwargs({})` — i.e. the model config's
engine-level `mm_processor_kwargs`, merged with any per-request override.

Per-request override *is* reachable from both sides:

- online — `mm_processor_kwargs` is a field on `ChatCompletionRequest`
  (`entrypoints/openai/chat_completion/protocol.py:343`)
- offline — `mm_processor_kwargs` is a key on the prompt dict, read by the renderer
  (`renderers/base.py:784`, `:847`)

**Neither of our code paths sets it.** `online_worker._build_chat_payload` sends only
`model` / `messages` / `temperature` / `max_tokens`; `vllm_config.get_prompt_with_image`
returns only `prompt` and `multi_modal_data`. So both fall through to the engine-level
value, which comes from the same `engine_args` block in `vllm_model.yaml`. Identical by
construction. **Not a divergence.**

When nothing is set anywhere, the fallback is the vision config
(`_compute_num_soft_tokens`, `gemma4_mm.py:277`):

```python
if max_soft_tokens is None:
    max_soft_tokens = vision_cfg.default_output_length
```

and `config.json`'s `vision_config.default_output_length` is `280` — matching the
processor default. Consistent from either direction.

### 3.4 Token-budget accounting is identical

`get_mm_max_tokens_per_item` (`gemma4_mm.py:240`) is what vLLM uses for memory planning,
and it reads the same merged kwargs:

```python
tokens_per_image = config.vision_config.default_output_length
merged_kwargs = self.ctx.get_merged_mm_kwargs({})
val, _ = _get_max_soft_tokens(merged_kwargs)
if isinstance(val, int) and val in _SUPPORTED_SOFT_TOKENS:
    tokens_per_image = val
```

Same input, same output, both backends. And `_compute_num_soft_tokens` clamps the
per-image count with `min(num_patches // pooling_kernel_size**2, max_soft_tokens)`,
guarding extreme aspect ratios (the source comment cites 3×900) from over-claiming
prompt-side placeholders.

**Conclusion for §3: for images there is no online-vs-offline quality gap in the vLLM
pipeline.** The gaps documented in the companion doc (scheduler defaults, sampling
params) still apply, and are model-wide rather than image-specific. The only
image-specific divergence is upstream of vLLM entirely — in our own decode.

---

## 4. The one genuine image divergence: our own decode

Offline we do not hand vLLM a file path; we hand it a PIL object we decoded ourselves.
That bypasses `ImageMediaIO` completely, so whatever `ImageMediaIO` would have done to
normalize the image simply does not happen.

**Ours** — `src/vision_ingest/utils/utils.py:18`, `load_cv2_pil`:

```python
img = cv2.imread(str(p))   # BGR, alpha channel DISCARDED
pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGB")
```

**Server** — `Image.open` → `exif_transpose` → `rgba_to_rgb(image, (255,255,255))`, where
(`multimodal/image.py:28`):

```python
def rgba_to_rgb(image, background_color=(255, 255, 255)):
    assert image.mode == "RGBA"
    converted = Image.new("RGB", image.size, background_color)
    converted.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return converted
```

`cv2.imread` with default flags drops alpha and returns the raw underlying colour
channels. Pillow composites over white. For any pixel that is not fully opaque, these
disagree.

### 4.1 Measured comparison

Both paths run against the same files
(script in [Appendix B](#appendix-b-reproduce-every-check-in-this-doc)):

| case | server (PIL path) | ours (cv2 path) | divergent? |
|---|---|---|---|
| RGBA, fully transparent red | `[255 255 255]` | `[255 0 0]` | **yes** |
| RGBA, 50% alpha blue | `[127 127 255]` | `[0 0 255]` | **yes** |
| Palette PNG + `tRNS` chunk | `[255 255 255]` | `[0 0 0]` | **yes** |
| EXIF orientation = 6 (rot 90) | shape `(80,40,3)` | shape `(80,40,3)` | no |
| 16-bit PNG | `uint8 (32,32,3)` | `uint8 (32,32,3)` | no |
| Grayscale `L` | `(32,32,3)`, `[128 128 128]` | `(32,32,3)`, `[128 128 128]` | no |
| CMYK JPEG | `[255 0 0]` | `[255 1 1]` | negligible (±1 rounding) |

The divergence is **alpha-handling, and only alpha-handling**.

The 50%-alpha row is the instructive one: `[127 127 255]` vs `[0 0 255]` shows the server
performing real alpha compositing, while cv2 returns the unblended colour. A
semi-transparent watermark, a logo with a soft edge, or an antialiased diagram exported
with transparency will reach the model looking materially different.

The palette row matters more than it looks. A palettized PNG with a `tRNS` chunk is not
mode `RGBA`, so a naive `image.mode == "RGBA"` check would miss it — vLLM catches it via
`_has_transparency`, which also tests `"transparency" in image.info`. Our cv2 path renders
those transparent pixels as black.

### 4.2 A correction to the previous doc

[`vllm_serve_offline_parity.md`](./vllm_serve_offline_parity.md) lists EXIF transpose as
part of the server pipeline in a way that implies offline lacks it. Measured, that is not
a divergence: OpenCV has applied EXIF orientation for JPEG/PNG in `imread` since 3.4.1,
and the test above confirms both paths return the identically rotated `(80, 40, 3)`.

Treat the image-decode divergence as **alpha only**. That doc's Divergence 3 should be
read with that narrowed scope.

### 4.3 Scope: how much does this matter?

Entirely dependent on your corpus:

- **JPEG-only corpus → completely inert.** JPEG has no alpha channel. Nothing to fix.
- **PNG / WebP with transparency → real, and silently wrong.** Renders, screenshots,
  charts, diagrams, logos, and anything exported from a design tool are the usual
  offenders.

This is cheap to check before spending any effort:

```bash
# Sample 500 paths from your corpus and count how many carry transparency
shuf -n 500 /path/to/images.txt | while read -r f; do
  python - "$f" <<'PY'
import sys
from PIL import Image
try:
    im = Image.open(sys.argv[1])
    if im.mode in ("RGBA","LA","PA") or "transparency" in im.info:
        print("ALPHA", sys.argv[1])
except Exception as e:
    print("ERR", sys.argv[1], e)
PY
done | grep -c ALPHA
```

If that returns 0, skip the fix. If it returns anything meaningful, the fix is three lines
and is in [§9](#9-recommended-changes).

---

## 5. The real lever: `max_soft_tokens`

Having established that online and offline are equivalent for images, the interesting
question stops being "which backend" and becomes "why is every image being squeezed into
645K pixels."

`image_processing_gemma4.py:29` defines the permitted values:

```python
_SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)
```

These are enforced twice — in `Gemma4ImageProcessor.__init__` and again in `_preprocess`
— so an unsupported value raises rather than silently degrading. vLLM validates against
the same tuple (`gemma4_mm.py`, "Unsupported max_soft_tokens value: …").

We are on `280`, the shipped default. There are two settings above it.

### 5.1 What the settings buy

Measured (1024×768 input; the resized output is the same for any 4:3 input, per §1.2):

| `max_soft_tokens` | max patches | pixel budget | 4:3 image → | tokens/image | vs default |
|---:|---:|---:|---|---:|---:|
| 70 | 630 | 161,280 px | 432×336 | 63 | 0.24× |
| 140 | 1,260 | 322,560 px | 624×480 | 130 | 0.49× |
| **280** *(current)* | 2,520 | 645,120 px | **912×672** | **266** | 1× |
| 560 | 5,040 | 1,290,240 px | 1296×960 | 540 | 2.03× |
| 1120 | 10,080 | 2,580,480 px | **1824×1344** | 1064 | **4.00×** |

Going 280 → 1120 quadruples the pixel budget: a 4:3 image is encoded at 1824×1344 instead
of 912×672. Linear resolution doubles in each dimension.

For any task where the answer depends on fine detail — small text, dense documents,
charts, thin strokes, small objects in a wide frame — this is a much larger lever than
anything in the online/offline comparison. It is also the lever most likely to explain a
"the model can't read this" style of failure.

The cost is exactly proportional: 1064 prompt tokens per image instead of 266, so ~4× the
vision-tower compute, ~4× the image-side prefill, and ~4× the KV cache footprint for image
tokens. Whether it pays for itself is empirical — but it should be A/B'd, and currently it
never has been.

At the other end, `70` and `140` exist for throughput. On a bulk captioning job where
images are large and simple, `140` halves image token cost. Worth testing in the same
sweep rather than assuming the default is optimal in either direction.

### 5.2 How to set it

It goes in `mm_processor_kwargs`, an engine-level arg, so one entry covers both backends:

```yaml
models:
  google/gemma-4-31B-it:
    is_llm: false
    engine_args:
      max_model_len: 32000
      gpu_memory_utilization: 0.90
      limit_mm_per_prompt:
        image: 1        # must be > 0 — see §8
        audio: 0
        video: 0        # see §7
      mm_processor_kwargs:
        max_soft_tokens: 1120
      async_scheduling: true
```

Verification: `_get_max_soft_tokens` accepts it at top level (returning
`is_top_level=True`), and `gemma4_mm.py:711-724` deliberately re-injects a top-level value
after merging so that it is not lost — there is an explicit comment about `_merge_kwargs`
routing it into `images_kwargs`. Both spellings work; top level is the documented one.

For `vllm serve`, `online_worker._engine_args_to_serve_flags()` translates this to
`--mm-processor-kwargs '{"max_soft_tokens": 1120}'`. Worth confirming the translator
handles a nested dict value the way it handles `limit_mm_per_prompt` — same JSON-encoding
requirement.

**One caveat before changing this:** `max_soft_tokens` alters the number of image
placeholder tokens in the prompt. It is a genuine change in model input, not a tuning
knob that leaves outputs invariant. Re-baseline any quality comparison after changing it,
and do not change it in the same experiment as the fixes from the companion doc.

### 5.3 The interaction with `max_num_batched_tokens`

This is the part that connects the two documents, and it is the reason to fix the
scheduler defaults *before* touching resolution.

Divergence 1 in [`vllm_serve_offline_parity.md`](./vllm_serve_offline_parity.md):
`AsyncLLM.from_engine_args` defaults to `UsageContext.ENGINE_CONTEXT`, which is absent
from vLLM's defaults table, so it silently falls through to
`DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048` and `DEFAULT_MAX_NUM_SEQS = 128` — where
`vllm serve` would have given 8192 and 1024.

Now put images on top of it:

| `max_soft_tokens` | tokens/image | images per step @ 2048 (offline default) | @ 8192 (serve default) |
|---:|---:|---:|---:|
| 280 | 266 | ~7 | ~30 |
| 560 | 540 | ~3 | ~15 |
| 1120 | 1064 | **~2** | ~8 |

At 1120 soft tokens with the un-fixed offline default, the scheduler can admit roughly
**two images per step**. Combined with `max_num_seqs` at 128 instead of 1024, multi-image
throughput collapses. This is very likely a large part of why the earlier manual
`max_num_batched_tokens=8192` experiment felt slow — it fixed one of the two numbers.

Also note vLLM's chunked-prefill behaviour here: `platforms/cuda.py:302-322` forces
`disable_chunked_mm_input`, meaning a single image's tokens cannot be split across
scheduler steps. An image must fit in one batch whole. At 1120 soft tokens that is a
1064-token indivisible unit, which is fine against 8192 and cramped against 2048.

**Ordering, therefore:**

1. Fix `usage_context` (companion doc, change 1). Re-baseline.
2. Only then sweep `max_soft_tokens` ∈ {140, 280, 560, 1120}.

Doing it in the other order measures the scheduler, not the resolution.

---

## 6. Multi-image specifics

Everything above holds per image; a few points are specific to multi-image prompts.

- **Per-image independent resize.** `_preprocess` loops one image at a time — "Images have
  different aspect ratios and thus different resized dimensions, so patchification and
  padding must happen per-image before stacking." Each image gets its own budget; they do
  not share one. Three images at 280 cost ~798 tokens, not 266.
- **Padding is per image, to `max_patches`.** `pad_along_first_dim` pads every image's
  patch sequence to the full `max_patches` with position IDs of `-1`. So a panoramic image
  that only fills 200 of 280 soft-token slots still occupies a full slot budget in the
  tensor. Padding is masked out, so it costs memory and not quality.
- **Placeholder expansion is per image and exact.** `Gemma4Processor.replace_image_token`:

  ```python
  num_soft_tokens = image_inputs["num_soft_tokens_per_image"][image_idx]
  return f"{self.boi_token}{self.image_token * num_soft_tokens}{self.eoi_token}"
  ```

  Each `<|image|>` placeholder expands to *its own* image's actual token count, not a
  uniform value. That is why `_compute_num_soft_tokens` has to replicate the resize
  arithmetic exactly — a mismatch between predicted and actual counts is a hard failure,
  not a quality issue.
- **Ordering is positional.** The template emits placeholders in content order and the
  processor consumes `image_idx` in order. Our
  `get_prompt_with_image` builds content as "all images, then text," which is a fixed
  convention — fine, but note that interleaved image/text ordering is expressible in the
  template and we never use it. If a task benefits from "image A, question about A, image
  B, question about B," that is available and currently unused.
- **`limit_mm_per_prompt.image` must be ≥ your true maximum.** It is a hard admission
  limit, not a hint.

---

## 7. Video, briefly

Not the focus, but two facts are worth recording because they affect image configs.

- **Video uses a different, smaller budget.** `gemma4_mm.py:91-92`:
  `_VIDEO_MAX_SOFT_TOKENS = 70` per frame and `_VIDEO_MAX_FRAMES = 32`. Frames are
  processed as images with `max_soft_tokens=70` forced
  (`video_mm_kwargs["max_soft_tokens"] = _VIDEO_MAX_SOFT_TOKENS`). So video is *not*
  simply "images repeated" — each frame is at quarter resolution relative to the image
  default.
- **Video inflates the minimum batched-token floor even when you never send video.** vLLM
  raises `max_num_batched_tokens` to fit the largest single multimodal item, and for
  Gemma 4 that is a 32-frame video: `32 × (70 + 2 + ~6) ≈ 2496` tokens. `get_supported_mm_limits`
  returns `limits["video"] = None` (unbounded) unless you constrain it. The current
  `vllm_model.yaml` sets `image: 0` and `audio: 0` but **omits `video`**, so the video
  floor still applies. Adding `video: 0` removes it.

---

## 8. Blockers in the current config

The current [`examples/gemma_testing/vllm_model.yaml`](../examples/gemma_testing/vllm_model.yaml)
cannot process images at all. Two independent reasons:

```yaml
is_llm: true                    # (1)
engine_args:
  limit_mm_per_prompt:
    image: 0                    # (2)
    audio: 0
```

1. **`is_llm: true`** makes `VLLMConfig.__init__` load `AutoTokenizer` instead of
   `AutoProcessor`, and makes `get_prompt_with_image` flatten message content to text
   only — dropping images silently, with no error. In `backend="online"` it also forces
   `"image_paths": []`.
2. **`limit_mm_per_prompt.image: 0`** makes the engine reject any request carrying an
   image.

This is correct and deliberate for the current job — it is a text-only English→Hindi
caption rewrite, and the file says so. But it means **no image path in this repo has been
exercised against Gemma 4 yet**, which is worth stating plainly: everything in this
document is derived from source and from isolated processor-level tests, not from a
completed image run. The first real image run may surface things static analysis cannot.

---

## 9. Recommended changes

Nothing below is applied. Ordered by value.

**1. (Prerequisite) Fix `usage_context`** — change 1 in the companion doc. Do this first
and re-baseline; it gates any meaningful image throughput measurement (see [§5.3](#53-the-interaction-with-max_num_batched_tokens)).

**2. Sweep `max_soft_tokens`.** Add to `vllm_model.yaml` under `engine_args`:

```yaml
mm_processor_kwargs:
  max_soft_tokens: 1120     # try 140 / 280 / 560 / 1120
```

Highest-value experiment available for image quality. Applies to both backends
identically, so it can be evaluated on whichever is more convenient.

**3. Composite alpha onto white in `load_cv2_pil`** — only if [§4.3](#43-scope-how-much-does-this-matter)
shows your corpus contains transparency. To match the server exactly, the decode needs to
read the alpha channel and composite over white rather than discarding it:

- read with `cv2.IMREAD_UNCHANGED` so the alpha channel survives, then composite; or
- decode with Pillow directly and reuse vLLM's own logic (`Image.open` →
  `ImageOps.exif_transpose` → composite over `(255,255,255)`), which also picks up the
  palette-`tRNS` case that a mode check alone misses.

The second is the safer match, since it is literally the same code path the server runs.

**4. Add `video: 0` to `limit_mm_per_prompt`.** Removes the ~2496-token video floor from
the batched-token calculation. Small, free, and unrelated to whether you use images.

**5. When enabling images:** set `is_llm: false` and `limit_mm_per_prompt.image` to the
real maximum images per prompt.

---

## Appendix A: the image pipeline end to end

Consolidated, in execution order, with the online/offline verdict at each stage.

| # | stage | code | online | offline | same? |
|---|---|---|---|---|---|
| 1 | decode bytes → PIL | `ImageMediaIO.load_bytes` / our `load_cv2_pil` | Pillow | OpenCV | **no** (alpha) |
| 2 | EXIF orientation | `normalize_image` / cv2 built-in | yes | yes | yes |
| 3 | mode → RGB | `rgba_to_rgb` / `.convert("RGB")` | composite on white | alpha dropped | **no** |
| 4 | pixel-count guard | `VLLM_MAX_IMAGE_PIXELS` (raise) | yes | n/a (not a resize) | n/a |
| 5 | chat template | `chat_template.jinja` | server-side | our `apply_chat_template` | yes (tested) |
| 6 | resolve `max_soft_tokens` | `_get_max_soft_tokens` | engine-level | engine-level | yes |
| 7 | aspect-preserving resize | `get_aspect_ratio_preserving_size` | shared | shared | yes |
| 8 | rescale to [0,1] | `rescale_and_normalize` | shared | shared | yes |
| 9 | patchify (16px) | `convert_image_to_patches` | shared | shared | yes |
| 10 | pad to `max_patches` | `pad_along_first_dim` | shared | shared | yes |
| 11 | placeholder expansion | `replace_image_token` | shared | shared | yes |
| 12 | tokenize | `add_special_tokens=False` | shared | shared | yes |

Stages 1 and 3 are ours to fix. Stages 5–12 are provably shared.

---

## Appendix B: reproduce every check in this doc

```bash
VENV=/fsxvision_new/srihari.bandarupalli/environments/gemma4_new
VLLM=$VENV/lib/python3.12/site-packages/vllm
TF=$VENV/lib/python3.12/site-packages/transformers
SNAP=/hf_cache/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475

# The processor config that drives all resizing
python -c "import json;print(json.dumps(json.load(open('$SNAP/processor_config.json'))['image_processor'],indent=2))"

# default_output_length = 280, the fallback budget
python -c "import json;print(json.load(open('$SNAP/config.json'))['vision_config']['default_output_length'])"

# The resize function and the supported-values tuple
sed -n '29,90p'   $TF/models/gemma4/image_processing_gemma4.py
sed -n '200,235p' $TF/models/gemma4/image_processing_gemma4.py

# vLLM's loader: confirm no resize, only guard + EXIF + mode
sed -n '73,92p' $VLLM/multimodal/media/image.py
sed -n '21,60p' $VLLM/multimodal/image.py
grep -n "VLLM_MAX_IMAGE_PIXELS" $VLLM/envs.py

# vLLM's Gemma4 plumbing
grep -n "max_soft_tokens\|_VIDEO_MAX\|add_special_tokens" $VLLM/model_executor/models/gemma4_mm.py

# Per-request override reachability (both backends)
grep -n "mm_processor_kwargs" $VLLM/entrypoints/openai/chat_completion/protocol.py
grep -n "mm_processor_kwargs" $VLLM/renderers/base.py
```

**Probe 1 — resize behaviour across sizes and budgets.**

```python
from transformers.models.gemma4.image_processing_gemma4 import (
    get_aspect_ratio_preserving_size as g)

for mst in (70, 140, 280, 560, 1120):
    mp = mst * 9                                  # pooling_kernel_size**2
    print(f"\n--- max_soft_tokens={mst} (max_patches={mp}, budget={mp*256:,} px) ---")
    for (w, h) in [(4096,3072),(2048,1536),(1024,768),(800,600),(512,384),(224,224)]:
        th, tw = g(height=h, width=w, patch_size=16,
                   max_patches=mp, pooling_kernel_size=3)
        n = (th//16)*(tw//16)//9
        print(f"  {w}x{h} ({w*h:>10,} px) -> {tw}x{th} "
              f"({tw*th:>9,} px)  scale={tw/w:.3f}  soft_tokens={n}")
```

**Probe 2 — decode divergence.** Note this reimplements `ImageMediaIO.load_bytes`
verbatim rather than importing it: `import vllm` fails outside a CUDA-capable environment
(`ImportError: libcudart.so.13`), and the logic is short enough to transcribe exactly.
Diff it against `$VLLM/multimodal/media/image.py:73` before trusting it.

```python
import contextlib, numpy as np, cv2
from io import BytesIO
from PIL import Image, ImageOps

def normalize_image(im):
    with contextlib.suppress(Exception):
        im = ImageOps.exif_transpose(im)
    return im

def rgba_to_rgb(im, bg=(255,255,255)):
    conv = Image.new("RGB", im.size, bg)
    conv.paste(im, mask=im.split()[3])
    return conv

def _has_transparency(im):
    return im.mode in ("RGBA","LA","PA") or "transparency" in getattr(im, "info", {})

def convert_image_mode(im, to="RGB", bg=(255,255,255)):
    if im.mode == to: return im
    if to == "RGB" and _has_transparency(im):
        if im.mode != "RGBA": im = im.convert("RGBA")
        return rgba_to_rgb(im, bg)
    return im.convert(to)

def srv(p):                       # vLLM server path
    im = Image.open(BytesIO(open(p, 'rb').read()))
    im = normalize_image(im); im.load()
    if im.mode != "RGB":
        im = rgba_to_rgb(im) if im.mode == "RGBA" else convert_image_mode(im, "RGB")
    return np.array(im)

def ours(p):                      # utils.load_cv2_pil path
    img = cv2.imread(str(p))
    return np.array(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGB"))

Image.new("RGBA", (64,64), (255,0,0,0)).save("t_rgba.png")
print("transparent :", srv("t_rgba.png")[0,0], ours("t_rgba.png")[0,0])
Image.new("RGBA", (64,64), (0,0,255,128)).save("t_a50.png")
print("50% alpha   :", srv("t_a50.png")[0,0], ours("t_a50.png")[0,0])
Image.new("P", (32,32)).save("t_pal.png", transparency=0)
print("palette tRNS:", srv("t_pal.png")[0,0], ours("t_pal.png")[0,0])
```

**Probe 3 — chat-template equivalence.**

```python
from transformers import AutoProcessor
from PIL import Image
SNAP = "/hf_cache/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475"
p  = AutoProcessor.from_pretrained(SNAP, trust_remote_code=True)
im = Image.new("RGB", (64,64))

ours = [{"role":"user","content":[{"type":"image","image":im},
                                  {"type":"image","image":im},
                                  {"type":"text","text":"Describe."}]}]
srv  = [{"role":"user","content":[{"type":"image"},
                                  {"type":"image"},
                                  {"type":"text","text":"Describe."}]}]
a = p.apply_chat_template(ours, tokenize=False, add_generation_prompt=True)
b = p.apply_chat_template(srv,  tokenize=False, add_generation_prompt=True)
print(repr(a)); print(repr(b)); print("IDENTICAL:", a == b)
```
