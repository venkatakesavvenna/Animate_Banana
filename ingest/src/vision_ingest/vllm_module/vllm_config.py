from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Union
from pathlib import Path
from .vllm import SamplingParams, StructuredOutputsParams
from .media import (
    DEFAULT_MAX_IMAGE_PIXELS,
    ImagePolicy,
    ImageValidationError,
    validate_image_path,
)
import os
import yaml
import time
import threading
from vision_ingest.utils.utils import GracefulShutdown
from transformers import AutoProcessor, AutoTokenizer, GenerationConfig

# Sampling-only knobs that live under a model's `engine_args:` yaml block
# (read by get_sampling_params()) but are not real vLLM engine kwargs — must
# never reach AsyncEngineArgs(**engine_args) / --serve-flag translation.
_SAMPLING_ONLY_ENGINE_ARG_KEYS = {"sampling_temperature", "sampling_max_tokens"}

# Mirrors vLLM's ModelConfig.get_diff_sampling_param(): the only fields the
# HTTP server ever merges in from generation_config.json.
_GENERATION_CONFIG_SAMPLING_KEYS = (
    "repetition_penalty", "temperature", "top_k", "top_p", "min_p",
)
# generation_config.json's token-budget field has a different name than
# SamplingParams'; map it explicitly rather than folding it into the
# same-named loop above.
_GENERATION_CONFIG_MAX_TOKENS_KEY = "max_new_tokens"

# vLLM 0.25+ turns async scheduling ON by default (config/vllm.py: an unset
# async_scheduling resolves to True unless some other option forbids it). It
# overlaps CPU scheduling with the previous forward pass and has been the cause
# of hard node-level hangs/crashes here, so this codebase pins it OFF unless a
# model yaml explicitly opts in with `async_scheduling: true`. See
# docs/v1.7_changes.md.
DEFAULT_ASYNC_SCHEDULING = False


class PromptConfigError(RuntimeError):
    """
    The prompt cannot be built because of a CONFIGURATION mistake, not because
    of anything about this particular image (is_llm set on a model being sent
    images, more images than limit_mm_per_prompt allows, ...).

    Deliberately distinct from a per-image failure: a config error affects every
    image equally, so failing them one at a time would quietly march the whole
    dataset into state=3. This propagates instead — the prep thread dies, the
    run aborts, and the in-flight paths are reset to their fetch state.
    """


class VLLMConfig():
    """
    Model configuration: engine args, sampling params, and image policy.

    As of v1.8 this no longer knows or cares which backend will serve the
    request. Every backend receives the same payload — raw user text plus image
    *paths* — and the worker turns that into whatever its engine needs. That is
    what let `backend`, `payload_kind`, and the whole offline prompt-building
    branch (PIL decode + `apply_chat_template` + placeholder assertion) leave
    this class: there is only one payload shape now, so there is nothing left to
    choose between. See `core/wire.py` and `docs/v1.8_changes.md`.
    """

    def __init__(self, model_name: str, config_path: str = "config/vllm_model.yaml",
                 allowed_media_roots=None,
                 max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS):
        self.model_name = self.get_model_path(model_name)
        if self.model_name is None:
            self.model_name = model_name # assume it's a valid huggingface model path or local path
        with open(config_path, "r", encoding="utf-8") as file:
            all_configs = yaml.safe_load(file)

        self.model_specific_config = all_configs.get('models').get(self.model_name)
        if not self.model_specific_config:
            raise ValueError(f"Model '{self.model_name}' not found in {config_path}")

        self.is_llm = self.model_specific_config.get('is_llm', False)
        self.model = None

        # Use AutoTokenizer for pure LLMs, AutoProcessor for multimodal models.
        # Since v1.8 nothing here applies a chat template (the worker does, with
        # vLLM's own renderer); the processor is kept because it is the only way
        # to discover the model's own image-resize budget for the startup log.
        if self.is_llm:
            self.processor = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)

        # How many images of each modality one prompt may carry, straight from
        # the engine_args the engine/server is actually started with. Sending
        # more is a 400 online and a ValueError offline; we catch it before
        # submission so it reads as one clear per-item error.
        self.max_images_per_prompt = self._max_images_per_prompt()

        # Everything an image path must satisfy before it may be submitted.
        # read_header is now unconditional: since v1.8 the stage never decodes
        # an image itself on any backend, so this header read is the ONLY local
        # chance to reject a corrupt or oversized file — on every backend, not
        # just online — before it becomes an opaque worker-side failure.
        self.allowed_media_roots = self._resolve_media_roots(allowed_media_roots)
        self.image_policy = ImagePolicy(
            allowed_media_roots=self.allowed_media_roots,
            max_pixels=max_image_pixels,
            processor_max_pixels=self._processor_max_pixels(),
            read_header=True,
        )

        self.sampling_params = self.get_sampling_params()
        # Built last: it folds allowed_media_roots into `allowed_local_media_path`
        # so the engine (offline) and `vllm serve` (online) are both configured
        # from the same single source as the stage-side path validation.
        self.engine_args = self.get_engine_args()

    def _resolve_media_roots(self, allowed_media_roots) -> tuple:
        """
        Normalise and validate the directories images may be read from.

        Since v1.8 *every* backend reads images from paths, so this is required
        for any vision model rather than being an online-only concern: vLLM's
        own MediaConnector refuses every `file://` URL unless
        `allowed_local_media_path` is set, whether it is running inside
        `vllm serve` or inside the in-process engine.
        """
        roots = ImagePolicy.normalise_roots(allowed_media_roots)
        if not self.is_llm and not roots:
            raise ValueError(
                "a vision model requires allowed_media_roots (the directories images "
                "live under). They become the engine's `allowed_local_media_path` / "
                "`vllm serve --allowed-local-media-path` AND the roots every image "
                "path is validated against before submission, so the two can never "
                "drift. Set args.allowed_media_roots."
            )
        return roots

    def engine_media_root(self) -> Optional[str]:
        """
        The single directory handed to vLLM as `allowed_local_media_path`.

        vLLM models this as ONE string, on both the engine config and the serve
        CLI (`config/model.py: allowed_local_media_path: str`). Passing the flag
        more than once therefore does not add a root — it overwrites, keeping
        only the last. That was a live bug before v1.8: a run configured with two
        roots had every image under the first one rejected by the server with
        "must be a subpath of --allowed-local-media-path", because only the
        second was ever in force.

        So when several roots are configured we hand vLLM their common ancestor
        and keep the exact set here. That is safe because the two checks are not
        peers: THIS process validates every path against the real root list
        before submitting it, so nothing outside them is ever sent, and vLLM's
        check is a backstop on a server we launched ourselves, on localhost,
        against paths we already vetted. Widening the backstop cannot widen what
        actually gets submitted.
        """
        if not self.allowed_media_roots:
            return None
        if len(self.allowed_media_roots) == 1:
            return self.allowed_media_roots[0]
        return os.path.commonpath(self.allowed_media_roots)

    def media_root_warning(self) -> Optional[str]:
        """A warning to log at startup when the engine's root is broader than
        the roots actually allowed here, so the widening is never invisible."""
        if len(self.allowed_media_roots) <= 1:
            return None
        return (
            f"allowed_media_roots has {len(self.allowed_media_roots)} entries "
            f"{list(self.allowed_media_roots)}, but vLLM's allowed_local_media_path "
            f"is a single directory — it is set to their common ancestor "
            f"{self.engine_media_root()!r}. Image paths are still validated against "
            "the exact list before submission, so nothing outside it is ever sent; "
            "only vLLM's own backstop is broader. Use one root if you want them to "
            "match exactly."
        )

    def get_model_path(self, model_name):
        model_paths = {
            "gemma": "google/gemma-3-27b-it",
            "qwen": "Qwen/Qwen2.5-VL-72B-Instruct",
            "qwen_14b_llm": "Qwen/Qwen3-14B",
            "qwen_3": "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "qwen_3_32b": "Qwen/Qwen3-VL-32B-Instruct",
            "qwen_3_235b": "Qwen/Qwen3-VL-235B-A22B-Instruct",
            "logics-parsing":"/projects/data/vision-team/raghuveer/model_cache/weights/Logics-MLLM_Logics-Parsing",
            "dots.ocr":"/projects/data/vision-team/raghuveer/weights/DotsOCR"
        }
        return model_paths.get(model_name, None)

    # ------------------------------------------------------------------
    # Input limits discovered from the model / engine args
    # ------------------------------------------------------------------

    def _max_images_per_prompt(self) -> Optional[int]:
        """
        The engine's own per-prompt image limit (`limit_mm_per_prompt.image`),
        or None when the yaml leaves it unset. vLLM's own default when unset is
        1 image per modality, but we do not assume that here — an unset limit
        means "let the engine decide and report", not "silently truncate".
        """
        limits = self.model_specific_config.get("engine_args", {}).get("limit_mm_per_prompt")
        if isinstance(limits, dict) and "image" in limits:
            return int(limits["image"])
        return None

    def _processor_max_pixels(self) -> Optional[int]:
        """
        The pixel budget the model's *own* image processor will downscale to.

        This is the "silent resize" that matters for OCR quality: HF image
        processors (e.g. Qwen2VLImageProcessorFast, which chandra-ocr-2 uses)
        smart-resize anything above it, identically for online and offline since
        both run the same processor inside vLLM. We cannot switch it off — it is
        how the model was trained — but reporting it at startup and counting how
        many inputs exceed it turns invisible quality loss into a logged number.
        """
        proc = getattr(self, "processor", None)
        image_proc = getattr(proc, "image_processor", proc)
        # Newer processors express it as size.longest_edge (a transformers
        # SizeDict — attribute access, NOT a real dict); older Qwen-VL ones as
        # max_pixels. Despite the name, longest_edge here is an AREA in pixels,
        # not an edge length.
        size = getattr(image_proc, "size", None)
        longest_edge = None
        if size is not None:
            longest_edge = (
                size.get("longest_edge") if isinstance(size, dict)
                else getattr(size, "longest_edge", None)
            )
        if longest_edge:
            return int(longest_edge)
        max_pixels = getattr(image_proc, "max_pixels", None)
        if max_pixels:
            return int(max_pixels)
        return None

    def describe_input_limits(self) -> dict:
        """Startup-log payload: everything that can silently alter an input."""
        return {
            "is_llm": self.is_llm,
            "max_images_per_prompt": self.max_images_per_prompt or "engine default",
            "image_policy": self.image_policy.describe(),
            "engine_allowed_local_media_path": self.engine_media_root(),
        }

    def _generation_config_defaults(self) -> dict:
        """
        The same fields, from the same source, that `vllm serve`'s HTTP layer
        merges into any request that leaves them unset
        (`ModelConfig.get_diff_sampling_param()` / `ChatCompletionRequest.
        to_sampling_params()` — see docs/vllm_serve_offline_parity.md,
        Divergence 2). Offline engines never do this merge on their own, so we
        replicate it here rather than falling back to vLLM's neutral
        SamplingParams defaults (temperature=1.0, top_p=1.0, top_k=0, ...)
        the way plain `SamplingParams(...)` construction would.
        """
        try:
            diff = GenerationConfig.from_pretrained(
                self.model_name, trust_remote_code=True
            ).to_diff_dict()
        except Exception:
            return {}
        params = {k: diff[k] for k in _GENERATION_CONFIG_SAMPLING_KEYS if k in diff}
        if _GENERATION_CONFIG_MAX_TOKENS_KEY in diff:
            params["max_tokens"] = diff[_GENERATION_CONFIG_MAX_TOKENS_KEY]
        return params

    def get_sampling_params(self):
        """
        Precedence (highest wins), matching `vllm serve`'s own per-field merge:
            explicit yaml value  >  model generation_config.json  >  vLLM neutral default
        Only fields explicitly present in the yaml override the
        generation_config-derived value; everything else is left exactly as
        `vllm serve` would compute it, so offline (AsyncLLM) and online
        (`vllm serve`) requests resolve to the same sampling params.
        """
        engine_args_cfg = self.model_specific_config.get("engine_args", {})
        params = self._generation_config_defaults()

        if "sampling_temperature" in engine_args_cfg:
            params["temperature"] = engine_args_cfg["sampling_temperature"]
        if "sampling_max_tokens" in engine_args_cfg:
            params["max_tokens"] = engine_args_cfg["sampling_max_tokens"]
        # No arbitrary cap here. If neither the yaml nor generation_config.json
        # (max_new_tokens) specify one, leave it explicitly None — vLLM's
        # InputProcessor (v1/engine/input_processor.py, shared by `vllm serve`
        # and offline engines alike) fills an unset max_tokens as
        # `max_model_len - prompt_len` at generation time. This must be an
        # explicit None, not an omitted kwarg: SamplingParams' own dataclass
        # default for max_tokens is 16, which would silently cut every
        # response to 16 tokens.
        params.setdefault("max_tokens", None)
        if "stop_token_ids" in self.model_specific_config:
            params["stop_token_ids"] = self.model_specific_config["stop_token_ids"]
        else:
            params.setdefault("stop_token_ids", [])
        if "repetition_penalty" in self.model_specific_config:
            params["repetition_penalty"] = self.model_specific_config["repetition_penalty"]

        return SamplingParams(**params)

    def get_engine_args(self):
        engine_args = {
            "model": self.model_name,
            "gpu_memory_utilization": self.model_specific_config.get("engine_args", {}).get("gpu_memory_utilization", 0.95),
        }
        for key, val in self.model_specific_config.get("engine_args", {}).items():
            # sampling_temperature/sampling_max_tokens are consumed by
            # get_sampling_params() above — they are not real engine kwargs
            # and must not reach AsyncEngineArgs(**engine_args) or be
            # translated into a bogus `vllm serve` flag.
            if key in _SAMPLING_ONLY_ENGINE_ARG_KEYS:
                continue
            engine_args[key] = val
        # Pin async scheduling off unless the yaml asked for it. vLLM leaves
        # this as None = "enable it unless something forbids it", which is not a
        # default we want to inherit implicitly on a multi-day run — see
        # DEFAULT_ASYNC_SCHEDULING above.
        engine_args.setdefault("async_scheduling", DEFAULT_ASYNC_SCHEDULING)
        # The directory images may be read from, as vLLM's own engine arg. Since
        # v1.8 the worker resolves `file://` URLs itself on every backend (the
        # in-process engine through MultiModalConfig, `vllm serve` through the
        # translated --allowed-local-media-path flag), and both refuse local
        # files outright when this is unset. Deriving it from the same
        # allowed_media_roots the stage validates against is the whole point:
        # the two cannot drift, because there is only one of them.
        media_root = self.engine_media_root()
        if media_root:
            engine_args.setdefault("allowed_local_media_path", media_root)
        return engine_args

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _check_image_inputs(self, img_paths: List[str]) -> None:
        """
        Everything that must be true before an image may be submitted, in one
        place for both backends. Raises ImageValidationError / ValueError; the
        caller turns that into a terminal failure for exactly this one item.
        """
        if self.is_llm and img_paths:
            # A text-only model (AutoTokenizer, no vision tower) cannot see
            # images. Online used to drop them on the floor and offline used to
            # attach multi_modal_data to a prompt with no image placeholder —
            # both produce confident answers about an image the model never got.
            raise PromptConfigError(
                f"is_llm=true for {self.model_name} but {len(img_paths)} image(s) were "
                "supplied for this prompt. A text-only model cannot receive images: "
                "either drop the images from get_image_specific_prompt(), or set "
                "is_llm: false in the model yaml if this really is a vision model. "
                "(Text-only prompts on a vision model are fine and need no flag.)"
            )
        if (
            self.max_images_per_prompt is not None
            and len(img_paths) > self.max_images_per_prompt
        ):
            raise PromptConfigError(
                f"prompt carries {len(img_paths)} images but the engine was started "
                f"with limit_mm_per_prompt.image={self.max_images_per_prompt}; raise "
                "the limit in the model yaml or send fewer images"
            )
        for p in img_paths:
            validate_image_path(p, self.image_policy)

    def get_prompt_with_image(self, image_paths: List[Union[str, List[str]]], prompts: List[str],
                            reader_pool: ThreadPoolExecutor, logger=None, shutdown_event: threading.Event = None,
                            enable_thinking = False):
        """
        Build final prompts with images attached.

        Args:
            image_paths: List where each element is either:
                - A single image path (str) for single-image prompts
                - A list of image paths (List[str]) for multi-image prompts
                - An empty list for text-only prompts
            prompts: List of text prompts (one per element in image_paths)
            reader_pool: ThreadPoolExecutor for parallel image loading
            logger: Optional logger for error reporting
            shutdown_event: Optional threading.Event to signal graceful shutdown

        Returns:
            List of unified request payloads, or None for failed items. Each
            payload is `{"prompt": <raw user text>, "image_paths": [...],
            "enable_thinking": bool}` — the same shape for every backend. See
            `core/wire.py`.
        """
        if len(image_paths) != len(prompts):
            raise ValueError("Number of images and prompts must match.")

        # -----------------------------------------
        # Internal worker with full error handling
        # -----------------------------------------

        def process_one(img_path_or_list, prompt):
            try:
                # Normalize to list for uniform processing
                if img_path_or_list is None:
                    img_paths = []
                elif isinstance(img_path_or_list, str):
                    img_paths = [img_path_or_list]
                elif isinstance(img_path_or_list, list):
                    img_paths = img_path_or_list
                else:
                    raise ValueError(f"Invalid image path type: {type(img_path_or_list)}")

                # Validate before the path goes anywhere near a backend. This is
                # all that is left of prompt "building" on the stage side: it is
                # cheap, and it is what turns a bad image into a precise terminal
                # failure for exactly this one item, with a reason, instead of an
                # opaque rejection from the worker several seconds later.
                self._check_image_inputs(img_paths)

                # One payload shape, every backend. The text is NOT chat-templated
                # and the images are NOT decoded — the worker does both, with
                # vLLM's own renderer, so online and offline run the identical
                # templating code path instead of two implementations that happen
                # to agree. enable_thinking rides along as a chat-template kwarg.
                #
                # Keeping the raw text under "prompt" is also what makes retries
                # correct: prediction.py appends the previous error to it, and
                # because templating now happens strictly afterwards on every
                # backend, the error text always lands in the user turn. Offline
                # it used to be appended to an already-templated string, i.e.
                # after `<|im_start|>assistant\n` — inside the model's own turn.
                return {
                    "prompt": prompt,
                    "image_paths": img_paths,
                    "enable_thinking": enable_thinking,
                }

            except PromptConfigError:
                # Not this image's fault — every image would fail the same way.
                # Let it out so the run stops instead of failing the dataset
                # one row at a time.
                raise
            except Exception as e:
                if logger:
                    logger.error(
                        f"process_one failed for image {img_path_or_list!r}: "
                        f"{type(e).__name__}: {e}"
                    )
                return None  # safe fallback.. those images will not be processed.

        # -----------------------------------------
        # Submit tasks and preserve order
        # -----------------------------------------
        futures = [
            reader_pool.submit(process_one, img_path, prompt)
            for idx, (img_path, prompt)
            in enumerate(zip(image_paths, prompts))
        ]

        final_prompts = [None] * len(image_paths)

        # Wait for all futures with shutdown checking
        for idx, future in enumerate(futures):
            while not future.done():
                if shutdown_event and shutdown_event.is_set():
                    raise GracefulShutdown("Graceful shutdown requested during image processing")
                time.sleep(0.1)

            final_prompts[idx] = future.result()  # always returns dict or None

        return final_prompts

# NOTE (v1.8): the image-placeholder assertion that used to live here is gone
# with the prompt building it guarded. It existed because this class applied the
# chat template itself and a template that ignored image parts would silently
# produce a prompt with no placeholder token — the model then answered from the
# text alone. Templating now runs inside vLLM's own renderer, the same code
# `vllm serve` uses, whose multimodal processor raises when the placeholder
# count and the attached items disagree. The check is enforced by the engine
# rather than re-implemented against it.
