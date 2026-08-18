from copy import deepcopy
import json, re, traceback, os
from vision_ingest.utils.main_logger import MainLogger
from vision_ingest.vllm_module.vllm import StructuredOutputsParams

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CAPTIONS_JSONL = os.path.join(_THIS_DIR, "inputs", "eval_500_diff.jsonl")


class PromptAndValidation:
    """
    Hindi caption-rewrite task: read English VLM captions from a JSONL, ask
    Gemma-4-31B (served as a text-only LLM, no images) to translate them.

    This replaces the earlier ad hoc setup — `docker run vllm/vllm-openai
    ... google/gemma-4-31B-it` plus a standalone script hammering
    /v1/chat/completions with a ThreadPoolExecutor — with the same
    `vllm serve` online serving under VIE's own online backend (v1.6,
    backend="online"): DB-tracked, crash-safe, retried on failure.

    There are no real images here, so `image_paths_source` (see main.py) is
    just a list of the JSONL's "path" values used as opaque DB keys — the
    ingest/DB layer never opens or validates them, and `is_llm: true` in
    vllm_model.yaml makes VLLMConfig send `image_paths: []` to the server
    regardless of what's passed through (see vllm_config.get_prompt_with_image's
    online branch). get_image_specific_prompt() below uses each key only to
    look up the matching record in the preloaded JSONL.
    """

    def __init__(self, prompt_path, captions_jsonl_path=DEFAULT_CAPTIONS_JSONL,
                 structured_outputs_params: StructuredOutputsParams = None):
        try:
            with open(prompt_path, 'r') as f:
                self.prompt_str = f.read()
            self.structured_outputs_params: StructuredOutputsParams = structured_outputs_params
            # None (not False) — matches the original bakeoff script's default
            # of not passing --thinking/--no-thinking at all, so no
            # chat_template_kwargs.enable_thinking is sent and the server's
            # own default applies (see online_worker._build_chat_payload).
            self.enable_thinking = None
        except Exception as e:
            print(f"Failed to read prompt file from {prompt_path}: {e}")
            raise

        self.english_captions_by_path = {}
        with open(captions_jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                path = record.get("path")
                if path:
                    self.english_captions_by_path[path] = record.get("english_captions") or {}

    def get_sampling_params(self):
        # None -> defer entirely to VLLMConfig.get_sampling_params(), which
        # merges google/gemma-4-31B-it's generation_config.json (temperature,
        # top_p, top_k, ...) the same way `vllm serve`'s HTTP layer does for
        # any field the request leaves unset (docs/vllm_serve_offline_parity.md,
        # Divergence 2) — instead of hardcoding values here that would only
        # ever fall back to vLLM's neutral SamplingParams defaults offline.
        # If a future task needs to deviate, return only the fields to
        # override (a dict, e.g. {"max_tokens": 16096}); every other field
        # still resolves from generation_config.json, not vLLM's neutral
        # defaults.
        if self.structured_outputs_params is not None:
            return {"structured_outputs": self.structured_outputs_params}
        return None

    def write_local_json(self, image_path: str, processed_output: str, logger: MainLogger):
        pass

    def get_modified_image_path(self, image_path):
        # Text-only task — no image ever gets sent, regardless of backend.
        return []

    def get_image_specific_prompt(self, image_path, logger):
        """
        `image_path` here is the JSONL record's "path" value, used purely as
        a lookup key — nothing is read from disk. Mirrors
        online_inference_bakeoff.py's build_prompts(): inject the record's
        english_captions JSON in place of <ENGLISH_VLM_OUTPUT> in the prompt
        template.
        """
        prompt = deepcopy(self.prompt_str)
        try:
            english_output = self.english_captions_by_path.get(image_path)
            if english_output is None:
                logger.warning(f"No English captions found for {image_path}, skipping.")
                english_output = {}
            english_json_str = json.dumps(english_output, ensure_ascii=False, indent=2)
            prompt = prompt.replace("<ENGLISH_VLM_OUTPUT>", english_json_str)
        except Exception as e:
            logger.error(f"get_image_specific_prompt() failed for image_path {image_path}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return prompt, self.get_modified_image_path(image_path)

    def preprocess_vlm_output(self, raw_vlm_output):
        return raw_vlm_output.strip()

    def validate_output(self, processed_output):
        if processed_output is None:
            raise ValueError("Output is None")
        if isinstance(processed_output, str) and not processed_output.strip():
            raise ValueError("Output is empty string")
        return True

    def wrap_and_validate_output(self, raw_vlm_output):
        try:
            processed_output = self.preprocess_vlm_output(raw_vlm_output)
            self.validate_output(processed_output)
            return processed_output
        except Exception as e:
            raise ValueError(f"Validation failed: {str(e)}")
