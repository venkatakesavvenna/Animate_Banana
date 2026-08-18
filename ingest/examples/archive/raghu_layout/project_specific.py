from PIL import Image
from copy import deepcopy
from pathlib import Path
import json, re, traceback, os
import ast
from vision_ingest.utils.main_logger import MainLogger
from vision_ingest.vllm_module.vllm import StructuredOutputsParams, SamplingParams

class PromptAndValidation:
    def __init__(self, prompt_path, structured_outputs_params: StructuredOutputsParams = None):
        """
        Load prompt text from file. Raises on missing/unreadable file.
        
        structured_outputs_params: Use this when you want the VLM to return output
        in a strict schema (e.g. a Pydantic model or JSON schema). Leave as None
        for free-form text or when you're handling structure yourself in
        preprocess_vlm_output().
        """
        try:
            with open(prompt_path, 'r') as f:
                self.prompt_str = f.read()
            self.structured_outputs_params: StructuredOutputsParams = structured_outputs_params
            self.enable_thinking = True
        except Exception as e:
            print(f"Failed to read prompt file from {prompt_path}: {e}")
            raise

    def get_sampling_params(self):
        return SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=40,
            min_p=0.01,
            repetition_penalty=1.0,
            max_tokens=128000,
            stop_token_ids=[],
            # structured_outputs=self.structured_outputs_params
        )
   

    def write_local_json(self, image_path: str, processed_output: str, logger: MainLogger):
        """
        The entire data is already being stored in sharded JSONL files automatically.
        Use this only if you also want to write each image's output to a separate
        image-specific file (e.g. a per-image JSON). The processed_output here is
        whatever wrap_and_validate_output() returned.

        Example:
            output_dir = Path(image_path).parent.parent / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            out = output_dir / (Path(image_path).stem + ".json")
            with open(out, 'w') as f:
                json.dump({"output": processed_output}, f)
        """
        pass

    def to_det_json(self, page_path: Path, root: Path = None, output_folder: str = "", file_suffix="", extension=".json", return_dir=False) -> Path:
        """
        Convert a page image path into a JSON detection output path.

        Example:
            page_path = /root/doc_id/pages/abc.png
            root = /root
            output_folder = "outputs/text_detections"

        Returns:
            /root/doc_id/outputs/text_detections/abc.json
        """
        if not isinstance(page_path, Path):
            print(page_path, type(page_path))
            
        doc_id_path = page_path.parent.parent / output_folder # rel.parent = "doc_id/pages" → rel.parent.parent = "doc_id"
        if return_dir:
            return doc_id_path
        return doc_id_path / f"{page_path.stem}{file_suffix}{extension}" # doc_id/outputs/text_detections/abc.json


    def get_modified_image_path(self, image_path):
        """
        Return the image path(s) to actually send to the VLM.

        Use this when the VLM should see a different file than the one stored
        in the database (e.g. a pre-processed variant, a SOM image, or
        multiple images for a multi-image prompt).

        Returns:
            list[str]: One or more image paths.
                - Single image  → [image_path] or [modified_path]
                - Multi-image   → [image_path, som_path, other_path]

        Default: passes the original path through unchanged.
        """
        # return [self.to_det_json(Path(image_path),
        #                         root=None,
        #                         output_folder="outputs/layout_detections_newensemble",
        #                         extension=".jpg",
        #                         file_suffix=""
        #                     )]
        return [image_path]

    def get_image_specific_prompt(self, image_path, logger):
        """
        Customize the prompt for each image before it is sent to the VLM.

        The prompt string and image paths returned by this function are what
        get passed directly to the VLM at inference time — so any per-image
        metadata, placeholder replacement, or path modification must happen here.

        Override this to replace placeholders in the prompt (e.g. <WIDTH>,
        <HEIGHT>, <IMAGE_PLACEHOLDER>) with actual values, or to append any
        image-specific context. Image path resolution is handled via
        get_modified_image_path().

        Returns:
            tuple[str, list[str]]: (prompt_to_run, image_paths_to_run_on)
        """
    
        prompt = deepcopy(self.prompt_str)
        image_path_to_send_to_vlm = deepcopy(image_path)
        return prompt, image_path_to_send_to_vlm
        try:
            mapping_path = self.to_det_json(Path(image_path), None, "outputs/layout_detections_newensemble", file_suffix="_color")
            with open(mapping_path, "r") as f:
                mapping_data = json.load(f)
            # Limit mapping_data to a maximum of 50 numbered entries (skip non-integer keys like "bbox_color").
            numbered_entries = {k: v for k, v in mapping_data.items() if k.isdigit()}
            ordered_keys = sorted(numbered_entries.keys(), key=lambda x: int(x))
            limited_keys = ordered_keys[:50]
            limited_mapping_data = {k: mapping_data[k] for k in limited_keys}
            # Retain extra keys present in the original (e.g., "bbox_color") if needed by the prompt or model.
            for k in mapping_data:
                if not k.isdigit():
                    limited_mapping_data[k] = mapping_data[k]
            if len(numbered_entries) > 50:
                logger.info(f"Truncating mapping_data from {len(numbered_entries)} entries to 50 for prompt input.{mapping_path}")

            prompt = prompt.replace("<INPUT_JSON>", json.dumps(limited_mapping_data)).replace("<max>", str(len(mapping_data)-1))

            image_path_to_send_to_vlm = self.get_modified_image_path(image_path_to_send_to_vlm)
        except Exception as e:
            logger.error(f"get_image_specific_prompt() failed for image_path {image_path}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        finally:
            return prompt, image_path_to_send_to_vlm

    def preprocess_vlm_output(self, raw_vlm_output):
        """
        Clean and extract the raw VLM response into your target format.

        This runs before validate_output(). Common things to do here: strip
        thinking tokens (</think>), pull JSON out of a ```json fence with
        json.loads(), or extract HTML with re.search(r'```html...'). Return
        whatever type validate_output() expects.

        Raises ValueError if extraction fails — triggers a retry.

        Default: returns the raw string stripped of whitespace.
        """
        return raw_vlm_output.strip()

    def validate_output(self, processed_output):
        """
        Check that the preprocessed output meets your requirements.

        Raise ValueError with a clear message if something is wrong — e.g.
        check that all expected JSON fields are present, or verify that the
        returned HTML parses without errors. Return True if everything is fine.

        Any exception raised here will trigger a prompt retry.

        Default: rejects None and empty strings only.
        """
        if processed_output is None:
            raise ValueError("Output is None")
        if isinstance(processed_output, str) and not processed_output.strip():
            raise ValueError("Output is empty string")
        return True

    def wrap_and_validate_output(self, raw_vlm_output):
        """
        Full output pipeline: preprocess → validate → return.

        The raw VLM output for each image is passed in here. Do all required
        cleaning, extraction, validation, and any wrapping (e.g. into an HTML
        format for saving) inside preprocess_vlm_output() and validate_output().

        If this function raises an exception, the prompt will be retried. If
        retries are exhausted, the image path is saved as failed.

        The value returned here is written via json.dumps() to the shared JSONL
        file and also passed to write_local_json() for any per-image saving.

        Do not override this. Override preprocess_vlm_output() and
        validate_output() instead.
        """
        # try:
        #     processed_output = self.preprocess_vlm_output(raw_vlm_output)
        #     self.validate_output(processed_output)
        #     return processed_output
        # except Exception as e:
        #     raise ValueError(f"Validation failed: {str(e)}")
        try:
            # html_str = self.extract_html_str(vlm_output)
            # self.validate_html(html_str)  # Validate HTML structure
            # if "</think>" in raw_vlm_output:
                # print("thinked")
            think_end = raw_vlm_output.rfind("</think>")
            if think_end != -1:
                raw_vlm_output = raw_vlm_output[think_end + len("</think>"):]
            start = raw_vlm_output.find("[")
            end = raw_vlm_output.rfind("]")

            if start != -1 and end != -1:
                data = json.loads(raw_vlm_output[start:end+1])
            return raw_vlm_output
        except Exception as e:
            raise ValueError(f"Validation failed: {str(e)}")
