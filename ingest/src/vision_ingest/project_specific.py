from PIL import Image
from copy import deepcopy
from pathlib import Path
import json, re, traceback, os
from html.parser import HTMLParser
from vision_ingest.utils.main_logger import MainLogger
from vision_ingest.vllm_module.vllm import StructuredOutputsParams, SamplingParams


class PromptAndValidation:
    def __init__(self, prompt_path, structured_outputs_params: StructuredOutputsParams = None):
        try:
            with open(prompt_path, 'r') as f:
                self.prompt_str = f.read()
            self.structured_outputs_params: StructuredOutputsParams = structured_outputs_params
            self.enable_thinking = False # Will be sent AutoProcessor/AutoTokenizer .apply_chat_template() in VLLM_Config.py
        except Exception as e:
            print(f"Failed to read prompt file from {prompt_path}: {e}")
            raise
    
    def get_sampling_params(self):
        """
        Optional sampling-param override, layered on top of
        VLLMConfig.get_sampling_params() (see prediction.py's
        _base_sampling_params()), which already merges the model's
        generation_config.json for temperature/top_p/top_k/min_p/
        repetition_penalty/max_tokens — the same per-field merge `vllm serve`
        does for any request field left unset. Only override something here
        if this task genuinely needs to deviate from what the model (and
        `vllm serve`) would otherwise do by default.

        Returns:
            None: (default) use the generation_config-derived base as-is.
            dict: override only these fields, e.g. {"max_tokens": 2048} —
                every other field still resolves from generation_config.json.
            SamplingParams: full replacement of the base (legacy — use a
                dict instead unless you deliberately want every field,
                including ones you didn't set, to fall back to vLLM's
                neutral SamplingParams defaults rather than the model's own
                generation_config.json).
        """
        if self.structured_outputs_params is not None:
            return {"structured_outputs": self.structured_outputs_params}
        return None

    def build_retry_request(self, sent_prompt: dict, sampling_params: SamplingParams,
                            attempt: int, last_error: str):
        """
        Build the request for one RETRY, given the original request and why the
        previous attempt failed. Never called for a fresh item (attempt == 0).

        The default reproduces what `prediction.py` used to hardcode: append the
        error to the prompt text and bump `repetition_penalty` by 0.5 per prior
        attempt. Override it when your task wants something else — lower the
        temperature instead, switch to a stricter instruction on attempt 2, add
        a few-shot correction example, or (often the right call for long-form
        OCR transcription) drop the repetition_penalty bump entirely, since
        attempt 2 otherwise runs at ~2.0 and will actively degrade the output.

        Args:
            sent_prompt:     a FRESH COPY of the original request payload
                             ({"prompt": raw text, "image_paths": [...],
                             "enable_thinking": bool}). `prompt` is raw,
                             un-templated user text on every backend, so text
                             appended here always lands in the user turn.
            sampling_params: a DEEP COPY of the resolved base sampling params.
            attempt:         how many attempts have already been made (>= 1).
            last_error:      the error text from the most recent attempt.

        Both arguments are private copies — `prediction.py` never hands over the
        originals, precisely so mutating them in place here is safe. Returns
        `(prompt_dict, sampling_params)`.
        """
        prompt = dict(sent_prompt)
        prompt["prompt"] = (
            sent_prompt["prompt"]
            + f"\n\nPrevious attempt failed: {last_error}"
            + "\nFix and regenerate valid output."
        )
        base_rp = getattr(sampling_params, "repetition_penalty", 1.0) or 1.0
        sampling_params.repetition_penalty = base_rp + attempt * 0.5
        return prompt, sampling_params

    def write_local_json(self, obj, logger: MainLogger):
        pass

    def get_image_data(self, image_path:Path):
        return None,None,None

    def get_modified_image_path(self, image_path):
        """ 
        For given image_path, you can either pass the original image as it is or pass the SOM image saved at a different location.
        But make sure that differnet location can be found out using the original image_path.
        """
        # implement according to user choice
        return [image_path]
    
    # ----- Input Prompt specific function ------
    def get_image_specific_prompt(self, image_path, logger):
        """
        Customizes the prompt for each image by replacing placeholders with image-specific data.
        
        Replaces <WIDTH>, <HEIGHT>, and <IMAGE_PLACEHOLDER> in the prompt with actual values.
        Also allows for image path modification via get_modified_image_path().
        
        Returns:
            tuple: (customized_prompt, image_path_to_send_to_vlm)
                - customized_prompt: str with placeholders replaced
                - image_path_to_send_to_vlm: str or list of str (for multi-image prompts)
            On error, returns the working copies at the point of failure (may be partially modified).
        """
        prompt  = deepcopy(self.prompt_str)
        image_path_to_send_to_vlm = deepcopy(image_path)
        try:
            width,height,image_ids_bboxes  = self.get_image_data(image_path)
            # Replace dimension placeholders
            if width is not None and height is not None:
                prompt = prompt.replace("<WIDTH>", str(width))
                prompt = prompt.replace("<HEIGHT>", str(height))

            # Remove residual image placeholder text if present
            prompt = prompt.replace("<IMAGE_PLACEHOLDER>", image_ids_bboxes).strip() 
            image_path_to_send_to_vlm = self.get_modified_image_path(image_path_to_send_to_vlm)
            
        except Exception as e:
            logger.error(f"get_image_specific_prompt() failed for image_path {image_path}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        finally:
            return prompt, image_path_to_send_to_vlm  # Returns working copies (may be partially modified if error occurred)

    # --- VLM output specific function ----
    def extract_html_str(self, response_text: str) -> str:
        """
        Extracts the HTML string from the model response with robust fallback mechanisms.
        """
        # 1. Clean the <think> block
        think_end = response_text.rfind("</think>")
        if think_end != -1:
            response_text = response_text[think_end + len("</think>"):]

        # 2. Pattern to match code blocks (Markdown)
        # We capture flags (like 'html') and the content.
        # We allow the closing ``` to be optional to handle truncated responses.
        code_block_pattern = re.compile(r'```(\w*)\n(.*?)(?:```|$)', re.DOTALL)
        
        matches = list(code_block_pattern.finditer(response_text))
        
        html_candidates = []

        # 3. Filter and collect candidates from Markdown blocks
        for match in matches:
            lang = match.group(1).lower().strip()
            content = match.group(2).strip()
            
            # Accept explicit HTML blocks or generic blocks that look like HTML
            if 'html' in lang or ('<!DOCTYPE' in content or '<html' in content):
                html_candidates.append(content)

        # 4. Selection Logic: Choose the longest candidate
        # This prevents selecting a small correction snippet at the end over the main code.
        if html_candidates:
            return max(html_candidates, key=len)

        # 5. Fallback: Direct HTML tag extraction
        # If no markdown blocks were used, look for the raw HTML structure.
        html_start = re.search(r'<!DOCTYPE html>|<html', response_text, re.IGNORECASE)
        html_end = re.search(r'</html>', response_text, re.IGNORECASE)

        if html_start:
            start_idx = html_start.start()
            # If we have an end tag, cut there; otherwise take everything (truncated)
            end_idx = html_end.end() if html_end else len(response_text)
            return response_text[start_idx:end_idx].strip()

        # 6. Final Fallback: Return cleaned text
        return response_text.strip()
    
    def validate_html(self, html_str: str) -> bool:
        """
        Validate if the string is valid HTML.
        Returns True if valid, raises ValueError if invalid.
        """
        class StrictHTMLParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.errors = []
            
            def error(self, message):
                self.errors.append(message)
        
        parser = StrictHTMLParser()
        try:
            parser.feed(html_str)
            parser.close()
            
            if parser.errors:
                raise ValueError(f"HTML parsing errors: {'; '.join(parser.errors)}")
            
            # Check if there's at least some HTML-like content
            if not re.search(r'<[^>]+>', html_str):
                raise ValueError("No HTML tags found in output")
            return True

        except Exception as e:
            raise ValueError(f"Invalid HTML: {str(e)}")

    def wrap_and_validate_output(self, vlm_output):
        """
        We pass the raw vlm output for each image to the output_str and here we do all the requried validation and in case of error we just raise an execption.
        We can also use this to wrap our validated outputs into an HTML format for saving.
        If this function raises an error.. then we will retry the prompt and if even retires fail, then we will save the image_path as failed.
        If this function return a string will be used with json.dumps(str) to save the output to a common jsonl file and at the image_path aslo.
        """
        try:
            html_str = self.extract_html_str(vlm_output)
            self.validate_html(html_str)  # Validate HTML structure
            return vlm_output
        except Exception as e:
            raise ValueError(f"Validation failed: {str(e)}")
        