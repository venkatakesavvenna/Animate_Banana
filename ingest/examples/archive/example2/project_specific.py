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
        return SamplingParams(
            temperature=0.2,
            max_tokens=4096,
            stop_token_ids=[],
            repetition_penalty=1,
            structured_outputs=self.structured_outputs_params
        )
    
    def to_det_json(self, page_path: Path, root: Path, output_folder: str, file_suffix="", extension=".json", return_dir=False) -> Path:
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
        rel = page_path.relative_to(root) # doc_id/pages/abc.png

        doc_id_path = rel.parent.parent # rel.parent = "doc_id/pages" → rel.parent.parent = "doc_id"
        
        stem = page_path.stem # Extract filename without extension (abc from abc.png)
        
        out_rel = doc_id_path / output_folder # doc_id/outputs/text_detections

        if return_dir:
            return out_rel

        out_rel = out_rel / f"{stem}{file_suffix}{extension}" # doc_id/outputs/text_detections/abc.json

        # Attach to root → /root/doc_id/outputs/text_detections/abc.json
        return root / out_rel

    def write_local_json(self, obj, logger: MainLogger):
        try:
            image_path = Path(obj["path"])
            # prompt_out_path = obj["path"].rsplit(".", 1)[0] + "_prompt.txt"
            # out_path = obj["path"].rsplit(".", 1)[0] + ".html"

            out_path = self.to_det_json(image_path, image_path.parent.parent, "outputs/vlm_html_v2", 
                                   file_suffix="", extension=".html")
            prompt_out_path = self.to_det_json(image_path, image_path.parent.parent, "outputs/vlm_html_v2",
                                      file_suffix="_prompt", extension=".txt")

            os.makedirs(os.path.dirname(prompt_out_path), exist_ok=True)
            with open(prompt_out_path, "w", encoding="utf-8") as f:
                f.write(obj["prompt_str"])
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(obj["vlm_output"])
        except Exception as e:
            logger.log_error(f"Error writing local JSON for path {obj['path']}", e)
            raise

    def get_image_data(self, image_path:Path):
        image_path = Path(image_path)
        with Image.open(image_path) as img:
            width, height = img.size
        # Example: generate a string of image IDs and bounding boxes
        json_path = self.to_det_json(image_path, image_path.parent.parent, "outputs/layout_detections", 
                                   file_suffix="", extension=".json")
        cnt = 0
        with open(json_path, 'r') as f:
            detections = json.load(f)
            image_ids_bboxes = ""
            for i,det in enumerate(detections):
                cls_name = det.get("class_name", "")[0]
                if cls_name.lower() != "picture":
                    continue
                
                x1, y1, x2, y2 = det.get("bbox", [])
                w, h = x2 - x1, y2 - y1
                image_ids_bboxes += f"image_src:img_{cnt}.png image_size: {w}x{h} px.\n"
                cnt+=1
        return width, height, image_ids_bboxes.strip()

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
        