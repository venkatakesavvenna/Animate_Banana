import time
from PIL import Image
from copy import deepcopy
from pathlib import Path
import hashlib
import json, re, traceback, os

from vision_ingest.utils.main_logger import MainLogger
from vision_ingest.vllm_module.vllm import StructuredOutputsParams

# Where rescaled/rendered copies are written. It must be inside
# args.allowed_media_roots (main_test.py adds it), because on the online backend
# `vllm serve` refuses to read any file outside --allowed-local-media-path — a
# scratch dir under /tmp made every single online request fail with a 400 that
# the stage could only report as "failed after 3 attempts".
# Keep it on local NVMe, not FSx: it is written once per rescaled image.
CHANDRA_SCRATCH_DIR = os.environ.get(
    "CHANDRA_SCRATCH_DIR", "/opt/dlami/nvme/chandra_ocr_scaled"
)

# Chandra's own inference code upsamples so the short edge is at least this many
# pixels. Images already at or above it are passed through untouched (see
# get_modified_image_path) — re-encoding them to JPEG would be a lossy round
# trip on every input, for nothing.
CHANDRA_MIN_SHORT_EDGE = 1536
CHANDRA_SCRATCH_JPEG_QUALITY = 95

# ── Chandra OCR Prompts ────────────────────────────────────────────────────────
ALLOWED_TAGS = ["math","br","i","b","u","del","sup","sub","table","tr","td","p","th","div","pre","h1","h2","h3","h4","h5","ul","ol","li","input","a","span","img","hr","tbody","small","caption","strong","thead","big","code","chem"]
ALLOWED_ATTRIBUTES = ["class","colspan","rowspan","display","checked","type","border","value","style","href","alt","align","data-bbox","data-label"]

PROMPT_ENDING = f"""
Only use these tags {ALLOWED_TAGS}, and these attributes {ALLOWED_ATTRIBUTES}.
Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property. Describe in detail inside the div tag. Also convert charts to high fidelity data, and convert diagrams to mermaid.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags. Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.
* Lists: Preserve indents and proper list markers.
* Rotated Text: If any text is rotated (e.g., vertical or angled), output only the rotated text exactly as it appears, without reordering or correcting orientation. Do not normalize it.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret. Reading order should be correct and natural.
""".strip()

OCR_LAYOUT_PROMPT = f"""
OCR this image to HTML, arranged as layout blocks. Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in x0 y0 x1 y1 format. Bboxes are normalized 0-1000. The data-label attribute is the label for the block.
Use the following labels:
- Caption
- Footnote
- Equation-Block
- List-Group
- Page-Header
- Page-Footer
- Image
- Section-Header
- Table
- Text
- Complex-Block
- Code-Block
- Form
- Table-Of-Contents
- Figure
- Chemical-Block
- Diagram
- Bibliography
- Blank-Page
{PROMPT_ENDING}
""".strip()


class PromptAndValidation:
    def __init__(self, prompt_path=None, structured_outputs_params: StructuredOutputsParams = None,
                 scratch_dir: str = CHANDRA_SCRATCH_DIR):
        """
        Initializes the Chandra OCR prompt and validation logic.
        """
        self.structured_outputs_params: StructuredOutputsParams = structured_outputs_params
        self.enable_thinking = False
        self.start_time = time.time()
        self.processed_count = 0

        if prompt_path and os.path.exists(prompt_path):
            try:
                with open(prompt_path, 'r', encoding="utf-8") as f:
                    self.prompt_str = f.read()
            except Exception as e:
                print(f"Failed to read prompt file from {prompt_path}: {e}")
                self.prompt_str = OCR_LAYOUT_PROMPT
        else:
            self.prompt_str = OCR_LAYOUT_PROMPT

        # Scratch dir for rescaled / PDF-rendered images.
        self.tmp_dir = Path(scratch_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def print_summary(self):
        """
        Prints a summary of the OCR run.
        """
        end_time = time.time()
        duration = end_time - self.start_time
        avg_time = duration / self.processed_count if self.processed_count > 0 else 0

        print("\n" + "━" * 50)
        print("🚀 CHANDRA OCR RUN SUMMARY")
        print("━" * 50)
        print(f"Total Images Processed: {self.processed_count}")
        print(f"Total Execution Time: {duration:.2f}s")
        print(f"Average Time per Image: {avg_time:.2f}s")
        print(f"Throughput: {self.processed_count/duration:.2f} images/s" if duration > 0 else "Throughput: 0 images/s")
        print("━" * 50 + "\n")

    def get_sampling_params(self):
        """
        Sampling overrides for chandra, as a dict overlay (the v1.6 contract).

        A dict overrides only these fields; everything else — top_k, min_p,
        stop_token_ids, and so on — still resolves from the model's own
        generation_config.json exactly the way `vllm serve` would fill it in.
        Returning a full SamplingParams here (as this used to) pinned *every*
        field, including ones nobody chose, to vLLM's neutral defaults.

        These values come from chandra's standalone inference code. If you are
        re-deriving them from the upstream repo (chandra/model/vllm.py), change
        them here — this is the only place they are set.
        """
        overrides = {
            "temperature": 0.0,
            "top_p": 0.1,
            "max_tokens": 12384,
            "repetition_penalty": 1.0,
        }
        if self.structured_outputs_params is not None:
            overrides["structured_outputs"] = self.structured_outputs_params
        return overrides

    # ------------------------------------------------------------------
    # Image preparation
    # ------------------------------------------------------------------

    def _scratch_path(self, source_path: str, suffix: str) -> Path:
        """
        Deterministic scratch filename derived from the source path.

        Deterministic, not uuid4: a uuid means a re-run (or a retry after a
        crash) writes a second copy of every image, so a long run silently fills
        the scratch filesystem. With a content-independent but path-derived name
        a replay overwrites its own previous file.
        """
        digest = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:24]
        return self.tmp_dir / f"{digest}{suffix}"

    def process_image(self, image_path: str, min_dim: int = CHANDRA_MIN_SHORT_EDGE) -> str:
        """
        Upscale so the short edge is at least `min_dim`, and return the path to
        send to the model.

        Images that are already large enough are returned UNCHANGED. The old
        version re-encoded every single image to JPEG q95 even when no resize
        was needed — a lossy round trip on 100% of inputs, plus one scratch file
        per image for a run that processes hundreds of millions of them.

        Raises on failure. Returning the original path on error (as this used
        to) hides a broken input: the model then either fails on a file the
        caller believed had been prepared, or silently OCRs the wrong pixels.
        """
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            w, h = image.size
            if min(w, h) >= min_dim:
                return image_path      # nothing to do — send the original bytes

            scale = min_dim / min(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        temp_path = self._scratch_path(image_path, ".jpg")
        image.save(temp_path, quality=CHANDRA_SCRATCH_JPEG_QUALITY)
        return str(temp_path)

    def get_modified_image_path(self, image_path: str):
        """
        Resolve the path(s) actually sent to the model: renders a PDF page when
        the path is a `<file>.pdf|<page>` reference, otherwise applies the
        short-edge upscale above. Raises on any failure — VIE turns that into a
        terminal failure for this one image, with the reason in the log.
        """
        if ".pdf|" in image_path.lower():
            import fitz

            pdf_path, page_str = image_path.rsplit("|", 1)
            page_num = int(page_str)

            doc = fitz.open(pdf_path)
            try:
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=200)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            finally:
                doc.close()

            w, h = image.size
            if min(w, h) < CHANDRA_MIN_SHORT_EDGE:
                scale = CHANDRA_MIN_SHORT_EDGE / min(w, h)
                image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            temp_path = self._scratch_path(image_path, f"_page_{page_num}.jpg")
            image.save(temp_path, quality=CHANDRA_SCRATCH_JPEG_QUALITY)
            return [str(temp_path)]

        return [self.process_image(image_path)]

    def get_image_specific_prompt(self, image_path: str, logger: MainLogger):
        """
        Generates the final prompt and resolves the (possibly rescaled) image path.

        Deliberately does NOT swallow errors: VIE's prompt-prep already catches
        per-image exceptions and terminal-fails just that image with the reason
        logged. Catching here and returning the original path instead meant a
        failed rescale was indistinguishable from a successful one.
        """
        prompt = deepcopy(self.prompt_str)
        return prompt, self.get_modified_image_path(image_path)

    # ------------------------------------------------------------------
    # Output handling
    # ------------------------------------------------------------------

    def preprocess_vlm_output(self, raw_vlm_output):
        """
        Basic cleanup of the VLM output.
        """
        return raw_vlm_output.strip()

    def validate_output(self, processed_output):
        """
        Basic validation for OCR output.
        """
        if processed_output is None:
            raise ValueError("Output is None")
        if isinstance(processed_output, str) and not processed_output.strip():
            raise ValueError("Output is empty string")
        return True

    def wrap_and_validate_output(self, raw_vlm_output):
        """
        Full output pipeline: preprocess → validate → return.
        """
        try:
            processed_output = self.preprocess_vlm_output(raw_vlm_output)
            self.validate_output(processed_output)
            self.processed_count += 1
            return processed_output
        except Exception as e:
            raise ValueError(f"Validation failed: {str(e)}")

    def write_local_json(self, obj: dict, logger: MainLogger):
        """
        Optional per-image sidecar output. No-op here — the shared JSONL is the
        deliverable.

        Signature note: VIE calls this as write_local_json(obj, logger), where
        obj is the full output record ({"path", "prompt_str", "vlm_output",
        "full_vlm_object"}). The previous three-argument signature here raised
        TypeError on every image, into a Future nobody read — invisible for the
        whole run. cli_utils now logs such failures (see
        _guarded_write_local_json).
        """
        pass
