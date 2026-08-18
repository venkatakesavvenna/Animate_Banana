"""
The main-process half of the HTML->PNG job (v1.9 §7).

`HtmlRenderTask` is to `backend="function"` what `PromptAndValidation` is to a
vLLM backend. It builds the request, and it decides whether what came back is
good enough to keep.
"""

from vision_ingest.core.task import FunctionTask


class HtmlRenderTask(FunctionTask):

    # A render smaller than this is a blank/near-blank page — chromium happily
    # produces a valid PNG of nothing when a document fails to load. Retrying is
    # worthwhile: the usual cause is a resource that timed out, not a document
    # that can never render.
    MIN_PNG_BYTES = 1024

    def __init__(self, min_png_bytes: int = MIN_PNG_BYTES, width: int = 1280, height: int = 1024):
        super().__init__()
        self.min_png_bytes = min_png_bytes
        # Initial render canvas. render_engine's TEXT AUTO-FIT pass (ported
        # from DTP) grows this automatically when content overflows it — see
        # render_engine.JS_FIT_TEXT — so these only need to be a reasonable
        # starting size, not an exact fit for every input.
        self.width = width
        self.height = height

    def build_request(self, path, logger):
        """
        One reservation per path, and the payload that tells the worker exactly
        which two files to touch.

        `reserve()` is what opts this task into shard allocation — nothing else
        in the pipeline branches on it. The returned record names the FINAL TAR
        (`shard_00007.tar`) rather than the folder the worker is about to write
        into, which is why the JSONL line stays correct after the shard seals
        and why sealing never has to rewrite an immutable JSONL shard.
        """
        member = self.member_name(path, ".png")
        output_path, record = self.reserve(path, member)
        return {
            "input_path": path,
            "output_path": output_path,
            "shard": record["shard"],
            "member": record["member"],
            "width": self.width,
            "height": self.height,
        }

    def wrap_and_validate_output(self, output):
        """
        Reject a zero-byte or blank render, which then RETRIES — exactly like a
        bad VLM answer does, on the same `max_attempts` budget, through the same
        code path. Raising ValueError is the whole interface.
        """
        if not isinstance(output, dict):
            raise ValueError(f"worker returned {type(output).__name__}, expected a dict")
        nbytes = output.get("bytes_written") or 0
        if nbytes < self.min_png_bytes:
            raise ValueError(
                f"render produced {nbytes} bytes (< {self.min_png_bytes}) — "
                "almost certainly a blank page"
            )
        return output
