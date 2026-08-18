"""
HTML -> PNG worker functions (v1.9 §7).

The three functions below are the entire process-side of a function backend.
They MUST be module-level: `WorkerSpec` is pickled into a spawned child, and a
lambda, a closure or a bound method is not picklable.

Rendering itself is `render_engine.py`, ported verbatim from
Digital-Twin-Pipeline's SMART render path (CDP viewport + TEXT AUTO-FIT).
Requires selenium plus a chrome/chromedriver build — see `main.py` for where
those paths come from — not playwright.

Structurally this mirrors `examples/vllm_testing/main.py`'s init/term pair — same
queues, same lifecycle. As of v1.9.2 it is not a mirror but the same code: one
`WorkerPool` runs this chrome spec and a vLLM spec alike, so the seam v1.9 claimed
is real rather than asserted.

Note what is NOT here. The worker never chooses a filename, never picks a
folder, never scans a directory, and never decides whether its output is good.
It is given the exact path to open and the exact path to write into (both
decided in the main process, see core/shards.py), and it returns a dict. That
is what keeps shard accounting under a single authority and keeps the queue
carrying paths instead of pixels.
"""

import os
from pathlib import Path

import render_engine


def model_init(args, rank, device):
    """
    One persistent headless Chrome (via CDP/selenium) per worker process,
    reused across every call this process ever handles — render_engine's own
    render_html_file() builds a fresh driver per call instead, which is fine
    for a smoke test but would pay Chrome's ~1s startup cost on every image
    here.

    `args` is a SimpleNamespace built from `WorkerSpec.args` (JSON-able, so it
    survives the spawn). `device` is None here — rendering is CPU work. It is
    the same signature a SAM3 worker uses to put its replica on `cuda:{rank}`,
    which is exactly why there is no branching in the framework.
    """
    import tempfile

    driver = render_engine.new_driver(
        getattr(args, "chromedriver", None), getattr(args, "chrome_binary", None)
    )
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"render_worker_{rank}_"))
    return {
        "driver": driver,
        "tmp_dir": tmp_dir,
        "dom_timeout_ms": getattr(args, "dom_timeout_ms", 5000),
        "rank": rank,
    }


def model_call(ctx, payload, params=None):
    """
    Render one HTML file to one PNG via render_engine's SMART-scaling path
    (CDP viewport + TEXT AUTO-FIT), fsync, and return render diagnostics.

    Reads `payload["input_path"]`, writes `payload["output_path"]` at
    `payload["width"] x payload["height"]` (auto-grown if content overflows —
    see render_engine.JS_FIT_TEXT).

    `params` is the request's inference parameters, which only a model has any use
    for — always None here. It is on the signature because v1.9.2 made vLLM an
    ordinary `WorkerSpec`, and a vLLM `call_fn` needs its `SamplingParams`; giving
    every backend the same three arguments is what keeps there being one worker
    loop rather than one per backend.

    The fsync is not optional and not a nicety: it is the single rule that
    keeps the existing commit protocol intact with zero new recovery code. By
    the time `JSONLWriter._flush_group` commits this path, its image bytes are
    already durable — whether or not its shard has sealed yet.

    Raising `InputRejected` means "this HTML is unrenderable and always will be" —
    terminal on the first occurrence. Raising `BackendGone` means "this renderer is
    dead", which stops this worker and lets the pool replace it. Anything else
    raised is treated as infrastructure (a crashed renderer, a full disk): retried
    on the infra budget, never marking a path FAILED.
    """
    from vision_ingest.core.task import InputRejected

    input_path = payload["input_path"]
    output_path = payload["output_path"]

    try:
        size = os.path.getsize(input_path)
    except OSError as e:
        raise InputRejected(f"cannot stat {input_path}: {e}")
    if size == 0:
        raise InputRejected(f"empty HTML file: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        html_original = f.read()

    result = render_engine.render_html(
        html_original=html_original,
        out_png=Path(output_path),
        width=payload["width"],
        height=payload["height"],
        driver=ctx["driver"],
        tmp_dir=ctx["tmp_dir"],
        dom_stable_timeout_ms=ctx["dom_timeout_ms"],
    )

    fd = os.open(output_path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

    return {
        "bytes_written": os.path.getsize(output_path),
        "w": result["render_width"],
        "h": result["render_height"],
        "text_fit": result["text_fit"],
    }


def model_term(ctx):
    """Close this worker's browser and scratch dir. Best-effort — the process
    is exiting either way, and a teardown failure must not mask the reason it
    is exiting."""
    try:
        ctx["driver"].quit()
    finally:
        import shutil
        shutil.rmtree(ctx["tmp_dir"], ignore_errors=True)
