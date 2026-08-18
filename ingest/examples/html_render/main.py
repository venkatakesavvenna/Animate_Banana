"""
HTML -> PNG at 32 workers (v1.9 §7).

Deliberately the same shape as `examples/vllm_testing/main.py`: the same queues
created in the same place for the same reason, the same init/term lifecycle, the
same `main_cli()` -> `cli()` handoff. As of v1.9.2 it is not merely the same shape
but the same class — one `WorkerPool` runs a chrome spec here and a vLLM spec
there, and nothing else moves.

Run it the same way:

    # once, on the host node
    python -m vision_ingest.drivers.db_driver_postgres serve \
      --data-dir /opt/dlami/nvme/<user>/db \
      --pg-host <HOST_IP> --pg-port 5432 --pg-dbname html_render --pg-user pipeline

    # on every node
    cd examples/html_render && python main.py

Requires `pip install selenium` plus a chrome/chromedriver build — CHROMEDRIVER
/ CHROME_BINARY below — and rendering is CPU-bound (a real headless Chrome
process per worker), not a GPU one; see worker.py / render_engine.py, the
latter a verbatim port of Digital-Twin-Pipeline's SMART render path (CDP
viewport + JS TEXT AUTO-FIT).
"""

import os, time, socket
import multiprocessing as mp
from types import SimpleNamespace

# Artifacts (file list, JSONL, image shards) live next to this script. Logs and
# the Postgres data dir deliberately do NOT — they stay on local NVMe, since
# os.fsync() on FSx is slow and slows it for everyone (see logs_path below).
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

from vision_ingest.core.channel import build_channel
from vision_ingest.core.service import WorkerPool, WorkerSpec
from vision_ingest.drivers.db_driver_postgres import ingest_images
from vision_ingest.drivers.cli import cli

# Module-level import, not a local one: these three go into a WorkerSpec that is
# pickled into each spawned child, so they must be importable by name there.
from worker import model_init, model_call, model_term
from task import HtmlRenderTask


# One headless Chrome per process. CPU-bound and I/O-bound, so this can go far
# above the core count before it stops paying; 32 is a reasonable starting point
# on a large node. Contrast SAM3, which would use devices=["cuda:0", ...] and get
# one replica per GPU from the identical machinery.
N_WORKERS = 32

# Point at a chrome/chromedriver build. No torch/vllm needed for this stage —
# pick whichever of your ENVIRONMENTS/* folders has `selenium` installed (see
# CHROMEDRIVER_PATH / CHROME_BINARY_PATH env vars, or hardcode below). Reuses
# the same Chrome for Testing build digital_twin_stage_weaver already has.
CHROMEDRIVER = os.environ.get(
    "CHROMEDRIVER_PATH",
    "/fsxvision_new/aryanjain.intern/Patram-Data-Engine/digital_twin_stage_weaver/"
    "chrome_cft/chromedriver-linux64/chromedriver",
)
CHROME_BINARY = os.environ.get(
    "CHROME_BINARY_PATH",
    "/fsxvision_new/aryanjain.intern/Patram-Data-Engine/digital_twin_stage_weaver/"
    "chrome_cft/chrome-linux64/chrome",
)
# Initial render canvas — render_engine's TEXT AUTO-FIT pass grows this
# automatically when content overflows it, so this only needs to be a
# reasonable starting size for your inputs, not an exact fit.
RENDER_WIDTH = 1280
RENDER_HEIGHT = 1024


def build_args():
    return SimpleNamespace(
        # DB details -> same details as what you used to serve.
        pg_host   = "10.20.213.80",
        pg_port   = "59324",
        pg_dbname = "html_render",
        pg_user   = "pipeline",
        pg_password = None,

        # logs on local nvme — os.fsync() on FSx is slow and slows it for everyone.
        logs_path = "/opt/dlami/nvme/srihari.bandarupalli/html_render_logs/",
        # The JSONL is ALWAYS the output, for every backend and every job.
        jsonl_output_path = os.path.join(ROOT_DIR, "jsonl_outputs"),
        # Images are ADDITIONAL, and only images go into folders-then-tars.
        # Omit this and the run still works — it just produces no shards.
        # Lands two trees under here:
        #   <this>/<host>/image_folders/shard_00007/a.png   (kept, see below)
        #   <this>/<host>/shards/shard_00007.tar + .csv
        image_output_path = os.path.join(ROOT_DIR, "image_output"),
        # Roll to a new shard folder once the current one holds this many bytes.
        # Same trigger JSONL rotation uses. Lower it to ~50 MB to watch rotation
        # happen in seconds during a smoke test.
        max_shard_bytes = 10 * 1024 * 1024 * 1024,
        # Keep the loose per-shard folders after their tar is sealed (the
        # default). Every image then exists twice — ~2x disk — in exchange for
        # files you can read without untarring and a tar you can rebuild. Flip
        # to True to reclaim the space; doing so later also cleans up folders
        # retained by earlier runs, at the next startup.
        delete_shard_folders = False,

        image_paths_source = os.path.join(ROOT_DIR, "html_files.txt"),
        ingest_batch_size = 100000,
        use_copy = False,
        run_ingest = True,

        batch_size = 32,
        local_json_threads = 16,
        fsync_every_lines = 250,
        fetch_state = 0,

        # THE discriminator. cli() builds a FunctionTask path instead of a
        # VLLMConfig, and VLMPrediction treats the payload as opaque.
        backend = "function",
    )


# ---------------------------------------------------------------------------
# Worker pool lifecycle — identical to the vLLM one, which is the point
# ---------------------------------------------------------------------------

def pool_init(args, logger, channel):
    """
    Same `(args, logger, channel) -> init_vars` shape StageWeaver locks for
    `init_fn` (v1.9.1 §4.5).

    As of v1.9.2 this is byte-for-byte the shape a vLLM group uses too (compare
    examples/vllm_testing/main.py): build a spec, hand it to a `WorkerPool`, start
    it. A headless-chrome renderer and a `vllm serve` process differ only in what
    their `init_fn` builds and whether their `call_fn` is `async def`.
    """
    spec = WorkerSpec(
        init_fn   = model_init,
        call_fn   = model_call,
        term_fn   = model_term,
        args      = {"chromedriver": CHROMEDRIVER, "chrome_binary": CHROME_BINARY,
                     "dom_timeout_ms": 30000},
        devices   = None,        # CPU work — no per-process device to pin
        n_workers = N_WORKERS,
        # This call_fn is synchronous, so the worker answers exactly one request at
        # a time and the pool never assigns it more. A backend that wanted real
        # concurrency would make call_fn `async def` and declare its true ceiling
        # here. See core/service.py.
        max_inflight_per_worker = 1,
        # 32 cheap, independent renderers: one dying costs throughput, not the run.
        # The pool re-emits its in-flight request as `infra` and respawns it. Only
        # when every rank is gone does the run abort — with rows reset, nothing
        # marked FAILED. (A vLLM pool defaults to "abort" instead, because a dead
        # GPU worker usually stays dead and a silent relaunch burns GPU minutes.)
        on_worker_death = "respawn",
    )
    pool = WorkerPool(spec, channel, logs_dir=args.logs_path)
    pool.start()
    return pool


def pool_term(pool, channel):
    """Signal the workers, wait for them to finish the item in hand, then opt out
    of the exit-time queue joins. No GPU cleanup — nothing here ever touched one."""
    pool.stop(timeout=60)

    channel.cancel_join_threads()
    print("----- Worker pool shut down cleanly. You can exit now. -----")


def main_cli(args, channel, stage_name, run_ingest=False):
    if run_ingest:
        db_path_or_config = {
            'host':   args.pg_host,
            'port':   args.pg_port,
            'dbname': args.pg_dbname,
            'user':   args.pg_user,
            **({"password": args.pg_password} if args.pg_password else {}),
        }
        ingest_images(db_path_or_config, args.image_paths_source, args.logs_path,
                      args.ingest_batch_size, run_specific_log_path_in_args=True)
        print("----- Ingestion Done -----")
    else:
        print("----- Ingestion Skipped -----")

    # For a function backend the task object IS the project-specific object —
    # it exposes the same hook names (wrap_and_validate_output,
    # build_retry_request, write_local_json) that PromptAndValidation does,
    # which is why prediction.py needs no second code path for them.
    task = HtmlRenderTask(width=RENDER_WIDTH, height=RENDER_HEIGHT)

    cli(
        args,
        task,
        channel,
        router_coordinator = None,
        stage_name      = stage_name,
        shutdown_event  = channel.stop_event,
        run_specific_log_path_in_args = True,
    )


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    args = build_args()
    stage_name = "standalone"

    run_id    = time.strftime("%Y%m%d-%H%M%S")
    node_name = socket.gethostname()
    args.logs_path = os.path.join(args.logs_path, node_name, run_id)

    # The channel MUST be created here, before any process is spawned — the
    # only window where fd inheritance works with the 'spawn' start method.
    channel = build_channel([stage_name])

    pool = pool_init(args, logger=None, channel=channel)
    try:
        main_cli(args, channel, stage_name, args.run_ingest)
    finally:
        pool_term(pool, channel)
