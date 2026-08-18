"""
Chandra OCR (`datalab-to/chandra-ocr-2`) via VIE, on the v1.7 entry-point shape.

1. First make sure you start a db server on some node
```bash
rm -rf /opt/dlami/nvme/srihari.bandarupalli/db_test
```

```bash
python -m vision_ingest.drivers.db_driver_postgres serve \
  --data-dir /opt/dlami/nvme/srihari.bandarupalli/db_test \
  --pg-host 10.20.213.80 \
  --pg-port 54326 \
  --pg-dbname phase_3 \
  --pg-user pipeline
```

 --> Get the IP address of the host in which you are running the serve command
 --> You will run the serve command on host node only once.
 --> Please keep the data-dir a local nvme path only, otherwise it will be very slow.

2. cd into `examples/chandra_testing` and run this `python main_test.py` in all the
   nodes you want to run(including host node)
    - make sure to use the same details (including host) you used in serve command in all the nodes.
3. Try to keep the logs_path in `/opt/dlami/nvme/<some dir>`
    - make sure you have write permission in that dir
    - please dont keep fsxvision path as logs path because this will slow down file system for both you and others as it involves os.fsync()
4.  Jsonl path can be on fsxvision as we write to it only at fixed checkpoints.
"""
# Standard imports
import os, time, socket
import gc
import torch
import multiprocessing as mp
from types import SimpleNamespace

# Dependency imports
from vision_ingest.vllm_module import VLLMConfig, vllm_spec
from vision_ingest.core.channel import build_channel
from vision_ingest.core.service import WorkerPool
from vision_ingest.drivers.db_driver_postgres import ingest_images
from vision_ingest.drivers.cli import cli

from chandra import PromptAndValidation, CHANDRA_SCRATCH_DIR

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def build_args():
    return SimpleNamespace(
        # DB details -> same details as what you used to serve.
        pg_host = "10.20.213.80",
        pg_port = "54326",
        pg_dbname = "phase_3",
        pg_user = "pipeline",
        pg_password = None,

        #logs path -> preferably nvme so that it does not slowdown the file system.
        logs_path = "/opt/dlami/nvme/srihari.bandarupalli/chandra_testing_logs/v1_7_online",
        jsonl_output_path = os.path.join(_THIS_DIR, "outputs_v1_7_online"),

        # Insert into the database.
        image_paths_source ="/fsxvision_new/srihari.bandarupalli/Patram-Data-Engine/example_data/images_regular.txt",
        ingest_batch_size= 100000, # batch size for insert only.. It can be high.
        use_copy = False, # can be True if you want faster insert but only if you are sure if image_paths_source has no duplicates with the paths already inserted into db.
        run_ingest = True,

        # Processing configuration
        batch_size = 128,
        local_json_threads = 32,
        fsync_every_lines = 250,
        fetch_state = 0,

        #VLM Configuration
        # Which WorkerSpec vllm_spec() builds. As of v1.8 it no longer selects a
        # prompt-payload shape — every backend receives the same payload (raw
        # text + image paths) and renders it itself — so there is nothing left
        # for a second call site to disagree with. As of v1.9.2 this string is
        # interpreted exactly once, in vllm_spec(), and reaches no worker at all.
        #   "online" -> local `vllm serve` over HTTP
        #   "async"  -> in-process AsyncLLM continuous batching
        # ("batch", the batch-boundary LLM.generate baseline, was retired in
        #  v1.9.2 — it cannot be a call-per-request worker.)
        backend = "online",

        # Directories images may be read from, on every backend (v1.8): the
        # roots each image path is validated against before submission, and the
        # source of vLLM's own allowed_local_media_path.
        # BOTH are genuinely needed here: chandra.py only rescales images below
        # its short-edge threshold, so it sends rescaled copies from the scratch
        # dir AND originals straight from /fsxvision_new.
        # vLLM's allowed_local_media_path is a single directory, so it gets the
        # common ancestor of these two and the exact list is enforced locally —
        # cli() logs that widening at startup. See VLLMConfig.engine_media_root.
        # (Before v1.8 the second --allowed-local-media-path flag simply
        # overwrote the first, so every original under /fsxvision_new was
        # rejected by the server.)
        allowed_media_roots = ["/fsxvision_new", CHANDRA_SCRATCH_DIR],

        vlm_model_name="datalab-to/chandra-ocr-2",
        vlm_gpus="0,1,2,3,4,5,6,7",
        vlm_config_path=os.path.join(_THIS_DIR, "vllm_model.yaml"),
        prompt_path=None # Chandra prompt is hardcoded in chandra.py by default
    )

# ---------------------------------------------------------------------------
# GPU process lifecycle
# ---------------------------------------------------------------------------

def model_init(args, logger, channel):
    """
    Start the backend: one `WorkerPool` running a vLLM `WorkerSpec`.

    Same `(args, logger, channel) -> init_vars` shape StageWeaver locks for
    `init_fn` (v1.9.1 §4.5). The channel must be created BEFORE calling this —
    its queues and stop_event must exist before any process is spawned so file
    descriptors are inherited correctly at spawn time.

    Nothing here builds or publishes a health object any more: `pool.start()`
    puts its own `ServiceStatus` on the channel before spawning any worker,
    since the pool is the only thing that writes it. The ordering rule — publish
    before the spawn that reads it — is unchanged, just handled by the pool.

    Args:
        args:           Parsed args (needs vlm_model_name, vlm_config_path,
                        vlm_gpus, batch_size, backend, allowed_media_roots,
                        logs_path — expected to already be the run's final,
                        timestamped log directory; see __main__).
        logger:         Logger for init errors (can be None).
        channel:        ServiceChannel carrying request_queue /
                        response_queues / stop_event / extras. Must exist
                        before any process is spawned (fd inheritance).

    Returns:
        pool (WorkerPool): Started pool. Caller passes it to model_term().
    """
    try:
        # VLLMConfig is only needed here to get engine_args for the spec.
        # cli() will create its own VLLMConfig instance for prompt preparation.
        # Same media roots on both: they become the engine's
        # allowed_local_media_path AND what cli() validates paths against.
        cfg_model = VLLMConfig(args.vlm_model_name, args.vlm_config_path,
                               allowed_media_roots=args.allowed_media_roots)

        pool = WorkerPool(
            vllm_spec(
                args.backend,
                engine_args  = cfg_model.engine_args,
                gpus         = args.vlm_gpus,
                # How many requests the one server holds at once. batch_size is
                # the natural value: it is what prediction.py's admission control
                # keeps in flight for this stage.
                max_inflight = args.batch_size,
                logs_dir     = args.logs_path,
            ),
            channel,
            logs_dir = args.logs_path,
        )
        pool.start()
        return pool

    except Exception as e:
        msg = f"Failed to start the vLLM WorkerPool: {e}"
        if logger is None:
            print(msg)
        else:
            logger.error(msg)
        raise


def model_term(pool, channel):
    """
    Cleanly shut down the backend.

    `pool.stop()` sets the stop event, joins every worker — each running its own
    term_fn in a `finally`, which is what terminates the `vllm serve` subprocess —
    and then joins the pool's own threads. It does not force-kill unless a worker
    overruns the join timeout.

    GPU memory cleanup runs AFTER stop() because the GPU memory is only
    actually released once the child processes are dead.

    Args:
        pool:    WorkerPool returned by model_init().
        channel: The same ServiceChannel passed to model_init().
    """
    pool.stop()

    # GPU cleanup — runs in the parent after the children have fully exited
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


    # Opt out of the exit-time flush-and-join for every queue this process
    # ever put() to: a queue holding unread items leaves this process's own
    # feeder thread blocked on a full pipe with no reader, hanging exit.
    # See docs/archive/exit_hang_postmortem.md.
    channel.cancel_join_threads()

    print("----- Backend also shut down cleanly. Model Term Complete. You can exit now. -----")


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def main_cli(args, channel, stage_name, run_ingest=False):
    """
    Handle optional ingestion, then run the pipeline via cli().

    args.logs_path is already the run's final, timestamped log directory by
    the time this is called — stamped once in __main__, before model_init()
    spawns any worker, so the pool's worker_pool.log / worker_0.log land in
    the same directory as cli()'s main.log/vlm_prediction.log/etc. cli() is
    told to use it as-is via run_specific_log_path_in_args=True.

    """
    if run_ingest:
        db_path_or_config = {
            'host':   args.pg_host,
            'port':   args.pg_port,
            'dbname': args.pg_dbname,
            'user':   args.pg_user,
            **({"password": args.pg_password} if args.pg_password else {}),
        }
        ingest_images(
            db_path_or_config,
            args.image_paths_source,
            args.logs_path,
            args.ingest_batch_size,
            run_specific_log_path_in_args=True,
        )
        print("----- Ingestion Done -----")
    else:
        print("----- Ingestion Skipped -----")

    prompt_validation_object = PromptAndValidation(args.prompt_path)

    # stop_event doubles as shutdown_event in standalone mode:
    #   - signals the worker loops in the WorkerPool (mp.Event behaviour)
    #   - signals the pipeline loop in cli() (threading.Event behaviour)
    # mp.Event is a superset of threading.Event — .is_set() / .set() work identically.
    #
    # cli() picks its own response queue and the ServiceStatus off the channel —
    # pool.start() published the latter before this ran.
    cli(
        args,
        prompt_validation_object,
        channel,
        router_coordinator = None,
        stage_name      = stage_name,
        shutdown_event  = channel.stop_event,        # same object — see comment above
        run_specific_log_path_in_args = True,
    )

    # Print the final OCR summary
    prompt_validation_object.print_summary()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    mp.set_start_method('spawn', force=True)

    args = build_args()
    stage_name = "standalone"

    # Stamped once, here, before model_init() spawns anything — so the pool's
    # own loggers (worker_pool.log, worker_<rank>.log) and cli()'s loggers
    # (main.log, vlm_prediction.log, ...) write into the same directory.
    # main_cli()/cli() are told this is already final via
    # run_specific_log_path_in_args=True and do not restamp it.
    run_id    = time.strftime("%Y%m%d-%H%M%S")
    node_name = socket.gethostname()
    args.logs_path = os.path.join(args.logs_path, node_name, run_id)

    # The channel MUST be created here — before any process is spawned. This
    # is the only window where fd inheritance works with the 'spawn' start
    # method. Standalone's __main__ plays the part StageWeaver's group leader
    # plays; the shape is identical either way (v1.9.1 §4.5).
    channel = build_channel([stage_name])

    # pool.start() publishes the ServiceStatus on the channel before spawning
    # any worker, so cli() — which runs after this returns — always finds it.
    # Without it a crashed backend is indistinguishable from bad data and the
    # whole DB walks into state=3.
    pool = model_init(args, logger=None, channel=channel)

    try:
        main_cli(args, channel, stage_name, args.run_ingest)
    finally:
        # model_term runs whether main_cli completes normally or raises.
        # stop_event may already be set if cli() hit a GracefulShutdown —
        # setting it again is a no-op, so this is always safe.
        model_term(pool, channel)
