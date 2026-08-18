"""
!!! ARCHIVED — DO NOT COPY THIS AS A TEMPLATE !!!

This file targets the pre-v1.9.1 locked signatures: flat positional
`init_fn(args, logger, request_queue, response_queues, stop_event, shm_name,
free_slots)` and an 8-argument stage `function`, plus the `Shm` shared-memory
pool. Patram-Ingest v1.9.1 / StageWeaver v1.1.0 replaced all of that with a
single `ServiceChannel` and deleted the shm subsystem entirely, so nothing here
will import, let alone run.

Kept only as a record of the older shape. For a current reference see:
  - Patram-Ingest `examples/vllm_testing/main.py`  (standalone, vLLM)
  - Patram-Ingest `examples/html_render/main.py`   (standalone, FunctionService)
  - `src/digital_twin/project_specific/*.py`       (StageWeaver DAG)
  - `src/captioning_dataset_1/stage_fns.py`        (stages sharing one service)
"""
"""
1. First make sure you start a db server on some node
```bash
rm -rf /opt/dlami/nvme/srihari.bandarupalli/db_test
```

```bash
python -m vision_ingest.drivers.db_driver_postgres serve \
  --data-dir /opt/dlami/nvme/srihari.bandarupalli/db_test \
  --pg-host 10.20.213.80 \
  --pg-port 59324 \
  --pg-dbname vllm_testing \
  --pg-user pipeline
```
 --> Get the IP address of the host in which you are running the serve command
 --> You will run the serve command on host node only once.
 --> Please keep the data-dir a local nvme path only, otherwise it will be very slow.

2. cd into `examples/vllm_testing` and run this `python main.py` in all the nodes you want to run(including host node)
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
from vision_ingest.vllm_module import VLLMConfig, VLLMService
from vision_ingest.shm import Shm
from vision_ingest.drivers.db_driver_postgres import ingest_images
from vision_ingest.drivers.cli import cli


# Project-specific — assumed to live alongside main.py
from project_specific import PromptAndValidation


def build_args():
    return SimpleNamespace(
        # DB details -> same details as what you used to serve.
        pg_host = "10.20.213.80",
        pg_port = "59324",
        pg_dbname = "vllm_testing",
        pg_user = "pipeline",
        pg_password = None,

        #logs path -> preferably nvme so that it does not slowdown the file system.
        logs_path = "/opt/dlami/nvme/srihari.bandarupalli/raghu_layout_testing_logs/",
        jsonl_output_path = "/code/dependencies/Vision-Ingestion-Engine/examples/archive/raghu_layout/jsonl_outputs",

        # Insert into the database.
        image_paths_source = "/code/dependencies/Vision-Ingestion-Engine/examples/archive/raghu_layout/images_small.txt",
        ingest_batch_size= 100000, # batch size for insert only.. It can be high.
        use_copy = False, # can be True if you want faster insert but only if you are sure if image_paths_source has no duplicates with the paths already inserted into db.
        run_ingest = True, 
        
        # Processing configuration
        batch_size = 32,
        local_json_threads = 32,
        fsync_every_lines = 32,
        fetch_state = 0,

        #VLM Configuration
        vlm_model_name="Qwen/Qwen3.6-27B",
        vlm_gpus="0,1,2,3,4,5,6,7",
        vlm_config_path="/code/dependencies/Vision-Ingestion-Engine/examples/archive/raghu_layout/vllm_model.yaml",

        prompt_path="/code/dependencies/Vision-Ingestion-Engine/examples/archive/raghu_layout/layout_vlm_stage_1.md",

    )

# ---------------------------------------------------------------------------
# GPU process lifecycle
# ---------------------------------------------------------------------------

def model_init(args, logger, request_queue, response_queues, stop_event, shm_name, free_slots):
    """
    Start the VLLMService process.

    Queues, stop_event and the shm pool must be created BEFORE calling this
    function — they must exist before any process is spawned so that file
    descriptors are inherited correctly at spawn time.

    Args:
        args:           Parsed args (needs vlm_model_name, vlm_config_path,
                        vlm_gpus, batch_size).
        logger:         Logger for init errors (can be None).
        request_queue:  Shared mp.Queue — all stages put requests here.
        response_queues: Dict mapping stage names to their response queues.
        stop_event:     mp.Event — set to stop VLLMService._worker.
        shm_name:       Name of the shared-memory pool VLLMService reconnects to.
        free_slots:     mp.Queue of free shm slot indices, shared with the pool.

    Returns:
        vllm_proc (VLLMService): Started process. Caller passes to model_term().
    """
    try:
        # VLLMConfig is only needed here to get engine_args for VLLMService.
        # cli() will create its own VLLMConfig instance for prompt preparation.
        cfg_model = VLLMConfig(args.vlm_model_name, args.vlm_config_path)

        vllm_proc = VLLMService(
            engine_args     = cfg_model.engine_args,
            gpus            = args.vlm_gpus,
            batch_size      = args.batch_size,
            request_queue   = request_queue,
            response_queues = response_queues,
            stop_event      = stop_event,
            shm_name        = shm_name,
            free_slots      = free_slots,
            lazy_loading    = True,
        )
        vllm_proc.start()
        return vllm_proc

    except Exception as e:
        msg = f"Failed to start VLLMService: {e}"
        if logger is None:
            print(msg)
        else:
            logger.error(msg)
        raise


def model_term(vllm_proc, stop_event, shm, request_queue, free_slots, response_queues):
    """
    Cleanly shut down the VLLMService process.

    stop_event signals _worker to exit its loop. Since _worker checks
    stop_event.is_set() at the top of every iteration, the process dies
    naturally once the current batch (if any) finishes. join() waits for
    that clean exit — it does not force-kill.

    GPU memory cleanup runs AFTER join() because the GPU memory is only
    actually released once the child process is dead. shm.cleanup() also
    runs after join() — nothing must still have the block mapped when it
    unlinks the name.

    Args:
        vllm_proc:       VLLMService instance returned by model_init().
        stop_event:      The same mp.Event passed to model_init().
        shm:             The Shm pool created before spawning vllm_proc.
        request_queue:   Shared mp.Queue this process calls .put() on via submit().
        free_slots:      mp.Queue this process calls .get()/.put() on via shm.acquire/release.
        response_queues: Dict of this stage's response mp.Queue(s).
    """
    stop_event.set()
    vllm_proc.join()

    # GPU cleanup — runs in the parent after the child has fully exited
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    print("----- Cleaning up the SHM -----")
    shm.cleanup()

    # Opt out of the exit-time flush-and-join for every queue this process
    # ever put() to. request_queue/free_slots can have unread items if
    # VLLMService died first, or if a stray submit() lands after this
    # process stops reading — either way, this process's own feeder thread
    # would otherwise block interpreter exit on a full pipe with no reader.
    # See src/vision_ingest/shm/exit_hang_postmortem.md.
    request_queue.cancel_join_thread()
    free_slots.cancel_join_thread()
    for q in response_queues.values():
        q.cancel_join_thread()

    print("----- VLLMService also shut down cleanly. Model Term Complete. You can exit now. -----")


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def main_cli(args, request_queue, response_queues, shm_name, free_slots, stop_event, stage_name, run_ingest=False):
    """
    Handle optional ingestion, then run the pipeline via cli().

    Log directory is stamped here (same as before) and written back into
    args so cli() can use run_specific_log_path_in_args=True.

    shm_name/free_slots (not a Shm object) are passed down — same pattern
    as model_init()/VLLMService — so VLMPrediction reconnects to the pool
    itself instead of receiving a live Shm instance.
    """
    run_id    = time.strftime("%Y%m%d-%H%M%S")
    node_name = socket.gethostname()
    node_run_specific_log_dir = os.path.join(args.logs_path, node_name, run_id)
    args.logs_path = node_run_specific_log_dir

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
    #   - signals _worker loop in VLLMService (mp.Event behaviour)
    #   - signals the pipeline loop in cli() (threading.Event behaviour)
    # mp.Event is a superset of threading.Event — .is_set() / .set() work identically.
    cli(
        args,
        prompt_validation_object,
        request_queue   = request_queue,
        response_queue  = response_queues[stage_name],
        shm_name        = shm_name,
        free_slots      = free_slots,
        router_coordinator = None,
        stage_name      = stage_name,
        shutdown_event  = stop_event,        # same object — see comment above
        run_specific_log_path_in_args = True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    mp.set_start_method('spawn', force=True)

    args = build_args()
    stage_name = "standalone"

    # Queues, stop_event and the shm pool MUST be created here — before any
    # process is spawned. This is the only window where fd inheritance works
    # correctly with the 'spawn' start method.
    request_queue  = mp.Queue()
    response_queue = mp.Queue()
    response_queues = {stage_name: response_queue}
    stop_event     = mp.Event()

    shm_name   = f"patram_shm_{stage_name}"
    free_slots = mp.Queue()
    shm        = Shm(name=shm_name, create=True, free_slots=free_slots)
    print(f"Shm pool '{shm_name}': {shm.n_slots} slots x {shm.slot_size // (1024 * 1024)}MB")

    # Phase A2 slot budget: simultaneously-held slots ≈ staged pools (≤ 2·B,
    # pre-staged requests awaiting submission) + in-flight responses not yet
    # collected (≤ B) ≈ 3·B + margin. acquire() is the backpressure, so
    # undersizing is not a deadlock — but it silently serializes the stage-side
    # pipeline back into lock-step, so keep n_slots comfortably above 3·B.
    min_slots = 3 * args.batch_size
    if shm.n_slots < min_slots:
        print(f"****WARNING**** Shm pool has {shm.n_slots} slots but Phase A2 wants "
              f">= 3*batch_size = {min_slots}; the stage-side pipeline may serialize.")

    vllm_proc = model_init(args, logger=None,
                           request_queue=request_queue,
                           response_queues=response_queues,
                           stop_event=stop_event,
                           shm_name=shm_name,
                           free_slots=free_slots)

    try:
        main_cli(args, request_queue, response_queues, shm_name, free_slots, stop_event, stage_name, args.run_ingest)
    finally:
        # model_term runs whether main_cli completes normally or raises.
        # stop_event may already be set if cli() hit a GracefulShutdown —
        # setting it again is a no-op, so this is always safe.
        model_term(vllm_proc, stop_event, shm, request_queue, free_slots, response_queues)