from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import signal
import threading
import time, os, socket

from vision_ingest.db.db import DB
from vision_ingest.modules.writer import JSONLWriter
from vision_ingest.modules.recovery import recover_last_shard
from vision_ingest.modules.prediction import VLMPrediction

from vision_ingest.utils import PipelineMetrics, MainLogger, GracefulShutdown
from vision_ingest.vllm_module import VLLMConfig
from vision_ingest.core.status import VLMServiceDown
from vision_ingest.vllm_module.media import DEFAULT_MAX_IMAGE_PIXELS
from vision_ingest.core.channel import ServiceChannel
from vision_ingest.core.task import FunctionTask
from vision_ingest.core.shards import ShardAllocator, DEFAULT_MAX_SHARD_BYTES

from vision_ingest.project_specific import PromptAndValidation
from vision_ingest.drivers.cli_utils import process_pipeline


@contextmanager
def _shielded_teardown(main_logger):
    """
    Make the shutdown sequence uninterruptible by Ctrl+C / SIGTERM.

    Teardown does the two things that keep the database honest — reset the
    in-flight paths from state=1 back to their fetch state, and flush the writer
    (which marks its own paths done). Both must complete. A second Ctrl+C
    landing in the middle of that used to raise straight through the `finally`,
    leaving rows stranded in state=1 (needing a manual `reset-stuck`) or, worse,
    interrupting the writer between its fsync and its mark_done.

    So we take the signals off the table for the duration and tell the user why
    the extra Ctrl+C did nothing. Only signals installed in the main thread can
    be replaced; when cli() runs off-thread this degrades to a no-op, which is
    fine — signals are delivered to the main thread anyway.
    """
    installed = []

    def _ignore(signum, frame):
        msg = (f"signal {signum} ignored — shutdown is already in progress and "
               "must finish (resetting in-flight paths + flushing the writer). "
               "Interrupting now would leave rows stuck in state=1.")
        try:
            main_logger.logger.warning(msg)
        except Exception:
            pass
        print(f"\n[shutdown] {msg}", flush=True)

    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                installed.append((sig, signal.signal(sig, _ignore)))
            except (ValueError, OSError):
                pass
    try:
        yield
    finally:
        for sig, previous in installed:
            try:
                signal.signal(sig, previous)
            except (ValueError, OSError):
                pass


def cli(
    args,
    prompt_validation_object: PromptAndValidation,
    channel: ServiceChannel,
    router_coordinator = None,
    stage_name: str = "standalone",
    run_specific_log_path_in_args: bool = False,
    shutdown_event: threading.Event = None, # or mp.Event
):
    """
    Pipeline orchestrator for one stage.
      - Owns VLLMConfig initialisation (moved from model_init / VLMPrediction).
      - shutdown_event and channel.stop_event are the same object when called
        from main.py (standalone mode). cli() keeps both names so the
        distinction remains clear: stop_event signals the backend process;
        shutdown_event signals only this current stage loop.

    Args:
        args:                         Parsed CLI arguments.
        prompt_validation_object:     Project-specific prompt/validation logic.
        channel:                      ServiceChannel carrying the shared IPC primitives
                                      (request_queue, response_queues, stop_event) plus
                                      `extras`. Created in the one fd-inheritance-safe
                                      window — the group leader under StageWeaver, or
                                      `__main__` standalone. cli() reads two things off it:
                                      its own response queue (`response_queues[stage_name]`)
                                      and `extras["health"]`, the ServiceStatus that lets
                                      the predictor tell "the backend died" apart from "this
                                      image failed" — so a crashed backend aborts the run
                                      instead of marking every remaining path FAILED. The
                                      pool publishes it itself (WorkerPool.start), so a
                                      group with a backend always has one; a group with no
                                      backend has none and simply does not wait.
        router_coordinator:           RouterCoordinator instance — JSONLWriter calls
                                      mark_done(paths, stage_name) on it after each
                                      batch (replaces legacy next_stage_db signaling).
        stage_name:                   Key in channel.response_queues AND the stage
                                      identifier passed to router_coordinator.mark_done.
                                      Defaults to "standalone".
        run_specific_log_path_in_args: Use args.logs_path as-is (no timestamp dir).
        shutdown_event:               threading.Event or mp.Event for graceful pipeline shutdown.
    """
    # StageWeaver builds its own four-field channel (it depends on nothing and
    # cannot import ours); adopt() wraps it so the conveniences below exist
    # regardless of who created it. A no-op when it is already ours.
    channel        = ServiceChannel.adopt(channel)
    request_queue  = channel.request_queue
    response_queue = channel.response_queue(stage_name)
    service_status = channel.health

    # ------------------------------------------------------------------
    # Database backend selection (SQLite or Postgres — unchanged)
    # ------------------------------------------------------------------
    if hasattr(args, 'pg_host') and args.pg_host:
        db_path_or_config = {
            'host':   args.pg_host,
            'port':   args.pg_port,
            'dbname': args.pg_dbname,
            'user':   args.pg_user,
            **({"password": args.pg_password} if args.pg_password else {}),
        }
    else:
        db_path_or_config = os.path.join(args.db_path, "images.db")
        os.makedirs(args.db_path, exist_ok=True)

    node_name = socket.gethostname()

    # ------------------------------------------------------------------
    # Log directory
    # ------------------------------------------------------------------
    if run_specific_log_path_in_args:
        node_run_specific_log_dir = args.logs_path
    else:
        run_id = time.strftime("%Y%m%d-%H%M%S")
        node_run_specific_log_dir = os.path.join(args.logs_path, node_name, run_id)

    jsonl_output_path = os.path.join(args.jsonl_output_path, node_name)
    json_prefix       = os.path.join(jsonl_output_path, "results_")

    if shutdown_event and shutdown_event.is_set():
        raise GracefulShutdown("shutdown_event set before starting MainLogger")

    # ------------------------------------------------------------------
    # MainLogger
    # ------------------------------------------------------------------
    main_logger = MainLogger(
        node_run_specific_log_dir,
        run_id=os.path.basename(node_run_specific_log_dir),
    )

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    fetch_state = getattr(args, 'fetch_state', 0)
    db = DB(db_path_or_config, main_logger=main_logger, fetch_state=fetch_state)

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------
    try:
        last_shard_path = recover_last_shard(
            db,
            prefix=json_prefix,
            fsync_lines=args.fsync_every_lines,
            log_dir=node_run_specific_log_dir,
        )
    except Exception as e:
        main_logger.log_error("Error during recovery of last shard", e)
        raise

    recovery_info = None
    if last_shard_path:
        recovery_info = {
            "recovery_performed": True,
            "last_shard_path":    last_shard_path,
            "last_shard_index":   int(
                last_shard_path
                .replace(json_prefix, "")
                .replace(".jsonl", "")
            ) if last_shard_path else 0,
        }

    # ------------------------------------------------------------------
    # Writer — downstream signaling is now driven by router_coordinator.
    # ------------------------------------------------------------------
    writer = JSONLWriter(
        db_path_or_config,
        router_coordinator=router_coordinator,
        stage_name=stage_name,
        prefix=json_prefix,
        fsync_every_lines=args.fsync_every_lines,
        last_shard_path=last_shard_path,
        logs_dir=node_run_specific_log_dir,
    )

    if shutdown_event and shutdown_event.is_set():
        raise GracefulShutdown("shutdown_event set before VLMPrediction initialisation")

    # ------------------------------------------------------------------
    # VLMPrediction
    # ------------------------------------------------------------------
    # VLLMConfig no longer takes a backend (v1.8): every backend receives the
    # same payload — raw text + image paths — and the worker turns it into what
    # its engine needs, so there is no longer a per-backend prompt shape to
    # choose. args.backend is still needed for the sampling-param capability
    # check inside VLMPrediction, and to pick the backend's WorkerSpec builder.
    backend = getattr(args, "backend", "async")
    main_logger.logger.info(f"VLM backend: {backend}")

    # v1.9 §2: the one branch a function backend adds to this file. Everything
    # after it — metrics, the health-readiness wait, process_pipeline, the whole
    # `finally` — is shared, because a FunctionTask exposes the same hook names
    # a PromptAndValidation does and VLMPrediction treats the payload as opaque.
    cfg_model       = None
    function_task   = None
    shard_allocator = None

    if backend == "function":
        if not isinstance(prompt_validation_object, FunctionTask):
            raise ValueError(
                "backend='function' requires prompt_validation_object to be a "
                f"core.task.FunctionTask subclass, got {type(prompt_validation_object).__name__}."
            )
        function_task = prompt_validation_object

        # Image outputs are OPTIONAL — a function task that only produces JSONL
        # (SAM3 returning COCO polygons, say) needs no allocator at all, and
        # asking for one it never uses would create empty shard folders. A task
        # opts in simply by calling self.reserve() in build_request(), which is
        # exactly what needs an allocator to exist.
        image_output_path = getattr(args, "image_output_path", None)
        if image_output_path:
            shard_allocator = ShardAllocator(
                root=image_output_path,
                host=node_name,
                max_shard_bytes=getattr(args, "max_shard_bytes", DEFAULT_MAX_SHARD_BYTES),
                logger=main_logger.logger,
                on_shard_sealed=getattr(args, "on_shard_sealed", None),
                # OFF by default: a sealed folder is kept, so every image exists
                # both loose and inside its tar (~2x disk). Set
                # args.delete_shard_folders = True to reclaim the space once the
                # tars are trusted.
                delete_folders_after_seal=getattr(args, "delete_shard_folders", False),
            )
            # Attached BEFORE VLMPrediction is constructed: its prep thread
            # starts inside __init__ and can call build_request -> reserve()
            # immediately.
            function_task.attach_shard_allocator(shard_allocator)
            main_logger.logger.info(
                f"Shard allocator: folders={shard_allocator.folders_dir}, "
                f"shards={shard_allocator.shards_dir}, "
                f"max_shard_bytes={shard_allocator.max_shard_bytes}, "
                f"delete_folders_after_seal={shard_allocator.delete_folders_after_seal}"
            )
        else:
            main_logger.logger.info(
                "No args.image_output_path — running without a shard allocator. "
                "The JSONL is still the output, as always; only image bytes need shards."
            )
    else:
        cfg_model = VLLMConfig(
            args.vlm_model_name,
            args.vlm_config_path,
            # One source for three things: the roots every image path is validated
            # against here, the engine's own `allowed_local_media_path`, and
            # `vllm serve --allowed-local-media-path`. Validating locally means an
            # out-of-root path is a precise pre-submit error rather than an opaque
            # rejection retried three times.
            allowed_media_roots=getattr(args, "allowed_media_roots", None),
            max_image_pixels=getattr(args, "max_image_pixels", DEFAULT_MAX_IMAGE_PIXELS),
        )
        # Log exactly what can silently alter an input on this run: whether images
        # are allowed at all (is_llm), the per-prompt image limit, the allowed media
        # roots, the hard pixel ceiling, and the resolution above which the MODEL'S
        # OWN processor downscales.
        main_logger.logger.info(f"VLM input limits: {cfg_model.describe_input_limits()}")
        media_root_warning = cfg_model.media_root_warning()
        if media_root_warning:
            main_logger.logger.warning(media_root_warning)

    prompt_pool   = ThreadPoolExecutor(max_workers=args.local_json_threads)
    vlm_predictor = VLMPrediction(
        request_queue=request_queue,
        response_queue=response_queue,
        stage_name=stage_name,
        reader_pool=prompt_pool,
        prompt_gen_and_output_validation_object=prompt_validation_object,
        batch_size=args.batch_size,   # hard cap on in-flight items (Phase A submission scheduler)
        vlm_config=cfg_model,         # built here so backend reaches prompt build
        model_name=getattr(args, "vlm_model_name", None) if cfg_model is not None else None,
        config_path=getattr(args, "vlm_config_path", None) if cfg_model is not None else None,
        function_task=function_task,  # mutually exclusive with the three above
        logs_dir=node_run_specific_log_dir,
        service_status=service_status,
        # Only used to refuse, at startup, a sampling-param/backend combination
        # the backend cannot honour (e.g. structured_outputs on "online"). That
        # check has to happen here rather than per request: a config mistake
        # fails every image identically, so failing them one at a time would
        # march the whole dataset into state=3.
        backend=backend,
    )

    # ------------------------------------------------------------------
    # Metrics + startup log
    # ------------------------------------------------------------------
    metrics             = PipelineMetrics()
    overall_start_time  = time.time()  # fallback; reset below once the VLM backend is ready
    cumulative_images   = 0
    cumulative_failed   = 0

    config = {
        "stage_name":         stage_name,
        "backend":            backend,
        "batch_size":         args.batch_size,
        "fsync_every_lines":  args.fsync_every_lines,
        "local_json_threads": args.local_json_threads,
        # getattr: a function backend has no model and no GPU list to report,
        # and requiring the args to carry empty ones would be a lie in the log.
        "vlm_model_name":     getattr(args, "vlm_model_name", None),
        "vlm_gpus":           getattr(args, "vlm_gpus", None),
        "output_path":        args.jsonl_output_path,
        "image_output_path":  getattr(args, "image_output_path", None),
        "db_details":         db_path_or_config,
        "logs_path":          node_run_specific_log_dir,
        "channel":            channel.describe(),
    }
    db_health = db.get_db_health()
    main_logger.log_startup(config, db_health, recovery_info)

    shutdown_reason = "complete"
    fetched_paths   = set()

    if shutdown_event and shutdown_event.is_set():
        raise GracefulShutdown("shutdown_event set before process_pipeline")

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    local_pool = ThreadPoolExecutor(max_workers=args.local_json_threads)
    try:
        # Wait for the backend to actually be ready (model loaded, or
        # `vllm serve` health-checked and verified) before starting the clock, so
        # total_runtime/throughput measure real processing time rather than
        # model-load time. `is_ready()` is set by the pool once every worker has
        # finished its init_fn — see core/status.py. This is what the old
        # `time.sleep(300)` guess used to stand in for.
        # Raising here (inside the try) is deliberate: it gets the same
        # teardown/log_shutdown treatment as any other run-aborting condition.
        if service_status is not None:
            wait_start = time.time()
            main_logger.logger.info("Waiting for the backend to finish loading...")
            while not service_status.is_ready():
                if shutdown_event and shutdown_event.is_set():
                    raise GracefulShutdown(
                        "shutdown_event set while waiting for the backend to become ready"
                    )
                reason = service_status.failure_reason()
                if reason:
                    raise VLMServiceDown(f"{reason} — the backend never became ready")
                time.sleep(1.0)
            main_logger.logger.info(
                f"backend ready after {time.time() - wait_start:.1f}s — starting the clock"
            )
            overall_start_time = time.time()
        else:
            # A CPU-only group publishes no status, and that is correct: there is
            # no backend to wait for. Say so plainly rather than as a warning —
            # the only case worth warning about (a group that HAS a backend but
            # published nothing) is no longer reachable, because the pool
            # publishes its own status in start().
            main_logger.logger.info(
                "No backend status on the channel (extras['health'] is unset) — this "
                "stage has no shared backend, so timing starts immediately."
            )
        print('***** Started Recording Time *****')

        cumulative_images, cumulative_failed = process_pipeline(
            db=db,
            writer=writer,
            local_pool=local_pool,
            vlm_predictor=vlm_predictor,
            prompt_validation_object=prompt_validation_object,
            metrics=metrics,
            main_logger=main_logger,
            overall_start_time=overall_start_time,
            batch_size=args.batch_size,
            checkpoint_interval=args.fsync_every_lines,
            fetched_paths=fetched_paths,
            shutdown_event=shutdown_event,
            # None for every vLLM run and for function tasks with no image
            # output; when present, process_iteration resolves each terminal
            # result's reservation from the very place it already sees them.
            shard_allocator=shard_allocator,
        )

    except VLMServiceDown as e:
        # The inference backend died. Every in-flight path is still state=1 and
        # is reset below — deliberately NOT marked FAILED, because none of them
        # were genuinely attempted. Re-run the pipeline once the GPU side is
        # healthy and they are picked up untouched.
        shutdown_reason = f"vlm_service_down: {e}"
        main_logger.log_error("VLM service is down — aborting run", e)

    except GracefulShutdown as e:
        shutdown_reason = str(e) if str(e) else "signal_interrupt"
        main_logger.log_error("Pipeline received shutdown signal", e)

    except KeyboardInterrupt:
        shutdown_reason = "keyboard_interrupt"

    except InterruptedError:
        shutdown_reason = "interrupted_error"

    except Exception as e:
        shutdown_reason = f"error: {str(e)}"
        main_logger.log_error("Fatal error during pipeline processing", e)

    finally:
      # Ctrl+C is off the table until teardown finishes: the two steps below are
      # what keep the DB and the JSONL consistent with each other. See
      # _shielded_teardown().
      with _shielded_teardown(main_logger):
        # Reset paths stuck in state 1 back to their original fetch state.
        #
        # This runs FIRST, before writer.close(), and that ordering is load
        # bearing: cli_utils removes a path from `fetched_paths` at the moment
        # it is handed to the writer, so everything left here is work the writer
        # will never see. Resetting first therefore cannot race the writer's own
        # mark_done(state 1 -> 2) — and it means the rows most likely to be
        # abandoned are freed before the (slower) writer flush.
        if fetched_paths:
            try:
                db.reset_paths_to_state(fetched_paths, target_state=fetch_state)
                main_logger.logger.warning(
                    f"Reset {len(fetched_paths)} stuck paths on shutdown "
                    f"(back to state {fetch_state})"
                )
            except Exception as e:
                main_logger.log_error(
                    "Failed to reset stuck paths on shutdown", e,
                    {"stuck_count": len(fetched_paths)},
                )

        writer.close()

        # Second sweep: anything the writer accepted but could not commit (its
        # thread died, or close() timed out mid-flush) is still state=1 and is
        # no longer tracked anywhere. Without this it stays stuck until someone
        # runs `reset-stuck` by hand.
        unflushed = writer.unflushed_paths()
        if unflushed:
            try:
                db.reset_paths_to_state(unflushed, target_state=fetch_state)
                main_logger.logger.warning(
                    f"Reset {len(unflushed)} paths the writer accepted but never "
                    f"committed (back to state {fetch_state}) — their JSONL lines "
                    "were not fsynced, so they are replayed, not duplicated"
                )
            except Exception as e:
                main_logger.log_error(
                    "Failed to reset uncommitted writer paths on shutdown", e,
                    {"stuck_count": len(unflushed)},
                )

        local_pool.shutdown(wait=True)
        prompt_pool.shutdown(wait=True)
        vlm_predictor.close(wait=False)

        # After the predictor stops, so nothing can reserve a new slot while we
        # are sealing. close() releases any reservation still unresolved — those
        # belong to paths just reset from state=1 back to fetch_state above, so
        # their half-written outputs must not ship inside a tar — and then seals
        # the still-open folder, which is what lets a clean run end with every
        # image inside a tar rather than one loose folder for the next startup.
        if shard_allocator is not None:
            try:
                shard_allocator.close()
            except Exception as e:
                main_logger.log_error(
                    "Failed to close the shard allocator; any unsealed folder is "
                    "sealed by the next startup's recovery", e,
                )

        # Opt out of the exit-time flush-and-join for the one queue this
        # process put() to (request_queue — the response queue is read-only
        # here, so it is left alone). Safe/idempotent even though standalone's
        # model_term() also calls this in the single-process setup
        # (cancel_join_thread() is a no-op on an already-cancelled Finalize
        # hook). Needed here specifically because under StageWeaver cli() DOES
        # run in its own spawned process, separate from the one that owns the
        # queues — and cancellation is process-local. See
        # docs/archive/exit_hang_postmortem.md.
        channel.cancel_join_threads(include_responses=False)

        db_health = db.get_db_health()
        db.close()

        total_runtime = time.time() - overall_start_time
        main_logger.log_shutdown(
            shutdown_reason,
            total_runtime,
            cumulative_images,
            cumulative_failed,
            db_health,
            writer,
        )
        main_logger.logger.warning(f"Logs path: {node_run_specific_log_dir}")
        main_logger.logger.warning(f"JSONL output path: {jsonl_output_path}")
        print(
            f"------ Pipeline process ended due to {shutdown_reason}. "
            f"Please wait before exiting ------\n"
        )