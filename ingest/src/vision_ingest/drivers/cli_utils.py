import time, threading, os, json, traceback
from concurrent.futures import ThreadPoolExecutor
from vision_ingest.db.db import DB
from vision_ingest.modules.writer import JSONLWriter
from vision_ingest.utils import PipelineMetrics, MainLogger
from vision_ingest.modules.prediction import VLMPrediction
from vision_ingest.project_specific import PromptAndValidation
from vision_ingest.utils.utils import GracefulShutdown

# Per-run cap on how many sidecar-write failures get logged in full, so one
# systematically broken write_local_json cannot flood the log with a million
# identical tracebacks. The count keeps rising either way.
_LOCAL_JSON_ERROR_LOG_LIMIT = 20
_local_json_errors = {"logged": 0, "total": 0}


def _guarded_write_local_json(prompt_validation_object, obj, main_logger):
    """Run the project's optional per-image sidecar write, reporting failures.

    write_local_json is user-supplied and optional, so a failure here must not
    fail the image — the shared JSONL is the real output. But it must not be
    invisible either: it used to be submitted bare to the pool, where any
    exception (a wrong signature, a full disk, a bad path) vanished into a
    Future that was never inspected.
    """
    try:
        prompt_validation_object.write_local_json(obj, main_logger)
    except Exception as e:
        _local_json_errors["total"] += 1
        if _local_json_errors["logged"] < _LOCAL_JSON_ERROR_LOG_LIMIT:
            _local_json_errors["logged"] += 1
            main_logger.log_error(
                "write_local_json failed (the shared JSONL is unaffected)", e,
                {"path": obj.get("path"),
                 "total_failures_so_far": _local_json_errors["total"],
                 "traceback": traceback.format_exc()},
            )

def  process_pipeline(
    db: DB,
    writer: JSONLWriter,
    local_pool: ThreadPoolExecutor,
    vlm_predictor: VLMPrediction,
    prompt_validation_object: PromptAndValidation,
    metrics: PipelineMetrics,
    main_logger: MainLogger,
    overall_start_time,
    batch_size,
    checkpoint_interval,
    fetched_paths: set,
    shutdown_event: threading.Event = None,
    shard_allocator=None,   # core.shards.ShardAllocator | None
):
    """
    Run the CPU/GPU pipeline as a single-batching-point STREAMING loop against
    the AsyncLLM continuous-batching worker (v1.5, Phase A + A2 + C).

    All batching lives entirely inside VLMPrediction, which runs two persistent
    background threads (prep + collector, see prediction.py) that continuously
    stage/submit fresh paths and retries, and drain/classify responses as they
    stream back — independent of this loop's own pacing. This loop stays a
    plain sequential `while True`; it only:
      1. Sizes each fetch to keep the predictor-owned total bounded
         (fetch_n = desired_total - owned, returned by the previous
         run_iteration — the predictor owns the formula via an exact
         conservation counter). Keeping this bounded is what stops fresh DB
         rows from piling up idle in state=1 ahead of the engine.
      2. Calls vlm_predictor.run_iteration(fresh_paths) — hands the fresh paths
         to the predictor's prep thread (instant) and blocks on its internal
         terminals queue until a chunky batch of finished results has
         accumulated (or nothing is left active, or a staleness bound elapses;
         see prediction.py's MAX_LINGER_S), returning terminal results + the
         next fetch_n. This paces the loop on real throughput without spinning.
      3. Handles the terminal results it returns (mark failed, enqueue success).

    Bootstrap / steady-state / flush collapse into one rule — "fetch keeps the
    owned total near desired_total; each run_iteration call drains what's ready
    and feeds what's new":
      - Bootstrap  = first iterations: fetch primes at 2*batch_size so the
        engine fills quickly while the first prompts are prepared+staged.
      - Flush/drain = fetch returns empty but pending/in-flight/in-prep still
        hold items: keep calling run_iteration(None) until everything drains.
      - Multi-stage waiting = fetch empty AND pending==0 AND inflight==0 AND
        in_prep==0: sleep 1s and retry. If any work remains in the predictor we
        keep pumping instead of sleeping (don't sleep on responses already
        streaming back).

    Args:
        local_pool: ThreadPoolExecutor for local JSON writes
        main_logger: MainLogger instance
        overall_start_time: Pipeline start time
        batch_size: Per-pump top-up granularity / fetch unit
        checkpoint_interval: Log checkpoint every N images

    Returns:
        tuple: (cumulative_images, cumulative_failed)
    """
    cumulative_images = 0
    cumulative_failed = 0

    def process_iteration(fresh_paths, batch_type="regular"):
        nonlocal cumulative_images, cumulative_failed

        # === ONE PUMP STEP (fire prep || drain+classify || top-up submit) ===
        batch_start = time.time()
        inference_start = time.time()
        # Terminal results are whatever finished/exhausted this pump step —
        # retried and still-generating items stay inside the predictor. fetch_n
        # sizes the next fetch.
        failed_paths, successful_objs, fetch_n = vlm_predictor.run_iteration(fresh_paths, shutdown_event)
        inference_time_ms = (time.time() - inference_start) * 1000

        # Mark failed paths in database (terminal: exhausted / oversized / prep-failed)
        failed_count = len(failed_paths)
        if failed_paths:
            db.mark_failed(failed_paths)
            fetched_paths.difference_update(failed_paths)

        # Metrics count terminal items only.
        images_processed = failed_count + len(successful_objs)
        if images_processed > 0:
            metrics.record_inference(inference_time_ms, images_processed, failed_count)

        # === ENQUEUE SUCCESSFUL RESULTS ===
        for obj in successful_objs:
            metrics.record_tokens(obj.get("full_vlm_object"))
            # _guarded so a broken write_local_json (e.g. a signature mismatch
            # in a project's PromptAndValidation) shows up in the log. A bare
            # pool.submit() drops the exception into a Future nobody ever reads,
            # so every per-image sidecar can fail for an entire run in silence.
            local_pool.submit(_guarded_write_local_json,
                              prompt_validation_object, obj, main_logger)
            writer.enqueue(obj)  # after writing to a jsonl file, it will mark those paths as done in db.
            fetched_paths.discard(obj['path'])

        # === RESOLVE SHARD RESERVATIONS (v1.9 §5) ===
        # This is the one place the pipeline sees BOTH terminal outcomes for an
        # item, which is exactly what a reservation needs: every reserve() must
        # resolve to either a success (bytes counted toward the byte-roll) or a
        # failure (output file deleted), or its folder never reaches
        # outstanding == 0 and never seals.
        #
        # Both calls are keyed on the input path and are no-ops for an item that
        # never reserved anything — so a vLLM run, or a function task that emits
        # only JSONL, passes through here untouched.
        if shard_allocator is not None:
            for path in failed_paths:
                shard_allocator.release_failed(path)
            for obj in successful_objs:
                shard_allocator.commit(obj["path"], obj.get("vlm_output"))

        # Record end-to-end batch time
        batch_e2e_ms = (time.time() - batch_start) * 1000
        metrics.record_batch_e2e(batch_e2e_ms)

        # Update cumulative counters (terminal items only)
        if images_processed > 0:
            cumulative_images += images_processed
            cumulative_failed += failed_count

            # === LOG CHECKPOINT ===
            # Crossed a checkpoint threshold? (handles batch_size not dividing evenly)
            previous_checkpoint = (cumulative_images - images_processed) // checkpoint_interval
            current_checkpoint = cumulative_images // checkpoint_interval
            if current_checkpoint > previous_checkpoint:
                db_health = db.get_db_health()
                main_logger.log_checkpoint(
                    metrics,
                    db_health,
                    writer,
                    cumulative_images,
                    cumulative_failed,
                    overall_start_time
                )
                metrics.reset()

        # === LOG BATCH ===
        # Log bootstrap (first) and drain (last) iterations specifically.
        if batch_type in ["first", "last"]:
            operation = "prompt_preparation" if batch_type == "first" else "vlm_inference"
            additional_info = {
                "failed_count": failed_count,
                "successful_count": len(successful_objs),
                "phase": "bootstrap" if batch_type == "first" else "flush",
                "fresh_paths_count": len(fresh_paths) if fresh_paths else 0,
                "pending_pool": vlm_predictor.pending_count(),
                "inflight": vlm_predictor.inflight_count(),
                "in_prep": vlm_predictor.in_prep_count(),
                "next_fetch_n": fetch_n,
            }
            main_logger.log_batch(
                batch_type=batch_type,
                batch_size=(len(fresh_paths) if fresh_paths else 0),
                operation=operation,
                duration_ms=batch_e2e_ms,
                additional_info=additional_info
            )

        return fetch_n

    first_iteration = True
    retry_count = 0
    # Prime the pipeline toward the in-flight concurrency target (2·batch_size)
    # on the first fetch so the engine fills quickly; the predictor returns
    # fetch_n thereafter and settles at ~consumption-rate fresh per pump step
    # (bounded by desired_total = 3·batch_size predictor-owned items).
    fetch_n = 2 * batch_size
    while True:
        if shutdown_event and shutdown_event.is_set():
            raise GracefulShutdown("shutdown_event_set inside process_pipeline before fetch_batch")

        # === FETCH BATCH (sized by the predictor to keep owned total bounded) ===
        if fetch_n < 0:
            fetch_n = 0

        fetch_start = time.time()
        fresh_paths = db.fetch_batch(fetch_n) if fetch_n > 0 else []
        fetch_time_ms = (time.time() - fetch_start) * 1000

        # Track fetched paths for cleanup on shutdown
        if fresh_paths:
            fetched_paths.update(fresh_paths)

        pending = vlm_predictor.pending_count()
        inflight = vlm_predictor.inflight_count()
        in_prep = vlm_predictor.in_prep_count()

        # Nothing fetched AND nothing staged AND nothing in flight AND nothing
        # still being prepped → either fully done or waiting on upstream. (If any
        # work is still in the predictor we fall through and run a drain/top-up
        # pump step instead of sleeping on responses already streaming back.)
        if not fresh_paths and pending == 0 and inflight == 0 and in_prep == 0:
            # Going idle: force out whatever's sitting in the writer's buffer
            # rather than let it wait for fsync_every_lines (which a small
            # trailing batch may never reach) or close(). Also runs
            # mark_done()'s downstream stage-DB activation, so a waiting
            # downstream stage doesn't sit idle any longer than it has to.
            writer.flush()
            # Exit only when no rows remain in state=0 (pending) AND no rows remain at the
            # fetch_state. For fetch_state=0 (standalone) this collapses to a single check.
            # For fetch_state=4 (multi-stage) we must also wait while state=0 rows still
            # exist, because upstream will promote them to state=4 later.
            #
            # Counts, not percentages: get_state_pct() rounds to 0.1%, so on a
            # 10M-row database this used to read "0.0% pending" — and exit — with
            # up to ~5000 images still unprocessed.
            no_pending = db.get_state_count(0) == 0
            no_fetchable = db.fetch_state == 0 or db.get_state_count(db.fetch_state) == 0
            if no_pending and no_fetchable:
                main_logger.logger.warning("No data to fetch, pools empty, nothing in flight; no rows remain in pending or fetchable states. Exiting main loop.")
                break
            # Record empty fetch with wait time and retry
            metrics.record_fetch(fetch_time_ms, is_empty=True, wait_time_sec=1.0)
            if retry_count % 50 == 0:
                main_logger.logger.info(f"Waited {retry_count} seconds for new data but no new data found. Will keep checking every 1 second.")
            retry_count += 1
            time.sleep(1)
            continue
        retry_count = 0

        # Record fetch (may be empty when we're draining the pipeline mid-run)
        metrics.record_fetch(fetch_time_ms, is_empty=(not fresh_paths))

        # batch_type: first iteration = bootstrap; no-fresh (draining) = flush.
        if first_iteration:
            batch_type = "first"
        elif not fresh_paths:
            batch_type = "last"
        else:
            batch_type = "regular"

        fetch_n = process_iteration(fresh_paths, batch_type)
        first_iteration = False

    return cumulative_images, cumulative_failed

