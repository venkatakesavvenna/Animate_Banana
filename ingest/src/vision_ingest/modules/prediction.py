import threading
import queue
import time
import copy
from multiprocessing import Queue
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import traceback

from vision_ingest.vllm_module.retry_validate import stage, submit_token, drain_available
from vision_ingest.vllm_module.vllm_config import VLLMConfig, PromptConfigError
from vision_ingest.core.status import ServiceStatus, VLMServiceDown
from vision_ingest.core.wire import KIND_INFRA, KIND_OK, KIND_REJECT
from vision_ingest.core.task import FunctionTask
from vision_ingest.utils.utils import get_logger, GracefulShutdown
from vision_ingest.project_specific import PromptAndValidation
from vision_ingest.vllm_module.vllm import StructuredOutputsParams, SamplingParams


# =============================================================================
# v1.5 — streaming predictor against the AsyncLLM continuous-batching worker.
#
# Three internal queues, two internal threads, one ownership counter:
#
#   paths_in ──► PREP THREAD ─────────────► request_queue / inflight{}
#   retry_q ──►  (retries first — cheap           │
#                re-stage, no prompt prep;        │  the ONLY submitter
#                then fresh paths in chunks       ▼
#                of ≤ batch_size)            AsyncLLM worker
#                                                 │
#   response_queue ◄───────────────────────────────┘
#        │
#        ▼
#   COLLECTOR THREAD — drain, release slot, validate, classify:
#     success / attempts exhausted / oversize ──► terminals ("ok"/"fail")
#     failed with attempts left ────────────────► retry_q
#
#   run_iteration(fresh_paths) — cli's only entry point:
#     push fresh_paths onto paths_in (instant), then block on terminals until
#     ~batch_size results accumulate (or nothing is active, or MAX_LINGER_S
#     after the first result), and return (failed, successful, fetch_n).
#
# INVARIANT: every path handed in is in exactly one of {paths_in / being
# prepped, retry_q, inflight, terminals, returned}. `_owned` counts everything
# not yet returned (+= on hand-in, -= on return), so
#     fetch_n = desired_total - _owned          (desired_total = 3·batch_size)
# is exact by conservation — it is cli's admission control: cli never fetches
# more than fetch_n, so _owned ≤ 3B and at most that many DB rows sit in
# state=1 at once. Since v1.9.1 this counter is the ONLY backpressure: the
# shared-memory free_slots pool that used to bound submission from underneath is
# gone, along with the payloads that justified it.
#
# Because submission runs on its own thread, the GPU stays fed no matter how
# long cli spends in fetch/writer/fsync — and run_iteration is therefore free
# to linger for a chunky return (fetch_n comes back ≈ batch_size, not 1–2, so
# DB fetches stay worthwhile). A straggler delays only itself; its cohort is
# classified, retried, or returned around it.
#
# Thread-safety notes:
#   - Exactly one thread writes request_queue / inserts into inflight (prep);
#     exactly one pops it (collector). The inflight insert happens BEFORE the
#     queue.put so a response can never race back to an unknown req_id.
#   - Either thread's crash is stored in _thread_error and re-raised into cli
#     by the next run_iteration → normal shutdown path (state 1→0 reset).
#
# Crash safety is unchanged: everything predictor-owned is state=1 in the DB;
# cli.py's finally resets 1→0 on any exit. See docs/v1.5_changes.md.
#
# A record is a plain dict, identical for fresh work and retries:
#   {
#     "path":         str,           # image path (identity through the pipeline)
#     "orig_prompt":  dict,          # unified payload (raw text + image paths,
#                                    #   see core/wire.py); never None
#     "attempts":     int,           # GPU attempts already made (0 == fresh)
#     "last_error":   Optional[str], # error text from the most recent attempt
#     "req_id":       str,           # uuid of the currently-staged request
#     "submit_token": int | payload, # what submit_token() puts on request_queue
#     "sent_payload": dict,          # the payload actually staged (for output records)
#   }
#
# v1.9: the payload is OPAQUE. This class touches its contents in exactly two
# places — it records `send` on the record for the output line, and it hands
# `response.output` to wrap_and_validate_output(). It never reads "prompt",
# never reads "image_paths", and never assumes the response is text. The one
# constraint that remains is structural: a payload is always a JSON-able dict.
#
# That is what lets `function_task` (see core/task.py) replace `vlm_config`
# wholesale: with a FunctionTask set, get_prompts() calls build_request() per
# path instead of the get_image_specific_prompt -> get_prompt_with_image
# two-step, sampling params resolve to None, and everything else below — the
# retry taxonomy, the infra fuse, the ownership counter, the terminals queue —
# runs unchanged and unaware of which backend is on the other end of the queue.
#
# v1.8: responses come back as VLLMResponse(req_id, kind, text, stats, error),
# already classified ok/reject/infra by the worker that ran them, so
# _validate_one only has to decide whether the model's OUTPUT is valid. And
# because "orig_prompt" now holds raw, un-templated text on every backend, the
# retry hook's error-append lands in the user turn everywhere — offline it used
# to be appended after the templated string's generation prompt, i.e. inside the
# assistant's own turn.
# =============================================================================


class VLMPrediction():

    # After the first terminal result of a run_iteration call arrives, keep
    # collecting at most this long before returning a partial chunk — bounds
    # how stale a finished result can get before it reaches the writer.
    MAX_LINGER_S = 5.0

    # How many times ONE item may be re-submitted after an infrastructure
    # failure (a response of None: engine/transport error, not a bad output).
    # These do not consume the validation budget — an item that never reached
    # the model has not been "attempted" in any meaningful sense.
    MAX_INFRA_ATTEMPTS = 5

    def __init__(
        self,
        request_queue: Queue,
        response_queue: Queue,
        stage_name: str,
        reader_pool: ThreadPoolExecutor,
        prompt_gen_and_output_validation_object: PromptAndValidation,
        batch_size: int,
        max_attempts: int = 3,
        # VLLMConfig is optional — if not passed, created internally from these args
        vlm_config: Optional[VLLMConfig] = None,
        model_name: Optional[str] = None,
        config_path: Optional[str] = None,
        # v1.9 §2: the function-backend counterpart of vlm_config, and mutually
        # exclusive with it. Set it and this predictor builds requests by calling
        # function_task.build_request(path, logger) instead of prompting a VLM.
        # Left None (the default) nothing below changes at all.
        function_task: Optional[FunctionTask] = None,
        logs_dir: str = "./logs",
        service_status: Optional[ServiceStatus] = None,
        backend: str = "async",
    ):
        self.logger = get_logger(logs_dir, "vlm_prediction")

        # --- backend discriminator (exactly one of these two is set) ---
        self.function_task: Optional[FunctionTask] = function_task
        if function_task is not None and (
            vlm_config is not None or model_name is not None or config_path is not None
        ):
            raise ValueError(
                "function_task is mutually exclusive with vlm_config / model_name / "
                "config_path — a function backend has no VLM to configure."
            )

        if function_task is not None:
            self.logger.info(
                f"Initializing VLMPrediction: stage={stage_name}, "
                f"function_task={type(function_task).__name__}"
            )
        else:
            self.logger.info(
                f"Initializing VLMPrediction: stage={stage_name}, "
                f"model={model_name if vlm_config is None else vlm_config.model_name}"
            )

        # --- IPC primitives (queues owned by caller, never closed here) ---
        self.request_queue  = request_queue
        self.response_queue = response_queue
        self.stage_name     = stage_name

        # --- VLLMConfig: accept externally or build from args ---
        # Skipped entirely for a function backend: there is no model, no chat
        # template and no image policy to resolve, and constructing one would
        # drag transformers + a model directory into a CPU-only deployment.
        self.cfg_model: VLLMConfig = vlm_config
        if self.cfg_model is None and function_task is None:
            if model_name is None or config_path is None:
                raise ValueError(
                    "Either vlm_config, both model_name and config_path, or "
                    "function_task must be provided."
                )
            try:
                self.cfg_model = VLLMConfig(model_name, config_path)
            except Exception as e:
                self.logger.error(f"Failed to initialize VLLMConfig: {e}")
                raise

        # --- prompt / validation hooks ---
        # A FunctionTask exposes the same hook names, so everything downstream
        # of this line is written once and serves both backends. cli() passes
        # the task as prompt_validation_object as well, so these normally
        # resolve to the same object; preferring function_task explicitly means
        # a caller that passes them separately still gets the coherent pair.
        hooks = function_task if function_task is not None else prompt_gen_and_output_validation_object
        self.enable_thinking = getattr(hooks, "enable_thinking", False)
        self.get_cur_img_prompt_fn   = getattr(hooks, "get_image_specific_prompt", None)
        self.validate_and_postporcess_fn = hooks.wrap_and_validate_output
        # How a retry differs from the attempt before it. Overridable per
        # project (v1.8 §3); the default reproduces the behaviour that used to
        # be hardcoded here — append the error to the prompt, bump
        # repetition_penalty by 0.5 per attempt. See project_specific.py.
        self.build_retry_request_fn = hooks.build_retry_request

        # --- sampling params (optional override from project_specific) ---
        self.sampling_params: SamplingParams = None
        if hasattr(hooks, "get_sampling_params"):
            self.sampling_params = hooks.get_sampling_params()

        # Resolved once, here, and reused for every record for this object's
        # whole lifetime — which is exactly why _stage_and_submit hands the
        # retry hook a deepcopy and never this object itself.
        self.backend = backend
        self.base_sampling_params: SamplingParams = self._base_sampling_params()
        self._check_backend_supports_sampling_params()

        # --- thread pools ---
        self.reader_pool = reader_pool      # image I/O, owned by caller

        # --- batching / admission state (see module docstring) ---
        self.batch_size    = batch_size     # fresh-prep chunk / return-chunk size
        self.max_attempts  = max_attempts   # total GPU attempts per item before terminal
        self.desired_total = 3 * batch_size # cap on predictor-owned items (== state=1 rows)

        # --- internal queues (see module docstring diagram) ---
        self.paths_in  = queue.Queue()      # fresh paths handed in by cli
        self.retry_q   = queue.Queue()      # records awaiting re-stage + resubmit
        self.terminals = queue.Queue()      # ("ok", obj) / ("fail", path) for cli
        self.inflight: dict = {}            # req_id → record (prep inserts, collector pops)

        self._owned      = 0                # items handed in and not yet returned
        self._owned_lock = threading.Lock()

        # --- backend liveness (see core/status.py) ---
        # An `infra` response means "the backend could not produce an answer".
        # That is retryable when it is a hiccup and FATAL when the engine has
        # died — and the difference matters enormously, because the old code
        # treated both as "retry then mark the path FAILED", which walked the
        # whole database into state=3 the moment vllm serve fell over.
        self.service_status = service_status
        # Consecutive infra failures with no success in between. Deliberately
        # independent of the status object above: this catches "the backend is
        # alive and failing everything", which no liveness signal can see — a
        # server answering 500s, a model returning empty output forever.
        self._consecutive_infra = 0
        self._infra_fatal_threshold = max(2 * batch_size, 64)
        # Total successful generations this run. Used to decide whether a
        # repeatedly-failing item is "the item" or "the backend": blaming the
        # item is only defensible once something, anything, has succeeded.
        self._total_success = 0

        # --- internal threads ---
        self._stop = threading.Event()      # set by close(); threads poll it
        self._thread_error: Optional[BaseException] = None
        self._prep_thread = threading.Thread(
            target=self._thread_main, args=("prep", self._prep_loop),
            name="vlm-prep", daemon=True)
        self._collect_thread = threading.Thread(
            target=self._thread_main, args=("collector", self._collect_loop),
            name="vlm-collect", daemon=True)
        self._prep_thread.start()
        self._collect_thread.start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self, wait: bool = False):
        """
        Stop the internal threads.

        Does NOT touch request_queue / response_queue — those are owned by the
        caller (the ServiceChannel, created in main.py / the group leader).
        Backend process teardown is model_term()'s / term_fn()'s job.
        """
        self._stop.set()
        for t in (self._prep_thread, self._collect_thread):
            try:
                if t is not None and t.is_alive():
                    t.join(timeout=5.0 if wait else 1.0)
            except Exception as e:
                self.logger.error(f"Failed to join {t.name}: {e}")
        self.logger.info("VLMPrediction internal threads stopped")

    def _thread_main(self, name: str, loop_fn):
        """Wrapper for both internal threads: exit quietly on shutdown, store
        any real exception for run_iteration to re-raise into cli."""
        try:
            loop_fn()
        except GracefulShutdown:
            pass
        except Exception as e:
            self.logger.error(
                f"{name} thread died: {type(e).__name__}: {e}\n{traceback.format_exc()}"
            )
            if self._thread_error is None:
                self._thread_error = e

    def _raise_if_thread_dead(self):
        err = self._thread_error
        if err is None:
            return
        # VLMServiceDown is not an internal-thread bug — it is the deliberate
        # "the backend is gone, stop the run" signal, and cli() reports it as
        # such. Re-raise it unwrapped so the shutdown reason stays readable.
        if isinstance(err, VLMServiceDown):
            raise err
        raise RuntimeError("VLMPrediction internal thread failed") from err

    # ------------------------------------------------------------------
    # Prompt preparation (CPU-bound)
    # ------------------------------------------------------------------

    def get_prompts(
        self,
        batch_paths: Optional[List[str]],
        shutdown_event: threading.Event = None,
    ) -> List:
        """
        Build the request payload for each path in a batch.

        Runs on the prep thread; per-item failures come back as None entries,
        which _prep_batch turns into terminal failures for exactly those paths.

        Two ways to build a payload, one discriminator (v1.9 §2):
          - function_task set → build_request(path, logger) per path, through
            the same reader_pool.
          - otherwise (the vLLM path, unchanged) → get_image_specific_prompt()
            per path, then one get_prompt_with_image() call for the batch.
        """
        if not batch_paths:
            return []

        if self.function_task is not None:
            return self._build_function_requests(batch_paths, shutdown_event)

        try:
            futures = [
                self.reader_pool.submit(self.get_cur_img_prompt_fn, p, self.logger)
                for p in batch_paths
            ]
            prompts = []
            image_paths_to_send_to_vlm = []

            for idx_f, f in enumerate(futures):
                # Wait on the future itself (blocks until ready, up to the
                # timeout), re-checking shutdown_event on the timeout boundary.
                while True:
                    if shutdown_event and shutdown_event.is_set():
                        raise GracefulShutdown(
                            "Graceful shutdown requested during prompt generation"
                        )
                    try:
                        cur_prompt, cur_img_path = f.result(timeout=0.25)
                        break
                    except FuturesTimeoutError:
                        continue
                    except Exception as e:
                        self.logger.error(
                            f"Prompt generation failed for image at path "
                            f"{batch_paths[idx_f]}: {e}"
                        )
                        cur_prompt = None
                        cur_img_path = None
                        break

                prompts.append(cur_prompt)
                image_paths_to_send_to_vlm.append(cur_img_path)

            final_prompts = self.cfg_model.get_prompt_with_image(
                image_paths_to_send_to_vlm,
                prompts,
                reader_pool=self.reader_pool,
                logger=self.logger,
                shutdown_event=shutdown_event,
                enable_thinking=self.enable_thinking,
            )
            return final_prompts

        except (GracefulShutdown, PromptConfigError):
            # PromptConfigError is a misconfiguration that would fail every
            # image identically (is_llm vs images, per-prompt image limit).
            # Let it kill the prep thread so the run aborts with in-flight paths
            # reset to their fetch state, rather than failing the dataset row by
            # row. Per-image problems come back as None entries instead.
            raise
        except Exception as e:
            self.logger.error(
                f"get_prompts() failed: {type(e).__name__}: {e}\n{traceback.format_exc()}"
            )
            return []

    def _build_function_requests(
        self,
        batch_paths: List[str],
        shutdown_event: threading.Event = None,
    ) -> List:
        """
        Function-backend payload build (v1.9 §2): one build_request() call per
        path, on the same reader_pool the vLLM path uses.

        Deliberately the same shape as the branch above — submit everything,
        then wait on each future with a 0.25s timeout so a shutdown is noticed
        promptly — because build_request() is where a shard slot gets reserved,
        and reserving is I/O-ish work worth overlapping across a batch.

        A path whose build_request raises (or returns None) comes back as None
        and is failed terminally by _prep_batch. If it had already reserved a
        shard slot before raising, that reservation is released by cli_utils
        from the terminal-failure path — reservations are keyed on the input
        path precisely so a half-built request cannot strand one.
        """
        def build_one(path):
            try:
                return self.function_task.build_request(path, self.logger)
            except Exception as e:
                self.logger.error(
                    f"build_request() failed for {path}: {type(e).__name__}: {e}\n"
                    f"{traceback.format_exc()}"
                )
                return None

        futures = [self.reader_pool.submit(build_one, p) for p in batch_paths]
        payloads = []
        for idx_f, f in enumerate(futures):
            while True:
                if shutdown_event and shutdown_event.is_set():
                    raise GracefulShutdown(
                        "Graceful shutdown requested during request build"
                    )
                try:
                    payloads.append(f.result(timeout=0.25))
                    break
                except FuturesTimeoutError:
                    continue
                except Exception as e:
                    # build_one already swallows task errors; reaching here means
                    # the pool itself failed for this item.
                    self.logger.error(
                        f"request build future failed for {batch_paths[idx_f]}: {e}"
                    )
                    payloads.append(None)
                    break
        return payloads

    # ------------------------------------------------------------------
    # Record helpers (prompt / params / staging)
    # ------------------------------------------------------------------

    def _base_sampling_params(self) -> SamplingParams:
        """
        cfg_model.get_sampling_params() is always the base — it merges the
        model's generation_config.json exactly like `vllm serve`'s HTTP layer
        does (see VLLMConfig.get_sampling_params), so the "no task override"
        case matches online serve automatically. The project's own
        get_sampling_params() (self.sampling_params, captured once in
        __init__) then layers on top of that base:
          - None            -> base is used untouched.
          - dict            -> only those fields overwrite the base; every
                                other field keeps its generation_config value.
          - SamplingParams  -> legacy full-replacement override (kept for
                                project_specific implementations that already
                                construct a complete SamplingParams object).

        Returns None for a function backend: sampling params are a vLLM concept
        with no meaning for an arbitrary Python function. `params` stays an
        opaque Optional slot on VLLMRequest, carried through to the worker and
        ignored there — exactly as it was already typed.
        """
        if self.function_task is not None:
            return None
        base = self.cfg_model.get_sampling_params()
        override = self.sampling_params
        if override is None:
            return base
        if isinstance(override, dict):
            for key, value in override.items():
                setattr(base, key, value)
            return base
        return override

    def _check_backend_supports_sampling_params(self) -> None:
        """
        Refuse, at startup, a sampling-param/backend combination the backend
        cannot honour.

        This is v1.8 Defect 1. `structured_outputs` has no equivalent in the
        online backend's request shape, so online_worker raised while building
        the body — once per request, which the stage could only read as "this
        item was rejected". A single misconfiguration therefore marched the
        entire dataset into state=3 at full speed, resetting the
        consecutive-infra fuse each time, which is precisely the failure mode
        v1.7's health work existed to stop. A configuration mistake affects
        every image equally, so it belongs here, before a single row is fetched.
        """
        if self.function_task is not None:
            # No sampling params exist to be unsupported.
            return
        if self.backend != "online":
            return
        if getattr(self.base_sampling_params, "structured_outputs", None) is None:
            return
        raise ValueError(
            "backend='online' cannot honour structured_outputs: the online worker "
            "drives `vllm serve` over /v1/chat/completions, which this request shape "
            "does not forward it through. Every request would generate unconstrained "
            "output and then fail your own validate_output(). Use backend='async', or "
            "drop structured_outputs from get_sampling_params()."
        )

    def _build_retry_request(self, rec: dict):
        """
        Build the (prompt, sampling_params) for one retry, via the project's
        `build_retry_request` hook. Fresh items (attempts == 0) never get here —
        they send the original prompt and the shared base params untouched.

        Copy-safety is owned HERE, not by the hook: the hook is handed a fresh
        shallow dict and a deepcopy of the base params, never `rec["orig_prompt"]`
        or `self.base_sampling_params` themselves. Those are reused for every
        record for this object's whole lifetime, so an override that mutated
        what it was given in place — an easy thing to write — would silently
        corrupt every subsequent *fresh* attempt too, not just this retry. The
        prompt is likewise always rebuilt from the ORIGINAL text, so errors from
        successive attempts can never stack up inside it.
        """
        send, sp = self.build_retry_request_fn(
            dict(rec["orig_prompt"]),
            copy.deepcopy(self.base_sampling_params),
            rec["attempts"],
            rec["last_error"],
        )
        return send, sp

    def _stage_and_submit(self, rec: dict):
        """
        Stage rec's current attempt and submit it. Fresh items send the original
        prompt and base sampling params; retries go through the project's
        build_retry_request hook. Runs ONLY on the prep thread (the single
        submitter).
        """
        if rec["attempts"] == 0:
            send, sp = rec["orig_prompt"], self.base_sampling_params
        else:
            send, sp = self._build_retry_request(rec)
        req_id, token = stage(self.stage_name, send, sp)
        rec["req_id"]       = req_id
        rec["submit_token"] = token
        # v1.9 §2: keep the whole payload, not one field of it. This is one of
        # exactly two places this class touches a payload, and it does not read
        # it here — the output record is built from it in _classify().
        rec["sent_payload"] = send
        # Insert into inflight BEFORE the queue.put — the collector runs on
        # another thread, so a response must never beat the bookkeeping.
        self.inflight[req_id] = rec
        submit_token(self.request_queue, token)

    def _validate_one(self, response):
        """
        Classify one collected response. Returns (status, payload):
          "ok"     → payload = (validated_output, stats dict)
          "fail"   → payload = error str; the model answered but the answer is
                     wrong/invalid. Retrying can plausibly help.
          "reject" → payload = error str; the backend refused THIS input
                     (bad/corrupt/oversized image, prompt too long, ...).
                     Deterministic — retrying re-sends the identical request,
                     so it is terminal on the first occurrence.
          "infra"  → payload = error str; the backend could not answer at all
                     (engine dead, connection refused, worker crash). Says
                     nothing about the input, so it must never mark a path
                     FAILED. See _classify().

        As of v1.8 the first three of those are decided by the WORKER and simply
        read off `response.kind` — this method no longer reverse-engineers them
        from the shape of an untyped object (a `type(...).__name__` string match
        for online rejections, `.usage` duck-typing to tell an online shim from
        an offline RequestOutput, `None` for everything else). All that is left
        here is the one decision the worker genuinely cannot make: whether the
        model's actual output passes THIS task's validation.
        """
        if response is None:
            # Should not happen — every worker emits exactly one tagged response
            # per request. Treat it as infra so it can never fail a path.
            return "infra", "backend returned no response object"
        if response.kind == KIND_INFRA:
            return "infra", response.error or "backend could not produce an output"
        if response.kind == KIND_REJECT:
            return "reject", response.error or "backend refused this request"
        if response.kind != KIND_OK:
            return "infra", f"backend sent an unknown response kind {response.kind!r}"
        try:
            # `.output` rather than `.text` (v1.9 §2): for a vLLM backend these
            # are the same string, for a function backend it is whatever call_fn
            # returned. Nothing here needs to know which.
            validated_output = self.validate_and_postporcess_fn(response.output)
        except Exception as e:
            return "fail", str(e)
        return "ok", (validated_output, dict(response.stats or {}))

    def _note_fatal(self, exc: BaseException):
        """Record the first fatal condition; run_iteration re-raises it into cli."""
        if self._thread_error is None:
            self._thread_error = exc
        self.logger.error(f"FATAL: {exc}")
        # Wake both loops so they stop doing work nobody will use.
        self._stop.set()

    def _check_service_alive(self, context: str) -> None:
        """Turn a dead backend into a fatal error rather than a wave of
        FAILED paths. Safe to call from either internal thread."""
        if self.service_status is None:
            return
        reason = self.service_status.failure_reason()
        if reason:
            self._note_fatal(VLMServiceDown(
                f"{reason} — aborting the run with in-flight paths left in state=1 "
                f"so they are reset to their fetch state and replayed, not marked "
                f"FAILED [{context}]"
            ))

    # ------------------------------------------------------------------
    # PREP THREAD — retries first, then fresh paths in chunks
    # ------------------------------------------------------------------

    def _prep_loop(self):
        while not self._stop.is_set():
            # Retries first: re-staging is cheap (no prompt prep) and a retry
            # rejoins the running batch immediately.
            worked = False
            while True:
                try:
                    rec = self.retry_q.get_nowait()
                except queue.Empty:
                    break
                self._stage_and_submit(rec)
                worked = True

            # Fresh paths in chunks of ≤ batch_size so get_prompts amortizes
            # the reader pool. Block briefly only when there was nothing to do
            # (keeps the loop from spinning while staying responsive to stop).
            paths: List[str] = []
            if not worked:
                try:
                    paths.append(self.paths_in.get(timeout=0.25))
                except queue.Empty:
                    continue
            while len(paths) < self.batch_size:
                try:
                    paths.append(self.paths_in.get_nowait())
                except queue.Empty:
                    break
            if paths:
                self._prep_batch(paths)

    def _prep_batch(self, paths: List[str]):
        """Prep + stage + submit one chunk of fresh paths. Prep failures (None
        prompt) and oversize are terminal immediately — a failed prep is never
        re-prepared, an oversized prompt only grows on retry."""
        prepared = self.get_prompts(paths, self._stop)
        if len(prepared) != len(paths):
            # get_prompts() returns [] only on a catastrophic internal failure
            # (per-image failures come back as None entries).
            raise Exception(
                "Error in get_prompts(): returned "
                f"{len(prepared)} prompts for {len(paths)} input paths."
            )
        for path, prompt in zip(paths, prepared):
            if self._stop.is_set():
                raise GracefulShutdown("stop requested during prep batch")
            if prompt is None:
                self.terminals.put(("fail", path))
                self.logger.error(f"Terminal fail (prompt preparation): {path}")
                continue
            rec = {
                "path":        path,
                "orig_prompt": prompt,
                "attempts":    0,
                "last_error":  None,
            }
            self._stage_and_submit(rec)

    # ------------------------------------------------------------------
    # COLLECTOR THREAD — drain responses, validate, classify
    # ------------------------------------------------------------------

    def _collect_loop(self):
        while not self._stop.is_set():
            # Blocks ≤0.25s for the first response, then greedily drains the
            # rest (may return empty) — so responses keep flowing back even
            # while cli is busy in fsync/DB.
            drained = drain_available(self.response_queue)
            if not drained:
                # Nothing arrived. If we are still waiting on submitted
                # requests, make sure the thing that owes them is still alive —
                # otherwise the pipeline blocks forever on a dead process
                # (which is what used to happen on a SIGKILLed backend process).
                if self.inflight:
                    self._check_service_alive("waiting on in-flight responses")
                continue
            for response in drained:
                rec = self.inflight.pop(response.req_id, None)
                if rec is None:
                    self.logger.error(
                        f"collector: response for unknown req_id {response.req_id}")
                    continue
                self._classify(rec, response)

    def _classify(self, rec: dict, response):
        """
        success                  → terminals ("ok", output obj)
        reject (bad input)       → terminals ("fail", path), immediately
        fail, attempts remain    → retry_q (prep thread re-stages + resubmits)
        fail, attempts exhausted → terminals ("fail", path)
        infra (backend failure)  → retry on its own budget; NEVER a terminal
                                   failure while the backend is at fault
        """
        status, payload = self._validate_one(response)
        if status == "ok":
            self._consecutive_infra = 0
            self._total_success += 1
            validated_output, full_vlm_obj = payload
            # Retries consumed before this success — thrown away otherwise,
            # but it's a useful diagnostic (how often is this task retrying?).
            if isinstance(full_vlm_obj, dict):
                full_vlm_obj["attempts"] = rec["attempts"]
            # The output record is built FROM the payload rather than from a
            # field the predictor stashed earlier, which is what keeps the vLLM
            # JSONL byte-identical to v1.8: `prompt_str` is still exactly
            # `send["prompt"]`. A function payload has no "prompt", so that key
            # comes out None and the payload itself is recorded instead — added
            # only on the function path, so the vLLM record gains no new key.
            payload = rec["sent_payload"]
            obj = {
                "path":            rec["path"],
                "prompt_str":      payload.get("prompt"),
                "vlm_output":      validated_output,
                "full_vlm_object": full_vlm_obj,
            }
            if self.function_task is not None:
                obj["sent_payload"] = payload
            self.terminals.put(("ok", obj))
            return

        if status == "infra":
            self._classify_infra(rec, payload)
            return

        # The model answered (or the server refused this specific input): the
        # backend is demonstrably working, so reset the infra streak.
        self._consecutive_infra = 0

        if status == "reject":
            # Deterministic per-input refusal. Re-sending the identical request
            # cannot change the answer, so burning two more attempts on it only
            # delays the report and hides the reason.
            self.terminals.put(("fail", rec["path"]))
            self.logger.error(
                f"VLM rejected the input for path: {rec['path']} — {payload} "
                "(not retried: the backend refused this exact request)"
            )
            return

        rec["attempts"]  += 1
        rec["last_error"] = payload
        if rec["attempts"] >= self.max_attempts:
            self.terminals.put(("fail", rec["path"]))
            self.logger.error(
                f"VLM validation failed after {rec['attempts']} attempts for path: "
                f"{rec['path']} (last error: {payload})"
            )
        else:
            self.retry_q.put(rec)

    def _classify_infra(self, rec: dict, payload: str):
        """
        Handle a response the backend could not produce.

        Rule: an infrastructure failure never marks a path FAILED. Either the
        backend recovers and the item is retried, or the backend is gone and the
        whole run aborts with the path still in state=1 — so cli()'s shutdown
        resets it to its fetch state and the next run picks it up untouched.
        """
        self._consecutive_infra += 1
        rec["infra_attempts"] = rec.get("infra_attempts", 0) + 1
        rec["last_error"] = payload

        # Did the service tell us (or the OS tell us) that it is gone?
        self._check_service_alive(f"backend failure on {rec['path']}")
        if self._thread_error is not None:
            return

        if self._consecutive_infra >= self._infra_fatal_threshold:
            # Nothing has succeeded for a long stretch. Whatever the health flag
            # says, this backend is not serving; stop before the failures start
            # looking like data problems.
            self._note_fatal(VLMServiceDown(
                f"{self._consecutive_infra} consecutive backend failures with no "
                f"successful generation (last error: {payload}) — treating the VLM "
                "service as down and aborting without marking any path FAILED"
            ))
            return

        # Blaming the item requires evidence that the backend is otherwise fine:
        # something must have succeeded this run, and we must not currently be
        # in a failure streak. Without that second condition a totally dead
        # backend still fails every path — each item just exhausts its own infra
        # budget before the global fuse (below) has seen enough failures.
        others_are_working = (
            self._total_success > 0
            and self._consecutive_infra <= self.MAX_INFRA_ATTEMPTS
        )
        if rec["infra_attempts"] > self.MAX_INFRA_ATTEMPTS and others_are_working:
            # This one item keeps failing at the backend while others succeed —
            # so it is the item, not the service. Report it as itself.
            self.terminals.put(("fail", rec["path"]))
            self.logger.error(
                f"backend failed {rec['infra_attempts']} times for path {rec['path']} "
                f"while other requests were succeeding (last error: {payload})"
            )
            return

        self.logger.warning(
            f"backend failure {rec['infra_attempts']}/{self.MAX_INFRA_ATTEMPTS} for "
            f"{rec['path']} ({payload}) — re-queued, attempt count untouched"
        )
        self.retry_q.put(rec)

    # ------------------------------------------------------------------
    # Introspection (cli uses these for its exit / sleep decision)
    # ------------------------------------------------------------------
    # The three counts partition _owned (in_prep is derived as the remainder,
    # so items mid-handoff between queues are never invisible): cli's guard
    # "pending==0 and inflight==0 and in_prep==0" is true iff owned == 0.

    def pending_count(self) -> int:
        """Items awaiting resubmission (retry_q) or pickup by cli (terminals)."""
        return self.retry_q.qsize() + self.terminals.qsize()

    def inflight_count(self) -> int:
        """Submitted requests still awaiting their streamed response."""
        return len(self.inflight)

    def in_prep_count(self) -> int:
        """Paths queued for / undergoing prompt prep (derived remainder)."""
        with self._owned_lock:
            owned = self._owned
        return max(0, owned - self.inflight_count() - self.pending_count())

    # ------------------------------------------------------------------
    # Main entry point — called by cli_utils.process_pipeline
    # ------------------------------------------------------------------

    def run_iteration(
        self,
        fresh_paths: Optional[List[str]] = None,
        shutdown_event: threading.Event = None,
    ):
        """
        Hand fresh_paths to the prep thread (instant), then block on the
        terminals queue and return a chunk of finished results.

        Returns when ANY of:
          - ≥ batch_size terminal results collected (a chunky return keeps the
            next fetch_n ≈ batch_size — no 1–2 row DB fetches), or
          - nothing is active anymore (everything owned is in hand — lets cli
            run its drain/exit checks), or
          - MAX_LINGER_S elapsed since this call's first terminal result
            (bounds writer-handoff latency on slow/straggler stretches).

        The GPU never waits on this method: submission runs on the prep thread.

        Returns (failed_paths, successful_objs, fetch_n):
          failed_paths / successful_objs — terminal results, from whichever
            submission they originated in.
          fetch_n = desired_total - owned. Exact by conservation (owned is a
            counter, += handed in, -= returned); caps state=1 rows at 3B.
        """
        self._raise_if_thread_dead()

        if fresh_paths:
            with self._owned_lock:
                self._owned += len(fresh_paths)
            for p in fresh_paths:
                self.paths_in.put(p)

        failed_paths: List[str] = []
        successful_objs: List[dict] = []
        first_at: Optional[float] = None

        def take(block: bool) -> bool:
            try:
                tag, item = (self.terminals.get(timeout=0.25) if block
                             else self.terminals.get_nowait())
            except queue.Empty:
                return False
            if tag == "ok":
                successful_objs.append(item)
            else:
                failed_paths.append(item)
            return True

        while True:
            if shutdown_event and shutdown_event.is_set():
                raise GracefulShutdown("Graceful shutdown requested in run_iteration")
            self._raise_if_thread_dead()

            while take(block=False):
                pass
            collected = len(failed_paths) + len(successful_objs)
            if collected and first_at is None:
                first_at = time.time()

            with self._owned_lock:
                owned = self._owned
            if collected >= self.batch_size:
                break
            if owned - collected <= 0:      # nothing active → nothing more will arrive
                break
            if first_at is not None and time.time() - first_at >= self.MAX_LINGER_S:
                break
            take(block=True)

        with self._owned_lock:
            self._owned -= len(failed_paths) + len(successful_objs)
            owned = self._owned
        fetch_n = max(0, self.desired_total - owned)

        self.logger.info(
            f"run_iteration returned: in_prep={self.in_prep_count()}, "
            f"pending={self.pending_count()}, in_flight={self.inflight_count()}, "
            f"successful_objs={len(successful_objs)}, failed_paths={len(failed_paths)}, "
            f"fetch_n={fetch_n}"
        )
        return failed_paths, successful_objs, fetch_n
