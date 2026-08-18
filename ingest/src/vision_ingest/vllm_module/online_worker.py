"""
The online vLLM backend as three functions: launch a local `vllm serve`, drive it
over HTTP, terminate it (v1.9.2 §4).

`vllm_module/specs.py` wraps these into a `WorkerSpec`. There is no worker class,
no accept loop, no receiver/sender bridge threads, and no subprocess poll timer —
`core/service.py`'s async worker loop owns all of that now, identically for every
async backend. What is left here is the only part that was ever about vLLM:

    online_init  launch `vllm serve` (this process owns the GPU subprocess),
                 wait for /health, prove it is OUR model, build an AsyncClient
    online_call  one /v1/chat/completions round trip, classified
    online_term  terminate the subprocess

`engine_args` (from the model yaml) is translated to serve flags by
`_engine_args_to_serve_flags()`, so the online server loads with the same
max_model_len / gpu_memory_utilization / limit_mm_per_prompt / etc. as the
in-process engine.

CONTRACT PRESERVED: exactly one `VLLMResponse` per request, tagged ok / reject /
infra by this worker — which is the only place that knows which it was. v1.8
removed the `_OnlineOutput` / `_OnlineCompletion` / `_OnlineFailure` shims that
made an HTTP response impersonate a vLLM `RequestOutput`. v1.9.2 removes the
remaining plumbing: the framework does the queueing, so `online_call` returns a
value or raises, and the taxonomy is applied in one place for all backends.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

import httpx

from vision_ingest.core.task import BackendGone, InputRejected
from vision_ingest.utils.utils import get_logger

from vision_ingest.core.wire import (
    CallResult,
    build_chat_messages,
    chat_template_kwargs,
    stats_from_usage,
)

# --- serve defaults. model/TP come from engine_args + the rank's device; the
#     rest of engine_args is translated to serve flags below. ---
ONLINE_HOST = "127.0.0.1"
# 0 = pick a free ephemeral port at launch. A fixed port is actively dangerous:
# if anything else (a stale server from a previous run, another user's job) is
# already listening there, /health answers 200 and this worker silently drives
# SOMEONE ELSE'S MODEL for the whole run. Port 0 removes the collision, and
# _verify_served_model() additionally proves the server is ours. It is also what
# lets n_workers=2 launch two servers on one node with no port arithmetic.
ONLINE_PORT = 0
ONLINE_REQUEST_TIMEOUT_S = 3600      # per-request HTTP timeout (gen can be slow)
ONLINE_HEALTH_TIMEOUT_S = 1800       # max wait for `vllm serve` to come up
ONLINE_MAX_CONNECTIONS = 512         # httpx pool cap (>= this worker's max_inflight)


def _sampling_field(sp, key):
    return getattr(sp, key, None)


def _build_chat_payload(model: str, prompt: dict, sp) -> dict:
    """
    Translate one request into an OpenAI /v1/chat/completions body.

    `prompt` is the unified payload every backend receives:
        {"prompt": <raw user text>, "image_paths": [<abs path>, ...],
         "enable_thinking": <bool>}
    (see vllm_config.get_prompt_with_image / wire.build_chat_messages). The text
    is NOT chat-templated — the server applies the template. Images are sent as
    percent-encoded file:// URLs so the server reads them itself (no base64, no
    copy); the offline backend builds the identical message list and hands it to
    vLLM's renderer, which resolves the same URLs the same way.
    """
    # sp is always a fully-resolved SamplingParams by the time it gets here
    # (VLLMConfig.get_sampling_params() already merged generation_config.json for
    # any field the task didn't override — see docs/vllm_serve_offline_parity.md,
    # Divergence 2), so forwarding these explicitly reproduces exactly what the
    # server would have filled in on its own, while still carrying real overrides
    # (e.g. prediction.py's per-retry repetition_penalty bump).
    payload = {
        "model": model,
        "messages": build_chat_messages(prompt),
    }
    # Omit anything left unset (None) rather than filling in a made-up number — a
    # request that never sends max_tokens gets exactly the same
    # `max_model_len - prompt_len` fill from the server's InputProcessor that
    # offline gets from the same shared code.
    for key in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty",
                "max_tokens"):
        val = _sampling_field(sp, key)
        if val is not None:
            payload[key] = val
    stop_token_ids = _sampling_field(sp, "stop_token_ids")
    if stop_token_ids:
        payload["stop_token_ids"] = list(stop_token_ids)
    # Structured outputs have no /v1/chat/completions equivalent in this worker's
    # request shape. Unreachable in practice since v1.8 — VLMPrediction.__init__
    # refuses the combination at startup, because failing it here made a
    # *configuration* mistake look like a per-item rejection and marched the whole
    # dataset into state=3 at full speed. Kept as a backstop so the combination
    # can never silently produce unconstrained output that then fails the task's
    # own validation on every retry.
    if _sampling_field(sp, "structured_outputs") is not None:
        raise ValueError(
            "sampling params request structured_outputs, which the online backend "
            "does not forward. Use the async backend, or drop structured_outputs "
            "from get_sampling_params()."
        )
    # Forward thinking preference to the server's chat template. Offline passes the
    # same kwarg to vLLM's renderer, so both reach the template identically.
    template_kwargs = chat_template_kwargs(prompt)
    if template_kwargs:
        payload["chat_template_kwargs"] = template_kwargs
    return payload


def _error_body(resp) -> str:
    """The server's own explanation, which raise_for_status() throws away."""
    try:
        data = resp.json()
    except Exception:
        return (resp.text or "")[:2000]
    if isinstance(data, dict):
        err = data.get("error", data)
        if isinstance(err, dict):
            return str(err.get("message") or err)[:2000]
        return str(err)[:2000]
    return str(data)[:2000]


# engine_args keys handled explicitly elsewhere (positional model / device-derived
# tensor-parallel-size), so excluded from the generic translation.
_SERVE_FLAG_SKIP = {"model", "tensor_parallel_size"}


def _engine_args_to_serve_flags(engine_args: dict) -> list:
    """
    Translate a vLLM engine_args dict (the same dict passed to offline
    LLM(**engine_args) / AsyncEngineArgs) into `vllm serve` CLI flags.

      snake_case key            -> --kebab-case flag
      bool True                 -> bare flag; bool False -> --no-<flag>
      dict / list value         -> flag + JSON string (e.g. limit_mm_per_prompt)
      scalar (int/float/str)    -> flag + str(value)
      None                      -> omitted (means "let vLLM decide")

    The False -> `--no-<flag>` mapping matters: vLLM registers bool config fields
    with argparse.BooleanOptionalAction, and several of them (notably
    async_scheduling) default to ENABLED when the flag is absent. Dropping `False`
    — as this used to — meant the offline engine honoured `async_scheduling: false`
    from the yaml while the online server quietly ran with it on, i.e. the two
    backends were not running the same configuration.

    Example (datalab-to/chandra-ocr-2):
      {max_model_len: 60000, gpu_memory_utilization: 0.95, trust_remote_code: True,
       async_scheduling: False}
      -> --max-model-len 60000 --gpu-memory-utilization 0.95 --trust-remote-code
         --no-async-scheduling
    """
    flags = []
    for key, val in engine_args.items():
        if key in _SERVE_FLAG_SKIP or val is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            flags.append(flag if val else "--no-" + key.replace("_", "-"))
        elif isinstance(val, (dict, list)):
            flags += [flag, json.dumps(val)]
        else:
            flags += [flag, str(val)]
    return flags


def _free_port() -> int:
    """Reserve an ephemeral port by binding and immediately releasing it. There is
    a small race before `vllm serve` binds it, which _verify_served_model() closes
    by proving the thing that answered is the server we launched."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((ONLINE_HOST, 0))
        return s.getsockname()[1]


def _verify_served_model(base_url: str, expected_model: str, logger) -> None:
    """
    Confirm the healthy server on this port is serving OUR model.

    Without this, any pre-existing listener (a stale serve from a crashed run,
    another job) satisfies /health and the entire run is generated by the wrong
    model with no error anywhere.
    """
    with httpx.Client(timeout=30.0) as client:
        data = client.get(f"{base_url}/v1/models").json()
    served = [m.get("id") for m in data.get("data", [])]
    if expected_model not in served:
        raise RuntimeError(
            f"the server on {base_url} is serving {served}, not {expected_model!r}. "
            "Refusing to send requests to a model we did not launch."
        )
    logger.info(f"verified server is serving {expected_model!r}")


def _venv_nvidia_lib_dirs(vllm_bin: str) -> list:
    """
    When `vllm_bin` is an absolute path into a specific venv's `bin/` (rather
    than the ambient `vllm` on PATH), that venv's own pip-installed
    `nvidia-*-cuXX` wheels are typically NOT on the dynamic linker's search
    path — only `import torch`'s own preloading puts them there, and a bare
    `subprocess.Popen([vllm_bin, ...])` skips that entirely. Symptom: `vllm
    serve` dies immediately with `ImportError: libcudart.so.13: cannot open
    shared object file` (or libnccl/libcublas/...) even though the .so is
    sitting right there in `site-packages/nvidia/<pkg>/lib`. Confirmed live
    against `/environments/gemma4_new` on 2026-08-12 (CUDA 13 torch build,
    driver reports CUDA 13.0) — fixed by prepending exactly these dirs to
    LD_LIBRARY_PATH. Returns [] for a bare "vllm" (PATH-resolved, ambient env
    already correct — this is what chandra/qwen use, unchanged).
    """
    bin_path = Path(vllm_bin)
    if not bin_path.is_absolute():
        return []
    venv_root = bin_path.parent.parent
    site_packages = list(venv_root.glob("lib/python3.*/site-packages"))
    if not site_packages:
        return []
    nvidia_root = site_packages[0] / "nvidia"
    if not nvidia_root.is_dir():
        return []
    return [str(p) for p in sorted(nvidia_root.glob("*/lib")) if p.is_dir()]


def _launch_server(engine_args: dict, gpus: str, stop_event, logger, vllm_bin: str = "vllm"):
    """
    Launch `vllm serve` as a subprocess and block until /health is OK and the
    served model matches. Returns (Popen, base_url).

    `gpus` is this rank's device string ("0,1,2,3"), handed to the child via
    CUDA_VISIBLE_DEVICES — which is what makes one pool of N ranks able to run N
    servers on disjoint GPU groups with no other coordination.

    `vllm_bin` picks which `vllm` executable to launch — defaults to the bare
    name (resolved off PATH, i.e. whatever env this process is already
    running in). Pass an absolute path (e.g.
    "/environments/gemma4_new/bin/vllm") to serve a model from a DIFFERENT
    venv than the one driving the pipeline — the only requirement is that
    `vllm serve` work as a standalone terminal command in that venv; see
    `_venv_nvidia_lib_dirs` for the one extra step (LD_LIBRARY_PATH) that
    needs, which is handled automatically here.

    All of engine_args (max_model_len, gpu_memory_utilization,
    limit_mm_per_prompt, trust_remote_code, async_scheduling,
    allowed_local_media_path, ...) is translated to serve flags;
    tensor-parallel-size is derived from `gpus`, falling back to engine_args when
    no GPUs are pinned.

    Cleans up after itself on every failure path: a Popen'd server that outlives
    the worker that launched it holds its GPUs until someone notices.
    """
    model = engine_args["model"]
    port = ONLINE_PORT or _free_port()
    base_url = f"http://{ONLINE_HOST}:{port}"

    cmd = [
        vllm_bin, "serve", str(model),
        "--host", ONLINE_HOST,
        "--port", str(port),
    ]
    # NOTE: --allowed-local-media-path is not appended here. As of v1.8 it is a
    # normal entry in engine_args (VLLMConfig.get_engine_args derives it from
    # allowed_media_roots) and reaches the server through the generic translation
    # below — the same value the in-process engine gets and the same value the
    # stage validates paths against, from one source, so they cannot drift.
    n_gpus = len(gpus.split(",")) if gpus else 0
    if n_gpus > 0:
        cmd += ["--tensor-parallel-size", str(n_gpus)]
    elif engine_args.get("tensor_parallel_size"):
        cmd += ["--tensor-parallel-size", str(engine_args["tensor_parallel_size"])]
    cmd += _engine_args_to_serve_flags(engine_args)

    env = os.environ.copy()
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["NCCL_NET_PLUGIN"] = ""
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    extra_lib_dirs = _venv_nvidia_lib_dirs(vllm_bin)
    if extra_lib_dirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(extra_lib_dirs + ([existing] if existing else []))

    logger.info(f"launching: {' '.join(cmd)}  (CUDA_VISIBLE_DEVICES={gpus})")
    proc = subprocess.Popen(cmd, env=env)

    try:
        deadline = time.time() + ONLINE_HEALTH_TIMEOUT_S
        with httpx.Client(timeout=5.0) as probe:
            while True:
                # Shutdown during a half-hour model load has to be honoured, or
                # Ctrl-C would hang for the rest of the load. The framework hands
                # init_fn the group's stop_event for exactly this.
                if stop_event is not None and stop_event.is_set():
                    raise RuntimeError(
                        "shutdown requested while waiting for `vllm serve` to come up"
                    )
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"vllm serve exited during startup with code {proc.returncode}"
                    )
                if time.time() > deadline:
                    raise RuntimeError(
                        f"vllm serve did not become healthy within "
                        f"{ONLINE_HEALTH_TIMEOUT_S}s"
                    )
                try:
                    if probe.get(f"{base_url}/health").status_code == 200:
                        logger.info(f"server healthy at {base_url}")
                        _verify_served_model(base_url, str(model), logger)
                        return proc, base_url
                except httpx.HTTPError:
                    pass
                time.sleep(2.0)
    except BaseException:
        _terminate_server(proc, logger)
        raise


def _terminate_server(proc, logger):
    """terminate() -> wait(timeout) -> kill() fallback."""
    if proc is None or proc.poll() is not None:
        return
    logger.info("terminating vllm serve subprocess")
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        logger.warning("serve did not exit in 30s — killing")
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------------------
# The WorkerSpec triple
# ---------------------------------------------------------------------------


def online_init(args, rank: int, device):
    """
    `init_fn`: one `vllm serve` subprocess, healthy and verified, plus the HTTP
    client that will drive it.

    Returns only once the server answers /health and reports our model —
    `vllm serve` does not open its port until the engine (weights included) has
    finished loading, so this is genuinely "ready", not a guess. The pool latches
    `ServiceStatus.ready` when every rank gets here.
    """
    logger = get_logger(args.logs_dir, f"vllm_online_{rank}")
    engine_args = dict(args.engine_args)
    gpus = device if device is not None else getattr(args, "gpus", "") or ""
    vllm_bin = getattr(args, "vllm_bin", None) or "vllm"

    proc, base_url = _launch_server(engine_args, gpus, args.stop_event, logger, vllm_bin=vllm_bin)
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(ONLINE_REQUEST_TIMEOUT_S),
        limits=httpx.Limits(max_connections=ONLINE_MAX_CONNECTIONS,
                            max_keepalive_connections=ONLINE_MAX_CONNECTIONS),
    )
    return {
        "proc": proc,
        "base_url": base_url,
        "model": str(engine_args["model"]),
        "client": client,
        "logger": logger,
    }


async def online_call(ctx, prompt: dict, params):
    """
    `call_fn`: one /v1/chat/completions round trip.

    Async, which is what tells `core/service.py` to run an event loop and hold
    `max_inflight_per_worker` of these at once — the continuous batching this
    backend exists for.

    The three outcomes, in the framework's own vocabulary:

      return          the server answered; `stats` is its own `usage` object,
                      already counted, so nothing is re-derived.
      InputRejected   the server rejected THIS request (4xx). Deterministic:
                      retrying re-sends the identical body, so the stage fails it
                      once and logs the server's own explanation.
      BackendGone     the subprocess is dead. The worker drains and exits; the
                      pool decides whether to replace it.
      anything else   infrastructure (a transport blip, a 5xx while the server is
                      still up). Retried on the infra budget, never FAILED.
    """
    logger = ctx["logger"]
    try:
        payload = _build_chat_payload(ctx["model"], prompt, params)
    except Exception as e:
        # Building the body failed. Nothing left here is per-item (v1.8 moved the
        # structured_outputs check to startup), so this is a configuration or
        # wire-format problem that would fail every image identically — let it
        # surface as infra, which retries and then trips the consecutive-infra
        # fuse, rather than as a rejection that fails paths one at a time.
        raise RuntimeError(
            f"could not build the request body ({type(e).__name__}: {e})"
        ) from e

    resp = await ctx["client"].post(f"{ctx['base_url']}/v1/chat/completions",
                                    json=payload)
    if resp.status_code >= 500 or resp.status_code == 404:
        # A 5xx (or a 404 from a server that has torn its routes down) means the
        # server, not the request. If the subprocess is gone it is terminal for
        # this worker; if it is still up, treat it as a retryable blip.
        body = _error_body(resp)
        if ctx["proc"].poll() is not None:
            raise BackendGone(
                f"`vllm serve` exited (code {ctx['proc'].returncode}); "
                f"HTTP {resp.status_code}: {body}"
            )
        raise RuntimeError(f"HTTP {resp.status_code} from vllm serve: {body}")
    if resp.status_code >= 400:
        message = _error_body(resp)
        logger.error(f"request rejected ({resp.status_code}): {message}")
        raise InputRejected(
            f"vllm serve rejected this request (HTTP {resp.status_code}): {message}"
        )

    data = resp.json()
    choice = data["choices"][0]
    return CallResult(
        choice["message"]["content"],
        stats_from_usage(
            data.get("usage"),
            finish_reason=choice.get("finish_reason"),
            stop_reason=choice.get("stop_reason"),
        ),
    )


async def online_term(ctx) -> None:
    """`term_fn`: close the client, then take the GPUs back."""
    try:
        await ctx["client"].aclose()
    except Exception:
        pass
    _terminate_server(ctx["proc"], ctx["logger"])


# ---------------------------------------------------------------------------
# Public API for stages that need a bare `vllm serve` subprocess without the
# rest of the online WorkerSpec (its call-per-request /v1/chat/completions
# contract, `max_inflight` semaphore, etc.) — e.g. a backend="function" stage
# that drives the server with its own request shape (see
# project_specific_translation.py). Both are thin wrappers so there is only
# one implementation of "launch/terminate a `vllm serve` subprocess".
# ---------------------------------------------------------------------------

def launch_vllm_serve(engine_args: dict, gpus: str, stop_event, logger, vllm_bin: str = "vllm"):
    """Launch `vllm serve` and block until healthy. Returns (Popen, base_url).
    See `_launch_server` for the full contract."""
    return _launch_server(engine_args, gpus, stop_event, logger, vllm_bin=vllm_bin)


def terminate_vllm_serve(proc, logger) -> None:
    """Terminate a subprocess returned by `launch_vllm_serve`."""
    _terminate_server(proc, logger)
