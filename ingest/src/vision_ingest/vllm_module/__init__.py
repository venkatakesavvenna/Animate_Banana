from .vllm_config import VLLMConfig
from .specs import vllm_async_spec, vllm_online_spec, vllm_spec
from .online_worker import launch_vllm_serve, terminate_vllm_serve
from vision_ingest.core.wire import VLLMRequest, VLLMResponse

# NOTE (v1.9.2): `VLLMService` and `VLMHealth` used to be re-exported from here.
# Both are gone. vLLM is a `WorkerSpec` now (`specs.py`), run by the one
# `core.service.WorkerPool` that runs every other backend, and liveness is a
# backend-agnostic `core.status.ServiceStatus` the pool publishes itself — so
# neither the service class nor a health object named after one backend has
# anything left to be. `VLMServiceDown` moved to `core.status` with the rest of
# the status machinery; it is `cli()`/`prediction.py`'s exception, not vLLM's.
# See docs/v1.9.2_changes.md.
#
# NOTE (v1.9.1): `Shm` used to be re-exported from here too. The shared-memory
# subsystem is gone — since v1.8 a request is paths + text and a response is text
# + stats, so there was no longer a payload big enough to justify moving it out
# of the mp.Queue. See docs/archive/shm_readme.md / exit_hang_postmortem.md for
# the design it replaced (the exit-hang lesson — cancel_join_thread() in every
# process that put() to a queue — is still live, and still implemented).
__all__ = [
    "VLLMConfig",
    "vllm_spec",
    "vllm_online_spec",
    "vllm_async_spec",
    "launch_vllm_serve",
    "terminate_vllm_serve",
    "VLLMRequest",
    "VLLMResponse",
]
