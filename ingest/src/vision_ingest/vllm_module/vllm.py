# v1.9 §1: the fallback below is chosen AUTOMATICALLY when vLLM is not
# installed, instead of by hand-flipping a module constant.
#
# Every vLLM-adjacent import in this package bottoms out here — prediction.py,
# retry_validate.py and vllm_config.py both import
# SamplingParams / StructuredOutputsParams / LLM / RequestOutput from this
# module and nowhere else. So degrading gracefully here is what makes the whole
# chain importable on a CPU-only node that will never run vLLM: a chrome
# rendering or SAM3 worker (backend="function", see core/service.py) can now
# `import vision_ingest.drivers.cli` without a full vLLM install.
#
# `local_testing` is kept as a manual override for developing against the dummy
# classes on a machine that DOES have vLLM installed. Leave it False otherwise —
# the ImportError path handles the real "no vLLM here" case on its own.
local_testing = False

VLLM_AVAILABLE = False

if not local_testing:
    try:
        from vllm import SamplingParams, RequestOutput, LLM
        from vllm.sampling_params import StructuredOutputsParams
        VLLM_AVAILABLE = True
    except ImportError:
        # Not an error: a function-backend deployment has no reason to install
        # vLLM. Anything that actually needs a real engine (a vLLM WorkerSpec's
        # workers) will fail loudly at model-load time instead, which is the
        # right place for that failure to surface.
        pass

if not VLLM_AVAILABLE:
    from typing import List
    from tqdm import tqdm
    import time
    random_global_counter=0

    class SamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def __repr__(self):
            return f"SamplingParams({self.__dict__})"
    
    class StructuredOutputsParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def __repr__(self):
            return f"StructuredOutputsParams({self.__dict__})"

    class DummyOutput:
        def __init__(self, text=None, **kwargs):
            self.text = text
            self.__dict__.update(kwargs)

    class RequestOutput:
        def __init__(self, outputs=None, **kwargs):
            self.outputs = outputs if outputs is not None else []
            self.__dict__.update(kwargs)


    class LLM:
        def __init__(self, **engine_args):
            # Store engine args for compatibility
            self.engine_args = engine_args

        def generate(
            self,
            prompts: List[str],
            sampling_params: List[SamplingParams] = None,
        ) -> List[RequestOutput]:
            """
            Dummy generate method that echoes prompts.
            Returns one RequestOutput per prompt.
            """
            global random_global_counter
            results = []

            sampling_params = sampling_params or [None] * len(prompts)

            for prompt, sparams in tqdm(zip(prompts, sampling_params)):
                results.append(
                    RequestOutput(outputs=[DummyOutput(text=f"```json\n{{\"str\": \"[DUMMY COMPLETION] - {random_global_counter}\"}}```")])
                )
                random_global_counter+=1
            time.sleep(5)
            return results