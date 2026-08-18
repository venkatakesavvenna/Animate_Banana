"""
§8 scenario 9: the wiring sweep. No GPUs, no vLLM engine, no pipeline run.

Checks that every consumer of the v1.9.2 API actually resolves:
  - StageWeaver's locked signatures are unchanged and cli() still matches them
  - every example / parent-repo init_fn + term_fn imports and has the right shape
  - the two-hop channel + adopt() path still yields a readable status
  - no module still references a deleted name
"""
import ast
import importlib
import inspect
import multiprocessing as mp
import pathlib
import sys

ING = "/fsxvision_new/srihari.bandarupalli/Patram-Data-Engine/dependencies/Patram-Ingest"
SW = "/fsxvision_new/srihari.bandarupalli/Patram-Data-Engine/dependencies/StageWeaver"
PARENT = "/fsxvision_new/srihari.bandarupalli/Patram-Data-Engine"
for pth in (f"{ING}/src", f"{SW}/src", PARENT, f"{PARENT}/src"):
    sys.path.insert(0, pth)

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def sweep_signatures():
    print("=== locked signatures")
    from vision_ingest.drivers.cli import cli
    params = list(inspect.signature(cli).parameters)
    expected = ['args', 'prompt_validation_object', 'channel', 'router_coordinator',
                'stage_name', 'run_specific_log_path_in_args', 'shutdown_event']
    check("cli() signature unchanged", params == expected, str(params))

    import stageweaver
    check("StageWeaver still exports ServiceChannel / MAX_STAGE_NAME_BYTES",
          hasattr(stageweaver, "ServiceChannel") and hasattr(stageweaver, "MAX_STAGE_NAME_BYTES"),
          f"version={stageweaver.__version__}")
    check("StageWeaver bumped to 1.1.1", stageweaver.__version__ == "1.1.1",
          stageweaver.__version__)


def sweep_init_fns():
    print("=== init_fn / term_fn call sites")
    targets = [
        # (module path or file, init name, term name)
        (f"{ING}/examples/vllm_testing/main.py", "model_init", "model_term"),
        (f"{ING}/examples/gemma_testing/main.py", "model_init", "model_term"),
        (f"{ING}/examples/chandra_testing/main_test.py", "model_init", "model_term"),
        (f"{ING}/examples/html_render/main.py", "pool_init", "pool_term"),
        (f"{PARENT}/src/digital_twin/project_specific/project_specific_layout.py",
         "init_fn", "termination_fn"),
        (f"{PARENT}/src/digital_twin/project_specific/project_specific_html_gen.py",
         "init_fn", "termination_fn"),
        (f"{PARENT}/src/digital_twin/project_specific/project_specific_translation.py",
         "init_fn", "termination_fn"),
        (f"{PARENT}/src/digital_twin/project_specific/project_specific_verifier.py",
         "init_fn", "termination_fn"),
        (f"{PARENT}/src/digital_twin/project_specific/project_specific_render.py",
         "init_fn", "termination_fn"),
        (f"{PARENT}/src/captioning_dataset_1/stage_fns.py", "model_init", "model_term"),
    ]
    dead = ("VLLMService", "VLMHealth", "FunctionService", "attach_process",
            "attach_worker_process", "set_health", "lazy_loading",
            "vllm_module.health", "vlm_service", "timed_worker",
            "debug_big_payload")
    for path, init_name, term_name in targets:
        rel = path.replace(PARENT + "/", "").replace(ING + "/", "ingest:")
        src = pathlib.Path(path).read_text()
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            check(f"{rel} parses", False, str(e)); continue
        funcs = {n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        ok_shape = init_name in funcs and term_name in funcs
        check(f"{rel}: {init_name}/{term_name} present", ok_shape, str(sorted(funcs))[:90])

        # dead names must not appear in CODE (docstrings/comments may cite history)
        code_lines = []
        for i, line in enumerate(src.split("\n"), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            code_lines.append((i, line))
        offenders = []
        # crude but effective: strip all string literals, then look for dead names
        stripped_src = src
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                seg = ast.get_source_segment(src, node)
                if seg:
                    stripped_src = stripped_src.replace(seg, "")
        for d in dead:
            if d in stripped_src:
                offenders.append(d)
        check(f"{rel}: no dead API in executable code", not offenders, str(offenders))

        # init_fn signature is (args, logger, channel)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == init_name:
                argnames = [a.arg for a in node.args.args]
                check(f"{rel}: {init_name}(args, logger, channel)",
                      argnames == ["args", "logger", "channel"], str(argnames))


def sweep_imports():
    print("=== module imports")
    mods = [
        "vision_ingest.core", "vision_ingest.core.service", "vision_ingest.core.status",
        "vision_ingest.core.wire", "vision_ingest.core.channel", "vision_ingest.core.task",
        "vision_ingest.core.shards",
        "vision_ingest.drivers.cli", "vision_ingest.modules.prediction",
        "vision_ingest.modules.writer", "vision_ingest.modules.recovery",
        "vision_ingest.vllm_module", "vision_ingest.vllm_module.specs",
        "vision_ingest.vllm_module.online_worker", "vision_ingest.vllm_module.async_worker",
        "vision_ingest.vllm_module.retry_validate", "vision_ingest.vllm_module.vllm_config",
        "vision_ingest.utils.pipeline_metrics",
    ]
    for m in mods:
        try:
            importlib.import_module(m)
            check(f"import {m}", True)
        except Exception as e:
            check(f"import {m}", False, f"{type(e).__name__}: {e}")

    # deleted modules must be gone
    for m in ("vision_ingest.vllm_module.health", "vision_ingest.vllm_module.vlm_service",
              "vision_ingest.vllm_module.timed_worker", "vision_ingest.vllm_module.wire"):
        try:
            importlib.import_module(m)
            check(f"{m} is gone", False, "still importable")
        except ImportError:
            check(f"{m} is gone", True)


def sweep_specs():
    print("=== vLLM specs build without a GPU")
    from vision_ingest.vllm_module.specs import vllm_spec, vllm_online_spec, vllm_async_spec
    from vision_ingest.core.service import ON_DEATH_ABORT
    ea = {"model": "some/model", "max_model_len": 4096}
    for backend, builder in (("online", vllm_online_spec), ("async", vllm_async_spec)):
        spec = vllm_spec(backend, engine_args=ea, gpus="0,1", max_inflight=16,
                         logs_dir="/tmp")
        check(f"{backend}: spec builds", spec is not None)
        check(f"{backend}: call_fn is async", spec.is_async())
        check(f"{backend}: max_inflight honoured", spec.resolved_max_inflight() == 16)
        check(f"{backend}: defaults to abort",
              spec.resolved_on_worker_death() == ON_DEATH_ABORT)
        check(f"{backend}: 1 worker by default", spec.resolved_n_workers() == 1)
        check(f"{backend}: args are JSON-able",
              set(spec.args) == {"engine_args", "gpus", "logs_dir"}, str(sorted(spec.args)))
    two = vllm_online_spec(engine_args=ea, gpus="", max_inflight=8, logs_dir="/tmp",
                           devices=["0,1,2,3", "4,5,6,7"])
    check("two servers => n_workers=2", two.resolved_n_workers() == 2)
    check("rank 1 gets its own GPU group", two.device_for(1) == "4,5,6,7")
    try:
        vllm_spec("batch", engine_args=ea, gpus="", max_inflight=1)
        check("backend='batch' is refused", False)
    except ValueError as e:
        check("backend='batch' is refused", "retired" in str(e))
    # pickling: a spec must survive spawn
    import pickle
    for backend in ("online", "async"):
        spec = vllm_spec(backend, engine_args=ea, gpus="0", max_inflight=4, logs_dir="/tmp")
        try:
            pickle.loads(pickle.dumps(spec))
            check(f"{backend}: spec pickles (spawn-safe)", True)
        except Exception as e:
            check(f"{backend}: spec pickles (spawn-safe)", False, str(e))


def _leaf(channel, out):
    from vision_ingest.core.channel import ServiceChannel
    ch = ServiceChannel.adopt(channel)
    st = ch.health
    out.put({
        "adopted_type": type(ch).__name__,
        "status_present": st is not None,
        "ready": bool(st and st.is_ready()),
        "reason": st.failure_reason() if st else "no-status",
        "resp_queue_ok": ch.response_queue("s1") is not None,
        "extras_keys": sorted(ch.extras),
    })


def sweep_two_hop():
    print("=== two-hop channel + adopt() with a real ServiceStatus")
    from stageweaver.datamodels.channel import ServiceChannel as SWChannel
    from vision_ingest.core.status import ServiceStatus
    st = ServiceStatus()
    st.mark_ready()
    st.beat()
    ch = SWChannel(request_queue=mp.Queue(),
                   response_queues={"s1": mp.Queue()},
                   stop_event=mp.Event(),
                   extras={"health": st})
    out = mp.Queue()
    p = mp.Process(target=_leaf, args=(ch, out))
    p.start()
    res = out.get(timeout=60)
    p.join(timeout=10)
    check("StageWeaver channel adopts into ours", res["adopted_type"] == "ServiceChannel")
    check("ServiceStatus survives spawn", res["status_present"])
    check("ready readable in the leaf", res["ready"], str(res))
    check("failure_reason readable in the leaf", res["reason"] is None, str(res["reason"]))
    check("response queue resolves", res["resp_queue_ok"])
    check("extras carries only 'health'", res["extras_keys"] == ["health"],
          str(res["extras_keys"]))


if __name__ == "__main__":
    mp.set_start_method("spawn")
    sweep_signatures()
    sweep_imports()
    sweep_specs()
    sweep_init_fns()
    sweep_two_hop()
    print()
    if FAIL:
        print(f"FAILURES ({len(FAIL)}):")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("ALL WIRING CHECKS PASSED")
