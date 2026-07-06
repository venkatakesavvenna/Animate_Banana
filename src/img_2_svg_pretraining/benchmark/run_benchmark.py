"""Benchmark CLI: image -> TikZ inference (via plain transformers generation)
+ compile + score (DINO visual similarity, VLM-as-judge) for one model
against a sample set.

Run inside the project docker container (needs transformers, torch, and the
LaTeX/poppler toolchain used by viewer/compile.py):

    python -m img_2_svg_pretraining.benchmark.run_benchmark --model qwen-3.5-9b --limit 100

Pass multiple --gpus to data-parallelize: one full model replica is loaded
per GPU, in its own process, and the sample list is sharded across them
(e.g. --gpus 1,2,3,4,5,6,7 loads 7 replicas of the model, each churning
through 1/7th of the samples independently -- this is throughput scaling,
not sharding one copy of the model across GPUs for memory capacity).

Results are written incrementally to benchmark/results/<model>.jsonl (one
JSON record per line, appended as each sample/batch finishes -- not held in
memory for the whole run, so this scales to arbitrarily large sample sets).
With multiple --gpus, each worker streams to its own private
<model>.gpu<id>.jsonl (concurrent appends to one shared file are not safe on
this project's Lustre-backed /code mount -- see _append_records), which are
concatenated into the final <model>.jsonl once all workers finish. Line
order in the merged file follows GPU/shard order, not original sample order
(every record carries its own sample_id, so downstream consumers don't need
file order). See benchmark/README.md for the full workflow, including
running the judge as a separate pass after the model-under-test is unloaded
from GPU memory.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

DEFAULT_GPUS = "1,2,3,4,5,6,7"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model registry key (see benchmark/models.py)")
    parser.add_argument("--data-root", type=str, default="/code/data/test_extracted/test")
    parser.add_argument("--limit", type=int, default=100, help="number of samples to evaluate (default 100 for a quick pass; use --limit 0 for all)")
    parser.add_argument("--gpus", type=str, default=DEFAULT_GPUS,
                         help="comma-separated physical GPU ids (default excludes GPU 0). "
                              "One model replica is loaded per id listed here, each in its own "
                              "process, and the sample set is sharded across them.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1, help="samples per generation batch, per GPU replica (default 1 = sequential, matching prior behavior)")
    return parser.parse_args()


def _select_samples(discover_samples, data_root: Path, limit: int | None, seed: int = 0):
    samples = discover_samples(data_root)
    samples = [s for s in samples if s.image_path is not None]
    if limit is not None and limit < len(samples):
        random.Random(seed).shuffle(samples)
        samples = samples[:limit]
    return samples


def _ok_record(sample, result, latency_s) -> dict:
    return {
        "sample_id": sample.id,
        "image_path": str(sample.image_path),
        "gt_tex_path": str(sample.tex_path),
        "raw_output": result.raw_output,
        "generated_tex": result.tex,
        "compiled_ok": result.compiled_ok,
        "rendered_png_path": str(result.png_path) if result.png_path else None,
        "compile_log": result.compile_log if not result.compiled_ok else "",
        "latency_s": latency_s,
        "error": None,
    }


def _error_record(sample, error: Exception, latency_s) -> dict:
    return {
        "sample_id": sample.id,
        "image_path": str(sample.image_path),
        "gt_tex_path": str(sample.tex_path),
        "raw_output": None,
        "generated_tex": None,
        "compiled_ok": False,
        "rendered_png_path": None,
        "compile_log": "",
        "latency_s": latency_s,
        "error": str(error),
    }


def _append_records(out_path: Path, records: list[dict]) -> None:
    """Appends records to `out_path` as soon as they're ready, rather than
    accumulating them in memory for the whole run. `out_path` must be
    exclusive to the calling process -- concurrent appends to one shared file
    from multiple processes are NOT safe here: /code is a Lustre mount, and
    Lustre does not give O_APPEND the atomicity a local ext4 disk would,
    so concurrent writers can silently clobber each other's lines (confirmed:
    a 4-process test lost 2 whole workers' output with no error raised).
    That's why each GPU worker gets its own file (see _gpu_worker) instead of
    sharing one via a lock."""
    lines = "".join(json.dumps(r) + "\n" for r in records)
    with out_path.open("a") as f:
        f.write(lines)


def _process_shard(spec, samples, model_cache_dir: Path, batch_size: int, log_prefix: str,
                    total: int, out_path: Path) -> tuple[int, int]:
    """Loads one model replica on whatever GPU is visible in this process
    (CUDA_VISIBLE_DEVICES must already be set to a single physical GPU by the
    caller) and runs inference over `samples` sequentially/batched, appending
    each batch's records to `out_path` as it completes. Shared by both the
    single-GPU path and each multi-GPU worker process; `out_path` must not be
    written by any other concurrent process (see _append_records). Returns
    (n_done, n_compiled_ok) instead of the records themselves, since records
    are streamed to disk rather than held in memory."""
    from img_2_svg_pretraining.benchmark.infer import ModelRunner, run_inference, run_inference_batch

    print(f"{log_prefix} loading {spec.hf_repo} (attn_implementation={spec.attn_implementation})...")
    runner = ModelRunner(spec)
    print(f"{log_prefix} model loaded.")

    n_done = 0
    n_compiled_ok = 0
    try:
        for batch_start in range(0, len(samples), batch_size):
            batch = samples[batch_start:batch_start + batch_size]
            t0 = time.time()
            try:
                if batch_size == 1:
                    results = [run_inference(runner, batch[0].id, batch[0].image_path, model_cache_dir)]
                else:
                    results = run_inference_batch(
                        runner,
                        [s.id for s in batch],
                        [s.image_path for s in batch],
                        model_cache_dir,
                    )
                latency_s = time.time() - t0
                per_sample_latency = latency_s / len(batch)
                batch_records = [_ok_record(sample, result, per_sample_latency)
                                  for sample, result in zip(batch, results)]
            except Exception as e:  # noqa: BLE001 -- one bad batch shouldn't kill the run
                latency_s = time.time() - t0
                per_sample_latency = latency_s / len(batch)
                batch_records = [_error_record(sample, e, per_sample_latency) for sample in batch]

            _append_records(out_path, batch_records)
            n_done += len(batch_records)
            n_compiled_ok += sum(r["compiled_ok"] for r in batch_records)
            for i, record in enumerate(batch_records):
                print(f"{log_prefix} [{n_done - len(batch_records) + i + 1}/{total}] {record['sample_id']}: "
                      f"{'ok' if record['compiled_ok'] else 'FAILED'} ({record['latency_s']:.1f}s)")
    finally:
        runner.unload()

    return n_done, n_compiled_ok


def _gpu_worker(gpu_id: str, spec, samples, model_cache_dir: Path, batch_size: int,
                 total: int, out_path: Path, result_queue) -> None:
    """Entry point for a spawned worker process: pins this process to exactly
    one physical GPU before any CUDA-touching import (torch locks in device
    visibility at first import), then runs its shard of samples, appending
    results to its own private `out_path` as they complete (never shared with
    other workers -- see _append_records for why). Only the (n_done,
    n_compiled_ok) counts go back through the queue -- not the records
    themselves -- since spawned processes don't share memory with the parent
    and the records already live on disk."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    log_prefix = f"[gpu {gpu_id}]"
    counts = _process_shard(spec, samples, model_cache_dir, batch_size, log_prefix,
                             total=total, out_path=out_path)
    result_queue.put(counts)


def _run(args) -> Path:
    """All torch/transformers imports are deferred to here (or into worker
    processes for the multi-GPU path), since torch locks in device visibility
    at first CUDA-touching import -- CUDA_VISIBLE_DEVICES must be set before
    that happens, either in main() for the single-process path or inside each
    worker for the multi-GPU path."""
    from img_2_svg_pretraining.benchmark.models import get_model, MODELS
    from img_2_svg_pretraining.viewer.samples import discover_samples

    if args.model not in MODELS:
        raise SystemExit(f"Unknown model '{args.model}'. Known: {sorted(MODELS)}")

    results_dir = Path(__file__).parent / "results"
    render_cache_dir = Path(__file__).parent / "render_cache"

    spec = get_model(args.model)
    limit = None if args.limit == 0 else args.limit
    samples = _select_samples(discover_samples, Path(args.data_root), limit, args.seed)

    results_dir.mkdir(parents=True, exist_ok=True)
    render_cache_dir.mkdir(parents=True, exist_ok=True)
    model_cache_dir = render_cache_dir / args.model

    batch_size = max(1, args.batch_size)
    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]

    out_path = results_dir / f"{args.model}.jsonl"

    if len(gpu_ids) <= 1:
        print(f"Running inference for {args.model} ({spec.hf_repo}) over {len(samples)} samples "
              f"[CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}]")
        out_path.write_text("")  # truncate/create -- only this process writes to it, so plain append is safe
        n_done, n_ok = _process_shard(spec, samples, model_cache_dir, batch_size, log_prefix="",
                                       total=len(samples), out_path=out_path)
    else:
        # Data-parallel: one full model replica per GPU, each in its own
        # process, each churning through its own shard of samples. Each
        # worker appends to its OWN .jsonl file, never a shared one --
        # /code is a Lustre mount, and concurrent O_APPEND writers on Lustre
        # can silently drop whole writes (verified experimentally: a shared
        # write_lock + shared file still lost 2 of 4 workers' output with no
        # error). CUDA also requires "spawn" (not fork) so each child gets a
        # clean CUDA context instead of inheriting the parent's.
        print(f"Running inference for {args.model} ({spec.hf_repo}) over {len(samples)} samples "
              f"across {len(gpu_ids)} GPU replicas ({gpu_ids})")
        shards = [samples[i::len(gpu_ids)] for i in range(len(gpu_ids))]
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        workers = []
        shard_paths = []
        for gpu_id, shard in zip(gpu_ids, shards):
            if not shard:
                continue
            shard_path = results_dir / f"{args.model}.gpu{gpu_id}.jsonl"
            shard_path.write_text("")  # truncate/create -- exclusive to this worker
            shard_paths.append(shard_path)
            p = ctx.Process(
                target=_gpu_worker,
                args=(gpu_id, spec, shard, model_cache_dir, batch_size, len(shard), shard_path, result_queue),
            )
            p.start()
            workers.append(p)

        n_done = 0
        n_ok = 0
        for _ in workers:
            shard_done, shard_ok = result_queue.get()
            n_done += shard_done
            n_ok += shard_ok
        for p in workers:
            p.join()

        failed = [p for p in workers if p.exitcode not in (0, None)]
        if failed:
            raise RuntimeError(f"{len(failed)} GPU worker process(es) exited with non-zero status")

        # Merge each worker's private file into the single output file the
        # rest of the pipeline (score.py, viewer) expects. Plain line-level
        # concatenation -- no JSON re-parsing needed.
        with out_path.open("w") as merged:
            for shard_path in shard_paths:
                merged.write(shard_path.read_text())
                shard_path.unlink()

    print(f"Done: {n_ok}/{n_done} compiled OK. Wrote {out_path}")
    return out_path


def main():
    args = _parse_args()
    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if len(gpu_ids) <= 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    _run(args)


if __name__ == "__main__":
    main()
