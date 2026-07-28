from __future__ import annotations

from scripts.run_parallel_test_matrix import build_jobs


def test_dynamic_matrix_defines_granular_single_gpu_jobs():
    jobs = build_jobs()

    one_gpu_jobs = [job for job in jobs if job.gpu_slots == 1]
    names = {job.name for job in one_gpu_jobs}

    assert len(one_gpu_jobs) >= 16
    assert "gpu-qwen3-load" in names
    assert "gpu-qwen3-moe-load" in names
    assert "gpu-qwen-eval-artifacts" in names
    assert "gpu-gemma-eval-artifacts" in names


def test_dynamic_matrix_defines_four_two_gpu_jobs():
    jobs = build_jobs()

    two_gpu_jobs = [job for job in jobs if job.gpu_slots == 2]

    assert [job.name for job in two_gpu_jobs] == [
        "gpu-qwen-ddp",
        "gpu-gemma-ddp",
        "gpu-qwen-dataparallel",
        "gpu-gemma-dataparallel",
    ]


def test_dynamic_matrix_keeps_cpu_jobs_gpu_free():
    jobs = build_jobs()

    cpu_jobs = [job for job in jobs if job.runtime == "cpu"]

    assert cpu_jobs
    assert all(job.gpu_slots == 0 for job in cpu_jobs)
