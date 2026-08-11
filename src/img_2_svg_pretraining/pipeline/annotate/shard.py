"""Deterministic sample-id sharding for multiple annotators working in parallel.

`--shard i/n` needs no coordination file: everyone runs the same sort over the
same dataset root, so `partition_ids` alone is enough to guarantee disjoint
subsets across processes, machines, or people.
"""
from __future__ import annotations


def partition_ids(all_ids: list[str], n: int, i: int) -> list[str]:
    """The i-th of n disjoint, deterministic shards of `all_ids` (0-indexed)."""
    if n < 1:
        raise ValueError(f"shard count must be >= 1, got {n}")
    if not (0 <= i < n):
        raise ValueError(f"shard index {i} out of range for {n} shard(s)")
    return sorted(all_ids)[i::n]


def parse_shard(spec: str) -> tuple[int, int]:
    """Parse 'i/n' (1-indexed, as typed on a CLI) into 0-indexed (i, n)."""
    try:
        i_str, n_str = spec.split("/")
        i, n = int(i_str), int(n_str)
    except ValueError:
        raise ValueError(f"--shard expects 'i/n' (e.g. '1/4'), got '{spec}'") from None
    if not (1 <= i <= n):
        raise ValueError(f"--shard '{spec}': i must be between 1 and n")
    return i - 1, n
