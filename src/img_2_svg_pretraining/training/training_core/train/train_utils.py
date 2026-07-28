import fnmatch
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-layer (param-group) learning rate utilities
# ---------------------------------------------------------------------------

def build_param_groups(
    model: torch.nn.Module,
    param_group_specs: List[Dict],
    base_lr: float,
    base_weight_decay: float = 0.0,
) -> Tuple[List[Dict], List[str]]:
    """Partition model parameters into optimizer param groups driven by config.

    Each entry in ``param_group_specs`` must have at minimum:
      - ``pattern``:      glob or regex matched against parameter names
      - ``lr``:           learning rate for matched params
      - ``weight_decay``: (optional) weight decay; falls back to ``base_weight_decay``

    Parameters not matched by any spec fall into a default group at ``base_lr``
    and ``base_weight_decay``.

    Returns:
        (param_groups, unmatched_names) — list ready for torch.optim, plus
        names of params not covered by any spec (for debugging).
    """
    matched: Dict[str, int] = {}  # param_name -> spec_index
    for i, spec in enumerate(param_group_specs):
        pattern = spec["pattern"]
        for name, _ in model.named_parameters():
            if name in matched:
                continue
            if fnmatch.fnmatch(name, pattern) or re.search(pattern, name):
                matched[name] = i

    # Build per-spec buckets: decay / no-decay split inside each spec
    buckets: Dict[int, Dict] = {}
    for i, spec in enumerate(param_group_specs):
        lr = float(spec["lr"])
        wd = float(spec.get("weight_decay", base_weight_decay))
        buckets[i] = {"decay_params": [], "nodecay_params": [], "lr": lr, "wd": wd}

    # Default bucket for unmatched params
    DEFAULT = -1
    buckets[DEFAULT] = {
        "decay_params": [], "nodecay_params": [],
        "lr": base_lr, "wd": base_weight_decay,
    }

    unmatched_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        spec_idx = matched.get(name, DEFAULT)
        if spec_idx == DEFAULT:
            unmatched_names.append(name)
        bucket = buckets[spec_idx]
        # Apply weight decay only to 2-D+ tensors (skip biases, norms)
        if param.ndim >= 2 and bucket["wd"] > 0:
            bucket["decay_params"].append(param)
        else:
            bucket["nodecay_params"].append(param)

    param_groups = []
    for spec_idx in sorted(buckets.keys()):
        b = buckets[spec_idx]
        label = param_group_specs[spec_idx]["pattern"] if spec_idx != DEFAULT else "default"
        if b["decay_params"]:
            param_groups.append({
                "params": b["decay_params"],
                "lr": b["lr"],
                "weight_decay": b["wd"],
                "_group_label": f"{label}/decay",
            })
        if b["nodecay_params"]:
            param_groups.append({
                "params": b["nodecay_params"],
                "lr": b["lr"],
                "weight_decay": 0.0,
                "_group_label": f"{label}/no_decay",
            })

    total = sum(len(g["params"]) for g in param_groups)
    _logger.info(
        "[ParamGroups] %d groups, %d trainable tensors (%d unmatched → default group)",
        len(param_groups), total, len(unmatched_names),
    )
    for g in param_groups:
        _logger.info(
            "  group=%s  lr=%.2e  wd=%.2e  params=%d",
            g["_group_label"], g["lr"], g["weight_decay"], len(g["params"]),
        )
    return param_groups, unmatched_names


def build_multi_dataset(
    dataset_specs,
    get_dataset_fn: Callable,
):
    """Build one dataset bundle per spec entry and collect weights.

    Returns:
        (datasets, weights) — list of dataset bundles and corresponding float weights.
    """
    datasets = []
    weights = []
    for spec in dataset_specs:
        ds = get_dataset_fn(spec)
        datasets.append(ds)
        weights.append(float(spec.get("weight", 1.0)))
    return datasets, weights


# ---------------------------------------------------------------------------
# Weighted map-style mixing
# ---------------------------------------------------------------------------

class WeightedMixDataset(Dataset):
    """Map-style dataset that samples from multiple datasets proportionally to their weights.

    Builds a shuffled flat index at construction time:
      dataset i contributes round(weight_i / sum(weights) * total_len) entries.
    Total length equals sum of individual dataset lengths (same as ConcatDataset).
    """

    def __init__(
        self,
        datasets: List[Dataset],
        weights: List[float],
        seed: int = 42,
    ):
        assert len(datasets) == len(weights) and len(datasets) > 0
        self._datasets = datasets
        total = sum(len(d) for d in datasets)
        w = np.array(weights, dtype=np.float64)
        w = w / w.sum()

        rng = np.random.RandomState(seed)
        # Build flat index list: (dataset_idx, sample_idx within that dataset)
        index: List[tuple] = []
        for di, (ds, frac) in enumerate(zip(datasets, w)):
            n = max(1, round(frac * total))
            # Sample with replacement from this dataset to fill its quota
            sample_idxs = rng.randint(0, len(ds), size=n)
            index.extend((di, int(si)) for si in sample_idxs)

        # Shuffle the combined index
        perm = rng.permutation(len(index))
        self._index = [index[i] for i in perm]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        di, si = self._index[idx]
        return self._datasets[di][si]


# ---------------------------------------------------------------------------
# Iterable mixture (streaming / IterableDataset)
# ---------------------------------------------------------------------------

class IterableDatasetMixture(torch.utils.data.IterableDataset):
    """Infinitely iterates over a mixture of IterableDatasets.

    Adapted from allenai/molmo olmo/data/iterable_dataset_mixture.py.
    Supports two strategies:
      'weighted'   — probabilistic sampling via mixture_rates
      'stratified' — always sample the most under-represented dataset
    """

    def __init__(
        self,
        datasets: List[torch.utils.data.IterableDataset],
        mixture_rates: Optional[List[float]] = None,
        seed: int = 0,
        strategy: str = "weighted",
        total_samples: Optional[int] = None,
    ):
        self.datasets = list(datasets)
        if mixture_rates is not None:
            rates = np.array(mixture_rates, dtype=np.float32)
            self.mixture_rates = rates / rates.sum()
        else:
            self.mixture_rates = np.ones(len(datasets), dtype=np.float32) / len(datasets)
        self.seed = seed
        self.strategy = strategy
        self.total_samples = total_samples
        self._iters: List[Any] = []

    def _next_dataset_idx(self, rng: np.random.RandomState, counts: np.ndarray) -> int:
        if len(self.datasets) == 1:
            return 0
        if self.strategy == "stratified":
            total = counts.sum()
            if total == 0:
                return int(np.argmax(self.mixture_rates))
            return int(np.argmax(np.abs(counts / total - self.mixture_rates)))
        return int(rng.choice(len(self.datasets), p=self.mixture_rates))

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        counts = np.zeros(len(self.datasets), dtype=np.int64)
        iters = [iter(ds) for ds in self.datasets]
        emitted = 0
        while self.total_samples is None or emitted < self.total_samples:
            di = self._next_dataset_idx(rng, counts)
            try:
                item = next(iters[di])
            except StopIteration:
                iters[di] = iter(self.datasets[di])
                item = next(iters[di])
            counts[di] += 1
            emitted += 1
            yield item


# ---------------------------------------------------------------------------
# High-level builder
# ---------------------------------------------------------------------------

def build_mixed_dataset(
    dataset_bundles: List[Dict],
    weights: List[float],
    split: str,
    mixing_cfg: Optional[Dict] = None,
    seed: int = 42,
):
    """Build a single mixed dataset from per-bundle dicts.

    Args:
        dataset_bundles: list of dicts with keys 'train_dataset', 'val_dataset'
        weights: per-bundle weight (same order as dataset_bundles)
        split: 'train' or 'val'
        mixing_cfg: optional dict with 'strategy' and/or 'seed' keys
        seed: fallback seed if not in mixing_cfg

    Returns:
        A single Dataset (ConcatDataset or WeightedMixDataset).
    """
    key = f"{split}_dataset"
    raw_datasets = [b[key] for b in dataset_bundles]

    strategy = (mixing_cfg or {}).get("strategy", "concat")
    mix_seed = int((mixing_cfg or {}).get("seed", seed))

    weights_uniform = all(abs(w - weights[0]) < 1e-6 for w in weights)

    if strategy == "concat" or weights_uniform:
        if not weights_uniform:
            _logger.info(
                "[Mixing] Non-uniform weights ignored because strategy=concat; "
                "falling back to ConcatDataset."
            )
        return ConcatDataset(raw_datasets)

    # Check if any dataset is iterable
    any_iterable = any(isinstance(d, torch.utils.data.IterableDataset) for d in raw_datasets)
    if any_iterable:
        _logger.info("[Mixing] Using IterableDatasetMixture (strategy=%s)", strategy)
        return IterableDatasetMixture(
            datasets=raw_datasets,
            mixture_rates=weights,
            seed=mix_seed,
            strategy=strategy,
            total_samples=(mixing_cfg or {}).get("total_samples"),
        )

    _logger.info("[Mixing] Using WeightedMixDataset (strategy=%s, seed=%d)", strategy, mix_seed)
    return WeightedMixDataset(raw_datasets, weights, seed=mix_seed)
