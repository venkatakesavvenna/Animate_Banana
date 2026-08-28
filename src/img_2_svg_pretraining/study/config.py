"""Study configuration: everything tunable without touching code.

Loaded from yaml, then written into `study_config` as a version. Every trial
records the version it ran under, so changing a parameter mid-study cannot
silently reinterpret data already collected.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

# Which narrative attribute each pairwise experiment varies. The scheduler
# builds pairs by finding two narratives for the same (diagram, style) that
# differ on exactly this field -- so the moment the -K sweep, a baseline or the
# verified references land in a bundle, the arm turns on with no code change.
PAIR_AXIS = {
    "exp3": ("context_condition", ("with_context", "without_context")),
    "exp4": ("method", ("animatebanana", "baseline")),
    "exp5": ("verification_state", ("verified", "pre_verification")),
}

ABSOLUTE_EXPERIMENTS = ("exp1", "exp2")

# Captions are the ONLY difference between Experiment 1 and Experiment 2. Same
# media, same frames, caption layer switched off -- so the two rate pixel-
# identical visuals and their scores are directly comparable. (Our animations
# carry no burned-in narration, verified against the animation source, so there
# is nothing to crop.)
# Experiment 4 hides captions on BOTH sides. The end-to-end baseline produces
# no narration, so showing captions would leave one panel with a caption bar
# and the other with an empty one -- which un-blinds the comparison at a
# glance, before the participant has watched anything. Exp4's questions ask
# about preference, visual quality and pacing, so nothing is lost by making it
# a visuals-only comparison on both sides.
SHOWS_CAPTIONS = {"exp1": False, "exp2": True,
                  "exp3": True, "exp4": False, "exp5": True}

SCREEN = {"exp1": "absolute", "exp2": "absolute",
          "exp3": "pairwise", "exp4": "pairwise", "exp5": "verification"}


@dataclass
class StudyConfig:
    study_version: str = "pilot-1"
    state: str = "open"                       # open | paused | closed

    experiment_order: tuple = ("exp1", "exp2", "exp3", "exp4", "exp5")
    enabled: dict = field(default_factory=lambda: {
        "exp1": True, "exp2": True, "exp3": True, "exp4": True, "exp5": True})

    # Per participant, per experiment.
    samples_per_experiment: dict = field(default_factory=lambda: {
        "exp1": 10, "exp2": 10, "exp3": 10, "exp4": 10, "exp5": 10})

    # Retire a sample once this many independent judgments exist, then draw a
    # replacement from the same stratification class. Reported as 5-6 in the
    # design discussion; kept configurable because the pool grows from 15 to
    # ~100 and the right number moves with it.
    judgments_per_sample: int = 6

    # Which fields must match when replacing a retired sample.
    stratum_fields: tuple = ("element_density", "connectivity_level", "has_raster")

    attention_check_fraction: float = 0.0     # off until degraded stimuli exist
    calibration_pass_threshold: float = 0.7
    open_trial_ttl_seconds: int = 7200

    def enabled_experiments(self) -> list[str]:
        return [e for e in self.experiment_order if self.enabled.get(e)]

    def target_for(self, experiment: str) -> int:
        return int(self.samples_per_experiment.get(experiment, 10))

    def to_dict(self) -> dict:
        out = asdict(self)
        out["experiment_order"] = list(self.experiment_order)
        out["stratum_fields"] = list(self.stratum_fields)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "StudyConfig":
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in (data or {}).items() if k in known}
        for tuple_field in ("experiment_order", "stratum_fields"):
            if tuple_field in kwargs and kwargs[tuple_field] is not None:
                kwargs[tuple_field] = tuple(kwargs[tuple_field])
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> "StudyConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)
