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
# Main study.
#   exp1  absolute   -- one narrative at a time, every method in the bundle
#   exp2  tournament -- AB vs second-best, winner vs the original talk
# Selective cohort.
#   context  pairwise, blind, sides randomised
#   bench    pairwise, NOT blind, sides fixed (original left, corrected right)
PAIR_AXIS = {
    "context": ("context_condition", ("with_context", "without_context")),
    "bench":   ("verification_state", ("pre_verification", "verified")),
}
# Pairwise experiments whose sides are deliberately not randomised.
FIXED_SIDES = {"bench"}

ABSOLUTE_EXPERIMENTS = ("exp1",)

# The ranking tournament: first round between the first two methods, the
# winner then meets the third. Order matters and is fixed by the design.
TOURNAMENT = {"exp2": ("animatebanana", "qwen38", "talk")}

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
# exp1 asks about narration (NAS), so captions are on. exp2's talk has no
# narration track, so captions are off on every side to keep it symmetric.
SHOWS_CAPTIONS = {"exp1": True, "exp2": False, "context": True, "bench": True}

SCREEN = {"exp1": "absolute", "exp2": "tournament",
          "context": "pairwise", "bench": "pairwise"}


@dataclass
class StudyConfig:
    study_version: str = "pilot-1"
    state: str = "open"                       # open | paused | closed

    experiment_order: tuple = ("exp1", "exp2", "context", "bench")
    enabled: dict = field(default_factory=lambda: {
        "exp1": True, "exp2": True, "context": False, "bench": False})

    # Per participant, per experiment.
    samples_per_experiment: dict = field(default_factory=lambda: {
        "exp1": 10, "exp2": 10, "context": 15, "bench": 15})

    # Retire a sample once this many independent judgments exist, then draw a
    # replacement from the same stratification class. Reported as 5-6 in the
    # design discussion; kept configurable because the pool grows from 15 to
    # ~100 and the right number moves with it.
    judgments_per_sample: int = 6

    # Which fields must match when replacing a retired sample.
    stratum_fields: tuple = ("element_density", "connectivity_level", "has_raster")

    # Main study only: serve just the diagrams stamped with this day. None
    # means no day filtering (the selective cohort, or a single-day pilot).
    study_day: int | None = None

    attention_check_fraction: float = 0.0     # off until degraded stimuli exist
    calibration_pass_threshold: float = 0.7
    open_trial_ttl_seconds: int = 900

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
