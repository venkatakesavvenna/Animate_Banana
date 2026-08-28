"""Artifact paths for stage-batch handoff.

Every agent reads its input and writes its output as files under
`cache/<dataset>/`. Paths encode the lineage of models that produced the
artifact, so runs with different configs never collide and a re-run with one
agent swapped doesn't silently reuse the other's output.

Two lineage rules matter:

- Stage 2a (strategizer) and 2b (parser) are independent -- neither consumes
  the other -- so their paths omit each other's model. Re-running the parser
  after a prompt tweak leaves cached strategies valid, and vice versa. Only
  from the sequencer onward do the two lineages merge.
- `resolve` never silently picks among several candidates. `data_engine`'s
  render/judge stages glob `*/<id>.ext` and take `candidates[0]`, so with two
  upstream models you get whichever sorts first; here an ambiguous match is
  an error naming the alternatives.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ConfigError, PipelineConfig

CODE_SUFFIX = {"tikz": ".tex", "svg": ".svg"}


def _slug(value: str) -> str:
    """Filesystem-safe path component."""
    return str(value).replace("/", "-").replace(" ", "_")


@dataclass
class CachePaths:
    """Resolves artifact locations for one config."""

    cfg: PipelineConfig
    dataset: str

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> "CachePaths":
        return cls(cfg=cfg, dataset=cfg.dataset_root.name)

    # -- lineage components ----------------------------------------------

    def _model_for(self, agent: str) -> str:
        backend = self.cfg.agent(agent).backend
        if not backend:
            raise ValueError(f"agent '{agent}' has no backend configured")
        return _slug(self.cfg.backend_model(backend))

    @property
    def root(self) -> Path:
        return Path(self.cfg.cache_root) / self.dataset

    @property
    def code_lineage(self) -> str:
        """Stage-1 lineage. The integrator's model is included only when the
        integrator is enabled, so disabling it doesn't invalidate Stage-1
        artifacts that never involved it.

        `code_from` in the config overrides this entirely, pointing the run at
        diagram code some *other* config produced. That is what makes a
        stage-2/3-only comparison possible: hold stage 1 constant across
        models, and any difference in the animation is attributable to
        planning and design rather than to how each model chose to redraw the
        figure. It is also the practical route when stage 1a is the expensive
        part -- a 16K-token TikZ document is far slower to generate locally
        than the planning stages that follow it.
        """
        shared = self.cfg.raw.get("code_from")
        if shared:
            return _slug(str(shared))

        parts = [self._model_for("code_converter")]
        integrator = self.cfg.agents.get("raster_integrator")
        if integrator and integrator.option("enabled", False):
            # Stage 1b detects raster regions with an ordinary chat backend, so
            # its lineage component is just that model.
            parts.append(self._model_for("raster_integrator"))
        if self.cfg.target != "tikz":
            # The target belongs in the lineage because it changes the artifact,
            # not just its file extension: a TikZ document and an SVG one carry
            # completely different element ids. Without this the two share every
            # downstream path, so an SVG run reuses the TikZ XML as "cached" and
            # every sequence `focus` then references ids the SVG does not have
            # (observed: 0 of 14 ids in common).
            #
            # Appended only for non-tikz so every existing TikZ artifact keeps
            # its current path and stays valid.
            parts.append(_slug(self.cfg.target))
        return "__".join(parts)

    @property
    def strategy_lineage(self) -> str:
        """VESTIGIAL. Stage 2a is deprecated and cannot run (see
        `run_pipeline.DEPRECATED_STAGES`), so nothing is stored at this
        lineage any more.

        It is still computed, and still feeds `sequence_lineage`, purely to
        keep existing artifacts addressable: dropping it would rewrite the
        path of every sequence, animation and export ever produced, orphaning
        a fully scored 48-cell bench to gain nothing but a shorter directory
        name. The `__<model>__<tier>` it contributes is now a constant in
        practice, since no config changes a stage it cannot run.

        Delete it only alongside a deliberate, announced cache migration.
        """
        agent = self.cfg.agent("strategizer")
        tier = agent.option("context_tier", "full")
        return f"{self.code_lineage}__{self._model_for('strategizer')}__{tier}"

    @property
    def xml_lineage(self) -> str:
        return f"{self.code_lineage}__{self._model_for('parser')}"

    @property
    def sequence_lineage(self) -> str:
        """Where the two Stage-2 lineages merge.

        The sequencer's `context_tier` is appended ONLY when it is set, the
        same trick `code_lineage` uses for a non-tikz target: a sequencer run
        with paper context and one without must not share a cache entry, but
        every artifact produced before the option existed keeps its path.
        """
        parts = [self.strategy_lineage, self._model_for("parser"),
                 self._model_for("sequencer"), self.cfg.style]
        sequencer = self.cfg.agents.get("sequencer")
        tier = sequencer.option("context_tier", None) if sequencer else None
        if tier:
            parts.append(f"ctx-{_slug(str(tier))}")
        return "__".join(parts)

    @property
    def narrative_lineage(self) -> str:
        """Stage 2e. Deliberately NOT part of `animation_lineage`.

        Narration is spoken text laid over the animation; it does not change a
        single frame. Folding it into the animation lineage would invalidate
        every cached render whenever the narrative prompt or tier changed, and
        re-compiling a run's TikZ is far more expensive than re-writing its
        script.
        """
        agent = self.cfg.agents.get("narrative_writer")
        tier = agent.option("context_tier", "full") if agent else "full"
        return f"{self.sequence_lineage}__{self._model_for('narrative_writer')}__{tier}"

    @property
    def animation_lineage(self) -> str:
        return f"{self.sequence_lineage}__{self._model_for('designer')}"

    # -- artifact paths ---------------------------------------------------

    @property
    def _ext(self) -> str:
        return CODE_SUFFIX[self.cfg.target]

    def code(self, sample_id: str) -> Path:
        # `code_from` names a full stage-1 lineage (converter, or
        # converter__integrator). Its first component is the converter, which
        # is what this directory is keyed by.
        shared = self.cfg.raw.get("code_from")
        converter = (_slug(str(shared)).split("__")[0] if shared
                     else self._model_for("code_converter"))
        return self.root / "code" / converter / f"{sample_id}{self._ext}"

    def raw_output(self, agent: str, sample_id: str) -> Path:
        """Unparsed model output, kept for debugging extraction failures."""
        return self.root / "raw" / agent / self._model_for(agent) / f"{sample_id}.txt"

    def rasters(self, sample_id: str) -> Path:
        return self.root / "rasters" / self.code_lineage / sample_id

    def code_final(self, sample_id: str) -> Path:
        return self.root / "code_final" / self.code_lineage / f"{sample_id}{self._ext}"

    def code_reviewed(self, sample_id: str) -> Path:
        """Stage 1c output: the critic's best-scoring version of the code."""
        return self.root / "code_reviewed" / self.code_lineage / f"{sample_id}{self._ext}"

    def strategy(self, sample_id: str) -> Path:
        return self.root / "strategy" / self.strategy_lineage / f"{sample_id}.json"

    def xml(self, sample_id: str) -> Path:
        return self.root / "xml" / self.xml_lineage / f"{sample_id}.xml"

    def sequence(self, sample_id: str, round_index: int | None = None) -> Path:
        """Sequencer output; `round_index` addresses one critic round."""
        name = f"{sample_id}.json" if round_index is None else f"{sample_id}.round{round_index}.json"
        return self.root / "sequence" / self.sequence_lineage / name

    def sequence_final(self, sample_id: str) -> Path:
        return self.root / "sequence_final" / self.sequence_lineage / f"{sample_id}.json"

    def sequence_narrated(self, sample_id: str) -> Path:
        """Stage 2e output: the same sequence with spoken narration filled in."""
        return self.root / "sequence_narrated" / self.narrative_lineage / f"{sample_id}.json"

    def narration_jsonl(self, sample_id: str) -> Path:
        """The narration alone, timestamp-indexed, as the TTS stage wants it."""
        return self.root / "narration" / self.narrative_lineage / f"{sample_id}.jsonl"

    def audio(self, sample_id: str) -> Path:
        """Per-timestamp narration clips: `audio_ts_<n>.wav`."""
        return self.root / "audio" / self.narrative_lineage / sample_id

    def animation(self, sample_id: str, round_index: int | None = None) -> Path:
        name = (f"{sample_id}{self._ext}" if round_index is None
                else f"{sample_id}.round{round_index}{self._ext}")
        return self.root / "animation" / self.animation_lineage / name

    def animation_final(self, sample_id: str) -> Path:
        return self.root / "animation_final" / self.animation_lineage / f"{sample_id}{self._ext}"

    def exports(self, sample_id: str) -> Path:
        return self.root / "exports" / self.animation_lineage / sample_id

    def compile_cache(self) -> Path:
        """Shared content-addressed render cache for compile checks."""
        return self.root / "renders"

    # -- input resolution -------------------------------------------------

    def resolve_code(self, sample_id: str, reviewed: bool = True) -> Path:
        """The diagram code to plan against, newest usable version first.

        Order is critic-reviewed, then raster-integrated, then the raw
        converter output. `reviewed=False` stops at the raster-integrated
        version, which is what the critic itself needs -- otherwise it would
        read its own output and refine its refinement.

        A `code_reviewed` older than the `code_final` it reviewed is skipped:
        it describes a diagram that no longer exists. This is what made a
        freshly spliced set of rasters vanish from an animation -- the splice
        was minutes old, the critique three days old, and stage 3 quietly
        planned against the critique.
        """
        candidates = []
        if reviewed:
            critiqued = self.code_reviewed(sample_id)
            spliced = self.code_final(sample_id)
            superseded = (critiqued.is_file() and spliced.is_file()
                          and critiqued.stat().st_mtime < spliced.stat().st_mtime)
            if not superseded:
                candidates.append(critiqued)
        candidates += [self.code_final(sample_id), self.code(sample_id)]

        for path in candidates:
            if path.exists():
                return path
        listed = "\n  ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"no diagram code for '{sample_id}'. Looked for:\n  {listed}\n"
            f"Run the code-converter stage first."
        )

    def describe(self, sample_id: str = "<id>") -> dict[str, str]:
        """Every path this config resolves to. Powers `--print-paths`."""
        out = {
            "code": self.code(sample_id),
            "code_final": self.code_final(sample_id),
            "rasters": self.rasters(sample_id),
            "xml": self.xml(sample_id),
            "sequence": self.sequence(sample_id),
            "sequence_final": self.sequence_final(sample_id),
            "animation": self.animation(sample_id),
            "animation_final": self.animation_final(sample_id),
            "exports": self.exports(sample_id),
        }
        try:
            out["strategy"] = self.strategy(sample_id)
        except ValueError:
            pass
        # Absent from configs that predate stage 2e, so a missing agent is a
        # gap in the listing rather than a failure to print any paths at all.
        try:
            out["sequence_narrated"] = self.sequence_narrated(sample_id)
            out["narration"] = self.narration_jsonl(sample_id)
            out["audio"] = self.audio(sample_id)
        except (ValueError, ConfigError):
            pass
        return {k: str(v) for k, v in out.items()}


def resolve(directory: Path, sample_id: str, suffix: str) -> Path:
    """Find `<directory>/*/<sample_id><suffix>`, requiring exactly one match.

    Used when an upstream artifact's lineage isn't fully known. Ambiguity is
    an error rather than a silent first-match pick.
    """
    candidates = sorted(directory.glob(f"*/{sample_id}{suffix}"))
    if not candidates:
        raise FileNotFoundError(f"no artifact '{sample_id}{suffix}' under {directory}")
    if len(candidates) > 1:
        listed = "\n  ".join(str(c) for c in candidates)
        raise ValueError(
            f"ambiguous artifact for '{sample_id}' under {directory} -- "
            f"{len(candidates)} candidates:\n  {listed}\n"
            f"Name the upstream model explicitly in the config."
        )
    return candidates[0]


def write_text(path: Path, text: str) -> Path:
    """Atomic text write, creating parents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path
