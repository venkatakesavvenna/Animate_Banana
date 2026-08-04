"""Pipeline configuration: load, validate, and hand out per-agent settings.

One YAML file drives a whole run. Every agent independently names a backend,
so a run can mix a commercial API for one agent with local open weights for
another.

Validation happens eagerly at load: unknown agent names, backend references
that don't resolve, prompt files or keys that don't exist, and unknown enum
values all raise here rather than three stages into a run.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .backends import CHAT_BACKENDS
from .prompts import PromptError, available_prompts, parse_ref
from .samples import TIERS

# agent name -> (section, requires_prompt)
AGENTS: dict[str, tuple[str, bool]] = {
    "code_converter": ("transmuter", True),
    "raster_integrator": ("transmuter", False),
    "strategizer": ("planner", True),
    "parser": ("planner", True),
    "sequencer": ("planner", True),
    "planner_critic": ("planner", True),
    "narrative_writer": ("planner", True),
    "designer": ("animator", True),
    "animator_critic": ("animator", True),
    "exporter": ("animator", False),
}

SECTIONS = ("transmuter", "planner", "animator")
TARGETS = ("tikz", "svg")
EXPORT_FORMATS = ("pdf", "mp4", "gif", "pptx", "narrated_mp4")

# Agent config keys understood by the runner; anything else is a typo.
_AGENT_KEYS = {
    "backend", "prompt", "params", "enabled", "context_tier", "max_rounds",
    "use_set_of_mark", "compile_check", "outputs", "fps", "dpi", "max_depth",
    "gpu",
}


class ConfigError(Exception):
    pass


@dataclass
class AgentConfig:
    name: str
    section: str
    backend: str | None = None
    prompt: str | None = None
    params: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)

    def option(self, key: str, default=None):
        return self.options.get(key, default)


@dataclass
class PipelineConfig:
    path: Path
    raw: dict
    dataset_root: Path
    limit: int
    style: str
    target: str
    backends: dict[str, dict]
    agents: dict[str, AgentConfig]
    cache_root: Path

    def agent(self, name: str) -> AgentConfig:
        if name not in self.agents:
            raise ConfigError(
                f"config has no settings for agent '{name}'. "
                f"Add it under '{AGENTS[name][0]}:' in {self.path}"
            )
        return self.agents[name]

    def backend_cfg(self, name: str) -> dict:
        if name not in self.backends:
            raise ConfigError(
                f"unknown backend '{name}'. Defined: {sorted(self.backends)}"
            )
        return self.backends[name]

    def backend_model(self, backend_name: str) -> str:
        """Short model label used in cache path lineage.

        Identified by the weights, not by the config's nickname for them, so
        two backends pointing at different models never share a cache path.
        """
        cfg = self.backend_cfg(backend_name)
        model = cfg.get("model") or cfg.get("hf_repo") or backend_name
        return str(model).replace("/", "-")

    def fingerprint(self) -> str:
        """Hash of the resolved config, recorded in artifact provenance."""
        return hashlib.sha256(
            json.dumps(self.raw, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]


# Agents that pick their prompt variant at runtime rather than in the config,
# and so may name a multi-variant file with no '#key'. The narrative writer
# selects by context tier, exactly as the strategizer would if its tier keys
# were not already spelled out per-config.
_RUNTIME_VARIANT_AGENTS = {"narrative_writer"}


def _validate_prompt_ref(agent_name: str, ref: str) -> None:
    file_part, key = parse_ref(ref)
    try:
        keys = available_prompts(file_part)
    except PromptError as e:
        raise ConfigError(f"agent '{agent_name}': {e}") from e
    if key is not None and key not in keys:
        raise ConfigError(
            f"agent '{agent_name}': prompt '{file_part}' has no key '{key}'. Available: {keys}"
        )
    if key is None and len(keys) > 1 and agent_name not in _RUNTIME_VARIANT_AGENTS:
        raise ConfigError(
            f"agent '{agent_name}': prompt '{file_part}' has {len(keys)} variants, "
            f"so the reference needs a '#key' suffix. Available: {keys}"
        )


def load_config(path: str | Path) -> PipelineConfig:
    import yaml

    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: invalid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")

    # -- dataset -------------------------------------------------------
    dataset = raw.get("dataset") or {}
    if not dataset.get("root"):
        raise ConfigError(f"{path}: missing required 'dataset.root'")
    dataset_root = Path(dataset["root"])
    limit = int(dataset.get("limit", 0) or 0)

    # -- global run settings -------------------------------------------
    # Imported here to avoid a circular import at module load.
    from .styles import STYLES

    style = raw.get("animation_style", "progressive_reveal")
    if style not in STYLES:
        raise ConfigError(
            f"{path}: unknown animation_style '{style}'. Known: {sorted(STYLES)}"
        )

    target = raw.get("target", "tikz")
    if target not in TARGETS:
        raise ConfigError(f"{path}: target must be one of {TARGETS}, got '{target}'")

    cache_root = Path(raw.get("cache_root") or (Path(__file__).parent / "cache"))

    # -- backends ------------------------------------------------------
    backends = raw.get("backends") or {}
    if not backends:
        raise ConfigError(f"{path}: no 'backends' defined")
    for name, cfg in backends.items():
        if not isinstance(cfg, dict):
            raise ConfigError(f"backend '{name}': must be a mapping")
        btype = cfg.get("type")
        if btype is None:
            raise ConfigError(f"backend '{name}': missing 'type'")
        # Vision backends are validated by the raster integrator, which owns
        # their registry; only chat types are checked here.
        if btype in CHAT_BACKENDS:
            if btype == "hf_local":
                if not cfg.get("hf_repo"):
                    raise ConfigError(f"backend '{name}': hf_local requires 'hf_repo'")
            elif not cfg.get("model"):
                raise ConfigError(f"backend '{name}': type '{btype}' requires 'model'")

    # -- agents --------------------------------------------------------
    agents: dict[str, AgentConfig] = {}
    for section in SECTIONS:
        block = raw.get(section) or {}
        if not isinstance(block, dict):
            raise ConfigError(f"{path}: section '{section}' must be a mapping")
        for agent_name, agent_cfg in block.items():
            qualified = _qualify(section, agent_name)
            if qualified not in AGENTS:
                known = sorted(n for n, (s, _) in AGENTS.items() if s == section)
                raise ConfigError(
                    f"{path}: unknown agent '{agent_name}' in section '{section}'. "
                    f"Known: {known}"
                )
            if not isinstance(agent_cfg, dict):
                raise ConfigError(f"agent '{qualified}': must be a mapping")

            unknown = sorted(set(agent_cfg) - _AGENT_KEYS)
            if unknown:
                raise ConfigError(
                    f"agent '{qualified}': unknown option(s) {unknown}. "
                    f"Known: {sorted(_AGENT_KEYS)}"
                )

            backend = agent_cfg.get("backend")
            if backend is not None and backend not in backends:
                raise ConfigError(
                    f"agent '{qualified}': backend '{backend}' is not defined. "
                    f"Defined: {sorted(backends)}"
                )

            prompt = agent_cfg.get("prompt")
            if prompt:
                _validate_prompt_ref(qualified, prompt)
            elif AGENTS[qualified][1] and agent_cfg.get("enabled", True):
                raise ConfigError(f"agent '{qualified}': missing required 'prompt'")

            tier = agent_cfg.get("context_tier")
            if tier is not None and tier not in TIERS:
                raise ConfigError(
                    f"agent '{qualified}': unknown context_tier '{tier}'. Known: {list(TIERS)}"
                )

            outputs = agent_cfg.get("outputs")
            if outputs is not None:
                bad = sorted(set(outputs) - set(EXPORT_FORMATS))
                if bad:
                    raise ConfigError(
                        f"agent '{qualified}': unknown output format(s) {bad}. "
                        f"Known: {list(EXPORT_FORMATS)}"
                    )

            agents[qualified] = AgentConfig(
                name=qualified,
                section=section,
                backend=backend,
                prompt=prompt,
                params=agent_cfg.get("params") or {},
                options={k: v for k, v in agent_cfg.items()
                         if k not in ("backend", "prompt", "params")},
            )

    return PipelineConfig(
        path=path, raw=raw, dataset_root=dataset_root, limit=limit,
        style=style, target=target, backends=backends, agents=agents,
        cache_root=cache_root,
    )


def _qualify(section: str, agent_name: str) -> str:
    """Map a section-local agent name to its globally unique key.

    Both the planner and the animator have an agent called `critic`; they are
    distinct agents with distinct prompts and cache entries, so they get
    distinct qualified names.
    """
    if agent_name == "critic":
        return "planner_critic" if section == "planner" else "animator_critic"
    return agent_name
