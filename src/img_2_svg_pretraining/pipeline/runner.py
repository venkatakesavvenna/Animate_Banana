"""Shared machinery for running one agent over a batch of samples.

Every agent has the same outer shape: skip samples whose artifact already
exists unless forced, build a request per remaining sample, send them through
the configured backend, parse each response, write the artifact, and keep
going when one sample fails. That loop lives here so each agent module only
supplies the parts that differ.

Failure isolation is deliberate: one sample raising must not abandon the
other ten, since a batch may represent hours of API calls.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .backends import ChatBackend, Message, make_backend
from .cache import CachePaths, write_text
from .config import PipelineConfig
from .samples import PaperSample


@dataclass
class SampleOutcome:
    sample_id: str
    # "ok"         produced a good artifact
    # "unresolved" produced an artifact that still has known problems
    # "failed"     produced nothing
    # "skipped"    already cached
    status: str
    path: Path | None = None
    detail: str = ""
    latency_s: float = 0.0


@dataclass
class StageReport:
    agent: str
    outcomes: list[SampleOutcome] = field(default_factory=list)

    @property
    def ok(self) -> list[SampleOutcome]:
        return [o for o in self.outcomes if o.status == "ok"]

    @property
    def failed(self) -> list[SampleOutcome]:
        return [o for o in self.outcomes if o.status == "failed"]

    @property
    def skipped(self) -> list[SampleOutcome]:
        return [o for o in self.outcomes if o.status == "skipped"]

    def summary(self) -> str:
        return (f"{self.agent}: {len(self.ok)} ok, {len(self.failed)} failed, "
                f"{len(self.skipped)} skipped")

    def print_report(self) -> None:
        print(f"\n=== {self.summary()} ===")
        for o in self.outcomes:
            if o.status == "failed":
                print(f"  FAIL {o.sample_id}: {o.detail}")
        for o in self.ok:
            note = f" ({o.detail})" if o.detail else ""
            print(f"  ok   {o.sample_id}{note} -> {o.path}")


class AgentContext:
    """Everything an agent needs: config, paths, and its backend.

    The backend is built lazily so a run that skips every sample (all cached)
    never constructs a client or loads a local model.
    """

    def __init__(self, cfg: PipelineConfig, agent_name: str):
        self.cfg = cfg
        self.agent_name = agent_name
        self.agent = cfg.agent(agent_name)
        self.paths = CachePaths.from_config(cfg)
        self._backend: ChatBackend | None = None

    @property
    def backend(self) -> ChatBackend:
        if self._backend is None:
            if not self.agent.backend:
                raise ValueError(f"agent '{self.agent_name}' has no backend configured")
            self._backend = make_backend(
                self.agent.backend,
                self.cfg.backend_cfg(self.agent.backend),
                cache_root=self.paths.root,
            )
        return self._backend

    def unload(self) -> None:
        if self._backend is not None:
            self._backend.unload()
            self._backend = None

    def params(self) -> dict:
        return dict(self.agent.params)

    def provenance(self, **extra) -> dict:
        """Run metadata stored alongside each artifact, so a result can be
        traced back to the config and models that produced it."""
        prov = {
            "agent": self.agent_name,
            "backend": self.agent.backend,
            "model": (self.cfg.backend_model(self.agent.backend)
                      if self.agent.backend else None),
            "prompt": self.agent.prompt,
            "style": self.cfg.style,
            "target": self.cfg.target,
            "config_fingerprint": self.cfg.fingerprint(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        prov.update(extra)
        return prov


def run_agent(
    ctx: AgentContext,
    samples: list[PaperSample],
    output_path: Callable[[PaperSample], Path],
    build_request: Callable[[PaperSample], list[Message]],
    handle_response: Callable[[PaperSample, str], str],
    force: bool = False,
    save_raw: bool = True,
) -> StageReport:
    """Run one chat agent over `samples`.

    `build_request` may raise to skip a sample it cannot handle (e.g. missing
    upstream artifact); `handle_response` may raise to reject an unparseable
    response. Both are reported as failures against that sample alone.

    Returns the text to write; writing and provenance are handled here.
    """
    report = StageReport(agent=ctx.agent_name)

    pending: list[PaperSample] = []
    for sample in samples:
        out = output_path(sample)
        if out.exists() and not force:
            report.outcomes.append(
                SampleOutcome(sample.id, "skipped", out, "cached"))
        else:
            pending.append(sample)

    if not pending:
        return report

    requests: list[list[Message]] = []
    prepared: list[PaperSample] = []
    for sample in pending:
        try:
            requests.append(build_request(sample))
            prepared.append(sample)
        except Exception as e:
            report.outcomes.append(
                SampleOutcome(sample.id, "failed", None, f"{type(e).__name__}: {e}"))

    if not prepared:
        return report

    print(f"[{ctx.agent_name}] {len(prepared)} sample(s) via "
          f"{ctx.agent.backend} ({ctx.cfg.backend_model(ctx.agent.backend)})")

    try:
        results = ctx.backend.generate_batch(requests, **ctx.params())
    except Exception as e:
        # Backend construction or a whole-batch failure: attribute to every
        # sample rather than losing the reason.
        for sample in prepared:
            report.outcomes.append(
                SampleOutcome(sample.id, "failed", None, f"{type(e).__name__}: {e}"))
        return report

    for sample, result in zip(prepared, results):
        if save_raw and result.text:
            write_text(ctx.paths.raw_output(ctx.agent_name, sample.id), result.text)

        if not result.ok:
            report.outcomes.append(
                SampleOutcome(sample.id, "failed", None, result.error or "unknown error",
                              result.latency_s))
            continue

        try:
            payload = handle_response(sample, result.text)
        except Exception as e:
            report.outcomes.append(
                SampleOutcome(sample.id, "failed", None, f"{type(e).__name__}: {e}",
                              result.latency_s))
            continue

        out = output_path(sample)
        write_text(out, payload)
        detail = "cached response" if result.from_cache else f"{result.latency_s:.1f}s"
        report.outcomes.append(SampleOutcome(sample.id, "ok", out, detail, result.latency_s))

    return report


def run_iterative_agent(
    ctx: AgentContext,
    samples: list[PaperSample],
    output_path: Callable[[PaperSample], Path],
    step: Callable[[PaperSample, int], tuple[bool, str, str]],
    max_rounds: int,
    force: bool = False,
) -> StageReport:
    """Run a critic-style agent that refines a sample over several rounds.

    `step(sample, round_index)` returns `(done, payload, detail)`. It is
    called until `done` or `max_rounds` is reached; the final payload is
    written. Per-round artifacts are the step's own responsibility, so the
    refinement history stays inspectable.

    A sample whose budget runs out without converging is reported as
    "unresolved", not "ok". Its payload is still written -- the best attempt
    is more useful downstream than nothing -- but it must not be silently
    counted as a success, since an animation that never compiled would
    otherwise sail through to the exporter looking healthy.
    """
    report = StageReport(agent=ctx.agent_name)

    for sample in samples:
        out = output_path(sample)
        if out.exists() and not force:
            report.outcomes.append(SampleOutcome(sample.id, "skipped", out, "cached"))
            continue

        payload: str | None = None
        detail = ""
        converged = False
        try:
            for round_index in range(max_rounds):
                converged, payload, detail = step(sample, round_index)
                if converged:
                    break
        except Exception as e:
            report.outcomes.append(
                SampleOutcome(sample.id, "failed", None, f"{type(e).__name__}: {e}"))
            continue

        if payload is None:
            report.outcomes.append(
                SampleOutcome(sample.id, "failed", None, detail or "no output produced"))
            continue

        write_text(out, payload)
        if converged:
            report.outcomes.append(SampleOutcome(sample.id, "ok", out, detail))
        else:
            report.outcomes.append(SampleOutcome(
                sample.id, "unresolved", out,
                f"{detail} (unresolved after {max_rounds} round(s))"))

    return report
