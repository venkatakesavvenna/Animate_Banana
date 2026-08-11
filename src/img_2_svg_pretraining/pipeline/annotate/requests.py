"""Turn a pipeline agent's exact outgoing request into something copy-pasteable.

Calls the same `build_request` the real pipeline call uses, so what's shown
here is never a re-derived approximation of the prompt -- it is the prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..runner import AgentContext
from ..samples import PaperSample


@dataclass
class CopyBlock:
    prompt_text: str
    model: str
    image_paths: list[Path] = field(default_factory=list)


def build_copyable_request(ctx: AgentContext, sample: PaperSample,
                           module) -> CopyBlock:
    """`module` is a pipeline agent exposing `build_request(ctx, sample)`.

    Agents differ in whether they send images: the code converter attaches the
    figure, while the parser sends text only (prompt + the resolved code), so
    `image_paths` is empty for it and the screen supplies the figure itself.
    """
    messages = module.build_request(ctx, sample)
    prompt_text = "\n\n".join(m.text() for m in messages if m.text())
    image_paths = [p.path for m in messages for p in m.images()]
    model = ctx.cfg.backend_model(ctx.agent.backend) if ctx.agent.backend else ""
    return CopyBlock(prompt_text=prompt_text, model=model, image_paths=image_paths)
