"""Prompt loading, addressed as `<relative/path.yaml>#<key>`.

The prompt files are flat mappings of variant name to a literal block
string, e.g. `animation_planner/strategizer.yaml` holds four keys
(`image-only-prompt`, `image-plus-caption-prompt`, ...). The `#key` suffix
addresses one variant, so a config selects a prompt without the loader
needing to know each file's internal structure.

Interpolation uses `{name}` placeholders filled from the sample's context.
Braces are pervasive in these prompts (they contain TikZ and JSON), so
`str.format` is not usable -- a literal `\\node[...]{...}` would raise or
silently mangle. Substitution is therefore an explicit replace of only the
placeholders the caller supplies.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

PROMPTS_ROOT = Path(__file__).parent / "prompts"

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class PromptError(Exception):
    pass


@lru_cache(maxsize=None)
def _load_file(path: Path) -> dict[str, str]:
    import yaml

    if not path.exists():
        raise PromptError(f"prompt file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise PromptError(f"{path}: invalid YAML: {e}") from e
    if not isinstance(data, dict):
        raise PromptError(f"{path}: expected a mapping of name -> prompt text")
    return {k: v for k, v in data.items() if isinstance(v, str)}


def has_placeholders(template: str) -> bool:
    """Whether a template expects `{name}` substitution at all.

    Some prompts -- the AnimateBench sequencers among them -- are written as
    standalone instructions that name their inputs in prose and expect them
    appended rather than interpolated. Callers use this to tell the two
    conventions apart, since rendering a placeholder-free template silently
    supplies nothing.
    """
    return bool(_PLACEHOLDER_RE.search(template))


def parse_ref(ref: str) -> tuple[str, str | None]:
    """Split `file.yaml#key` into (file, key). Key is optional."""
    if "#" in ref:
        file_part, key = ref.split("#", 1)
        return file_part, key or None
    return ref, None


def load_prompt(ref: str, root: Path | None = None) -> str:
    """Resolve `file.yaml#key` to the raw prompt text.

    A file with exactly one key may omit `#key`; otherwise the key is
    required, so an ambiguous reference fails loudly rather than picking one.
    """
    file_part, key = parse_ref(ref)
    path = (Path(root) if root else PROMPTS_ROOT) / file_part
    prompts = _load_file(path)

    if not prompts:
        raise PromptError(f"{path}: contains no string-valued prompts")

    if key is None:
        if len(prompts) == 1:
            return next(iter(prompts.values()))
        raise PromptError(
            f"{path}: has {len(prompts)} prompts, so a #key is required. "
            f"Available: {sorted(prompts)}"
        )

    if key not in prompts:
        raise PromptError(f"{path}: no prompt named '{key}'. Available: {sorted(prompts)}")
    return prompts[key]


def render(template: str, context: dict[str, str] | None = None,
           strict: bool = False) -> str:
    """Substitute `{name}` placeholders from `context`.

    Only the keys present in `context` are replaced; every other brace in the
    template -- TikZ groups, JSON objects, LaTeX macros -- is left untouched,
    which is why this is not `str.format`.

    With `strict=True`, a context key that appears nowhere in the template
    raises. That catches a renamed placeholder silently dropping context that
    the caller believed it was passing.
    """
    if not context:
        return template

    if strict:
        present = set(_PLACEHOLDER_RE.findall(template))
        unused = sorted(set(context) - present)
        if unused:
            raise PromptError(
                f"context keys not present in template: {unused}. "
                f"Template placeholders: {sorted(present)}"
            )

    out = template
    for name, value in context.items():
        out = out.replace("{" + name + "}", str(value))
    return out


def load_and_render(ref: str, context: dict[str, str] | None = None,
                    root: Path | None = None, strict: bool = False) -> str:
    return render(load_prompt(ref, root=root), context, strict=strict)


def available_prompts(ref_or_file: str, root: Path | None = None) -> list[str]:
    """List the variant keys in a prompt file. Used by config validation."""
    file_part, _ = parse_ref(ref_or_file)
    path = (Path(root) if root else PROMPTS_ROOT) / file_part
    return sorted(_load_file(path))
