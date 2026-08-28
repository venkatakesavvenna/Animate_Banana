"""Question sets, loaded from yaml.

An experiment is a question set plus a caption flag -- nothing in the code
branches on which experiment is running. That is deliberate: the wording will
be rewritten several times before the study runs (the team's note was that
model-drafted questions "read very GPT"), and rewording must never mean
touching Python.

Placeholders in a prompt or help string are filled from the trial: `{style_name}`
and `{style_description}` are the only ones today. A missing placeholder leaves
the text alone rather than raising -- a half-rendered question is still
answerable, an exception mid-session is not.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

QUESTION_DIR = Path(__file__).parent / "questions"

VALID_TYPES = {"likert5", "yesno", "choice_ab", "select", "text"}

# Asked once per diagram, before the trial itself. Kept here rather than in each
# experiment's file because it is the same question everywhere and must stay
# comparable across all five.
FAMILIARITY = {
    "id": "familiarity",
    "prompt": "How familiar are you with the topic of this figure?",
    "type": "select",
    "options": [
        {"value": "not_familiar", "label": "Not familiar"},
        {"value": "somewhat", "label": "Somewhat familiar"},
        {"value": "familiar", "label": "Familiar"},
    ],
}


@lru_cache(maxsize=None)
def _load_raw(experiment: str) -> dict:
    path = QUESTION_DIR / f"{experiment}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no question set for '{experiment}' at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    seen = set()
    for q in data.get("questions", []):
        if q.get("type") not in VALID_TYPES:
            raise ValueError(f"{path.name}: question '{q.get('id')}' has unknown "
                             f"type '{q.get('type')}'. Known: {sorted(VALID_TYPES)}")
        if q["id"] in seen:
            raise ValueError(f"{path.name}: duplicate question id '{q['id']}'")
        seen.add(q["id"])
        if q["type"] == "select" and not q.get("options"):
            raise ValueError(f"{path.name}: select question '{q['id']}' has no options")
    return data


def _fill(text: str, values: dict) -> str:
    if not text:
        return text
    try:
        return text.format(**values)
    except (KeyError, IndexError):
        return text


def question_set(experiment: str, *, style_name: str = "",
                 style_description: str = "") -> dict:
    """The questions for one experiment, ready to render."""
    data = _load_raw(experiment)
    values = {"style_name": style_name.replace("_", " "),
              "style_description": style_description}

    questions = []
    for q in data.get("questions", []):
        item = dict(q)
        item["prompt"] = _fill(item.get("prompt", ""), values)
        if item.get("help"):
            item["help"] = _fill(item["help"], values)
        # `metric` is bookkeeping for the human<->LLM correlation and has no
        # business reaching the participant's browser.
        item.pop("metric", None)   # `section` is kept: it groups the UI
        questions.append(item)

    return {
        "experiment": experiment,
        "title": _fill(data.get("title", ""), values),
        "intro": _fill(data.get("intro", ""), values),
        "familiarity": FAMILIARITY,
        "questions": questions,
    }


def metric_map(experiment: str) -> dict:
    """question_id -> the AnimateBench metric it should correlate with."""
    return {q["id"]: q.get("metric") for q in _load_raw(experiment).get("questions", [])
            if q.get("metric")}


def required_ids(experiment: str) -> list[str]:
    return [q["id"] for q in _load_raw(experiment).get("questions", [])
            if not q.get("optional")]


def all_experiments() -> list[str]:
    return sorted(p.stem for p in QUESTION_DIR.glob("exp*.yaml"))
