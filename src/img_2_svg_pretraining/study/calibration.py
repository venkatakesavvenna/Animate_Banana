"""Preparation examples and the calibration quiz.

The team's decision was that the worked examples should be *our own* recorded
answers, not invented ones: the researchers annotate through the same tool with
`is_expert` set, and those sessions become both the teaching examples and the
answer key. So there is nothing to author by hand, and the examples cannot drift
from what the tool actually asks.

That has a consequence worth stating plainly: **until at least one expert
session exists, there is no key, and calibration cannot score anyone.** It then
reports itself unavailable and lets participants straight through rather than
blocking the study on missing data. The admin dashboard says so loudly. Silently
passing everyone while looking like a working gate would be the worse failure.

Scoring is per question, tolerant where tolerance is meaningful:
  * likert5   -- within +/-1 of the expert median counts as agreement, because
                 a 4-vs-5 disagreement is not a misunderstanding of the task
  * yesno / select / choice_ab -- exact match
Free text is never scored.
"""
from __future__ import annotations

import json
import statistics

from img_2_svg_pretraining.study.questions import question_set

# A calibration item is one already-judged trial, replayed to a new participant.
EXPERT_MIN = 1          # expert sessions needed before a key exists
TOLERANCE_LIKERT = 1


def expert_answers(db, bundle_id: str) -> dict:
    """cell_key -> {question_id: [expert values]} over expert-flagged sessions."""
    with db._connect() as conn:
        rows = conn.execute("""
            SELECT t.cell_key, t.experiment, t.diagram_id, t.animation_style,
                   t.presentation_a_id, t.presentation_b_id, t.show_captions,
                   r.question_id, r.value, r.revision, r.trial_id
            FROM trial t
            JOIN participant p ON p.participant_id = t.participant_id
            JOIN response r    ON r.trial_id = t.trial_id
            WHERE p.is_expert = 1 AND t.status = 'submitted' AND t.bundle_id = ?
            ORDER BY r.revision
        """, (bundle_id,)).fetchall()

    # Latest revision wins per (trial, question); several experts then pool.
    latest: dict[tuple, tuple] = {}
    meta: dict[str, dict] = {}
    for r in rows:
        latest[(r["trial_id"], r["question_id"])] = (r["cell_key"], r["value"])
        meta[r["cell_key"]] = {
            "experiment": r["experiment"], "diagram_id": r["diagram_id"],
            "animation_style": r["animation_style"],
            "presentation_a_id": r["presentation_a_id"],
            "presentation_b_id": r["presentation_b_id"],
            "show_captions": bool(r["show_captions"])}

    key: dict = {}
    for (_, question_id), (cell_key, value) in latest.items():
        key.setdefault(cell_key, {}).setdefault(question_id, []).append(
            json.loads(value))
    return {"key": key, "meta": meta}


def available(db, bundle_id: str) -> bool:
    with db._connect() as conn:
        n = conn.execute(
            "SELECT COUNT(DISTINCT t.participant_id) c FROM trial t"
            " JOIN participant p ON p.participant_id = t.participant_id"
            " WHERE p.is_expert = 1 AND t.status='submitted' AND t.bundle_id=?",
            (bundle_id,)).fetchone()["c"]
    return n >= EXPERT_MIN


def _agrees(question: dict, given, expected: list) -> bool:
    if not expected:
        return True                     # nothing to disagree with
    if question["type"] == "likert5":
        try:
            target = statistics.median([float(v) for v in expected])
            return abs(float(given) - target) <= TOLERANCE_LIKERT
        except (TypeError, ValueError):
            return False
    if question["type"] == "text":
        return True                     # never scored
    # Majority answer for everything categorical.
    counts: dict = {}
    for v in expected:
        counts[json.dumps(v)] = counts.get(json.dumps(v), 0) + 1
    winner = max(counts, key=counts.get)
    return json.dumps(given) == winner


def build_items(db, bundle_id: str, cfg, styles: dict, limit_per_experiment=1) -> list:
    """One calibration item per enabled, stocked experiment."""
    data = expert_answers(db, bundle_id)
    items, used = [], set()
    for experiment in cfg.enabled_experiments():
        picked = 0
        for cell_key, answers in data["key"].items():
            info = data["meta"].get(cell_key, {})
            if info.get("experiment") != experiment or cell_key in used:
                continue
            style = info.get("animation_style", "")
            items.append({
                "cell_key": cell_key,
                "experiment": experiment,
                "diagram_id": info["diagram_id"],
                "animation_style": style,
                "presentation_a_id": info["presentation_a_id"],
                "presentation_b_id": info["presentation_b_id"],
                "show_captions": info["show_captions"],
                "questions": question_set(experiment, style_name=style,
                                          style_description=styles.get(style, "")),
                "expected": answers,
            })
            used.add(cell_key)
            picked += 1
            if picked >= limit_per_experiment:
                break
    return items


def score(items: list, submitted: dict) -> dict:
    """Fraction of scorable questions on which the participant agrees."""
    total = hit = 0
    detail = []
    for item in items:
        answers = submitted.get(item["cell_key"], {})
        for question in item["questions"]["questions"]:
            if question["type"] == "text" or question.get("optional"):
                continue
            expected = item["expected"].get(question["id"])
            if not expected:
                continue
            total += 1
            ok = _agrees(question, answers.get(question["id"]), expected)
            hit += bool(ok)
            detail.append({"cell_key": item["cell_key"],
                           "question_id": question["id"], "agreed": bool(ok)})
    return {"score": (hit / total) if total else 1.0,
            "scored": total, "agreed": hit, "detail": detail}
