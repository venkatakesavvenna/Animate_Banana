"""Thin reuse of benchmark.metrics.judge for the data engine's final
visual-fidelity filter: render the generated code, compare against the
source image, keep/discard by score threshold. No new judging logic --
`JudgeRunner`/`JUDGE_PROMPT` are already backend-agnostic (they just compare
two images), so they're reused as-is rather than reimplemented here.
"""
from __future__ import annotations

from pathlib import Path

from img_2_svg_pretraining.benchmark.metrics.judge import JudgeResult, JudgeRunner

DEFAULT_KEEP_THRESHOLD = 4


def judge_and_filter(
    runner: JudgeRunner,
    source_image_path: Path,
    rendered_image_path: Path,
    keep_threshold: int = DEFAULT_KEEP_THRESHOLD,
) -> tuple[bool, JudgeResult]:
    result = runner.judge(source_image_path, rendered_image_path)
    keep = result.score is not None and result.score >= keep_threshold
    return keep, result
