"""Compiles standalone TikZ/LaTeX source into a PNG for preview.

Requires `pdflatex` (or `latexmk`) and poppler's `pdftoppm` to be on PATH.
Both are expected to be installed inside the project's docker image.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CompileResult:
    ok: bool
    png_path: Path | None
    log: str


def _source_hash(tex_source: str) -> str:
    return hashlib.sha256(tex_source.encode("utf-8")).hexdigest()[:16]


def compile_tikz(tex_source: str, cache_dir: Path, dpi: int = 150, timeout: int = 60) -> CompileResult:
    """Compile a .tex source string to PNG, caching by content hash.

    Returns a CompileResult with ok=False and the pdflatex log on failure so
    the viewer can surface compile errors to the user for correction.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = _source_hash(tex_source)
    png_path = cache_dir / f"{digest}.png"
    if png_path.exists():
        return CompileResult(ok=True, png_path=png_path, log="")

    work_dir = cache_dir / f"_work_{digest}"
    work_dir.mkdir(parents=True, exist_ok=True)
    tex_file = work_dir / "doc.tex"
    tex_file.write_text(tex_source, encoding="utf-8")

    try:
        proc = subprocess.run(
            [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory=" + str(work_dir),
                str(tex_file),
            ],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        log = proc.stdout + "\n" + proc.stderr
        pdf_file = work_dir / "doc.pdf"
        if proc.returncode != 0 or not pdf_file.exists():
            return CompileResult(ok=False, png_path=None, log=log)

        convert = subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-r",
                str(dpi),
                "-singlefile",
                str(pdf_file),
                str(cache_dir / digest),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if convert.returncode != 0 or not png_path.exists():
            return CompileResult(ok=False, png_path=None, log=log + "\n" + convert.stdout + convert.stderr)

        return CompileResult(ok=True, png_path=png_path, log=log)
    except subprocess.TimeoutExpired as e:
        return CompileResult(ok=False, png_path=None, log=f"Compilation timed out: {e}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
