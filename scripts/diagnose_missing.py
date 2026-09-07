#!/usr/bin/env python3
"""Explain, per (model, sample), WHY no animation exists -- one reason each.

Medha's `*_missing.txt` lists name samples with no video. A list of ids does
not say whether the model refused, looped, or the render broke, and those
have different fixes. This walks each missing cell through the artifact chain
and reports the FIRST stage that has no output, then reads that stage's raw
response to classify the failure concretely.

Reasons are derived from artifacts, never guessed: the raw stage-1 response is
kept on disk precisely so a parse failure can be explained afterwards.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

LIN2NAME = {"Qwen-Qwen3-VL-235B-A22B-Instruct": "Qwen3-VL-235B",
            "google-gemma-4-31B-it": "Gemma-4-31B",
            "Qwen-Qwen3.8-27B": "Qwen3.8-27B",
            "zai-org-GLM-4.6V": "GLM-4.6V",
            "google-gemini-3.7-flash": "Gemini-3.7-Flash"}
NAME2LIN = {v: k for k, v in LIN2NAME.items()}


def looping_tail(text: str) -> str | None:
    """Detect a degenerate repetition loop at the end of a generation.

    A model that loses coherence emits the same short token run until the cap.
    Checked on the tail only: a legitimate SVG repeats short substrings all
    over (path commands, colours), so a whole-document frequency test would
    flag healthy output.
    """
    tail = text[-400:]
    for size in range(4, 60):
        unit = tail[-size:]
        if unit.strip() and tail.endswith(unit * 5):
            return unit.strip()
    return None


STAGE_LABEL = {"xml": "stage 2b parse (SVG->structure XML)",
               "sequence": "stage 2c sequencer (traversal order)",
               "animation": "stage 3a designer (animation code)",
               "export": "stage 3c exporter (frames -> mp4)"}


def usage_index(cache: Path) -> dict:
    """md5(full response text) -> usage, so a raw file can be tied to its
    token count. Hashing the FULL text matters: every SVG opens with the same
    `<svg xmlns=...` boilerplate, so a prefix key collides across samples and
    attributes one sample's token count to another."""
    idx = {}
    for f in glob.glob(str(cache / "responses/served/*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        t = d.get("text") or ""
        if t:
            idx[hashlib.md5(t.encode("utf-8", "replace")).hexdigest()] = \
                d.get("usage") or {}
    return idx


CAP = 65536


def classify_stage1(raw: Path, usage: dict | None = None) -> tuple[str, str]:
    if not raw.exists():
        return ("no_response", "stage 1a produced no stored response "
                               "(request failed or was never made)")
    text = raw.read_text(errors="replace")
    n = len(text)
    tok = (usage or {}).get(
        hashlib.md5(text.encode("utf-8", "replace")).hexdigest(), {}
    ).get("completion_tokens")
    # "At the cap" is read from the token count, not guessed from length: a
    # long document and a truncated one look identical by character count.
    at_cap = tok == CAP
    budget = (f"generation stopped at the {CAP:,}-token output cap"
              if at_cap else
              f"generation stopped early at {tok:,} tokens (below the "
              f"{CAP:,}-token cap)" if tok else "generation stopped")
    if not text.strip():
        return "empty_response", "stage 1a returned an empty response"
    has_open, has_close = "<svg" in text, "</svg>" in text
    if has_open and not has_close:
        loop = looping_tail(text)
        if loop:
            return ("truncated_repetition_loop",
                    f"stage 1a image->SVG: model degenerated into a repetition loop "
                    f"(repeating {loop!r}); {budget}, so <svg> was never closed "
                    f"and no document could be parsed")
        return ("truncated" if at_cap else "stopped_early",
                f"stage 1a image->SVG: {budget} with <svg> still open "
                f"({n:,} chars emitted), so no document could be parsed")
    if not has_open:
        head = " ".join(text.strip()[:120].split())
        return ("no_svg_emitted",
                f"stage 1a image->SVG: model emitted no <svg> element at all "
                f"(response begins {head!r})")
    return ("unparseable_svg",
            "stage 1a (image->SVG): <svg> present but the document did not parse")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/animatebench_zs_cache/animatebench_zs")
    ap.add_argument("--dataset", default="data/animatebench_zs")
    ap.add_argument("--style-map", default="data/zs_style_map.json")
    ap.add_argument("--logs", default="logs/bench_zs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cache, out = REPO / args.cache, Path(args.out)
    styles = json.loads((REPO / args.style_map).read_text())
    all_samples = set(styles)

    usage = usage_index(cache)
    have = defaultdict(set)
    for mp4 in cache.glob("exports/*/*/animation.mp4"):
        have[LIN2NAME.get(mp4.parent.parent.name.split("__")[0])].add(mp4.parent.name)

    # Stage chain in pipeline order; the first one MISSING is where it stopped.
    def first_gap(model_lin: str, sample: str, style: str) -> str:
        for stage, pat in (("code", f"code/{model_lin}/{sample}.svg"),
                           ("xml", f"xml/{model_lin}__*/{sample}.xml"),
                           ("sequence", f"sequence/{model_lin}__*{style}/{sample}.json"),
                           ("animation", f"animation/{model_lin}__*{style}*/{sample}.*"),
                           ("export", f"exports/{model_lin}__*{style}*/{sample}/animation.mp4")):
            if not list(cache.glob(pat)):
                return stage
        return "complete"

    rows = []
    for model, lin in NAME2LIN.items():
        if model not in have and not list(cache.glob(f"code/{lin}")):
            continue
        for sample in sorted(all_samples - have.get(model, set())):
            style = styles[sample]
            gap = first_gap(lin, sample, style)
            if gap == "code":
                reason_key, reason = classify_stage1(
                    cache / "raw/code_converter" / lin / f"{sample}.txt", usage)
            elif gap == "complete":
                reason_key, reason = ("export_missing_only",
                                      "all stages present but no mp4 -- exporter failed")
            else:
                # Downstream: quote the pipeline log line for this sample.
                msg = ""
                for log in (REPO / args.logs).glob(f"pipe_*{style}.log"):
                    for line in log.read_text(errors="replace").splitlines():
                        if sample in line and "FAIL" in line:
                            msg = line.split("FAIL", 1)[1].strip()
                            msg = msg.split(f"{sample}:", 1)[-1].strip()
                            # The stored-raw path is an internal breadcrumb and
                            # swamps the actual error in a one-line summary.
                            msg = re.split(r";?\s*raw output kept at", msg)[0].strip()
                            msg = " ".join(msg.split())
                            # A context-overflow 400 carries the whole vLLM
                            # payload; the fact that matters is the arithmetic.
                            m = re.search(r"maximum context length is ([\d,]+) tokens.*?"
                                          r"requested ([\d,]+) output tokens.*?"
                                          r"prompt contains at least ([\d,]+) input",
                                          msg)
                            if m:
                                msg = (f"prompt too long for the judgeable window -- "
                                       f"the generated SVG produced {m.group(3)} input "
                                       f"tokens which, with {m.group(2)} requested output "
                                       f"tokens, exceeds the model's {m.group(1)}-token "
                                       f"context (HTTP 400)")
                            msg = msg[:260]
                            break
                    if msg:
                        break
                reason_key = f"{gap}_stage_failed"
                reason = (STAGE_LABEL.get(gap, f"stage '{gap}'") + " failed"
                          + (f": {msg}" if msg else " (no artifact produced)"))
            rows.append({"model": model, "sample": sample, "style": style,
                         "failed_stage": gap, "reason_key": reason_key,
                         "reason": reason})

    out.mkdir(parents=True, exist_ok=True)
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    for model, rs in sorted(by_model.items()):
        # <sample>: <reason>, matching the shape of the lists this answers.
        (out / f"{model}_missing_reasons.txt").write_text(
            "".join(f"{r['sample']}: {r['reason']}\n" for r in rs))
        (out / f"{model}_missing_reasons.json").write_text(
            json.dumps(rs, indent=2))
    (out / "ALL_missing_reasons.json").write_text(json.dumps(rows, indent=2))

    print(f"{'model':16s} missing  reasons")
    for model, rs in sorted(by_model.items()):
        c = defaultdict(int)
        for r in rs:
            c[r["reason_key"]] += 1
        print(f"{model:16s} {len(rs):5d}    " +
              ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
