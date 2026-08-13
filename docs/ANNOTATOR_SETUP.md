# Annotation tool — setup, data format, and ingestion

Everything needed to run the tool on your own machine, no Docker. Target is
~20 minutes, most of it waiting on TeX Live.

---

## 1. System packages

Two things pip cannot install. Both are needed for the **TikZ** target; skip
them only if you will exclusively annotate SVG.

| | why |
|---|---|
| **TeX Live** (`latexmk`, `pdflatex`, `pgf`/`tikz`, `animate`, `standalone`) | compiles diagram and animation code |
| **poppler** (`pdftoppm`) | rasterises the compiled PDF into frames |

**macOS**
```bash
brew install --cask mactex-no-gui     # ~4 GB, or basictex + the packages below
brew install poppler
```
With `basictex` instead, add the packages the pipeline uses:
```bash
sudo tlmgr update --self
sudo tlmgr install latexmk standalone animate pgf preview amsmath bm xcolor
```

**Ubuntu / Debian**
```bash
sudo apt update
sudo apt install -y texlive-latex-extra texlive-pictures latexmk poppler-utils
```

**Verify** — all four must print a path:
```bash
for c in latexmk pdflatex pdftoppm python3; do printf '%-10s %s\n' "$c" "$(command -v $c || echo MISSING)"; done
```

> `ffmpeg` is **not** required. `imageio-ffmpeg` ships its own binary.

---

## 2. Python environment

Python **3.10+**.

```bash
git clone <repo-url> img_2_svg_pretraining
cd img_2_svg_pretraining

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -U pip
pip install -r requirements-annotate.txt
pip install -e . --no-deps           # puts the package on the path
```

`--no-deps` matters. Without it pip installs `pyproject.toml`'s full list —
torch, transformers, flash-attn, accelerate — several GB the annotation tool
never imports. Those exist for local open-weights inference; `hf_local.py`
imports torch *inside* the loader function, so a Gemini-only config never
reaches it.

**Optional, SVG only.** SVG animations use CSS `@keyframes`, which no
rasterizer can seek, so their frames come from a headless browser:
```bash
playwright install chromium          # ~150 MB
```
Skip it and SVG still annotates; only SVG frame export fails.

**Verify:**
```bash
python -c "import flask, yaml, filelock, PIL, cairosvg, cv2, fitz, pptx, google.genai; print('ok')"
```

---

## 3. API key

The tool calls Gemini for every stage that runs a model. Reviewing existing
artifacts, rendering and compiling need no key — only re-running a stage does.

Pick **one** of these; they are tried in this order:

**A. `api_keys.csv` in the repo root** (what this project uses — pools several
keys and rotates on quota errors):
```csv
Key, Service
AIzaSy...your-key..., Google
AIzaSy...second-key..., Google
```

**B. Environment variable:**
```bash
export GOOGLE_API_KEY=AIzaSy...
```

**C. Inline in a config** under `backends:` — `api_key:` or `api_key_env:`.

> `api_keys.csv` is gitignored. Do not commit a key.

**Verify:**
```bash
python -c "
from img_2_svg_pretraining.pipeline.backends.keys import resolve_keys
print(len(resolve_keys('gemini', {}, 'GOOGLE_API_KEY')), 'Google key(s) found')"
```

---

## 4. Data format

One directory per sample under a dataset root. The directory name **is** the
sample id, and appears throughout the tool and on every artifact path.

```
data/my_benchmark/
├── CVPR_2025_pipe00002/
│   ├── CVPR_2025_pipe00002.png      ← REQUIRED: the figure
│   ├── caption.txt                  ← optional
│   ├── abstract.tex                 ← optional
│   ├── methods.tex                  ← optional
│   ├── title.txt                    ← optional
│   └── arxiv_src/                   ← optional: LaTeX source tree
└── CVPR_2025_pipe00003/
    └── ...
```

**The image is the only hard requirement.** A directory without one is skipped
silently — that is the first thing to check if a sample does not appear.

| file | accepted as | used for |
|---|---|---|
| `<id>.png` | `.png .jpg .jpeg .webp` | the figure being annotated |
| `caption` | `.tex .txt .md` | narration context |
| `abstract` | `.tex .txt .md` | narration context |
| `methods` | `.tex .txt .md` | narration context |
| `title` | `.tex .txt .md` | display name; else parsed from `arxiv_src` |
| `arxiv_src/` | directory | title extraction |

Naming notes worth knowing:

- The image is found as `<id>.<ext>` first. Failing that, a **single** loose
  image in the directory is accepted — deliberate tolerance, since one sample
  here has a typo'd filename (`...pipe000011.png`, an extra zero).
- Context files are optional per sample. Missing ones reduce what narration
  can draw on; nothing errors.
- Ids should avoid spaces and `/` — they become path components.

---

## 5. Point the tool at your data

Copy the config and edit the root:

```bash
cp src/img_2_svg_pretraining/pipeline/configs/default.yaml \
   src/img_2_svg_pretraining/pipeline/configs/local.yaml
```

```yaml
dataset:
  root: /absolute/path/to/data/my_benchmark    # ← the only line you must change
  limit: 0                                     # 0 = all samples

animation_style: progressive_reveal
target: tikz                                   # tikz | svg
```

Everything else — models, prompts, token limits — is already set.

**Verify the tool sees your samples:**
```bash
python -m img_2_svg_pretraining.pipeline.run_pipeline samples \
  --config src/img_2_svg_pretraining/pipeline/configs/local.yaml
```

---

## 6. Ingest: generate the artifacts to review

The tool reviews pipeline output, so each sample needs artifacts before there
is anything to annotate. From the repo root:

```bash
# Stage 1 — image -> diagram code -> rasters -> critique
python -m img_2_svg_pretraining.pipeline.run_pipeline stage1 \
  --config src/img_2_svg_pretraining/pipeline/configs/local.yaml

# Stage 2 — structure XML -> sequence -> critique -> narration
python -m img_2_svg_pretraining.pipeline.run_pipeline stage2 \
  --config src/img_2_svg_pretraining/pipeline/configs/local.yaml \
  --style progressive_reveal

# Stage 3 — animation code -> critique -> export
python -m img_2_svg_pretraining.pipeline.run_pipeline stage3 \
  --config src/img_2_svg_pretraining/pipeline/configs/local.yaml \
  --style progressive_reveal
```

Useful flags: `--only <id> [<id> ...]` for a subset, `--limit N` for the first
N, `--force` to recompute cached artifacts.

Budget roughly **3–6 minutes per sample** for all three stages, mostly model
latency. Start with `--limit 2` to confirm the setup end to end before
committing to a full run.

Artifacts land under `src/img_2_svg_pretraining/pipeline/cache/<dataset-name>/`,
keyed by a lineage string encoding the models that produced them — so two
configs never overwrite each other.

---

## 7. Run the tool

```bash
python -m img_2_svg_pretraining.pipeline.annotate.app \
  --config src/img_2_svg_pretraining/pipeline/configs/local.yaml \
  --port 8602 \
  --styles progressive_reveal
```

Open **http://localhost:8602**.

| flag | effect |
|---|---|
| `--styles progressive_reveal` | pin the animation style; omit and each sample draws randomly from all 8 |
| `--only <id> ...` | restrict this instance to specific samples |
| `--shard 1/4` | take the 1st of 4 deterministic shards — how several annotators split a set without overlapping |
| `--host 127.0.0.1` | local only (default `0.0.0.0` is reachable from your network) |

---

## Splitting work across annotators

Each person runs their own instance against their own copy of the data, using
`--shard i/N`:

```bash
# annotator 1 of 3
--shard 1/3
# annotator 2 of 3
--shard 2/3
```

Sharding is deterministic on the sample id, so the three sets are disjoint with
no coordination. Merge afterwards by collecting each person's
`cache/<dataset>/annotations/*.json` plus the `*_human/` directories — they are
per-sample files, so a union is a straight copy with no conflicts.

---

## Troubleshooting

**A sample is missing from the list.** Its directory has no readable image, or
has several images and none named `<id>.<ext>`.

**"no API key" on any run.** See §3, then verify with the snippet there.
Reviewing existing artifacts never needs a key — only re-running a stage does.

**Rendering fails, everything else works.** TeX Live is missing or incomplete.
`latexmk -v` and `pdflatex -v` must both succeed. On a `basictex` install, the
`tlmgr install` line in §1 is usually what is missing.

**The frame deck stays empty for an SVG sample.** `playwright install chromium`
was not run.

**A gate will not open.** Both halves must pass — the machine check *and* your
approval. The panel lists exactly which is unmet.

**A stage refuses to run (HTTP 409).** An earlier stage is incomplete; the
message names it. Re-running an *earlier* stage is always allowed — that is how
you fix what is blocking you.
