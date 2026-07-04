# TikZ Sample Viewer

Rudimentary viewer for scrutinizing and correcting TikZ code against ground-truth
architecture diagram images (Stage 1 of the pipeline: diagram image -> TikZ).

Scans a data directory recursively for `.tex` files, pairs each with a
best-guess ground-truth image (same basename, or a conventional name like
`input.png`/`image.png` in the same folder, or the first image under a
`crops/` subfolder), and lets you inspect/edit/recompile the TikZ side by
side with the source image and a rendered preview.

## Run (inside the project docker container)

```bash
cd /code/viewer
pip install -r requirements.txt
python app.py --data-root /code/examples --port 7860
```

Then open `http://<host>:7860`.

Point `--data-root` at `/code/data/<extracted-set>` once the real Stage-1
dataset is unpacked; the scanner does not assume any particular naming
convention beyond "a `.tex` file plus a nearby image".

## Requirements

Compiling TikZ requires a LaTeX toolchain (`latexmk`/`pdflatex`) and
poppler's `pdftoppm`, both installed via `docker/Dockerfile`. This is why the
viewer must be run inside the container, not on bare metal.

## Notes

- Compiled PNGs are cached in `viewer/cache/` keyed by a hash of the TikZ
  source, so re-viewing an unmodified sample is instant.
- "Save" writes the edited TikZ back to the original `.tex` file on disk —
  use it once you've corrected a sample.
