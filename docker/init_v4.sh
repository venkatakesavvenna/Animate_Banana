#!/usr/bin/env bash
# The v4 container: runs the pipeline AND serves vLLM, on one node.
#
# WHY NOT docker/init.sh
# ----------------------
# That script builds `img-2-svg-pretraining` from docker/Dockerfile
# (FROM nvcr.io/nvidia/pytorch:25.03-py3 + a full TeX Live). Three reasons not
# to here:
#
#   1. The v4 bench is SVG-only. SVG animations render through headless
#      chromium, not LaTeX, so the entire TeX Live layer -- the slowest part of
#      that build -- would be dead weight.
#   2. The 25.03 base is not on this node; 25.02 is, already pulled (37GB).
#   3. **25.02 ships Python 3.12.3**, which is exactly what every venv under
#      /environments was built against (`pyvenv.cfg: version = 3.12.3`). That
#      match is the whole ballgame: on a host with only python3.10 those venvs'
#      site-packages silently vanish from sys.path and their editable install
#      of /code/src resolves to nothing. docs/INFRA.md lists this as failure
#      mode #2 -- "check python3 -V inside the container before debugging
#      anything else".
#
# WHY --network host
# ------------------
# vLLM binds 127.0.0.1. A bridge-networked container cannot reach it -- that is
# recorded as a live failure in scripts/run_qwen_intermediate.sh. Host
# networking also means the -p mappings in init.sh are meaningless here, and
# that ports are the whole node's: 8010/8011 are used rather than 8000, which
# something else on this node already listens on.
#
# WHY EVERY CACHE PATH IS AN EXPLICIT -e
# --------------------------------------
# Modelled on vlm-ingest-pipeline-aryanjain.intern, the container that actually
# serves vLLM successfully on this node. The image PRESETS HF_HUB_CACHE, and it
# overrides HF_HOME -- so setting HF_HOME alone sends a 200GB download somewhere
# unintended, which INFRA.md records happening into a colleague's cache
# directory. Setting all of them makes that mistake unrepresentable rather than
# merely documented. TMPDIR, Triton and Inductor caches join them on node-local
# NVMe so JIT artifacts stay off Lustre and out of the container's overlay.
#
# WHY /opt/dlami/nvme AND NOT /fsxvision_new FOR WEIGHTS
# ------------------------------------------------------
# NVMe is node-local and fast (~1.2GB/s pulls) with 2.7T free; /fsxvision_new is
# shared Lustre with only 2.5T free. The trade is that weights do not survive a
# node move -- which is exactly how the previous 440GB Qwen cache was lost.
set -euo pipefail

USER_NAME=venkat.kesav
CONTAINER=animatebanana-v4
IMAGE=nvcr.io/nvidia/pytorch:25.02-py3
REPO=/fsxvision_new/${USER_NAME}/img_2_svg_pretraining
NVME=/opt/dlami/nvme/${USER_NAME}

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "✓ $CONTAINER exists"
  docker ps --format '{{.Names}}' | grep -qx "$CONTAINER" || docker start "$CONTAINER"
else
  mkdir -p "$NVME"/{hf_cache,vllm_cache,triton_cache,tmp}
  # Owned by us, before any container writes here: the pipeline/cache tree is
  # root-owned from an earlier root container and there is no passwordless sudo
  # on this node to chown it back (INFRA.md #4).
  mkdir -p "$REPO"/data/animatebench_v4_cache "$REPO"/logs/bench_v4

  docker run -d --name "$CONTAINER" \
    --gpus all --network host --shm-size 64g \
    -e HF_HOME=/hf_cache \
    -e HF_HUB_CACHE=/hf_cache/hub \
    -e HUGGINGFACE_HUB_CACHE=/hf_cache/hub \
    -e HF_DATASETS_CACHE=/hf_cache/datasets \
    -e XDG_CACHE_HOME=/hf_cache/xdg \
    -e TMPDIR=/tmp \
    -e TRITON_CACHE_DIR=/triton_cache \
    -e VLLM_CACHE_ROOT=/vllm_cache \
    -e TORCHINDUCTOR_CACHE_DIR=/vllm_cache/torch_inductor \
    -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers \
    -e PYTHONPATH=/code/src \
    -v "$NVME/hf_cache":/hf_cache \
    -v "$NVME/vllm_cache":/vllm_cache \
    -v "$NVME/triton_cache":/triton_cache \
    -v "$NVME/tmp":/tmp \
    -v /fsxvision_new:/fsxvision_new \
    -v "$REPO":/code \
    -v /fsxvision_new/${USER_NAME}/environments:/environments \
    -v /opt/dlami/nvme:/opt/dlami/nvme \
    -w /code "$IMAGE" sleep infinity
  echo "✓ $CONTAINER created"

  # /tmp is a bind mount, so it arrives with the host directory's 0775 and our
  # uid -- not the 1777 every /tmp is assumed to have. apt drops privileges to
  # the `_apt` user, which then cannot write there, and the failure surfaces as
  # "repository is not signed" rather than as a permissions error.
  docker exec "$CONTAINER" chmod 1777 /tmp

  # cairosvg is a CFFI binding, so pip installing it into the venv did not
  # bring the C library. It is the STATIC svg renderer (pipeline/svg_render.py)
  # behind stage 1's `csr` and `rendering_fidelity`, so without this those two
  # metrics are zero for every sample and the cause is a dlopen error buried in
  # a compile log. Animated frames go through chromium instead and are
  # unaffected, which is what would make this easy to miss.
  echo "installing libcairo2 (cairosvg's C library)..."
  docker exec "$CONTAINER" bash -lc \
    'apt-get update -qq && apt-get install -y -qq --no-install-recommends libcairo2' >/dev/null

  # Same shape of gap on the animated side: the browser BINARIES are already on
  # disk under /environments/playwright-browsers, but a bare CUDA image has
  # none of the X/NSS/GTK libraries chrome links against, so it dies at exec
  # with "libnspr4.so: cannot open shared object file". `install-deps` is
  # playwright's own list for exactly this, and is preferred over guessing at
  # package names.
  echo "installing chromium's system libraries..."
  docker exec "$CONTAINER" bash -lc \
    'PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers \
     /environments/img_2_svg_pretraining/bin/python -m playwright install-deps chromium' >/dev/null

  # The API clients. Absent from the pipeline venv itself -- on the HOST they
  # appear to be present, but that is an illusion: the host has no python3.12,
  # so `<venv>/bin/python` falls through to the system 3.10 and picks up its
  # dist-packages. Inside the container the venv is real, and `openai` is not
  # in it. `openai_compat` is how every served model is reached, so without
  # this NOTHING can talk to vLLM -- and the error arrives only at the first
  # model call, after a seven-minute load.
  echo "installing API clients into the pipeline venv..."
  docker exec "$CONTAINER" bash -lc \
    '/environments/img_2_svg_pretraining/bin/pip install --quiet \
       openai google-genai anthropic' >/dev/null

  # The gemma4 venv. Not the default serving env -- ocr_env_vllm is -- but it is
  # the ONLY one that can serve google/gemma-4-31B-it, because vLLM defers config
  # parsing to transformers and only 5.13.0 knows the `gemma4` architecture.
  #
  # Three packages, all from the same cause: the venv is --system-site-packages
  # and was built against a different base image, so it shadows some of this
  # image's packages while inheriting others, and the halves disagree.
  #   scipy            its numpy 2.4.4 vs the image's numpy-1.x-built scipy
  #   aiohappyeyeballs its aiohttp needs a newer one than the image ships
  #   distro           its openai client imports it; the image has no copy
  echo "repairing the gemma4 venv (needed only for google/gemma-4-31B-it)..."
  docker exec "$CONTAINER" bash -lc \
    '/environments/gemma4/bin/pip install --quiet "scipy>=1.14" -U aiohappyeyeballs distro' \
    >/dev/null 2>&1 || true
fi

echo
echo "--- sanity ---"
# The 3.12.3 check first, deliberately: everything below it is meaningless if
# the interpreter does not match what the venvs were built against.
docker exec "$CONTAINER" python3 -V
docker exec "$CONTAINER" /environments/img_2_svg_pretraining/bin/python -c \
  "import playwright, cairosvg, lxml; print('pipeline venv: playwright + cairosvg + lxml ok')"
docker exec "$CONTAINER" /environments/ocr_env_vllm/bin/python -c \
  "import torch, vllm; print(f'serving venv: torch {torch.__version__}, vllm {vllm.__version__}')"

cat <<EOF

--- how to use it ---
Pipeline / eval (note -u: writes must land as you, not root):
  docker exec -u \$(id -u):\$(id -g) $CONTAINER bash -lc \\
    'cd /code && /environments/img_2_svg_pretraining/bin/python -u -m <module> ...'

Serve a generation model (needs the GPUs free):
  docker exec -d $CONTAINER bash -lc \\
    '/environments/ocr_env_vllm/bin/python -m vllm.entrypoints.openai.api_server \\
       --model <repo> --served-model-name <repo> --tensor-parallel-size 8 \\
       --gpu-memory-utilization 0.90 --max-model-len 65536 \\
       --limit-mm-per-prompt.image 2 --max-num-seqs 16 \\
       --port 8011 --host 127.0.0.1 > /tmp/vllm_gen.log 2>&1'
EOF
