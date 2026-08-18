# Quick Start Guide

> **This guide is only about how to run the repo.**
> For full architecture, system design, and in-depth understanding, refer to [`src/vision_ingest/README.md`](src/vision_ingest/README.md).

---

## Overview

The reference entrypoint is [`examples/vllm_testing/main.py`](examples/vllm_testing/main.py). All steps below follow the docstring in that file exactly.

The pipeline requires a **PostgreSQL server** running on one host node, and then `main.py` is run on **all worker nodes** (including the host node itself).

---

## Step 1: Environment Setup

Go to `bash_scripts/` and open `docker_env_setup.sh`. Change `USER_NAME` to your username, then run:

```bash
cd bash_scripts
bash docker_env_setup.sh
```

---

## Step 2: Start the PostgreSQL DB Server (Host Node Only)

Pick one node to be the **host**. Get its IP address — you will need it in all subsequent steps.

**First, clear any existing data directory (if starting fresh):**

```bash
rm -rf /opt/dlami/nvme/<your_username>/db_test
```

**Then start the server:**

```bash
python -m vision_ingest.drivers.db_driver_postgres serve \
  --data-dir /opt/dlami/nvme/<your_username>/db_test \
  --pg-host <HOST_NODE_IP> \
  --pg-port 5432 \
  --pg-dbname vllm_testing \
  --pg-user pipeline
```

> **Important notes:**
> - Run this command **only once**, on the host node only. Keep this terminal alive — the server runs as long as this process is running.
> - `--data-dir` **must be a local NVMe path** (e.g. `/opt/dlami/nvme/...`). Do not use a shared filesystem path like `/fsxvision` — it will be very slow.
> - Note down `<HOST_NODE_IP>` — you will use the same host IP in `build_args()` on every worker node.

---

## Step 3: Write Your Project-Specific Code (`project_specific.py`)

Every project requires a custom `PromptAndValidation` class in [`examples/vllm_testing/project_specific.py`](examples/vllm_testing/project_specific.py). The three methods to override are:

- `get_modified_image_path(image_path)` — return modified path(s) to send to the VLM (e.g. SOM image, multi-image list)
- `preprocess_vlm_output(raw_vlm_output)` — clean/extract the raw VLM response (strip think tokens, parse JSON, etc.)
- `validate_output(processed_output)` — raise `ValueError` if output is invalid (triggers a retry); return `True` if OK

The defaults (pass-through path, strip whitespace, reject empty) work out of the box for basic use. Override only what you need.

Make sure `main.py` imports from this file at the top:

```python
from project_specific import PromptAndValidation
```

> This import assumes you run `python main.py` from inside `examples/vllm_testing/` so that `project_specific.py` is on the Python path.

---

## Step 4: Configure `build_args()` in `main.py`

Open [`examples/vllm_testing/main.py`](examples/vllm_testing/main.py) and edit the `build_args()` function. The key fields to update:

```python
def build_args():
    return SimpleNamespace(
        # DB connection — must match the serve command above
        pg_host = "<HOST_NODE_IP>",      # IP of the node running the DB server
        pg_port = "5432",
        pg_dbname = "vllm_testing",
        pg_user = "pipeline",
        pg_password = None,              # None = trust auth (no password)

        # Logs path — MUST be local NVMe, not fsxvision (os.fsync is called here)
        logs_path = "/opt/dlami/nvme/<your_username>/vllm_testing_logs/",

        # JSONL output — can be on fsxvision (written only at fixed checkpoints)
        jsonl_output_path = "/fsxvision_new/<your_username>/Vision-Ingestion-Engine/examples/vllm_testing/jsonl_outputs",

        # Image paths source — file with one image path per line
        image_paths_source = "/path/to/images.txt",
        ingest_batch_size = 100000,      # Batch size for DB inserts (can be high)
        use_copy = False,                # True = faster COPY insert, but no dedup

        run_ingest = True,               # Set True on first run, False to skip re-ingestion

        # Processing
        batch_size = 32,
        local_json_threads = 1,
        fsync_every_lines = 250,
        fetch_state = 0,

        # VLM
        vlm_model_name = "qwen_3_32b",
        vlm_gpus = "0,1,2,3,4,5,6,7",
        vlm_config_path = "/path/to/Vision-Ingestion-Engine/examples/vllm_testing/vllm_model.yaml",
        prompt_path = "/path/to/prompts/v1.md",
    )
```

> **`run_ingest` flag:** Set to `True` on the first run (or whenever you want to ingest new image paths). Set to `False` on subsequent runs to skip re-ingestion and go straight to processing.

---

## Step 5: Prepare Your Image List

For large datasets, always generate an image list file upfront (`os.walk` is significantly slower):

```bash
find /path/to/your/images -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) > images.txt
```

Set `image_paths_source` in `build_args()` to the full path of this `images.txt`.

---

## Step 6: Configure the VLM Model (`vllm_model.yaml`)

The model config file lives at [`examples/vllm_testing/vllm_model.yaml`](examples/vllm_testing/vllm_model.yaml). It already has entries for common models. Add or edit an entry for your model:

```yaml
models:
  /path/to/your/model:
    is_llm: false          # false for VLMs (vision), true for text-only LLMs
    engine_args:
      limit_mm_per_prompt:
        image: 1
      gpu_memory_utilization: 0.95
```

> **Note:** `tensor_parallel_size` is automatically set based on the number of GPUs you pass via `vlm_gpus`. No need to set it manually unless you want to override.

Set `vlm_config_path` in `build_args()` to the full path of this YAML file, and `vlm_model_name` to the key you defined.

---

## Step 7: Run on All Nodes

`cd` into `examples/vllm_testing` and run `main.py` on **every node** you want to use for processing — including the host node:

```bash
cd examples/vllm_testing
python main.py
```

> All nodes must use the same `pg_host`, `pg_port`, `pg_dbname`, and `pg_user` values — the same ones you used in the `serve` command on the host.

---

## Key Arguments Reference

All arguments are set inside `build_args()` in your `main.py` (not via CLI flags):

| Argument | Description | Default |
|----------|-------------|---------|
| `pg_host` | IP/hostname of the PostgreSQL server | **Required** |
| `pg_port` | PostgreSQL port | `5432` |
| `pg_dbname` | Database name | **Required** |
| `pg_user` | PostgreSQL user | **Required** |
| `pg_password` | Password (`None` for trust auth) | `None` |
| `logs_path` | Directory for logs — **use local NVMe** | **Required** |
| `jsonl_output_path` | Directory for JSONL output shards | **Required** |
| `image_paths_source` | File with image paths (one per line) or folder path | **Required** |
| `ingest_batch_size` | Batch size for DB inserts | `100000` |
| `use_copy` | Use `COPY FROM STDIN` for faster insert (no dedup) | `False` |
| `run_ingest` | Whether to run ingestion before processing | `True` |
| `batch_size` | Images per batch for VLM inference | `32` |
| `local_json_threads` | Threads for reading/writing per-image JSONs | `1` |
| `fsync_every_lines` | Lines to write before fsyncing (crash boundary) | `250` |
| `fetch_state` | State to fetch from DB (`0` = pending) | `0` |
| `vlm_model_name` | Model name key in `vllm_model.yaml` | `qwen_3_32b` |
| `vlm_gpus` | Comma-separated GPU IDs | `0,1,2,3,4,5,6,7` |
| `vlm_config_path` | Path to `vllm_model.yaml` | **Required** |
| `prompt_path` | Path to prompt template file | **Required** |

---

## Output Structure

```
{logs_path}/{node-hostname}/{timestamp}/
├── main.log
├── vlm_prediction.log
├── jsonl_writer.log
└── recovery.log

{jsonl_output_path}/{node-hostname}/
├── results_00000.jsonl    # Output shard (rotates at 1GB)
└── results_00001.jsonl
```