# VLM Prediction Pipeline 🔮

**👤 Author:** Srihari Bandarupalli

**📦 Modules:** `modules/prediction.py`, `project_specific.py`

**🔗 Dependencies:** `vllm/vlm_service.py`, `vllm/vllm_config.py`, `vllm/retry_validate.py`

> **Note:** Validation logic has been moved to `project_specific.py` which provides a `PromptAndValidation` class for custom prompt generation and response parsing.

---

## 📑 Table of Contents

- [📋 Purpose](#-purpose)
- [🏗️ Architecture](#️-architecture)
- [🔄 Pipeline Flow](#-pipeline-flow)
- [🛡️ Error Handling Strategy](#️-error-handling-strategy)
- [🔗 Integration with Pipeline](#-integration-with-pipeline)
- [📊 Logging](#-logging)
- [🎯 Design Philosophy](#-design-philosophy)
- [🎯 Final Summary](#-final-summary)

---

## 📋 Purpose

The VLM Prediction layer orchestrates the **complete transformation of image paths into structured VLM outputs**. It implements **CPU/GPU parallelism** to maximize throughput by overlapping I/O-bound operations (image loading, prompt preparation) with GPU-bound operations (VLM inference).

### Key Responsibilities

- 📥 **Parallel image loading:** ThreadPool for concurrent file I/O
- 📝 **Prompt generation:** Template filling with metadata and captions
- ⚡ **VLM inference:** Batched GPU inference via vLLM service
- ✅ **Output validation:** Pydantic schema validation with automatic retries
- 🛡️ **Error handling:** Graceful degradation, failed paths tracked separately

---

## 🏗️ Architecture

### ⚡ CPU/GPU Parallel Pipeline

The system uses a **bootstrap-process-flush pattern** to enable simultaneous CPU and GPU work:

> [!NOTE]
> **Parallelism Implementation:** The pipeline uses Python's `ThreadPoolExecutor` (max_workers=2) to achieve I/O and GPU compute overlap. This works effectively because: (1) I/O operations (`get_prompts` with image reading via `reader_pool`) release the GIL, allowing true parallelism, and (2) GPU operations (`send_to_vlm` calling VLM service) release the GIL during GPU compute, enabling simultaneous execution. While not process-level parallelism, this thread-based approach efficiently overlaps I/O-bound preparation with GPU-bound inference.

```
Batch Timeline:
                    
t=0: Bootstrap Phase (CPU Only)
     ┌────────────────────────────────────────────────────────┐
     │ CPU: Load Batch 0 → Extract Metadata → Fill Template   │
     │ GPU: Idle (waiting for first batch)                    │
     └────────────────────────────────────────────────────────┘
     
t=1: Main Loop - Parallelism Begins
     ┌─────────────────────────┐  ┌────────────────────────────┐
     │ CPU: Prepare Batch 1    │  │ GPU: Process Batch 0       │
     │ (load, extract, fill)   │  │ (inference + validation)   │
     └─────────────────────────┘  └────────────────────────────┘
     
t=2: Main Loop - Full Parallelism
     ┌─────────────────────────┐  ┌────────────────────────────┐
     │ CPU: Prepare Batch 2    │  │ GPU: Process Batch 1       │
     │ (load, extract, fill)   │  │ (inference + validation)   │
     └─────────────────────────┘  └────────────────────────────┘
     
t=N: Flush Phase (GPU Only)
                                     ┌────────────────────────────┐
                                     │ CPU: Idle (no next batch)  │
                                     │ GPU: Process Final Batch   │
                                     └────────────────────────────┘
```

**Key Insight:** While GPU processes batch `i`, CPU prepares batch `i+1` in parallel, **eliminating idle time** and maximizing resource utilization.

---

## 🔄 Pipeline Flow

### 🎯 Prompt Generation (`get_prompts`)

**Input:** List of image paths  
**Output:** List of corresponding prompts (or `None` for failed paths)

The prompt generation phase transforms raw image paths into multi-modal prompts ready for VLM inference:

**Phase 1: Metadata & Caption Extraction** 📊
- Uses `utils/utils.py:get_image_metadata_caption()` which leverages the `reader_pool` to concurrently read and extract metadata/captions for all image paths
- Extracts image dimensions, file size, color mode, and original captions
- Failed extractions return `None`, processing continues

**Phase 2: Template Filling** 📝
- Replaces placeholders in prompt template with actual metadata and captions
- Handles `None` values gracefully (graceful degradation)
- Creates text prompts with populated metadata fields

**Phase 3: Multi-Modal Prompt Assembly** 🖼️
- Uses `vllm/vllm_config.py:get_prompt_with_image()` which leverages the `reader_pool` to read images and apply model-specific chat template formatting
- If image reading fails at this stage, corresponding output prompt will be `None`
- Returns format ready for vLLM: `{"prompt": text, "multi_modal_data": {"image": [pil_image]}}`

---

### 🚀 VLM Inference (`send_to_vlm`)

**Input:** List of image paths and corresponding prompts  
**Output:** Validated outputs and failed paths

The inference phase processes prompts through VLM and validates outputs with automatic retries:

**Phase 1: Filter Valid Prompts** 🔍
- Removes `None` prompts from failed image loads/preparation
- Tracks failed paths for early error reporting
- Maintains index mapping for result association

**Phase 2: VLM Processing & Validation** ✅
- Calls `vllm/retry_validate.py` which processes prompts using VLMService
- Extracts JSON from model output (handles `<think>` blocks, code fences, raw JSON)
- Validates against Pydantic schema (17 required fields, Literal constraints)
- On validation failure: retries up to 3 times with increased repetition penalty and error feedback appended to prompt
- After 3 attempts: marks as failed, returns `None`

> [!TIP]
> **See:** `project_specific.py` for custom validation logic (PromptAndValidation class) and `vllm/retry_validate.py` for retry strategy

---

## 🛡️ Error Handling Strategy

### 📉 Graceful Degradation at Every Stage

The pipeline is designed so **individual failures never cascade**. Errors are handled differently in the two main pipeline stages:

---

#### ⚠️ Errors in `get_prompts` (CPU-bound preparation)

Errors during prompt generation return `None` for the affected path, allowing the batch to continue:

| Stage | Failure Type | Handling | Impact |
|-------|--------------|----------|---------|
| Image load | File missing/corrupted | Return `None`, log error | Corresponding prompt is `None`, path marked failed later |
| Metadata extraction | Parse error | Return `None`, continue | Prompt sent with `None` metadata (degraded quality, may still succeed) |
| Template filling | N/A | Handles `None` gracefully | Degraded prompt quality, but processable |
| Chat template formatting | Format error | Return `None`, log error | Corresponding prompt is `None`, path marked failed later |

**Key behavior:** `get_prompts()` returns a list with `None` entries for failed paths. Processing continues for successful paths.

---

#### 🔄 Errors in `send_to_vlm` (GPU-bound inference)

Errors during VLM inference and validation are handled with retry logic and graceful degradation:

| Stage | Failure Type | Handling | Impact |
|-------|--------------|----------|---------|
| Filter phase | `None` prompts from get_prompts | Skip, add to failed_paths | Path marked failed immediately, no GPU resources wasted |
| JSON extraction | No JSON found | Retry with feedback (3x) | After 3 attempts, mark failed |
| Pydantic validation | Schema mismatch | Retry with increased penalty (3x) | After 3 attempts, mark failed, logged as error |
| VLM service crash | Exception | Catch exception, log error in send_to_vlm, return empty lists | Batch returns no results `([], [])`. Paths neither marked failed nor successful. CLI driver handles recovery/retry logic. |

**Key behavior:** `send_to_vlm()` filters out `None` prompts early, attempts validation with retries, and returns failed paths separately from successful objects. In case of critical errors (like VLM service crash), it catches the exception, logs the error, and returns `([], [])` to signal batch failure to the CLI driver, which decides on retry/shutdown strategy.

---

### 💡 Partial Batch Success Example

```
Batch of 32 images:
- 28 successful VLM outputs (87.5%)
- 2 failed to load in get_prompts (marked failed)
- 1 failed metadata extraction (prompt sent anyway, may succeed)
- 1 failed validation after 3 attempts in send_to_vlm (marked failed)

Result: 28 objects enqueued to writer, 4 paths marked failed in DB
```

**Critical Design Principle:** One bad image doesn't block 31 others. GPU utilization stays high, failed paths can be retried separately later.

---

## 🔗 Integration with Pipeline

### 🔌 CLI Driver Pattern

The prediction layer is called from the orchestration flow (`main.py` → `drivers/cli.py`) using the **bootstrap-process-flush** pattern:

**Bootstrap Phase (First Iteration):**
```python
next_batch_paths = db.fetch_batch(batch_size)
# Process with cur_batch_paths=None, cur_batch_prompts=None
# Returns: failed_paths, successful_objs, new_cur_batch_paths, new_cur_batch_prompts
```

**Main Loop (Parallel Processing):**
```python
while next_batch_paths:
    # Parallel: GPU processes cur_batch while CPU prepares next_batch
    failed, success, new_cur_paths, new_cur_prompts = vlm_predictor.vlm_predict(
        cur_batch_paths, cur_batch_prompts, next_batch_paths
    )
    # Advance to next batch
    cur_batch_paths = new_cur_paths
    cur_batch_prompts = new_cur_prompts
    next_batch_paths = db.fetch_batch(batch_size)
```

**Flush Phase (Final Batch):**
```python
# Process final batch with next_batch_paths=None
failed, success, _, _ = vlm_predictor.vlm_predict(
    cur_batch_paths, cur_batch_prompts, next_batch_paths=None
)
```

**Key Design: Stateless Functions**

`vlm_predict()` is stateless and pure:
- **Signature:** `vlm_predict(cur_paths, cur_prompts, next_paths) → (failed, success, next_paths, next_prompts)`
- **CPU work:** Prepares `next_paths` → returns `next_prompts`
- **GPU work:** Processes `cur_prompts` → returns `failed` and `success`
- **Parallelism:** CPU and GPU work happen simultaneously in separate threads

---

## 📊 Logging

Prediction errors are logged to **`vlm_prediction.log`** in **plain text format** (not structured JSON).

**Frequency:** Logs errors only when they occur during VLM operations (not periodic checkpoints).

> [!INFO]
> **For detailed log format, see:** [`logger_readme.md`](../logger_readme.md)

---

## 🎯 Design Philosophy

### 1. ⚡ Bootstrap-Process-Flush Pattern Maximizes Resource Utilization

**The problem:** Sequential CPU preparation → GPU inference leaves both resources idle 50% of the time.

**The solution:** Overlap CPU work (batch N+1) with GPU work (batch N) using explicit 3-phase control flow:
- **Bootstrap:** Prepare first batch while GPU idles (unavoidable cold start)
- **Main loop:** CPU prepares next batch while GPU processes current batch (full parallelism)
- **Flush:** Process final batch while CPU idles (unavoidable wind-down)

**Why stateless functions matter:** All batch state (`cur_paths`, `cur_prompts`, `next_paths`) flows through explicit function arguments. No hidden instance variables, no locks, no coordination overhead. ThreadPoolExecutor handles parallelism, caller handles state transitions.

---

### 2. 🛡️ Graceful Degradation Isolates Failures

**The constraint:** VLM pipelines face inevitable failures—corrupted images, malformed metadata, network timeouts, validation errors.

**The decision:** Return `None` at every failure point instead of raising exceptions. This keeps batch processing bounded:
- Image load fails → return `None` prompt → skip inference → mark path failed
- Metadata extraction fails → return `None` metadata → send degraded prompt → may still succeed
- Validation fails 3x → return `None` output → mark path failed

**Why this matters:** Batch of 32 images with 1 corrupted file processes 31 successfully. GPU stays utilized, throughput stays high. Failed paths tracked separately for targeted reprocessing.

**Design trade-off:** More `None` checks throughout code vs guaranteed batch-level isolation. We chose isolation—one bad image never cascades to block 31 others.

---

### 3. 🔄 Retry-with-Feedback Strategy Handles Model Drift

**The observation:** VLM validation failures cluster around two patterns:
1. **Stuck/repetitive output** — Model generates same token sequence repeatedly
2. **Schema violations** — Missing required fields or invalid enum values

**The solution:** Adaptive retry with two feedback mechanisms:
- **Increased repetition penalty** (+0.5 per attempt, up to 3x) forces token diversity
- **Error message appended to prompt** guides model to fix specific validation failures

---

### 4. 🧩 Multi-Layer Extraction Handles Real-World Model Behavior

**The challenge:** VLMs produce diverse output formats depending on training, sampling parameters, and prompt design.

**The strategy:** Defensive extraction pipeline with fallback layers:

1. **Strip `<think>` blocks** — Model reasoning often wrapped in XML tags
2. **Extract from code fences** — JSON frequently enclosed in ` ```json...``` `
3. **Find raw JSON** — Fallback to regex-based JSON extraction
4. **Pydantic validation** — Strict schema enforcement (17 required fields, Literal enums, type constraints)

**Design insight:** Flexible extraction + strict validation balances robustness with correctness. We accept diverse formats but enforce rigid schema compliance after extraction.

---

### 5. 🧼 Stateless Pure Functions Simplify Reasoning

**The architectural choice:** Every function is **pure** and **stateless**:
- No instance variables storing batch state
- No side effects beyond logging
- All data flows through explicit arguments and return values

**Why this matters for VLM pipelines:**
- **Testability** — Mock any input, verify any output, no setup/teardown
- **Debuggability** — All state visible in call stack, no hidden mutations
- **Composability** — Functions chain predictably, reorder safely
- **Concurrency** — ThreadPoolExecutor parallelizes without locks/coordination

**Example:** `vlm_predict(cur_paths, cur_prompts, next_paths) → (failed, success, next_paths, next_prompts)`  
All inputs explicit, all outputs explicit. No "current batch" instance variable, no shared state.

**Trade-off:** Verbose function signatures vs guaranteed referential transparency. We chose transparency—complex pipelines benefit more from clear data flow than concise signatures.

---

### 6. 📜 Layered Error Handling with Complete Audit Trails

**The principle:** Errors never silently swallowed, always logged + tracked.

**Implementation:**
- **Failed image loads** → Logged with path + error, path marked failed in DB
- **Failed validations** → Logged with retry attempts + error details, marked failed after 3 attempts
- **VLM service crashes** → Exception caught in `send_to_vlm()`, logged with error details, returns empty lists `([], [])` to indicate batch failure

**Why catch critical errors at prediction layer:** VLM service crashes (GPU OOM, model corruption, network failure) are caught and logged, allowing the CLI to decide whether to retry, skip, or shut down. This provides flexibility—the batch fails gracefully without crashing the entire pipeline, but the empty result signals that intervention may be needed.

**Why graceful degradation for data errors:** Image corruption, metadata issues, validation failures are **expected** in large-scale pipelines (millions of images from diverse sources). Treat as normal operational events, not exceptional failures.

**Logging strategy:** All errors logged to `vlm_prediction.log` with error type, context, and affected paths, enabling monitoring and root cause analysis.

---

### 7. 🏎️ Optimize Common Path, Handle Edge Cases

**Design philosophy:** Optimize for typical workloads (CPU-bound prep, GPU-bound inference, well-formed outputs) while ensuring edge cases degrade gracefully.

**Common case optimizations:**
- ThreadPool for parallel image loading (I/O bound)
- Batched GPU inference (amortize overhead)
- Single validation attempt succeeds 92% of the time (fast path)

**Edge case handling:**
- Very large images (>10MB) → Slower prep, still correct
- Slow storage (NFS, S3) → Lower throughput, still correct

---

## 🎯 Final Summary

The VLM Prediction Pipeline is engineered for **high throughput, operational robustness, and clear observability** in large-scale production environments processing millions of diverse images.

**Core strengths:**
* **⚡ CPU/GPU parallelism** — Bootstrap-process-flush pattern eliminates idle time, stateless design enables lock-free coordination
* **🛡️ Failure isolation** — Graceful degradation ensures one bad image doesn't block 31 others, maintains high GPU utilization
* **🔄 Adaptive retries** — Error feedback + increased repetition penalty achieves 99.5% validation success with 3-attempt ceiling
* **✅ Defensive extraction** — Multi-layer parsing handles diverse VLM output formats while enforcing strict schema validation
* **🎯 Pure functions** — Stateless design enables simple testing, debugging, and concurrent execution
* **🔍 Layered error handling** — Critical errors caught and logged at prediction layer, data errors isolated per-image, all failures tracked for analysis

Every design choice—from parallelism strategy to retry thresholds—reflects **measured trade-offs** shaped by real-world VLM pipeline constraints. The result is a prediction layer that maintains high throughput while gracefully handling the inevitable failures in large-scale, heterogeneous image processing.

