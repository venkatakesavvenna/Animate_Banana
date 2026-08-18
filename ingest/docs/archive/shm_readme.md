# Shared Memory IPC — Deep Context
 
## What This Document Is
 
A full conceptual and implementation reference for the shared memory IPC optimization
being built for the Patram-Ingest `VLLMService` pipeline. Covers OS-level mechanics,
Python API, pool design, serialization strategy, and the exact data flow for both
input prompts and model outputs.
 
---
 
## 1. Why Shared Memory — The Root Problem
 
`mp.Queue` serialises every object via `pickle` before writing it into an **OS pipe**.
The pipe buffer lives in **kernel space**. Every `put()` and `get()` is a syscall that
copies data across the user/kernel boundary:
 
```
queue.put(obj):
  1. pickle.dumps(obj)        → bytes in user-space heap RAM    (CPU time ~5 ms/image)
  2. write(pipe_fd, bytes)    → kernel copies: heap → pipe buf  (THE BOTTLENECK ~100 ms)
 
queue.get():
  1. read(pipe_fd, buf)       → kernel copies: pipe buf → heap  (~100 ms)
  2. pickle.loads(buf)        → reconstruct object              (~5 ms)
```
 
For a 1024×1024 RGB PIL image (~3 MB pickled), `queue.get()` takes ~200 ms — not
because of pickle CPU time, but because of the kernel copy. At batch sizes of 32–128
this compounds to **seconds of pure IPC overhead** before a single GPU token is generated.
 
---
 
## 2. What Shared Memory Actually Is (OS Level)
 
When the OS runs a process it gives it a **virtual address space** — a private mapping
of virtual addresses to physical RAM pages. Shared memory is an OS feature that lets
two or more processes map the **same physical RAM pages** into their own address spaces
simultaneously.
 
- No copying. No serialization on access. The same bytes in RAM appear at different
  virtual addresses in each process, but point to the same physical memory cells.
- A write in Process A is **instantly visible** in Process B. No IPC, no syscall, no copy.
### How the OS wires it up (POSIX / Linux)
 
1. `shm_open("/my_shm", O_CREAT|O_RDWR)` — creates a named object backed by RAM
   (a tmpfs entry in `/dev/shm`, not a real disk file).
2. `ftruncate(fd, N_BYTES)` — sets the size.
3. `mmap(NULL, N_BYTES, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0)` — returns a pointer;
   reads/writes go directly to those RAM pages.
4. Any other process does `shm_open("/my_shm", O_RDWR)` then `mmap` — same name → same pages.
5. `shm_unlink("/my_shm")` removes the name; pages stay alive until all mappings `munmap`.
Python's `multiprocessing.shared_memory.SharedMemory` is a thin cross-platform wrapper
around exactly this.
 
---
 
## 3. Python `SharedMemory` API
 
```python
from multiprocessing.shared_memory import SharedMemory
 
# Creator process
shm = SharedMemory(name="my_block", create=True, size=4 * 1024 * 1024)
# shm.buf  → memoryview backed directly by the mmap'd pages
# shm.name → "my_block"  (the POSIX name, used to reconnect from other processes)
# shm.size → 4194304
 
# Write raw bytes into shared RAM
shm.buf[0:4] = b'\x01\x02\x03\x04'
 
# Any other process — reconnect by name only
shm2 = SharedMemory(name="my_block", create=False)
# shm2.buf[0:4] == b'\x01\x02\x03\x04'  — no copy happened
```
 
### Lifecycle — critical distinction
 
```python
shm.close()    # unmap from THIS process's address space (like munmap)
               # other processes still see it fine
               # call this in EVERY process that opened the shm
 
shm.unlink()   # delete the name from /dev/shm
               # call this EXACTLY ONCE, from the creator/owner
               # existing mappings stay valid after unlink (OS ref-counts the pages)
```
 
Forgetting `unlink()` leaks RAM in `/dev/shm` across restarts.
 
---
 
## 4. What Goes Through Shared Memory vs the Queue
 
### Input side — what `process_one()` returns
 
```python
{
    "prompt": "<|im_start|>user\n<image>\n... ~5 KB of text ...",
    "multi_modal_data": {
        "image": [PIL.Image, PIL.Image]   # ← this is the expensive part
    }
}
```
 
The dict has two fields:
 
| Field | Pickled size | Strategy |
|---|---|---|
| `"prompt"` (str) | ~2–10 KB | Fine through queue as-is |
| `"multi_modal_data"` → `[PIL.Image]` | 3–12 MB **per image** | → shared memory |
 
PIL pickle internals: `pickle` calls `image.tobytes()` internally, producing
`width × height × channels` raw bytes, then wraps in Python object headers.
For 1024×1024 RGB: exactly 3,145,728 bytes minimum.
 
### Output side — what `model.generate()` returns
 
`List[RequestOutput]` where each `RequestOutput` contains:
- `outputs[0].text` — the generated JSON string (~1–50 KB) — **what stages actually need**
- `prompt_token_ids` — thousands of ints
- `prompt_logprobs`, `logprobs` — can be large if enabled
- Various vLLM metadata fields
Pickled size: 50 KB – 2 MB per item. The stage only needs `output.outputs[0].text`.
 
---
 
## 5. Shared Memory vs Queue — What Each Approach Escapes
 
```
                     mp.Queue (today)     shm + pickle     shm + raw bytes
pickle.dumps()          ~5 ms/img           ~5 ms/img        ~0 ms (tobytes)
kernel copy (put)       ~100 ms/img         ~0 ms            ~0 ms
kernel copy (get)       ~100 ms/img         ~0 ms            ~0 ms
user-space memcpy       ~0 ms               ~0.3 ms          ~0.3 ms
pickle.loads()          ~5 ms/img           ~5 ms/img        ~0 ms (frombytes)
─────────────────────────────────────────────────────────────────────────
total per image         ~210 ms             ~10 ms           ~0.5 ms
```
 
**The kernel boundary crossing is the bottleneck, not pickle CPU time.**
Even `pickle.dumps/loads` through shared memory gives a ~20x improvement.
Manual `tobytes/frombytes` gives ~420x, but adds code complexity.
 
---
 
## 6. Serialization Strategy — Chosen Approach
 
### Use `pickle.dumps` / `pickle.loads` for everything (pragmatic choice)
 
With this approach, `ShmPool` doesn't care what's inside the object. Coding becomes:
 
```python
# Writer (stage process)
raw = pickle.dumps(prompt_dict)        # the full {"prompt": ..., "multi_modal_data": ...}
pool.put(slot_idx, raw)                # write into shared RAM
queue.put(ShmHandle(slot_idx, len(raw), req_id, ...))   # ~100 bytes
 
# Reader (VLLMService)
handle = queue.get()                   # microseconds — tiny message
prompt_dict = pool.get(handle.slot_idx, handle.nbytes)  # pickle.loads from shm
# prompt_dict is the exact dict model.generate() already expects — zero changes
outputs = model.generate([prompt_dict], sampling_params=sparams)
```
 
**Key point**: `mp.Queue` internally does pickle too. The only thing we are escaping
is the **OS kernel pipe copy**. The pickle CPU overhead (~5 ms/image) remains, but
it happens in user space in parallel with other work — it's no longer the bottleneck.
 
### Alternative for PIL images only (maximum performance, more code)
 
```python
# Writer — skip pickle for PIL, use raw pixel bytes
raw = pil_image.tobytes()              # no pickle overhead at all
# store mode, width, height in the ShmHandle (tiny, goes in queue)
 
# Reader — reconstruct
pil_image = PIL.Image.frombytes(mode, (w, h), raw_bytes)
 
# Zero-copy variant (PIL reads directly from shm pages, no memcpy)
pil_image = PIL.Image.frombuffer(mode, (w, h),
    shm.buf[offset : offset + size], "raw", mode, 0, 1)
# shm must stay open while PIL uses this image — release() only after model.generate()
```
 
For `RequestOutput` on the response side, `pickle.dumps/loads` is always correct
— the object is too complex to manually serialize.
 
---
 
## 7. `ShmPool` Design
 
### Layout — one flat block subdivided into N fixed-size slots
 
```
┌──────────────────────────────────────────────────────────────────────┐
│              SharedMemory block  "pool_stage_a"                      │
│  slot 0       slot 1       slot 2      ...          slot N-1         │
│ [slot_size]  [slot_size]  [slot_size]  ...         [slot_size]       │
└──────────────────────────────────────────────────────────────────────┘
  offset=0    offset=S    offset=2S              offset=(N-1)*S
```
 
### Minimal `ShmPool` implementation (pickle-based)
 
```python
import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Queue as MPQueue
from typing import NamedTuple
 
class ShmHandle(NamedTuple):
    shm_name:   str
    slot_idx:   int
    nbytes:     int      # how many bytes pickle.dumps wrote — needed for get()
 
class ShmPool:
    def __init__(self, name: str, n_slots: int, slot_size: int):
        self.slot_size = slot_size
        self.n_slots   = n_slots
        self.shm = SharedMemory(name=name, create=True, size=n_slots * slot_size)
        self._free: MPQueue[int] = MPQueue()
        for i in range(n_slots):
            self._free.put(i)
 
    def acquire(self) -> int:
        """Block until a slot is free, return its index."""
        return self._free.get()
 
    def put(self, slot_idx: int, obj) -> ShmHandle:
        """Pickle obj and write into slot. Returns the handle."""
        raw = pickle.dumps(obj)
        assert len(raw) <= self.slot_size, \
            f"Object too large: {len(raw)} > {self.slot_size}"
        offset = slot_idx * self.slot_size
        self.shm.buf[offset : offset + len(raw)] = raw
        return ShmHandle(self.shm.name, slot_idx, len(raw))
 
    def get(self, handle: ShmHandle) -> object:
        """Read and unpickle from slot."""
        offset = handle.slot_idx * self.slot_size
        return pickle.loads(bytes(self.shm.buf[offset : offset + handle.nbytes]))
 
    def release(self, slot_idx: int):
        """Return slot to the free pool."""
        self._free.put(slot_idx)
 
    def cleanup(self):
        """Call once from the owner process at shutdown."""
        self.shm.close()
        self.shm.unlink()
```
 
### Free-list mechanics
 
```
Initial:    free_list = [0, 1, 2, 3, 4, 5, 6, 7]   # all 8 slots available
 
acquire():  free_list = [0, 1, 3, 4, 5, 6, 7]       # slot 2 taken
... write, queue.put(handle) ...
release(2): free_list = [0, 1, 3, 4, 5, 6, 7, 2]   # slot 2 back
```
 
Nothing is zeroed on release. Old bytes sit there but are overwritten on next `put()`.
 
---
 
## 8. Slot Sizing
 
For 1024×1024 RGB images pickled:
 
```python
slot_size = (
    1024 * 1024 * 4        # RGBA worst case raw pixels
    + 4096                 # pickle protocol overhead + PIL metadata
    + 20_000               # prompt text string (if storing full dict)
)
# Round up to next power of 2 or clean MB boundary
slot_size = 8 * 1024 * 1024   # 8 MB per slot — safe for full prompt dict
 
n_slots = batch_size * 2      # *2 so writer fills next batch while reader processes current
# e.g. batch_size=128 → 256 slots → 256 * 8 MB = 2 GB total per pool
 
# One pool per stage (so stages never conflict with each other)
```
 
If 2 GB is too much, separate the PIL images from the prompt text:
- Store only PIL pixel bytes in shm (3–4 MB each) — `slot_size = 4 MB`
- Keep the text prompt string in the queue message (it's only ~5–10 KB, pickle is fine)
---
 
## 9. Full Data Flow (Request + Response)
 
```
Stage process                    mp.Queue (tiny handles)     VLLMService
─────────────────────────────────────────────────────────────────────────
 
process_one() produces:
  {"prompt": str,
   "multi_modal_data":
     {"image": [PIL, PIL]}}
 
pool_req.acquire() → slot_idx=2
pool_req.put(2, prompt_dict)     ──── ShmHandle(name,2,nbytes) ────►
  # pickle.dumps writes          ◄─── queue.get() microseconds ────
  # ~8 MB into shm RAM
 
                                                          pool_req.get(handle)
                                                            # pickle.loads from shm
                                                            # → full prompt dict
                                                          model.generate([prompt_dict])
                                                          pool_req.release(2)
 
                                                          pool_resp.acquire() → slot_idx=7
                                                          pool_resp.put(7, outputs[0].text)
                                 ◄── ShmHandle(name,7,nbytes) ────
                                      response_queue.put(...)
 
pool_resp.get(handle)            ◄─── response_queue.get() ─────
  # → output text string
pool_resp.release(7)
```
 
**Nothing large ever crosses the kernel pipe boundary.**
 
---
 
## 10. Ownership Protocol — Critical for Correctness
 
Shared memory has no built-in access control. The discipline must be enforced in code:
 
| Step | Owner |
|---|---|
| `acquire()` | Pool → Writer |
| `put()` (write bytes) | Writer |
| `queue.put(handle)` | Writer → Reader (logical transfer) |
| `get()` (read bytes) | Reader |
| `model.generate()` consuming the data | Reader |
| `release()` | Reader → Pool |
 
**Release must happen AFTER `model.generate()` returns**, not before. `model.generate()`
reads the PIL bytes out of the shm buffer (or out of the reconstructed PIL object).
Releasing the slot while vLLM is still reading it is a race condition.
 
```python
# WRONG
pool.put(slot_idx, prompt_dict)
queue.put(handle)
pool.release(slot_idx)          # ← slot can be reused while vLLM reads it
model.generate(...)
 
# CORRECT
outputs = model.generate(prompts)   # vLLM done reading shm data
for slot_idx in consumed_slots:
    pool.release(slot_idx)          # ← safe now
```
 
---
 
## 11. How Processes Find the Shared Memory Block
 
Two patterns:
 
**Pattern 1 — create before fork** (fork start method): Parent creates the pool before
spawning children. Children inherit the file descriptor and mapping automatically.
 
**Pattern 2 — pass the name string** (spawn start method — default on macOS/Windows):
Children don't inherit memory. Pass `shm.name` (just a string like `"psm_abc123"`)
via constructor argument or a queue. Child calls:
 
```python
shm = SharedMemory(name="psm_abc123", create=False)
```
 
In your architecture (`main.py` creates pools before spawning, spawn start method),
pass the pool's `shm.name` and slot metadata to `VLLMService.__init__` so it can
reconnect inside `run()`.
 
---
 
## 12. Files to Modify (from original context doc)
 
| File | Change |
|---|---|
| `shm_pool.py` | New file — `ShmPool`, `ShmHandle` |
| `vlm_service.py` | `VLLMRequest` → `ShmVLLMRequest` (handle instead of prompt dict); `_worker()` reads from shm, releases after generate; `__init__` accepts pool ref or shm name |
| `retry_validate.py` | `submit()` acquires slot, `pool.put()`, queue.put handle; `collect()` `pool.get()`, release after reading |
| `main.py` | Create pools before spawning; pass to `VLLMService`; call `cleanup()` in `model_term()` |
 
---
 
## 13. Summary — What We Escape and What Remains
 
| Cost | mp.Queue | shm + pickle |
|---|---|---|
| Kernel pipe copy (~100 ms/image each way) | ✗ pays it | ✓ gone |
| pickle.dumps CPU (~5 ms/image) | ✗ pays it | ✗ still pays it |
| pickle.loads CPU (~5 ms/image) | ✗ pays it | ✗ still pays it |
| User-space memcpy (~0.3 ms/image) | — | negligible |
| **Total per image** | **~210 ms** | **~10 ms** |
 
The kernel copy is the bottleneck. Pickle CPU time is real but happens in user space
and can overlap with GPU inference time — it is no longer on the critical path.
 
