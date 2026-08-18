"""
`ShardAllocator` — folders, then tars, with one authority (v1.9 §5).

The output rule, stated once: **the JSONL is always the output, for every
backend and every job.** Images are ADDITIONAL. Only images go into
folders-then-tars. A job may produce both: the image bytes land in a shard, and
the JSONL record says which shard and which member. Nothing about the JSONL
mechanism, its shard rotation, its fsync boundary, or its recovery changes.

On-disk layout — two directories, not one
-----------------------------------------
The loose per-shard folders and the packed shards live in SEPARATE trees:

    {root}/{host}/image_folders/shard_00007/a.png     <- what the worker writes
    {root}/{host}/shards/shard_00007.tar              <- the packed shard
    {root}/{host}/shards/shard_00007.csv              <- its byte-offset index

Keeping them apart means `shards/` is exactly the set of shippable artifacts —
`ls`, `rsync`, `aws s3 cp --recursive` and a glob all do the obvious thing with
no filtering — while `image_folders/` stays the working tree. It also removes an
ambiguity the flat layout had: with folder retention on (see below), a folder and
its tar coexist permanently, so "is there a folder next to this tar?" stopped
being a usable crash signal. Separate trees make the question moot.

**Folder retention (`delete_folders_after_seal`, default `False`).** By default a
sealed folder is KEPT. Every image therefore exists twice — loose and inside the
tar — which is a deliberate ~2x disk cost in exchange for: the loose files stay
directly readable with no untarring, a bad tar can be rebuilt from the folder,
and a downstream stage that resolves the folder path keeps working forever rather
than only until the seal. Set it `True` to reclaim the space once you trust the
tars.

The main process owns folder assignment, not the workers
--------------------------------------------------------
A per-worker directory scheme would mean N independent shard sequences and no
single answer to "how full is the current folder". Instead there is one
allocator in the cli process; workers only perform I/O on paths handed to them
and report back how many bytes they wrote.

Four operations, all in the cli process:

    reserve(input_path, name)  during request build, on the prep thread. Hands
                               out `{root}/{host}/image_folders/shard_00007/{name}`
                               plus the record
                               `{"shard": "shard_00007.tar", "member": name}`,
                               and increments outstanding[7].

    commit(input_path, output) the worker returned `bytes_written`; add it to
                               bytes[idx] and decrement outstanding[idx]. Rolls
                               by byte size — same trigger as JSONL rotation.

    release_failed(input_path) any item that reserved a slot but ended terminally
                               failed (reject / attempts-exhausted / prep-failed).
                               Deletes the file if it exists, then decrements
                               outstanding — the SAME accounting as a success.
                               Miss the decrement and a folder with one failed
                               image never seals; miss the delete and a failed
                               item's partial output ships inside the tar anyway.

    seal(idx)                  once the folder is closed for allocation AND
                               outstanding[idx] == 0.

Two names for the same artifact
-------------------------------
A shard is much bigger than a JSONL flush group (10 GB vs 250 lines), so when a
stage commits a row — and hands it downstream via `mark_done()` — that row's
shard is almost always still OPEN. Rather than making downstream wait for a seal
(which turns StageWeaver's streaming handoff into one-shard-at-a-time batching)
or letting it resolve a tar that does not exist yet (a real correctness bug),
both states are made resolvable:

  - The JSONL record always names the FINAL TAR, from the moment of reservation.
    It never changes, which is what preserves shard immutability.
  - The DB column downstream stages read starts out holding the FOLDER path,
    which is guaranteed to exist the instant the JSONL record is durable (the
    worker fsynced the file before responding). Once the shard seals,
    `on_shard_sealed` rewrites those rows to the tar path.
  - A downstream stage tells them apart by what it resolves to: a directory-shaped
    path means "read the loose file", a `.tar` path means "read this member out
    of the tar". Both are always valid at read time.

The seal -> update -> delete ordering makes the pointer swap crash-safe for
free: die between `os.rename` and the DB update and the tar exists, the folder
still exists untouched, and the DB still says "folder" — downstream keeps
reading correctly until the next startup retries the swap. `rmtree` (when
enabled at all) only ever runs after the DB update returns.

Seal order, and the completion marker
-------------------------------------
    write .tar.tmp -> fsync -> rename to .tar -> fsync(shards dir)
      -> on_shard_sealed()  (the DB pointer swap)
      -> write .csv         (THE COMPLETION MARKER)
      -> rmtree folder      (only if delete_folders_after_seal)

The CSV is written LAST of the durable steps precisely so its presence means
"this shard is completely finished". Startup recovery reads it that way, which is
what keeps restarts cheap once folders are retained: without a marker, a run with
10,000 retained folders would re-call the pointer-swap hook for all 10,000 of them
on every single startup.

Durability ordering
-------------------
The worker fsyncs the file it wrote BEFORE putting its response on the queue.
That single rule keeps the existing commit protocol intact with zero new
recovery code: by the time `JSONLWriter._flush_group` commits a path, its image
bytes are already durable — whether or not its shard has sealed yet.

Byte-offset index (`shard_00007.csv`, next to `shard_00007.tar`)
------------------------------------------------------------------
Written once, right after the tar itself becomes durable — see
`_write_shard_csv`. One row per file: `image_name_in_shard`, `byte_offset`,
`byte_length` (the file's CONTENT span, header excluded), `img_tar_shard_path`,
and `original_image_path` when that provenance is still in memory. That is
enough for a consumer to seek/Range-read one image straight out of the tar —
local or on S3 — with no untarring. Column names follow
Unified-Vision-Dataset-Repo's `images.csv` convention (standardisation_spec.md
§7.3) for the subset of columns that make sense here.
"""

from __future__ import annotations

import csv
import os
import shutil
import socket
import tarfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, NamedTuple, Optional, Tuple

DEFAULT_MAX_SHARD_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB


class ShardReservation(NamedTuple):
    """One outstanding reservation, keyed on the INPUT path.

    Keyed on the input path rather than a handle because that is the identity
    the pipeline carries end to end: cli_utils sees terminal results as input
    paths, so both `commit` and `release_failed` can be driven from them with no
    extra plumbing and nothing for a task to remember.
    """
    shard_idx: int
    output_path: str
    member: str


def default_on_shard_sealed(shard_id: str, folder_path: str, tar_path: str) -> None:
    """
    Hook: bulk-update every DB row referencing this shard from `folder_path` to
    `tar_path`. Called once per shard, after the tar is durable on disk and
    BEFORE its folder is removed.

    STUB — the real DB schema for the downstream-readable image-path column is
    TBD, so the shipped default only records that the swap point was reached.
    Install a real one with `ShardAllocator(on_shard_sealed=...)`.

    Contract for whoever implements it:
      - Must be safe to call TWICE. A crash between this call and the following
        `rmtree` makes startup recovery call it again, and the second call must
        be a no-op.
      - Must be synchronous and durable before it returns. `rmtree` runs
        immediately after, so the folder must not disappear while a DB row still
        points at it.

    (This ships as a no-op rather than the `raise NotImplementedError` the v1.9
    proposal sketched: raising would make every seal fail and take the tar
    creation down with it, which is a strictly worse default than sealing
    correctly and leaving the pointer swap to be wired up.)
    """
    return None


class ShardAllocator:
    """
    The single authority on which folder an output file goes into.

    Thread-safe: `reserve()` is called from the predictor's prep thread while
    `commit()` / `release_failed()` are called from the cli main loop, and
    sealing runs on its own background thread so it never stalls the pipeline.
    """

    def __init__(
        self,
        root: str,
        host: Optional[str] = None,
        max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
        digits: int = 5,
        logger=None,
        on_shard_sealed: Optional[Callable[[str, str, str], None]] = None,
        compress: bool = False,
        delete_folders_after_seal: bool = False,
        folders_dirname: str = "image_folders",
        shards_dirname: str = "shards",
    ):
        self.root = root
        self.host = host or socket.gethostname()
        self.base = os.path.join(root, self.host)
        # Two trees: the working one and the shippable one. See the module
        # docstring for why they are kept apart.
        self.folders_dir = os.path.join(self.base, folders_dirname)
        self.shards_dir = os.path.join(self.base, shards_dirname)
        self.max_shard_bytes = max_shard_bytes
        self.digits = digits
        self.logger = logger
        self.on_shard_sealed = on_shard_sealed or default_on_shard_sealed
        # Uncompressed on purpose. These are PNG/JPEG bytes — already compressed —
        # so gzip buys ~nothing and costs a full re-read of 10 GB of CPU time per
        # shard, on the same box that is trying to keep 32 renderers fed.
        self.compress = compress
        # OFF by default: a sealed folder is kept, so every image exists both
        # loose and inside its tar. That is a deliberate ~2x disk cost — it buys
        # directly-readable files, the ability to rebuild a bad tar from source,
        # and a folder path that stays resolvable forever rather than only until
        # the seal. Turn it on to reclaim the space once the tars are trusted.
        self.delete_folders_after_seal = delete_folders_after_seal

        self._lock = threading.Lock()
        self._bytes: Dict[int, int] = {}
        self._outstanding: Dict[int, int] = {}
        self._reservations: Dict[str, ShardReservation] = {}
        # shard_idx -> {member_name: input_path}, drained into the shard's CSV
        # at seal time — see commit() and _write_shard_csv().
        self._csv_rows: Dict[int, Dict[str, str]] = {}
        self._closed_for_alloc: set = set()
        self._sealed: set = set()
        self._sealing: set = set()
        self._current = 0
        self._stopped = False

        # One thread: sealing is I/O bound and serialising it keeps the disk
        # doing one large sequential write at a time rather than N interleaved.
        self._seal_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shard-seal")

        os.makedirs(self.folders_dir, exist_ok=True)
        os.makedirs(self.shards_dir, exist_ok=True)
        self._recover_on_startup()

    # ------------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------------

    def _stem(self, idx: int) -> str:
        return f"shard_{idx:0{self.digits}d}"

    def _folder(self, idx: int) -> str:
        return os.path.join(self.folders_dir, self._stem(idx))

    def _tar(self, idx: int) -> str:
        return os.path.join(self.shards_dir, self._stem(idx) + ".tar")

    def _csv(self, idx: int) -> str:
        """The shard's byte-offset index — and its completion marker. Present
        iff the seal ran all the way through; see _recover_on_startup()."""
        return os.path.join(self.shards_dir, self._stem(idx) + ".csv")

    def _shard_id(self, idx: int) -> str:
        return f"shard_{idx:0{self.digits}d}.tar"

    def _log(self, level: str, msg: str) -> None:
        if self.logger is None:
            return
        try:
            getattr(self.logger, level)(f"[shards] {msg}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Startup recovery
    # ------------------------------------------------------------------

    def _recover_on_startup(self) -> None:
        """
        Resolve whatever the previous process left behind.

        The CSV is the completion marker (see the module docstring), which is
        what makes this cheap now that folders are retained by default — a shard
        that finished is identified by one `os.path.exists` and then skipped
        entirely, rather than having its pointer-swap hook re-called on every
        startup for the rest of the dataset's life.

          - `.tar` + `.csv`          — fully sealed. Nothing to do, except remove
                                       a folder that retention says should be
                                       gone (i.e. we died between the CSV write
                                       and the rmtree).
          - `.tar`, no `.csv`        — died between `os.rename` and the CSV write.
                                       Re-run the tail: pointer swap (idempotent
                                       by contract), CSV, then retention.
          - folder, no `.tar`        — died with the folder still open. Seal it
                                       now, and resume allocation at NN+1.
          - a stray `.tar.tmp`       — an interrupted seal that never got renamed,
                                       so nothing can reference it. Delete it; the
                                       folder, if present, is re-sealed above.
        """
        max_idx = -1
        folders: Dict[int, str] = {}
        tars: set = set()

        for name in self._safe_listdir(self.shards_dir):
            full = os.path.join(self.shards_dir, name)
            if name.endswith(".tar.tmp"):
                try:
                    os.remove(full)
                    self._log("warning", f"removed stray partial tar {name}")
                except OSError as e:
                    self._log("error", f"could not remove stray {name}: {e}")
                continue
            if name.endswith(".tar") and name.startswith("shard_"):
                idx = self._parse_idx(name[: -len(".tar")])
                if idx is not None:
                    tars.add(idx)
                    max_idx = max(max_idx, idx)

        for name in self._safe_listdir(self.folders_dir):
            full = os.path.join(self.folders_dir, name)
            if os.path.isdir(full) and name.startswith("shard_"):
                idx = self._parse_idx(name)
                if idx is not None:
                    folders[idx] = full
                    max_idx = max(max_idx, idx)

        for idx in sorted(tars):
            folder = folders.get(idx)
            if os.path.exists(self._csv(idx)):
                # Fully sealed on a previous run. The common case once folders
                # are retained — say nothing and do nothing.
                self._sealed.add(idx)
                if folder and self.delete_folders_after_seal:
                    self._log("warning",
                              f"{self._stem(idx)} was fully sealed but its folder "
                              "survived — removing it now (retention is off)")
                    shutil.rmtree(folder, ignore_errors=True)
                continue

            # Tar exists but the marker does not: the seal was interrupted after
            # the rename. Finish it.
            tar_path = self._tar(idx)
            self._log("warning",
                      f"{self._stem(idx)} has a tar but no CSV — finishing the "
                      "interrupted seal")
            try:
                self.on_shard_sealed(self._shard_id(idx), folder or self._folder(idx), tar_path)
                self._write_shard_csv(idx, tar_path)
                if folder and self.delete_folders_after_seal:
                    shutil.rmtree(folder, ignore_errors=True)
                self._sealed.add(idx)
            except Exception as e:
                self._log("error", f"failed to finish seal for shard {idx}: {e}")

        for idx, folder in sorted(folders.items()):
            if idx in tars:
                continue
            self._log("warning",
                      f"sealing leftover folder {self._stem(idx)} from a previous run")
            self._closed_for_alloc.add(idx)
            self._outstanding[idx] = 0
            self._seal_now(idx)

        self._current = max_idx + 1 if max_idx >= 0 else 0
        if max_idx >= 0:
            self._log("info", f"resuming shard allocation at index {self._current}")

    @staticmethod
    def _safe_listdir(path: str):
        try:
            return os.listdir(path)
        except OSError:
            return []

    def _parse_idx(self, stem: str) -> Optional[int]:
        try:
            return int(stem.split("_")[-1])
        except (ValueError, IndexError):
            return None

    # ------------------------------------------------------------------
    # The four operations
    # ------------------------------------------------------------------

    def reserve(self, input_path: str, name: str) -> Tuple[str, dict]:
        """
        Reserve one output slot. Returns `(abs_output_path, record)`.

        The record names the FINAL TAR from the moment of reservation, not after
        sealing — so a JSONL line is correct both before and after the seal, and
        sealing never has to touch an immutable shard. This is the detail that
        keeps "JSONL shards are immutable after fsync" true.
        """
        with self._lock:
            if self._stopped:
                raise RuntimeError("ShardAllocator is closed; cannot reserve")
            idx = self._current
            folder = self._folder(idx)
            os.makedirs(folder, exist_ok=True)
            output_path = os.path.join(folder, name)
            self._outstanding[idx] = self._outstanding.get(idx, 0) + 1
            prev = self._reservations.get(input_path)
            self._reservations[input_path] = ShardReservation(idx, output_path, name)
        if prev is not None:
            # A retry re-runs build_request, so the same input path can reserve
            # more than once. Release the previous one or its folder never seals.
            self._log("info",
                      f"{input_path} re-reserved; releasing its previous slot in "
                      f"shard_{prev.shard_idx:0{self.digits}d}")
            self._release(prev, delete_file=True)
        return output_path, {"shard": self._shard_id(idx), "member": name}

    def commit(self, input_path: str, output=None) -> None:
        """
        Resolve a reservation as a success. `output` is whatever `call_fn`
        returned; `bytes_written` is read off it when present, else the file is
        stat'ed — a task that forgets to report its size still rolls correctly.
        """
        with self._lock:
            res = self._reservations.pop(input_path, None)
        if res is None:
            return  # this item produced no image output — nothing to account for

        nbytes = None
        if isinstance(output, dict):
            nbytes = output.get("bytes_written")
        if nbytes is None:
            try:
                nbytes = os.path.getsize(res.output_path)
            except OSError:
                nbytes = 0

        seal_candidates = []
        with self._lock:
            self._bytes[res.shard_idx] = self._bytes.get(res.shard_idx, 0) + int(nbytes)
            self._outstanding[res.shard_idx] = max(
                0, self._outstanding.get(res.shard_idx, 0) - 1
            )
            # Remembered only so the CSV this shard seals into (see
            # _write_shard_csv) can carry provenance. Best-effort: lost across a
            # crash/restart like the rest of this in-memory state, in which case
            # the CSV still gets correct byte_offset/byte_length (read back from
            # the tar itself) with an empty original_image_path.
            self._csv_rows.setdefault(res.shard_idx, {})[res.member] = input_path
            # Roll by byte size — the same trigger JSONL rotation uses. Once the
            # current folder is over budget it stops accepting new reservations;
            # the ones already outstanding still land in it and must all return
            # before it can seal.
            if (res.shard_idx == self._current
                    and self._bytes[res.shard_idx] >= self.max_shard_bytes):
                self._closed_for_alloc.add(self._current)
                self._log("info",
                          f"shard_{self._current:0{self.digits}d} reached "
                          f"{self._bytes[res.shard_idx]} bytes — closing for allocation")
                self._current += 1
            seal_candidates = self._sealable_locked()
        for idx in seal_candidates:
            self._seal_async(idx)

    def release_failed(self, input_path: str) -> None:
        """
        Resolve a reservation as a terminal failure (reject / attempts exhausted
        / prep failed). Deletes the output file if it exists — a prep-level
        failure never got far enough to write anything, so this is a GUARDED
        remove, not an assumed one — then decrements outstanding, same
        accounting as a success.

        This is what keeps a sealed tar free of orphans: every reservation
        resolves to either a JSONL success record backed by a real file, or
        nothing at all.
        """
        with self._lock:
            res = self._reservations.pop(input_path, None)
        if res is None:
            return
        self._release(res, delete_file=True)

    def _release(self, res: ShardReservation, delete_file: bool) -> None:
        if delete_file:
            try:
                os.remove(res.output_path)
                self._log("info", f"deleted failed item's output {res.output_path}")
            except FileNotFoundError:
                pass  # never got far enough to write — the common case
            except OSError as e:
                self._log("error", f"could not delete {res.output_path}: {e}")
        with self._lock:
            self._outstanding[res.shard_idx] = max(
                0, self._outstanding.get(res.shard_idx, 0) - 1
            )
            seal_candidates = self._sealable_locked()
        for idx in seal_candidates:
            self._seal_async(idx)

    # ------------------------------------------------------------------
    # Sealing
    # ------------------------------------------------------------------

    def _sealable_locked(self):
        """Caller must hold the lock. A folder seals once it is closed for
        allocation AND every request that reserved into it has reached a
        terminal state, success or failure."""
        ready = []
        for idx in sorted(self._closed_for_alloc):
            if idx in self._sealed or idx in self._sealing:
                continue
            if self._outstanding.get(idx, 0) == 0:
                self._sealing.add(idx)
                ready.append(idx)
        return ready

    def _seal_async(self, idx: int) -> None:
        try:
            self._seal_pool.submit(self._seal_now, idx)
        except RuntimeError:
            # Pool already shut down (we are in close()); seal inline instead so
            # the folder is not left behind.
            self._seal_now(idx)

    def _seal_now(self, idx: int) -> None:
        """
        `tarfile`, uncompressed, into `.tar.tmp` -> fsync -> `os.rename` to
        `.tar` -> fsync(shards dir) -> the DB pointer swap -> the CSV (the
        completion marker) -> `rmtree` the folder, and that last step only when
        `delete_folders_after_seal` is on.

        That ordering is the whole crash-safety argument; see the module
        docstring. Runs on a background thread so it never stalls the pipeline,
        and happens continuously through the run as folders fill — not batched
        to end-of-run, the same way JSONL shards rotate on the go.
        """
        folder = self._folder(idx)
        tar_path = self._tar(idx)
        tmp_path = tar_path + ".tmp"

        if not os.path.isdir(folder):
            with self._lock:
                self._sealing.discard(idx)
                self._sealed.add(idx)
            return

        t0 = time.time()
        try:
            mode = "w:gz" if self.compress else "w"
            with open(tmp_path, "wb") as raw:
                with tarfile.open(fileobj=raw, mode=mode) as tar:
                    for name in sorted(os.listdir(folder)):
                        full = os.path.join(folder, name)
                        if os.path.isfile(full):
                            tar.add(full, arcname=name)
                raw.flush()
                os.fsync(raw.fileno())

            os.rename(tmp_path, tar_path)
            # fsync the DIRECTORY too: a rename is only durable once the parent
            # directory entry is. Without this a crash right here can leave
            # neither the tmp nor the final name on disk.
            self._fsync_dir(self.shards_dir)

            # Pointer swap first, so a downstream row is never left pointing at a
            # folder that the next step might remove.
            self.on_shard_sealed(self._shard_id(idx), folder, tar_path)

            # The CSV is written LAST of the durable steps: its presence is what
            # tells the next startup this shard is completely finished.
            self._write_shard_csv(idx, tar_path)

            if self.delete_folders_after_seal:
                shutil.rmtree(folder, ignore_errors=True)

            size = os.path.getsize(tar_path)
            self._log("info",
                      f"sealed shard_{idx:0{self.digits}d} -> {os.path.basename(tar_path)} "
                      f"({size} bytes) in {time.time() - t0:.1f}s"
                      f"{'' if self.delete_folders_after_seal else ' (folder kept)'}")
            with self._lock:
                self._sealed.add(idx)
        except Exception as e:
            self._log("error",
                      f"failed to seal shard_{idx:0{self.digits}d}: "
                      f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            # Leave the folder alone. Startup recovery re-seals it next run;
            # losing a tar is recoverable, losing the folder is not.
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
        finally:
            with self._lock:
                self._sealing.discard(idx)

    def _write_shard_csv(self, idx: int, tar_path: str) -> None:
        """
        One row per file in the just-sealed tar, naming exactly where its bytes
        sit: `byte_offset` is where its CONTENT starts (header excluded) and
        `byte_length` is the content length — a consumer seeks there and reads
        `byte_length` bytes, no untarring, and the same numbers work as an S3
        Range request once the tar is uploaded. Column names follow
        Unified-Vision-Dataset-Repo's `images.csv` convention (see
        standardisation_spec.md §7.3) for whatever subset applies here — no
        `parent_doc_path`/`page_number`/S3 columns, since this allocator has no
        document-linkage concept and never pushes to S3 itself.

        Deliberately reads the offsets back from the tar we just wrote, rather
        than tracking them while writing it (`TarFile.addfile()` copies the
        `TarInfo` it's given and never reports position back to the caller —
        verified empirically, not from memory). That also makes this method
        reusable, unchanged, from the crash-recovery path in
        _recover_on_startup(), where the tar exists but this object's own
        commit()-time bookkeeping does not (a fresh process has no memory of
        the one that died) — the tar on disk is always the source of truth.

        original_image_path is filled from _csv_rows when known (the normal
        case) and left empty on a recovery-sealed shard, where that in-memory
        mapping did not survive the crash — everything else in the row is
        still exact, since it comes from the tar itself.
        """
        csv_path = self._csv(idx)
        tmp_path = csv_path + ".tmp"
        with self._lock:
            member_to_input = self._csv_rows.pop(idx, {})
        try:
            with tarfile.open(tar_path, "r") as tar:
                members = [m for m in tar.getmembers() if m.isfile()]
        except Exception as e:
            self._log("error", f"could not reopen {tar_path} to build its CSV: {e}")
            return
        try:
            with open(tmp_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "image_name_in_shard", "original_image_path",
                    "img_tar_shard_path", "byte_offset", "byte_length",
                ])
                for m in members:
                    w.writerow([
                        m.name, member_to_input.get(m.name, ""),
                        tar_path, m.offset_data, m.size,
                    ])
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp_path, csv_path)
            # This file is the seal's completion marker, so its directory entry
            # has to be durable — otherwise a crash here leaves a finished shard
            # looking unfinished, and the next startup redoes the pointer swap.
            # (Harmless, since the hook is idempotent by contract, but pointless.)
            self._fsync_dir(self.shards_dir)
        except OSError as e:
            self._log("error", f"could not write CSV for {tar_path}: {e}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _fsync_dir(path: str) -> None:
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def outstanding(self) -> Dict[int, int]:
        with self._lock:
            return {k: v for k, v in self._outstanding.items() if v}

    def close(self, seal_current: bool = True, timeout: float = 600.0) -> None:
        """
        Stop allocating and finish the sealing that is already queued.

        `seal_current` also seals the still-open folder, which is what makes a
        clean run end with every image inside a tar instead of one loose folder
        left over for the next startup to notice. Unresolved reservations are
        released first — on a clean shutdown those are items cli() is about to
        reset from state=1 back to fetch_state, so their half-written outputs
        must not ship.
        """
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            leftovers = list(self._reservations.items())
            self._reservations = {}

        if leftovers:
            self._log("warning",
                      f"releasing {len(leftovers)} unresolved reservations at shutdown "
                      "(their paths are reset to fetch_state and will be replayed)")
            for _, res in leftovers:
                self._release(res, delete_file=True)

        if seal_current:
            with self._lock:
                self._closed_for_alloc.add(self._current)
                ready = self._sealable_locked()
            for idx in ready:
                self._seal_async(idx)

        self._seal_pool.shutdown(wait=True)
        self._log("info", "ShardAllocator closed")
