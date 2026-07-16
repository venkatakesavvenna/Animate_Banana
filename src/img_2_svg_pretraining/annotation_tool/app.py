"""Raster-region annotation review app (Streamlit).

One canvas, two layers (points + masks) always drawn together. One instance
at a time: fully resolve the current instance (point confirmed, mask
accepted or fixed) before moving on -- there is no place-all-points-first
mode. Any point add/move/delete recomputes that instance's mask via SAM3
automatically; there is no "run segmentation" button.

Run (inside the docker container):
    streamlit run src/img_2_svg_pretraining/annotation_tool/app.py \
        --server.port 8600 --server.address 0.0.0.0

Env config:
    IMAGES_DIR       image directory (default: data/train/original_images)
    ANNOTATIONS_DIR  annotation JSON directory (default: data/annotations)
    ANNOTATOR        actor name recorded in edit logs (default: $USER)
    SAM3_REPO        SAM3 checkpoint to load -- HF repo id or local directory
                     (default: the sam_finetuning/ fine-tuned checkpoint,
                     falling back to facebook/sam3 if that checkpoint isn't
                     present yet)

Molmo proposals come from ingest.py (offline, separate venv) -- there is
deliberately no "run Molmo" button in this app.
"""
from __future__ import annotations

import copy
import os
import time
from pathlib import Path

import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_shortcuts import shortcut_button

from img_2_svg_pretraining.annotation_tool import store
from img_2_svg_pretraining.annotation_tool.compositor import (
    compose, instance_thumbnail,
)
from img_2_svg_pretraining.annotation_tool.datamodel import (
    ImageRecord, Instance, SessionStats, has_human_point_action,
    log_event, make_empty_instance, mask_to_rle, transition_instance,
)
from img_2_svg_pretraining.annotation_tool.pointops import (
    apply_add, apply_delete, apply_label_toggle, apply_move, can_accept,
    positive_points, resolve_click,
)
from img_2_svg_pretraining.annotation_tool.sam3_backend import Sam3InteractiveBackend

_REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGES_DIR = Path(os.environ.get("IMAGES_DIR",
                                 _REPO_ROOT / "data" / "train" / "original_images"))
ACTOR = os.environ.get("ANNOTATOR", os.environ.get("USER", "human"))
UNDO_CAP = 20
UNRESOLVED = ("proposed", "needs_point_review", "needs_mask_review")

# Fine-tuned checkpoint from sam_finetuning/ (see its README) -- falls back
# to the base facebook/sam3 repo if the checkpoint hasn't been trained/copied
# into place yet, so this doesn't break environments without it.
_FINETUNED_SAM3_CHECKPOINT = (
    _REPO_ROOT / "src" / "img_2_svg_pretraining" / "data_engine" / "sam_finetuning"
    / "checkpoints" / "full_run_v1" / "best"
)
SAM3_REPO = os.environ.get(
    "SAM3_REPO",
    str(_FINETUNED_SAM3_CHECKPOINT) if _FINETUNED_SAM3_CHECKPOINT.exists() else "facebook/sam3",
)

st.set_page_config(page_title="Raster Region Annotator", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def load_sam3() -> Sam3InteractiveBackend:
    # Self-hosted transformers Sam3Tracker backend (point/box interactive
    # mode ONLY -- never the text/concept mode; see sam3_backend docstring).
    # Logs SAM3_LOAD exactly once per server process (acceptance criterion 1).
    return Sam3InteractiveBackend(hf_repo=SAM3_REPO)


@st.cache_data(show_spinner=False)
def list_image_ids() -> list[str]:
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(p.stem for p in IMAGES_DIR.iterdir() if p.suffix.lower() in exts)


@st.cache_data(show_spinner=False, max_entries=4)
def load_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def image_path(image_id: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg"):
        p = IMAGES_DIR / f"{image_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(image_id)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state():
    ss = st.session_state
    ss.setdefault("current_image_id", None)
    ss.setdefault("image_record", None)
    ss.setdefault("instances", {})
    ss.setdefault("active_instance_id", None)
    ss.setdefault("selected_point_id", None)
    ss.setdefault("negative_point_mode", False)
    ss.setdefault("undo_stack", [])
    ss.setdefault("redo_stack", [])
    ss.setdefault("sam3_embedded_image_id", None)
    ss.setdefault("last_click_coords", None)
    ss.setdefault("box_corner", None)          # first corner of two-click box
    ss.setdefault("_last_action_ts", time.time())


def open_image(image_id: str):
    """Load an image's record: resume from its JSON if present (spec section
    12 resumability), else start an empty record. Never regenerates from
    Molmo output here -- that's ingest.py's job."""
    ss = st.session_state
    loaded = store.load_image_record(image_id)
    if loaded is not None:
        record, instances = loaded
    else:
        img = load_pil(str(image_path(image_id)))
        record = ImageRecord(id=image_id, file_path=str(image_path(image_id)),
                             width=img.width, height=img.height,
                             instances=[], session_stats=SessionStats())
        instances = {}
    ss.current_image_id = image_id
    ss.image_record = record
    ss.instances = instances
    ss.active_instance_id = None
    ss.selected_point_id = None
    ss.undo_stack = []
    ss.redo_stack = []
    ss.last_click_coords = None
    ss.box_corner = None


# ---------------------------------------------------------------------------
# Undo / redo (spec section 11): per-active-instance snapshots, cap 20
# ---------------------------------------------------------------------------

def push_undo_snapshot():
    ss = st.session_state
    aid = ss.active_instance_id
    if aid is None:
        return
    ss.undo_stack.append({"instance_id": aid,
                          "snapshot": copy.deepcopy(ss.instances[aid])})
    if len(ss.undo_stack) > UNDO_CAP:
        ss.undo_stack.pop(0)
    ss.redo_stack.clear()


def undo():
    ss = st.session_state
    if not ss.undo_stack:
        return
    entry = ss.undo_stack.pop()
    aid = entry["instance_id"]
    if aid in ss.instances:
        ss.redo_stack.append({"instance_id": aid,
                              "snapshot": copy.deepcopy(ss.instances[aid])})
        ss.instances[aid] = entry["snapshot"]
        ss.selected_point_id = None


def redo():
    ss = st.session_state
    if not ss.redo_stack:
        return
    entry = ss.redo_stack.pop()
    aid = entry["instance_id"]
    if aid in ss.instances:
        ss.undo_stack.append({"instance_id": aid,
                              "snapshot": copy.deepcopy(ss.instances[aid])})
        ss.instances[aid] = entry["snapshot"]
        ss.selected_point_id = None


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _tick_time():
    ss = st.session_state
    now = time.time()
    delta = now - ss._last_action_ts
    if 0 < delta < 120:                       # idle gaps don't count
        ss.image_record.session_stats.time_seconds += delta
    ss._last_action_ts = now


def repredict_active(backend: Sam3InteractiveBackend):
    """Auto-recompute the active instance's mask from its current points.
    Clears the mask (and drops back to point review) when no positive
    points remain."""
    ss = st.session_state
    inst = ss.instances.get(ss.active_instance_id)
    if inst is None:
        return
    if not positive_points(inst):
        inst.mask_rle = None
        inst.mask_source = None
        if inst.state == "needs_mask_review":
            transition_instance(inst, "back_to_points", actor=ACTOR)
        return
    result = backend.segment(
        ss.current_image_id,
        points=[(p.x, p.y, p.label == 1) for p in inst.points],
    )
    inst.mask_rle = mask_to_rle(result.mask)
    inst.mask_source = "sam3_auto"


def activate_instance(inst: Instance):
    ss = st.session_state
    ss.active_instance_id = inst.id
    ss.selected_point_id = None
    ss.box_corner = None
    if inst.state == "proposed":  # first open (spec section 6)
        transition_instance(
            inst, "open_mask" if inst.mask_rle else "open_points", actor=ACTOR)


def new_instance():
    ss = st.session_state
    inst = make_empty_instance(ss.current_image_id)
    ss.instances[inst.id] = inst
    ss.image_record.instances.append(inst.id)
    activate_instance(inst)


def save_now():
    store.save_image_record(st.session_state.image_record,
                            st.session_state.instances)


def accept_active():
    ss = st.session_state
    inst = ss.instances.get(ss.active_instance_id)
    if inst is None or inst.state != "needs_mask_review" or not can_accept(inst):
        return
    # Criterion 6 backstop: an instance can only be accepted with at least
    # one logged human action on its points (confirm_points itself counts).
    if not has_human_point_action(inst):
        st.warning("No recorded human action on this instance's points.")
        return
    log_event(inst, ACTOR, "mask_accept", {"mask_source": inst.mask_source})
    transition_instance(inst, "accept", actor=ACTOR)
    _tick_time()
    save_now()  # write immediately -- never batch accepts
    _advance_to_next_unresolved()


def _advance_to_next_unresolved():
    ss = st.session_state
    for iid in ss.image_record.instances:
        inst = ss.instances[iid]
        if inst.state in UNRESOLVED and iid != ss.active_instance_id:
            activate_instance(inst)
            return
    ss.active_instance_id = None


def reject_active():
    ss = st.session_state
    inst = ss.instances.get(ss.active_instance_id)
    if inst is None or inst.state != "needs_point_review":
        return
    transition_instance(inst, "reject", actor=ACTOR)
    _tick_time()
    save_now()
    _advance_to_next_unresolved()


def merge_active(target_id: str):
    ss = st.session_state
    inst = ss.instances.get(ss.active_instance_id)
    if inst is None or inst.state != "needs_point_review":
        return
    inst.merged_into_id = target_id
    transition_instance(inst, "merge", actor=ACTOR)
    _tick_time()
    save_now()
    _advance_to_next_unresolved()


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

_init_state()
backend = load_sam3()
image_ids = list_image_ids()
if not image_ids:
    st.error(f"No images found in {IMAGES_DIR}")
    st.stop()

ss = st.session_state

# Deferred negative-mode flip (from the "n" shortcut): a widget-backed key
# can only be written BEFORE its widget is instantiated in a run, so the
# shortcut sets this flag and we apply it here, at the top of the script.
if ss.pop("_pending_neg_flip", False):
    ss["neg_toggle_widget"] = not ss.get("neg_toggle_widget", False)

# --- image navigation row ---------------------------------------------------
nav = st.columns([1, 1, 4, 2])
idx = image_ids.index(ss.current_image_id) if ss.current_image_id in image_ids else 0
if nav[0].button("⬅ Prev", disabled=idx == 0):
    open_image(image_ids[idx - 1])
    st.rerun()
if nav[1].button("Next ➡", disabled=idx >= len(image_ids) - 1):
    open_image(image_ids[idx + 1])
    st.rerun()
# Key the selectbox to the current image so programmatic navigation
# (Prev/Next/Next-flagged) creates a fresh widget defaulting to the new
# image -- otherwise the widget's persisted state would immediately
# navigate back to the old selection.
picked = nav[2].selectbox("Image", image_ids, index=idx,
                          key=f"imgsel_{ss.current_image_id}",
                          label_visibility="collapsed")
annotated = set(store.list_annotated_image_ids())
nav[3].caption(f"{len(annotated)} / {len(image_ids)} images have annotations")

if ss.current_image_id is None or (picked != ss.current_image_id):
    open_image(picked)

record: ImageRecord = ss.image_record
pil = load_pil(record.file_path)

# --- SAM3 embedding guard (spec section 9) -----------------------------------
# Fires the encoder once per image, NOT on point-only reruns (criterion 2);
# sam3_backend logs SAM3_EMBED whenever it actually computes.
if ss.sam3_embedded_image_id != ss.current_image_id:
    backend.ensure_embedded(ss.current_image_id, pil)
    ss.sam3_embedded_image_id = ss.current_image_id

active: Instance | None = ss.instances.get(ss.active_instance_id)

left_col, right_col = st.columns([3, 1])

# ============================================================================
# LEFT: toolbar + canvas
# ============================================================================
with left_col:
    toolbar = st.columns([1, 1, 1, 1, 2])
    add_mode = toolbar[0].toggle("Add point", value=True)
    # The widget key is the single source of truth for negative mode; the
    # "n" shortcut flips it via the _pending_neg_flip flag handled above.
    negative_mode = toolbar[1].toggle("Negative pt", key="neg_toggle_widget")
    ss.negative_point_mode = negative_mode
    box_mode = toolbar[2].toggle("Draw box", value=False)
    if toolbar[3].button("↩ Undo", disabled=not ss.undo_stack,
                         key="undo_toolbar"):
        undo()
        if ss.active_instance_id:
            repredict_active(backend)
        st.rerun()
    with toolbar[4]:
        c1, c2 = st.columns([1, 1])
        c1.markdown(f"**Image {idx + 1} / {len(image_ids)}**")
        if c2.button("Next flagged ⏭"):
            # next image (wrapping) with any unresolved instance on disk
            for step in range(1, len(image_ids) + 1):
                cand = image_ids[(idx + step) % len(image_ids)]
                loaded = store.load_image_record(cand)
                if loaded and any(i.state in UNRESOLVED
                                  for i in loaded[1].values()):
                    open_image(cand)
                    st.rerun()
            st.toast("No other image with unresolved instances found.")

    zoom_col, _ = st.columns([1, 4])
    zoom_label = zoom_col.selectbox("Zoom", ["Fit (100%)", "200%", "400%"],
                                    label_visibility="collapsed")
    zoom = {"Fit (100%)": 1.0, "200%": 2.0, "400%": 4.0}[zoom_label]

    composite, tf = compose(
        pil, ss.instances, ss.active_instance_id, ss.selected_point_id,
        zoom=zoom, crop_to_active=(zoom > 1.0),
    )
    coords = streamlit_image_coordinates(
        composite, key=f"canvas_{ss.current_image_id}")

    if box_mode:
        st.caption("**Box mode (fallback):** click two opposite corners of the "
                   "raster region; SAM3 re-predicts from the box. Use only "
                   "when a correctly-placed point still gives a bad mask.")
    else:
        st.caption(
            "Confirmed point (green) · flagged point (red, dashed) · negative "
            "(purple) — **move a point in two clicks:** click it to select "
            "(amber ring), then click its new location. Click empty canvas in "
            "Add mode to add a point.")

    # ---- click handling ----------------------------------------------------
    if coords is not None:
        click_key = (coords["x"], coords["y"])
        # streamlit-image-coordinates re-returns the same value on reruns
        # triggered by other widgets -- dedupe against the last-seen click.
        if click_key != ss.last_click_coords:
            ss.last_click_coords = click_key
            ix, iy = tf.to_image(coords["x"], coords["y"])
            ix = min(max(ix, 0), record.width - 1)
            iy = min(max(iy, 0), record.height - 1)

            if box_mode and active is not None:
                # Two-click manual box: secondary fallback path, section 10.
                # (streamlit-drawable-canvas is unmaintained against current
                # Streamlit; two corner clicks on the SAME canvas keeps the
                # fallback usable without a second drawing surface.)
                if ss.box_corner is None:
                    ss.box_corner = (ix, iy)
                    st.toast("Corner 1 set — click the opposite corner.")
                else:
                    x0, y0 = ss.box_corner
                    box = (min(x0, ix), min(y0, iy), max(x0, ix), max(y0, iy))
                    ss.box_corner = None
                    push_undo_snapshot()
                    result = backend.segment(
                        ss.current_image_id,
                        points=[(p.x, p.y, p.label == 1) for p in active.points]
                               or None,
                        box=box,
                    )
                    active.mask_rle = mask_to_rle(result.mask)
                    active.mask_source = "human_box"
                    log_event(active, ACTOR, "mask_reject_manual_box",
                              {"box": list(box)})
                    record.session_stats.masks_manually_fixed += 1
                    _tick_time()
                    st.rerun()
            elif not box_mode:
                if active is None and add_mode:
                    new_instance()          # first click starts a new instance
                    active = ss.instances[ss.active_instance_id]
                action = resolve_click(active, ss.selected_point_id, ix, iy,
                                       add_mode)
                if action.kind == "select":
                    ss.selected_point_id = action.point_id
                    st.rerun()
                elif action.kind == "deselect":
                    ss.selected_point_id = None
                    st.rerun()
                elif action.kind == "move":
                    push_undo_snapshot()
                    apply_move(active, action.point_id, ix, iy,
                               record.session_stats, actor=ACTOR)
                    ss.selected_point_id = None
                    if active.state == "needs_mask_review":
                        transition_instance(active, "back_to_points", actor=ACTOR)
                    repredict_active(backend)
                    _tick_time()
                    st.rerun()
                elif action.kind == "add":
                    push_undo_snapshot()
                    apply_add(active, ix, iy, ss.negative_point_mode,
                              record.session_stats, actor=ACTOR)
                    if active.state == "needs_mask_review":
                        transition_instance(active, "back_to_points", actor=ACTOR)
                    repredict_active(backend)
                    _tick_time()
                    st.rerun()

# ============================================================================
# RIGHT: status chips, active-instance controls, instance list, stats
# ============================================================================
with right_col:
    # --- status chips ---
    counts: dict[str, int] = {}
    for inst in ss.instances.values():
        counts[inst.state] = counts.get(inst.state, 0) + 1
    chips = " ".join(
        f"`{state}: {counts[state]}`"
        for state in ("proposed", "needs_point_review", "needs_mask_review",
                      "accepted", "rejected", "merged") if counts.get(state))
    st.markdown(chips or "`no instances`")

    # --- active instance controls ---
    if active is not None:
        st.markdown(f"**Active: `{active.id}`** — *{active.state}*")
        if active.state == "needs_point_review":
            if shortcut_button("Confirm points ✔", "c", key="confirm_btn",
                               disabled=not positive_points(active)):
                push_undo_snapshot()
                transition_instance(active, "confirm_points", actor=ACTOR)
                repredict_active(backend)
                _tick_time()
                st.rerun()
            rc1, rc2 = st.columns(2)
            if rc1.button("Reject ✖", key="reject_btn"):
                reject_active()
                st.rerun()
            others = [i for i in ss.instances
                      if i != active.id and ss.instances[i].state != "merged"]
            if others:
                tgt = rc2.selectbox("Merge into", others, key="merge_tgt",
                                    label_visibility="collapsed")
                if rc2.button("Merge ⇒", key="merge_btn"):
                    merge_active(tgt)
                    st.rerun()
        elif active.state == "needs_mask_review":
            # can_accept recomputed every rerun -- never cached (criterion 4)
            if shortcut_button("Accept ✔ (a)", "a", key="accept_btn",
                               disabled=not can_accept(active)):
                accept_active()
                st.rerun()
            if not positive_points(active):
                st.caption("⚠ needs ≥1 positive point to accept")
            if st.button("Back to points ↩", key="back_btn"):
                transition_instance(active, "back_to_points", actor=ACTOR)
                st.rerun()
        else:
            if st.button("Reopen ♻", key="reopen_btn"):
                transition_instance(active, "reopen", actor=ACTOR)
                save_now()
                st.rerun()

        if ss.selected_point_id and any(p.id == ss.selected_point_id
                                        for p in active.points):
            dc1, dc2 = st.columns(2)
            if dc1.button("Delete point 🗑", key="delpt_btn"):
                push_undo_snapshot()
                apply_delete(active, ss.selected_point_id,
                             record.session_stats, actor=ACTOR)
                ss.selected_point_id = None
                if active.state == "needs_mask_review":
                    transition_instance(active, "back_to_points", actor=ACTOR)
                repredict_active(backend)
                _tick_time()
                st.rerun()
            if dc2.button("Toggle ± label", key="togglept_btn"):
                push_undo_snapshot()
                apply_label_toggle(active, ss.selected_point_id, actor=ACTOR)
                if active.state == "needs_mask_review":
                    transition_instance(active, "back_to_points", actor=ACTOR)
                repredict_active(backend)
                _tick_time()
                st.rerun()

    # --- global shortcut buttons (work without widget focus, criterion 7) ---
    gc1, gc2 = st.columns(2)
    with gc1:
        if shortcut_button("Undo ↩", "ctrl+z", key="undo_sc",
                           disabled=not ss.undo_stack):
            undo()
            if ss.active_instance_id:
                repredict_active(backend)
            st.rerun()
    with gc2:
        if shortcut_button("Neg mode (n)", "n", key="neg_sc"):
            ss._pending_neg_flip = True
            st.rerun()
    if ss.redo_stack and st.button("Redo ↪", key="redo_btn"):
        redo()
        if ss.active_instance_id:
            repredict_active(backend)
        st.rerun()

    st.divider()
    if st.button("➕ New instance", key="newinst_btn"):
        new_instance()
        st.rerun()

    # --- instance list: thumbnail + state badge, no text field --------------
    _BADGE = {"proposed": "🔴", "needs_point_review": "🟠",
              "needs_mask_review": "🟡", "accepted": "🟢",
              "rejected": "⚫", "merged": "🔗"}

    def _sort_key(iid: str):
        order = {"needs_point_review": 0, "needs_mask_review": 1,
                 "proposed": 2, "accepted": 3, "rejected": 4, "merged": 5}
        return order.get(ss.instances[iid].state, 9)

    for n, iid in enumerate(sorted(record.instances, key=_sort_key)):
        inst = ss.instances[iid]
        row = st.columns([1, 2, 1])
        if inst.points:
            row[0].image(instance_thumbnail(pil, inst))
        row[1].markdown(
            f"{_BADGE.get(inst.state, '·')} `#{n} {inst.id}`<br/>"
            f"<small>{inst.state} · {len(inst.points)} pt</small>",
            unsafe_allow_html=True)
        if iid != ss.active_instance_id:
            if row[2].button("open", key=f"open_{iid}"):
                activate_instance(inst)
                st.rerun()
        else:
            row[2].markdown("**◀ active**")

    # --- session stats footer ---
    s = record.session_stats
    st.divider()
    st.caption(
        f"session: +{s.points_added} pts · {s.points_moved} moved · "
        f"{s.points_deleted} deleted · {s.masks_manually_fixed} box-fixed · "
        f"{s.time_seconds / 60:.1f} min")
