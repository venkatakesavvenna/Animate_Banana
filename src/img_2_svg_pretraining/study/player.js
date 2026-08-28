/* The study player.
 *
 * One component, used identically by the absolute and pairwise screens.
 *
 * It never plays the exported mp4. At fps:2 those run 7.5s against ~40s of
 * authored narration -- unreadable, and "narration alignment" would be
 * unratable. Instead it drives the frame deck from the timeline computed at
 * bundle-build time, so pacing follows the narration and a caption can never
 * drift from the frame it describes.
 *
 * Two tabs over the same data, as the team asked:
 *   Video  -- timed playback, 0.5x default, speed slider
 *   Slider -- manual scrub, one frame at a time
 * Both read the same holds[] and cues[], so they cannot disagree.
 *
 * First pass runs WITHOUT seeking. "Submit disabled until it has been watched"
 * is theatre against a native scrub bar -- you drag to the end in 200ms. Only
 * once a slot has genuinely played through does its scrub unlock.
 *
 * Contract:
 *   const p = mountPlayer(el, spec, {onEvent, onWatched});
 *   p.watched()  -> bool   p.pause()   p.destroy()
 * spec = {slot, frames[], holds[], cues[], duration, frame_w, frame_h}
 */

function mountPlayer(el, spec, hooks) {
  hooks = hooks || {};
  const slot = spec.slot || "SINGLE";
  const frames = spec.frames || [];
  const holds = spec.holds || [];
  const cues = spec.cues || [];
  const duration = spec.duration || 0;

  // Cumulative frame end times, so a clock position maps to a frame without
  // re-summing the holds on every animation tick.
  const ends = [];
  let acc = 0;
  for (let i = 0; i < holds.length; i++) { acc += holds[i]; ends.push(acc); }

  // Slides only. Timed playback is gone: participants set their own pace, and
  // reaching the last frame is a definite act rather than a timer expiring
  // while they looked away.
  let t = 0, seq = 0;
  let everComplete = false, maxFrameSeen = 0;

  el.classList.add("player");
  el.innerHTML = `
    <div class="pl-tabs">
      <span class="pl-label" data-role="label"></span>
      <span class="pl-badge" data-role="badge">not viewed yet</span>
    </div>
    <div class="pl-stage" data-role="stage" title="Click to view full screen">
      <img class="pl-frame" alt="" draggable="false">
      <div class="pl-hint">Click to enlarge</div>
      <div class="pl-step" data-role="step"></div>
    </div>
    <div class="pl-cue" data-role="cue"></div>
    <div class="pl-controls">
      <input class="pl-seek" data-role="seek" type="range" min="0"
             max="${Math.max(frames.length - 1, 0)}" value="0" step="1">
    </div>
    <div class="pl-keys">
      <span class="pl-key" data-role="kleft">&larr;</span>
      <span class="pl-key" data-role="kright">&rarr;</span>
      <span class="pl-keyhint">step through</span>
      <span class="pl-progress" data-role="progress"></span>
    </div>`;

  const $ = (role) => el.querySelector(`[data-role="${role}"]`);
  const img = el.querySelector(".pl-frame");
  const stage = el.querySelector(".pl-stage");
  if (spec.frame_w && spec.frame_h) {
    stage.style.aspectRatio = `${spec.frame_w} / ${spec.frame_h}`;
  }
  if (spec.label) $("label").textContent = spec.label;

  // Preload every frame. A stall mid-playback would desync the caption from
  // the picture, which is precisely the artefact this study is measuring.
  const images = frames.map((id) => { const i = new Image(); i.src = `/media/${id}`; return i; });

  function emit(type, payload) {
    if (hooks.onEvent) {
      hooks.onEvent({ type, slot, client_seq: seq++, t_video: round2(t),
                      t_wall_client: new Date().toISOString(), payload: payload || {} });
    }
  }
  const round2 = (x) => Math.round(x * 100) / 100;

  function frameAt(time) {
    for (let i = 0; i < ends.length; i++) if (time < ends[i]) return i;
    return Math.max(frames.length - 1, 0);
  }
  function cueAt(time) {
    for (let i = 0; i < cues.length; i++) {
      if (time >= cues[i].start && time < cues[i].end) return cues[i];
    }
    return cues.length && time >= duration ? cues[cues.length - 1] : null;
  }
  function fmt(s) {
    const m = Math.floor(s / 60), r = Math.floor(s % 60);
    return `${m}:${String(r).padStart(2, "0")}`;
  }

  function render() {
    const i = frameAt(t);
    if (i > maxFrameSeen) { maxFrameSeen = i; }
    if (images[i] && img.src !== images[i].src) img.src = images[i].src;
    $("seek").value = i;
    $("step").textContent = `${i + 1} / ${frames.length}`;
    const cue = cueAt(t);
    $("cue").textContent = cue ? cue.text : "";
    $("cue").style.display = cues.length ? "block" : "none";
    if (fullscreenOf === slot) paintFullscreen(i, cue);
  }

  function setWatched() {
    const done = everComplete || maxFrameSeen >= frames.length - 1;
    if (done) everComplete = true;
    $("badge").textContent = done ? "viewed" : "not viewed yet";
    $("badge").classList.toggle("pl-ok", done);
    const pct = frames.length > 1
      ? Math.round(100 * maxFrameSeen / (frames.length - 1)) : 100;
    $("progress").textContent = done ? "" : `${pct}% of the way through`;
    if (done && hooks.onWatched) hooks.onWatched(slot);
  }

  function noteFrame(i) {
    if (i > maxFrameSeen) maxFrameSeen = i;
    setWatched();
  }

  function flashKey(role) {
    const k = $(role);
    if (!k) return;
    k.classList.add("pl-key-on");
    setTimeout(() => k.classList.remove("pl-key-on"), 180);
  }

  function step(delta) {
    const i = Math.min(Math.max(Number($("seek").value) + delta, 0), frames.length - 1);
    $("seek").value = i;
    t = i === 0 ? 0 : ends[i - 1];
    el.classList.add("pl-started");
    render();
    noteFrame(i);
    emit(delta > 0 ? "step_forward" : "step_back", { frame: i });
  }

  $("seek").oninput = (e) => {
    el.classList.add("pl-started");
    const i = Number(e.target.value);
    t = i === 0 ? 0 : ends[i - 1];
    render();
    noteFrame(i);
    emit("scrub", { frame: i });
  };

  // Arrow keys drive the slider natively; light the matching arrow so it is
  // discoverable that they work at all.
  $("seek").addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") flashKey("kleft");
    if (e.key === "ArrowRight" || e.key === "ArrowUp") flashKey("kright");
  });
  el.querySelectorAll(".pl-key").forEach((k) => {
    k.onclick = () => {
      step(k.dataset.role === "kleft" ? -1 : 1);
      flashKey(k.dataset.role);
      $("seek").focus();
    };
  });

  stage.onclick = () => openFullscreen(slot, api);

  const api = {
    slot,
    label: spec.label || "",
    frames,
    imageAt: (i) => images[i] && images[i].src,
    cueAt: (i) => cueAt(i === 0 ? 0 : ends[i - 1]),
    current: () => frameAt(t),
    count: () => frames.length,
    goto: (i) => { $("seek").value = Math.min(Math.max(i, 0), frames.length - 1);
                   $("seek").dispatchEvent(new Event("input")); },
    stepBy: (d) => step(d),
    watched: () => everComplete || maxFrameSeen >= frames.length - 1,
    maxSeen: () => maxFrameSeen,
    destroy: () => { el.innerHTML = ""; },
  };

  render();
  setWatched();
  return api;
}

/* ---------------------------------------------------------------- fullscreen --
 * One overlay shared by the figure and every player. Clicking a panel opens it
 * large; Esc (or the close button) returns to the questions.
 *
 * The slider stays live inside the overlay -- inspecting a dense figure at
 * full size is exactly when you want to step frame by frame, and forcing a
 * close-and-reopen to move one frame would make the feature useless.
 */
let fullscreenOf = null;
let fullscreenApi = null;

function ensureOverlay() {
  let ov = document.getElementById("fsview");
  if (ov) return ov;
  ov = document.createElement("div");
  ov.id = "fsview";
  ov.innerHTML = `
    <div class="fs-bar">
      <span class="fs-title" data-role="fstitle"></span>
      <span class="fs-step" data-role="fsstep"></span>
      <button class="fs-close" data-role="fsclose">Close (Esc)</button>
    </div>
    <div class="fs-stage"><img data-role="fsimg" alt=""></div>
    <div class="fs-cue" data-role="fscue"></div>
    <div class="fs-controls">
      <span class="pl-key" data-role="fsleft">&larr;</span>
      <input class="fs-seek" data-role="fsseek" type="range" min="0" max="0" value="0" step="1">
      <span class="pl-key" data-role="fsright">&rarr;</span>
    </div>`;
  document.body.appendChild(ov);

  const q = (r) => ov.querySelector(`[data-role="${r}"]`);
  q("fsclose").onclick = closeFullscreen;
  q("fsleft").onclick = () => fullscreenApi && fullscreenApi.stepBy(-1);
  q("fsright").onclick = () => fullscreenApi && fullscreenApi.stepBy(1);
  q("fsseek").oninput = (e) => fullscreenApi && fullscreenApi.goto(Number(e.target.value));
  ov.addEventListener("click", (e) => { if (e.target === ov) closeFullscreen(); });
  return ov;
}

function paintFullscreen(index, cue) {
  const ov = document.getElementById("fsview");
  if (!ov || !fullscreenApi) return;
  const q = (r) => ov.querySelector(`[data-role="${r}"]`);
  const src = fullscreenApi.imageAt(index);
  if (src && q("fsimg").src !== src) q("fsimg").src = src;
  q("fsseek").max = String(Math.max(fullscreenApi.count() - 1, 0));
  q("fsseek").value = String(index);
  q("fsstep").textContent = `${index + 1} / ${fullscreenApi.count()}`;
  q("fscue").textContent = cue ? cue.text : "";
  q("fscue").style.display = cue && cue.text ? "block" : "none";
}

function openFullscreen(slot, api) {
  const ov = ensureOverlay();
  fullscreenOf = slot;
  fullscreenApi = api;
  ov.querySelector('[data-role="fstitle"]').textContent = api.label || "";
  ov.classList.add("fs-on");
  document.body.classList.add("fs-locked");
  const i = api.current();
  paintFullscreen(i, api.cueAt(i));
  if (window.Events) Events.push({ type: "fullscreen_open", slot });
}

/* A still image (the source figure) has no slider -- open it on its own. */
function openFullscreenImage(src, title) {
  const ov = ensureOverlay();
  fullscreenOf = "FIGURE";
  fullscreenApi = null;
  const q = (r) => ov.querySelector(`[data-role="${r}"]`);
  q("fstitle").textContent = title || "Original figure";
  q("fsimg").src = src;
  q("fsstep").textContent = "";
  q("fscue").style.display = "none";
  ov.classList.add("fs-on", "fs-still");
  document.body.classList.add("fs-locked");
  if (window.Events) Events.push({ type: "fullscreen_open", slot: "FIGURE" });
}

function closeFullscreen() {
  const ov = document.getElementById("fsview");
  if (!ov) return;
  if (window.Events && fullscreenOf) {
    Events.push({ type: "fullscreen_close", slot: fullscreenOf });
  }
  ov.classList.remove("fs-on", "fs-still");
  document.body.classList.remove("fs-locked");
  fullscreenOf = null;
  fullscreenApi = null;
}

document.addEventListener("keydown", (e) => {
  if (!fullscreenOf) return;
  if (e.key === "Escape") { closeFullscreen(); return; }
  if (!fullscreenApi) return;
  if (e.key === "ArrowLeft" || e.key === "ArrowDown") { fullscreenApi.stepBy(-1); e.preventDefault(); }
  if (e.key === "ArrowRight" || e.key === "ArrowUp") { fullscreenApi.stepBy(1); e.preventDefault(); }
});
