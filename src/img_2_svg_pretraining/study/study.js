/* Shared helpers: participant session, fetch, and the event queue.
 *
 * Events are batched and flushed on pause/complete/submit and on pagehide via
 * sendBeacon, so closing the tab mid-trial does not lose the playback record
 * that the QC checks are computed from.
 */

const PID_KEY = "animatebanana_study_pid";

const Session = {
  get: () => localStorage.getItem(PID_KEY),
  set: (pid) => localStorage.setItem(PID_KEY, pid),
  clear: () => localStorage.removeItem(PID_KEY),
};

async function api(path, opts) {
  opts = opts || {};
  const headers = Object.assign({ "Content-Type": "application/json" }, opts.headers || {});
  const pid = Session.get();
  if (pid) headers["X-Participant"] = pid;
  const res = await fetch(path, Object.assign({}, opts, { headers }));
  if (!res.ok) {
    let detail = "";
    try { detail = JSON.stringify(await res.json()); } catch (e) { detail = res.statusText; }
    const err = new Error(`${res.status} ${detail}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

const Events = {
  queue: [],
  trialId: null,
  push(e) {
    if (!this.trialId) return;
    this.queue.push(Object.assign({ trial_id: this.trialId }, e));
    if (this.queue.length >= 25) this.flush();
  },
  flush(useBeacon) {
    if (!this.queue.length) return;
    const body = JSON.stringify({ events: this.queue.splice(0, this.queue.length) });
    const pid = Session.get();
    if (useBeacon && navigator.sendBeacon) {
      // Beacon cannot carry a header, so the id rides in the query string.
      navigator.sendBeacon(`/api/events?pid=${encodeURIComponent(pid)}`,
                           new Blob([body], { type: "application/json" }));
      return;
    }
    fetch("/api/events", { method: "POST", keepalive: true,
      headers: { "Content-Type": "application/json", "X-Participant": pid },
      body }).catch(() => {});
  },
};

window.addEventListener("pagehide", () => {
  Events.push({ type: "page_hide" });
  Events.flush(true);
});
document.addEventListener("visibilitychange", () => {
  // Tab switching is a real signal for QC -- a trial answered while the tab
  // was hidden was not answered from the stimulus.
  Events.push({ type: document.hidden ? "tab_hidden" : "tab_visible" });
  if (document.hidden) Events.flush();
});
window.addEventListener("resize", () => {
  clearTimeout(window.__rz);
  window.__rz = setTimeout(() => Events.push({
    type: "viewport_resize",
    payload: { w: window.innerWidth, h: window.innerHeight } }), 400);
});
setInterval(() => Events.flush(), 5000);

function el(sel) { return document.querySelector(sel); }
function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
