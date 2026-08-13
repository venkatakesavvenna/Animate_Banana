/* Shared stage gate: the checks, the sign-off, and the way forward.
 *
 * Every screen shows the same panel for its own stage, because a gate that is
 * only visible on some screens is worse than no gate at all -- 1a/1b/2b all
 * block downstream work, but for a while only the sequence and animation
 * screens could actually approve anything, so the one affirmative button on
 * the raster screen was "Promote", which does something else entirely.
 *
 * Promote and approve are deliberately separate:
 *   promote  = copy my edit over the pipeline's own file  (changes artifacts)
 *   approve  = I have looked at this and it is correct     (opens the gate)
 * Promoting without approving leaves the gate shut, which is what this panel
 * has to make obvious.
 *
 * Depends on page globals: SAMPLE_ID, and a banner(msg, kind) function.
 */

function gatePanelMarkup(stage, opts) {
  const o = opts || {};
  return `
    <div class="card" id="gp-card">
      <h3>
        <span>${o.title || "Stage"} <span class="hint" id="gp-summary"></span></span>
      </h3>
      <div id="gp-checks"></div>
      <div class="toolbar" style="margin-top:8px">
        <button id="gp-approve" onclick="gateApprove('${stage}')">Approve stage</button>
        ${o.nextHref
          ? `<button class="primary" id="gp-next" onclick="gateNext()">${
               o.nextLabel || "Next"} &rarr;</button>`
          : ""}
      </div>
      <p class="hint" id="gp-note"></p>
    </div>`;
}

let GATE_STATE = null;
let GATE_NEXT_HREF = null;

function gateNext() {
  if (GATE_NEXT_HREF) location.href = GATE_NEXT_HREF;
}

async function loadGate(stage, nextHref) {
  GATE_NEXT_HREF = nextHref || null;
  let d;
  try {
    d = await (await fetch(`/api/workflow-state/${SAMPLE_ID}`)).json();
  } catch (e) {
    return;
  }
  const g = d.gates[stage];
  if (!g || !document.querySelector("#gp-checks")) return;
  GATE_STATE = g;

  document.querySelector("#gp-summary").textContent =
    g.passed ? "— complete" : "— incomplete";
  document.querySelector("#gp-checks").innerHTML = g.checks.map(c => `
    <div style="font-size:12px;margin:2px 0;${c.passed ? "" : "color:var(--bad)"}">
      ${c.passed ? "&#10003;" : "&#10007;"} ${gpEsc(c.name)}
      ${c.detail ? `<span class="hint">— ${gpEsc(c.detail)}</span>` : ""}
    </div>`).join("");

  const approved = g.checks.some(c => c.name === "reviewer approved" && c.passed);
  const btn = document.querySelector("#gp-approve");
  btn.textContent = approved ? "Un-approve stage" : "Approve stage";
  btn.classList.toggle("promote", !approved);

  const next = document.querySelector("#gp-next");
  if (next) next.disabled = !g.passed;

  document.querySelector("#gp-note").textContent = g.passed
    ? "This stage is complete — you can move on."
    : `Blocked on: ${g.blocking.join(", ")}. Approving is separate from promoting:`
      + ` promote replaces the pipeline's file, approve says the result is correct.`;
}

async function gateApprove(stage) {
  const approved = GATE_STATE
    && GATE_STATE.checks.some(c => c.name === "reviewer approved" && c.passed);
  const r = await fetch(`/api/approve/${SAMPLE_ID}`, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({stage, approved: !approved}),
  });
  if (!r.ok) {
    gpBanner(`approve failed: ${r.status}`, false);
    return;
  }
  await loadGate(stage, GATE_NEXT_HREF);
  // The sample list colours finished samples green, so an approval has to
  // refresh it -- otherwise the sidebar keeps claiming work is outstanding
  // until the page is reloaded.
  if (typeof boot === "function" && document.querySelector("#list")) {
    boot();
  }
  gpBanner(approved ? "approval withdrawn" : "stage approved", true);
}

/* The screens disagree about `banner`'s second argument -- some take a boolean,
 * some take a class name -- so normalise here rather than making every caller
 * remember which one it is looking at. */
function gpBanner(msg, ok) {
  if (typeof banner !== "function") return;
  try {
    banner(msg, ok ? "ok" : "err");
  } catch (e) {
    banner(msg, !!ok);
  }
  // A boolean-style banner given "ok"/"err" produces a class that matches
  // neither, so call it again with the boolean when that is what it wants.
  const el = document.querySelector("#banner .banner");
  if (el && !el.classList.contains("ok") && !el.classList.contains("err")) {
    banner(msg, !!ok);
  }
}

function gpEsc(s) {
  return (s ?? "").toString().replace(
    /[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
}

/* ---- live run progress -------------------------------------------------
 *
 * A run can take minutes. "42s" says nothing about whether the sequencer is
 * still thinking or the critic has already sent a repair back, so poll the
 * server's own per-stage record and show the transitions as they happen.
 */

let RUN_POLL = null;

function runProgressStart(targetSel) {
  const el = document.querySelector(targetSel);
  if (!el) return;
  clearInterval(RUN_POLL);
  const tick = async () => {
    let p;
    try {
      p = await (await fetch(`/api/run-progress/${SAMPLE_ID}`)).json();
    } catch (e) { return; }
    el.innerHTML = runProgressMarkup(p);
  };
  tick();
  RUN_POLL = setInterval(tick, 1000);
}

async function runProgressStop(targetSel) {
  clearInterval(RUN_POLL);
  RUN_POLL = null;
  // One final read after the run returns. The 1s poll can otherwise take its
  // last sample just before the closing status is written, leaving the last
  // stage stuck on "running" forever even though it finished.
  if (targetSel) {
    const el = document.querySelector(targetSel);
    if (el) {
      try {
        const p = await (await fetch(`/api/run-progress/${SAMPLE_ID}`)).json();
        el.innerHTML = runProgressMarkup(p);
      } catch (e) { /* leave whatever the last poll showed */ }
    }
  }
}

const RP_MARK = {
  pending: "·", running: "▶", ok: "&#10003;", skipped: "–",
  unresolved: "!", failed: "&#10007;",
};
const RP_COLOR = {
  running: "var(--accent)", ok: "var(--ok)", failed: "var(--bad)",
  unresolved: "var(--warn)", skipped: "var(--dim)", pending: "var(--dim)",
};

function runProgressMarkup(p) {
  if (!p || !p.planned || !p.planned.length) return "";
  const rows = p.planned.map(r => `
    <div style="display:flex;gap:8px;align-items:baseline;font-size:12px;margin:3px 0;
                color:${RP_COLOR[r.status] || "var(--dim)"}">
      <span style="width:14px">${RP_MARK[r.status] || "·"}</span>
      <span style="font-family:ui-monospace,monospace;width:26px">${gpEsc(r.stage)}</span>
      <span style="flex:1">${gpEsc(r.label)}${
        r.status === "running" ? " — running…" : ""}</span>
      <span class="hint">${r.at != null ? r.at + "s" : ""}</span>
    </div>
    ${r.detail ? `<div class="hint" style="margin:0 0 4px 48px">${gpEsc(r.detail)}</div>` : ""}`
  ).join("");
  const head = p.running
    ? `<div class="hint" style="margin-bottom:6px">running… ${p.elapsed ?? 0}s</div>`
    : (p.error
        ? `<div class="hint" style="margin-bottom:6px;color:var(--bad)">stopped</div>`
        : `<div class="hint" style="margin-bottom:6px">finished in ${p.elapsed ?? 0}s</div>`);
  return `<div style="margin-top:10px;border-top:1px solid var(--line);padding-top:8px">
            ${head}${rows}</div>`;
}
