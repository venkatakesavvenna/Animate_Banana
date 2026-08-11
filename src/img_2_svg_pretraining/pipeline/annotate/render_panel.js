/* Shared render + discard controls.
 *
 * Both Stage-1 screens need the same two actions, and they need to behave
 * identically wherever they appear -- a render button that means something
 * different on each screen is the bug this replaces. The `source` argument is
 * what differs: each screen renders the artifact it is actually editing.
 */

function renderPanelMarkup(source) {
  return `
    <div class="card">
      <h3>Render <span style="font-weight:400;color:var(--dim)" id="rp-src"></span></h3>
      <button class="primary" style="width:100%" onclick="doRender('${source}')">Render diagram</button>
      <div id="rp-out" style="margin-top:10px"></div>
    </div>
    <div class="card">
      <h3>Discard</h3>
      <button style="width:100%;border-color:var(--bad);color:var(--bad)" onclick="doDiscard()">
        Discard this annotation
      </button>
      <p class="hint">Writes the sample off as unsalvageable: deletes every human artifact,
        reverts a promotion, and marks it discarded. The record is kept so nobody redoes it.</p>
    </div>`;
}

async function doRender(source) {
  const out = document.querySelector("#rp-out");
  const label = document.querySelector("#rp-src");
  out.innerHTML = `<div class="hint">compiling…</div>`;
  if (label) label.textContent = "";

  // Cache-bust: the artifact changes under the same URL after every edit.
  const url = `/api/render/${SAMPLE_ID}?source=${source}&t=${Date.now()}`;
  let r;
  try {
    r = await fetch(url);
  } catch (e) {
    out.innerHTML = `<div class="hint" style="color:var(--bad)">render request failed: ${e}</div>`;
    return;
  }

  if (!r.ok) {
    let d = {};
    try { d = await r.json(); } catch (e) { /* non-JSON error body */ }
    const msg = d.log || d.error || `HTTP ${r.status}`;
    out.innerHTML =
      `<pre style="white-space:pre-wrap;font-size:11px;color:var(--bad);margin:0;
                   max-height:320px;overflow:auto">${escapeHtml(String(msg).slice(-3000))}</pre>`;
    if (label && d.from) label.textContent = `(${d.from} — does not compile)`;
    return;
  }

  const from = r.headers.get("X-Render-Source");
  if (label && from) label.textContent = `(${from})`;
  const blob = await r.blob();
  out.innerHTML = `<img src="${URL.createObjectURL(blob)}"
                        style="max-width:100%;border-radius:4px;background:#fff;display:block">`;
}

async function doDiscard() {
  const reason = prompt(
    "Discard this annotation as unsalvageable?\n\n" +
    "Deletes all human edits, reverts any promotion, and marks the sample discarded.\n" +
    "Reason (optional):");
  if (reason === null) return;   // cancelled

  const r = await fetch(`/api/discard/${SAMPLE_ID}`, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({reason, author: "me"}),
  });
  const d = await r.json().catch(() => ({}));
  if (!r.ok || !d.ok) {
    alert(`discard failed: ${d.error || r.status}`);
    return;
  }
  alert(`Discarded. Removed ${d.removed.length} file(s).`);
  location.reload();
}

function escapeHtml(s) {
  return (s ?? "").toString().replace(
    /[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
}
