/* Question form: every question for the sample on ONE page.
 *
 * The earlier design showed one question at a time with a chip strip. The
 * team asked for all of a sample's questions on a single page so people flip
 * only between samples, with questions appearing PROGRESSIVELY: a question
 * carrying `show_if` renders only once the answers it depends on hold. A gate
 * answered "No" closes everything behind it and shows `gate_closed_text`.
 *
 * Every answer is POSTed as it is given (revisions append server-side), so a
 * reload rehydrates from `initial`.
 *
 * Contract:
 *   const f = mountForm(el, questionSet, {onAnswer, onComplete, canSubmit}, initial);
 *   f.answers() -> {id: value}   f.refreshGate()
 */

function mountForm(el, qset, hooks, initial) {
  hooks = hooks || {};
  const items = (qset.familiarity ? [qset.familiarity] : []).concat(qset.questions);
  const answers = Object.assign({}, initial || {});

  el.classList.add("qform");
  el.innerHTML = `
    <div class="qf-list" data-role="list"></div>
    <div class="qf-foot">
      <button class="qf-submit" data-role="submit" disabled>Submit and continue</button>
      <span class="qf-note" data-role="note"></span>
    </div>`;
  const $ = (r) => el.querySelector(`[data-role="${r}"]`);
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const answered = (q) => Object.prototype.hasOwnProperty.call(answers, q.id);
  const required = (q) => !q.optional;
  // A question is visible when every `show_if` condition holds.
  const visible = (q) => Object.entries(q.show_if || {}).every(([k, v]) => answers[k] === v);
  // A gate is a question others depend on; it "closed" when answered to a
  // value that no dependent accepts.
  const gates = new Set(items.flatMap((q) => Object.keys(q.show_if || {})));
  const gateClosed = () => items.some((q) => gates.has(q.id) && answered(q)
    && !items.some((d) => (d.show_if || {})[q.id] === answers[q.id]));

  function optionsHtml(q) {
    const sel = (v) => (answers[q.id] === v ? " qf-sel" : "");
    if (q.type === "yesno") {
      return `<div class="qf-row qf-yn">
        <button class="qf-opt qf-yes${sel(true)}" data-v="true">Yes</button>
        <button class="qf-opt qf-no${sel(false)}" data-v="false">No</button></div>`;
    }
    if (q.type === "score10") {
      const a = q.anchors || {};
      const opts = Array.from({ length: 11 }, (_, i) =>
        `<option value="${i}"${answers[q.id] === i ? " selected" : ""}>${i}</option>`).join("");
      return `<div class="qf-score">
        <select class="qf-dd" data-role="dd">
          <option value=""${answered(q) ? "" : " selected"} disabled>Score 0–10</option>${opts}</select>
        <span class="qf-anchor">0 = ${esc(a[0] || "")}</span>
        <span class="qf-anchor">10 = ${esc(a[10] || "")}</span></div>`;
    }
    if (q.type === "likert5") {
      const labels = q.labels || ["1", "2", "3", "4", "5"];
      return `<div class="qf-scale">` + labels.map((lab, i) =>
        `<button class="qf-opt${sel(i + 1)}" data-v="${i + 1}"><b>${i + 1}</b><span>${esc(lab)}</span></button>`
      ).join("") + `</div>`;
    }
    if (q.type === "choice_ab" || q.type === "choice_pair" || q.type === "select") {
      const opts = q.options || (q.type === "choice_ab"
        ? [{ value: "A", label: "Left (A)" }, { value: "no_preference", label: "No preference" },
           { value: "B", label: "Right (B)" }] : []);
      return `<div class="qf-row">` + opts.map((o) =>
        `<button class="qf-opt${sel(o.value)}" data-v="${esc(o.value)}">${esc(o.label)}</button>`
      ).join("") + `</div>`;
    }
    if (q.type === "text") {
      return `<textarea class="qf-text" data-role="freetext" rows="3"
                placeholder="Optional">${esc(answers[q.id] || "")}</textarea>
              <div><button class="qf-opt" data-role="savetext">Save</button></div>`;
    }
    return "";
  }

  function render() {
    const list = $("list");
    let html = "", n = 0;
    for (const q of items) {
      if (!visible(q)) continue;
      n += 1;
      html += `<div class="qf-q${answered(q) ? " qf-done" : ""}" data-q="${esc(q.id)}">
        <div class="qf-prompt"><span class="qf-num">Q${n}.</span> ${esc(q.prompt)}</div>
        ${q.help ? `<div class="qf-help">${esc(q.help)}</div>` : ""}
        ${optionsHtml(q)}</div>`;
    }
    if (gateClosed() && qset.gate_closed_text) {
      html += `<div class="qf-closed">${esc(qset.gate_closed_text)}</div>`;
    }
    list.innerHTML = html;

    list.querySelectorAll(".qf-q").forEach((box) => {
      const q = items.find((x) => x.id === box.dataset.q);
      box.querySelectorAll(".qf-opt[data-v]").forEach((b) => {
        b.onclick = () => {
          let v = b.dataset.v;
          if (q.type === "likert5") v = Number(v);
          if (q.type === "yesno") v = v === "true";
          record(q, v);
        };
      });
      const dd = box.querySelector('[data-role="dd"]');
      if (dd) dd.onchange = () => { if (dd.value !== "") record(q, Number(dd.value)); };
      const save = box.querySelector('[data-role="savetext"]');
      if (save) save.onclick = () => record(q, box.querySelector('[data-role="freetext"]').value);
    });
    refreshGate();
  }

  function record(q, value) {
    const revision = answered(q);
    const previous = answers[q.id];
    answers[q.id] = value;
    // Answering a gate differently hides its dependents; their stale answers
    // must not linger and count as "answered" for an invisible question.
    for (const d of items) {
      if (Object.keys(d.show_if || {}).includes(q.id) && !visible(d) && answered(d)) {
        delete answers[d.id];
      }
    }
    if (window.Events) {
      Events.push({ type: revision ? "answer_revised" : "answer_given",
                    payload: { question_id: q.id, value, previous: revision ? previous : undefined } });
    }
    if (hooks.onAnswer) hooks.onAnswer(q.id, value, revision);
    render();
  }

  function complete() {
    return items.filter((q) => required(q) && visible(q)).every(answered);
  }

  function refreshGate() {
    const watched = hooks.canSubmit ? hooks.canSubmit() : true;
    const missing = items.filter((q) => required(q) && visible(q) && !answered(q));
    const ok = missing.length === 0 && watched;
    $("submit").disabled = !ok;
    $("note").textContent = !watched
      ? "Move the slider to the end of each animation first."
      : (missing.length ? `${missing.length} question${missing.length > 1 ? "s" : ""} left` : "");
    // Nothing is asked before the stimulus has been seen.
    const list = $("list");
    if (!watched) {
      if (!list.querySelector(".qf-locked")) {
        list.innerHTML = `<div class="qf-locked">Move the slider all the way to the end to see the questions.</div>`;
      }
    } else if (list.querySelector(".qf-locked")) {
      render();
    }
  }

  $("submit").onclick = () => { if (hooks.onComplete) hooks.onComplete(answers); };
  render();
  return { complete, refreshGate, answers: () => answers };
}
