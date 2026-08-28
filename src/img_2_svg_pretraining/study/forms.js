/* Question flow.
 *
 * One question at a time. Answering it collapses it into a history strip of
 * chips; clicking a chip reopens that question to revise it.
 *
 * Revision is expected, not exceptional: a participant understands the figure
 * better by question eight than they did at question one, and the team asked
 * for exactly this. Every revision POSTs a new answer -- the server appends
 * rather than overwrites, so the whole trail survives and analysis reads the
 * newest.
 *
 * Contract:
 *   const f = mountForm(el, questionSet, {onAnswer, onComplete, canSubmit});
 *   f.complete() -> bool     f.refreshGate()
 */

function mountForm(el, qset, hooks) {
  hooks = hooks || {};
  // Familiarity first: it is asked before the participant has formed an
  // opinion of the animation, so it measures the topic rather than the video.
  // familiarity is null once this participant has already rated this
  // diagram's topic in an earlier section.
  const items = (qset.familiarity ? [qset.familiarity] : []).concat(qset.questions);
  const answers = {};
  let cursor = 0;

  el.classList.add("qform");
  el.innerHTML = `
    <div class="qf-strip" data-role="strip"></div>
    <div class="qf-current" data-role="current"></div>
    <div class="qf-nav">
      <button class="qf-move" data-role="prev">&larr; Previous</button>
      <button class="qf-move" data-role="next">Next &rarr;</button>
    </div>
    <div class="qf-foot">
      <button class="qf-submit" data-role="submit" disabled>Submit and continue</button>
      <span class="qf-note" data-role="note"></span>
    </div>`;
  const $ = (role) => el.querySelector(`[data-role="${role}"]`);

  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function isAnswered(q) {
    return Object.prototype.hasOwnProperty.call(answers, q.id);
  }
  function required(q) { return !q.optional; }

  function shortValue(q, v) {
    if (q.type === "likert5") return String(v);
    if (q.type === "yesno") return v ? "Yes" : "No";
    if (q.type === "choice_ab") return v === "no_preference" ? "=" : v;
    if (q.type === "select") {
      const o = (q.options || []).find((o) => o.value === v);
      return o ? o.label.split(" ").slice(0, 3).join(" ") : String(v);
    }
    if (q.type === "text") return v ? "written" : "-";
    return String(v);
  }

  function optionsHtml(q) {
    if (q.type === "likert5") {
      const labels = q.labels || ["1", "2", "3", "4", "5"];
      return `<div class="qf-scale">` + labels.map((lab, i) =>
        `<button class="qf-opt${answers[q.id] === i + 1 ? " qf-sel" : ""}" data-v="${i + 1}">
           <b>${i + 1}</b><span>${esc(lab)}</span></button>`).join("") + `</div>`;
    }
    if (q.type === "yesno") {
      return `<div class="qf-row">
        <button class="qf-opt${answers[q.id] === true ? " qf-sel" : ""}" data-v="true">Yes</button>
        <button class="qf-opt${answers[q.id] === false ? " qf-sel" : ""}" data-v="false">No</button></div>`;
    }
    if (q.type === "choice_ab") {
      return `<div class="qf-row">
        <button class="qf-opt qf-a${answers[q.id] === "A" ? " qf-sel" : ""}" data-v="A">Left (A)</button>
        <button class="qf-opt${answers[q.id] === "no_preference" ? " qf-sel" : ""}" data-v="no_preference">No preference</button>
        <button class="qf-opt qf-b${answers[q.id] === "B" ? " qf-sel" : ""}" data-v="B">Right (B)</button></div>`;
    }
    if (q.type === "select") {
      return `<div class="qf-col">` + (q.options || []).map((o) =>
        `<button class="qf-opt${answers[q.id] === o.value ? " qf-sel" : ""}" data-v="${esc(o.value)}">${esc(o.label)}</button>`
      ).join("") + `</div>`;
    }
    if (q.type === "text") {
      return `<textarea class="qf-text" data-role="freetext" rows="3"
                placeholder="Optional">${esc(answers[q.id] || "")}</textarea>
              <div><button class="qf-opt" data-role="savetext">Save</button></div>`;
    }
    return "";
  }

  function renderStrip() {
    $("strip").innerHTML = items.map((q, i) => {
      if (!isAnswered(q)) return "";
      return `<button class="qf-chip${i === cursor ? " qf-chip-on" : ""}" data-i="${i}"
                title="${esc(q.prompt)}">${esc(q.id)} <b>${esc(shortValue(q, answers[q.id]))}</b></button>`;
    }).join("");
    $("strip").querySelectorAll(".qf-chip").forEach((chip) => {
      chip.onclick = () => {
        cursor = Number(chip.dataset.i);
        if (window.Events) {
          Events.push({ type: "question_revisit",
                        payload: { question_id: items[cursor].id, index: cursor } });
        }
        renderCurrent();
      };
    });
  }

  function renderCurrent() {
    // Nothing is asked before the participant has seen the thing they are
    // being asked about. Showing the questions first invites answering from
    // the thumbnail.
    if (hooks.canSubmit && !hooks.canSubmit()) {
      $("current").innerHTML = `<div class="qf-locked">
        Move the slider all the way to the end to see the questions.</div>`;
      $("strip").innerHTML = "";
      $("prev").disabled = $("next").disabled = true;
      refreshGate();
      return;
    }
    const q = items[cursor];
    if (!q) {
      $("current").innerHTML = `<div class="qf-done">All questions answered.
        You can revisit any of them above before submitting.</div>`;
      renderStrip(); refreshGate(); return;
    }
    const n = items.filter(isAnswered).length;
    $("current").innerHTML = `
      <div class="qf-meta">${q.section ? esc(q.section) + " · " : ""}question ${cursor + 1} of ${items.length}
        <span class="qf-count">${n} answered</span></div>
      <div class="qf-prompt">${esc(q.prompt)}</div>
      ${q.help ? `<div class="qf-help">${esc(q.help)}</div>` : ""}
      ${optionsHtml(q)}
      ${q.optional ? `<div class="qf-skip"><button data-role="skip">Skip</button></div>` : ""}`;

    $("current").querySelectorAll(".qf-opt[data-v]").forEach((b) => {
      b.onclick = () => {
        let v = b.dataset.v;
        if (q.type === "likert5") v = Number(v);
        if (q.type === "yesno") v = v === "true";
        record(q, v);
      };
    });
    const save = el.querySelector('[data-role="savetext"]');
    if (save) save.onclick = () => record(q, el.querySelector('[data-role="freetext"]').value);
    const skip = el.querySelector('[data-role="skip"]');
    if (skip) skip.onclick = () => record(q, "");

    renderStrip();
    refreshGate();
    renderNav();
  }

  function renderNav() {
    $("prev").disabled = cursor <= 0;
    // Forward is allowed onto any question already answered, and onto the
    // first unanswered one -- so revising an early answer never traps you.
    $("next").disabled = cursor >= items.length - 1;
  }

  function go(delta) {
    const next = cursor + delta;
    if (next < 0 || next > items.length - 1) return;
    cursor = next;
    if (window.Events) {
      Events.push({ type: delta > 0 ? "question_next" : "question_prev",
                    payload: { to: items[cursor].id, index: cursor } });
    }
    renderCurrent();
  }

  function record(q, value) {
    const revision = isAnswered(q);
    const previous = answers[q.id];
    answers[q.id] = value;
    if (window.Events) {
      Events.push({ type: revision ? "answer_revised" : "answer_given",
                    payload: { question_id: q.id, value,
                               previous: revision ? previous : undefined,
                               question_index: cursor } });
    }
    if (hooks.onAnswer) hooks.onAnswer(q.id, value, revision);
    // On a revision, stay put so the participant sees the change land; on a
    // first answer, advance to the next unanswered question.
    if (!revision) {
      const next = items.findIndex((it, i) => i > cursor && !isAnswered(it));
      cursor = next === -1 ? items.findIndex((it) => !isAnswered(it)) : next;
      if (cursor === -1) cursor = items.length;
    }
    renderCurrent();
  }

  function complete() {
    return items.filter(required).every(isAnswered);
  }

  function refreshGate() {
    const missing = items.filter(required).filter((q) => !isAnswered(q));
    const watched = hooks.canSubmit ? hooks.canSubmit() : true;
    const ok = missing.length === 0 && watched;
    $("submit").disabled = !ok;
    $("note").textContent = !watched
      ? "Move the slider to the end of each animation first."
      : (missing.length ? `${missing.length} question${missing.length > 1 ? "s" : ""} left` : "");
    // The gate opening is what reveals the questions.
    if (watched && el.querySelector(".qf-locked")) renderCurrent();
  }

  $("prev").onclick = () => go(-1);
  $("next").onclick = () => go(1);
  $("submit").onclick = () => { if (hooks.onComplete) hooks.onComplete(answers); };

  renderCurrent();
  return { complete, refreshGate, answers: () => answers };
}
