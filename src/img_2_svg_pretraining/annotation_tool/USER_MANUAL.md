# Raster Region Annotator — User Manual

*A guide for reviewers. No technical background needed.*

---

## What is this tool, and why does your work matter?

We are teaching a computer to read scientific diagrams. Some diagrams have
**embedded raster content** — real photos, plots, charts, screenshots,
icons, or logos pasted into the figure, as opposed to the clean boxes and
arrows drawn around them. To learn what those regions look like, the
computer needs thousands of correct examples, and that's what you create
with this tool.

**This tool is only for that embedded raster content** — photos, plots,
charts, screenshots, icons, logos. It is **not** for the boxes, arrows, or
labels that make up the diagram's structure — leave those alone even if you
see them on screen.

The computer has already made a first guess: for each diagram, it has
marked where it *thinks* the raster regions are. Your job is to **check
every guess, fix the wrong ones, and approve the good ones**. For each
region you approve, the tool saves two things:

1. **A dot** — a point sitting on the region.
2. **A highlight** — a colored overlay covering exactly that region, which
   the tool draws automatically from your dot.

You never draw the highlight by hand. You place and adjust dots; the tool
redraws the highlight instantly every time you change a dot.

---

## Opening the tool

1. Open the web link your team lead gave you (it looks like
   `http://something:8600`) in Chrome or any modern browser.
2. The first page of the day can take **up to a minute** to appear while the
   system warms up. After that it's fast.
3. You'll see a diagram on the left and a list of raster regions on the
   right.

> **Tip:** your approved work is saved the moment you approve it. You can
> close the tab at any time and nothing approved is ever lost.

---

## The screen at a glance

```
┌────────────────────────────────────────────┬──────────────────┐
│  ⬅ Prev   Next ➡   [image chooser]         │  counts summary  │
│  Add point | Negative pt | Draw box | Undo │  buttons for the │
│  [zoom]                                    │  region you're   │
│                                            │  on              │
│                                            │──────────────────│
│           THE DIAGRAM (click here)         │  list of raster  │
│                                            │  regions (one    │
│  caption explaining the dot colors         │  row each)       │
│                                            │  session totals  │
└────────────────────────────────────────────┴──────────────────┘
```

- **Left side** — the diagram. All your clicking happens here.
- **Right side** — one row per raster region, with a small picture and a
  colored status icon, plus the action buttons for the region you're
  working on.

### What the colors mean

| On the diagram | Meaning |
|---|---|
| 🔴 **Red dashed circle** | The computer's guess — not checked by a human yet |
| 🟢 **Green dot** | A dot a human placed or confirmed |
| 🟣 **Purple dot with a bar** | A "NOT this" marker (see *Fixing a bad highlight*) |
| 🟡 **Amber ring around a dot** | The dot you currently have selected |
| **Blue highlight** | The area the tool thinks is the current raster region |
| **Faint green highlight** | Regions you've already approved |

| In the right-hand list | Meaning |
|---|---|
| 🔴 | Not looked at yet |
| 🟠 | You're checking the dot |
| 🟡 | You're checking the highlight |
| 🟢 | Approved — done! |
| ⚫ | Rejected (it wasn't a raster region) |
| 🔗 | Merged (it was a duplicate) |

---

## The basic routine — one region at a time

You always work on **one raster region at a time**, start to finish, then
move to the next. Click **open** on a row in the right-hand list to start
it.

### Step 1 — Check the dot

A red dashed circle shows the computer's guess. Ask yourself:

- **Is the dot sitting on a photo, plot, chart, screenshot, icon, or logo
  embedded in the diagram?**
  - ✔ **Yes** → click **Confirm points** (or press the **`c`** key).
  - ✖ **It's in the wrong place, but still on a real raster region** → move
    it (see below), then Confirm.
  - ✖ **It's not on a raster region at all** — it's on a plain box, an
    arrow, a text label, or empty space that's part of the diagram's
    structure, not embedded content → click **Reject ✖**.
  - ✖ **It's a second dot on a raster region that already has one** → pick
    the other region's ID in the small dropdown and click **Merge ⇒**.

**To move a dot — two clicks, not a drag:**
1. Click the dot once. An amber ring appears around it — it's selected.
2. Click where the dot *should* be. It jumps there.

(Dragging doesn't work in this tool — it's always click, then click.)

### Step 2 — Check the highlight

The moment you confirm, a **blue highlight** appears over the region. Ask:

- **Does the blue cover the whole raster region — and nothing else** (no
  spilling onto the diagram's boxes, arrows, or background)?
  - ✔ **Yes** → click **Accept ✔** (or press the **`a`** key). The row turns
    green, your work is saved, and the tool moves you to the next region
    automatically.
  - ✖ **No** → fix it (next section), then Accept.

That's the whole job: *check the dot, check the highlight, accept.* Most
regions take a few seconds.

---

## Fixing a bad highlight

The highlight redraws instantly after every change, so just keep adjusting
until it looks right.

**The highlight spills outside the raster region** (covers a neighboring
box, an arrow, the background):
1. Turn on **Negative pt** in the toolbar (or press **`n`**).
2. Click on the *wrongly covered* area. A purple dot appears meaning
   "NOT this part", and the highlight shrinks away from it.
3. Turn **Negative pt** off (press **`n`** again) before placing normal dots.

**The highlight misses part of the raster region:**
- Simply click on the missed part (with **Add point** on, which it is by
  default). A green dot appears there and the highlight grows to include it.

**The highlight is still wrong no matter what** (rare — last resort):
1. Turn on **Draw box** in the toolbar.
2. Click one corner of the raster region, then the opposite corner.
3. The tool redraws the highlight from your box. Turn **Draw box** off and
   continue.

**Deleted or misplaced something?** Click **Undo ↩** (or press
**Ctrl+Z**). You can undo your last 20 changes on the current region.

**To remove a dot:** click it (amber ring appears), then click
**Delete point 🗑** in the right panel.

---

## Other situations

**The computer missed a raster region entirely**
Click **➕ New instance** in the right panel, then click on the missed
photo/plot/icon/screenshot in the diagram. A dot and highlight appear —
review and accept as usual.

**You approved something by mistake**
Open that row again and click **Reopen ♻**. It goes back to Step 1 so you
can fix and re-approve it (or reject it).

**The raster region is tiny and hard to see**
Use the **zoom dropdown** above the diagram (200% or 400%). The view centers
on the region you're working on.

**Finished all raster regions on this image?**
When every row is 🟢, ⚫, or 🔗, click **Next ➡** for the next image — or
**Next flagged ⏭** to jump straight to the next image that still has
unchecked regions.

---

## Keyboard cheat sheet

| Key | What it does |
|---|---|
| `c` | Confirm the dot(s) — "yes, this is on a raster region" |
| `a` | Accept — "the highlight is perfect, save it" |
| `n` | Switch the purple "NOT this" mode on/off |
| `Ctrl+Z` | Undo your last change |

---

## Good to know

- **This tool is for embedded raster content only** — photos, plots,
  charts, screenshots, icons, logos. If the computer's guess landed on a
  plain diagram box, an arrow, or a text label, that's a wrong guess:
  reject it.
- **Saving is automatic.** Every Accept, Reject, or Merge is saved
  immediately. There is no Save button and no way to lose approved work.
- **Closing the tab is safe.** At worst, unfinished edits on the *one*
  region you were in the middle of are lost — everything approved stays.
- **One region at a time.** Finish the region you're on before opening
  another; the tool is built around this rhythm.
- **The Accept button refuses to work** if a region has no green (positive)
  dot — that's intentional. Add a dot on the region first.
- **Don't over-use Draw box.** It's a rescue tool. Dots (green + purple)
  handle almost every case and give better training data.
- **Quality beats speed.** A wrong approval teaches the computer the wrong
  thing. If you're unsure whether something counts as a raster region, ask
  your team lead rather than guessing.

## If something looks broken

| Problem | What to do |
|---|---|
| Page won't load / spinner forever | Wait a minute (first load is slow), then refresh the browser tab |
| Clicked the diagram, nothing happened | Check **Add point** is switched on; if a dot got an amber ring instead, you clicked too close to it — click elsewhere, then try again |
| Highlight looks stuck / stale | Move any dot slightly — the highlight always redraws after a change |
| Everything frozen | Refresh the tab. Your approved work is safe. If it persists, tell your team lead |

Happy annotating! Every raster region you approve makes the computer a
little smarter.
