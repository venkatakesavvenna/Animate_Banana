# Debugging a library from source: how the vLLM parity investigation was actually done

**Purpose:** the two parity documents
([`vllm_serve_offline_parity.md`](./vllm_serve_offline_parity.md),
[`vllm_image_pipeline_parity.md`](./vllm_image_pipeline_parity.md)) present *conclusions*.
This document presents the *method* — the specific commands, the reasoning that chose each
one, the probe scripts, and the wrong turns. The goal is that you can run this playbook
yourself on a different model, a different vLLM feature, or a completely different library.

**The one-sentence version:** when two code paths that should behave the same don't, find
where they converge, then binary-search upward for the first place they differ — reading
source as ground truth and confirming every step with an executable probe.

---

## Table of contents

- [Part 1 — Principles](#part-1--principles)
- [Part 2 — The playbook](#part-2--the-playbook)
  - [Step 0: Find the actual source](#step-0-find-the-actual-source)
  - [Step 1: Turn the symptom into a search string](#step-1-turn-the-symptom-into-a-search-string)
  - [Step 2: Map the two paths and find the junction](#step-2-map-the-two-paths-and-find-the-junction)
  - [Step 3: Enumerate the inputs to the junction](#step-3-enumerate-the-inputs-to-the-junction)
  - [Step 4: Read defaults like an adversary](#step-4-read-defaults-like-an-adversary)
  - [Step 5: Bound the blast radius](#step-5-bound-the-blast-radius)
  - [Step 6: Write a probe](#step-6-write-a-probe)
  - [Step 7: Differential-test the edges](#step-7-differential-test-the-edges)
  - [Step 8: Rule things out, in writing](#step-8-rule-things-out-in-writing)
- [Part 3 — Techniques worth internalizing](#part-3--techniques-worth-internalizing)
- [Part 4 — The wrong turns](#part-4--the-wrong-turns)
- [Part 5 — Command cookbook](#part-5--command-cookbook)
- [Part 6 — Applying this elsewhere](#part-6--applying-this-elsewhere)

---

## Part 1 — Principles

**1. Installed source is ground truth. Docs and release notes are hearsay.**
Every claim in the parity docs is anchored to a `file.py:line` in the venv that will
actually run. Documentation describes intent, lags reality, and is often written against a
different version. The code in `site-packages` is what executes tonight.

**2. Read the code, but confirm by executing it.**
Reading tells you what *should* happen; running tells you what *does*. Every significant
claim in those docs has a probe script behind it. The chat-template equivalence claim
looked obviously true from reading the Jinja — it was still worth 10 lines to print both
strings and compare, because "obviously true" is where bugs hide.

**3. A behavioural difference must have a mechanical cause.**
"Online just feels better" is not an explanation. Two processes running identical weights
on identical hardware produce different output only because some byte reaching the model
differs, or some kernel ran with different shapes. Refuse to accept a difference you can't
point at in code. This is what turned a vague "quality is worse" into
`UsageContext.ENGINE_CONTEXT` missing from a dict.

**4. Prefer eliminating hypotheses to confirming them.**
Confirming your favourite theory stops early and misses the real cause. Aim to *kill*
candidates. Roughly half of each parity doc is a "ruled out" section, and that half is
what prevents a colleague — or you in three months — from re-investigating the chat
template for the fourth time.

**5. Suspect your own code before the library's.**
Both real image-side divergences turned out to be in `load_cv2_pil`, ours, not vLLM's.
Library code has been run by thousands of people; glue code has been run by you.

**6. Write down the negative results.**
A finding of "these are identical, here is the proof" is nearly as valuable as a bug, and
far more perishable — nobody re-derives it, they just re-suspect it.

---

## Part 2 — The playbook

### Step 0: Find the actual source

You cannot grep what you cannot locate. Two things must be pinned before anything else:
the library source, and the model files.

```bash
# Which interpreter, which version, where does the package live?
VENV=/fsxvision_new/srihari.bandarupalli/environments/gemma4_new
$VENV/bin/python -c "import vllm, transformers, os
print('vllm', vllm.__version__, os.path.dirname(vllm.__file__))
print('tf  ', transformers.__version__, os.path.dirname(transformers.__file__))"
```

Set a shell variable for the root immediately — you will type it fifty times:

```bash
VLLM=$VENV/lib/python3.12/site-packages/vllm
TF=$VENV/lib/python3.12/site-packages/transformers
```

For model files, **do not go hunting with `find`.** HuggingFace tells you where it put
them:

```bash
env | grep -i "^HF_\|^TRANSFORMERS_\|^HUGGINGFACE_"
# HF_HOME=/hf_cache   →  weights live under /hf_cache/hub/
ls /hf_cache/hub/ | grep -i gemma
ls /hf_cache/hub/models--google--gemma-4-31B-it/snapshots/*/
```

That last listing is worth reading carefully rather than skimming — its *absences* are
informative. This model ships `processor_config.json` but **no** `preprocessor_config.json`,
and ships `chat_template.jinja` as a standalone file rather than embedding the template in
`tokenizer_config.json`. Both facts changed where to look next.

> **Why this matters:** an early attempt used `find` across `/fsxvision_new` and
> `/projects` to locate the weights. It ran past 120 seconds on network storage and had to
> be backgrounded. Checking `HF_HOME` took two seconds and was exact. On FSx/Lustre,
> treat any unscoped `find` as a mistake.

### Step 1: Turn the symptom into a search string

The best entry point into an unfamiliar codebase is a string you have literally seen. Log
lines are perfect: they are unique, greppable, and sit exactly at the code that made the
decision you care about.

The investigation started from a real `vllm serve` log line:

```
Default vLLM sampling parameters have been overridden by the model's `generation_config.json`:
{'temperature': 1.0, 'top_k': 64, 'top_p': 0.95}
```

The obvious grep is to pick a chunk that reads like a fixed phrase:

```bash
grep -rn "have been overridden by the model" $VLLM/     # ← returns NOTHING
```

Zero hits — and this is the single most common way this step fails, so it is worth
dwelling on. The source is:

```python
logger.warning_once(
    "Default vLLM sampling parameters have been overridden by %s: `%s`. "
    "If this is not intended, please relaunch vLLM instance "
    "with `--generation-config vllm`.",
    "the model's `generation_config.json`" if src == "auto" else src,
    str(diff_sampling_param),
)
```

`"the model's generation_config.json"` is a **`%s` argument**, not part of the format
string. The chosen fragment straddled the substitution point, so it exists only in the
rendered output and never in the code.

The fix is to grep the longest run you are confident is *literal*, and stop before any
value that looks like it varies:

```bash
grep -rn "sampling parameters have been overridden" $VLLM/ --include=*.py
# → config/model.py:1529
```

**Heuristic:** anything in a log line that is a filename, number, enum value, or model
name is probably interpolated. Cut your search string before it. If a grep of a message
you have definitely seen returns nothing, assume interpolation before assuming you're in
the wrong tree.

That one hit lands inside the function performing the override, which gives the mechanism
(`get_diff_sampling_param`), which gives the caller, which answers the real question:
*does that caller exist on the other path?* It did not — the merge lives in the HTTP
protocol layer, so offline never reaches it. One grep, one whole divergence.

The same line also hands you the config surface for free: `--generation-config vllm`
appears right there in the message, which is how you learn the flag exists without
touching the docs.

When you have no log line, grep for the config key or CLI flag instead:

```bash
grep -rn "max_num_batched_tokens" $VLLM/ --include=*.py | grep -v test | head -30
```

Filter aggressively. `--include=*.py`, excluding tests, and `head` are what keep the
signal readable.

### Step 2: Map the two paths and find the junction

This is the core move for any "A behaves differently from B" problem.

Draw both paths from entry point downward until they meet. Everything *below* the junction
is shared code and cannot be the cause. Everything *above* it is your search space. This
converts an unbounded question into a bounded one.

For vLLM:

```
ONLINE                                    OFFLINE
vllm serve → api_server.py                LLM / AsyncLLM
  → ChatCompletionRequest                   → your prompt dict
  → to_sampling_params()                    → SamplingParams you built
  → renderer.render_chat()                  → your apply_chat_template
        └──────────────┬─────────────────────────────┘
                       ▼
              input_processor.py      ← JUNCTION
                       ▼
              shared: processor, scheduler, model
```

Finding the junction is usually a matter of following calls downward from both ends until
you hit the same filename. Then verify it really is shared:

```bash
grep -rn "input_processor" $VLLM/v1/engine/ $VLLM/entrypoints/ | grep -v test
```

Once located, the discipline is: **any hypothesis about code below the junction is dead on
arrival.** That is what let the image doc assert stages 5–12 are identical without testing
each one — they are literally the same function invocation.

### Step 3: Enumerate the inputs to the junction

If both paths call the same function, they can only differ in what they pass to it. So
list the arguments. That list *is* your complete hypothesis space.

`MultiModalProcessor.apply(prompt, mm_data, hf_processor_mm_kwargs)` takes three things,
so there are exactly three candidate divergences for images:

| argument | check | verdict |
|---|---|---|
| `prompt` | render both ways, compare strings | identical |
| `mm_data` | compare decoded pixel arrays | **differs** (alpha) |
| `hf_processor_mm_kwargs` | trace who sets it on each side | identical |

Three checks, complete coverage, no guessing. This is dramatically better than reading
code hoping to notice something.

For each argument, ask *who can set this*, and grep for all the setters:

```bash
grep -n "mm_processor_kwargs" $VLLM/entrypoints/openai/chat_completion/protocol.py
grep -n "mm_processor_kwargs" $VLLM/renderers/base.py
```

Both paths *can* set it; neither of ours *does*. That's the answer — and note it required
checking our code, not just vLLM's.

### Step 4: Read defaults like an adversary

The single highest-value finding in this whole investigation came from reading a defaults
lookup with suspicion.

```bash
grep -n "def get_batch_defaults" -A 30 $VLLM/engine/arg_utils.py
```

which revealed a dict keyed by `UsageContext`, and then:

```python
max_num_batched_tokens = defaults.get(usage_context, DEFAULT_MAX_NUM_BATCHED_TOKENS)
```

The bug pattern: **`.get()` with a fallback silently succeeds on a missing key.** So the
question "what is `usage_context` here?" becomes critical:

```bash
grep -n "usage_context" $VLLM/v1/engine/async_llm.py | head
grep -n "UsageContext\." $VLLM/entrypoints/llm.py $VLLM/entrypoints/openai/api_server.py
```

Three different entry points, three different values, and one of them
(`ENGINE_CONTEXT`, the `AsyncLLM` default) is not a key in the dict — so it silently gets
2048 instead of 8192. No warning, no error.

**Generalize this.** When tracking down a defaults discrepancy, grep for these patterns
specifically:

```bash
grep -rn "\.get(.*, *[A-Z_]*DEFAULT" $VLLM/ --include=*.py | head -20   # silent fallthrough
grep -rn "if .* is None:" $VLLM/config/*.py | head -20                  # conditional defaults
grep -rn "__post_init__" $VLLM/config/*.py                              # derived/validated config
```

Also: always find *where the default is defined*, not just where it is consumed. The
consumer says `DEFAULT_MAX_NUM_BATCHED_TOKENS`; only the definition tells you it is 2048.

```bash
grep -rn "DEFAULT_MAX_NUM_BATCHED_TOKENS\s*=" $VLLM/      # ← returns NOTHING
```

Another zero-hit trap, for a different reason. The definition is:

```python
DEFAULT_MAX_NUM_BATCHED_TOKENS: ClassVar[int] = 2048     # config/scheduler.py:42
DEFAULT_MAX_NUM_SEQS: ClassVar[int] = 128                # config/scheduler.py:44
```

`NAME\s*=` assumes the `=` directly follows the name, but a **type annotation sits between
them**. In modern typed Python this is the norm, not the exception. Allow for it:

```bash
grep -rn "DEFAULT_MAX_NUM_BATCHED_TOKENS[^_]*=\s*[0-9]" $VLLM/ --include=*.py
```

The `[^_]*` skips the annotation while the `[^_]` guards against also matching
`DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP`, a real neighbouring constant with a
different value (256). Prefix collisions between constants are common; check for them
whenever a name is a prefix of another name.

### Step 5: Bound the blast radius

Before proposing a fix, establish what else it touches. This is what makes a
recommendation trustworthy rather than a hopeful one-liner.

```bash
grep -n "usage_context" $VLLM/engine/arg_utils.py
```

Seven hits — but the count is not the answer, the *classification* is. Read each one and
bucket it:

```
1820:  usage_context: UsageContext | None = None,     # signature
2136:  usage_context,                                  # → _set_default_max_num_seqs_and_batched_tokens_args
2592:  usage_context: UsageContext | None,             # that function's signature
2612:  usage_context,                                  # → get_batch_defaults
2618:  usage_context,                                  # → get_batch_defaults
2668:  usage_context.value if usage_context else None, # logging only
2678:  usage_context.value if usage_context else None, # logging only
```

Two signatures, three hops along a single call chain, two log statements. There is
**exactly one call site in `create_engine_config`** (2136), and it leads only into
scheduler-default selection. Nothing branches on `usage_context` for engine construction,
memory planning, or model loading.

That is what converts "change `usage_context` and hope" into "this changes two numbers and
nothing else" — a claim defensible in review.

Do this for every proposed change. The useful output is not the hit count but the answer
to "how many *independent* things does this feed?" Seven hits along one chain is safer
than two hits in two unrelated subsystems.

### Step 6: Write a probe

Reading gives a hypothesis. A probe gives a fact. Probes should be small, printed, and
comparative.

The most valuable probe in this investigation was ten lines and needed no GPU and no model
weights — just the processor:

```python
from transformers.models.gemma4.image_processing_gemma4 import (
    get_aspect_ratio_preserving_size as g)

for mst in (280, 560, 1120):
    mp = mst * 9
    print(f"--- max_soft_tokens={mst}, budget={mp*256:,} px ---")
    for (w, h) in [(4096,3072),(1024,768),(512,384),(224,224)]:
        th, tw = g(height=h, width=w, patch_size=16,
                   max_patches=mp, pooling_kernel_size=3)
        print(f"  {w}x{h} -> {tw}x{th}  scale={tw/w:.3f}")
```

Three lessons in that snippet:

- **Import the leaf function, not the framework.** `get_aspect_ratio_preserving_size` is a
  pure function of five integers. No model, no GPU, no engine startup. Runs in a second.
  Always look for the pure-function core of the behaviour you're testing.
- **Print a table across a range, not one value.** One input tells you a number; a range
  reveals the *shape*. The `scale > 1` entries — proving small images get **upscaled** —
  were the finding that killed the "HF uses native resolution" assumption, and a single
  4096×3072 test case would have missed it entirely.
- **Print the derived quantities too** (`scale`, budget in pixels), not just raw outputs.
  The insight is usually in the ratio, not the number.

### Step 7: Differential-test the edges

When comparing two implementations, do not test the happy path — it agrees by
construction, which is why the bug survived. Construct inputs that target the *specific
mechanism* where they might differ.

The decode comparison tested seven cases, chosen deliberately:

| case | what it probes |
|---|---|
| RGBA fully transparent | alpha handling at the extreme |
| RGBA 50% alpha | whether real compositing happens |
| palette + `tRNS` | transparency that is *not* mode RGBA |
| EXIF orientation=6 | metadata-driven transforms |
| 16-bit PNG | bit-depth downconversion |
| grayscale `L` | channel expansion |
| CMYK JPEG | colour-space conversion |

Four came back identical. That is a *good* outcome — it is what let the doc state the
divergence is "alpha-handling, and only alpha-handling" rather than vaguely gesturing at
"image loading differences."

The palette + `tRNS` case is the one worth studying. It came from reading vLLM's helper:

```python
def _has_transparency(image):
    return image.mode in ("RGBA","LA","PA") or "transparency" in getattr(image, "info", {})
```

That `or` clause is doing real work — it catches transparency that a naive
`mode == "RGBA"` check misses. Seeing defensive code in a library is a signal: **it is
there because that case occurs in the wild.** Turn every such branch into a test case.
That is how you find edges you would never have imagined.

### Step 8: Rule things out, in writing

Each "ruled out" entry needs the *evidence*, not just the verdict, or it will be
re-litigated. Compare:

- ✗ "Chat template is fine."
- ✓ "Rendered both the offline `{"type":"image","image":PIL}` form and the server's parsed
  `{"type":"image"}` form through the model's real `chat_template.jinja` with two images;
  output byte-identical including placeholder count. Template branches only on
  `content.type`."

The second survives scrutiny and stops the next person from redoing it.

---

## Part 3 — Techniques worth internalizing

**Grep the format string, not the formatted output.** `"Detected the chat template content
format"` matches; `"format to be 'openai'"` may not, because the value is interpolated.
Pick the longest literal run.

**`grep -rn` then `sed -n 'A,Bp'`.** Locate with grep, read with sed. Reading a 3000-line
file top to bottom is almost never the right move:

```bash
grep -n "def create_engine_config" $VLLM/engine/arg_utils.py    # → 2340
sed -n '2340,2420p' $VLLM/engine/arg_utils.py
```

**Grep for many symbols at once** when orienting in an unfamiliar file:

```bash
grep -n "max_soft_tokens\|pooling_kernel\|patch_size\|mm_processor_kwargs" \
  $VLLM/model_executor/models/gemma4_mm.py | head -60
```

The line-number distribution alone shows you the file's structure before you read a word
of it.

**Find the model-specific hook.** Most ML libraries have a per-architecture patch registry.
Knowing it exists is half the battle:

```bash
grep -rn "MODELS_CONFIG_MAP" $VLLM/model_executor/models/config.py
```

Then check whether *your* architecture is in it. Anything there runs for your model and no
other, which makes it a prime suspect for "why does this model behave oddly."

**Read config JSON with `python -c`, not `cat`.** Targeted and readable:

```bash
python -c "import json;print(json.dumps(json.load(open('$SNAP/config.json'))['vision_config'],indent=2))"
```

**When you can't import the library, transcribe it.** `import vllm` fails outside a
CUDA-capable environment:

```
ImportError: libcudart.so.13: cannot open shared object file
```

Rather than fighting `LD_LIBRARY_PATH`, the decode probe reimplemented
`ImageMediaIO.load_bytes` verbatim from the source. This is legitimate *if* you copy
exactly and say so — the doc labels it "faithful reimplementation" and tells the reader to
diff it against the original. Transcribing 20 lines of pure logic is fine; transcribing
anything with hidden state is not.

**Check absences, not just presences.** No `preprocessor_config.json` meant resize config
lived elsewhere. No `video: 0` in `limit_mm_per_prompt` meant the video token floor still
applied. Missing keys are findings.

**Let the maintainers' comments lead you.** From `gemma4_mm.py`:

```python
# mm_processor_kwargs from the model config on every call via:
# If we strip max_soft_tokens from incoming, the re-merge puts ...
```

Someone hit a bug there and left a note. Comments explaining *why* mark the places where
behaviour is surprising — exactly where you should be looking.

---

## Part 4 — The wrong turns

Recorded because the recovery is the transferable part.

**1. `find` on network storage.** Searching `/fsxvision_new` and `/projects` for model
files blew past a 120-second timeout. *Fix:* check `HF_HOME` / `TRANSFORMERS_CACHE` first.
*Lesson:* on distributed filesystems, prefer an authoritative index (env var, config,
package metadata) over a filesystem scan, always.

**2. Guessing file paths.** `entrypoints/openai/protocol.py` did not exist; the real path
was `entrypoints/openai/chat_completion/protocol.py`. Same for
`multimodal/processing.py`, actually `multimodal/processing/context.py`. *Fix:* `ls` the
directory instead of guessing:

```bash
ls $VLLM/entrypoints/openai/
```

*Lesson:* one `ls` beats three failed `Read`s. Libraries refactor modules into packages
constantly.

**3. Stopping at the first plausible cause.** The first pass concluded the problem was
sampling-parameter merging — which was real, but minor. Only the empirical clue about
`num_batched_tokens` (8192 online vs ~2048 offline) forced discovery of the
`UsageContext` mechanism, which was the dominant effect. *Lesson:* a plausible cause that
does not *quantitatively* account for the observed symptom is an incomplete cause. Ask:
"would this fully explain what I saw?" A sampling-params difference does not explain a 4×
scheduler difference.

**4. Trusting a partial fix.** Forcing `max_num_batched_tokens=8192` manually improved
quality but stayed slow. That residual was itself a clue — it pointed at `max_num_seqs`
still sitting at 128 against serve's 1024. *Lesson:* when a fix helps but doesn't close
the gap, the remaining gap is data. Don't declare victory; ask what the fix *didn't*
cover.

**5. Sloppy heredoc under time pressure.** A probe script was written with
`import piexif if False else None` — not valid Python — and again with
`import imageio.v2 as iio if False else None`. Two wasted cycles on `SyntaxError`.
*Lesson:* write the probe cleanly the first time. A probe you have to debug is a probe
that isn't testing what you think it is.

**6. Two zero-hit greps, both self-inflicted.** While writing *this* document, the two
example commands in [Step 1](#step-1-turn-the-symptom-into-a-search-string) and
[Step 4](#step-4-read-defaults-like-an-adversary) were written from memory and both
returned nothing when finally run: one straddled a `%s` substitution, the other assumed
`NAME =` where the source has `NAME: ClassVar[int] =`. Both are now written up as the
lessons they are. *Lesson:* a grep you *remember* working is not a grep that works — and
if you are about to hand a command to someone else, run it first. More importantly: a
zero-hit grep is almost never proof of absence. It is usually proof your pattern is wrong.

**7. Overstating a blast radius.** An earlier draft claimed `usage_context` was
"referenced exactly once" in `arg_utils.py`. It appears seven times. The *conclusion* was
still right — one call site, one call chain, two log statements — but the stated evidence
was wrong, which is worse than a wrong conclusion because it is harder to catch.
*Lesson:* if you claim a count, run the count.

**8. Overstating a finding.** The first doc implied EXIF orientation was an
online/offline divergence. Testing showed OpenCV applies EXIF too — no divergence. *Fix:*
[`vllm_image_pipeline_parity.md` §4.2](./vllm_image_pipeline_parity.md#42-a-correction-to-the-previous-doc)
corrects it explicitly. *Lesson:* claims derived from reading one side only are half-tested.
The fix is cheap — run the *other* side too, before writing it down.

---

## Part 5 — Command cookbook

```bash
# ─── Orientation ────────────────────────────────────────────────────────────
VENV=/path/to/venv
$VENV/bin/python -c "import LIB, os; print(LIB.__version__, os.path.dirname(LIB.__file__))"
LIB=$VENV/lib/python3.12/site-packages/LIB
env | grep -i "^HF_\|^TRANSFORMERS_\|CACHE"          # where are model files?
ls $LIB/                                             # top-level structure
ls $LIB/subpkg/                                      # before guessing a path

# ─── Symptom → code ─────────────────────────────────────────────────────────
# Longest LITERAL run only — stop before any filename/number/enum (it's a %s).
grep -rn "sampling parameters have been overridden" $LIB/ --include=*.py
grep -rn "config_key_name" $LIB/ --include=*.py | grep -v test | head -30
grep -rn "\-\-cli-flag-name" $LIB/ --include=*.py | head

# ─── Defaults hunting ───────────────────────────────────────────────────────
# NOTE: "NAME\s*=" MISSES "NAME: ClassVar[int] = 2048". Allow the annotation.
grep -rn "SOME_DEFAULT_CONST[^_]*=\s*[0-9]" $LIB/ --include=*.py   # where defined
grep -rn "\.get(.*, *[A-Z_]*DEFAULT" $LIB/ --include=*.py          # silent fallthrough
grep -rn "__post_init__" $LIB/config/*.py                          # derived config
grep -rn "if .* is None:" $LIB/config/*.py | head -20              # conditional defaults

# ─── Read precisely ─────────────────────────────────────────────────────────
grep -n "def target_function" $LIB/module.py               # → LINE
sed -n 'LINE,LINE+60p' $LIB/module.py
grep -n "symA\|symB\|symC" $LIB/module.py | head -60       # structure at a glance

# ─── Blast radius ───────────────────────────────────────────────────────────
grep -rn "the_parameter" $LIB/ --include=*.py | grep -v test   # all uses

# ─── Model-specific hooks ───────────────────────────────────────────────────
grep -rn "MODELS_CONFIG_MAP\|ARCH_TO\|_REGISTRY" $LIB/ | head

# ─── Config files ───────────────────────────────────────────────────────────
ls  $SNAP/                                                  # absences matter
python -c "import json;print(json.dumps(json.load(open('$SNAP/config.json'))['sub'],indent=2))"
```

---

## Part 6 — Applying this elsewhere

The playbook is library-agnostic. To reuse it:

**For a different model on vLLM.** Steps 0–3 are unchanged. The model-specific parts are:
`$VLLM/model_executor/models/<arch>.py` (per-model processing info, token accounting), the
`MODELS_CONFIG_MAP` entry (per-architecture config patching), and the HF
`image_processing_<arch>.py` / `processing_<arch>.py`. Ask the same four questions:

1. Does the processor resize, and against what budget?
2. Is there a `_SUPPORTED_*` tuple exposing a quality/cost knob?
3. Does the model override tokenization defaults (`add_special_tokens`, special tokens)?
4. Is the architecture in the config-patch registry, and what does the patch change?

**For a different vLLM feature.** The junction technique generalizes to any "path A vs path
B" question: LoRA vs base, speculative decoding on vs off, chunked prefill on vs off, TP=1
vs TP=4. Map both, find where they meet, enumerate the inputs to the junction.

**For a completely different library.** The transferable core is four moves:

1. Locate installed source and pin the version.
2. Grep a symptom string to get an entry point.
3. Find where the paths converge to bound the search space.
4. Probe the pure-function core to convert reading into evidence.

Everything else is grep syntax.

**A note on when to stop.** This investigation stopped at "three divergences, each with a
mechanism and a proposed fix," and explicitly did *not* apply them. That was right: the
findings were worth more verified than applied fast, and one of them
([`vllm_image_pipeline_parity.md` §8](./vllm_image_pipeline_parity.md#8-blockers-in-the-current-config))
is that the image path has never actually been run in this repo — so static analysis has a
hard ceiling here. Know which of your conclusions are proven by execution and which are
proven by reading, and label them differently. The parity docs do this deliberately:
"measured" and "verified identical (tested)" mean something stronger than "per source."
