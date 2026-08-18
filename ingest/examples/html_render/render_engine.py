"""
HTML -> PNG render engine, ported verbatim from Digital-Twin-Pipeline's
"SMART" render path (render_new_fitted_tight.py, via
digital_twin_stage_weaver/project_specific/project_specific_render.py) rather
than reimplemented: CDP-driven viewport + screenshot, a JS TEXT AUTO-FIT pass
(grow the overflowing box -> grow the page -> shrink the font, in that order —
never silently clips text) and a CONTENT-TIGHT CROP pass (off by default, see
ENABLE_TIGHT_CROP below) that trims trailing whitespace the page-grow step may
introduce, without ever cutting real content.

Trimmed from the source relative to project_specific_render.py: the
chandra-detection width/height lookup, the DB-polling ThreadPoolExecutor loop,
and the JS_TREE_WITH_BBOX layout-bbox export — all specific to DTP's own
OCR-grounding pipeline, not to HTML rendering itself. `worker.py` supplies
width/height per call instead, and drives this module through
the pool's one-process-per-worker model.
"""
import base64
import io
import os
import re
import tempfile
from pathlib import Path

# Headless Chrome needs shared libs (libatk-1.0.so.0 etc.) not present in a
# bare python venv — same LD_LIBRARY_PATH trick DTP's own renderer uses. Must
# run at import time: under multiprocessing's 'spawn' start method each worker
# re-imports this module before model_init() ever launches Chrome.
_CHROME_DEPS_LIB = (
    "/fsxvision_new/aryanjain.intern/Patram-Data-Engine/Digital-Twin-Pipeline/"
    "chrome-deps/extracted/usr/lib/x86_64-linux-gnu"
)
if os.path.isdir(_CHROME_DEPS_LIB) and _CHROME_DEPS_LIB not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = _CHROME_DEPS_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")

DATA_URI_BYTE_LIMIT = 0  # always file:// — avoids Chrome's large-data-URI tab crashes
FIT_TOLERANCE_PX = 1.0
FIT_PAD_PX = 2.0
NEIGHBOUR_GAP_PX = 2.0
# "Never crop, never shrink the font" setup: GROW THE BOX / GROW THE PAGE
# (Levers 1-2) only ever ADD room, so they're safe to keep on unconditionally
# — nothing is ever visually cut by growing. What's disabled/tuned instead:
#   - MAX_PAGE_GROW_FACTOR raised way past DTP's default 2.5x so the page can
#     basically always grow enough to fit overflowing text, meaning Lever 3
#     (SHRINK THE FONT, last resort) almost never actually fires.
#   - ENABLE_TIGHT_CROP = False — the crop pass only ever shrinks the canvas
#     after the fact, which is the one thing that previously cut real text off
#     (mismeasured "ink" extent on a flex layout). Off entirely by default.
MAX_PAGE_GROW_FACTOR = 50.0
MAX_DIM = 12000  # absolute safety ceiling (px) — effectively unreachable in practice
MIN_FONT_PX = 6.0
ENABLE_TEXT_AUTOFIT = True
ENABLE_TIGHT_CROP = False
TIGHT_CROP_PAD_PX = 5.0

# ---------------------------------------------------------------------------
# JS — verbatim from Digital-Twin-Pipeline/render_new_fitted_tight.py
# ---------------------------------------------------------------------------

JS_WAIT_STABLE = """
return await new Promise((resolve) => {
    const timeout = arguments[0];
    const deadline = Date.now() + timeout;

    if (document.readyState !== 'complete') {
        window.addEventListener('load', check, { once: true });
    } else {
        check();
    }

    function check() {
        let timer = null;
        const obs = new MutationObserver(() => {
            clearTimeout(timer);
            timer = setTimeout(done, 200);
        });
        obs.observe(document.body, { childList: true, subtree: true, attributes: true });
        timer = setTimeout(done, 300);

        function done() {
            obs.disconnect();
            resolve({ stable: true, elapsed: Date.now() - (deadline - timeout) });
        }

        if (Date.now() >= deadline) {
            obs.disconnect();
            resolve({ stable: false, elapsed: timeout });
        }
    }
});
"""

JS_FIT_TEXT = r"""
const opts = arguments[0] || {};
const TOL          = (typeof opts.tol === 'number') ? opts.tol : 1.0;
const PAD          = (typeof opts.pad === 'number') ? opts.pad : 2.0;
const GAP          = (typeof opts.gap === 'number') ? opts.gap : 2.0;
const MAX_PG_GROW  = (typeof opts.maxPageGrow === 'number') ? opts.maxPageGrow : 2.5;
const MIN_FONT     = (typeof opts.minFont === 'number') ? opts.minFont : 6.0;

let scanned = 0, grewBox = 0, grewLeft = 0, shrunk = 0, floored = 0;

const pageEl = document.querySelector('.document-page') || document.body;
const pageRect0 = pageEl.getBoundingClientRect();
const pageOrigW = pageRect0.width;
const pageOrigH = pageRect0.height;
const pageMaxW = pageOrigW * MAX_PG_GROW;
const pageMaxH = pageOrigH * MAX_PG_GROW;

function hasText(el) {
    const t = (el.innerText || el.textContent || '');
    return t.trim().length > 0;
}
function overflowAmount(el) {
    const dx = el.scrollWidth  - el.clientWidth;
    const dy = el.scrollHeight - el.clientHeight;
    return { dx, dy, over: (dx > TOL || dy > TOL) };
}
function isLeafTextBox(el) {
    if (!hasText(el)) return false;
    for (const c of el.children) { if (hasText(c)) return false; }
    const r = el.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) return false;
    return true;
}

const all = Array.from(document.body.querySelectorAll('*'));
const leaves = all.filter(isLeafTextBox);
const allBoxes = leaves.map(el => ({ el, r: el.getBoundingClientRect() }));

function alignOf(el) {
    const a = (window.getComputedStyle(el).textAlign || '').toLowerCase();
    if (a === 'right' || a === 'end') return 'right';
    if (a === 'center') return 'center';
    return 'left';
}

function freeRight(b) {
    const r = b.r;
    let limit = pageRect0.right;
    for (const o of allBoxes) {
        if (o.el === b.el) continue;
        const orr = o.r;
        if (orr.bottom <= r.top + 1 || orr.top >= r.bottom - 1) continue;
        if (orr.left >= r.right - 1) {
            limit = Math.min(limit, orr.left - GAP);
        }
    }
    return Math.max(0, limit - r.right);
}
function freeLeft(b) {
    const r = b.r;
    let limit = pageRect0.left;
    for (const o of allBoxes) {
        if (o.el === b.el) continue;
        const orr = o.r;
        if (orr.bottom <= r.top + 1 || orr.top >= r.bottom - 1) continue;
        if (orr.right <= r.left + 1) {
            limit = Math.max(limit, orr.right + GAP);
        }
    }
    return Math.max(0, r.left - limit);
}
function freeDown(b) {
    const r = b.r;
    let limit = pageRect0.bottom;
    for (const o of allBoxes) {
        if (o.el === b.el) continue;
        const orr = o.r;
        if (orr.right <= r.left + 1 || orr.left >= r.right - 1) continue;
        if (orr.top >= r.bottom - 1) {
            limit = Math.min(limit, orr.top - GAP);
        }
    }
    return Math.max(0, limit - r.bottom);
}

let pageNeedW = pageOrigW;
let pageNeedH = pageOrigH;

function setLeft(el, px)  { el.style.setProperty('left',  px + 'px', 'important'); }
function setWidth(el, px) { el.style.setProperty('width', px + 'px', 'important'); }
function setHeight(el, px){ el.style.setProperty('height', px + 'px', 'important'); }

for (const b of allBoxes) {
    const el = b.el;
    scanned++;
    let info = overflowAmount(el);
    if (!info.over) continue;

    const cs0 = window.getComputedStyle(el);
    const align = alignOf(el);
    const startRect = el.getBoundingClientRect();

    let didGrow = false;
    if (info.dx > TOL) {
        const needX = info.dx + 2 * PAD;
        if (align === 'right') {
            const room = freeLeft(b);
            const take = Math.min(needX, room);
            if (take > 0.5) {
                const curLeft  = parseFloat(cs0.left)  || startRect.left;
                const curWidth = parseFloat(cs0.width) || startRect.width;
                setLeft(el,  curLeft - take);
                setWidth(el, curWidth + take);
                grewLeft++;
                didGrow = true;
            }
            info = overflowAmount(el);
            if (info.dx > TOL) {
                const room2 = freeRight(b);
                const take2 = Math.min(info.dx + 2 * PAD, room2);
                if (take2 > 0.5) {
                    const cw = parseFloat(window.getComputedStyle(el).width) || el.clientWidth;
                    setWidth(el, cw + take2);
                    didGrow = true;
                }
            }
        } else {
            const room = freeRight(b);
            const take = Math.min(needX, room);
            if (take > 0.5) {
                const curWidth = parseFloat(cs0.width) || startRect.width;
                setWidth(el, curWidth + take);
                didGrow = true;
            }
        }
    }
    info = overflowAmount(el);
    if (info.dy > TOL) {
        const room = freeDown(b);
        const take = Math.min(info.dy + 2 * PAD, room);
        if (take > 0.5) {
            const cs1 = window.getComputedStyle(el);
            const curHeight = parseFloat(cs1.height) || el.getBoundingClientRect().height;
            setHeight(el, curHeight + take);
            didGrow = true;
        }
    }
    if (didGrow) grewBox++;
    info = overflowAmount(el);

    if (info.over) {
        const r = el.getBoundingClientRect();
        if (info.dx > TOL) {
            const wantRight = r.left - pageRect0.left + el.scrollWidth + 2 * PAD;
            pageNeedW = Math.max(pageNeedW, Math.min(pageMaxW, wantRight));
        }
        if (info.dy > TOL) {
            const wantBottom = r.top - pageRect0.top + el.scrollHeight + 2 * PAD;
            pageNeedH = Math.max(pageNeedH, Math.min(pageMaxH, wantBottom));
        }
    }
}

const grewPage = (pageNeedW > pageOrigW + TOL) || (pageNeedH > pageOrigH + TOL);
if (grewPage) {
    if (pageNeedW > pageOrigW + TOL) {
        pageEl.style.setProperty('width', Math.ceil(pageNeedW) + 'px', 'important');
    }
    if (pageNeedH > pageOrigH + TOL) {
        pageEl.style.setProperty('height', Math.ceil(pageNeedH) + 'px', 'important');
    }
    pageEl.style.setProperty('overflow', 'hidden', 'important');
}

const pageRect1 = pageEl.getBoundingClientRect();
for (const b of allBoxes) {
    const el = b.el;
    let info = overflowAmount(el);
    if (!info.over) continue;

    const cs = window.getComputedStyle(el);
    const align = alignOf(el);
    const r = el.getBoundingClientRect();

    if (info.dx > TOL) {
        const roomToPageEdge = pageRect1.right - r.right;
        const take = Math.min(info.dx + 2 * PAD, Math.max(0, roomToPageEdge));
        if (take > 0.5) {
            const cw = parseFloat(cs.width) || el.clientWidth;
            el.style.setProperty('width', (cw + take) + 'px', 'important');
        }
    }
    info = overflowAmount(el);
    if (info.dy > TOL) {
        const roomToPageBottom = pageRect1.bottom - el.getBoundingClientRect().bottom;
        const take = Math.min(info.dy + 2 * PAD, Math.max(0, roomToPageBottom));
        if (take > 0.5) {
            const ch = parseFloat(window.getComputedStyle(el).height) || el.getBoundingClientRect().height;
            el.style.setProperty('height', (ch + take) + 'px', 'important');
        }
    }
    info = overflowAmount(el);

    if (info.over) {
        let fontPx = parseFloat(window.getComputedStyle(el).fontSize) || 16;
        const origFontPx = fontPx;
        let guard = 0;
        while (info.over && fontPx > MIN_FONT && guard < 60) {
            guard++;
            const wRatio = el.clientWidth  > 0 ? el.scrollWidth  / el.clientWidth  : 1;
            const hRatio = el.clientHeight > 0 ? el.scrollHeight / el.clientHeight : 1;
            const ratio  = Math.max(wRatio, hRatio, 1.02);
            let next = fontPx / Math.min(ratio, 1.5);
            if (next >= fontPx) next = fontPx - 0.5;
            if (next < MIN_FONT) next = MIN_FONT;
            fontPx = next;
            el.style.setProperty('font-size', fontPx + 'px', 'important');
            info = overflowAmount(el);
        }
        guard = 0;
        while (!info.over && fontPx < origFontPx && guard < 60) {
            guard++;
            const trial = Math.min(origFontPx, fontPx + 0.5);
            el.style.setProperty('font-size', trial + 'px', 'important');
            const ti = overflowAmount(el);
            if (ti.over) { el.style.setProperty('font-size', fontPx + 'px', 'important'); break; }
            fontPx = trial; info = ti;
        }
        if (fontPx < origFontPx - 0.01) shrunk++;

        info = overflowAmount(el);
        if (info.over) {
            el.style.setProperty('white-space', 'normal', 'important');
            el.style.setProperty('overflow-wrap', 'anywhere', 'important');
            el.style.setProperty('word-break', 'break-word', 'important');
            info = overflowAmount(el);
        }
        if (info.over) {
            floored++;
            el.style.setProperty('overflow', 'visible', 'important');
        }
    }
}

const pageRectF = pageEl.getBoundingClientRect();
return {
    scanned, grewBox, grewLeft, shrunk, floored,
    pageOrigW: Math.round(pageOrigW), pageOrigH: Math.round(pageOrigH),
    pageNeedW: Math.ceil(pageRectF.width), pageNeedH: Math.ceil(pageRectF.height),
    grewPage
};
"""

JS_CONTENT_EXTENT = r"""
const opts = arguments[0] || {};
const PAD = (typeof opts.pad === 'number') ? opts.pad : 2.0;

const pageEl = document.querySelector('.document-page') || document.body;
const pageRect = pageEl.getBoundingClientRect();

const marginL = Math.max(0, pageRect.left);
const marginT = Math.max(0, pageRect.top);

function isVisible(el) {
    const s = window.getComputedStyle(el);
    if (!s) return false;
    if (s.display === 'none' || s.visibility === 'hidden' || s.opacity === '0') return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
}
function paintsInk(el) {
    const tag = el.tagName.toLowerCase();
    if (tag === 'img' || tag === 'hr' || tag === 'svg' || tag === 'canvas' ||
        tag === 'input' || tag === 'table' || tag === 'td' || tag === 'th') return true;
    for (const n of el.childNodes) {
        if (n.nodeType === 3 && n.textContent.trim().length > 0) return true;
    }
    const s = window.getComputedStyle(el);
    if (s.backgroundImage && s.backgroundImage !== 'none') return true;
    const bw = ['borderTopWidth','borderRightWidth','borderBottomWidth','borderLeftWidth']
        .map(k => parseFloat(s[k]) || 0);
    if (bw.some(w => w > 0) && s.borderStyle !== 'none') {
        if (el !== pageEl) return true;
    }
    return false;
}

let contentRight = 0, contentBottom = 0;
const all = document.body.querySelectorAll('*');
for (const el of all) {
    if (!isVisible(el)) continue;
    if (!paintsInk(el)) continue;
    const r = el.getBoundingClientRect();
    const right  = Math.max(r.right,  r.left + el.scrollWidth);
    const bottom = Math.max(r.bottom, r.top  + el.scrollHeight);
    if (right  > contentRight)  contentRight  = right;
    if (bottom > contentBottom) contentBottom = bottom;
}

if (contentRight  <= 0) contentRight  = pageRect.right;
if (contentBottom <= 0) contentBottom = pageRect.bottom;

let cropW = Math.ceil(contentRight  + marginL + PAD);
let cropH = Math.ceil(contentBottom + marginT + PAD);
cropW = Math.min(cropW, Math.ceil(pageRect.right));
cropH = Math.min(cropH, Math.ceil(pageRect.bottom));

return {
    cropW, cropH,
    contentRight: Math.ceil(contentRight),
    contentBottom: Math.ceil(contentBottom),
    pageW: Math.ceil(pageRect.width),
    pageH: Math.ceil(pageRect.height),
    marginL: Math.round(marginL),
    marginT: Math.round(marginT)
};
"""


def extract_html(text: str) -> str:
    m = re.search(r"```html(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _inject_smart_scaling(html: str, zoom: float) -> str:
    css = f"""
<style>
html, body {{
    margin: 0 !important;
    padding: 0 !important;
    background-color: white;
}}
body {{
    position: relative;
    zoom: {zoom:.4f};
    transform-origin: 0 0;
}}
</style>"""
    if "</head>" in html:
        return html.replace("</head>", f"{css}</head>")
    elif "<body>" in html:
        return html.replace("<body>", f"<body>{css}")
    return css + html


def _cdp_set_viewport(driver, width: int, height: int, dpr: int = 1):
    driver.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {
        "width": int(width),
        "height": int(height),
        "deviceScaleFactor": int(dpr),
        "mobile": False,
    })


def _cdp_capture_viewport_png(driver, width: int, height: int) -> bytes:
    res = driver.execute_cdp_cmd("Page.captureScreenshot", {
        "format": "png",
        "fromSurface": True,
        "clip": {"x": 0, "y": 0, "width": int(width), "height": int(height), "scale": 1},
    })
    return base64.b64decode(res["data"])


def _load_html_in_driver(driver, html: str, tmp_dir: Path):
    html_bytes = html.encode("utf-8")
    if len(html_bytes) <= DATA_URI_BYTE_LIMIT:
        b64 = base64.b64encode(html_bytes).decode("utf-8")
        driver.get(f"data:text/html;charset=utf-8;base64,{b64}")
    else:
        tmp_file = tmp_dir / "page.html"
        tmp_file.write_bytes(html_bytes)
        driver.get(tmp_file.as_uri())


def _wait_for_dom_stable(driver, timeout_ms: int = 5000):
    try:
        driver.execute_script(JS_WAIT_STABLE, timeout_ms)
    except Exception:
        pass


def _fit_text_in_driver(driver) -> dict:
    try:
        diag = driver.execute_script(JS_FIT_TEXT, {
            "tol": FIT_TOLERANCE_PX, "pad": FIT_PAD_PX, "gap": NEIGHBOUR_GAP_PX,
            "maxPageGrow": MAX_PAGE_GROW_FACTOR, "minFont": MIN_FONT_PX,
        })
        return diag or {}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _measure_content_extent(driver) -> dict:
    try:
        info = driver.execute_script(JS_CONTENT_EXTENT, {"pad": TIGHT_CROP_PAD_PX})
        return info or {}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def new_driver(chromedriver: str, chrome_binary: str):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--autoplay-policy=no-user-gesture-required")
    options.add_argument("--no-zygote")
    options.add_argument("--disable-renderer-backgrounding")
    options.add_argument("--disable-background-timer-throttling")
    options.add_argument("--disable-ipc-flooding-protection")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--hide-scrollbars")
    if chrome_binary:
        options.binary_location = chrome_binary
    service = Service(executable_path=chromedriver) if chromedriver else Service()
    return webdriver.Chrome(service=service, options=options)


def render_html(html_original: str, out_png: Path, width: int, height: int,
                 driver, tmp_dir: Path, dom_stable_timeout_ms: int = 5000) -> dict:
    """Ports render_new_fitted_tight.render_html's SMART-scaling path
    (use_smart_scaling=True): render at the source canvas size, TEXT AUTO-FIT
    any overflowing box (grow box -> grow page -> shrink font), then
    CONTENT-TIGHT CROP trailing whitespace (if enabled) before the final
    screenshot.
    """
    MEASURE_VIEWPORT = max(int(width), 2048)
    _cdp_set_viewport(driver, MEASURE_VIEWPORT, MEASURE_VIEWPORT, dpr=1)
    _load_html_in_driver(driver, html_original, tmp_dir)
    driver.execute_script("""
        window.scrollTo(0,0);
        document.documentElement.style.margin='0';
        document.documentElement.style.padding='0';
        document.body.style.margin='0';
        document.body.style.padding='0';
        document.body.style.position='relative';
        document.documentElement.style.backgroundColor='white';
        document.body.style.backgroundColor='white';
    """)
    _wait_for_dom_stable(driver, dom_stable_timeout_ms)

    zoom = 1.0
    render_width = min(width, MAX_DIM)
    render_height = min(height, MAX_DIM)
    html_to_render = _inject_smart_scaling(html_original, zoom)

    _cdp_set_viewport(driver, render_width, render_height, dpr=1)
    _load_html_in_driver(driver, html_to_render, tmp_dir)
    driver.execute_script("""
        window.scrollTo(0,0);
        document.documentElement.style.margin='0';
        document.documentElement.style.padding='0';
        document.body.style.margin='0';
        document.body.style.padding='0';
        document.documentElement.style.backgroundColor='white';
        document.body.style.backgroundColor='white';
        document.body.style.overflow='visible';
    """)
    _wait_for_dom_stable(driver, dom_stable_timeout_ms)

    fit_diag = _fit_text_in_driver(driver) if ENABLE_TEXT_AUTOFIT else {}
    _wait_for_dom_stable(driver, min(dom_stable_timeout_ms, 1500))

    if isinstance(fit_diag, dict) and fit_diag.get("grewPage"):
        need_w = int(fit_diag.get("pageNeedW", render_width))
        need_h = int(fit_diag.get("pageNeedH", render_height))
        new_w = min(max(render_width, need_w), MAX_DIM)
        new_h = min(max(render_height, need_h), MAX_DIM)
        if new_w != render_width or new_h != render_height:
            render_width, render_height = new_w, new_h
            _cdp_set_viewport(driver, render_width, render_height, dpr=1)
            driver.execute_script("""
                document.documentElement.style.backgroundColor='white';
                document.body.style.backgroundColor='white';
                window.scrollTo(0,0);
            """)
            _wait_for_dom_stable(driver, min(dom_stable_timeout_ms, 1500))

    # Container-level safety net (not in DTP's original engine): DTP's TEXT
    # AUTO-FIT only checks PER-LEAF-BOX overflow (scrollWidth vs clientWidth on
    # each individual absolutely-positioned div) — correct for DTP's own HTML
    # convention. A flexbox multi-column layout can spill past the render
    # canvas without any single leaf box reporting overflow, so compare the
    # whole document's natural size against the current canvas too.
    natural = driver.execute_script(
        "return {w: document.documentElement.scrollWidth, h: document.documentElement.scrollHeight};"
    )
    nat_w, nat_h = int(natural.get("w", render_width)), int(natural.get("h", render_height))
    if nat_w > render_width or nat_h > render_height:
        new_w = min(max(render_width, nat_w), MAX_DIM)
        new_h = min(max(render_height, nat_h), MAX_DIM)
        if new_w != render_width or new_h != render_height:
            render_width, render_height = new_w, new_h
            _cdp_set_viewport(driver, render_width, render_height, dpr=1)
            driver.execute_script("""
                document.documentElement.style.backgroundColor='white';
                document.body.style.backgroundColor='white';
                window.scrollTo(0,0);
            """)
            _wait_for_dom_stable(driver, min(dom_stable_timeout_ms, 1500))

    crop_diag = {}
    if ENABLE_TIGHT_CROP:
        crop_diag = _measure_content_extent(driver)
        if isinstance(crop_diag, dict) and not crop_diag.get("error"):
            crop_w = max(1, min(int(crop_diag.get("cropW", render_width)), render_width))
            crop_h = max(1, min(int(crop_diag.get("cropH", render_height)), render_height))
            if crop_w < render_width or crop_h < render_height:
                render_width, render_height = crop_w, crop_h
                _cdp_set_viewport(driver, render_width, render_height, dpr=1)
                driver.execute_script("""
                    document.documentElement.style.backgroundColor='white';
                    document.body.style.backgroundColor='white';
                    window.scrollTo(0,0);
                """)
                _wait_for_dom_stable(driver, min(dom_stable_timeout_ms, 1500))

    png_bytes = _cdp_capture_viewport_png(driver, render_width, render_height)
    from PIL import Image
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)

    return {"render_width": render_width, "render_height": render_height,
            "text_fit": fit_diag, "tight_crop": crop_diag}


def render_html_file(html_path: Path, width: int, height: int, chromedriver: str,
                      chrome_binary: str, out_png: Path,
                      dom_stable_timeout_ms: int = 5000) -> dict:
    """Standalone entry point — builds its own driver and tmp_dir, renders one
    file, and tears down. Used by smoke tests; the pipeline itself reuses a
    persistent per-worker driver via worker.py instead (one Chrome per process,
    not per call)."""
    driver = new_driver(chromedriver, chrome_binary)
    tmp_dir = Path(tempfile.mkdtemp(prefix="render_tmp_"))
    try:
        html_original = html_path.read_text(encoding="utf-8")
        result = render_html(
            html_original=html_original, out_png=out_png,
            width=width, height=height, driver=driver, tmp_dir=tmp_dir,
            dom_stable_timeout_ms=dom_stable_timeout_ms,
        )
        return {"ok": True, "render": str(out_png), **result}
    finally:
        driver.quit()
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
