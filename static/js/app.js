/* Data Annotation Tool — single-page app (v1.1.0)
 *
 * Sections
 *   1. Constants and small utilities
 *   2. App state
 *   3. Toasts, loading overlay, modals
 *   4. Undo / redo and the history panel
 *   5. Preferences (local) and detection settings (server)
 *   6. Classes: class bar, dropdowns, classes editor, model class picker
 *   7. Setup screen
 *   8. Opening the annotator, app modes
 *   9. Canvas: transforms, geometry, rendering, minimap
 *  10. Canvas input (mouse) and drawing
 *  11. Keyboard (one handler)
 *  12. Image list, navigation, loading and saving
 *  13. Shapes panel, status bar
 *  14. Stats, class chart, split panel
 *  15. Grid view and image modal
 *  16. Health dashboard, export, add images
 *  17. Train, resize, compress
 *  18. Shortcut sheet and help
 *  19. Init
 */
"use strict";

// ═══ 1. CONSTANTS AND UTILITIES ═══════════════════════════════════════════════
const COLORS = ["#4C9AFF", "#F0883E", "#3FB950", "#E5534B", "#2DB5A3", "#D9B526", "#5BC0EB", "#8FC93A", "#C8875A", "#FF9E9E", "#9DB0C4", "#E0C3A0"];
const ITEM_H = 42;            // sidebar row height (virtualised list)
const HANDLE = 8;             // handle size in screen px
const CLOSE_RADIUS = 9;       // px distance to first vertex that closes a polygon
const MIN_DRAW_PX = 4;        // minimum on-screen size of a new box

const $ = id => document.getElementById(id);
const clamp01 = v => Math.max(0, Math.min(1, v));
const clone = o => JSON.parse(JSON.stringify(o));
const colorOf = cls => COLORS[((cls % COLORS.length) + COLORS.length) % COLORS.length];

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/** Reads a CSS custom property from :root (canvas drawing cannot use var()). */
const _tokCache = {};
function tok(name) {
    if (!_tokCache[name]) {
        _tokCache[name] = getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "white";
    }
    return _tokCache[name];
}

/** fetch wrapper: JSON in/out, throws Error(message) on HTTP errors using the server's {error}. */
async function api(url, opts = {}) {
    const init = { method: opts.method || "GET", headers: {} };
    if (opts.json !== undefined) {
        init.method = opts.method || "POST";
        init.headers["Content-Type"] = "application/json";
        init.body = JSON.stringify(opts.json);
    }
    // Optional timeout so one hung request shows an error instead of waiting forever.
    let ctl = null, timer = null;
    if (opts.timeout) {
        ctl = new AbortController();
        init.signal = ctl.signal;
        timer = setTimeout(() => ctl.abort(), opts.timeout);
    }
    const timedOut = () => { const err = new Error("The server took too long to answer."); err.timeout = true; return err; };
    let res;
    try {
        res = await fetch(url, init);
    } catch (e) {
        clearTimeout(timer);
        if (ctl && ctl.signal.aborted) throw timedOut();
        throw new Error("Can't reach the server. Check that the app is still running.");
    }
    let data = null;
    try { data = await res.json(); } catch (e) { data = null; }
    clearTimeout(timer);
    if (ctl && ctl.signal.aborted && data === null) throw timedOut();
    if (!res.ok) {
        const err = new Error((data && data.error) || `Request failed (HTTP ${res.status})`);
        err.status = res.status;
        err.data = data;
        throw err;
    }
    return data;
}

/** Toggle visibility through the d-none utility (features.css makes it !important). */
function show(el, on) {
    if (typeof el === "string") el = $(el);
    if (el) el.classList.toggle("d-none", !on);
}

/** Sets a .field-msg element's text and tone (ok | warn | err | muted). */
function setMsg(el, tone, html) {
    if (typeof el === "string") el = $(el);
    if (!el) return;
    el.className = el.className.replace(/\b(ok|warn|err|muted)\b/g, "").trim();
    if (!el.classList.contains("field-msg")) el.classList.add("field-msg");
    el.classList.add(tone);
    el.innerHTML = html;
    show(el, !!html);
}

function debounce(fn, ms) {
    let t = null;
    const d = (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
    d.cancel = () => clearTimeout(t);
    return d;
}

function isTypingTarget(el) {
    if (!el) return false;
    if (el.isContentEditable) return true;
    const tag = el.tagName;
    if (tag === "TEXTAREA" || tag === "SELECT") return true;
    if (tag === "INPUT") {
        const t = (el.type || "text").toLowerCase();
        return !["range", "checkbox", "radio", "button", "submit", "reset", "color"].includes(t);
    }
    return false;
}

function fmtPct(v) { return Number(v).toFixed(2); }

// ═══ 2. APP STATE ═════════════════════════════════════════════════════════════
let CLASS_NAMES = [];
let datasetType = "bbox";          // 'bbox' | 'polygon'
let datasetName = "";
let currentSplit = "train";
let appMode = "annotate";          // 'annotate' | 'grid'
let annotatorOpen = false;

let allImages = [];                // [{name, annotated, reviewed}]
let filteredImages = [];
let currentIndex = -1;
let sidebarFilter = "all";         // 'all' | 'toreview' | 'reviewed'

let boxes = [];                    // shapes of the current image: {cls,x,y,w,h} or {cls,pts:[{x,y}]}
let selectedBoxes = [];
let activeClass = 0;
let dirty = false;
let editVersion = 0;               // bumps on each edit, so a late save does not clear newer edits
let saveState = "saved";           // 'saved' | 'unsaved' | 'saving' | 'error'
let autoSave = true;
let drawMode = false;
let ctrlHeld = false;
let spaceHeld = false;
let hideShapes = false;
let showLabels = true;
let askClassAfterDraw = false;
let shapeClipboard = [];

let imgNatW = 1, imgNatH = 1;
let imgReady = false;
let zoomScale = 1.0, panX = 0, panY = 0;
let bcBright = 1.0, bcContrast = 1.0;
let extraFilterType = "none", extraFilterVal = 0;
let cursorX = -1, cursorY = -1;    // screen px within the canvas, -1 when outside
let isMouseInCanvas = false;

let drag = { on: false };          // active mouse gesture (see section 10)
let polyDraft = null;              // polygon being drawn: {pts:[{x,y}]}
let _loadId = 0;
let _sessionStart = Date.now();
let serverSettings = { auto_annotate: true, conf: 0.25, iou: 0.7, model_loaded: false };

const canvas = $("annot-canvas");
const ctx = canvas.getContext("2d");
let imgEl = new Image();
const _preloadSlots = [new Image(), new Image(), new Image()];

// ═══ 3. TOASTS, LOADING, MODALS ═══════════════════════════════════════════════
/** kind: 'ok' | 'err' | 'info' */
function showToast(msg, kind = "info") {
    const t = $("toast");
    if (!t) return;
    t.textContent = msg;
    t.className = "show " + (kind === "ok" || kind === "err" ? kind : "info");
    clearTimeout(t._t);
    t._t = setTimeout(() => { t.className = ""; }, kind === "err" ? 4200 : 2400);
}

function showLoading(m) { $("loading-text").textContent = m; const o = $("loading-overlay"); o.classList.remove("is-error"); o.style.display = "flex"; }
function hideLoading() { const o = $("loading-overlay"); o.style.display = "none"; o.classList.remove("is-error"); }
/** Shows a message over the canvas without a spinner; a click dismisses it. */
function showOverlayMessage(m) { $("loading-text").textContent = m; const o = $("loading-overlay"); o.classList.add("is-error"); o.style.display = "flex"; }

const _modalStack = [];
function openModal(id) {
    const el = $(id);
    if (!el) return;
    el.classList.remove("hidden", "d-none");
    el.classList.add("is-open");
    el.style.display = "flex";
    const i = _modalStack.indexOf(id);
    if (i >= 0) _modalStack.splice(i, 1);
    _modalStack.push(id);
}
function closeModal(id) {
    const el = $(id);
    if (!el) return;
    if (el.contains(document.activeElement)) document.activeElement.blur();
    el.classList.remove("is-open");
    el.style.display = "none";
    const i = _modalStack.indexOf(id);
    if (i >= 0) _modalStack.splice(i, 1);
}
function isModalOpen(id) { return _modalStack.includes(id); }
function anyModalOpen() { return _modalStack.length > 0; }
function closeTopModal() {
    const id = _modalStack[_modalStack.length - 1];
    if (!id) return false;
    if (id === "image-modal") closeImageModal(); else closeModal(id);
    return true;
}
function bindModalChrome() {
    ["add-images-modal", "classes-modal", "shortcuts-modal", "help-modal", "health-modal", "image-modal"].forEach(id => {
        const el = $(id);
        if (!el) return;
        el.addEventListener("click", e => {
            if (e.target === el) { if (id === "image-modal") closeImageModal(); else closeModal(id); }
        });
        el.querySelectorAll("[data-close-modal]").forEach(b => b.addEventListener("click", () => closeModal(id)));
    });
}

// ═══ 4. UNDO / REDO ═══════════════════════════════════════════════════════════
class StateSnapshotCommand {
    constructor(prev, next, actionName) {
        this.prevBoxes = clone(prev);
        this.newBoxes = clone(next);
        this.actionName = actionName || "Edited shapes";
        this.time = Date.now();
        const d = new Date();
        this.timestamp = [d.getHours(), d.getMinutes(), d.getSeconds()].map(n => String(n).padStart(2, "0")).join(":");
    }
}

class CommandManager {
    constructor() { this.undoStack = []; this.redoStack = []; this.maxHistory = 50; }
    /** Returns true if a step was recorded. coalesce: merge into the previous step when it has the same name and is recent. */
    pushState(prev, next, actionName, coalesce = false) {
        if (JSON.stringify(prev) === JSON.stringify(next)) return false;
        const last = this.undoStack[this.undoStack.length - 1];
        if (coalesce && last && last.actionName === actionName && Date.now() - last.time < 1200) {
            last.newBoxes = clone(next);
            last.time = Date.now();
        } else {
            this.undoStack.push(new StateSnapshotCommand(prev, next, actionName));
            if (this.undoStack.length > this.maxHistory) this.undoStack.shift();
        }
        this.redoStack = [];
        renderHistoryPanel();
        return true;
    }
    undo() {
        const cmd = this.undoStack.pop();
        if (!cmd) { showToast("Nothing to undo"); return; }
        boxes = clone(cmd.prevBoxes);
        this.redoStack.push(cmd);
        afterHistoryJump("Undid: " + cmd.actionName);
    }
    redo() {
        const cmd = this.redoStack.pop();
        if (!cmd) { showToast("Nothing to redo"); return; }
        boxes = clone(cmd.newBoxes);
        this.undoStack.push(cmd);
        afterHistoryJump("Redid: " + cmd.actionName);
    }
    clear() { this.undoStack = []; this.redoStack = []; renderHistoryPanel(); }
}
const cmdManager = new CommandManager();

function afterHistoryJump(msg) {
    selectedBoxes = [];
    polyDraft = null;
    markDirty();
    triggerAutoSave();
    updateBoxPanel();
    renderCanvas();
    renderHistoryPanel();
    showToast(msg);
}

function renderHistoryPanel() {
    const list = $("history-list");
    if (!list) return;
    list.innerHTML = "";
    const cnt = $("history-count");
    if (cnt) cnt.textContent = cmdManager.undoStack.length ? `(${cmdManager.undoStack.length})` : "";
    if (cmdManager.undoStack.length === 0) {
        list.innerHTML = '<div class="hist-empty">No edits on this image yet.</div>';
        return;
    }
    cmdManager.undoStack.forEach((cmd, i) => {
        const item = document.createElement("div");
        item.className = "hist-item" + (i === cmdManager.undoStack.length - 1 ? " latest" : "");
        item.innerHTML = `<span class="hist-time">${cmd.timestamp}</span> ${escapeHtml(cmd.actionName)}`;
        list.appendChild(item);
    });
    const panel = $("history-panel");
    if (panel) panel.scrollTop = panel.scrollHeight;
}

let _pendingUndoState = null;
function pushUndo() { _pendingUndoState = clone(boxes); }
/** Commits the snapshot taken by pushUndo(). Returns true when something changed. */
function commitUndo(actionName = "Edited shapes", coalesce = false) {
    if (!_pendingUndoState) return false;
    const changed = cmdManager.pushState(_pendingUndoState, boxes, actionName, coalesce);
    _pendingUndoState = null;
    return changed;
}
/** Runs a mutation as one undoable, auto-saved step. */
function editShapes(actionName, mutate, coalesce = false) {
    if (currentIndex < 0) return false;
    pushUndo();
    mutate();
    const changed = commitUndo(actionName, coalesce);
    if (changed) { markDirty(); triggerAutoSave(); }
    updateBoxPanel();
    renderCanvas();
    return changed;
}
function doUndo() { cmdManager.undo(); }
function doRedo() { cmdManager.redo(); }

// ═══ 5. PREFERENCES AND DETECTION SETTINGS ════════════════════════════════════
function loadPrefs() {
    try {
        askClassAfterDraw = localStorage.getItem("askClassAfterDraw") === "1";
        showLabels = localStorage.getItem("showLabels") !== "0";
    } catch (e) { /* storage unavailable: keep defaults */ }
    $("ask-class-toggle").checked = askClassAfterDraw;
    try {
        if (localStorage.getItem("historyOpen") === "1") {
            $("history-panel").classList.remove("collapsed");
            $("history-toggle").setAttribute("aria-expanded", "true");
            $("history-toggle").querySelector("i").className = "fa-solid fa-chevron-down";
        }
    } catch (e) { /* ignore */ }
}
function savePref(key, val) { try { localStorage.setItem(key, val); } catch (e) { /* ignore */ } }

const SETTINGS_UI = [
    { conf: "conf-slider", confVal: "conf-value", iou: "iou-slider", iouVal: "iou-value", auto: "auto-label-toggle", saved: "settings-saved" },
    { conf: "det-conf", confVal: "det-conf-value", iou: "det-iou", iouVal: "det-iou-value", auto: "det-auto", saved: "det-saved" },
];

function renderSettings() {
    const s = serverSettings;
    SETTINGS_UI.forEach(u => {
        if ($(u.conf)) $(u.conf).value = s.conf;
        if ($(u.confVal)) $(u.confVal).textContent = fmtPct(s.conf);
        if ($(u.iou) && s.iou != null) $(u.iou).value = s.iou;
        if ($(u.iouVal) && s.iou != null) $(u.iouVal).textContent = fmtPct(s.iou);
        if ($(u.auto)) $(u.auto).checked = !!s.auto_annotate;
    });
    const hint = $("settings-hint");
    if (hint) {
        if (!s.model_loaded) setMsg(hint, "muted", "No model loaded. Load one above to auto-label, or label by hand.");
        else if (s.auto_annotate) setMsg(hint, "muted", "The model labels each image the first time you open it.");
        else setMsg(hint, "muted", "Auto-labelling is off. New images open without labels.");
    }
}

async function loadSettings() {
    try {
        const s = await api("/api/settings");
        serverSettings = Object.assign({}, serverSettings, s || {});
    } catch (e) { /* older backend: keep defaults */ }
    renderSettings();
    setSavedNote("");
}

function setSavedNote(text) {
    SETTINGS_UI.forEach(u => { const el = $(u.saved); if (el) { el.textContent = text; show(el, !!text); } });
}

const _pendingSettings = {};
const _flushSettings = debounce(async () => {
    const body = Object.assign({}, _pendingSettings);
    Object.keys(_pendingSettings).forEach(k => delete _pendingSettings[k]);
    setSavedNote("Saving…");
    try {
        const s = await api("/api/settings", { json: body });
        serverSettings = Object.assign({}, serverSettings, s || body);
        renderSettings();
        const where = serverSettings.dataset_name || datasetName;
        setSavedNote(where ? `Saved to ${where}` : "Saved");
    } catch (e) {
        setSavedNote("");
        showToast("Couldn't save detection settings: " + e.message, "err");
    }
}, 300);

function queueSetting(key, val) {
    serverSettings[key] = val;
    _pendingSettings[key] = val;
    renderSettings();
    _flushSettings();
}

function bindSettingsControls() {
    SETTINGS_UI.forEach(u => {
        $(u.conf)?.addEventListener("input", e => queueSetting("conf", Math.round(parseFloat(e.target.value) * 100) / 100));
        $(u.iou)?.addEventListener("input", e => queueSetting("iou", Math.round(parseFloat(e.target.value) * 100) / 100));
        $(u.auto)?.addEventListener("change", e => queueSetting("auto_annotate", e.target.checked));
    });
}

// ═══ 6. CLASSES ═══════════════════════════════════════════════════════════════
function className(cls) { return CLASS_NAMES[cls] != null ? CLASS_NAMES[cls] : `class ${cls}`; }
const fmtNum = n => (n == null ? "–" : Number(n).toLocaleString("en-US"));

let classUsage = {};       // class id -> box count (from /api/classes/usage)
let classSlots = [];       // 9 key slots -> class id | null (persisted per dataset)
let recentClasses = [];    // most recent first (persisted per dataset)

function dsKey(k) { return `${k}:${datasetName || "default"}`; }
function loadClassPrefs() {
    let slots = null, recent = null;
    try {
        slots = JSON.parse(localStorage.getItem(dsKey("classSlots")) || "null");
        recent = JSON.parse(localStorage.getItem(dsKey("recentClasses")) || "null");
    } catch (e) { /* ignore */ }
    classSlots = Array.isArray(slots) && slots.length === 9 ? slots : [0, 1, 2, 3, 4, 5, 6, 7, 8];
    recentClasses = Array.isArray(recent) ? recent : [];
    normalizeClassPrefs();
}
function saveClassPrefs() {
    savePref(dsKey("classSlots"), JSON.stringify(classSlots));
    savePref(dsKey("recentClasses"), JSON.stringify(recentClasses));
}
function normalizeClassPrefs() {
    const n = CLASS_NAMES.length;
    classSlots = classSlots.map(id => (Number.isInteger(id) && id >= 0 && id < n ? id : null));
    recentClasses = recentClasses.filter(id => Number.isInteger(id) && id >= 0 && id < n).slice(0, 8);
}
/** After merge/delete removed some ids, shift the remembered ids down so they keep pointing at the same names. */
function remapClassIds(removed, mergedInto) {
    const rm = [...removed].sort((a, b) => a - b);
    const map = id => {
        if (id == null) return null;
        if (rm.includes(id)) id = mergedInto != null ? mergedInto : null;
        if (id == null) return null;
        return id - rm.filter(r => r < id).length;
    };
    classSlots = classSlots.map(map);
    recentClasses = recentClasses.map(map).filter(id => id != null);
    activeClass = map(activeClass) || 0;
}
function slotOf(id) { return classSlots.indexOf(id); }
function pushRecent(id) {
    recentClasses = [id, ...recentClasses.filter(x => x !== id)].slice(0, 8);
    saveClassPrefs();
}

function setClasses(list) {
    CLASS_NAMES = Array.isArray(list) ? list.slice() : [];
    if (activeClass >= CLASS_NAMES.length) activeClass = 0;
    normalizeClassPrefs();
    buildClassDropdowns();
    renderClassBar();
    renderClassTags();
    renderSlotHotkeys();
    if (palette.open) renderPalette();
    if (annotatorOpen) { updateBoxPanel(); renderCanvas(); }
}

let gridClassFilterMode = "all"; // "all" or "custom"
let gridSelectedClasses = new Set();

function updateGridClassLabel() {
    const lbl = $("grid-class-label");
    if (!lbl) return;
    if (gridClassFilterMode === "all" || gridSelectedClasses.size === CLASS_NAMES.length) {
        lbl.textContent = "All classes";
    } else if (gridSelectedClasses.size === 0) {
        lbl.textContent = "No classes";
    } else if (gridSelectedClasses.size === 1) {
        const id = [...gridSelectedClasses][0];
        lbl.textContent = className(id);
    } else {
        lbl.textContent = `${gridSelectedClasses.size} classes`;
    }
}

function syncGridClassCheckboxes() {
    const list = $("grid-class-list");
    if (!list) return;
    list.querySelectorAll("input[type=checkbox]").forEach(cb => {
        const i = Number(cb.value);
        cb.checked = gridClassFilterMode === "all" || gridSelectedClasses.has(i);
    });
    updateGridClassLabel();
}

function setGridClassFilterAll() {
    gridClassFilterMode = "all";
    gridSelectedClasses.clear();
    syncGridClassCheckboxes();
    loadGridPage(1);
}

function clearGridClassFilter() {
    gridClassFilterMode = "custom";
    gridSelectedClasses.clear();
    syncGridClassCheckboxes();
    loadGridPage(1);
}

function closeGridClassPop() {
    const pop = $("grid-class-pop");
    if (pop && !pop.hidden) {
        pop.hidden = true;
        $("btn-grid-class-filter")?.setAttribute("aria-expanded", "false");
        return true;
    }
    return false;
}

function toggleGridClassPop(e) {
    if (e) {
        e.preventDefault();
        e.stopPropagation();
    }
    const pop = $("grid-class-pop");
    if (!pop) return;
    const isHidden = pop.hidden;
    if (isHidden) {
        const list = $("grid-class-list");
        if (list && (!list.children.length || list.querySelector(".mc-empty"))) {
            buildClassDropdowns();
        }
        pop.hidden = false;
        $("btn-grid-class-filter")?.setAttribute("aria-expanded", "true");
    } else {
        pop.hidden = true;
        $("btn-grid-class-filter")?.setAttribute("aria-expanded", "false");
    }
}

function buildClassDropdowns() {
    const gcsel = $("grid-class-filter");
    if (gcsel && gcsel.tagName === "SELECT") {
        const prev = gcsel.value;
        gcsel.innerHTML = '<option value="">Any class</option>';
        CLASS_NAMES.forEach((c, i) => {
            const o = document.createElement("option");
            o.value = i; o.textContent = `${i}: ${c}`;
            gcsel.appendChild(o);
        });
        gcsel.value = prev && Number(prev) < CLASS_NAMES.length ? prev : "";
    }
    const list = $("grid-class-list");
    if (!list) return;
    list.innerHTML = "";
    if (!CLASS_NAMES.length) {
        list.innerHTML = '<div class="mc-empty">No classes in dataset.</div>';
        updateGridClassLabel();
        return;
    }
    CLASS_NAMES.forEach((c, i) => {
        const row = document.createElement("label");
        row.className = "grid-class-item";
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.value = i;
        cb.checked = gridClassFilterMode === "all" || gridSelectedClasses.has(i);
        cb.addEventListener("change", () => {
            if (gridClassFilterMode === "all") {
                gridClassFilterMode = "custom";
                CLASS_NAMES.forEach((_, j) => {
                    if (j !== i) gridSelectedClasses.add(j);
                });
                if (cb.checked) gridSelectedClasses.add(i);
            } else {
                if (cb.checked) gridSelectedClasses.add(i);
                else gridSelectedClasses.delete(i);
            }
            if (gridSelectedClasses.size === CLASS_NAMES.length) {
                gridClassFilterMode = "all";
                gridSelectedClasses.clear();
            }
            syncGridClassCheckboxes();
            loadGridPage(1);
        });
        const sw = document.createElement("span");
        sw.className = "swatch";
        sw.style.background = colorOf(i);
        const nameSpan = document.createElement("span");
        nameSpan.className = "grid-class-name";
        nameSpan.textContent = `${i}: ${c}`;
        row.append(cb, sw, nameSpan);
        list.appendChild(row);
    });
    updateGridClassLabel();
}

function renderClassTags() {
    const tags = $("classes-tags");
    if (!tags) return;
    tags.innerHTML = "";
    if (CLASS_NAMES.length === 0) {
        tags.innerHTML = '<div class="field-msg muted">No classes yet. Load a model, or click Manage classes to add them.</div>';
        return;
    }
    const MAX = 24;
    CLASS_NAMES.slice(0, MAX).forEach((c, i) => {
        const t = document.createElement("span");
        t.className = "class-tag";
        t.innerHTML = `<span class="swatch" style="background:${colorOf(i)}"></span>${i}: ${escapeHtml(c)}`;
        tags.appendChild(t);
    });
    if (CLASS_NAMES.length > MAX) {
        const more = document.createElement("span");
        more.className = "chip-more";
        more.textContent = `+${CLASS_NAMES.length - MAX} more`;
        tags.appendChild(more);
    }
}

async function loadClassUsage() {
    try {
        const r = await api("/api/classes/usage");
        classUsage = {};
        (r.classes || []).forEach(c => { classUsage[c.id] = c.boxes || 0; });
    } catch (e) { /* counts are optional */ }
    renderClassBar();
}

/** One fixed row: active pill | chips that fit | +N more. */
function renderClassBar() {
    const wrap = $("class-chips");
    if (!wrap) return;
    const n = CLASS_NAMES.length;
    $("acp-swatch").style.background = n ? colorOf(activeClass) : "var(--text-3)";
    $("acp-name").textContent = n ? className(activeClass) : "No classes";
    const k = slotOf(activeClass);
    $("acp-key").textContent = k >= 0 ? k + 1 : "";
    show("acp-key", n > 0 && k >= 0);
    wrap.innerHTML = "";
    if (!n) {
        wrap.innerHTML = '<span class="class-bar-empty">No classes yet. <button type="button" class="link-btn" id="class-bar-edit">Add classes</button></span>';
        $("class-bar-edit").addEventListener("click", openClassesManager);
        show("class-more", false);
        return;
    }
    const order = [];
    const add = id => { if (id != null && id < n && !order.includes(id)) order.push(id); };
    classSlots.forEach(add);
    Object.keys(classUsage).map(Number).sort((a, b) => (classUsage[b] || 0) - (classUsage[a] || 0)).slice(0, 30).forEach(add);
    recentClasses.forEach(add);
    for (let i = 0; i < n && order.length < 30; i++) add(i);
    order.forEach(id => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "class-chip" + (id === activeClass ? " active" : "");
        const s = slotOf(id);
        chip.title = s >= 0 ? `${className(id)} (key ${s + 1})` : className(id);
        chip.innerHTML = `<span class="swatch" style="background:${colorOf(id)}"></span>` +
            (s >= 0 ? `<kbd>${s + 1}</kbd>` : "") + `<span class="class-chip-name">${escapeHtml(className(id))}</span>`;
        chip.addEventListener("click", () => applyClassKey(id));
        wrap.appendChild(chip);
    });
    fitClassChips();
}

function fitClassChips() {
    const wrap = $("class-chips");
    const chips = [...wrap.querySelectorAll(".class-chip")];
    if (!chips.length) return;
    chips.forEach(c => { c.style.display = ""; });
    const more = $("class-more");
    show(more, true);
    more.textContent = `+${CLASS_NAMES.length} more`;
    const W = wrap.clientWidth;
    let visible = 0, overflow = false;
    chips.forEach(c => {
        if (overflow || c.offsetLeft + c.offsetWidth > W) { overflow = true; c.style.display = "none"; }
        else visible++;
    });
    const rest = CLASS_NAMES.length - visible;
    more.textContent = `+${rest} more`;
    show(more, rest > 0);
}

function renderSlotHotkeys() {
    const el = $("slot-hotkeys");
    if (!el) return;
    el.innerHTML = "";
    classSlots.forEach((id, k) => {
        if (id == null) return;
        const d = document.createElement("div");
        d.className = "slot-row";
        d.title = `Press ${k + 1} for ${className(id)}`;
        d.innerHTML = `<kbd>${k + 1}</kbd><span class="swatch" style="background:${colorOf(id)}"></span><span class="slot-name">${escapeHtml(className(id))}</span>`;
        el.appendChild(d);
    });
}

function setActiveClass(i) {
    if (i == null || i < 0 || (CLASS_NAMES.length && i >= CLASS_NAMES.length)) return;
    activeClass = i;
    if (CLASS_NAMES.length) pushRecent(i);
    renderClassBar();
    updateStatusBar();
    renderCanvas();
}

/** Chip click / palette choice: reassigns the selection, or sets the active class when nothing is selected. */
function applyClassKey(i) {
    if (i == null || i >= CLASS_NAMES.length) return;
    if (selectedBoxes.length > 0 && currentIndex >= 0) {
        editShapes(`Set class to ${className(i)}`, () => selectedBoxes.forEach(j => { if (boxes[j]) boxes[j].cls = i; }));
    }
    setActiveClass(i);
}
function applySlotKey(k) {
    const id = classSlots[k];
    if (id == null) { showToast(`Key ${k + 1} has no class. Press / and use Alt+${k + 1} on a class to pin it.`); return; }
    applyClassKey(id);
}

function cycleClass(dir) {
    const n = CLASS_NAMES.length;
    if (n === 0) return;
    if (selectedBoxes.length > 0) {
        editShapes("Changed class", () => selectedBoxes.forEach(j => { boxes[j].cls = ((boxes[j].cls + dir) % n + n) % n; }));
    } else {
        setActiveClass(((activeClass + dir) % n + n) % n);
    }
}

// ─── Class palette (searchable, keyboard driven) ──────────────────────────────
const palette = { open: false, mode: "set", items: [], hi: 0 };

function openPalette(mode = "set") {
    if (!CLASS_NAMES.length) { openClassesManager(); return; }
    palette.open = true;
    palette.mode = mode;
    palette.hi = 0;
    $("class-palette").hidden = false;
    $("active-class-pill").setAttribute("aria-expanded", "true");
    $("cp-head").textContent = mode === "assign" ? "Class for the new shape"
        : selectedBoxes.length ? `Reassign ${selectedBoxes.length} selected ${selectedBoxes.length === 1 ? "shape" : "shapes"}` : "Set active class";
    $("cp-search").value = "";
    renderPalette();
    setTimeout(() => $("cp-search").focus(), 0);
}
function closePalette() {
    if (!palette.open) return false;
    palette.open = false;
    $("class-palette").hidden = true;
    $("active-class-pill").setAttribute("aria-expanded", "false");
    if (document.activeElement && $("class-palette").contains(document.activeElement)) document.activeElement.blur();
    if (palette.mode === "assign") triggerAutoSave();
    return true;
}
function paletteRank(q, id) {
    const name = className(id).toLowerCase();
    if (String(id) === q) return 0;
    if (name.startsWith(q)) return 1;
    if (name.split(/[\s_-]+/).some(w => w.startsWith(q))) return 2;
    if (name.includes(q)) return 3;
    return -1;
}
function renderPalette() {
    const list = $("cp-list");
    const q = $("cp-search").value.trim().toLowerCase();
    const sections = [];
    if (!q) {
        const rec = recentClasses.filter(id => id < CLASS_NAMES.length).slice(0, 6);
        if (rec.length) sections.push(["Recent", rec]);
        sections.push(["All classes", CLASS_NAMES.map((_, i) => i)]);
    } else {
        const hits = CLASS_NAMES.map((_, i) => [i, paletteRank(q, i)]).filter(([, r]) => r >= 0)
            .sort((a, b) => a[1] - b[1] || a[0] - b[0]).map(([i]) => i);
        sections.push([`${hits.length} ${hits.length === 1 ? "match" : "matches"}`, hits]);
    }
    palette.items = [];
    list.innerHTML = "";
    sections.forEach(([title, ids]) => {
        const h = document.createElement("div");
        h.className = "cp-section";
        h.textContent = title;
        list.appendChild(h);
        if (!ids.length) {
            const e = document.createElement("div");
            e.className = "mc-empty";
            e.textContent = "No class matches. Add it with Manage classes.";
            list.appendChild(e);
        }
        ids.forEach(id => {
            const idx = palette.items.length;
            palette.items.push(id);
            const row = document.createElement("div");
            row.className = "cp-row" + (idx === palette.hi ? " hi" : "") + (id === activeClass ? " active" : "");
            row.setAttribute("role", "option");
            row.dataset.idx = idx;
            const s = slotOf(id);
            row.innerHTML = `<span class="swatch" style="background:${colorOf(id)}"></span><span class="cp-id">${id}</span>
                <span class="cp-name">${escapeHtml(className(id))}</span>
                <span class="cp-count">${classUsage[id] != null ? fmtNum(classUsage[id]) : ""}</span>`;
            const pin = document.createElement("button");
            pin.type = "button";
            pin.className = "cp-pin" + (s >= 0 ? " pinned" : "");
            pin.textContent = s >= 0 ? `Key ${s + 1}` : "Pin to key";
            pin.title = "Choose which number key selects this class";
            pin.addEventListener("mousedown", e => e.preventDefault());
            pin.addEventListener("click", e => { e.stopPropagation(); showPinChooser(pin, id); });
            row.appendChild(pin);
            row.addEventListener("mousedown", e => e.preventDefault());   // keep focus in the search box
            row.addEventListener("click", () => choosePalette(id));
            row.addEventListener("mousemove", () => { if (palette.hi !== idx) { palette.hi = idx; markPaletteHi(); } });
            list.appendChild(row);
        });
    });
    if (palette.hi >= palette.items.length) palette.hi = Math.max(0, palette.items.length - 1);
    markPaletteHi();
}
function markPaletteHi() {
    $("cp-list").querySelectorAll(".cp-row").forEach(r => r.classList.toggle("hi", Number(r.dataset.idx) === palette.hi));
    const el = $("cp-list").querySelector(".cp-row.hi");
    if (el) el.scrollIntoView({ block: "nearest" });
}
function showPinChooser(btn, id) {
    const box = document.createElement("span");
    box.className = "cp-keys";
    for (let k = 0; k < 9; k++) {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = k + 1;
        b.title = classSlots[k] != null ? `Key ${k + 1}: ${className(classSlots[k])}` : `Key ${k + 1}: empty`;
        if (classSlots[k] === id) b.classList.add("on");
        b.addEventListener("mousedown", e => e.preventDefault());
        b.addEventListener("click", e => { e.stopPropagation(); pinSlot(id, k); });
        box.appendChild(b);
    }
    btn.replaceWith(box);
}
function pinSlot(id, k) {
    const old = slotOf(id);
    if (old >= 0) classSlots[old] = classSlots[k] === id ? null : classSlots[k];
    classSlots[k] = id;
    saveClassPrefs();
    renderClassBar();
    renderSlotHotkeys();
    if (palette.open) renderPalette();
    showToast(`Key ${k + 1} now selects ${className(id)}`);
}
function choosePalette(id) {
    if (id == null) return;
    const mode = palette.mode;
    palette.mode = "set";   // the shape gets its class below; closing must not re-trigger
    closePalette();
    if (mode === "assign" && selectedBoxes.length) {
        editShapes(`Set class to ${className(id)}`, () => selectedBoxes.forEach(j => { if (boxes[j]) boxes[j].cls = id; }));
        setActiveClass(id);
        triggerAutoSave();
    } else {
        applyClassKey(id);
    }
}
function onPaletteKey(e) {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        const n = palette.items.length;
        if (n) palette.hi = (palette.hi + (e.key === "ArrowDown" ? 1 : -1) + n) % n;
        markPaletteHi();
    } else if (e.key === "Enter") {
        e.preventDefault();
        choosePalette(palette.items[palette.hi]);
    } else if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        closePalette();
    } else if (e.altKey && /^[1-9]$/.test(e.key)) {
        e.preventDefault();
        const id = palette.items[palette.hi];
        if (id != null) pinSlot(id, parseInt(e.key, 10) - 1);
    } else if (e.altKey && /^Digit[1-9]$/.test(e.code)) {
        e.preventDefault();
        const id = palette.items[palette.hi];
        if (id != null) pinSlot(id, parseInt(e.code.slice(5), 10) - 1);
    }
}
function bindClassBar() {
    $("active-class-pill").addEventListener("click", e => { e.stopPropagation(); if (palette.open) closePalette(); else openPalette("set"); });
    $("class-more").addEventListener("click", e => { e.stopPropagation(); if (palette.open) closePalette(); else openPalette("set"); });
    $("class-palette").addEventListener("click", e => e.stopPropagation());
    $("cp-search").addEventListener("input", () => { palette.hi = 0; renderPalette(); });
    $("cp-search").addEventListener("keydown", onPaletteKey);
    if (window.ResizeObserver) new ResizeObserver(() => fitClassChips()).observe($("class-bar"));
    else window.addEventListener("resize", fitClassChips);
}

// ─── Manage classes modal ─────────────────────────────────────────────────────
let clsUsage = { classes: [], unknown_ids: [] };
const clsSelected = new Set();

async function openClassesManager() {
    clsSelected.clear();
    closeClsAction();
    setMsg("classes-modal-info", "muted", "Click a name to rename it. Renaming never changes labels. Merge and delete rewrite label files, and a backup is saved first.");
    openModal("classes-modal");
    await refreshClassesManager();
}

async function refreshClassesManager() {
    try {
        clsUsage = await api("/api/classes/usage");
        const names = (clsUsage.classes || []).map(c => c.name);
        if (JSON.stringify(names) !== JSON.stringify(CLASS_NAMES)) setClasses(names);
        classUsage = {};
        (clsUsage.classes || []).forEach(c => { classUsage[c.id] = c.boxes || 0; });
        renderClassBar();
    } catch (e) {
        clsUsage = { classes: CLASS_NAMES.map((name, id) => ({ id, name, boxes: null, images: null })), unknown_ids: [] };
        setMsg("classes-modal-info", "err", `Couldn't count how often each class is used: ${escapeHtml(e.message)}. Open a dataset first to manage its classes.`);
    }
    renderClsTable();
}

function renderClsTable() {
    const tb = $("cls-tbody");
    tb.innerHTML = "";
    const rows = clsUsage.classes || [];
    if (!rows.length) {
        tb.innerHTML = '<tr><td colspan="5" class="cls-empty">No classes yet. Click Add class to create one, or load a model in Setup.</td></tr>';
    }
    rows.forEach(c => {
        const tr = document.createElement("tr");
        if (clsSelected.has(c.id)) tr.className = "sel";
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = clsSelected.has(c.id);
        cb.setAttribute("aria-label", `Select ${c.name}`);
        cb.addEventListener("change", () => { if (cb.checked) clsSelected.add(c.id); else clsSelected.delete(c.id); tr.classList.toggle("sel", cb.checked); updateClsToolbar(); });
        const td0 = document.createElement("td"); td0.className = "cls-col-check"; td0.appendChild(cb);
        const td1 = document.createElement("td"); td1.className = "cls-col-id";
        td1.innerHTML = `<span class="swatch" style="background:${colorOf(c.id)}"></span> ${c.id}`;
        const td2 = document.createElement("td");
        const inp = document.createElement("input");
        inp.type = "text";
        inp.className = "cls-name";
        inp.value = c.name;
        inp.spellcheck = false;
        inp.setAttribute("aria-label", `Name of class ${c.id}`);
        inp.addEventListener("keydown", e => {
            if (e.key === "Enter") { e.preventDefault(); inp.blur(); }
            if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); inp.value = c.name; inp.blur(); }
        });
        inp.addEventListener("blur", () => renameClass(c, inp));
        td2.appendChild(inp);
        const td3 = document.createElement("td"); td3.className = "cls-col-num"; td3.textContent = fmtNum(c.boxes);
        const td4 = document.createElement("td"); td4.className = "cls-col-num"; td4.textContent = fmtNum(c.images);
        tr.append(td0, td1, td2, td3, td4);
        tb.appendChild(tr);
    });
    const unk = clsUsage.unknown_ids || [];
    if (unk.length) {
        setMsg("cls-unknown", "warn", `Labels use class IDs that aren't in this list: ${unk.map(u => `${u.id} (${fmtNum(u.boxes)} boxes)`).join(", ")}. Add classes until those IDs exist, or fix the label files.`);
    } else show("cls-unknown", false);
    $("cls-check-all").checked = rows.length > 0 && clsSelected.size === rows.length;
    updateClsToolbar();
}

function updateClsToolbar() {
    const n = (clsUsage.classes || []).length;
    $("btn-cls-merge").disabled = !(clsSelected.size >= 1 && clsSelected.size < n);
    $("btn-cls-delete").disabled = clsSelected.size !== 1;
    $("btn-cls-delete").title = clsSelected.size > 1 ? "Delete works on one class at a time. To combine several, use Merge." : "Delete the selected class";
}

async function renameClass(c, inp) {
    const name = inp.value.trim();
    if (name === c.name) return;
    if (!name) { inp.value = c.name; setMsg("classes-modal-info", "err", "A class name can't be empty."); return; }
    try {
        const r = await api("/api/classes/rename", { json: { id: c.id, name } });
        c.name = name;
        setClasses(r.classes || CLASS_NAMES.map((x, i) => (i === c.id ? name : x)));
        loadModelClasses(false);
        if (appMode === "grid") loadGridPage(gridPage);
        setMsg("classes-modal-info", "ok", `Renamed class ${c.id} to "${escapeHtml(name)}". Labels didn't change.`);
    } catch (e) {
        inp.value = c.name;
        setMsg("classes-modal-info", "err", `Couldn't rename: ${escapeHtml(e.message)}`);
    }
}

function closeClsAction() { const a = $("cls-action"); a.innerHTML = ""; show(a, false); }
function openClsAction(html) {
    const a = $("cls-action");
    a.innerHTML = html;
    show(a, true);
    return a;
}
function classOptions(excludeIds) {
    return (clsUsage.classes || []).filter(c => !excludeIds.includes(c.id))
        .map(c => `<option value="${c.id}">${c.id}: ${escapeHtml(c.name)}</option>`).join("");
}

function clsAddForm() {
    const a = openClsAction(`<div class="cls-action-row"><label for="cls-add-input">New class name</label>
        <input type="text" id="cls-add-input" class="glass-input" spellcheck="false" placeholder="e.g. bike">
        <button class="btn btn-primary btn-sm" id="cls-add-ok"><i class="fa-solid fa-plus"></i> Add</button>
        <button class="btn btn-secondary btn-sm" id="cls-add-cancel">Cancel</button></div>`);
    const inp = a.querySelector("#cls-add-input");
    const go = async () => {
        const name = inp.value.trim();
        if (!name) { setMsg("classes-modal-info", "err", "Type a name for the new class."); return; }
        try {
            const r = await api("/api/classes/add", { json: { name } });
            await afterClassChange(r, `Added class "${name}" as ID ${(r.classes || []).length - 1}.`);
        } catch (e) { setMsg("classes-modal-info", "err", `Couldn't add the class: ${escapeHtml(e.message)}`); }
    };
    a.querySelector("#cls-add-ok").addEventListener("click", go);
    a.querySelector("#cls-add-cancel").addEventListener("click", closeClsAction);
    inp.addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); go(); } if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); closeClsAction(); } });
    inp.focus();
}

function clsByIds(ids) { return ids.map(id => (clsUsage.classes || []).find(c => c.id === id)).filter(Boolean); }

function mergeConfirmText(sources, target) {
    const names = sources.map(c => c.name).join(", ");
    const boxesN = sources.reduce((s, c) => s + (c.boxes || 0), 0);
    const files = sources.reduce((s, c) => s + (c.images || 0), 0);
    if (!boxesN) return `Merge ${names} into ${target.name}? No labels use ${sources.length === 1 ? "it" : "them"}, so only the class list changes. A backup is saved first.`;
    const filesText = sources.length === 1 ? fmtNum(files) : `up to ${fmtNum(files)}`;
    return `Merge ${names} into ${target.name}? ${fmtNum(boxesN)} boxes in ${filesText} files will change. A backup is saved first.`;
}

function clsMergeForm() {
    const ids = [...clsSelected].sort((a, b) => a - b);
    const sources = clsByIds(ids);
    const a = openClsAction(`<div class="cls-action-row"><span>Merge <b>${escapeHtml(sources.map(c => c.name).join(", "))}</b> into</span>
        <select id="cls-merge-target" class="field-select glass-input">${classOptions(ids)}</select>
        <button class="btn btn-primary btn-sm" id="cls-merge-ok"><i class="fa-solid fa-code-merge"></i> Merge…</button>
        <button class="btn btn-secondary btn-sm" id="cls-merge-cancel">Cancel</button></div>
        <div class="field-msg muted">The merged classes are removed and the IDs after them move up, so IDs stay in order.</div>`);
    a.querySelector("#cls-merge-cancel").addEventListener("click", closeClsAction);
    a.querySelector("#cls-merge-ok").addEventListener("click", async () => {
        const tid = parseInt(a.querySelector("#cls-merge-target").value, 10);
        const target = clsByIds([tid])[0];
        if (!target || !confirm(mergeConfirmText(sources, target))) return;
        if (!(await resolveDirty())) return;
        try {
            const r = await api("/api/classes/merge", { json: { sources: ids, target: tid } });
            remapClassIds(ids, tid);
            await afterClassChange(r, `Merged ${sources.map(c => c.name).join(", ")} into ${target.name}.`);
        } catch (e) { setMsg("classes-modal-info", "err", `Merge failed: ${escapeHtml(e.message)}`); }
    });
}

function clsDeleteForm() {
    const id = [...clsSelected][0];
    const c = clsByIds([id])[0];
    if (!c) return;
    const others = classOptions([id]);
    const a = openClsAction(`<div class="cls-action-col"><div>Delete <b>${escapeHtml(c.name)}</b> (${fmtNum(c.boxes)} boxes in ${fmtNum(c.images)} files). What should happen to its boxes?</div>
        <label class="switch-row"><input type="radio" name="cls-del-mode" value="remove" checked> Delete its boxes</label>
        <label class="switch-row"><input type="radio" name="cls-del-mode" value="merge" ${others ? "" : "disabled"}> Move its boxes to
          <select id="cls-del-target" class="field-select glass-input" ${others ? "" : "disabled"}>${others}</select></label>
        <div class="cls-action-row"><button class="btn btn-danger btn-sm" id="cls-del-ok"><i class="fa-solid fa-trash"></i> Delete class…</button>
        <button class="btn btn-secondary btn-sm" id="cls-del-cancel">Cancel</button></div></div>`);
    a.querySelector("#cls-del-target")?.addEventListener("change", () => { a.querySelector('input[value="merge"]').checked = true; });
    a.querySelector("#cls-del-cancel").addEventListener("click", closeClsAction);
    a.querySelector("#cls-del-ok").addEventListener("click", async () => {
        const mode = a.querySelector('input[name="cls-del-mode"]:checked').value;
        let msg, body;
        if (mode === "merge") {
            const tid = parseInt(a.querySelector("#cls-del-target").value, 10);
            const t = clsByIds([tid])[0];
            msg = c.boxes ? `Delete ${c.name} and move its ${fmtNum(c.boxes)} boxes in ${fmtNum(c.images)} files to ${t.name}? Classes after it move up one ID. A backup is saved first.`
                : `Delete ${c.name}? No labels use it. Classes after it move up one ID. A backup is saved first.`;
            body = { id, boxes: "merge", merge_into: tid };
        } else {
            msg = c.boxes ? `Delete ${c.name}? Its ${fmtNum(c.boxes)} boxes in ${fmtNum(c.images)} files will be removed, and classes after it move up one ID. A backup is saved first.`
                : `Delete ${c.name}? No labels use it. Classes after it move up one ID. A backup is saved first.`;
            body = { id, boxes: "remove" };
        }
        if (!confirm(msg)) return;
        if (!(await resolveDirty())) return;
        try {
            const r = await api("/api/classes/delete", { json: body });
            remapClassIds([id], mode === "merge" ? body.merge_into : null);
            await afterClassChange(r, `Deleted class ${c.name}.`);
        } catch (e) { setMsg("classes-modal-info", "err", `Delete failed: ${escapeHtml(e.message)}`); }
    });
}

async function clsBackupsForm() {
    const a = openClsAction('<div class="field-msg muted">Loading backups…</div>');
    let list;
    try {
        const r = await api("/api/classes/backups");
        list = Array.isArray(r) ? r : (r && r.backups) || [];
    } catch (e) {
        a.innerHTML = `<div class="field-msg err">Couldn't list backups: ${escapeHtml(e.message)}</div>`;
        return;
    }
    if (!list.length) { a.innerHTML = '<div class="field-msg muted">No backups yet. One is saved automatically before every merge or delete.</div>'; return; }
    a.innerHTML = '<div class="cls-backups"></div><div class="cls-action-row"><button class="btn btn-secondary btn-sm" id="cls-bk-close">Close</button></div>';
    a.querySelector("#cls-bk-close").addEventListener("click", closeClsAction);
    const box = a.querySelector(".cls-backups");
    list.forEach(b => {
        const name = typeof b === "string" ? b : (b.name || b.file || b.filename || "");
        const when = typeof b === "object" ? (b.created || b.time || b.mtime || b.modified) : null;
        const row = document.createElement("div");
        row.className = "cls-backup-row";
        const label = when ? `${escapeHtml(name)} <span class="muted">· ${escapeHtml(typeof when === "number" ? relTime(when) : String(when))}</span>` : escapeHtml(name);
        row.innerHTML = `<span class="mono">${label}</span>`;
        const btn = document.createElement("button");
        btn.className = "btn btn-secondary btn-sm";
        btn.innerHTML = '<i class="fa-solid fa-clock-rotate-left"></i> Restore…';
        btn.addEventListener("click", async () => {
            if (!confirm(`Restore labels and the class list from ${name}? The current labels in every split are replaced by this backup. A backup of the current labels is saved first, so you can undo this.`)) return;
            if (!(await resolveDirty())) return;
            try {
                const r = await api("/api/classes/restore", { json: { backup: name.split(/[\\/]/).pop() } });
                classSlots = [0, 1, 2, 3, 4, 5, 6, 7, 8];
                await afterClassChange(r, `Restored labels and classes from ${name}.`);
            } catch (e) { setMsg("classes-modal-info", "err", `Restore failed: ${escapeHtml(e.message)}`); }
        });
        row.appendChild(btn);
        box.appendChild(row);
    });
}

/** Refreshes everything that depends on class ids after a class operation. */
async function afterClassChange(r, msg) {
    if (r && Array.isArray(r.classes)) setClasses(r.classes);
    else { const c = await api("/api/classes").catch(() => null); if (c && c.classes) setClasses(c.classes); }
    saveClassPrefs();
    clsSelected.clear();
    closeClsAction();
    let detail = "";
    if (r && r.changed_boxes != null) detail = ` ${fmtNum(r.changed_boxes)} boxes in ${fmtNum(r.changed_files)} files changed.`;
    if (r && r.backup) detail += ` Backup: ${r.backup}`;
    setMsg("classes-modal-info", "ok", escapeHtml(msg + detail));
    showToast(msg, "ok");
    await refreshClassesManager();
    await reloadCurrentLabels();
    if (appMode === "grid") loadGridPage(gridPage);
    if ($("panel-split").classList.contains("active")) updateClassChart();
    loadModelClasses(false);
}

/** Re-reads the current image's labels from disk without resetting zoom (ids may have shifted). */
async function reloadCurrentLabels() {
    if (!current.name || !annotatorOpen) return;
    const target = { ...current };
    try {
        const data = await api(`/api/labels/${target.split}/${encodeURIComponent(target.name)}`);
        if (current.name !== target.name) return;
        boxes = normalizeShapes(data);
        selectedBoxes = [];
        cmdManager.clear();
        dirty = false;
        editVersion++;
        setSaveState("saved");
        updateBoxPanel();
        renderCanvas();
    } catch (e) { /* the next navigation reloads */ }
}

function bindClassesManager() {
    ["btn-edit-classes-setup", "btn-edit-classes-panel", "btn-manage-classes-chart", "btn-grid-manage-classes", "btn-edit-classes-top"].forEach(id => {
        const el = $(id);
        if (el) el.addEventListener("click", openClassesManager);
    });
    $("btn-cls-add").addEventListener("click", clsAddForm);
    $("btn-cls-merge").addEventListener("click", clsMergeForm);
    $("btn-cls-delete").addEventListener("click", clsDeleteForm);
    $("btn-cls-backups").addEventListener("click", clsBackupsForm);
    $("cls-check-all").addEventListener("change", e => {
        clsSelected.clear();
        if (e.target.checked) (clsUsage.classes || []).forEach(c => clsSelected.add(c.id));
        renderClsTable();
    });
}

function relTime(epochSec) {
    const s = Math.max(0, Date.now() / 1000 - epochSec);
    if (s < 60) return "just now";
    if (s < 3600) return `${Math.floor(s / 60)} min ago`;
    if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
    if (s < 86400 * 30) return `${Math.floor(s / 86400)} d ago`;
    return new Date(epochSec * 1000).toLocaleDateString();
}

// ─── Model class picker (inline popover, mirrors the Data Capture Tool) ───────
const MC_MAX_CHIPS = 10;
const mc = { loaded: false, modelPath: "", names: [], defaults: [], kind: "custom", map: new Map(), seq: 0 };

function mcDefaultTarget(modelName) {
    const hit = CLASS_NAMES.find(c => c.toLowerCase() === String(modelName).toLowerCase());
    return hit || modelName;
}

function mcApplyResponse(r, showWarnings) {
    if (!r) return;
    mc.loaded = !!r.loaded;
    mc.modelPath = r.model_path || "";
    mc.names = r.model_classes || [];
    mc.defaults = r.default || [];
    mc.kind = r.kind || (mc.names.includes("person") && mc.names.includes("traffic light") ? "coco" : "custom");
    mc.map = new Map((r.mapping || []).map(m => [Number(m.model_idx), m.target]));
    if (Array.isArray(r.classes)) setClasses(r.classes);
    if (showWarnings && r.warnings && r.warnings.length) showToast(r.warnings.join("; "), "err");
    mcRender();
}

async function loadModelClasses(showWarnings = true) {
    try {
        mcApplyResponse(await api("/api/model_classes"), showWarnings);
    } catch (e) {
        mc.loaded = false;
        mcRender();
    }
}

const mcPush = debounce(async () => {
    const mapping = [...mc.map.entries()].sort((a, b) => a[0] - b[0]).map(([i, t]) => ({ model_idx: i, target: t }));
    if (mapping.length === 0) return;
    const sentSeq = mc.seq;
    try {
        const r = await api("/api/model_classes", { json: { mapping } });
        if (Array.isArray(r.classes)) setClasses(r.classes);
        if (r.warnings && r.warnings.length) showToast(r.warnings.join("; "), "err");
        // Only adopt the server's mapping if nothing changed locally since this request was sent.
        const typing = document.activeElement && $("mc-list").contains(document.activeElement) && document.activeElement.tagName === "INPUT";
        if (sentSeq === mc.seq && !typing) { mc.map = new Map((r.mapping || []).map(m => [Number(m.model_idx), m.target])); mcRender(); }
        else mcRenderChips();
    } catch (e) {
        showToast("Couldn't update classes to auto-label: " + e.message, "err");
        loadModelClasses(false);
    }
}, 300);

function mcToggle(idx, on) {
    mc.seq++;
    if (on) mc.map.set(idx, mcDefaultTarget(mc.names[idx]));
    else {
        if (mc.map.size <= 1) { showToast("Keep at least one class selected", "err"); mcRender(); return; }
        mc.map.delete(idx);
    }
    mcRender();
    mcPush();
}

function mcPreset(kind) {
    mc.seq++;
    const names = kind === "all" ? mc.names : mc.defaults;
    const next = new Map();
    mc.names.forEach((n, i) => { if (names.includes(n)) next.set(i, mc.map.get(i) || mcDefaultTarget(n)); });
    if (next.size === 0) { showToast("This model has no default classes. Pick them from the list.", "err"); return; }
    mc.map = next;
    mcRender();
    mcPush();
}

function mcSetTarget(idx, target) {
    mc.seq++;
    target = String(target || "").trim();
    if (!target) { showToast("Type a class name, or pick one from the list", "err"); mcRender(); return; }
    mc.map.set(idx, target);
    mcRender();
    mcPush();
}

function mcRenderChips() {
    const chipsEl = $("mc-chips");
    const sel = [...mc.map.keys()].sort((a, b) => a - b);
    chipsEl.innerHTML = "";
    sel.slice(0, MC_MAX_CHIPS).forEach(i => {
        const n = mc.names[i] || `#${i}`;
        const t = mc.map.get(i);
        const chip = document.createElement("span");
        chip.className = "chip-x";
        chip.textContent = t && t.toLowerCase() !== String(n).toLowerCase() ? `${n} → ${t}` : n;
        const x = document.createElement("button");
        x.type = "button";
        x.title = `Remove ${n}`;
        x.setAttribute("aria-label", `Remove ${n}`);
        x.innerHTML = '<i class="fa-solid fa-xmark"></i>';
        x.addEventListener("click", e => { e.stopPropagation(); mcToggle(i, false); });
        chip.appendChild(x);
        chipsEl.appendChild(chip);
    });
    if (sel.length > MC_MAX_CHIPS) {
        const more = document.createElement("span");
        more.className = "chip-more";
        more.textContent = `+${sel.length - MC_MAX_CHIPS} more`;
        chipsEl.appendChild(more);
    }
    const label = mc.loaded ? `Choose classes (${sel.length} of ${mc.names.length})` : "Choose classes";
    $("mc-btn-text").textContent = label;
    $("mc-panel-text").textContent = label;
    $("mc-btn").disabled = !mc.loaded;
    // Summary: N model classes -> M dataset classes
    const targets = [...new Set(sel.map(i => mc.map.get(i)))];
    $("mc-summary").textContent = sel.length
        ? `${sel.length} model ${sel.length === 1 ? "class" : "classes"} → ${targets.length} dataset ${targets.length === 1 ? "class" : "classes"}: ${targets.join(", ")}`
        : "";
}

function mcRender() {
    const info = $("mc-info");
    if (!mc.loaded) {
        setMsg(info, "muted", "Load a model to choose which of its classes are labelled.");
    } else {
        setMsg(info, "muted", `${mc.kind === "coco" ? "COCO" : "Custom"} · ${mc.names.length} classes`);
    }
    $("mc-preset-default-text").textContent = mc.kind === "coco" ? "Vehicles" : "Model defaults";
    mcRenderChips();
    mcRenderList();
}

function mcRenderList() {
    const list = $("mc-list");
    if ($("mc-pop").hidden) { list.innerHTML = ""; return; }
    const q = $("mc-search").value.trim().toLowerCase();
    list.innerHTML = "";
    const rows = mc.names.map((n, i) => [n, i]).filter(([n]) => n.toLowerCase().includes(q));
    if (rows.length === 0) {
        list.innerHTML = '<div class="mc-empty">No classes match your search.</div>';
        return;
    }
    rows.forEach(([n, i]) => {
        const row = document.createElement("div");
        row.className = "mc-row" + (mc.map.has(i) ? " checked" : "");
        const lab = document.createElement("label");
        lab.className = "mc-check";
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = mc.map.has(i);
        cb.addEventListener("change", () => mcToggle(i, cb.checked));
        lab.append(cb);
        const idx = document.createElement("span"); idx.className = "mc-idx"; idx.textContent = i;
        const nm = document.createElement("span"); nm.className = "mc-name"; nm.textContent = n;
        lab.append(idx, nm);
        row.appendChild(lab);
        if (mc.map.has(i)) row.appendChild(mcTargetControl(i, n));
        list.appendChild(row);
    });
}

function mcTargetControl(i, modelName) {
    const target = mc.map.get(i);
    const wrap = document.createElement("span");
    wrap.className = "mc-target";
    const sel = document.createElement("select");
    sel.setAttribute("aria-label", `Save ${modelName} as`);
    sel.title = "Save as";
    const lower = CLASS_NAMES.map(c => c.toLowerCase());
    CLASS_NAMES.forEach(c => { const o = document.createElement("option"); o.value = "c:" + c; o.textContent = c; sel.appendChild(o); });
    const news = [];
    if (!lower.includes(String(modelName).toLowerCase())) news.push(modelName);
    if (target && !lower.includes(target.toLowerCase()) && !news.includes(target)) news.push(target);
    news.forEach(nm => { const o = document.createElement("option"); o.value = "n:" + nm; o.textContent = `New: ${nm}`; sel.appendChild(o); });
    const oc = document.createElement("option"); oc.value = "custom"; oc.textContent = "Custom name…"; sel.appendChild(oc);
    const exist = CLASS_NAMES.find(c => c.toLowerCase() === String(target).toLowerCase());
    sel.value = exist ? "c:" + exist : "n:" + target;
    sel.addEventListener("change", () => {
        if (sel.value === "custom") {
            const inp = document.createElement("input");
            inp.type = "text";
            inp.placeholder = "New class name";
            inp.className = "mc-custom";
            inp.setAttribute("aria-label", `New class name for ${modelName}`);
            let done = false;
            const commit = () => { if (done) return; done = true; if (inp.value.trim()) mcSetTarget(i, inp.value); else mcRender(); };
            inp.addEventListener("keydown", e => {
                if (e.key === "Enter") { e.preventDefault(); commit(); }
                if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); done = true; mcRender(); }
            });
            inp.addEventListener("blur", commit);
            wrap.replaceChildren(inp);
            inp.focus();
        } else {
            mcSetTarget(i, sel.value.slice(2));
        }
    });
    wrap.appendChild(sel);
    return wrap;
}

let _mcOpener = null;
function mcOpen(anchorId, opener) {
    if (!mc.loaded) { showToast("Load a model in Setup first, then choose its classes", "err"); return; }
    const pop = $("mc-pop");
    $(anchorId).appendChild(pop);
    pop.hidden = false;
    _mcOpener = opener;
    opener.setAttribute("aria-expanded", "true");
    $("mc-search").value = "";
    mcRenderList();
    $("mc-search").focus();
}
function mcClose() {
    const pop = $("mc-pop");
    if (pop.hidden) return false;
    pop.hidden = true;
    if (_mcOpener) _mcOpener.setAttribute("aria-expanded", "false");
    $("mc-list").innerHTML = "";
    return true;
}
function bindModelClassPicker() {
    $("mc-btn").addEventListener("click", e => {
        e.stopPropagation();
        if (!$("mc-pop").hidden) mcClose(); else mcOpen("mc-anchor-setup", $("mc-btn"));
    });
    $("btn-choose-classes-panel").addEventListener("click", e => {
        e.stopPropagation();
        if (!$("mc-pop").hidden) mcClose(); else mcOpen("mc-anchor-panel", $("btn-choose-classes-panel"));
    });
    $("mc-pop").addEventListener("click", e => e.stopPropagation());
    $("mc-search").addEventListener("input", mcRenderList);
    document.querySelectorAll("#mc-pop [data-mc-preset]").forEach(b => b.addEventListener("click", () => mcPreset(b.dataset.mcPreset)));
}

// ═══ 7. SETUP SCREEN ══════════════════════════════════════════════════════════
const MODEL_PRESETS = {
    "YOLO26": ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"],
    "YOLO11": ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
    "YOLOv8": ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
};

/** Native folder picker via the backend; falls back to a text prompt when no dialog is available. */
async function browseFolder(promptText, current) {
    let res;
    try { res = await api("/api/browse_folder"); } catch (e) { res = { error: e.message }; }
    if (res && res.path) return res.path;
    if (res && res.error) {
        const p = prompt(`${promptText}\n\nThe folder picker isn't available, so paste the full path here.`, current || "");
        return p ? p.trim() : "";
    }
    return "";
}

// ─── Home: datasets rail, "Open existing" / "Create new", independent loading ─
// Every section (datasets, open dataset state, local models, devices, settings, model
// classes) loads on its own with its own loading/error/retry state, so one slow or hung
// request (e.g. a dataset locked by a frozen copy of the app) never blanks the screen.
let homeDatasets = [];       // rows from /api/datasets?stats=1 plus extras
let homeSelected = null;     // path of the selected dataset
let homeActivePath = null;   // path of the dataset the server has open
let homeBaseDir = "";        // BASE_DIR from /api/state.base_dir
let homeMode = null;         // "open" | "create"
let homeModeChosen = false;  // the user picked a mode in this session
let homePreferActive = true; // each visit starts on the open dataset
const homeExtras = new Map(); // normPath -> dataset outside the list (other folder, open elsewhere)
const homeLoad = { datasets: "idle", seq: 0, error: "", slow: false };
const normPath = p => String(p || "").replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();
const HOME_LOCK_REASON = "Another program is holding its database open, usually an old copy of this app that froze.";
const baseName = p => String(p || "").replace(/[\\/]+$/, "").split(/[\\/]/).pop() || String(p || "");
const sentence = m => { m = String(m || "Unknown error").trim(); return /[.!?]$/.test(m) ? m : m + "."; };
const retryLink = what => `<button type="button" class="link-btn" data-home-retry="${what}">Retry</button>`;

function selectedHomeDataset() {
    return homeDatasets.find(x => normPath(x.path) === normPath(homeSelected)) || null;
}

function setHomeMode(mode, opts = {}) {
    homeMode = mode === "create" ? "create" : "open";
    if (opts.user) homeModeChosen = true;
    [["choice-open", "open"], ["choice-create", "create"]].forEach(([id, m]) => {
        const b = $(id);
        b.setAttribute("aria-pressed", String(homeMode === m));
        b.classList.toggle("active", homeMode === m);
    });
    show("home-open-panel", homeMode === "open");
    show("new-dataset-card", homeMode === "create");
    if (opts.focus) { const el = $(opts.focus); if (el) el.focus(); }
}

function homeBaseLabel() {
    if (homeBaseDir) return homeBaseDir;
    // Older backend without base_dir: the most common parent folder of the listed datasets.
    const counts = {};
    homeDatasets.filter(d => !d.extra).forEach(d => {
        const parent = String(d.path).replace(/[\\/]+$/, "").replace(/[\\/][^\\/]*$/, "");
        counts[parent] = (counts[parent] || 0) + 1;
    });
    return Object.keys(counts).sort((a, b) => counts[b] - counts[a])[0] || "";
}

function renderBaseHint() {
    const base = homeBaseLabel();
    $("home-base-hint").innerHTML = base
        ? `The Datasets list shows every dataset in <span class="mono">${escapeHtml(base)}</span>. For a dataset stored somewhere else, choose its folder.`
        : "The Datasets list shows every dataset in the datasets folder. For a dataset stored somewhere else, choose its folder.";
}

/** Adds datasets that aren't in the server list (another folder, or the open dataset) back in. */
function mergeHomeExtras(list) {
    const have = new Set(list.map(d => normPath(d.path)));
    homeExtras.forEach((d, key) => { if (have.has(key)) homeExtras.delete(key); });
    return list.concat([...homeExtras.values()]);
}

function pickHomeSelection() {
    const has = p => p && homeDatasets.some(d => normPath(d.path) === normPath(p));
    if (homePreferActive && has(homeActivePath)) { homeSelected = homeActivePath; homePreferActive = false; return; }
    if (has(homeSelected)) return;
    const recent = homeDatasets.filter(d => !d.extra).sort((a, b) => (b.modified || 0) - (a.modified || 0))[0] || homeDatasets[0];
    homeSelected = recent ? recent.path : null;
}

function renderHome() {
    renderHomeDatasets();
    renderContinueCard();
    renderBaseHint();
}

function homeSkeletonRows(n) {
    let html = "";
    for (let i = 0; i < n; i++) html += '<div class="home-skel-row" aria-hidden="true"><div class="skel-line w60"></div><div class="skel-line w90 thin"></div><div class="skel-line w40 thin"></div></div>';
    return html;
}

function renderHomeDatasets() {
    const list = $("home-ds-list");
    const q = $("home-ds-search").value.trim().toLowerCase();
    const loading = homeLoad.datasets === "idle" || homeLoad.datasets === "loading";
    list.setAttribute("aria-busy", String(loading && !homeDatasets.length));
    list.innerHTML = "";
    if (!homeDatasets.length) {
        if (loading) {
            list.innerHTML = `<div class="home-list-note">${homeLoad.slow ? "Still loading datasets…" : "Loading datasets…"}</div>` + homeSkeletonRows(3);
        } else if (homeLoad.datasets === "error") {
            list.innerHTML = `<div class="home-list-note err"><i class="fa-solid fa-triangle-exclamation"></i> Couldn't load datasets. ${retryLink("datasets")}</div>`;
        } else {
            list.innerHTML = '<div class="home-empty">No datasets yet. <button type="button" class="link-btn" data-home-mode-go="create">Create one from a folder of images</button> <i class="fa-solid fa-arrow-right"></i></div>';
        }
        return;
    }
    if (homeLoad.datasets === "error") {
        list.insertAdjacentHTML("beforeend", `<div class="home-list-note err"><i class="fa-solid fa-triangle-exclamation"></i> Couldn't refresh the list. ${retryLink("datasets")}</div>`);
    }
    const rows = homeDatasets.filter(d => !q || d.name.toLowerCase().includes(q));
    if (!rows.length) { list.insertAdjacentHTML("beforeend", '<div class="home-empty">No dataset matches your search.</div>'); return; }
    rows.forEach(d => {
        const known = d.total != null;
        const total = d.total || 0, rev = d.reviewed || 0;
        const pct = total ? Math.round(rev / total * 100) : 0;
        const row = document.createElement("div");
        const isSel = normPath(d.path) === normPath(homeSelected);
        const isActive = normPath(d.path) === normPath(homeActivePath);
        row.className = "home-ds-row" + (isSel ? " sel" : "") + (isActive ? " active" : "") + (d.locked ? " locked" : "");
        row.tabIndex = 0;
        row.setAttribute("role", "option");
        row.setAttribute("aria-selected", String(isSel));
        row.title = d.locked ? `${d.path}\nLocked by another program` : d.path;
        const meta = [known ? `${fmtNum(total)} images` : "Other folder"];
        if (d.modified) meta.push(relTime(d.modified));
        if (isActive) meta.push("open");
        row.innerHTML = `<div class="home-ds-top">
              <i class="fa-solid ${d.type === "polygon" ? "fa-draw-polygon" : "fa-vector-square"} home-ds-type" title="${d.type === "polygon" ? "Polygon" : "Bounding box"} dataset"></i>
              <span class="home-ds-name">${escapeHtml(d.name)}</span>
              ${d.locked ? '<span class="home-ds-badge locked"><i class="fa-solid fa-lock" aria-hidden="true"></i> Locked</span>' : ""}
              ${known ? `<span class="home-ds-pct">${pct}%</span>` : ""}</div>
            ${known ? `<div class="list-progress-track"><div class="list-progress-bar" style="width:${pct}%"></div></div>` : ""}
            <div class="home-ds-meta">${meta.join(" · ")}</div>`;
        row.addEventListener("click", () => selectHomeDataset(d.path));
        row.addEventListener("dblclick", () => { selectHomeDataset(d.path); openSelectedDataset("annotate"); });
        row.addEventListener("keydown", e => {
            if (e.key === "Enter") { e.preventDefault(); e.stopPropagation(); selectHomeDataset(d.path); openSelectedDataset("annotate"); }
            if (e.key === "ArrowDown" || e.key === "ArrowUp") {
                e.preventDefault();
                const step = el => (e.key === "ArrowDown" ? el.nextElementSibling : el.previousElementSibling);
                let sib = step(row);
                while (sib && !sib.classList.contains("home-ds-row")) sib = step(sib);
                if (sib) sib.focus();
            }
        });
        list.appendChild(row);
    });
}

function selectHomeDataset(path) {
    homeSelected = path;
    if (homeMode !== "open") setHomeMode("open", { user: true });
    renderHomeDatasets();
    renderContinueCard();
}

function homeCanOpenSelected() {
    const d = selectedHomeDataset();
    return !!d && !d.locked && homeMode === "open" && !$("continue-card").classList.contains("d-none");
}

function renderContinueCard() {
    const d = selectedHomeDataset();
    const stateEl = $("home-open-state");
    show("continue-card", !!d);
    if (!d) {
        let html;
        if (homeLoad.datasets === "error") {
            html = `<div class="home-state err"><i class="fa-solid fa-triangle-exclamation home-state-icon" aria-hidden="true"></i><div>
                <div class="home-state-title">Couldn't load the datasets list</div>
                <p class="home-state-text">${escapeHtml(sentence(homeLoad.error))}</p>
                <button type="button" class="btn btn-secondary btn-sm" data-home-retry="datasets"><i class="fa-solid fa-rotate-right"></i> Retry</button></div></div>`;
        } else if (homeLoad.datasets !== "ok") {
            html = `<div class="home-skel" aria-busy="true"><div class="skel-line w40 tall"></div><div class="skel-line w60"></div><div class="skel-line w90 thin"></div>
                <p class="field-msg muted">${homeLoad.slow
                    ? `Still loading datasets… Large datasets take a moment, or one may be locked by another program. ${retryLink("datasets")}`
                    : "Loading datasets…"}</p></div>`;
        } else if (!homeDatasets.length) {
            const base = homeBaseLabel();
            html = `<div class="home-state"><i class="fa-solid fa-database home-state-icon" aria-hidden="true"></i><div>
                <div class="home-state-title">No datasets found</div>
                <p class="home-state-text">There are no datasets in ${base ? `<span class="mono">${escapeHtml(base)}</span>` : "the datasets folder"} yet. Create one from a folder of images, or choose a dataset folder stored somewhere else.</p>
                <button type="button" class="btn btn-primary btn-sm" data-home-mode-go="create"><i class="fa-solid fa-plus"></i> Create new dataset</button></div></div>`;
        } else {
            html = '<p class="field-msg muted">Select a dataset in the list to open it.</p>';
        }
        stateEl.innerHTML = html;
        show(stateEl, true);
        return;
    }
    stateEl.innerHTML = "";
    show(stateEl, false);
    const known = d.total != null;
    const total = d.total || 0, rev = d.reviewed || 0;
    const pct = total ? Math.round(rev / total * 100) : 0;
    const isActive = normPath(d.path) === normPath(homeActivePath);
    const locked = !!d.locked;
    $("continue-name").textContent = d.name;
    const metaBits = [];
    if (d.type || known) metaBits.push(`<i class="fa-solid ${d.type === "polygon" ? "fa-draw-polygon" : "fa-vector-square"}"></i> ${d.type === "polygon" ? "Polygon" : "Bounding box"}`);
    if (d.classes != null) metaBits.push(`${fmtNum(d.classes)} ${d.classes === 1 ? "class" : "classes"}`);
    if (isActive) metaBits.push('<span class="home-open-now">Open now</span>');
    $("continue-meta").innerHTML = metaBits.join(" · ");
    show($("continue-count").parentElement, known);
    $("continue-count").textContent = `${fmtNum(rev)} of ${fmtNum(total)} reviewed`;
    $("continue-bar").style.width = pct + "%";
    $("continue-status").textContent = `${d.path}${d.modified ? ` · changed ${relTime(d.modified)}` : ""}`;
    $("continue-status").title = d.path;
    show("continue-locked", locked);
    $("continue-locked-reason").textContent = HOME_LOCK_REASON;
    ["btn-continue", "btn-view-dataset"].forEach(id => { $(id).disabled = locked; });
    $("btn-continue").title = locked ? "Locked by another program. Close it, then click Retry." : "Open annotator (Enter)";
}

/** Marks a dataset locked (409 from load_existing_dataset) and shows the fix. */
function markHomeLocked(path) {
    let d = homeDatasets.find(x => normPath(x.path) === normPath(path));
    if (!d) {
        d = { name: baseName(path), path, locked: true, extra: true };
        homeExtras.set(normPath(path), d);
        homeDatasets = homeDatasets.concat([d]);
    }
    d.locked = true;
    homeSelected = d.path;
    setHomeMode("open");
    renderHomeDatasets();
    renderContinueCard();
}

const isLockedError = e => !!(e && e.status === 409 && e.data && e.data.locked);

async function openSelectedDataset(mode) {
    const d = selectedHomeDataset();
    if (!d || d.locked) return;
    $("btn-continue").disabled = true;
    $("btn-view-dataset").disabled = true;
    try {
        if (normPath(d.path) !== normPath(homeActivePath)) {
            $("continue-status").textContent = "Opening…";
            await api("/api/load_existing_dataset", { json: { dataset_dir: d.path } });
            homeActivePath = d.path;
        }
        await openAnnotator({ mode });
        renderContinueCard();
    } catch (e) {
        if (isLockedError(e)) { markHomeLocked(d.path); $("btn-lock-retry").focus(); return; }
        renderContinueCard();
        $("continue-status").textContent = `Couldn't open the dataset: ${e.message}`;
        showToast(`Couldn't open the dataset: ${e.message}`, "err");
    }
}

/** Opens a dataset folder chosen by path (not necessarily in the list). */
async function openFolderPath(p) {
    try {
        await api("/api/load_existing_dataset", { json: { dataset_dir: p } });
    } catch (e) {
        if (isLockedError(e)) { markHomeLocked(p); $("btn-lock-retry").focus(); return false; }
        if (e.status === 400 && /state\.db/i.test(e.message)) {
            showToast("That folder has no state.db, so it isn't a dataset made by this tool. To use its images, create a new dataset from it.", "err");
            setHomeMode("create", { user: true });
            $("src-input").value = p;
            validateSourcePath(p, "src-info", "btn-ingest");
        } else {
            showToast(`Couldn't open the dataset: ${e.message}`, "err");
        }
        return false;
    }
    homeActivePath = p;
    const ex = homeExtras.get(normPath(p));
    if (ex) ex.locked = false;
    await openAnnotator({ mode: "annotate" });
    return true;
}

async function openOtherFolder() {
    const p = await browseFolder("Dataset folder (it must contain state.db):", "");
    if (p) await openFolderPath(p);
}

/** Retry for a locked dataset: re-check the list (or re-try opening a folder outside it). */
async function retryLockedDataset() {
    const d = selectedHomeDataset();
    if (!d) { await loadHomeDatasets({ refresh: true }); return; }
    const btn = $("btn-lock-retry");
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Checking…';
    try {
        if (d.extra) {
            await openFolderPath(d.path);
        } else {
            await loadHomeDatasets({ refresh: true });
            const nd = selectedHomeDataset();
            if (nd && nd.locked) showToast(`${nd.name} is still locked. Close the other program first.`, "err");
            else if (nd) { showToast(`${nd.name} is free now`, "ok"); $("btn-continue").focus(); }
        }
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-rotate-right"></i> Retry';
    }
}

async function loadHomeDatasets(opts = {}) {
    const seq = ++homeLoad.seq;
    homeLoad.datasets = "loading";
    homeLoad.slow = false;
    if (!homeDatasets.length) { renderHomeDatasets(); renderContinueCard(); }
    const slowTimer = setTimeout(() => {
        if (seq !== homeLoad.seq || homeLoad.datasets !== "loading") return;
        homeLoad.slow = true;
        if (!homeDatasets.length) { renderHomeDatasets(); renderContinueCard(); }
    }, 4000);
    let ok = false;
    try {
        const data = await api("/api/datasets?stats=1" + (opts.refresh ? "&refresh=1" : ""), { timeout: 45000 });
        if (seq !== homeLoad.seq) return;
        homeDatasets = mergeHomeExtras(Array.isArray(data) ? data : []);
        homeLoad.datasets = "ok";
        homeLoad.error = "";
        ok = true;
    } catch (e) {
        if (seq !== homeLoad.seq) return;
        homeLoad.datasets = "error";
        homeLoad.error = e.message;
    } finally {
        clearTimeout(slowTimer);
    }
    const act = homeDatasets.find(d => d.active && !d.extra);
    if (act) homeActivePath = act.path;
    pickHomeSelection();
    // Default choice: open an existing dataset when there is one, else create a new one.
    const typing = document.activeElement && $("new-dataset-card").contains(document.activeElement);
    if (ok && !homeModeChosen && !typing) setHomeMode(homeDatasets.length ? "open" : "create");
    renderHome();
}

async function loadHomeState() {
    show("classes-msg", false);
    let st;
    try {
        st = await api("/api/state", { timeout: 20000 });
    } catch (e) {
        setMsg("classes-msg", "err", `Couldn't read the open dataset: ${escapeHtml(sentence(e.message))} ${retryLink("state")}`);
        return;
    }
    if (st.base_dir) homeBaseDir = st.base_dir;
    if (st.ready) {
        datasetType = st.dataset_type || "bbox";
        datasetName = st.dataset_name || "";
    }
    if (Array.isArray(st.classes)) setClasses(st.classes);
    else { try { const r = await api("/api/classes", { timeout: 20000 }); setClasses(r.classes || []); } catch (e) { /* keep */ } }
    if (st.ready && st.dataset_dir) {
        homeActivePath = st.dataset_dir;
        const key = normPath(st.dataset_dir);
        if (homeLoad.datasets === "ok" && !homeDatasets.some(d => normPath(d.path) === key)) {
            // The open dataset lives outside the datasets folder: list it anyway.
            const sum = k => ["train", "valid", "test"].reduce((a, s) => a + (((st.counts || {})[s] || {})[k] || 0), 0);
            const extra = {
                name: st.dataset_name || baseName(st.dataset_dir), path: st.dataset_dir, type: st.dataset_type || "bbox",
                total: sum("total"), reviewed: sum("reviewed"), annotated: sum("annotated"),
                classes: Array.isArray(st.classes) ? st.classes.length : null, active: true, extra: true,
            };
            homeExtras.set(key, extra);
            homeDatasets = homeDatasets.concat([extra]);
        }
        if (homePreferActive) pickHomeSelection();
    }
    renderHome();
}

function fillModelPresets() {
    const mpre = $("model-preset");
    const keep = mpre.value;
    mpre.innerHTML = '<option value="">Choose a pre-trained model…</option>';
    Object.entries(MODEL_PRESETS).forEach(([label, models]) => {
        const grp = document.createElement("optgroup");
        grp.label = label;
        models.forEach(m => { const o = document.createElement("option"); o.value = m; o.textContent = m; grp.appendChild(o); });
        mpre.appendChild(grp);
    });
    if (keep) mpre.value = keep;
}

async function loadHomeModels() {
    const msel = $("model-select");
    const cur = msel.selectedOptions[0];
    const keepOpt = cur && msel.value ? { value: msel.value, text: cur.textContent } : null;
    const firstLoad = msel.options.length <= 1;
    if (firstLoad) { msel.innerHTML = '<option value="">Loading local models…</option>'; msel.disabled = true; }
    show("models-msg", false);
    let mdata;
    try {
        mdata = await api("/api/models", { timeout: 20000 });
    } catch (e) {
        msel.disabled = false;
        if (firstLoad) msel.innerHTML = '<option value="">Local models unavailable</option>';
        setMsg("models-msg", "err", `Couldn't list local models: ${escapeHtml(sentence(e.message))} ${retryLink("models")}`);
        return;
    }
    const files = (mdata && mdata.files) || [];
    msel.innerHTML = `<option value="">${files.length ? "Choose a local model…" : "No local models found"}</option>`;
    const byExt = {};
    files.forEach(f => { const ext = f.split(".").pop().toLowerCase(); (byExt[ext] = byExt[ext] || []).push(f); });
    Object.entries(byExt).forEach(([ext, list]) => {
        const grp = document.createElement("optgroup");
        grp.label = `.${ext} files`;
        list.forEach(f => { const o = document.createElement("option"); o.value = f; o.textContent = f; grp.appendChild(o); });
        msel.appendChild(grp);
    });
    if (keepOpt) {
        if (![...msel.options].some(o => o.value === keepOpt.value)) {
            const o = document.createElement("option");
            o.value = keepOpt.value;
            o.textContent = keepOpt.text;
            msel.insertBefore(o, msel.options[1] || null);
        }
        msel.value = keepOpt.value;
    }
    msel.disabled = false;

    const csel = $("tr-custom");
    const keepCustom = csel.value;
    csel.innerHTML = '<option value="">None (use base model)</option>';
    files.forEach(f => { const o = document.createElement("option"); o.value = f; o.textContent = f; csel.appendChild(o); });
    if (keepCustom) csel.value = keepCustom;
}

async function loadHomeDevices() {
    const devEl = $("device-autodetect");
    setMsg(devEl, "muted", "Checking for GPUs…");
    let devInfo;
    try {
        devInfo = (await api("/api/device_info", { timeout: 60000 })) || {};
    } catch (e) {
        setMsg(devEl, "err", `Couldn't check for GPUs: ${escapeHtml(sentence(e.message))} Auto still works. ${retryLink("devices")}`);
        return;
    }
    const dsel = $("device-select");
    const keepDev = dsel.value || "auto";
    dsel.innerHTML = '<option value="auto">Auto (recommended)</option><option value="cpu">CPU</option>';
    (devInfo.cuda || []).forEach(g => {
        const o = document.createElement("option");
        o.value = String(g.id);
        o.textContent = `GPU ${g.id}: ${g.name}`;
        dsel.appendChild(o);
    });
    if (devInfo.mps) { const o = document.createElement("option"); o.value = "mps"; o.textContent = "Apple Silicon (MPS)"; dsel.appendChild(o); }
    dsel.value = [...dsel.options].some(o => o.value === keepDev) ? keepDev : "auto";
    if (devInfo.cuda && devInfo.cuda.length) {
        const names = devInfo.cuda.map(g => `GPU ${g.id}: ${escapeHtml(g.name)} (${g.free_gb} of ${g.total_gb} GB free)`).join(", ");
        setMsg(devEl, "ok", `Found ${names}. Auto uses the GPU.`);
    } else if (devInfo.mps) {
        setMsg(devEl, "ok", "Found Apple Silicon (MPS). Auto uses it.");
    } else {
        setMsg(devEl, "muted", "No GPU found. Inference runs on the CPU, which is slower but works.");
    }
    updateDeviceChip(devInfo.selected);
}

const HOME_LOADERS = {
    datasets: () => loadHomeDatasets({ refresh: true }),
    state: () => loadHomeState(),
    models: () => loadHomeModels(),
    devices: () => loadHomeDevices(),
};

/** Refreshes setup data. Listeners are bound once in bindSetup(). Never blocks on one slow call. */
function initSetup() {
    $("mc-anchor-setup").appendChild($("mc-pop"));
    mcClose();
    fillModelPresets();
    homePreferActive = true;
    if (!homeMode) setHomeMode("open");
    renderHome();
    return Promise.allSettled([
        loadHomeDatasets(),
        loadHomeState(),
        loadHomeModels(),
        loadHomeDevices(),
        loadSettings(),
        loadModelClasses(true),
    ]);
}

function updateDeviceChip(dev) {
    if (dev == null) return;
    dev = String(dev);
    const isMps = dev === "mps";
    const isGpu = dev !== "cpu" && !isMps;
    ["device-chip", "home-device-chip"].forEach(id => {
        const chip = $(id);
        if (!chip) return;
        chip.textContent = isGpu ? `GPU ${dev.replace(/^cuda:?/, "")}` : isMps ? "MPS" : "CPU";
        chip.className = "device-chip " + (isGpu || isMps ? "gpu" : "cpu");
    });
}

async function validateSourcePath(path, infoId, btnId) {
    const info = $(infoId), btn = $(btnId);
    path = (path || "").trim();
    if (!path) { show(info, false); btn.disabled = true; return false; }
    setMsg(info, "muted", "Scanning folder…");
    try {
        const r = await api(`/api/source_info?path=${encodeURIComponent(path)}`);
        if (r.valid && r.count > 0) {
            setMsg(info, "ok", `Found ${r.count} images.`);
            btn.disabled = false;
            return true;
        }
        if (r.valid) setMsg(info, "warn", "No supported images (.jpg, .png, .bmp, .webp) in this folder. Choose a folder that contains images.");
        else setMsg(info, "err", "Can't read this folder. Check the path and that you have access to it.");
    } catch (e) {
        setMsg(info, "err", escapeHtml(e.message));
    }
    btn.disabled = true;
    return false;
}

/** Follows a /api/progress SSE stream. */
function followProgress(jobId, { onTick, onDone, onError }) {
    const es = new EventSource(`/api/progress/${encodeURIComponent(jobId)}`);
    let finished = false;
    es.onmessage = e => {
        let p;
        try { p = JSON.parse(e.data); } catch (err) { return; }
        const st = String(p.status || "");
        if (st === "done") { finished = true; es.close(); if (onDone) onDone(p); return; }
        if (st.startsWith("error") || st === "unknown") {
            finished = true; es.close();
            if (onError) onError(st.replace(/^error:?\s*/, "") || "unknown job");
            return;
        }
        if (onTick) onTick(p);
    };
    es.onerror = () => {
        if (finished) return;
        finished = true; es.close();
        if (onError) onError("lost connection to the server");
    };
    return es;
}

function progressText(p, pct) {
    if (p.status === "linking") return `Linking images… ${pct}%`;
    if (p.status === "copying") return `Copying images… ${pct}%`;
    if (p.status === "annotating" || p.status === "running") return `Labelling… ${pct}%`;
    return p.status ? String(p.status) : `${pct}%`;
}

async function loadModel() {
    const modelPath = $("model-select").value || $("model-preset").value;
    if (!modelPath) { showToast("Choose a pre-trained or local model first", "err"); return; }
    const device = $("device-select").value;
    const btn = $("btn-load-model");
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Loading model…';
    try {
        const data = await api("/api/set_model", { json: { model_path: modelPath, device } });
        if (Array.isArray(data.classes)) setClasses(data.classes);
        updateDeviceChip(data.device);
        await Promise.all([loadModelClasses(true), loadSettings()]);
        const n = (data.model_classes || mc.names).length;
        showToast(`Model loaded on ${String(data.device).toUpperCase()} with ${n} classes`, "ok");
    } catch (e) {
        showToast(`Couldn't load the model: ${e.message}`, "err");
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-bolt"></i> Load model';
    }
}

async function browseModelFile() {
    let res;
    try { res = await api("/api/browse_model_file"); } catch (e) { res = { error: e.message }; }
    let path = res && res.path;
    if (!path && res && res.error) {
        const p = prompt("The file picker isn't available. Paste the full path to a .pt, .onnx or .engine file:", "");
        path = p ? p.trim() : "";
    }
    if (!path) return;
    const msel = $("model-select");
    $("model-preset").value = "";
    let found = [...msel.options].find(o => o.value === path || o.value === path.replace(/\\/g, "/"));
    if (!found) {
        found = document.createElement("option");
        found.value = path;
        found.textContent = `${path.split(/[\\/]/).pop()} (${path})`;
        msel.insertBefore(found, msel.options[1] || null);
    }
    msel.value = found.value;
    showToast(`Selected ${path.split(/[\\/]/).pop()}. Click Load model to use it.`);
}

function readRatios(ids) {
    const v = ids.map(id => Math.max(0, parseFloat($(id).value) || 0));
    const tot = v[0] + v[1] + v[2];
    if (tot <= 0) return null;
    return v.map(x => Math.round((x / tot) * 1000) / 1000);
}

async function startIngest() {
    const src = $("src-input").value.trim();
    if (!src) return;
    const ratios = readRatios(["r-train", "r-valid", "r-test"]);
    if (!ratios) { showToast("Set at least one split percentage above zero", "err"); return; }
    const btn = $("btn-ingest");
    btn.disabled = true;
    show("ingest-prog", true);
    $("ip-bar").style.width = "0%";
    $("ip-num").textContent = "0 / 0";
    $("ip-status").textContent = "Starting…";
    let job;
    try {
        job = await api("/api/ingest", { json: { source_dir: src, dataset_type: $("dataset-type-select").value, split_ratios: ratios } });
    } catch (e) {
        $("ip-status").textContent = `Couldn't start the import: ${e.message}`;
        btn.disabled = false;
        return;
    }
    followProgress(job.job_id, {
        onTick: p => {
            const pct = p.total > 0 ? Math.round(p.done / p.total * 100) : 0;
            $("ip-bar").style.width = pct + "%";
            $("ip-num").textContent = `${p.done} / ${p.total}`;
            $("ip-status").textContent = progressText(p, pct);
        },
        onDone: p => {
            $("ip-bar").style.width = "100%";
            if (p && p.total != null) $("ip-num").textContent = `${p.done} / ${p.total}`;
            const summary = p && p.added != null ? `Added ${p.added}, skipped ${p.skipped || 0} already in the dataset.` : "Import finished.";
            $("ip-status").textContent = summary + " Opening the annotator…";
            showToast(summary, "ok");
            setTimeout(async () => {
                await openAnnotator();
                show("ingest-prog", false);
                $("src-input").value = "";
                show("src-info", false);
                btn.disabled = true;
            }, 400);
        },
        onError: msg => {
            $("ip-status").textContent = `Import failed: ${msg}. Check the folder and try again.`;
            btn.disabled = false;
        },
    });
}

function bindSetup() {
    $("btn-load-model").addEventListener("click", loadModel);
    $("btn-browse-model").addEventListener("click", browseModelFile);
    $("model-select").addEventListener("change", () => { if ($("model-select").value) $("model-preset").value = ""; });
    $("model-preset").addEventListener("change", () => { if ($("model-preset").value) $("model-select").value = ""; });
    $("btn-help-setup").addEventListener("click", () => openModal("help-modal"));
    $("btn-continue").addEventListener("click", () => openSelectedDataset("annotate"));
    $("btn-view-dataset").addEventListener("click", () => openSelectedDataset("grid"));
    $("btn-open-other").addEventListener("click", openOtherFolder);
    $("btn-lock-retry").addEventListener("click", retryLockedDataset);
    $("btn-home-new").addEventListener("click", () => setHomeMode("create", { user: true, focus: "src-input" }));
    $("home-ds-search").addEventListener("input", renderHomeDatasets);

    // Open existing / Create new: toggle buttons; arrow keys move between them (and pick).
    const choices = [$("choice-open"), $("choice-create")];
    choices.forEach(b => b.addEventListener("click", () => setHomeMode(b.dataset.homeMode, { user: true })));
    $("home-choice").addEventListener("keydown", e => {
        const i = choices.indexOf(document.activeElement);
        if (i < 0) return;
        let j = null;
        if (e.key === "ArrowRight" || e.key === "ArrowDown") j = (i + 1) % choices.length;
        else if (e.key === "ArrowLeft" || e.key === "ArrowUp") j = (i + choices.length - 1) % choices.length;
        else if (e.key === "Home") j = 0;
        else if (e.key === "End") j = choices.length - 1;
        if (j == null) return;
        e.preventDefault();
        choices[j].focus();
        setHomeMode(choices[j].dataset.homeMode, { user: true });
    });
    // Per-section Retry links and "Create new dataset" shortcuts rendered into the home screen.
    $("setup-screen").addEventListener("click", e => {
        const r = e.target.closest("[data-home-retry]");
        if (r && HOME_LOADERS[r.dataset.homeRetry]) { e.preventDefault(); HOME_LOADERS[r.dataset.homeRetry](); return; }
        const g = e.target.closest("[data-home-mode-go]");
        if (g) { e.preventDefault(); setHomeMode(g.dataset.homeModeGo, { user: true, focus: g.dataset.homeModeGo === "create" ? "src-input" : null }); }
    });

    const sinp = $("src-input");
    sinp.addEventListener("change", () => validateSourcePath(sinp.value, "src-info", "btn-ingest"));
    $("btn-browse-src").addEventListener("click", async () => {
        const p = await browseFolder("Image folder:", sinp.value);
        if (p) { sinp.value = p; validateSourcePath(p, "src-info", "btn-ingest"); }
    });
    $("btn-ingest").addEventListener("click", startIngest);
}

// ═══ 8. OPENING THE ANNOTATOR, APP MODES ══════════════════════════════════════
function showScreen(which) {
    $("setup-screen").style.display = which === "setup" ? "" : "none";
    $("app").style.display = which === "app" ? "grid" : "none";
}

async function openAnnotator(opts = {}) {
    try {
        const st = await api("/api/state").catch(() => ({ ready: false }));
        if (!st.ready) { showToast("This dataset has no images yet. Add a folder of images with New dataset, or pick another dataset.", "err"); return; }
        datasetType = st.dataset_type || "bbox";
        datasetName = st.dataset_name || "";
        if (Array.isArray(st.classes)) setClasses(st.classes);
        else { const r = await api("/api/classes").catch(() => ({ classes: [] })); setClasses(r.classes || []); }
        if (Array.isArray(st.split_ratios) && st.split_ratios.length === 3) {
            ["sp-train", "sp-valid", "sp-test"].forEach((id, i) => { $(id).value = Math.round(st.split_ratios[i] * 100); });
        }
        mcClose();
        showScreen("app");
        annotatorOpen = true;
        $("add-box-label").textContent = datasetType === "polygon" ? "Add polygon" : "Add box";
        const shapeIcon = datasetType === "polygon" ? "fa-solid fa-draw-polygon" : "fa-solid fa-vector-square";
        $("btn-draw-icon").className = shapeIcon;
        $("add-box-icon").className = shapeIcon;
        $("rtab-boxes").textContent = datasetType === "polygon" ? "Polygons" : "Boxes";
        _sessionStart = Date.now();
        api("/api/device_info").then(d => updateDeviceChip(d.selected)).catch(() => {});
        loadSettings();
        loadModelClasses(true);
        let split = "train";
        try { split = localStorage.getItem("lastSplit") || "train"; } catch (e) { /* ignore */ }
        if (!["train", "valid", "test"].includes(split)) split = "train";
        currentSplit = split;
        document.querySelectorAll(".split-btn").forEach(t => t.classList.toggle("active", t.dataset.split === split));
        await setAppMode(opts.mode === "grid" ? "grid" : "annotate", { force: true });
        loadClassPrefs();
        renderClassBar();
        renderSlotHotkeys();
        loadClassUsage();
        updateStats(true);
    } catch (err) {
        console.error("openAnnotator error:", err);
        showToast("Couldn't open the annotator: " + err.message, "err");
    }
}

async function goToSetup() {
    if (!(await resolveDirty())) return;
    annotatorOpen = false;
    setDrawMode(false);
    showScreen("setup");
    initSetup();
}

async function setAppMode(mode, opts = {}) {
    if (mode === appMode && !opts.force) return;
    if (mode === "grid" && !(await resolveDirty())) return;
    appMode = mode;
    $("mode-annotate").classList.toggle("active", mode === "annotate");
    $("mode-grid").classList.toggle("active", mode === "grid");
    const grid = mode === "grid";
    $("left").style.display = grid ? "none" : "";
    $("canvas-area").style.display = grid ? "none" : "";
    $("right").style.display = grid ? "none" : "";
    $("grid-view").style.display = grid ? "flex" : "none";
    if (grid) {
        setDrawMode(false);
        buildClassDropdowns();
        updateRoiButtons();
        await loadGridPage(1);
    } else {
        await loadSplitImages(currentSplit, opts.preferName);
    }
}

async function switchSplit(split) {
    if (split === currentSplit && appMode === "annotate" && allImages.length) return;
    if (!(await resolveDirty())) return;
    document.querySelectorAll(".split-btn").forEach(t => t.classList.toggle("active", t.dataset.split === split));
    currentSplit = split;
    if (appMode === "grid") {
        gridSelectedImages.clear();
        lastGridSelectedIndex = -1;
        updateGridBatchBar();
        loadGridPage(1);
    } else {
        loadSplitImages(split);
    }
}

// ═══ 9. CANVAS: TRANSFORMS, GEOMETRY, RENDERING ═══════════════════════════════
function getTransform() {
    const W = canvas.width, H = canvas.height;
    const base = Math.max(0.0001, Math.min((W - 64) / imgNatW, (H - 6) / imgNatH, 1));
    const scale = base * zoomScale;
    const ox = W / 2 + panX - (imgNatW * scale) / 2;
    const oy = H / 2 + panY - (imgNatH * scale) / 2;
    return { ox, oy, scale };
}
function toScreen(pt) {
    const { ox, oy, scale } = getTransform();
    return { x: ox + pt.x * imgNatW * scale, y: oy + pt.y * imgNatH * scale };
}
function toNorm(p) {
    const { ox, oy, scale } = getTransform();
    return { x: (p.x - ox) / (imgNatW * scale), y: (p.y - oy) / (imgNatH * scale) };
}
const clampPt = p => ({ x: clamp01(p.x), y: clamp01(p.y) });
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const isPoly = b => Array.isArray(b && b.pts);

function shapeBounds(b) {
    if (isPoly(b)) {
        const xs = b.pts.map(p => p.x), ys = b.pts.map(p => p.y);
        return { l: Math.min(...xs), t: Math.min(...ys), r: Math.max(...xs), b: Math.max(...ys) };
    }
    return { l: b.x - b.w / 2, t: b.y - b.h / 2, r: b.x + b.w / 2, b: b.y + b.h / 2 };
}
function boxToPoly(b) {
    const { l, t, r, b: bt } = shapeBounds(b);
    return { cls: b.cls, pts: [{ x: l, y: t }, { x: r, y: t }, { x: r, y: bt }, { x: l, y: bt }] };
}
function polyToBox(p) {
    const { l, t, r, b } = shapeBounds(p);
    return { cls: p.cls, x: (l + r) / 2, y: (t + b) / 2, w: r - l, h: b - t };
}
function translateShape(s, dx, dy) {
    const o = clone(s);
    if (isPoly(o)) o.pts = o.pts.map(p => ({ x: p.x + dx, y: p.y + dy }));
    else { o.x += dx; o.y += dy; }
    return o;
}
/** Clamps a translation so the whole group of shapes stays inside the image. */
function clampDelta(shapes, dx, dy) {
    let l = 1, t = 1, r = 0, b = 0;
    shapes.forEach(s => { const bb = shapeBounds(s); l = Math.min(l, bb.l); t = Math.min(t, bb.t); r = Math.max(r, bb.r); b = Math.max(b, bb.b); });
    return { dx: Math.max(-l, Math.min(1 - r, dx)), dy: Math.max(-t, Math.min(1 - b, dy)) };
}
/** Cleans shapes from the server; polygon datasets convert plain boxes to 4-point polygons. */
function normalizeShapes(list) {
    if (!Array.isArray(list)) return [];
    return list.filter(b => b && (isPoly(b) ? b.pts.length >= 3 : isFinite(b.x) && isFinite(b.w))).map(b => {
        const s = isPoly(b) ? { cls: parseInt(b.cls, 10) || 0, pts: b.pts.map(p => ({ x: +p.x, y: +p.y })) }
            : { cls: parseInt(b.cls, 10) || 0, x: +b.x, y: +b.y, w: +b.w, h: +b.h };
        return datasetType === "polygon" && !isPoly(s) ? boxToPoly(s) : s;
    });
}
function shapeForDataset(s) {
    if (datasetType === "polygon" && !isPoly(s)) return boxToPoly(s);
    return clone(s);
}

function boxScreenRect(b) {
    const { ox, oy, scale } = getTransform();
    return { x: ox + (b.x - b.w / 2) * imgNatW * scale, y: oy + (b.y - b.h / 2) * imgNatH * scale, w: b.w * imgNatW * scale, h: b.h * imgNatH * scale };
}
function pointInPolygon(p, pts) {
    let inside = false;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
        const a = pts[i], c = pts[j];
        if ((a.y > p.y) !== (c.y > p.y) && p.x < ((c.x - a.x) * (p.y - a.y)) / (c.y - a.y) + a.x) inside = !inside;
    }
    return inside;
}
function distToSegment(p, a, b) {
    const dx = b.x - a.x, dy = b.y - a.y;
    const len2 = dx * dx + dy * dy;
    const t = len2 ? Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2)) : 0;
    return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}
function hitShape(p) {
    if (hideShapes) return -1;
    for (let i = boxes.length - 1; i >= 0; i--) {
        const b = boxes[i];
        if (isPoly(b)) { if (pointInPolygon(p, b.pts.map(toScreen))) return i; }
        else { const d = boxScreenRect(b); if (p.x >= d.x && p.x <= d.x + d.w && p.y >= d.y && p.y <= d.y + d.h) return i; }
    }
    return -1;
}
function hitHandle(i, p) {
    const d = boxScreenRect(boxes[i]);
    const pts = { tl: [d.x, d.y], tr: [d.x + d.w, d.y], bl: [d.x, d.y + d.h], br: [d.x + d.w, d.y + d.h] };
    for (const [k, [cx, cy]] of Object.entries(pts)) if (Math.abs(p.x - cx) <= HANDLE && Math.abs(p.y - cy) <= HANDLE) return k;
    return null;
}
function hitVertex(i, p) {
    const b = boxes[i];
    if (!isPoly(b)) return -1;
    for (let v = b.pts.length - 1; v >= 0; v--) if (dist(toScreen(b.pts[v]), p) <= HANDLE) return v;
    return -1;
}
function hitEdge(i, p) {
    const b = boxes[i];
    if (!isPoly(b)) return null;
    const s = b.pts.map(toScreen);
    for (let e = 0; e < s.length; e++) if (distToSegment(p, s[e], s[(e + 1) % s.length]) <= 6) return { edge: e };
    return null;
}

function isDrawing() { return drawMode || ctrlHeld; }

let _rafPending = false;
function renderCanvas() {
    if (_rafPending) return;
    _rafPending = true;
    requestAnimationFrame(doRender);
}

function doRender() {
    _rafPending = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!imgReady) { updateMinimap(); updateStatusBar(); return; }
    const { ox, oy, scale } = getTransform();
    const dW = imgNatW * scale, dH = imgNatH * scale;

    let filterStr = `brightness(${bcBright}) contrast(${bcContrast})`;
    if (extraFilterType !== "none") filterStr += ` ${extraFilterType}(${extraFilterVal}%)`;
    ctx.filter = filterStr;
    updateAdjustIndicator();
    ctx.drawImage(imgEl, ox, oy, dW, dH);
    ctx.filter = "none";

    if (!hideShapes) boxes.forEach((b, i) => drawShape(b, selectedBoxes.includes(i)));
    if (drag.on && drag.type === "draw") drawBoxDraft();
    if (polyDraft) drawPolyDraft();

    if ((isDrawing() || polyDraft) && cursorX >= 0 && !(drag.on && drag.type === "pan")) {
        ctx.save();
        ctx.globalCompositeOperation = "difference";
        ctx.strokeStyle = tok("--text-1"); ctx.globalAlpha = 0.75;
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(cursorX + 0.5, 0); ctx.lineTo(cursorX + 0.5, canvas.height); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, cursorY + 0.5); ctx.lineTo(canvas.width, cursorY + 0.5); ctx.stroke();
        ctx.restore();
    }
    updateMinimap();
    updateStatusBar();
}

function drawHandle(x, y, col, big) {
    const s = big ? HANDLE + 2 : HANDLE;
    ctx.fillStyle = tok("--text-1");
    ctx.fillRect(x - s / 2, y - s / 2, s, s);
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x - s / 2, y - s / 2, s, s);
}

function drawLabel(text, x, y, col) {
    ctx.font = `500 11px ${tok("--font-sans")}`;
    const tw = ctx.measureText(text).width;
    const ty = y > 16 ? y - 15 : y + 1;
    ctx.fillStyle = col;
    ctx.fillRect(x, ty, tw + 8, 15);
    ctx.fillStyle = tok("--bg");
    ctx.fillText(text, x + 4, ty + 11);
}

function drawShape(b, sel) {
    const col = colorOf(b.cls);
    ctx.lineWidth = sel ? 2.5 : 1.5;
    ctx.strokeStyle = col;
    ctx.fillStyle = col + (sel ? "40" : "1F");
    let lx, ly;
    if (isPoly(b)) {
        const s = b.pts.map(toScreen);
        ctx.beginPath();
        s.forEach((p, k) => (k ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
        ctx.closePath();
        ctx.fill(); ctx.stroke();
        const top = s.reduce((a, p) => (p.y < a.y ? p : a), s[0]);
        lx = top.x; ly = top.y;
        if (sel) {
            const hv = cursorX >= 0 ? s.findIndex(p => dist(p, { x: cursorX, y: cursorY }) <= HANDLE) : -1;
            s.forEach((p, k) => drawHandle(p.x, p.y, col, k === hv));
        }
    } else {
        const d = boxScreenRect(b);
        ctx.fillRect(d.x, d.y, d.w, d.h);
        ctx.strokeRect(d.x, d.y, d.w, d.h);
        lx = d.x; ly = d.y;
        if (sel) [[d.x, d.y], [d.x + d.w, d.y], [d.x, d.y + d.h], [d.x + d.w, d.y + d.h]].forEach(([x, y]) => drawHandle(x, y, col));
    }
    if (showLabels) drawLabel(className(b.cls), lx, ly, col);
}

function drawPill(text, x, y) {
    ctx.font = `500 11px ${tok("--font-mono")}`;
    const tw = ctx.measureText(text).width;
    let px = x + 14, py = y + 14;
    if (px + tw + 10 > canvas.width) px = x - tw - 20;
    if (py + 18 > canvas.height) py = y - 26;
    ctx.fillStyle = tok("--surface-1");
    ctx.fillRect(px, py, tw + 10, 18);
    ctx.strokeStyle = tok("--border-strong");
    ctx.lineWidth = 1;
    ctx.strokeRect(px + 0.5, py + 0.5, tw + 9, 17);
    ctx.fillStyle = tok("--text-1");
    ctx.fillText(text, px + 5, py + 13);
}

function drawBoxDraft() {
    const { scale } = getTransform();
    const col = colorOf(activeClass);
    const x = Math.min(drag.sx, drag.cx), y = Math.min(drag.sy, drag.cy);
    const w = Math.abs(drag.cx - drag.sx), h = Math.abs(drag.cy - drag.sy);
    ctx.strokeStyle = col;
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
    ctx.fillStyle = col + "1F";
    ctx.fillRect(x, y, w, h);
    drawPill(`${Math.round(w / scale)} × ${Math.round(h / scale)} px`, drag.cx, drag.cy);
}

function drawPolyDraft() {
    const col = colorOf(activeClass);
    const s = polyDraft.pts.map(toScreen);
    ctx.strokeStyle = col;
    ctx.lineWidth = 2;
    ctx.beginPath();
    s.forEach((p, k) => (k ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
    ctx.stroke();
    if (cursorX >= 0 && s.length) {
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(s[s.length - 1].x, s[s.length - 1].y);
        ctx.lineTo(cursorX, cursorY);
        ctx.stroke();
        ctx.setLineDash([]);
    }
    const nearFirst = s.length >= 3 && cursorX >= 0 && dist(s[0], { x: cursorX, y: cursorY }) <= CLOSE_RADIUS;
    s.forEach((p, k) => {
        if (k === 0) {
            ctx.beginPath();
            ctx.arc(p.x, p.y, nearFirst ? 7 : 5, 0, Math.PI * 2);
            ctx.fillStyle = nearFirst ? tok("--accent-text") : tok("--text-1");
            ctx.fill();
            ctx.strokeStyle = col;
            ctx.lineWidth = 2;
            ctx.stroke();
        } else {
            drawHandle(p.x, p.y, col);
        }
    });
    if (cursorX >= 0) drawPill(nearFirst ? "Click to close" : `${s.length} points`, cursorX, cursorY);
}

// Minimap (drawn by doRender, not monkey-patched)
const minimap = $("minimap");
const mctx = minimap.getContext("2d");
function updateMinimap() {
    if (!imgReady || zoomScale <= 1.01) { minimap.style.display = "none"; return; }
    minimap.style.display = "block";
    const mw = 140, mh = Math.max(20, Math.round(140 * imgNatH / imgNatW));
    if (minimap.width !== mw) minimap.width = mw;
    if (minimap.height !== mh) minimap.height = mh;
    mctx.clearRect(0, 0, mw, mh);
    mctx.drawImage(imgEl, 0, 0, mw, mh);
    const { ox, oy, scale } = getTransform();
    const dW = imgNatW * scale, dH = imgNatH * scale;
    const vx = Math.max(0, -ox / dW), vy = Math.max(0, -oy / dH);
    const vw = Math.min(1 - vx, canvas.width / dW), vh = Math.min(1 - vy, canvas.height / dH);
    mctx.strokeStyle = tok("--accent-text");
    mctx.lineWidth = 2;
    mctx.strokeRect(vx * mw, vy * mh, vw * mw, vh * mh);
}
function centerViewOn(nx, ny) {
    const { scale } = getTransform();
    const dW = imgNatW * scale, dH = imgNatH * scale;
    panX = dW / 2 - nx * dW;
    panY = dH / 2 - ny * dH;
    renderCanvas();
}

function resizeCanvas() {
    const wrap = $("canvas-wrapper");
    const w = wrap.clientWidth, h = wrap.clientHeight;
    if (w && h && (canvas.width !== w || canvas.height !== h)) { canvas.width = w; canvas.height = h; }
    renderCanvas();
}
function fitView() { zoomScale = 1.0; panX = 0; panY = 0; resizeCanvas(); }
function zoomBy(f, cx, cy) {
    const nz = Math.max(0.2, Math.min(30, zoomScale * f));
    const k = nz / zoomScale;
    if (cx == null) { cx = canvas.width / 2; cy = canvas.height / 2; }
    panX = panX * k - (cx - canvas.width / 2) * (k - 1);
    panY = panY * k - (cy - canvas.height / 2) * (k - 1);
    zoomScale = nz;
    renderCanvas();
}

// ═══ 10. CANVAS INPUT (MOUSE) AND DRAWING ═════════════════════════════════════
function getPos(e) { const r = canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }

function updateCursor(p) {
    let c = "default";
    if (drag.on && drag.type === "pan") c = "grabbing";
    else if (spaceHeld) c = "grab";
    else if (isDrawing() || polyDraft) c = "crosshair";
    else if (p) {
        if (selectedBoxes.length === 1 && boxes[selectedBoxes[0]]) {
            const i = selectedBoxes[0];
            if (isPoly(boxes[i])) { if (hitVertex(i, p) >= 0) c = "pointer"; }
            else { const h = hitHandle(i, p); if (h) c = h === "tl" || h === "br" ? "nwse-resize" : "nesw-resize"; }
        }
        if (c === "default" && hitShape(p) >= 0) c = "move";
    }
    canvas.style.cursor = c;
}

function updateCursorReadout() {
    const el = $("sb-cursor");
    if (!el) return;
    if (cursorX < 0 || !imgReady) { el.textContent = "–"; return; }
    const n = toNorm({ x: cursorX, y: cursorY });
    if (n.x < 0 || n.y < 0 || n.x > 1 || n.y > 1) { el.textContent = "–"; return; }
    el.textContent = `${Math.floor(n.x * imgNatW)}, ${Math.floor(n.y * imgNatH)}`;
}

function onCanvasDown(e) {
    if (currentIndex < 0 || !imgReady) return;
    if (palette.open) closePalette();   // in assign mode the new shape keeps the active class
    const p = getPos(e);
    if (e.button === 1 || (e.button === 0 && spaceHeld)) {
        e.preventDefault();
        drag = { on: true, type: "pan", sx: p.x, sy: p.y, px: panX, py: panY };
        updateCursor(p);
        return;
    }
    if (e.button === 2) {
        if (polyDraft) return;
        drag = { on: true, type: "draw", sx: p.x, sy: p.y, cx: p.x, cy: p.y };
        return;
    }
    if (e.button !== 0) return;

    if (polyDraft || (isDrawing() && datasetType === "polygon")) { polygonClick(p); return; }
    if (isDrawing()) { drag = { on: true, type: "draw", sx: p.x, sy: p.y, cx: p.x, cy: p.y }; return; }

    // Vertex / handle editing on the single selected shape
    if (selectedBoxes.length === 1 && !e.shiftKey && boxes[selectedBoxes[0]]) {
        const i = selectedBoxes[0];
        if (isPoly(boxes[i])) {
            const v = hitVertex(i, p);
            if (v >= 0) {
                if (e.altKey) { deleteVertex(i, v); return; }
                pushUndo();
                drag = { on: true, type: "vertex", i, v };
                return;
            }
        } else {
            const h = hitHandle(i, p);
            if (h) { pushUndo(); drag = { on: true, type: "resize", i, h, orig: clone(boxes[i]) }; return; }
        }
    }
    // Alt+click on any polygon vertex deletes it
    if (e.altKey && !hideShapes) {
        for (let i = boxes.length - 1; i >= 0; i--) {
            const v = hitVertex(i, p);
            if (v >= 0) { selectedBoxes = [i]; deleteVertex(i, v); return; }
        }
    }
    const hit = hitShape(p);
    if (hit >= 0) {
        if (e.shiftKey) {
            selectedBoxes = selectedBoxes.includes(hit) ? selectedBoxes.filter(j => j !== hit) : [...selectedBoxes, hit];
        } else {
            if (!selectedBoxes.includes(hit)) selectedBoxes = [hit];
            pushUndo();
            const orig = selectedBoxes.map(j => [j, clone(boxes[j])]);
            drag = { on: true, type: "move", sx: p.x, sy: p.y, orig };
        }
        updateBoxPanel(); renderCanvas();
        return;
    }
    if (!e.shiftKey && selectedBoxes.length) { selectedBoxes = []; updateBoxPanel(); renderCanvas(); }
}

function deleteVertex(i, v) {
    if (boxes[i].pts.length <= 3) { showToast("A polygon needs at least 3 points"); return; }
    editShapes("Deleted point", () => boxes[i].pts.splice(v, 1));
}

function onWindowMove(e) {
    if (!drag.on) return;
    const p = getPos(e);
    cursorX = p.x; cursorY = p.y;
    updateCursorReadout();
    const { scale } = getTransform();
    if (drag.type === "pan") {
        panX = drag.px + (p.x - drag.sx);
        panY = drag.py + (p.y - drag.sy);
    } else if (drag.type === "draw") {
        drag.cx = Math.max(0, Math.min(canvas.width, p.x));
        drag.cy = Math.max(0, Math.min(canvas.height, p.y));
    } else if (drag.type === "move") {
        const d = clampDelta(drag.orig.map(([, s]) => s), (p.x - drag.sx) / (imgNatW * scale), (p.y - drag.sy) / (imgNatH * scale));
        drag.orig.forEach(([j, s]) => { boxes[j] = translateShape(s, d.dx, d.dy); });
    } else if (drag.type === "resize") {
        const n = clampPt(toNorm(p));
        const o = drag.orig;
        let l = o.x - o.w / 2, t = o.y - o.h / 2, r = o.x + o.w / 2, b = o.y + o.h / 2;
        if (drag.h[0] === "t") t = n.y; else b = n.y;
        if (drag.h[1] === "l") l = n.x; else r = n.x;
        let L = Math.min(l, r), R = Math.max(l, r), T = Math.min(t, b), B = Math.max(t, b);
        const minW = 2 / imgNatW, minH = 2 / imgNatH;
        if (R - L < minW) R = Math.min(1, L + minW);
        if (B - T < minH) B = Math.min(1, T + minH);
        boxes[drag.i] = Object.assign({}, o, { x: (L + R) / 2, y: (T + B) / 2, w: R - L, h: B - T });
    } else if (drag.type === "vertex") {
        boxes[drag.i].pts[drag.v] = clampPt(toNorm(p));
    }
    renderCanvas();
}

function onWindowUp(e) {
    if (!drag.on) return;
    const d = drag;
    drag = { on: false };
    if (d.type === "pan") { updateCursor(isMouseInCanvas ? getPos(e) : null); return; }
    if (d.type === "draw") { finishBoxDraw(d); renderCanvas(); return; }
    const names = { move: "Moved shape", resize: "Resized box", vertex: "Moved point" };
    if (commitUndo(names[d.type] || "Edited shape")) { markDirty(); triggerAutoSave(); }
    updateBoxPanel();
    renderCanvas();
}

function finishBoxDraw(d) {
    if (Math.abs(d.cx - d.sx) < MIN_DRAW_PX || Math.abs(d.cy - d.sy) < MIN_DRAW_PX) return;
    const a = clampPt(toNorm({ x: d.sx, y: d.sy })), b = clampPt(toNorm({ x: d.cx, y: d.cy }));
    const l = Math.min(a.x, b.x), r = Math.max(a.x, b.x), t = Math.min(a.y, b.y), bt = Math.max(a.y, b.y);
    if (r - l <= 0 || bt - t <= 0) return;
    const box = { cls: activeClass, x: (l + r) / 2, y: (t + bt) / 2, w: r - l, h: bt - t };
    addNewShape(datasetType === "polygon" ? boxToPoly(box) : box, datasetType === "polygon" ? "Drew polygon" : "Drew box");
}

function polygonClick(p) {
    const n = clampPt(toNorm(p));
    if (!polyDraft) { polyDraft = { pts: [n] }; renderCanvas(); updateStatusBar(); return; }
    const s0 = toScreen(polyDraft.pts[0]);
    if (polyDraft.pts.length >= 3 && dist(p, s0) <= CLOSE_RADIUS) { finishPolygon(); return; }
    const last = toScreen(polyDraft.pts[polyDraft.pts.length - 1]);
    if (dist(p, last) < 4) return;   // second click of a double-click
    polyDraft.pts.push(n);
    renderCanvas();
}

function finishPolygon() {
    if (!polyDraft) return;
    if (polyDraft.pts.length < 3) { showToast("A polygon needs at least 3 points. Keep clicking, or press Esc to cancel."); return; }
    const shape = { cls: activeClass, pts: polyDraft.pts };
    polyDraft = null;
    addNewShape(shape, "Drew polygon");
}

function cancelDraft() {
    if (polyDraft) { polyDraft = null; renderCanvas(); updateStatusBar(); return true; }
    if (drag.on && drag.type === "draw") { drag = { on: false }; renderCanvas(); return true; }
    return false;
}

function addNewShape(shape, label) {
    pushUndo();
    boxes.push(shape);
    commitUndo(label);
    selectedBoxes = [boxes.length - 1];
    markDirty();
    if (drawMode && !ctrlHeld) setDrawMode(false);
    updateBoxPanel();
    renderCanvas();
    if (askClassAfterDraw && CLASS_NAMES.length > 1) openPalette("assign");
    else triggerAutoSave();
}

function onCanvasDblClick(e) {
    if (currentIndex < 0 || !imgReady) return;
    if (polyDraft) { finishPolygon(); return; }
    if (isDrawing() || hideShapes) return;
    const p = getPos(e);
    const order = [...selectedBoxes, ...boxes.map((_, i) => i).filter(i => !selectedBoxes.includes(i))];
    for (const i of order) {
        if (!boxes[i] || !isPoly(boxes[i])) continue;
        const h = hitEdge(i, p);
        if (h && hitVertex(i, p) < 0) {
            selectedBoxes = [i];
            editShapes("Added point", () => boxes[i].pts.splice(h.edge + 1, 0, clampPt(toNorm(p))));
            return;
        }
    }
}

function onCanvasHover(e) {
    const p = getPos(e);
    cursorX = p.x; cursorY = p.y;
    updateCursorReadout();
    if (drag.on) return;
    updateCursor(p);
    renderCanvas();
}

function onCanvasWheel(e) {
    e.preventDefault();
    if (!imgReady) return;
    const p = getPos(e);
    zoomBy(e.deltaY < 0 ? 1.12 : 1 / 1.12, p.x, p.y);
}

function bindCanvas() {
    canvas.addEventListener("mousedown", onCanvasDown);
    canvas.addEventListener("mousemove", onCanvasHover);
    canvas.addEventListener("dblclick", onCanvasDblClick);
    canvas.addEventListener("wheel", onCanvasWheel, { passive: false });
    canvas.addEventListener("contextmenu", e => e.preventDefault());
    canvas.addEventListener("mouseenter", () => { isMouseInCanvas = true; });
    canvas.addEventListener("mouseleave", () => {
        isMouseInCanvas = false;
        if (!drag.on) { cursorX = -1; cursorY = -1; updateCursorReadout(); renderCanvas(); }
    });
    window.addEventListener("mousemove", onWindowMove);
    window.addEventListener("mouseup", onWindowUp);
    minimap.addEventListener("mousedown", e => {
        e.preventDefault();
        e.stopPropagation();
        centerViewOn(e.offsetX / minimap.width, e.offsetY / minimap.height);
    });
    if (window.ResizeObserver) new ResizeObserver(() => resizeCanvas()).observe($("canvas-wrapper"));
    else window.addEventListener("resize", resizeCanvas);
}

// "Ask class after drawing" uses the class palette in assign mode.
function isPopupOpen() { return palette.open && palette.mode === "assign"; }
function hidePopup() { closePalette(); }
function popupApply() { closePalette(); }

// ═══ 11. KEYBOARD (single handler) ════════════════════════════════════════════
function annotatorActive() { return annotatorOpen && appMode === "annotate"; }

function onKeyDown(e) {
    const key = e.key;
    const k = key && key.length === 1 ? key.toLowerCase() : key;   // Caps Lock safe
    const mod = e.ctrlKey || e.metaKey;

    if (key === "Escape") {
        const closedPop = closePalette() | mcClose() | detClose() | adjustClose() | (closeGridClassPop() ? 1 : 0);
        if (closedPop) { e.preventDefault(); return; }
        if (isPopupOpen()) { hidePopup(); triggerAutoSave(); return; }
        if (closeTopModal()) { e.preventDefault(); return; }
        if (isTypingTarget(e.target)) { e.target.blur(); return; }
        if (!annotatorActive()) return;
        if (cancelDraft()) return;
        if (selectedBoxes.length) { selectedBoxes = []; updateBoxPanel(); renderCanvas(); return; }
        if (drawMode) setDrawMode(false);
        return;
    }
    if (isTypingTarget(e.target)) return;
    // A keyboard-focused button/link keeps its native Enter/Space behaviour.
    if ((key === "Enter" || key === " ") && e.target.closest && e.target.closest("button, a, summary")) return;

    if (key === "?") {
        e.preventDefault();
        if (isModalOpen("shortcuts-modal")) closeModal("shortcuts-modal"); else openModal("shortcuts-modal");
        return;
    }

    if (!annotatorOpen && !anyModalOpen() && key === "Enter" && homeCanOpenSelected()) {
        e.preventDefault();
        openSelectedDataset("annotate");
        return;
    }
    if (_modalStack[_modalStack.length - 1] === "image-modal") {
        if (key === "ArrowRight" || k === "n") { e.preventDefault(); stepImageModal(1); }
        else if (key === "ArrowLeft" || k === "p") { e.preventDefault(); stepImageModal(-1); }
        else if (key === "Enter") { e.preventDefault(); $("btn-jump-anno").click(); }
        else if (key === "Delete" || key === "Backspace") { e.preventDefault(); deleteModalCurrentImage(); }
        else if (k === "h") { e.preventDefault(); toggleGridRoi(); }
        return;
    }
    if (appMode === "grid" && !anyModalOpen()) {
        if (k === "h") { e.preventDefault(); toggleGridRoi(); return; }
    }
    if (anyModalOpen() || !annotatorActive()) return;

    if (key === "Control" || key === "Meta") {
        if (isMouseInCanvas && !ctrlHeld) { ctrlHeld = true; updateCursor(null); renderCanvas(); }
        return;
    }
    if (key === " ") {
        e.preventDefault();
        if (!spaceHeld) { spaceHeld = true; updateCursor(null); }
        return;
    }

    if (mod) {
        switch (k) {
            case "z": e.preventDefault(); if (e.shiftKey) doRedo(); else doUndo(); return;
            case "y": e.preventDefault(); doRedo(); return;
            case "s": e.preventDefault(); saveAnnotations(false); return;
            case "a": e.preventDefault(); selectAll(); return;
            case "c": e.preventDefault(); copySelection(); return;
            case "v": e.preventDefault(); pasteShapes(); return;
            case "d": e.preventDefault(); duplicateSelection(); return;
            default: return;   // leave other browser shortcuts alone
        }
    }

    if (key.startsWith("Arrow")) {
        if (e.shiftKey) {
            if (selectedBoxes.length) {
                e.preventDefault();
                const step = e.altKey ? 10 : 1;
                const dir = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[key];
                nudgeSelection(dir[0] * step, dir[1] * step);
            }
            return;
        }
        if (e.altKey) return;
        if (key === "ArrowRight") { e.preventDefault(); navigateTo(currentIndex + 1); }
        else if (key === "ArrowLeft") { e.preventDefault(); navigateTo(currentIndex - 1); }
        return;
    }
    if (e.altKey) return;

    if (key === "Tab") { e.preventDefault(); cycleSelection(e.shiftKey ? -1 : 1); return; }
    if (key === "Enter") { e.preventDefault(); if (polyDraft) finishPolygon(); else saveAndNext(); return; }
    if (key === "Backspace" && polyDraft) {
        e.preventDefault();
        polyDraft.pts.pop();
        if (!polyDraft.pts.length) polyDraft = null;
        renderCanvas();
        return;
    }
    if (key === "Delete" || key === "Backspace") { e.preventDefault(); deleteSelected(); return; }
    if (/^[1-9]$/.test(key)) { applySlotKey(parseInt(key, 10) - 1); return; }
    if (key === "/" || k === "k") { e.preventDefault(); openPalette("set"); return; }

    switch (k) {
        case "0": case "f": fitView(); break;
        case "+": case "=": zoomBy(1.2); break;
        case "-": case "_": zoomBy(1 / 1.2); break;
        case "d": toggleDrawMode(); break;
        case "j": jumpToNextUnannotated(); break;
        case "r": toggleReviewed(); break;
        case "h": toggleHideShapes(); break;
        case "l": toggleLabels(); break;
        case "c": copyFromPrevious(); break;
        case "q": cycleClass(-1); break;
        case "e": cycleClass(1); break;
        case "n": navigateTo(currentIndex + 1); break;
        case "p": navigateTo(currentIndex - 1); break;
        default: return;
    }
}

function onKeyUp(e) {
    if ((e.key === "Control" || e.key === "Meta") && ctrlHeld) {
        ctrlHeld = false;
        updateCursor(cursorX >= 0 ? { x: cursorX, y: cursorY } : null);
        renderCanvas();
    }
    if (e.key === " " && spaceHeld) {
        spaceHeld = false;
        updateCursor(cursorX >= 0 ? { x: cursorX, y: cursorY } : null);
    }
}

function setDrawMode(on) {
    drawMode = !!on;
    const btn = $("btn-draw");
    btn.classList.toggle("active-tool", drawMode);
    btn.querySelector(".tbtn-label").textContent = drawMode ? "Drawing" : "Draw";
    btn.setAttribute("aria-pressed", String(drawMode));
    updateCursor(cursorX >= 0 ? { x: cursorX, y: cursorY } : null);
    renderCanvas();
}
function toggleDrawMode() {
    if (drawMode && polyDraft) polyDraft = null;
    setDrawMode(!drawMode);
}

function selectAll() {
    selectedBoxes = boxes.map((_, i) => i);
    updateBoxPanel(); renderCanvas();
}
function cycleSelection(dir) {
    const n = boxes.length;
    if (!n) return;
    const cur = selectedBoxes.length ? selectedBoxes[selectedBoxes.length - 1] : (dir > 0 ? -1 : 0);
    selectedBoxes = [((cur + dir) % n + n) % n];
    updateBoxPanel(); renderCanvas();
}
function deleteSelected() {
    if (!selectedBoxes.length) return;
    const n = selectedBoxes.length;
    editShapes(n === 1 ? "Deleted shape" : `Deleted ${n} shapes`, () => {
        boxes = boxes.filter((_, i) => !selectedBoxes.includes(i));
        selectedBoxes = [];
    });
}
function nudgeSelection(pxX, pxY) {
    editShapes("Nudged", () => {
        const sel = selectedBoxes.map(i => boxes[i]);
        const d = clampDelta(sel, pxX / imgNatW, pxY / imgNatH);
        selectedBoxes.forEach(i => { boxes[i] = translateShape(boxes[i], d.dx, d.dy); });
    }, true);
}
function copySelection() {
    if (!selectedBoxes.length) { showToast("Select shapes first, then press Ctrl+C"); return; }
    shapeClipboard = selectedBoxes.map(i => clone(boxes[i]));
    showToast(`Copied ${shapeClipboard.length} ${shapeClipboard.length === 1 ? "shape" : "shapes"}`);
}
function pasteShapes() {
    if (!shapeClipboard.length) { showToast("Nothing to paste. Select shapes and press Ctrl+C first."); return; }
    if (currentIndex < 0) return;
    editShapes(`Pasted ${shapeClipboard.length} shapes`, () => {
        const start = boxes.length;
        shapeClipboard.forEach(s => boxes.push(shapeForDataset(s)));
        selectedBoxes = boxes.map((_, i) => i).filter(i => i >= start);
    });
}
function duplicateSelection() {
    if (!selectedBoxes.length) { showToast("Select shapes first, then press Ctrl+D"); return; }
    editShapes("Duplicated", () => {
        const start = boxes.length;
        selectedBoxes.forEach(i => {
            const s = boxes[i];
            const d = clampDelta([s], 10 / imgNatW, 10 / imgNatH);
            boxes.push(translateShape(s, d.dx, d.dy));
        });
        selectedBoxes = boxes.map((_, i) => i).filter(i => i >= start);
    });
}
function toggleHideShapes() {
    hideShapes = !hideShapes;
    const b = $("btn-toggle-shapes");
    b.classList.toggle("active-tool", hideShapes);
    b.querySelector("i").className = hideShapes ? "fa-solid fa-eye-slash" : "fa-solid fa-eye";
    showToast(hideShapes ? "Annotations hidden. Press H to show them." : "Annotations shown");
    renderCanvas();
}
function toggleLabels() {
    showLabels = !showLabels;
    savePref("showLabels", showLabels ? "1" : "0");
    showToast(showLabels ? "Class labels shown" : "Class labels hidden. Press L to show them.");
    renderCanvas();
}

async function copyFromPrevious() {
    if (currentIndex <= 0) { showToast("There is no previous image in this list"); return; }
    const prev = filteredImages[currentIndex - 1];
    try {
        const data = await api(`/api/labels/${currentSplit}/${encodeURIComponent(prev.name)}`);
        const shapes = normalizeShapes(data);
        if (!shapes.length) { showToast(`${prev.name} has no shapes to copy`); return; }
        editShapes(`Copied ${shapes.length} shapes from previous image`, () => { boxes = shapes; selectedBoxes = []; });
        showToast(`Copied ${shapes.length} shapes from ${prev.name}`);
    } catch (e) {
        showToast(`Couldn't read the previous image's labels: ${e.message}`, "err");
    }
}

// ═══ 12. IMAGE LIST, NAVIGATION, LOADING AND SAVING ═══════════════════════════
let current = { split: null, name: null };   // the image whose shapes are in `boxes`

function imgStatus(item) { return !item ? "pending" : item.reviewed ? "reviewed" : item.annotated ? "auto" : "pending"; }
const STATUS_LABEL = { pending: "Pending", auto: "Auto-labelled", reviewed: "Reviewed" };
function currentItem() {
    if (!current.name || current.split !== currentSplit) return null;
    return allImages.find(i => i.name === current.name) || null;
}

async function loadSplitImages(split, preferName) {
    currentSplit = split;
    savePref("lastSplit", split);
    currentIndex = -1;
    boxes = []; selectedBoxes = []; dirty = false; imgReady = false; polyDraft = null;
    current = { split: null, name: null };
    cmdManager.clear();
    setSaveState("saved");
    $("canvas-filename").textContent = "";
    showLoading("Loading image list…");
    try {
        const list = await api(`/api/images?split=${encodeURIComponent(split)}`);
        allImages = (list || []).map(i => ({ name: i.name, annotated: !!i.annotated, reviewed: !!i.reviewed }));
    } catch (e) {
        allImages = [];
        showToast(`Couldn't load the ${split} list: ${e.message}`, "err");
    }
    renderImageList();
    updateBoxPanel();
    renderCanvas();
    if (!filteredImages.length) {
        showOverlayMessage(allImages.length
            ? "No images match this filter. Clear the search or choose another tab."
            : `The ${split} split has no images. Pick another split, or use Add images.`);
        $("canvas-filename").textContent = "No image";
        return;
    }
    let target = preferName;
    if (!target) { try { target = localStorage.getItem("lastImage_" + split); } catch (e) { target = null; } }
    let idx = target ? filteredImages.findIndex(i => i.name === target) : -1;
    if (idx < 0 && preferName && allImages.some(i => i.name === preferName)) {
        setSidebarFilter("all");
        $("search").value = "";
        renderImageList();
        idx = filteredImages.findIndex(i => i.name === preferName);
    }
    hideLoading();
    navigateTo(Math.max(0, idx), { skipGuard: true, force: true });
}

function setSidebarFilter(f) {
    sidebarFilter = f;
    document.querySelectorAll(".ftab").forEach(t => t.classList.toggle("active", t.dataset.filter === f));
}

/** Re-applies search + filter. Status changes do NOT call this, so "Save and next" keeps its place. */
function renderImageList(opts = {}) {
    const want = (currentIndex >= 0 && filteredImages[currentIndex] ? filteredImages[currentIndex].name : null) || current.name;
    const q = $("search").value.trim().toLowerCase();
    let base = q ? allImages.filter(i => i.name.toLowerCase().includes(q)) : allImages.slice();
    if (sidebarFilter === "toreview") base = base.filter(i => !i.reviewed);
    if (sidebarFilter === "reviewed") base = base.filter(i => i.reviewed);
    filteredImages = base;
    currentIndex = want ? filteredImages.findIndex(i => i.name === want) : -1;
    $("image-list-spacer").style.height = filteredImages.length * ITEM_H + "px";
    refreshListRows();
    if (!opts.autoNav || !annotatorActive()) return;
    if (currentIndex >= 0) { if (imgReady) hideLoading(); scrollToCurrent(); return; }
    if (filteredImages.length) navigateTo(0, { force: true });
    else { showOverlayMessage("No images match this filter. Clear the search or choose another tab."); $("canvas-filename").textContent = "No image"; }
}

function updateListCounts() {
    const n = allImages.length;
    const r = allImages.filter(i => i.reviewed).length;
    $("fc-all").textContent = n;
    $("fc-toreview").textContent = n - r;
    $("fc-reviewed").textContent = r;
    $("img-count").textContent = `${r} / ${n} reviewed`;
    $("img-count-pct").textContent = n ? Math.round(r / n * 100) + "%" : "";
    $("list-progress-bar").style.width = (n ? r / n * 100 : 0) + "%";
}

let _rowsKey = "";
function renderVisibleRows(force) {
    const viewport = $("image-list-viewport");
    const inner = $("image-list-inner");
    const start = Math.max(0, Math.floor(viewport.scrollTop / ITEM_H) - 3);
    const end = Math.min(filteredImages.length - 1, Math.ceil((viewport.scrollTop + viewport.clientHeight) / ITEM_H) + 3);
    const key = `${start}:${end}`;
    if (!force && key === _rowsKey) return;
    _rowsKey = key;
    inner.style.top = start * ITEM_H + "px";
    const frag = document.createDocumentFragment();
    for (let i = start; i <= end; i++) {
        const item = filteredImages[i];
        if (!item) continue;
        const st = imgStatus(item);
        const div = document.createElement("div");
        div.className = "img-item" + (i === currentIndex ? " active" : "");
        div.style.height = ITEM_H + "px";
        div.title = `${item.name} · ${STATUS_LABEL[st]}`;
        const th = document.createElement("img");
        th.className = "thumb";
        th.loading = "lazy";
        th.alt = "";
        th.src = `/api/thumb/${currentSplit}/${encodeURIComponent(item.name)}`;
        th.onerror = () => { th.classList.add("thumb-missing"); };
        const nm = document.createElement("div");
        nm.className = "iname";
        nm.textContent = item.name;
        const dot = document.createElement("div");
        dot.className = "dot " + st;
        dot.setAttribute("aria-label", STATUS_LABEL[st]);
        div.append(th, nm, dot);
        div.addEventListener("click", () => navigateTo(i));
        frag.appendChild(div);
    }
    inner.replaceChildren(frag);
}

function refreshListRows() {
    updateListCounts();
    renderVisibleRows(true);
    updateStatusBar();
}

function scrollToCurrent() {
    const viewport = $("image-list-viewport");
    const top = currentIndex * ITEM_H;
    if (top < viewport.scrollTop || top > viewport.scrollTop + viewport.clientHeight - ITEM_H) {
        viewport.scrollTop = Math.max(0, top - viewport.clientHeight / 2 + ITEM_H / 2);
    }
    renderVisibleRows(true);
}

/** Handles unsaved edits before leaving the current image. Returns false to stay. */
async function resolveDirty() {
    if (!dirty || !current.name) return true;
    if (autoSave) {
        const ok = await flushSave();
        if (ok) return true;
        return confirm(`Saving "${current.name}" failed. Leave anyway and lose the changes?`);
    }
    const save = confirm(`"${current.name}" has unsaved changes.\n\nOK: save them and continue.\nCancel: discard them and continue.`);
    if (save) {
        const ok = await saveAnnotations(true);
        if (!ok) return confirm("Saving failed. Leave anyway and lose the changes?");
    } else {
        dirty = false;
        setSaveState("saved");
    }
    return true;
}

async function navigateTo(idx, opts = {}) {
    if (!filteredImages.length) return;
    if (idx >= filteredImages.length && !opts.force) { showToast("This is the last image in the list"); return; }
    if (idx < 0 && !opts.force) { showToast("This is the first image in the list"); return; }
    idx = Math.max(0, Math.min(idx, filteredImages.length - 1));
    if (idx === currentIndex && !opts.force) return;
    const targetName = filteredImages[idx].name;
    if (!opts.skipGuard && !(await resolveDirty())) return;
    // the list may have changed while saving; re-resolve by name
    const j = filteredImages.findIndex(i => i.name === targetName);
    if (j < 0) return;
    if (isPopupOpen()) hidePopup();
    polyDraft = null;
    drag = { on: false };
    currentIndex = j;
    savePref("lastImage_" + currentSplit, targetName);
    savePref("lastSplit", currentSplit);
    scrollToCurrent();
    loadImage(targetName);
    preloadAround(j);
}

function preloadAround(idx) {
    [1, 2, -1].forEach((off, s) => {
        const it = filteredImages[idx + off];
        if (it) _preloadSlots[s].src = `/api/image/${currentSplit}/${encodeURIComponent(it.name)}`;
    });
}

async function loadImage(filename) {
    const myId = ++_loadId;
    const split = currentSplit;
    imgReady = false;
    boxes = []; selectedBoxes = []; dirty = false; editVersion++;
    current = { split, name: filename };
    cmdManager.clear();
    setSaveState("saved");
    zoomScale = 1.0; panX = 0; panY = 0;
    bcBright = 1.0; $("bc-bright").value = 1;
    bcContrast = 1.0; $("bc-contrast").value = 1;
    extraFilterType = "none"; $("extra-filter-type").value = "none";
    extraFilterVal = 0; $("extra-filter-val").value = 0;
    const willLabel = serverSettings.model_loaded && serverSettings.auto_annotate && !(currentItem() || {}).annotated;
    showLoading(willLabel ? "Loading and auto-labelling…" : "Loading…");
    $("canvas-filename").textContent = filename;
    updateBoxPanel();
    renderCanvas();

    let labelsErr = null;
    const labelsP = api(`/api/labels/${split}/${encodeURIComponent(filename)}`).catch(e => { labelsErr = e; return []; });
    const im = new Image();
    const imgP = new Promise(res => { im.onload = () => res(true); im.onerror = () => res(false); });
    im.src = `/api/image/${split}/${encodeURIComponent(filename)}`;
    const [labels, ok] = await Promise.all([labelsP, imgP]);
    if (myId !== _loadId) return;

    if (!ok) {
        showOverlayMessage("This image couldn't be loaded. It may be missing or corrupt. Press the right arrow to skip it, or delete it.");
        renderCanvas();
        return;
    }
    imgEl = im;
    imgNatW = im.naturalWidth || 1;
    imgNatH = im.naturalHeight || 1;
    imgReady = true;
    hideLoading();
    boxes = normalizeShapes(labels);
    const it = currentItem();
    if (it && boxes.length) it.annotated = true;
    if (labelsErr) showOverlayMessage(`Couldn't load labels: ${labelsErr.message}. Click to dismiss and label by hand, or try Detection, Re-run on this image.`);
    else if (!boxes.length && serverSettings.model_loaded && serverSettings.auto_annotate && mc.loaded && mc.map.size === 0) {
        showToast("No classes selected for auto-labelling. Choose classes to turn it on.", "err");
    }
    resizeCanvas();
    updateBoxPanel();
    refreshListRows();
    updateStats();
}

function setSaveState(s) { saveState = s; updateStatusBar(); }
function markDirty() { dirty = true; editVersion++; setSaveState("unsaved"); }

const _autoSaveDebounced = debounce(() => { if (dirty) saveAnnotations(true); }, 350);
function triggerAutoSave() {
    if (autoSave && dirty && current.name) _autoSaveDebounced();
}
async function flushSave() {
    _autoSaveDebounced.cancel();
    return dirty ? saveAnnotations(true) : true;
}

async function saveAnnotations(silent = false) {
    if (!current.name) return false;
    _autoSaveDebounced.cancel();
    const target = { split: current.split, name: current.name };
    const ver = editVersion;
    const payload = boxes.map(b => (isPoly(b) ? { cls: b.cls, pts: b.pts } : { cls: b.cls, x: b.x, y: b.y, w: b.w, h: b.h }));
    setSaveState("saving");
    try {
        const r = await api("/api/save", { json: { split: target.split, filename: target.name, boxes: payload } });
        if (r && r.skipped > 0) showToast(`Saved, but ${r.skipped} malformed ${r.skipped === 1 ? "shape was" : "shapes were"} skipped`, "err");
        if (target.split === currentSplit) {
            const it = allImages.find(i => i.name === target.name);
            if (it) { it.annotated = true; it.reviewed = true; }
        }
        if (current.name === target.name && current.split === target.split && editVersion === ver) {
            dirty = false;
            setSaveState("saved");
        } else if (current.name === target.name) {
            setSaveState("unsaved");
        }
        refreshListRows();
        updateStats();
        if (!silent) showToast("Saved and marked reviewed", "ok");
        return true;
    } catch (e) {
        if (current.name === target.name) setSaveState("error");
        showToast(`Couldn't save ${target.name}: ${e.message}`, "err");
        return false;
    }
}

async function saveAndNext() {
    if (currentIndex < 0) return;
    if (!(await saveAnnotations(true))) return;
    if (currentIndex >= filteredImages.length - 1) { showToast("Saved. That was the last image in this list.", "ok"); return; }
    navigateTo(currentIndex + 1, { skipGuard: true });
}

async function toggleReviewed() {
    const it = currentItem();
    if (!it) return;
    if (dirty) { await saveAnnotations(false); return; }
    const want = !it.reviewed;
    try {
        const r = await api("/api/mark_reviewed", { json: { split: current.split, filename: current.name, reviewed: want } });
        it.reviewed = r && r.reviewed != null ? !!r.reviewed : want;
        if (r && r.annotated != null) it.annotated = !!r.annotated;
        else if (want) it.annotated = true;
        refreshListRows();
        updateStats();
        showToast(want ? "Marked reviewed" : "Marked as to review", want ? "ok" : "info");
    } catch (e) {
        showToast(`Couldn't change the review state: ${e.message}`, "err");
    }
}

async function deleteCurrentImage() {
    if (currentIndex < 0 || !current.name) return;
    const fn = current.name;
    if (!confirm(`Delete "${fn}" and its labels from the dataset?\n\nThis can't be undone.`)) return;
    try {
        await api("/api/delete_image", { json: { split: current.split, filename: fn } });
    } catch (e) {
        showToast(`Couldn't delete ${fn}: ${e.message}`, "err");
        return;
    }
    dirty = false;
    _autoSaveDebounced.cancel();
    const idx = currentIndex;
    allImages = allImages.filter(i => i.name !== fn);
    filteredImages = filteredImages.filter(i => i.name !== fn);
    currentIndex = -1;
    current = { split: null, name: null };
    $("image-list-spacer").style.height = filteredImages.length * ITEM_H + "px";
    refreshListRows();
    updateStats();
    showToast(`Deleted ${fn}`);
    if (filteredImages.length) navigateTo(Math.min(idx, filteredImages.length - 1), { skipGuard: true, force: true });
    else { boxes = []; imgReady = false; updateBoxPanel(); renderCanvas(); showOverlayMessage("No images left in this list."); }
}

/** Jumps to the next image that is not reviewed, wrapping around the current list. */
function jumpToNextUnannotated() {
    const n = filteredImages.length;
    for (let k = 1; k <= n; k++) {
        const i = (currentIndex + k) % n;
        if (!filteredImages[i].reviewed && i !== currentIndex) { navigateTo(i); return; }
    }
    showToast("Every image in this list is reviewed", "ok");
}

// Detection popover: re-run auto-labelling
function detClose() {
    const pop = $("det-pop");
    if (pop.hidden) return false;
    pop.hidden = true;
    $("det-btn").setAttribute("aria-expanded", "false");
    return true;
}
function detToggle(e) {
    e.stopPropagation();
    const pop = $("det-pop");
    if (!pop.hidden) { detClose(); return; }
    adjustClose();
    pop.hidden = false;
    $("det-btn").setAttribute("aria-expanded", "true");
    renderSettings();
}

function adjustClose() {
    const pop = $("adjust-pop");
    if (pop.hidden) return false;
    pop.hidden = true;
    $("adjust-btn").setAttribute("aria-expanded", "false");
    return true;
}
function adjustToggle(e) {
    e.stopPropagation();
    const pop = $("adjust-pop");
    if (!pop.hidden) { adjustClose(); return; }
    detClose();
    pop.hidden = false;
    $("adjust-btn").setAttribute("aria-expanded", "true");
}
function updateAdjustIndicator() {
    const on = bcBright !== 1 || bcContrast !== 1 || extraFilterType !== "none";
    $("adjust-btn").classList.toggle("active-tool", on);
}
function resetAdjustments() {
    bcBright = 1; $("bc-bright").value = 1;
    bcContrast = 1; $("bc-contrast").value = 1;
    extraFilterType = "none"; $("extra-filter-type").value = "none";
    extraFilterVal = 0; $("extra-filter-val").max = 100; $("extra-filter-val").value = 0;
    updateAdjustIndicator();
    renderCanvas();
}

async function relabelCurrent(force = false) {
    if (!current.name) { showToast("Open an image first"); return; }
    if (!serverSettings.model_loaded) { showToast("Load a model in Setup first", "err"); return; }
    if (!force && dirty && !confirm("Re-running replaces your unsaved changes on this image. Continue?")) return;
    const btn = $("btn-relabel-one");
    btn.disabled = true;
    const target = { split: current.split, name: current.name };
    try {
        const r = await api("/api/relabel", { json: { split: target.split, filename: target.name, force } });
        if (current.name !== target.name) return;
        const prev = clone(boxes);
        boxes = normalizeShapes(Array.isArray(r) ? r : (r && r.boxes) || []);
        selectedBoxes = [];
        cmdManager.pushState(prev, boxes, "Re-ran auto-labelling");
        _autoSaveDebounced.cancel();
        dirty = false;
        editVersion++;
        setSaveState("saved");
        const it = currentItem();
        if (it) { it.annotated = true; it.reviewed = false; }
        refreshListRows();
        updateBoxPanel();
        renderCanvas();
        updateStats();
        showToast(`Auto-labelling found ${boxes.length} ${boxes.length === 1 ? "object" : "objects"}`, "ok");
    } catch (e) {
        if (e.status === 409 && !force) {
            btn.disabled = false;
            if (confirm("This image is reviewed. Re-running replaces its labels and marks it as to review. Continue?")) return relabelCurrent(true);
            return;
        }
        showToast(`Couldn't re-run auto-labelling: ${e.message}`, "err");
    } finally {
        btn.disabled = false;
    }
}

async function relabelAll() {
    if (!serverSettings.model_loaded) { showToast("Load a model in Setup first", "err"); return; }
    if (!confirm(`Re-run auto-labelling on every image in the ${currentSplit} split that isn't reviewed?\n\nReviewed images are not touched.`)) return;
    if (!(await resolveDirty())) return;
    const btn = $("btn-relabel-all");
    btn.disabled = true;
    show("relabel-prog", true);
    $("relabel-bar").style.width = "0%";
    $("relabel-num").textContent = "0 / 0";
    $("relabel-label").textContent = "Starting…";
    let job;
    try {
        job = await api("/api/relabel_all", { json: { split: currentSplit } });
    } catch (e) {
        $("relabel-label").textContent = `Couldn't start: ${e.message}`;
        btn.disabled = false;
        return;
    }
    const keep = current.name;
    followProgress(job.job_id, {
        onTick: p => {
            const pct = p.total > 0 ? Math.round(p.done / p.total * 100) : 0;
            $("relabel-bar").style.width = pct + "%";
            $("relabel-num").textContent = `${p.done} / ${p.total}`;
            $("relabel-label").textContent = `Re-running… ${pct}%`;
        },
        onDone: p => {
            $("relabel-bar").style.width = "100%";
            $("relabel-label").textContent = "Finished";
            btn.disabled = false;
            showToast(`Re-ran auto-labelling on ${p && p.total != null ? p.total : "all unreviewed"} images`, "ok");
            if (appMode === "annotate") loadSplitImages(currentSplit, keep);
            updateStats(true);
            setTimeout(() => show("relabel-prog", false), 2500);
        },
        onError: msg => {
            $("relabel-label").textContent = `Failed: ${msg}`;
            btn.disabled = false;
        },
    });
}

// ═══ 13. SHAPES PANEL, STATUS BAR ═════════════════════════════════════════════
function updateBoxPanel() {
    const panel = $("boxes-panel");
    panel.innerHTML = "";
    const poly = datasetType === "polygon";
    const noun = n => (poly ? (n === 1 ? "polygon" : "polygons") : (n === 1 ? "box" : "boxes"));
    const head = document.createElement("div");
    head.className = "shapes-head";
    head.textContent = `${boxes.length} ${noun(boxes.length)}` + (selectedBoxes.length ? ` · ${selectedBoxes.length} selected` : "");
    panel.appendChild(head);
    if (boxes.length === 0) {
        const empty = document.createElement("div");
        empty.className = "shapes-empty";
        empty.textContent = !current.name ? "Open an image to see its shapes."
            : poly ? "No polygons yet. Press D, then click to place points. Click the first point or press Enter to close."
                : "No boxes yet. Press D and drag on the image, or hold Ctrl and drag.";
        panel.appendChild(empty);
        return;
    }
    boxes.forEach((b, i) => {
        const isSel = selectedBoxes.includes(i);
        const card = document.createElement("div");
        card.className = "box-card" + (isSel ? " sel" : "");
        const hdr = document.createElement("div");
        hdr.className = "bch";
        hdr.addEventListener("click", e => {
            if (e.shiftKey) selectedBoxes = selectedBoxes.includes(i) ? selectedBoxes.filter(x => x !== i) : [...selectedBoxes, i];
            else selectedBoxes = [i];
            updateBoxPanel(); renderCanvas();
        });
        const num = document.createElement("div");
        num.className = "bnum";
        num.textContent = i + 1;
        num.style.background = colorOf(b.cls);
        const cls = document.createElement("div");
        cls.className = "bcls";
        cls.textContent = className(b.cls);
        cls.title = `Class ID ${b.cls}`;
        const meta = document.createElement("div");
        meta.className = "bmeta";
        meta.textContent = isPoly(b) ? `${b.pts.length} pts` : `${Math.round(b.w * imgNatW)}×${Math.round(b.h * imgNatH)}`;
        const del = document.createElement("button");
        del.className = "bdel";
        del.title = "Delete shape";
        del.setAttribute("aria-label", "Delete shape");
        del.innerHTML = '<i class="fa-solid fa-xmark"></i>';
        del.addEventListener("click", e => {
            e.stopPropagation();
            editShapes("Deleted shape", () => { boxes.splice(i, 1); selectedBoxes = []; });
        });
        hdr.append(num, cls, meta, del);
        card.append(hdr);
        if (isSel && selectedBoxes.length === 1) {
            const body = document.createElement("div");
            body.className = "bcb";
            const sel = document.createElement("select");
            sel.className = "cls-sel";
            sel.setAttribute("aria-label", "Class");
            const n = Math.max(CLASS_NAMES.length, b.cls + 1);
            for (let ci = 0; ci < n; ci++) {
                const o = document.createElement("option");
                o.value = ci;
                o.textContent = `${ci}: ${className(ci)}`;
                if (ci === b.cls) o.selected = true;
                sel.appendChild(o);
            }
            sel.addEventListener("change", () => {
                const v = parseInt(sel.value, 10);
                editShapes(`Set class to ${className(v)}`, () => { boxes[i].cls = v; });
                sel.blur();
            });
            const coords = document.createElement("div");
            coords.className = "bcoords";
            coords.textContent = isPoly(b)
                ? `${b.pts.length} points · drag a point to move it, Alt+click to delete it, double-click an edge to add one`
                : `x ${b.x.toFixed(4)}  y ${b.y.toFixed(4)}  w ${b.w.toFixed(4)}  h ${b.h.toFixed(4)}`;
            body.append(sel, coords);
            card.append(body);
        }
        panel.appendChild(card);
    });
}

function updateStatusBar() {
    const sbIndex = $("sb-index");
    if (!sbIndex) return;
    sbIndex.textContent = filteredImages.length && currentIndex >= 0 ? `Image ${currentIndex + 1} / ${filteredImages.length}` : "No image";
    $("sb-size").textContent = imgReady ? `${imgNatW} × ${imgNatH}` : "–";
    $("sb-zoom").textContent = imgReady ? `${Math.round(getTransform().scale * 100)}%` : "–";
    let mode = "Select";
    if (polyDraft) mode = `Polygon: ${polyDraft.pts.length} points. Enter or click the first point to close`;
    else if (isDrawing()) mode = `${datasetType === "polygon" ? "Draw polygon" : "Draw box"} · ${CLASS_NAMES.length ? className(activeClass) : "class 1"}`;
    else if (spaceHeld) mode = "Pan";
    $("sb-mode").textContent = mode;
    $("sb-mode").classList.toggle("drawing", !!(polyDraft || isDrawing()));
    const save = $("sb-save");
    const st = !current.name ? "saved" : saveState;
    save.className = "sb-item sb-save " + st;
    save.textContent = { saved: "Saved", unsaved: "Unsaved changes", saving: "Saving…", error: "Save failed" }[st];
    const rv = $("sb-review");
    const it = currentItem();
    const rs = imgStatus(it);
    rv.className = "sb-review " + rs;
    rv.textContent = it ? STATUS_LABEL[rs] : "–";
    rv.disabled = !it;
}

// ═══ 14. STATS, CLASS CHART, SPLIT PANEL ══════════════════════════════════════
let _statsTimer = null;
function updateStats(immediate) {
    clearTimeout(_statsTimer);
    _statsTimer = setTimeout(_doUpdateStats, immediate ? 0 : 800);
}
async function _doUpdateStats() {
    let d;
    try { d = await api("/api/stats"); } catch (e) { return; }
    const g = (s, k) => (d[s] && d[s][k]) || 0;
    ["train", "valid", "test"].forEach(s => {
        $("s-" + s).textContent = d[s] ? `${g(s, "reviewed")}/${g(s, "total")}` : "–";
    });
    const tot = g("train", "total") + g("valid", "total") + g("test", "total");
    const ann = g("train", "annotated") + g("valid", "annotated") + g("test", "annotated");
    const rev = g("train", "reviewed") + g("valid", "reviewed") + g("test", "reviewed");
    $("sst-total").textContent = tot;
    $("sst-anno").textContent = ann;
    $("sst-rev").textContent = rev;
    $("sst-tr").textContent = g("train", "total");
    $("sst-vl").textContent = g("valid", "total");
    $("sst-ts").textContent = g("test", "total");
    if (tot > 0) {
        $("sbar-t").style.width = (g("train", "total") / tot * 100) + "%";
        $("sbar-v").style.width = (g("valid", "total") / tot * 100) + "%";
        $("sbar-x").style.width = (g("test", "total") / tot * 100) + "%";
    }
    const pct = tot > 0 ? Math.round(rev / tot * 100) : 0;
    const ring = $("prog-ring-circle");
    ring.style.strokeDashoffset = 56.5 - (pct / 100) * 56.5;
    ring.classList.toggle("done", pct >= 100);
    $("prog-pct").textContent = pct + "%";
    if ($("panel-split").classList.contains("active")) updateClassChart();
}

async function updateClassChart() {
    const body = $("cls-chart-body");
    let counts;
    try { counts = await api("/api/class_stats"); } catch (e) { body.innerHTML = '<div class="shapes-empty">Couldn\'t load class counts.</div>'; return; }
    const total = Object.values(counts || {}).reduce((a, b) => a + b, 0);
    if (!total) { body.innerHTML = '<div class="shapes-empty">No labels yet.</div>'; return; }
    body.innerHTML = "";
    Object.entries(counts).sort(([, a], [, b]) => b - a).forEach(([name, cnt]) => {
        const idx = CLASS_NAMES.indexOf(name);
        const col = idx >= 0 ? colorOf(idx) : "var(--text-3)";
        const row = document.createElement("div");
        row.className = "cls-bar-row";
        row.innerHTML = `<div class="cls-bar-label" title="${escapeHtml(name)}">${escapeHtml(name)}</div>
            <div class="cls-bar-track"><div class="cls-bar-fill" style="width:${Math.round(cnt / total * 100)}%;background:${col}"></div></div>
            <div class="cls-bar-count">${cnt}</div>`;
        body.appendChild(row);
    });
}

async function doAutoSplit() {
    const ratios = readRatios(["sp-train", "sp-valid", "sp-test"]);
    if (!ratios) { showToast("Set at least one split percentage above zero", "err"); return; }
    if (!(await resolveDirty())) return;
    const btn = $("btn-auto-split");
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Splitting…';
    try {
        const d = await api("/api/auto_split", { json: { ratios } });
        const c = d.counts || {};
        setMsg("split-result", "ok", `Done. Train ${c.train || 0}, valid ${c.valid || 0}, test ${c.test || 0}. Skipped ${c.unannotated_skipped || 0} images without labels` + (c.duplicates_skipped ? ` and ${c.duplicates_skipped} duplicates.` : ".") );
        updateStats(true);
        await loadSplitImages(currentSplit, current.name);
        showToast("Dataset split", "ok");
    } catch (e) {
        setMsg("split-result", "err", `Split failed: ${escapeHtml(e.message)}`);
        showToast(`Split failed: ${e.message}`, "err");
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-shuffle"></i> Auto-split dataset';
    }
}

// ═══ 15. GRID VIEW AND IMAGE MODAL ════════════════════════════════════════════
let gridPage = 1;
let gridPages = 1;
let currentGridImages = [];
let currentGridIndex = -1;
let _healthFilterFiles = null;
let gridSelectedImages = new Set();
let lastGridSelectedIndex = -1;
let gridRoiVisible = true;
try {
    const savedRoi = localStorage.getItem("annotator_browse_roi");
    if (savedRoi !== null) gridRoiVisible = savedRoi === "true";
} catch (e) { /* ignore */ }
const SVG_NS = "http://www.w3.org/2000/svg";

function updateRoiButtons() {
    const iconCls = gridRoiVisible ? "fa-solid fa-eye" : "fa-solid fa-eye-slash";
    const labelText = gridRoiVisible ? "ROI: On" : "ROI: Off";

    // Main gallery button
    const gridBtn = $("btn-grid-toggle-roi");
    if (gridBtn) {
        gridBtn.classList.toggle("roi-off", !gridRoiVisible);
        const icon = $("grid-roi-icon");
        if (icon) icon.className = iconCls;
        const lbl = $("grid-roi-label");
        if (lbl) lbl.textContent = labelText;
        gridBtn.title = gridRoiVisible ? "Hide ROI / bounding boxes (H)" : "Show ROI / bounding boxes (H)";
    }

    // Grid container
    const gridView = $("grid-view");
    if (gridView) {
        gridView.classList.toggle("hide-roi", !gridRoiVisible);
    }

    // Modal button in sidebar
    const modalBtn = $("btn-modal-toggle-roi");
    if (modalBtn) {
        modalBtn.classList.toggle("roi-off", !gridRoiVisible);
        const icon = $("modal-roi-icon");
        if (icon) icon.className = iconCls;
        const lbl = $("modal-roi-label");
        if (lbl) lbl.textContent = labelText;
        modalBtn.title = gridRoiVisible ? "Hide ROI on this image (H)" : "Show ROI on this image (H)";
    }

    // Modal floating button on stage
    const stageBtn = $("btn-modal-stage-roi");
    if (stageBtn) {
        stageBtn.classList.toggle("roi-off", !gridRoiVisible);
        stageBtn.innerHTML = `<i class="${iconCls}"></i> <span>${labelText}</span>`;
        stageBtn.title = gridRoiVisible ? "Hide ROI on this image (H)" : "Show ROI on this image (H)";
    }

    // Modal container
    const modal = $("image-modal");
    if (modal) {
        modal.classList.toggle("hide-roi", !gridRoiVisible);
    }
}

function setGridRoi(visible, showToastMsg = true) {
    gridRoiVisible = !!visible;
    try {
        localStorage.setItem("annotator_browse_roi", String(gridRoiVisible));
    } catch (e) { /* ignore */ }
    updateRoiButtons();
    if (showToastMsg) {
        showToast(gridRoiVisible ? "ROI overlays shown" : "ROI overlays hidden");
    }
}

function toggleGridRoi() {
    setGridRoi(!gridRoiVisible);
}

function updateGridBatchBar() {
    const bar = $("grid-batch-bar");
    if (!bar) return;
    const count = gridSelectedImages.size;
    if (count > 0) {
        bar.classList.remove("d-none");
        const cntEl = $("grid-batch-count");
        if (cntEl) cntEl.textContent = count;
        const delNum = $("grid-batch-del-num");
        if (delNum) delNum.textContent = count;
    } else {
        bar.classList.add("d-none");
    }
    document.querySelectorAll(".grid-card").forEach(card => {
        const fn = card.dataset.filename;
        const isSel = fn && gridSelectedImages.has(fn);
        card.classList.toggle("selected", !!isSel);
        const cb = card.querySelector(".grid-card-cb");
        if (cb) cb.checked = !!isSel;
        const sw = card.querySelector(".grid-card-select-wrap");
        if (sw) sw.setAttribute("aria-checked", String(!!isSel));
    });
}

function selectAllGridPage() {
    currentGridImages.forEach(img => {
        if (img && img.name) gridSelectedImages.add(img.name);
    });
    lastGridSelectedIndex = -1;
    updateGridBatchBar();
}

function deselectAllGridImages() {
    gridSelectedImages.clear();
    lastGridSelectedIndex = -1;
    updateGridBatchBar();
}

async function deleteSelectedGridImages() {
    const count = gridSelectedImages.size;
    if (!count) return;
    if (!confirm(`Delete ${count} selected images and their labels from ${currentSplit}? This cannot be undone.`)) return;
    const btn = $("btn-grid-batch-delete");
    if (btn) btn.disabled = true;
    try {
        const filenames = Array.from(gridSelectedImages);
        const res = await api("/api/delete_images", { json: { split: currentSplit, filenames } });
        showToast(`Deleted ${res.deleted_count || count} images`, "ok");
        gridSelectedImages.clear();
        lastGridSelectedIndex = -1;
        updateGridBatchBar();
        updateStats(true);
        await loadGridPage(gridPage);
    } catch (e) {
        showToast(`Failed to delete images: ${e.message}`, "err");
    } finally {
        if (btn) btn.disabled = false;
    }
}

async function deleteModalCurrentImage() {
    const img = currentGridImages[currentGridIndex];
    if (!img) return;
    if (!confirm(`Delete ${img.name} and its labels from ${currentSplit}? This cannot be undone.`)) return;
    const btn = $("btn-modal-delete");
    if (btn) btn.disabled = true;
    try {
        await api("/api/delete_image", { json: { split: currentSplit, filename: img.name } });
        showToast(`Deleted ${img.name}`, "ok");
        if (gridSelectedImages.has(img.name)) {
            gridSelectedImages.delete(img.name);
            updateGridBatchBar();
        }
        currentGridImages.splice(currentGridIndex, 1);
        if (currentGridImages.length === 0) {
            closeImageModal();
        } else {
            if (currentGridIndex >= currentGridImages.length) {
                currentGridIndex = currentGridImages.length - 1;
            }
            openImageModal(currentGridIndex);
        }
        updateStats(true);
        loadGridPage(gridPage);
    } catch (e) {
        showToast(`Couldn't delete image: ${e.message}`, "err");
    } finally {
        if (btn) btn.disabled = false;
    }
}

/** SVG overlay in image-pixel space; preserveAspectRatio=meet letterboxes exactly like object-fit: contain. */
function buildOverlaySvg(shapes, W, H, withLabels) {
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    svg.setAttribute("class", "shape-overlay");
    svg.setAttribute("aria-hidden", "true");
    const fs = Math.max(10, Math.round(Math.max(W, H) / 55));
    (shapes || []).forEach(s => {
        const col = colorOf(s.cls);
        let el, lx, ly;
        if (isPoly(s)) {
            el = document.createElementNS(SVG_NS, "polygon");
            el.setAttribute("points", s.pts.map(p => `${p.x * W},${p.y * H}`).join(" "));
            const top = s.pts.reduce((a, p) => (p.y < a.y ? p : a), s.pts[0]);
            lx = top.x * W; ly = top.y * H;
        } else {
            el = document.createElementNS(SVG_NS, "rect");
            lx = (s.x - s.w / 2) * W; ly = (s.y - s.h / 2) * H;
            el.setAttribute("x", lx); el.setAttribute("y", ly);
            el.setAttribute("width", s.w * W); el.setAttribute("height", s.h * H);
        }
        el.setAttribute("fill", col + "26");
        el.setAttribute("stroke", col);
        el.setAttribute("stroke-width", "2");
        el.setAttribute("vector-effect", "non-scaling-stroke");
        svg.appendChild(el);
        if (withLabels) {
            const t = document.createElementNS(SVG_NS, "text");
            t.setAttribute("x", lx + 2);
            t.setAttribute("y", ly > fs + 2 ? ly - 4 : ly + fs);
            t.setAttribute("font-size", fs);
            t.setAttribute("fill", col);
            t.setAttribute("class", "ov-label");
            t.textContent = className(s.cls);
            svg.appendChild(t);
        }
    });
    return svg;
}

function gridItemStatus(img) { return img.reviewed ? "reviewed" : img.annotated ? "auto" : "pending"; }

async function loadGridPage(page) {
    if (page < 1) page = 1;
    const cont = $("grid-content");
    const limit = parseInt($("grid-limit").value, 10) || 50;
    let classIdsParam = null;
    if (gridClassFilterMode === "custom") {
        classIdsParam = gridSelectedClasses.size > 0 ? Array.from(gridSelectedClasses) : [-999];
    }
    const body = {
        split: currentSplit,
        page,
        limit,
        filter: $("grid-filter").value,
        class_ids: classIdsParam,
        class_id: $("grid-class-filter") ? $("grid-class-filter").value : ""
    };
    if (_healthFilterFiles !== null) body.filter_files = _healthFilterFiles;
    show("grid-health-chip", _healthFilterFiles !== null);
    if (_healthFilterFiles !== null) $("grid-health-text").textContent = `Health check: ${_healthFilterFiles.length} flagged files`;
    cont.innerHTML = '<div class="grid-empty">Loading…</div>';
    let res;
    try {
        res = await api("/api/dataset_page", { json: body });
    } catch (e) {
        cont.innerHTML = `<div class="grid-empty">Couldn't load images: ${escapeHtml(e.message)}. Click Refresh to try again.</div>`;
        return;
    }
    gridPage = res.page;
    gridPages = res.pages;
    const pageText = `Page ${res.page} / ${res.pages}`;
    $("grid-page-info").textContent = pageText;
    $("grid-page-info-btm").textContent = pageText;
    $("grid-total").textContent = `${res.total} ${res.total === 1 ? "image" : "images"}`;
    $("grid-prev").disabled = $("grid-prev-btm").disabled = res.page <= 1;
    $("grid-next").disabled = $("grid-next-btm").disabled = res.page >= res.pages;
    currentGridImages = res.images || [];
    cont.innerHTML = "";
    if (!currentGridImages.length) {
        cont.innerHTML = '<div class="grid-empty">No images match these filters. Change the filters above or pick another split.</div>';
        updateGridBatchBar();
        return;
    }
    currentGridImages.forEach((img, idx) => {
        const st = gridItemStatus(img);
        const card = document.createElement("div");
        card.className = "grid-card" + (gridSelectedImages.has(img.name) ? " selected" : "");
        card.dataset.filename = img.name;
        card.tabIndex = 0;
        card.setAttribute("role", "button");
        card.setAttribute("aria-label", `${img.name}, ${STATUS_LABEL[st]}`);
        card.addEventListener("click", e => {
            if (e.target.closest(".grid-card-select-wrap")) return;
            openImageModal(idx);
        });
        card.addEventListener("keydown", e => {
            if (e.target.closest(".grid-card-select-wrap")) return;
            if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openImageModal(idx); }
        });

        const selWrap = document.createElement("div");
        selWrap.className = "grid-card-select-wrap";
        selWrap.title = "Select image (Shift+click for range)";
        selWrap.setAttribute("role", "checkbox");
        selWrap.setAttribute("aria-checked", String(gridSelectedImages.has(img.name)));
        selWrap.tabIndex = 0;

        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.className = "grid-card-cb";
        cb.checked = gridSelectedImages.has(img.name);
        cb.tabIndex = -1;
        cb.setAttribute("aria-label", `Select ${img.name}`);

        const toggleSelection = e => {
            if (e) {
                e.stopPropagation();
                if (e.stopImmediatePropagation) e.stopImmediatePropagation();
            }
            if (e && e.target === cb) {
                // Native input click toggled cb.checked already
            } else {
                cb.checked = !cb.checked;
            }
            const isNowChecked = cb.checked;
            if (e && e.shiftKey && lastGridSelectedIndex >= 0 && lastGridSelectedIndex !== idx) {
                const start = Math.min(lastGridSelectedIndex, idx);
                const end = Math.max(lastGridSelectedIndex, idx);
                for (let k = start; k <= end; k++) {
                    const targetImg = currentGridImages[k];
                    if (targetImg && targetImg.name) {
                        if (isNowChecked) gridSelectedImages.add(targetImg.name);
                        else gridSelectedImages.delete(targetImg.name);
                    }
                }
            } else {
                if (isNowChecked) {
                    gridSelectedImages.add(img.name);
                } else {
                    gridSelectedImages.delete(img.name);
                }
            }
            lastGridSelectedIndex = idx;
            updateGridBatchBar();
        };

        selWrap.addEventListener("click", toggleSelection);
        cb.addEventListener("click", toggleSelection);
        cb.addEventListener("change", e => e.stopPropagation());
        selWrap.addEventListener("keydown", e => {
            if (e.key === " " || e.key === "Enter") {
                e.preventDefault();
                e.stopPropagation();
                toggleSelection(e);
            }
        });
        selWrap.appendChild(cb);

        const wrap = document.createElement("div");
        wrap.className = "grid-thumb-wrap";
        const timg = document.createElement("img");
        timg.className = "grid-thumb";
        timg.loading = "lazy";
        timg.alt = "";
        timg.src = `/api/thumb/${currentSplit}/${encodeURIComponent(img.name)}`;
        timg.addEventListener("load", () => {
            if (img.boxes && img.boxes.length) wrap.appendChild(buildOverlaySvg(img.boxes, timg.naturalWidth || 1, timg.naturalHeight || 1, false));
        });
        timg.addEventListener("error", () => wrap.classList.add("thumb-missing"));
        wrap.appendChild(timg);
        const badge = document.createElement("span");
        badge.className = "gbadge " + st;
        badge.textContent = STATUS_LABEL[st];
        wrap.appendChild(badge);
        wrap.appendChild(selWrap);
        const meta = document.createElement("div");
        meta.className = "grid-meta";
        const nm = document.createElement("span");
        nm.className = "grid-name";
        nm.textContent = img.name;
        nm.title = img.name;
        const cnt = document.createElement("span");
        cnt.className = "grid-count";
        const n = (img.boxes || []).length;
        cnt.textContent = `${n} ${n === 1 ? "shape" : "shapes"}`;
        meta.append(nm, cnt);
        card.append(wrap, meta);
        cont.appendChild(card);
    });
    updateGridBatchBar();
}

function openImageModal(idx) {
    const img = currentGridImages[idx];
    if (!img) return;
    currentGridIndex = idx;
    if (!isModalOpen("image-modal")) openModal("image-modal");
    const st = gridItemStatus(img);
    const n = (img.boxes || []).length;
    $("im-title").textContent = img.name;
    $("im-meta").textContent = `${STATUS_LABEL[st]} · ${currentSplit} · ${n} ${n === 1 ? "shape" : "shapes"}`;
    const left = $("im-left");
    left.innerHTML = "";

    // Floating ROI toggle button on top-left of image stage
    const stageRoiBtn = document.createElement("button");
    stageRoiBtn.type = "button";
    stageRoiBtn.className = "im-stage-roi-btn" + (gridRoiVisible ? "" : " roi-off");
    stageRoiBtn.id = "btn-modal-stage-roi";
    stageRoiBtn.title = gridRoiVisible ? "Hide ROI / bounding boxes (H)" : "Show ROI / bounding boxes (H)";
    stageRoiBtn.innerHTML = `<i class="fa-solid ${gridRoiVisible ? "fa-eye" : "fa-eye-slash"}"></i> <span>ROI: ${gridRoiVisible ? "On" : "Off"}</span>`;
    stageRoiBtn.addEventListener("click", e => {
        e.stopPropagation();
        toggleGridRoi();
    });
    left.appendChild(stageRoiBtn);

    const stage = document.createElement("div");
    stage.className = "im-stage";
    const full = document.createElement("img");
    full.className = "im-img";
    full.alt = img.name;
    full.src = `/api/image/${currentSplit}/${encodeURIComponent(img.name)}`;
    full.addEventListener("load", () => {
        if (currentGridIndex !== idx) return;
        stage.appendChild(buildOverlaySvg(img.boxes, full.naturalWidth || 1, full.naturalHeight || 1, true));
        $("im-meta").textContent = `${full.naturalWidth} × ${full.naturalHeight} · ${STATUS_LABEL[st]} · ${currentSplit} · ${n} ${n === 1 ? "shape" : "shapes"}`;
    });
    stage.appendChild(full);
    left.appendChild(stage);

    const cls = $("im-classes");
    cls.innerHTML = "";
    if (n) {
        const counts = {};
        img.boxes.forEach(b => { counts[b.cls] = (counts[b.cls] || 0) + 1; });
        Object.entries(counts).sort((a, b) => b[1] - a[1]).forEach(([c, k]) => {
            const t = document.createElement("span");
            t.className = "class-tag";
            t.innerHTML = `<span class="swatch" style="background:${colorOf(+c)}"></span>${escapeHtml(className(+c))} <b>${k}</b>`;
            cls.appendChild(t);
        });
    } else {
        cls.innerHTML = '<div class="field-msg muted">No objects labelled.</div>';
    }
    $("btn-modal-prev").disabled = idx <= 0;
    $("btn-modal-next").disabled = idx >= currentGridImages.length - 1;
    updateRoiButtons();
}
function stepImageModal(dir) {
    const i = currentGridIndex + dir;
    if (i >= 0 && i < currentGridImages.length) openImageModal(i);
}
function closeImageModal() { closeModal("image-modal"); $("im-left").innerHTML = ""; }

async function jumpToAnnotate() {
    const img = currentGridImages[currentGridIndex];
    if (!img) return;
    closeImageModal();
    await setAppMode("annotate", { preferName: img.name });
}

function bindGrid() {
    $("mode-annotate").addEventListener("click", () => setAppMode("annotate"));
    $("mode-grid").addEventListener("click", () => setAppMode("grid"));
    $("grid-filter").addEventListener("change", () => loadGridPage(1));
    const gcsel = $("grid-class-filter");
    if (gcsel) gcsel.addEventListener("change", () => loadGridPage(1));
    $("grid-limit").addEventListener("change", () => loadGridPage(1));
    $("btn-grid-refresh").addEventListener("click", () => loadGridPage(gridPage));
    ["grid-prev", "grid-prev-btm"].forEach(id => $(id).addEventListener("click", () => loadGridPage(gridPage - 1)));
    ["grid-next", "grid-next-btm"].forEach(id => $(id).addEventListener("click", () => loadGridPage(gridPage + 1)));
    $("btn-clear-health-filter").addEventListener("click", () => { _healthFilterFiles = null; loadGridPage(1); });
    $("btn-open-health").addEventListener("click", openHealthDashboard);
    $("btn-export-dataset").addEventListener("click", exportDataset);
    $("btn-close-image-modal").addEventListener("click", closeImageModal);
    $("btn-modal-prev").addEventListener("click", () => stepImageModal(-1));
    $("btn-modal-next").addEventListener("click", () => stepImageModal(1));
    $("btn-modal-toggle-roi")?.addEventListener("click", toggleGridRoi);
    $("btn-modal-delete")?.addEventListener("click", deleteModalCurrentImage);
    $("btn-jump-anno").addEventListener("click", jumpToAnnotate);

    // ROI toggle in Browse toolbar
    $("btn-grid-toggle-roi")?.addEventListener("click", toggleGridRoi);

    // Class management in Browse
    $("btn-grid-manage-classes")?.addEventListener("click", openClassesManager);

    // Class filter popover & buttons
    $("btn-grid-class-filter")?.addEventListener("click", toggleGridClassPop);
    $("grid-class-pop")?.addEventListener("click", e => e.stopPropagation());
    $("btn-grid-cls-all")?.addEventListener("click", setGridClassFilterAll);
    $("btn-grid-cls-clear")?.addEventListener("click", clearGridClassFilter);

    // Batch actions
    $("btn-grid-select-page")?.addEventListener("click", selectAllGridPage);
    $("btn-grid-deselect-all")?.addEventListener("click", deselectAllGridImages);
    $("btn-grid-batch-delete")?.addEventListener("click", deleteSelectedGridImages);
}

// ═══ 16. HEALTH DASHBOARD, EXPORT, ADD IMAGES ═════════════════════════════════
async function openHealthDashboard() {
    openModal("health-modal");
    const loading = $("health-loading");
    loading.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Scanning all splits…';
    show(loading, true);
    show("health-results", false);
    let data;
    try {
        data = await api("/api/dataset_health");
    } catch (e) {
        loading.innerHTML = `<span class="field-msg err">Couldn't run the health check: ${escapeHtml(e.message)}. Close this and try again.</span>`;
        return;
    }
    show(loading, false);
    show("health-results", true);
    const score = $("health-score");
    score.textContent = data.score;
    score.className = "health-score " + (data.score > 90 ? "good" : data.score > 70 ? "fair" : "poor");
    $("health-total-imgs").textContent = data.total_images;
    $("health-total-boxes").textContent = data.total_boxes;
    const list = $("health-issues-list");
    list.innerHTML = "";

    const addIssue = (title, count, icon, tone, desc, files) => {
        files = files || [];
        const row = document.createElement("div");
        const clickable = files.length > 0;
        row.className = `health-row ${tone}` + (clickable ? " clickable" : "") + (count && count !== "OK" ? "" : " quiet");
        row.innerHTML = `<div><div class="health-row-title"><i class="${icon}"></i> ${escapeHtml(title)}</div>
            <div class="health-row-desc">${escapeHtml(desc)}</div></div>
            <div class="health-row-count">${escapeHtml(String(count))}</div>`;
        if (clickable) {
            row.tabIndex = 0;
            row.setAttribute("role", "button");
            row.title = "Show these images in Browse";
            const go = () => {
                _healthFilterFiles = files;
                $("grid-filter").value = "all";
                $("grid-class-filter").value = "";
                closeModal("health-modal");
                loadGridPage(1);
                showToast(`Showing ${files.length} flagged images in this split`);
            };
            row.addEventListener("click", go);
            row.addEventListener("keydown", e => { if (e.key === "Enter") go(); });
        }
        list.appendChild(row);
    };
    const g = k => data[k] || { count: 0, files: [] };
    addIssue("Corrupt images", g("corrupt_images").count, "fa-solid fa-file-circle-xmark", "danger", "Empty or unreadable image files.", g("corrupt_images").files);
    addIssue("Malformed label lines", g("malformed").count, "fa-solid fa-file-circle-exclamation", "danger", "Label files with lines that can't be parsed.", g("malformed").files);
    addIssue("Images without labels", g("empty_images").count, "fa-regular fa-image", "info", "Images that have no shapes yet.", g("empty_images").files);
    addIssue("Tiny boxes", g("small_boxes").count, "fa-solid fa-down-left-and-up-right-to-center", "warning", "Boxes smaller than 0.05% of the image.", g("small_boxes").files);
    addIssue("Out-of-bounds boxes", g("oob_boxes").count, "fa-solid fa-vector-square", "warning", "Boxes that extend past the image edge.", g("oob_boxes").files);
    const clsCounts = Object.values(data.classes || {});
    if (clsCounts.length) {
        const ratio = Math.max(...clsCounts) / Math.max(Math.min(...clsCounts), 1);
        if (ratio > 5) addIssue("Class imbalance", Math.round(ratio) + "×", "fa-solid fa-scale-unbalanced", "warning", "The most common class has far more labels than the rarest one.");
        else addIssue("Class balance", "OK", "fa-solid fa-scale-balanced", "success", "Classes are reasonably balanced.");
    }
}

function exportDataset() {
    const format = $("export-format").value;
    const btn = $("btn-export-dataset");
    const old = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Exporting…';
    window.location.href = `/api/export?format=${encodeURIComponent(format)}`;
    showToast("Preparing the export. The download starts when it's ready.");
    setTimeout(() => { btn.disabled = false; btn.innerHTML = old; }, 4000);
}

function openAddImages() {
    $("add-src-input").value = "";
    show("new-info", false);
    show("add-ingest-prog", false);
    $("btn-confirm-add").disabled = true;
    openModal("add-images-modal");
    setTimeout(() => $("add-src-input").focus(), 30);
}

async function confirmAddImages() {
    const src = $("add-src-input").value.trim();
    if (!src) return;
    const btn = $("btn-confirm-add");
    btn.disabled = true;
    show("add-ingest-prog", true);
    $("add-ip-bar").style.width = "0%";
    $("add-ip-num").textContent = "0 / 0";
    $("add-ip-status").textContent = "Starting…";
    let job;
    try {
        job = await api("/api/ingest_append", { json: { source_dir: src, split: "train" } });
    } catch (e) {
        $("add-ip-status").textContent = `Couldn't start: ${e.message}`;
        btn.disabled = false;
        return;
    }
    followProgress(job.job_id, {
        onTick: p => {
            const pct = p.total > 0 ? Math.round(p.done / p.total * 100) : 0;
            $("add-ip-bar").style.width = pct + "%";
            $("add-ip-num").textContent = `${p.done} / ${p.total}`;
            $("add-ip-status").textContent = progressText(p, pct);
        },
        onDone: p => {
            $("add-ip-bar").style.width = "100%";
            const msg = p && p.added != null ? `Added ${p.added}, skipped ${p.skipped || 0} already in the dataset.` : "Images added to the train split.";
            $("add-ip-status").textContent = msg;
            showToast(msg, "ok");
            setTimeout(async () => {
                closeModal("add-images-modal");
                updateStats(true);
                if (appMode === "grid") loadGridPage(1);
                else if (currentSplit === "train") { if (await resolveDirty()) loadSplitImages("train", current.name); }
            }, 600);
        },
        onError: msg => {
            $("add-ip-status").textContent = `Failed: ${msg}. Check the folder and try again.`;
            btn.disabled = false;
        },
    });
}

function bindAddImages() {
    $("btn-add-images").addEventListener("click", openAddImages);
    const inp = $("add-src-input");
    inp.addEventListener("change", () => validateSourcePath(inp.value, "new-info", "btn-confirm-add"));
    inp.addEventListener("keydown", e => {
        if (e.key === "Enter") { e.preventDefault(); validateSourcePath(inp.value, "new-info", "btn-confirm-add").then(ok => { if (ok) confirmAddImages(); }); }
    });
    $("btn-browse-add-src").addEventListener("click", async () => {
        const p = await browseFolder("Folder with images to add:", inp.value);
        if (p) { inp.value = p; validateSourcePath(p, "new-info", "btn-confirm-add"); }
    });
    $("btn-confirm-add").addEventListener("click", confirmAddImages);
}

// ═══ 17. TRAIN, RESIZE, COMPRESS ══════════════════════════════════════════════
let _trainEventSource = null;

async function checkGpuTrainingSupport() {
    const devSelect = $("tr-device"), btnTrain = $("btn-train");
    let info = {};
    try { info = await api("/api/device_info"); } catch (e) { /* show as no GPU */ }
    const keep = devSelect.value;
    devSelect.innerHTML = "";
    const gpus = info.cuda || [];
    gpus.forEach(g => {
        const o = document.createElement("option");
        o.value = String(g.id);
        o.textContent = `GPU ${g.id}: ${g.name} (${g.total_gb} GB)`;
        devSelect.appendChild(o);
    });
    if (gpus.length > 1) {
        const o = document.createElement("option");
        o.value = gpus.map(g => g.id).join(",");
        o.textContent = `All GPUs (${o.value})`;
        devSelect.appendChild(o);
    }
    if (info.mps) {
        const o = document.createElement("option");
        o.value = "mps"; o.textContent = "Apple Silicon (MPS)";
        devSelect.appendChild(o);
    }
    const hasGpu = gpus.length > 0 || !!info.mps;
    if (!hasGpu) {
        const o = document.createElement("option");
        o.value = ""; o.textContent = "No GPU available"; o.disabled = true;
        devSelect.appendChild(o);
    } else if ([...devSelect.options].some(o => o.value === keep)) {
        devSelect.value = keep;
    }
    show("gpu-train-warning", !hasGpu);
    btnTrain.disabled = !hasGpu;
    btnTrain.title = hasGpu ? "Start training" : "Training needs a CUDA GPU or Apple Silicon (MPS)";

    try {
        const st = await api("/api/train/status");
        if (st.status === "running") {
            btnTrain.disabled = true;
            $("train-status").textContent = "Training is running. Showing its log.";
            if (!_trainEventSource) connectTrainLogs();
        }
    } catch (e) { /* ignore */ }
}

function connectTrainLogs() {
    const logBox = $("train-log"), statusBox = $("train-status"), btnTrain = $("btn-train");
    if (_trainEventSource) _trainEventSource.close();
    logBox.innerHTML = "";
    const es = new EventSource("/api/train/logs");
    _trainEventSource = es;
    es.onmessage = ev => {
        let parsed;
        try { parsed = JSON.parse(ev.data); } catch (e) { return; }
        if (parsed.line != null) {
            const stick = logBox.scrollTop + logBox.clientHeight >= logBox.scrollHeight - 20;
            const line = document.createElement("div");
            line.className = "log-line";
            line.textContent = parsed.line;
            logBox.appendChild(line);
            while (logBox.childNodes.length > 2000) logBox.removeChild(logBox.firstChild);
            if (stick) logBox.scrollTop = logBox.scrollHeight;
        }
        if (parsed.done) {
            es.close();
            _trainEventSource = null;
            btnTrain.disabled = false;
            api("/api/train/status").then(st => {
                const ok = st.exit_code == null || st.exit_code === 0;
                statusBox.textContent = ok ? "Training finished." : `Training stopped with exit code ${st.exit_code}. Check the log above.`;
                showToast(ok ? "Training finished" : "Training ended with an error. Check the log.", ok ? "ok" : "err");
            }).catch(() => { statusBox.textContent = "Training finished."; });
        }
    };
    es.onerror = () => {
        es.close();
        if (_trainEventSource === es) _trainEventSource = null;
        btnTrain.disabled = false;
        statusBox.textContent = "Lost the connection to the training log. Open this tab again to reconnect.";
    };
}

async function startTraining() {
    const device = $("tr-device").value;
    if (!device || device === "cpu") { showToast("Training needs a GPU. Pick a GPU or MPS device.", "err"); return; }
    const payload = {
        model: $("tr-model").value || "yolov8n.pt",
        custom_model: $("tr-custom").value || "",
        epochs: parseInt($("tr-epochs").value, 10) || 50,
        batch: parseInt($("tr-batch").value, 10) || 16,
        imgsz: parseInt($("tr-imgsz").value, 10) || 640,
        lr0: parseFloat($("tr-lr").value) || 0.01,
        device,
        name: $("tr-name").value.trim() || "run_v1",
    };
    const btn = $("btn-train");
    btn.disabled = true;
    $("train-status").textContent = "Starting training…";
    try {
        await api("/api/train", { json: payload });
        $("train-status").textContent = "Training is running.";
        connectTrainLogs();
    } catch (e) {
        btn.disabled = false;
        $("train-status").textContent = `Couldn't start training: ${e.message}`;
        showToast(`Couldn't start training: ${e.message}`, "err");
    }
}

async function stopTraining() {
    try {
        const r = await api("/api/train/stop", { json: {} });
        if (_trainEventSource) { _trainEventSource.close(); _trainEventSource = null; }
        $("btn-train").disabled = false;
        $("train-status").textContent = r.status === "stopped" ? "Training stopped." : "No training was running.";
        showToast(r.status === "stopped" ? "Training stopped" : "No training was running");
    } catch (e) {
        showToast(`Couldn't stop training: ${e.message}`, "err");
    }
}

/** Starts a tool job and polls /api/tools/status until it finishes. */
function runToolJob(cfg) {
    let timer = null;
    return async () => {
        const btn = $(cfg.btn), wrap = $(cfg.wrap), bar = $(cfg.bar), num = $(cfg.num), status = $(cfg.status);
        btn.disabled = true;
        btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> ${cfg.busy}`;
        show(wrap, true);
        if (cfg.onStart) cfg.onStart();
        bar.style.width = "0%"; num.textContent = "0%"; status.textContent = "Starting…";
        const reset = () => { btn.disabled = false; btn.innerHTML = cfg.idle; };
        try {
            await api(cfg.url, { json: cfg.body() });
        } catch (e) {
            reset();
            status.textContent = `Couldn't start: ${e.message}`;
            showToast(`Couldn't start: ${e.message}`, "err");
            return;
        }
        clearInterval(timer);
        timer = setInterval(async () => {
            let s;
            try { s = await api(`/api/tools/status?job=${cfg.job}`); } catch (e) { return; }
            bar.style.width = `${s.percent || 0}%`;
            num.textContent = `${s.percent || 0}%`;
            status.textContent = s.status || "Working…";
            if (s.running) return;
            clearInterval(timer);
            reset();
            if (s.error) {
                status.textContent = `Failed: ${s.error}`;
                showToast(`${cfg.label} failed: ${s.error}`, "err");
            } else {
                status.textContent = s.status || "Done";
                showToast(`${cfg.label} finished`, "ok");
                if (cfg.onDone) cfg.onDone(s);
            }
        }, 1000);
    };
}

function bindTools() {
    $("btn-train").addEventListener("click", startTraining);
    $("btn-stop-train").addEventListener("click", stopTraining);
    document.querySelectorAll(".tsubtab").forEach(tab => {
        tab.addEventListener("click", () => {
            document.querySelectorAll(".tsubtab").forEach(t => t.classList.toggle("active", t === tab));
            document.querySelectorAll(".tsubpanel").forEach(p => p.classList.add("d-none"));
            show(`tsub-panel-${tab.dataset.tsub}`, true);
        });
    });
    $("btn-tool-resize").addEventListener("click", runToolJob({
        btn: "btn-tool-resize", wrap: "resize-prog-wrap", bar: "resize-prog-bar", num: "resize-prog-num", status: "resize-prog-status",
        url: "/api/tools/resize", job: "resize", label: "Resize", busy: "Resizing…",
        idle: '<i class="fa-solid fa-crop-simple"></i> Resize dataset',
        body: () => ({
            size: parseInt($("tool-resize-size").value, 10) || 640,
            mode: $("tool-resize-mode").value,
            quality: parseInt($("tool-resize-quality").value, 10) || 95,
            workers: parseInt($("tool-resize-workers").value, 10) || 8,
        }),
    }));
    $("btn-tool-compress").addEventListener("click", runToolJob({
        btn: "btn-tool-compress", wrap: "compress-prog-wrap", bar: "compress-prog-bar", num: "compress-prog-num", status: "compress-prog-status",
        url: "/api/tools/compress", job: "compress", label: "Compression", busy: "Compressing…",
        idle: '<i class="fa-solid fa-file-zipper"></i> Compress and download',
        onStart: () => show("compress-download-btn-wrap", false),
        body: () => ({
            format: $("tool-compress-format").value,
            method: $("tool-compress-method").value,
            verify: $("tool-compress-verify").checked,
        }),
        onDone: s => {
            if (s.download_url) {
                $("compress-download-link").href = s.download_url;
                show("compress-download-btn-wrap", true);
                window.location.href = s.download_url;
            }
        },
    }));
}

// ═══ 18. SHORTCUT SHEET AND HELP ══════════════════════════════════════════════
/** Single source of truth for the "?" sheet and the help screen. */
const SHORTCUTS = [
    {
        group: "Images", items: [
            { keys: [["←"], ["→"]], desc: "Previous / next image" },
            { keys: [["Enter"]], desc: "Save (marks reviewed) and go to the next image" },
            { keys: [["J"]], desc: "Next image to review" },
            { keys: [["R"]], desc: "Mark reviewed without changes (press again to undo)" },
            { keys: [["Ctrl", "S"]], desc: "Save" },
            { keys: [["C"]], desc: "Copy all shapes from the previous image" },
        ],
    },
    {
        group: "Drawing", items: [
            { keys: [["D"]], desc: "Toggle draw mode" },
            { keys: [["Ctrl"]], desc: "Hold to draw while the pointer is on the image" },
            { keys: [["Right-drag"]], desc: "Draw a box" },
            { keys: [["1"], ["9"]], sep: "–", desc: "Pinned class keys: set the active class, or reassign selected shapes" },
            { keys: [["/"], ["K"]], desc: "Open the class palette (type to filter, Enter to choose, Alt+1–9 to pin)" },
            { keys: [["Q"], ["E"]], desc: "Previous / next class" },
            { keys: [["Esc"]], desc: "Close popups, cancel drawing, then deselect" },
        ],
    },
    {
        group: "Polygons", items: [
            { keys: [["Click"]], desc: "Add a point (in draw mode)" },
            { keys: [["Enter"]], desc: "Close the polygon (or click the first point, or double-click)" },
            { keys: [["Backspace"]], desc: "Remove the last point while drawing" },
            { keys: [["Alt", "Click"]], desc: "Delete a point (polygons keep at least 3)" },
            { keys: [["Double-click"]], desc: "Add a point on an edge" },
        ],
    },
    {
        group: "Editing", items: [
            { keys: [["Del"]], desc: "Delete selected shapes" },
            { keys: [["Ctrl", "Z"]], desc: "Undo" },
            { keys: [["Ctrl", "Y"], ["Ctrl", "Shift", "Z"]], desc: "Redo" },
            { keys: [["Ctrl", "A"]], desc: "Select all shapes" },
            { keys: [["Ctrl", "C"], ["Ctrl", "V"]], desc: "Copy / paste shapes (works across images)" },
            { keys: [["Ctrl", "D"]], desc: "Duplicate selected shapes" },
            { keys: [["Tab"], ["Shift", "Tab"]], desc: "Select next / previous shape" },
            { keys: [["Shift", "Arrows"]], desc: "Nudge selection 1 px (add Alt for 10 px)" },
            { keys: [["Shift", "Click"]], desc: "Add or remove a shape from the selection" },
        ],
    },
    {
        group: "View", items: [
            { keys: [["Wheel"]], desc: "Zoom at the pointer" },
            { keys: [["Space", "Drag"]], desc: "Pan (or drag with the middle button)" },
            { keys: [["0"], ["F"]], desc: "Fit image to the window" },
            { keys: [["+"], ["-"]], desc: "Zoom in / out" },
            { keys: [["H"]], desc: "Hide or show annotations" },
            { keys: [["L"]], desc: "Hide or show class labels" },
            { keys: [["?"]], desc: "Show this list" },
        ],
    },
];

function renderShortcuts(containerId) {
    const el = $(containerId);
    if (!el) return;
    el.innerHTML = SHORTCUTS.map(g => `
        <section class="shortcut-group">
          <h4>${escapeHtml(g.group)}</h4>
          ${g.items.map(it => `<div class="shortcut-row">
              <span class="shortcut-keys">${it.keys.map(combo => combo.map(k => `<kbd>${escapeHtml(k)}</kbd>`).join("+")).join(`<span class="shortcut-sep">${it.sep || "/"}</span>`)}</span>
              <span class="shortcut-desc">${escapeHtml(it.desc)}</span></div>`).join("")}
        </section>`).join("");
}

// ═══ 19. INIT ═════════════════════════════════════════════════════════════════
function bindToolbar() {
    $("btn-draw").addEventListener("click", toggleDrawMode);
    $("btn-next-unanno").addEventListener("click", jumpToNextUnannotated);
    $("btn-toggle-shapes").addEventListener("click", toggleHideShapes);
    $("btn-zoom-in").addEventListener("click", () => zoomBy(1.3));
    $("btn-zoom-out").addEventListener("click", () => zoomBy(1 / 1.3));
    $("btn-fit").addEventListener("click", fitView);
    $("btn-del-img").addEventListener("click", deleteCurrentImage);
    $("btn-save-next").addEventListener("click", saveAndNext);
    $("nav-prev").addEventListener("click", () => navigateTo(currentIndex - 1));
    $("nav-next").addEventListener("click", () => navigateTo(currentIndex + 1));
    $("det-btn").addEventListener("click", detToggle);
    $("adjust-btn").addEventListener("click", adjustToggle);
    $("adjust-pop").addEventListener("click", e => e.stopPropagation());
    $("btn-adjust-reset").addEventListener("click", resetAdjustments);
    $("history-toggle").addEventListener("click", () => {
        const p = $("history-panel");
        const open = p.classList.toggle("collapsed") === false;
        $("history-toggle").setAttribute("aria-expanded", String(open));
        $("history-toggle").querySelector("i").className = open ? "fa-solid fa-chevron-down" : "fa-solid fa-chevron-right";
        savePref("historyOpen", open ? "1" : "0");
    });
    $("det-pop").addEventListener("click", e => e.stopPropagation());
    $("btn-relabel-one").addEventListener("click", () => relabelCurrent(false));
    $("btn-relabel-all").addEventListener("click", relabelAll);
    $("ask-class-toggle").addEventListener("change", e => {
        askClassAfterDraw = e.target.checked;
        savePref("askClassAfterDraw", askClassAfterDraw ? "1" : "0");
        e.target.blur();
    });

    $("bc-bright").addEventListener("input", e => { bcBright = parseFloat(e.target.value); renderCanvas(); });
    $("bc-contrast").addEventListener("input", e => { bcContrast = parseFloat(e.target.value); renderCanvas(); });
    $("bc-bright-reset").addEventListener("click", () => { bcBright = 1; $("bc-bright").value = 1; renderCanvas(); });
    $("bc-contrast-reset").addEventListener("click", () => { bcContrast = 1; $("bc-contrast").value = 1; renderCanvas(); });
    const fType = $("extra-filter-type"), fVal = $("extra-filter-val");
    fType.addEventListener("change", () => {
        extraFilterType = fType.value;
        if (fType.value === "saturate") { fVal.max = 300; fVal.value = 100; } else { fVal.max = 100; fVal.value = fType.value === "none" ? 0 : 100; }
        extraFilterVal = parseFloat(fVal.value);
        fType.blur();
        renderCanvas();
    });
    fVal.addEventListener("input", () => {
        extraFilterVal = parseFloat(fVal.value);
        if (extraFilterType === "none") { fType.value = "saturate"; extraFilterType = "saturate"; fVal.max = 300; }
        renderCanvas();
    });
    $("extra-filter-reset").addEventListener("click", () => {
        fType.value = "none"; extraFilterType = "none"; extraFilterVal = 0; fVal.max = 100; fVal.value = 0; renderCanvas();
    });
    // Sliders give keyboard focus back to the shortcuts once released.
    document.querySelectorAll("#canvas-toolbar input[type=range]").forEach(r => r.addEventListener("change", () => r.blur()));
}

function bindSidebar() {
    document.querySelectorAll(".split-btn").forEach(tab => tab.addEventListener("click", () => switchSplit(tab.dataset.split)));
    document.querySelectorAll(".ftab").forEach(tab => tab.addEventListener("click", () => {
        setSidebarFilter(tab.dataset.filter);
        renderImageList({ autoNav: true });
        tab.blur();
    }));
    $("search").addEventListener("input", debounce(() => renderImageList({ autoNav: true }), 150));
    $("image-list-viewport").addEventListener("scroll", () => renderVisibleRows(false));
}

function bindRightPanel() {
    document.querySelectorAll(".rtab").forEach(tab => {
        tab.addEventListener("click", () => {
            document.querySelectorAll(".rtab").forEach(t => t.classList.toggle("active", t === tab));
            document.querySelectorAll(".rpanel").forEach(p => p.classList.remove("active"));
            $("panel-" + tab.dataset.panel).classList.add("active");
            if (tab.dataset.panel === "split") { updateStats(true); updateClassChart(); }
            if (tab.dataset.panel === "train") checkGpuTrainingSupport();
        });
    });
    $("add-box-btn").addEventListener("click", () => {
        if (currentIndex < 0 || !imgReady) return;
        const box = { cls: activeClass, x: 0.5, y: 0.5, w: 0.15, h: 0.15 };
        addNewShape(datasetType === "polygon" ? boxToPoly(box) : box, datasetType === "polygon" ? "Added polygon" : "Added box");
    });
    $("btn-save").addEventListener("click", () => saveAnnotations(false));
    $("btn-auto-split").addEventListener("click", doAutoSplit);
}

function bindGlobal() {
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", () => {
        if (ctrlHeld || spaceHeld) { ctrlHeld = false; spaceHeld = false; updateCursor(null); renderCanvas(); }
    });
    window.addEventListener("beforeunload", e => {
        if (dirty) { e.preventDefault(); e.returnValue = ""; }
    });
    document.addEventListener("click", e => {
        mcClose();
        detClose();
        adjustClose();
        if (!e.target.closest("#grid-class-anchor")) closeGridClassPop();
        if (palette.open && !e.target.closest("#class-bar") && e.target !== canvas) closePalette();
        // Mouse clicks should not leave focus on buttons (Enter/Space would re-trigger them).
        if (e.detail > 0) { const b = e.target.closest && e.target.closest("button"); if (b) b.blur(); }
    });
    $("autosave-toggle").addEventListener("change", e => {
        autoSave = e.target.checked;
        $("autosave-label").classList.toggle("on", autoSave);
        e.target.blur();
        showToast(autoSave ? "Auto-save on" : "Auto-save off. You'll be asked before unsaved changes are left behind.");
        if (autoSave) triggerAutoSave();
    });
    $("btn-goto-setup").addEventListener("click", goToSetup);
    $("btn-fullscreen").addEventListener("click", () => {
        if (!document.fullscreenElement) document.documentElement.requestFullscreen().catch(err => showToast(`Full screen isn't available: ${err.message}`, "err"));
        else document.exitFullscreen();
    });
    document.addEventListener("fullscreenchange", () => {
        $("btn-fullscreen").querySelector("i").className = document.fullscreenElement ? "fa-solid fa-minimize" : "fa-solid fa-maximize";
    });
    $("btn-shortcuts").addEventListener("click", () => openModal("shortcuts-modal"));
    $("sb-review").addEventListener("click", toggleReviewed);
    setInterval(() => {
        const s = Math.floor((Date.now() - _sessionStart) / 1000);
        const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
        $("session-timer").textContent = (h ? `${h}:` : "") + `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
    }, 1000);
}

function init() {
    loadPrefs();
    bindModalChrome();
    bindSettingsControls();
    bindModelClassPicker();
    bindSetup();
    bindCanvas();
    bindClassBar();
    bindClassesManager();
    bindToolbar();
    bindSidebar();
    bindRightPanel();
    bindGrid();
    updateRoiButtons();
    bindAddImages();
    bindTools();
    bindGlobal();
    $("loading-overlay").addEventListener("click", () => { if ($("loading-overlay").classList.contains("is-error") && imgReady) hideLoading(); });
    renderShortcuts("shortcuts-body");
    renderShortcuts("help-shortcuts");
    setClasses([]);
    renderHistoryPanel();
    updateBoxPanel();
    updateStatusBar();
    showScreen("setup");
    initSetup();
}

init();
