
// ─── DYNAMIC CLASSES (from server, updated after model load) ─────────────────
let CLASS_NAMES = [];
const COLORS = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#06b6d4", "#3b82f6", "#a855f7", "#ec4899",
    "#14b8a6", "#f43f5e", "#84cc16", "#fb923c"];

// ─── STATE ───────────────────────────────────────────────────────────────────
let currentSplit = "train";
let allImages = [];
let filteredImages = [];
let currentIndex = -1;
let boxes = [];
let selectedBoxes = [];
let dirty = false;
let drawMode = false;
let autoSave = true;
let sidebarFilter = "all";   // 'all' | 'done' | 'pending'
let datasetType = "bbox"; // 'bbox' | 'polygon'

// Session timer
let _sessionStart = Date.now();
setInterval(() => {
    const s = Math.floor((Date.now() - _sessionStart) / 1000);
    const mm = String(Math.floor(s / 60)).padStart(2, '0');
    const ss = String(s % 60).padStart(2, '0');
    const el = document.getElementById('session-timer');
    if (el) el.textContent = `${mm}:${ss}`;
}, 1000);

// Undo/Redo Stack (Command Pattern)
class StateSnapshotCommand {
    constructor(prevBoxes, newBoxes, actionName) {
        this.prevBoxes = JSON.parse(JSON.stringify(prevBoxes));
        this.newBoxes = JSON.parse(JSON.stringify(newBoxes));
        this.actionName = actionName || "Edited annotations";
        
        const now = new Date();
        this.timestamp = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}:${String(now.getSeconds()).padStart(2,'0')}`;
    }
    execute() { boxes = JSON.parse(JSON.stringify(this.newBoxes)); }
    undo() { boxes = JSON.parse(JSON.stringify(this.prevBoxes)); }
}

class CommandManager {
    constructor() { this.undoStack = []; this.redoStack = []; this.maxHistory = 20; }
    pushState(prevBoxes, newBoxes, actionName) {
        if (JSON.stringify(prevBoxes) === JSON.stringify(newBoxes)) return;
        this.undoStack.push(new StateSnapshotCommand(prevBoxes, newBoxes, actionName));
        if (this.undoStack.length > this.maxHistory) this.undoStack.shift();
        this.redoStack = [];
        renderHistoryPanel();
    }
    undo() {
        if (this.undoStack.length === 0) return;
        const cmd = this.undoStack.pop();
        cmd.undo();
        this.redoStack.push(cmd);
        selectedBoxes = []; dirty = true; triggerAutoSave(); updateBoxPanel(); renderCanvas(); renderHistoryPanel();
        showToast("↩ Undo");
    }
    redo() {
        if (this.redoStack.length === 0) return;
        const cmd = this.redoStack.pop();
        cmd.execute();
        this.undoStack.push(cmd);
        selectedBoxes = []; dirty = true; triggerAutoSave(); updateBoxPanel(); renderCanvas(); renderHistoryPanel();
        showToast("↪ Redo");
    }
    clear() { this.undoStack = []; this.redoStack = []; renderHistoryPanel(); }
}
const cmdManager = new CommandManager();

function renderHistoryPanel() {
    const list = document.getElementById("history-list");
    if (!list) return;
    list.innerHTML = "";
    if (cmdManager.undoStack.length === 0) {
        list.innerHTML = '<div style="font-size:11px;color:#475569;">No edits yet for this image.</div>';
        return;
    }
    cmdManager.undoStack.forEach((cmd, i) => {
        const item = document.createElement("div");
        item.style.fontSize = "11px";
        item.style.color = i === cmdManager.undoStack.length - 1 ? "#86efac" : "#94a3b8";
        item.style.fontFamily = "'JetBrains Mono', monospace";
        item.innerHTML = `<span style="color:#475569;">${cmd.timestamp}</span> &nbsp; ${cmd.actionName}`;
        list.appendChild(item);
    });
    // Auto scroll to bottom
    const panel = document.getElementById("history-panel");
    if (panel) panel.scrollTop = panel.scrollHeight;
}

let _pendingUndoState = null;
function pushUndo() {
    _pendingUndoState = JSON.parse(JSON.stringify(boxes));
}
function commitUndo(actionName = "Edited annotations") {
    if (_pendingUndoState) {
        cmdManager.pushState(_pendingUndoState, boxes, actionName);
        _pendingUndoState = null;
    }
}
function doUndo() { cmdManager.undo(); }
function doRedo() { cmdManager.redo(); }

// Canvas
const canvas = document.getElementById("annot-canvas");
const ctx = canvas.getContext("2d");
const imgEl = new Image();
let imgNatW = 1, imgNatH = 1;
const HANDLE = 9;
let drag = { on: false, type: null, bIdx: -1, sx: 0, sy: 0, ob: null, ds: null };

// Zoom / Pan state
let zoomScale = 1.0;
let panX = 0, panY = 0;
let isPanning = false;
let panStart = { x: 0, y: 0 };
let isMouseInCanvas = false;

// Brightness/Contrast
let bcBright = 1.0, bcContrast = 1.0;

// Pre-load next images
let _preloadSlots = [new Image(), new Image(), new Image()];

// Virtualization
const ITEM_H = 42;
let _lastScrollTop = -1;

// _loadId for stale request guard
let _loadId = 0;
let _imgReady = false, _labelsReady = false;

// ─── SETUP SCREEN ────────────────────────────────────────────────────────────
async function initSetup() {
    const st = await fetch("/api/state").then(r => r.json()).catch(() => ({ ready: false }));
    if (st.ready) {
        datasetType = st.dataset_type || "bbox";
        const c = st.counts;
        document.getElementById("existing-text").textContent =
            `Dataset found — Train:${c.train.total}  Valid:${c.valid.total}  Test:${c.test.total}`;
        document.getElementById("existing-banner").style.display = "flex";
    }

    // Load available model files (all formats)
    const mdata = await fetch("/api/models").then(r => r.json()).catch(() => ({ files: [], presets: [] }));
    const msel = document.getElementById("model-select");
    const mpre = document.getElementById("model-preset");
    
    msel.innerHTML = '<option value="">— select local model —</option>';
    if (mpre) mpre.innerHTML = '<option value="">— select preset to download —</option>';
    
    // Group by format
    const byExt = {};
    mdata.files.forEach(f => {
        const ext = f.split('.').pop().toLowerCase();
        (byExt[ext] = byExt[ext] || []).push(f);
    });
    Object.entries(byExt).forEach(([ext, files]) => {
        const grp = document.createElement('optgroup');
        grp.label = `.${ext.toUpperCase()} files`;
        files.forEach(f => { const o = document.createElement('option'); o.value = f; o.textContent = f; grp.appendChild(o); });
        msel.appendChild(grp);
    });
    
    if (mpre) {
        mdata.presets.forEach(f => {
            const o = document.createElement('option'); o.value = f; o.textContent = f; mpre.appendChild(o);
        });
        
        // Mutually exclusive
        msel.addEventListener("change", () => { if(msel.value) mpre.value = ""; });
        mpre.addEventListener("change", () => { if(mpre.value) msel.value = ""; });
    }

    // Show auto-detected device info
    const devInfo = await fetch("/api/device_info").then(r => r.json()).catch(() => ({}));
    const devEl = document.getElementById("device-autodetect");
    if (devInfo.cuda && devInfo.cuda.length > 0) {
        const gpuNames = devInfo.cuda.map(g => `GPU ${g.id}: ${g.name} (${g.free_gb}/${g.total_gb} GB free)`).join(', ');
        devEl.innerHTML = `<span style="color:#86efac;font-weight:700;">🟢 GPU detected: ${gpuNames} — will auto-select GPU for inference</span>`;
    } else if (devInfo.mps) {
        devEl.innerHTML = `<span style="color:#86efac;font-weight:700;">🟢 Apple MPS detected — will auto-select MPS for inference</span>`;
    } else {
        devEl.innerHTML = `<span style="color:#f59e0b;">⚡ No GPU detected — will use CPU for inference</span>`;
    }
    
    // Helper to check dataset info
    async function checkDatasetInfo(path) {
        const inputExisting = document.getElementById("existing-dataset-input");
        const btnLoadExisting = document.getElementById("btn-load-existing");
        const infoBox = document.getElementById("existing-info");
        
        inputExisting.value = path;
        btnLoadExisting.disabled = false;
        infoBox.style.display = "block";
        infoBox.textContent = "Checking dataset...";
        infoBox.style.color = "#94a3b8";
        
        try {
            const info = await fetch(`/api/dataset_info?path=${encodeURIComponent(path)}`).then(r => r.json());
            if (info.valid) {
                infoBox.style.color = "#86efac";
                infoBox.innerHTML = `✅ Found Dataset — <b>Type:</b> ${info.type === 'polygon' ? 'Polygon (Seg)' : 'Bounding Box (Det)'} | <b>Images:</b> ${info.count}`;
            } else {
                infoBox.style.color = "#f87171";
                infoBox.innerHTML = `⚠️ No valid state.db found. Are you sure this is a dataset folder?`;
            }
        } catch(e) {
            infoBox.style.display = "none";
        }
    }

    // Populate existing datasets dropdown
    const dsel = document.getElementById("existing-dataset-select");
    const ddata = await fetch("/api/datasets").then(r => r.json()).catch(() => []);
    dsel.innerHTML = '<option value="">— choose an existing dataset from root —</option>';
    ddata.forEach(d => {
        const o = document.createElement("option");
        o.value = d.path;
        o.textContent = d.name;
        dsel.appendChild(o);
    });
    
    if (dsel) {
        dsel.addEventListener("change", () => {
            if (dsel.value) {
                checkDatasetInfo(dsel.value);
            }
        });
    }

    // Setup Browse Button for Existing Datasets
    const btnBrowseExisting = document.getElementById("btn-browse-existing");
    if (btnBrowseExisting) {
        btnBrowseExisting.addEventListener("click", async () => {
            const res = await fetch("/api/browse_folder").then(r => r.json());
            if (res.path) {
                if (dsel) dsel.value = "";
                checkDatasetInfo(res.path);
            }
        });
    }

    // Setup Browse Button for Source Folders
    const btnBrowseSrc = document.getElementById("btn-browse-src");
    const inputSrc = document.getElementById("src-input");
    const btnIngest = document.getElementById("btn-ingest");
    if (btnBrowseSrc) {
        btnBrowseSrc.addEventListener("click", async () => {
            const res = await fetch("/api/browse_folder").then(r => r.json());
            if (res.path) {
                inputSrc.value = res.path;
                btnIngest.disabled = false;
                const badge = document.getElementById("img-badge");
                badge.innerHTML = `<span class="badge badge-blue">${res.path.replace(/\\\\/g, '/').split('/').pop()}</span>`;
                badge.style.display = "block";
                
                // Fetch info
                const infoBox = document.getElementById("src-info");
                infoBox.style.display = "block";
                infoBox.textContent = "Scanning folder...";
                infoBox.style.color = "#94a3b8";
                
                try {
                    const info = await fetch(`/api/source_info?path=${encodeURIComponent(res.path)}`).then(r => r.json());
                    if (info.valid) {
                        if (info.count > 0) {
                            infoBox.style.color = "#86efac";
                            infoBox.innerHTML = `✅ Found <b>${info.count}</b> images ready to load.`;
                        } else {
                            infoBox.style.color = "#f59e0b";
                            infoBox.innerHTML = `⚠️ No supported images (.jpg, .png, etc.) found in this folder.`;
                        }
                    } else {
                        infoBox.style.color = "#f87171";
                        infoBox.innerHTML = `❌ Failed to read folder contents.`;
                    }
                } catch(e) {
                    infoBox.style.display = "none";
                }
            }
        });
    }

    // Setup Browse Button for Appending Source Folders
    const btnBrowseAddSrc = document.getElementById("btn-browse-add-src");
    const inputAddSrc = document.getElementById("add-src-input");
    if (btnBrowseAddSrc) {
        btnBrowseAddSrc.addEventListener("click", async () => {
            const res = await fetch("/api/browse_folder").then(r => r.json());
            if (res.path) {
                inputAddSrc.value = res.path;
            }
        });
    }

    // Training panel custom model list
    const csel = document.getElementById("tr-custom");
    csel.innerHTML = '<option value="">— use preset above —</option>';
    mdata.files.forEach(f => {
        const o = document.createElement("option"); o.value = f; o.textContent = f; csel.appendChild(o);
    });
}

// ─── LOAD MODEL BUTTON ───────────────────────────────────────────────────────
document.getElementById("btn-load-model").addEventListener("click", async () => {
    const msel = document.getElementById("model-select").value;
    const mpre = document.getElementById("model-preset") ? document.getElementById("model-preset").value : "";
    const modelPath = msel || mpre;
    
    if (!modelPath) {
        showToast("❌ Please select a model to load", true);
        return;
    }

    const device = document.getElementById("device-select").value;
    const btn = document.getElementById("btn-load-model");
    btn.disabled = true; 
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Downloading & Loading...';

    const res = await fetch("/api/set_model", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_path: modelPath, device })
    });
    const data = await res.json();
    btn.disabled = false; 
    btn.innerHTML = '<i class="fa-solid fa-bolt"></i> Load Model';

    if (data.error) { showToast("❌ " + data.error, true); return; }

    CLASS_NAMES = data.classes;
    buildClassDropdowns();

    const preview = document.getElementById("classes-preview");
    const tags = document.getElementById("classes-tags");
    tags.innerHTML = "";
    data.classes.forEach((c, i) => {
        tags.innerHTML += `<span class="badge badge-blue" style="background:${COLORS[i % COLORS.length]}22;border:1px solid ${COLORS[i % COLORS.length]}44;color:${COLORS[i % COLORS.length]}">${i}: ${c}</span>`;
    });
    preview.style.display = "block";
    showToast(`✅ Loaded ${data.classes.length} classes  |  Device: ${data.device}`, false, "ok");

    // Update device chip in topbar
    const chip = document.getElementById('device-chip');
    if (chip) {
        const isGpu = data.device !== 'cpu' && data.device !== 'mps';
        chip.textContent = isGpu ? `⚡ GPU ${data.device}` : data.device === 'mps' ? '🍎 MPS' : '🖥 CPU';
        chip.className = 'device-chip ' + (isGpu || data.device === 'mps' ? 'gpu' : 'cpu');
    }
});

// Build all class dropdowns dynamically
function buildClassDropdowns() {
    // Popup selector
    const psel = document.getElementById("popup-cls-sel");
    psel.innerHTML = "";
    CLASS_NAMES.forEach((c, i) => {
        const o = document.createElement("option"); o.value = i; o.textContent = `${i}: ${c}`; psel.appendChild(o);
    });

    // Grid Class filter selector
    const gcsel = document.getElementById("grid-class-filter");
    if (gcsel) {
        const prev = gcsel.value;
        gcsel.innerHTML = '<option value="">Any Class</option>';
        CLASS_NAMES.forEach((c, i) => {
            const o = document.createElement("option"); o.value = i; o.textContent = `[${i + 1}] ${c}`; gcsel.appendChild(o);
        });
        gcsel.value = prev;
    }

    // Dynamic Class hotkeys
    const chk = document.getElementById("dynamic-class-hotkeys");
    if (chk) {
        chk.innerHTML = "";
        CLASS_NAMES.forEach((c, i) => {
            if (i < 9) {
                chk.innerHTML += `<div style="display: flex; align-items: center; gap: 6px;"><span style="background: #1e2535; color: #e2e8f0; font-family: 'JetBrains Mono', monospace; font-weight: 700; padding: 2px 6px; border-radius: 4px; font-size: 10px;">${i + 1}</span> <span style="color:${COLORS[i % COLORS.length]}; font-weight: 600;">${c}</span></div>`;
            }
        });
    }

    // Sidebar box selectors are rebuilt in updateBoxPanel
}

document.getElementById("btn-open").addEventListener("click", openAnnotator);
document.getElementById("btn-goto-setup").addEventListener("click", () => {
    document.getElementById("app").style.display = "none";
    document.getElementById("setup-screen").classList.remove("hidden");
    initSetup();
});

document.getElementById("btn-load-existing").addEventListener("click", async () => {
    const src = document.getElementById("existing-dataset-input").value;
    if (!src) return;
    document.getElementById("btn-load-existing").disabled = true;
    document.getElementById("btn-load-existing").innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Loading...`;
    
    const res = await fetch("/api/load_existing_dataset", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_dir: src })
    });
    
    if (res.ok) {
        openAnnotator();
    } else {
        const err = await res.json();
        showToast("❌ Error: " + err.error, true);
        document.getElementById("btn-load-existing").disabled = false;
        document.getElementById("btn-load-existing").innerHTML = `<i class="fa-solid fa-folder-open"></i> Open Selected Dataset`;
    }
});

document.getElementById("btn-ingest").addEventListener("click", async () => {
    const src = document.getElementById("src-input").value;
    const type = document.getElementById("dataset-type-select").value;
    if (!src) return;
    document.getElementById("btn-ingest").disabled = true;
    document.getElementById("ingest-prog").style.display = "block";
    document.getElementById("ip-status").textContent = "Starting…";

    const { job_id } = await fetch("/api/ingest", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_dir: src, dataset_type: type })
    }).then(r => r.json());

    const es = new EventSource(`/api/progress/${job_id}`);
    es.onmessage = e => {
        const p = JSON.parse(e.data);
        const pct = p.total > 0 ? Math.round(p.done / p.total * 100) : 0;
        document.getElementById("ip-bar").style.width = pct + "%";
        document.getElementById("ip-num").textContent = `${p.done} / ${p.total}`;
        document.getElementById("ip-status").innerHTML =
            p.status === "linking" ? `<i class="fa-solid fa-link"></i> Linking images… ${pct}%` :
                p.status === "copying" ? `<i class="fa-solid fa-folder-open"></i> Copying images… ${pct}%` : p.status;
        if (p.status === "done") {
            es.close();
            document.getElementById("ip-status").innerHTML = `<i class="fa-solid fa-check-circle"></i> Done! Opening annotator…`;
            setTimeout(openAnnotator, 600);
        }
        if (String(p.status).startsWith("error")) {
            es.close();
            document.getElementById("ip-status").innerHTML = `<i class="fa-solid fa-circle-xmark"></i> ` + p.status;
            document.getElementById("btn-ingest").disabled = false;
        }
    };
});

async function openAnnotator() {
    // Fetch classes from backend if they aren't loaded yet
    if (CLASS_NAMES.length === 0) {
        const res = await fetch("/api/classes").then(r => r.json()).catch(() => ({ classes: [] }));
        if (res.classes && res.classes.length > 0) {
            CLASS_NAMES = res.classes;
        }
    }

    document.getElementById("setup-screen").classList.add("hidden");
    document.getElementById("app").style.display = "grid";
    buildClassDropdowns();
    _sessionStart = Date.now();
    // Restore last active split (default to train)
    const lastSplit = localStorage.getItem("lastSplit") || "train";
    currentSplit = lastSplit;
    // Highlight the correct split tab
    document.querySelectorAll(".stab").forEach(t => t.classList.toggle("active", t.dataset.split === lastSplit));
    await loadSplitImages(lastSplit);
    updateStats();
}

// ─── AUTO-SAVE TOGGLE ────────────────────────────────────────────────────────
document.getElementById("autosave-toggle").addEventListener("change", e => {
    autoSave = e.target.checked;
    const lbl = document.getElementById("autosave-label");
    lbl.classList.toggle("on", autoSave);
    showToast(autoSave ? "🟢 Auto-Save ON" : "⚪ Auto-Save OFF");
});

function triggerAutoSave() {
    if (autoSave && currentIndex >= 0) saveAnnotations(true);
}

// ─── ZOOM & PAN ──────────────────────────────────────────────────────────────
function getTransform() {
    // Returns the effective drawing parameters (origin, scale)
    const wW = canvas.width, wH = canvas.height;
    const baseScale = Math.min((wW - 64) / imgNatW, (wH - 6) / imgNatH, 1);
    const scale = baseScale * zoomScale;
    const cx = wW / 2 + panX, cy = wH / 2 + panY;
    const ox = cx - (imgNatW * scale) / 2;
    const oy = cy - (imgNatH * scale) / 2;
    return { ox, oy, scale };
}

function fitView() {
    zoomScale = 1.0; panX = 0; panY = 0;
    const wrap = document.getElementById("canvas-wrapper");
    canvas.width = wrap.clientWidth;
    canvas.height = wrap.clientHeight;
    renderCanvas();
}

function n2d(b) {
    const { ox, oy, scale } = getTransform();
    return {
        x: ox + (b.x - b.w / 2) * imgNatW * scale,
        y: oy + (b.y - b.h / 2) * imgNatH * scale,
        w: b.w * imgNatW * scale,
        h: b.h * imgNatH * scale
    };
}
function d2n(px, py, pw, ph) {
    const { ox, oy, scale } = getTransform();
    return {
        x: (px - ox + pw / 2) / (imgNatW * scale),
        y: (py - oy + ph / 2) / (imgNatH * scale),
        w: pw / (imgNatW * scale),
        h: ph / (imgNatH * scale)
    };
}

// ─── CANVAS RENDER ───────────────────────────────────────────────────────────
let _rafPending = false;
let _crossX = -1, _crossY = -1;

function renderCanvas() {
    if (_rafPending) return;
    _rafPending = true;
    requestAnimationFrame(_doRender);
}

function _doRender() {
    _rafPending = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!imgEl.src || !imgEl.complete) return;

    const { ox, oy, scale } = getTransform();
    const dW = imgNatW * scale, dH = imgNatH * scale;

    // Apply brightness/contrast via global composite
    ctx.filter = `brightness(${bcBright}) contrast(${bcContrast})`;
    ctx.drawImage(imgEl, ox, oy, dW, dH);
    ctx.filter = "none";

    // Draw boxes or polygons
    boxes.forEach((b, i) => {
        const col = COLORS[b.cls % COLORS.length];
        const sel = selectedBoxes.includes(i);
        ctx.strokeStyle = sel ? "#fff" : col; ctx.lineWidth = sel ? 2.5 : 1.5;
        ctx.fillStyle = col + (sel ? "4a" : "25");
        
        let tx, ty;
        
        if (datasetType === "polygon" && b.pts) {
            ctx.beginPath();
            b.pts.forEach((pt, pi) => {
                const px = ox + pt.x * imgNatW * scale;
                const py = oy + pt.y * imgNatH * scale;
                if (pi === 0) { ctx.moveTo(px, py); tx = px; ty = py; }
                else { ctx.lineTo(px, py); if (py < ty) { tx = px; ty = py; } }
            });
            ctx.closePath();
            ctx.fill(); ctx.stroke();
            
            if (sel) {
                b.pts.forEach((pt) => {
                    const px = ox + pt.x * imgNatW * scale;
                    const py = oy + pt.y * imgNatH * scale;
                    ctx.fillStyle = "#fff"; ctx.fillRect(px - HANDLE / 2, py - HANDLE / 2, HANDLE, HANDLE);
                    ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.strokeRect(px - HANDLE / 2, py - HANDLE / 2, HANDLE, HANDLE);
                });
            }
        } else {
            const d = n2d(b);
            ctx.strokeRect(d.x, d.y, d.w, d.h);
            ctx.fillRect(d.x, d.y, d.w, d.h);
            tx = d.x; ty = d.y;
            if (sel) {
                [[d.x, d.y], [d.x + d.w, d.y], [d.x, d.y + d.h], [d.x + d.w, d.y + d.h]].forEach(([cx, cy]) => {
                    ctx.fillStyle = "#fff"; ctx.fillRect(cx - HANDLE / 2, cy - HANDLE / 2, HANDLE, HANDLE);
                    ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.strokeRect(cx - HANDLE / 2, cy - HANDLE / 2, HANDLE, HANDLE);
                });
            }
        }
        
        const lbl = `[${b.cls + 1}] ${CLASS_NAMES[b.cls] || b.cls}`;
        ctx.font = "bold 11px Inter,sans-serif";
        const tw = ctx.measureText(lbl).width;
        ty = ty > 16 ? ty - 3 : ty + 15;
        ctx.fillStyle = col; ctx.fillRect(tx, ty - 12, tw + 8, 14);
        ctx.fillStyle = "#fff"; ctx.fillText(lbl, tx + 4, ty);
    });

    // Crosshairs
    if (_crossX >= 0 && !drag.on) {
        ctx.save();
        ctx.globalCompositeOperation = 'difference';
        ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(_crossX, 0); ctx.lineTo(_crossX, canvas.height); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, _crossY); ctx.lineTo(canvas.width, _crossY); ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
    }
}

// ─── CANVAS MOUSE ────────────────────────────────────────────────────────────
function hitHandle(i, mx, my) {
    const d = n2d(boxes[i]);
    const pts = { tl: [d.x, d.y], tr: [d.x + d.w, d.y], bl: [d.x, d.y + d.h], br: [d.x + d.w, d.y + d.h] };
    for (const [k, [cx, cy]] of Object.entries(pts))
        if (Math.abs(mx - cx) <= HANDLE && Math.abs(my - cy) <= HANDLE) return k;
    return null;
}
function hitBox(mx, my) {
    for (let i = boxes.length - 1; i >= 0; i--) { const d = n2d(boxes[i]); if (mx >= d.x && mx <= d.x + d.w && my >= d.y && my <= d.y + d.h) return i; }
    return -1;
}
function getPos(e) { const r = canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }

canvas.addEventListener("wheel", e => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    const { x, y } = getPos(e);
    // Correct zoom around cursor math:
    // newPan = oldPan * factor - (mouse - center) * (factor - 1)
    panX = panX * factor - (x - canvas.width / 2) * (factor - 1);
    panY = panY * factor - (y - canvas.height / 2) * (factor - 1);
    zoomScale = Math.max(0.2, Math.min(20, zoomScale * factor));
    renderCanvas();
}, { passive: false });

canvas.addEventListener("mousedown", e => {
    if (currentIndex < 0) return;
    const { x, y } = getPos(e);

    // Middle button = pan
    if (e.button === 1) {
        e.preventDefault();
        isPanning = true;
        panStart = { x: x - panX, y: y - panY };
        canvas.style.cursor = "grabbing";
        return;
    }

    if (drawMode) { drag = { on: true, type: "draw", bIdx: -1, sx: x, sy: y, ds: { x, y } }; return; }

    if (selectedBoxes.length > 0 && !e.shiftKey) {
        const h = hitHandle(selectedBoxes[0], x, y);
        if (h && selectedBoxes.length === 1) { drag = { on: true, type: h, bIdx: selectedBoxes[0], sx: x, sy: y, ob: { ...boxes[selectedBoxes[0]] } }; return; }
    }
    const hit = hitBox(x, y);
    if (hit >= 0) {
        pushUndo();
        if (e.shiftKey) {
            if (selectedBoxes.includes(hit)) selectedBoxes = selectedBoxes.filter(i => i !== hit);
            else selectedBoxes.push(hit);
        } else {
            selectedBoxes = [hit];
        }
        if (selectedBoxes.length === 1) {
            drag = { on: true, type: "move", bIdx: hit, sx: x, sy: y, ob: { ...boxes[hit] } };
        }
        updateBoxPanel(); renderCanvas(); return;
    }
    selectedBoxes = []; updateBoxPanel(); renderCanvas();
});

canvas.addEventListener("mousemove", e => {
    const { x, y } = getPos(e);
    _crossX = x; _crossY = y;

    if (isPanning) {
        panX = x - panStart.x; panY = y - panStart.y;
        renderCanvas(); return;
    }

    if (!drag.on) {
        canvas.style.cursor = drawMode ? "crosshair" :
            selectedBoxes.length === 1 && hitHandle(selectedBoxes[0], x, y) ? "nwse-resize" :
                hitBox(x, y) >= 0 ? "move" : "default";
        renderCanvas(); return;
    }

    const dx = x - drag.sx, dy = y - drag.sy;
    if (drag.type === "draw") {
        _doRender();
        ctx.strokeStyle = "#60a5fa"; ctx.lineWidth = 2; ctx.setLineDash([5, 3]);
        ctx.strokeRect(drag.ds.x, drag.ds.y, x - drag.ds.x, y - drag.ds.y);
        ctx.setLineDash([]); return;
    }

    const ob = drag.ob;
    const { ox, oy, scale } = getTransform();
    let px = (ob.x - ob.w / 2) * imgNatW * scale + ox, py = (ob.y - ob.h / 2) * imgNatH * scale + oy,
        pw = ob.w * imgNatW * scale, ph = ob.h * imgNatH * scale;

    if (drag.type === "move") { px += dx; py += dy; }
    else if (drag.type === "tl") { px += dx; py += dy; pw -= dx; ph -= dy; }
    else if (drag.type === "tr") { pw += dx; py += dy; ph -= dy; }
    else if (drag.type === "bl") { px += dx; pw -= dx; ph += dy; }
    else if (drag.type === "br") { pw += dx; ph += dy; }
    if (pw < 5) pw = 5; if (ph < 5) ph = 5;
    px = Math.max(ox, Math.min(px, ox + imgNatW * scale - pw));
    py = Math.max(oy, Math.min(py, oy + imgNatH * scale - ph));
    const n = d2n(px, py, pw, ph);
    n.x = Math.max(n.w / 2, Math.min(1 - n.w / 2, n.x));
    n.y = Math.max(n.h / 2, Math.min(1 - n.h / 2, n.y));
    boxes[drag.bIdx] = { ...boxes[drag.bIdx], ...n };
    dirty = true; renderCanvas();
});

canvas.addEventListener("mouseup", e => {
    if (isPanning) { isPanning = false; canvas.style.cursor = "default"; return; }
    if (!drag.on) return;
    const { x, y } = getPos(e);
    if (drag.type === "draw") {
        const ew = Math.abs(x - drag.ds.x), eh = Math.abs(y - drag.ds.y);
        if (ew > 8 && eh > 8) {
            const sx = Math.min(x, drag.ds.x), sy = Math.min(y, drag.ds.y);
            const n = d2n(sx, sy, ew, eh);
            n.x = Math.max(n.w / 2, Math.min(1 - n.w / 2, n.x));
            n.y = Math.max(n.h / 2, Math.min(1 - n.h / 2, n.y));
            pushUndo();
            boxes.push({ id: boxes.length, cls: 0, ...n });
            commitUndo("Drew new box");
            selectedBoxes = [boxes.length - 1]; dirty = true;
            // Show inline popup
            showClassPopup(drag.ds.x, drag.ds.y, x, y);
        }
        setDrawMode(false);
    } else {
        triggerAutoSave();
        commitUndo("Moved/Resized box");
    }
    drag = { on: false }; updateBoxPanel(); renderCanvas();
});

// Help Modal
document.getElementById("btn-help-setup")?.addEventListener("click", () => {
    document.getElementById("help-modal").style.display = "flex";
});
document.getElementById("help-modal")?.addEventListener("click", (e) => {
    if (e.target.id === "help-modal") {
        document.getElementById("help-modal").style.display = "none";
    }
});

canvas.addEventListener("mouseenter", () => { isMouseInCanvas = true; });
canvas.addEventListener("mouseleave", () => { drag.on = false; isPanning = false; _crossX = -1; _crossY = -1; isMouseInCanvas = false; renderCanvas(); });
window.addEventListener("resize", () => { if (imgEl.src) fitView(); });

// ─── INLINE CLASS POPUP ───────────────────────────────────────────────────────
function showClassPopup(x1, y1, x2, y2) {
    const popup = document.getElementById("class-popup");
    const wrapper = document.getElementById("canvas-wrapper");
    const wr = wrapper.getBoundingClientRect();
    const cr = canvas.getBoundingClientRect();

    // Position popup near the drawn box
    let left = Math.max(x1, x2) - cr.left + wr.left + 8;
    let top = Math.min(y1, y2) - cr.top + wr.top;

    if (left + 190 > wr.right) left = Math.min(x1, x2) - cr.left + wr.left - 195;
    if (top + 130 > wr.bottom) top = wr.bottom - 130;

    let finalLeft = left - wr.left;
    let finalTop = top - wr.top;

    // Prevent hiding at the top or left side
    if (finalLeft < 8) finalLeft = 8;
    if (finalTop < 8) finalTop = 8;

    popup.style.left = finalLeft + "px";
    popup.style.top = finalTop + "px";
    popup.style.display = "block";

    // Pre-select class 0
    document.getElementById("popup-cls-sel").value = "0";
}

document.getElementById("popup-confirm").addEventListener("click", () => {
    if (boxes.length > 0 && selectedBoxes.length > 0) {
        pushUndo();
        const cls = parseInt(document.getElementById("popup-cls-sel").value);
        selectedBoxes.forEach(i => boxes[i].cls = cls);
        commitUndo("Set class from popup");
        dirty = true;
        triggerAutoSave();
        updateBoxPanel(); renderCanvas();
    }
    document.getElementById("class-popup").style.display = "none";
});

document.getElementById("popup-cancel").addEventListener("click", () => {
    if (boxes.length > 0 && selectedBoxes.length > 0) {
        pushUndo();
        boxes = boxes.filter((_, i) => !selectedBoxes.includes(i));
        commitUndo("Canceled new box");
        selectedBoxes = boxes.length > 0 ? [boxes.length - 1] : [];
    }
    document.getElementById("class-popup").style.display = "none";
    dirty = true; updateBoxPanel(); renderCanvas();
});

// ─── KEYBOARD ────────────────────────────────────────────────────────────────
window.addEventListener("keyup", e => {
    if (e.key === "Control") {
        setDrawMode(false);
    }
});

window.addEventListener("keydown", e => {
    if (e.key === "Control" && isMouseInCanvas) {
        setDrawMode(true);
    }

    // Close popup with Escape
    if (e.key === "Escape") {
        document.getElementById("class-popup").style.display = "none"; return;
    }
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

    // Modal keyboard navigation
    if (document.getElementById("image-modal").style.display === "flex") {
        if (e.key === "Escape") { closeImageModal(); return; }
        if (e.key === "ArrowRight" || e.key === "n") {
            const btn = document.getElementById("btn-modal-next");
            if (btn && !btn.disabled) btn.click();
        }
        if (e.key === "ArrowLeft" || e.key === "p") {
            const btn = document.getElementById("btn-modal-prev");
            if (btn && !btn.disabled) btn.click();
        }
        return;
    }

    if (e.key === "ArrowRight" || e.key === "n") navigateTo(currentIndex + 1);
    if (e.key === "ArrowLeft" || e.key === "p") navigateTo(currentIndex - 1);
    if (e.key === "s" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); saveAnnotations(); }
    if (e.key === "z" && (e.ctrlKey || e.metaKey)) { 
        e.preventDefault(); 
        if (e.shiftKey) { doRedo(); } else { doUndo(); } 
    }
    if (e.key === "y" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); doRedo(); }
    if (e.key === "d") setDrawMode(!drawMode);
    if (e.key === "f") fitView();
    if (e.key === "=" || e.key === "+") { zoomScale = Math.min(20, zoomScale * 1.2); renderCanvas(); }
    if (e.key === "-") { zoomScale = Math.max(0.2, zoomScale / 1.2); renderCanvas(); }
    if (e.key === "Enter") saveAndNext();
    if ((e.key === "Delete" || e.key === "Backspace") && selectedBoxes.length > 0) { 
        pushUndo(); 
        boxes = boxes.filter((_, i) => !selectedBoxes.includes(i));
        selectedBoxes = [];
        commitUndo("Deleted box(es)"); 
        dirty = true; updateBoxPanel(); renderCanvas();
    }

    // Number keys 1-9 → set class of selected box
    const num = parseInt(e.key);
    if (!isNaN(num) && num >= 1 && num <= CLASS_NAMES.length && selectedBoxes.length > 0) {
        pushUndo();
        selectedBoxes.forEach(i => boxes[i].cls = num - 1);
        dirty = true; triggerAutoSave(); updateBoxPanel(); renderCanvas();
        commitUndo("Changed class via hotkey");
    }

    // Q/E Class Cycling Hotkeys
    if (e.key === "e" && selectedBoxes.length > 0) {
        pushUndo();
        selectedBoxes.forEach(i => boxes[i].cls = (boxes[i].cls + 1) % CLASS_NAMES.length);
        commitUndo("Cycled class via hotkey (E)");
        dirty = true; triggerAutoSave(); updateBoxPanel(); renderCanvas();
    }
    if (e.key === "q" && selectedBoxes.length > 0) {
        pushUndo();
        selectedBoxes.forEach(i => boxes[i].cls = (boxes[i].cls - 1 + CLASS_NAMES.length) % CLASS_NAMES.length);
        commitUndo("Cycled class via hotkey (Q)");
        dirty = true; triggerAutoSave(); updateBoxPanel(); renderCanvas();
    }
});

// ─── API ─────────────────────────────────────────────────────────────────────
async function loadSplitImages(split) {
    currentSplit = split;
    localStorage.setItem("lastSplit", split);
    currentIndex = -1; boxes = []; selectedBoxes = [];
    showLoading("Loading list…");
    allImages = await fetch(`/api/images?split=${split}`).then(r => r.json());
    filteredImages = [...allImages];
    renderImageList();
    hideLoading();
    if (filteredImages.length > 0) {
        const lastOpened = localStorage.getItem("lastImage_" + split);
        let startIdx = 0;
        if (lastOpened) {
            const found = filteredImages.findIndex(i => i.name === lastOpened);
            if (found >= 0) startIdx = found;
        }
        navigateTo(startIdx);
    }
}

async function navigateTo(idx) {
    if (filteredImages.length === 0) return;
    idx = Math.max(0, Math.min(idx, filteredImages.length - 1));
    if (idx === currentIndex) return;
    if (dirty && currentIndex >= 0) await saveAnnotations(true);
    currentIndex = idx;
    localStorage.setItem("lastImage_" + currentSplit, filteredImages[currentIndex].name);
    localStorage.setItem("lastSplit", currentSplit);
    loadImage(filteredImages[currentIndex].name);
    preloadAround(idx);
}

function preloadAround(idx) {
    [-1, 1, 2].forEach((offset, i) => {
        const ni = idx + offset;
        if (ni >= 0 && ni < filteredImages.length) {
            _preloadSlots[i].src = `/api/image/${currentSplit}/${encodeURIComponent(filteredImages[ni].name)}`;
        }
    });
}

async function loadImage(filename) {
    const myId = ++_loadId;
    _imgReady = false; _labelsReady = false;
    boxes = []; selectedBoxes = []; dirty = false;
    cmdManager.clear();
    zoomScale = 1.0; panX = 0; panY = 0;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    showLoading("Auto-annotating…");
    document.getElementById("canvas-filename").textContent = filename;
    highlightActive();
    updateBoxPanel();

    const [fetchedBoxes] = await Promise.all([
        fetch(`/api/labels/${currentSplit}/${encodeURIComponent(filename)}`).then(r => r.json()),
        new Promise(resolve => {
            imgEl.onload = () => {
                if (myId !== _loadId) { resolve(); return; }
                imgNatW = imgEl.naturalWidth; imgNatH = imgEl.naturalHeight;
                const wrap = document.getElementById("canvas-wrapper");
                canvas.width = wrap.clientWidth;
                canvas.height = wrap.clientHeight;
                _imgReady = true;
                if (_labelsReady) { renderCanvas(); hideLoading(); }
                resolve();
            };
            imgEl.onerror = () => { hideLoading(); resolve(); };
            imgEl.src = `/api/image/${currentSplit}/${encodeURIComponent(filename)}`;
        })
    ]);

    if (myId !== _loadId) return;

    boxes = fetchedBoxes; _labelsReady = true;
    const li = filteredImages.find(i => i.name === filename); if (li) li.annotated = true;
    const ai = allImages.find(i => i.name === filename); if (ai) ai.annotated = true;
    if (_imgReady) { renderCanvas(); hideLoading(); }
    updateBoxPanel(); renderImageList(); updateStats();
}

async function saveAnnotations(silent = false) {
    if (currentIndex < 0) return;
    try {
        await fetch("/api/save", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ split: currentSplit, filename: filteredImages[currentIndex].name, boxes })
        });
        dirty = false;
        if (!silent) showToast("✅ Saved!", false, "ok");
    } catch (err) {
        if (!silent) showToast("❌ Save failed — check connection", true);
    }
}

async function saveAndNext() {
    await saveAnnotations(true);
    navigateTo(currentIndex + 1);
}

// Debounced updateStats — avoid hammering the server on rapid navigation
let _statsTimer = null;
function updateStats() {
    clearTimeout(_statsTimer);
    _statsTimer = setTimeout(_doUpdateStats, 2000);
}
async function _doUpdateStats() {
    const d = await fetch("/api/stats").then(r => r.json());
    const fmt = k => d[k] ? `${d[k].annotated}/${d[k].total}` : "–";
    document.getElementById("s-train").textContent = fmt("train");
    document.getElementById("s-valid").textContent = fmt("valid");
    document.getElementById("s-test").textContent = fmt("test");
    const tot = (d.train?.total || 0) + (d.valid?.total || 0) + (d.test?.total || 0);
    const ann = (d.train?.annotated || 0) + (d.valid?.annotated || 0) + (d.test?.annotated || 0);
    document.getElementById("sst-total").textContent = tot;
    document.getElementById("sst-anno").textContent = ann;
    document.getElementById("sst-tr").textContent = d.train?.total || 0;
    document.getElementById("sst-vl").textContent = d.valid?.total || 0;
    document.getElementById("sst-ts").textContent = d.test?.total || 0;
    if (tot > 0) {
        document.getElementById("sbar-t").style.width = ((d.train?.total || 0) / tot * 100) + "%";
        document.getElementById("sbar-v").style.width = ((d.valid?.total || 0) / tot * 100) + "%";
        document.getElementById("sbar-x").style.width = ((d.test?.total || 0) / tot * 100) + "%";
    }
    // Update progress ring
    const pct = tot > 0 ? Math.round(ann / tot * 100) : 0;
    const circumference = 56.5;
    const offset = circumference - (pct / 100) * circumference;
    const ring = document.getElementById('prog-ring-circle');
    if (ring) { ring.style.strokeDashoffset = offset; ring.style.stroke = pct > 80 ? '#22c55e' : pct > 50 ? '#f59e0b' : '#3b82f6'; }
    const pctEl = document.getElementById('prog-pct');
    if (pctEl) pctEl.textContent = pct + '%';
    // Refresh class distribution chart if visible
    const splitPanel = document.getElementById('panel-split');
    if (splitPanel && splitPanel.classList.contains('active')) updateClassChart();
}

async function deleteCurrentImage() {
    if (currentIndex < 0) return;
    const fn = filteredImages[currentIndex].name;
    if (!confirm(`Delete "${fn}"?`)) return;
    await fetch("/api/delete_image", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ split: currentSplit, filename: fn }) });
    filteredImages.splice(currentIndex, 1);
    allImages = allImages.filter(i => i.name !== fn);
    if (currentIndex >= filteredImages.length) currentIndex = filteredImages.length - 1;
    renderImageList();
    if (currentIndex >= 0) { const n = currentIndex; currentIndex = -1; navigateTo(n); }
    else { boxes = []; renderCanvas(); updateBoxPanel(); }
    showToast("🗑 Deleted");
}

// ─── JUMP TO NEXT UNANNOTATED ────────────────────────────────────────────────
async function jumpToNextUnannotated() {
    const start = currentIndex + 1;
    for (let i = start; i < allImages.length; i++) {
        if (!allImages[i].annotated) {
            // Find this index in filteredImages
            const fi = filteredImages.findIndex(x => x.name === allImages[i].name);
            if (fi >= 0) { navigateTo(fi); return; }
        }
    }
    showToast('✔ All images annotated!', false, 'ok');
}

// ─── CLASS DISTRIBUTION CHART ───────────────────────────────────────────────
async function updateClassChart() {
    const body = document.getElementById('cls-chart-body');
    if (!body) return;
    const counts = await fetch('/api/class_stats').then(r => r.json()).catch(() => ({}));
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    if (total === 0) { body.innerHTML = '<div style="font-size:11px;color:#374151;text-align:center;padding:6px;">No annotations yet</div>'; return; }
    body.innerHTML = '';
    const sorted = Object.entries(counts).sort(([, a], [, b]) => b - a);
    sorted.forEach(([name, cnt], i) => {
        const pct = Math.round(cnt / total * 100);
        const col = COLORS[CLASS_NAMES.indexOf(name) % COLORS.length] || '#3b82f6';
        body.innerHTML += `
      <div class="cls-bar-row">
        <div class="cls-bar-label" title="${name}">${name}</div>
        <div class="cls-bar-track"><div class="cls-bar-fill" style="width:${pct}%;background:${col}"></div></div>
        <div class="cls-bar-count">${cnt}</div>
      </div>`;
    });
}

// ─── UI HELPERS ───────────────────────────────────────────────────────────────
function renderImageList() {
    const currentName = (currentIndex >= 0 && currentIndex < filteredImages.length) ? filteredImages[currentIndex].name : null;

    const q = document.getElementById("search").value.toLowerCase();
    let base = allImages.filter(i => !q || i.name.toLowerCase().includes(q));

    if (sidebarFilter === 'done') base = base.filter(i => i.annotated);
    if (sidebarFilter === 'pending') base = base.filter(i => !i.annotated);

    filteredImages = base;
    document.getElementById("img-count").textContent = `(${filteredImages.length})`;

    if (currentName) {
        const newIdx = filteredImages.findIndex(i => i.name === currentName);
        if (newIdx >= 0) {
            currentIndex = newIdx;
        } else {
            currentIndex = -1;
            if (filteredImages.length > 0) setTimeout(() => navigateTo(0), 10);
            else {
                boxes = []; ctx.clearRect(0, 0, canvas.width, canvas.height); updateBoxPanel();
                document.getElementById("canvas-filename").textContent = "No images match filter";
            }
        }
    }

    const viewport = document.getElementById("image-list-viewport");
    const spacer = document.getElementById("image-list-spacer");
    const inner = document.getElementById("image-list-inner");
    const totalH = filteredImages.length * ITEM_H;
    spacer.style.height = totalH + "px";
    _lastScrollTop = -1; // force re-render

    function renderVisible() {
        const scrollTop = viewport.scrollTop;
        if (Math.abs(scrollTop - _lastScrollTop) < 2 && inner.childNodes.length > 0) return;
        _lastScrollTop = scrollTop;

        const startIdx = Math.max(0, Math.floor(scrollTop / ITEM_H) - 2);
        const endIdx = Math.min(filteredImages.length - 1, Math.ceil((scrollTop + viewport.clientHeight) / ITEM_H) + 2);

        inner.style.top = (startIdx * ITEM_H) + "px";
        inner.innerHTML = "";

        for (let i = startIdx; i <= endIdx; i++) {
            const item = filteredImages[i];
            const div = document.createElement("div");
            div.className = "img-item" + (i === currentIndex ? " active" : "");
            div.style.height = ITEM_H + "px";

            const th = document.createElement("img"); th.className = "thumb"; th.loading = "lazy";
            th.src = `/api/thumb/${currentSplit}/${encodeURIComponent(item.name)}`;
            th.onerror = () => { th.style.opacity = ".3"; };
            const nm = document.createElement("div"); nm.className = "iname"; nm.textContent = item.name; nm.title = item.name;
            const dot = document.createElement("div"); dot.className = "dot " + (item.annotated ? "done" : "pending");
            div.append(th, nm, dot);
            div.addEventListener("click", () => navigateTo(i));
            inner.appendChild(div);
        }
    }

    renderVisible();
    viewport.onscroll = renderVisible;
}

function highlightActive() {
    document.querySelectorAll(".img-item").forEach((el, i) => {
        const abs = parseInt(el.parentElement.style.top) / ITEM_H;
        // We only have rendered items; match by checking if offset + index = currentIndex
        const viewport = document.getElementById("image-list-viewport");
        const startIdx = Math.max(0, Math.floor(viewport.scrollTop / ITEM_H) - 2);
        el.classList.toggle("active", (startIdx + i) === currentIndex);
    });
    // Scroll viewport to make item visible
    const viewport = document.getElementById("image-list-viewport");
    const itemTop = currentIndex * ITEM_H;
    if (itemTop < viewport.scrollTop || itemTop > viewport.scrollTop + viewport.clientHeight - ITEM_H) {
        viewport.scrollTop = itemTop - viewport.clientHeight / 2 + ITEM_H / 2;
    }
}

function updateBoxPanel() {
    const panel = document.getElementById("boxes-panel");
    panel.innerHTML = "";
    if (boxes.length === 0) {
        panel.innerHTML = '<div style="padding:12px;text-align:center;font-size:11px;color:#374151;">No boxes. Draw one or wait for auto-annotation.</div>';
        return;
    }
    boxes.forEach((b, i) => {
        const card = document.createElement("div"); card.className = "box-card" + (selectedBoxes.includes(i) ? " sel" : "");
        const hdr = document.createElement("div"); hdr.className = "bch";
        hdr.addEventListener("click", (e) => { 
            if (e.shiftKey) {
                if (selectedBoxes.includes(i)) selectedBoxes = selectedBoxes.filter(x => x !== i);
                else selectedBoxes.push(i);
            } else {
                selectedBoxes = [i]; 
            }
            updateBoxPanel(); renderCanvas(); 
        });
        const num = document.createElement("div"); num.className = "bnum";
        num.textContent = i + 1; num.style.background = COLORS[b.cls % COLORS.length];
        const cls = document.createElement("div"); cls.className = "bcls"; cls.textContent = `[${b.cls + 1}] ${CLASS_NAMES[b.cls] || `cls${b.cls}`}`;
        const del = document.createElement("button"); del.className = "bdel"; del.innerHTML = "✕";
        del.addEventListener("click", e => { e.stopPropagation(); pushUndo(); deleteBox(i); });
        hdr.append(num, cls, del); card.append(hdr);
        if (selectedBoxes.length === 1 && selectedBoxes.includes(i)) {
            const body = document.createElement("div"); body.className = "bcb";
            const sel = document.createElement("select"); sel.className = "cls-sel";
            CLASS_NAMES.forEach((cn, ci) => { const o = document.createElement("option"); o.value = ci; o.textContent = `[${ci + 1}] ${cn}`; if (ci === b.cls) o.selected = true; sel.appendChild(o); });
            sel.addEventListener("change", () => { pushUndo(); boxes[i].cls = parseInt(sel.value); cls.textContent = `[${boxes[i].cls + 1}] ${CLASS_NAMES[boxes[i].cls]}`; num.style.background = COLORS[boxes[i].cls % COLORS.length]; commitUndo("Changed class via dropdown"); dirty = true; triggerAutoSave(); renderCanvas(); });
            const coords = document.createElement("div"); coords.className = "bcoords";
            coords.textContent = `x:${b.x.toFixed(4)} y:${b.y.toFixed(4)} w:${b.w.toFixed(4)} h:${b.h.toFixed(4)}`;
            body.append(sel, coords); card.append(body);
        }
        panel.appendChild(card);
    });
}

function deleteBox(idx) {
    boxes.splice(idx, 1);
    selectedBoxes = [];
    commitUndo("Deleted box from panel");
    dirty = true; triggerAutoSave(); updateBoxPanel(); renderCanvas();
}

function setDrawMode(on) {
    drawMode = on;
    const pill = document.getElementById("mode-pill");
    const btn = document.getElementById("btn-draw");
    if (on) {
        if (pill) { pill.textContent = "DRAW"; pill.className = "mpill draw"; }
        if (btn) { btn.classList.add("active-tool"); btn.textContent = "✖ Cancel (D)"; }
        canvas.style.cursor = "crosshair";
    } else {
        if (pill) { pill.textContent = "SELECT"; pill.className = "mpill"; }
        if (btn) { btn.classList.remove("active-tool"); btn.textContent = "✏ Draw (D)"; }
        canvas.style.cursor = "default";
    }
}

function showLoading(m) { document.getElementById("loading-text").textContent = m; document.getElementById("loading-overlay").style.display = "flex"; }
function hideLoading() { document.getElementById("loading-overlay").style.display = "none"; }
function showToast(m, err = false, type = "") {
    const t = document.getElementById("toast"); t.textContent = m;
    t.className = "show" + (err ? " err" : type === "ok" ? " ok" : "");
    clearTimeout(t._t); t._t = setTimeout(() => t.className = "", 2400);
}

// ─── BRIGHTNESS / CONTRAST ───────────────────────────────────────────────────
document.getElementById("bc-bright").addEventListener("input", e => { bcBright = parseFloat(e.target.value); renderCanvas(); });
document.getElementById("bc-contrast").addEventListener("input", e => { bcContrast = parseFloat(e.target.value); renderCanvas(); });
document.getElementById("bc-bright-reset").addEventListener("click", () => { bcBright = 1; document.getElementById("bc-bright").value = 1; renderCanvas(); });
document.getElementById("bc-contrast-reset").addEventListener("click", () => { bcContrast = 1; document.getElementById("bc-contrast").value = 1; renderCanvas(); });

// ─── ZOOM BUTTONS ────────────────────────────────────────────────────────────
document.getElementById("btn-zoom-in").addEventListener("click", () => { zoomScale = Math.min(20, zoomScale * 1.3); renderCanvas(); });
document.getElementById("btn-zoom-out").addEventListener("click", () => { zoomScale = Math.max(0.2, zoomScale / 1.3); renderCanvas(); });
document.getElementById("btn-fit").addEventListener("click", fitView);

// ─── TRAINING ────────────────────────────────────────────────────────────────
let trainES = null;

async function startTraining() {
    const model = document.getElementById("tr-custom").value || document.getElementById("tr-model").value;
    const cfg = {
        model, epochs: document.getElementById("tr-epochs").value,
        batch: document.getElementById("tr-batch").value,
        imgsz: document.getElementById("tr-imgsz").value,
        lr0: document.getElementById("tr-lr").value,
        device: document.getElementById("tr-device").value,
        name: document.getElementById("tr-name").value,
    };
    const log = document.getElementById("train-log");
    log.innerHTML = '<span class="train-log-line highlight">Starting training…</span>\n';
    document.getElementById("btn-train").disabled = true;
    document.getElementById("train-status").textContent = "⏳ Training in progress…";

    const res = await fetch("/api/train", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cfg) });
    const d = await res.json();
    if (d.error) { showToast("❌ " + d.error, true); document.getElementById("btn-train").disabled = false; return; }

    if (trainES) trainES.close();
    trainES = new EventSource("/api/train/logs");
    trainES.onmessage = e => {
        const p = JSON.parse(e.data);
        const line = document.createElement("div");
        line.className = "train-log-line" + (p.line?.includes("Epoch") || (p.line || "").includes("mAP") ? " highlight" : "");
        line.textContent = p.line || "";
        log.appendChild(line); log.scrollTop = log.scrollHeight;
        if (p.done) { trainES.close(); document.getElementById("btn-train").disabled = false; document.getElementById("train-status").textContent = "✅ Training complete!"; }
    };
}

async function stopTraining() {
    await fetch("/api/train/stop", { method: "POST" });
    if (trainES) trainES.close();
    document.getElementById("btn-train").disabled = false;
    document.getElementById("train-status").textContent = "⏹ Stopped.";
}

// ─── AUTO-SPLIT ──────────────────────────────────────────────────────────────
async function doAutoSplit() {
    const rT = parseInt(document.getElementById("sp-train").value) || 70;
    const rV = parseInt(document.getElementById("sp-valid").value) || 20;
    const rX = parseInt(document.getElementById("sp-test").value) || 10;
    const tot = rT + rV + rX;
    const ratios = [rT / tot, rV / tot, rX / tot];
    const btn = document.getElementById("btn-auto-split");
    btn.disabled = true; btn.textContent = "🔀 Splitting…";

    const res = await fetch("/api/auto_split", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ratios }) });
    const d = await res.json();
    if (d.status === "ok") {
        const el = document.getElementById("split-result");
        el.textContent = `✅ Done — Train:${d.counts.train}  Valid:${d.counts.valid}  Test:${d.counts.test}  Skipped(unannotated):${d.counts.unannotated_skipped || 0}`;
        el.style.display = "block";
        await updateStats();
        currentIndex = -1; boxes = []; allImages = []; filteredImages = [];
        await loadSplitImages(currentSplit);
        showToast("✅ Dataset split complete!", false, "ok");
    } else {
        showToast("❌ " + JSON.stringify(d), true);
    }
    btn.disabled = false; btn.textContent = "🔀 Auto-Split Dataset";
}

// ─── WIRING ──────────────────────────────────────────────────────────────────
document.querySelectorAll(".split-btn").forEach(tab => {
    tab.addEventListener("click", () => {
        document.querySelectorAll(".split-btn").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        currentSplit = tab.dataset.split;
        dirty = false; boxes = []; ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (appMode === "grid") {
            loadGridPage(1);
        } else {
            loadSplitImages(currentSplit);
        }
    });
});

document.querySelectorAll(".rtab").forEach(tab => {
    tab.addEventListener("click", () => {
        document.querySelectorAll(".rtab").forEach(t => t.classList.remove("active"));
        document.querySelectorAll(".rpanel").forEach(p => p.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById("panel-" + tab.dataset.panel).classList.add("active");
        if (tab.dataset.panel === "split") { updateStats(); updateClassChart(); }
        if (tab.dataset.panel === "train") pollTrainStatus();
    });
});

// Sidebar filter tabs
document.querySelectorAll(".ftab").forEach(tab => {
    tab.addEventListener("click", () => {
        document.querySelectorAll(".ftab").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        sidebarFilter = tab.dataset.filter;
        _lastScrollTop = -1;
        renderImageList();
    });
});

document.getElementById("search").addEventListener("input", renderImageList);

document.getElementById("btn-fullscreen")?.addEventListener("click", () => {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(err => {
            console.error(`Error attempting to enable fullscreen mode: ${err.message}`);
        });
    } else {
        document.exitFullscreen();
    }
});

document.addEventListener("fullscreenchange", () => {
    const icon = document.querySelector("#btn-fullscreen i");
    if (icon) {
        if (document.fullscreenElement) {
            icon.className = "fa-solid fa-compress";
        } else {
            icon.className = "fa-solid fa-expand";
        }
    }
});
document.getElementById("btn-draw").addEventListener("click", () => setDrawMode(!drawMode));
document.getElementById("btn-next-unanno").addEventListener("click", jumpToNextUnannotated);

document.getElementById("btn-del-img").addEventListener("click", deleteCurrentImage);
document.getElementById("btn-save").addEventListener("click", () => saveAnnotations());
document.getElementById("btn-save-next").addEventListener("click", saveAndNext);
document.getElementById("add-box-btn").addEventListener("click", () => {
    if (currentIndex < 0) return;
    pushUndo();
    boxes.push({ id: boxes.length, cls: 0, x: 0.5, y: 0.5, w: 0.15, h: 0.15 });
    selectedBoxes = [boxes.length - 1]; dirty = true; updateBoxPanel(); renderCanvas();
});
document.getElementById("nav-prev").addEventListener("click", () => navigateTo(currentIndex - 1));
document.getElementById("nav-next").addEventListener("click", () => navigateTo(currentIndex + 1));
document.getElementById("btn-auto-split").addEventListener("click", doAutoSplit);
document.getElementById("btn-train").addEventListener("click", startTraining);
document.getElementById("btn-stop-train").addEventListener("click", stopTraining);

// Extra keyboard shortcuts (J=jump unannotated)
window.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "j") jumpToNextUnannotated();
}, true);

async function pollTrainStatus() {
    const d = await fetch("/api/train/status").then(r => r.json()).catch(() => ({ status: "idle" }));
    document.getElementById("train-status").textContent =
        d.status === "running" ? "⏳ Training in progress…" :
            d.status === "done" ? `✅ Done (exit code ${d.exit_code})` : "";
            
    const btnTrain = document.getElementById("btn-train");
    const dev = document.getElementById("tr-device").value;
    if (d.status === "running") {
        btnTrain.disabled = true;
        btnTrain.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Training...';
    } else if (dev === "cpu") {
        btnTrain.disabled = true;
        btnTrain.innerHTML = '<i class="fa-solid fa-ban"></i> CPU Training Disabled';
    } else {
        btnTrain.disabled = false;
        btnTrain.innerHTML = '<i class="fa-solid fa-play"></i> Start Training';
    }
}

document.getElementById("tr-device").addEventListener("change", () => {
    pollTrainStatus(); // Re-evaluates button state
});

// ─── GRID VIEW LOGIC ─────────────────────────────────────────────────────────
let appMode = "annotate"; // 'annotate' | 'grid'
let gridPage = 1;
let currentGridImages = [];
let currentGridIndex = -1;

async function setAppMode(mode) {
    appMode = mode;
    document.getElementById("mode-annotate").classList.toggle("active", mode === "annotate");
    document.getElementById("mode-grid").classList.toggle("active", mode === "grid");

    if (mode === "grid") {
        document.getElementById("left").style.display = "none";
        document.getElementById("canvas-area").style.display = "none";
        document.getElementById("right").style.display = "none";
        const gv = document.getElementById("grid-view");
        gv.style.display = "flex";
        await loadGridPage(1);
    } else {
        document.getElementById("grid-view").style.display = "none";
        document.getElementById("left").style.display = "";
        document.getElementById("canvas-area").style.display = "";
        document.getElementById("right").style.display = "";
        await loadSplitImages(currentSplit);
    }
}



let _healthFilterFiles = null; // Store files to filter from health dashboard

async function loadGridPage(page) {
    if (page < 1) page = 1;
    const filter = document.getElementById("grid-filter").value;
    const clsFilter = document.getElementById("grid-class-filter").value;
    const gridLimit = document.getElementById("grid-limit") ? parseInt(document.getElementById("grid-limit").value) : 50;
    showLoading("Loading grid...");
    try {
        const bodyPayload = {
            split: currentSplit,
            page: page,
            limit: gridLimit,
            filter: filter,
            class_id: clsFilter
        };
        
        if (_healthFilterFiles !== null) {
            bodyPayload.filter_files = _healthFilterFiles;
        }

        const res = await fetch("/api/dataset_page", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(bodyPayload)
        }).then(r => {
            if (!r.ok) throw new Error("API returned " + r.status);
            return r.json();
        });

        gridPage = res.page;
        const pageText = `Page ${res.page} / ${res.pages}`;
        
        document.getElementById("grid-page-info").textContent = pageText;
        document.getElementById("grid-prev").disabled = (res.page <= 1);
        document.getElementById("grid-next").disabled = (res.page >= res.pages);
        
        const infoBtm = document.getElementById("grid-page-info-btm");
        if (infoBtm) infoBtm.textContent = pageText;
        
        const prevBtm = document.getElementById("grid-prev-btm");
        if (prevBtm) prevBtm.disabled = (res.page <= 1);
        
        const nextBtm = document.getElementById("grid-next-btm");
        if (nextBtm) nextBtm.disabled = (res.page >= res.pages);

        const cont = document.getElementById("grid-content");
        cont.innerHTML = "";
        currentGridImages = res.images;

        res.images.forEach((img, idx) => {
            const card = document.createElement("div");
            card.className = "grid-card";
            card.onclick = () => {
                currentGridIndex = idx;
                openImageModal(img);
            };

            // Thumbnail wrapper
            const twrap = document.createElement("div");
            twrap.className = "grid-thumb-wrap";

            const timg = document.createElement("img");
            timg.className = "grid-thumb";
            timg.loading = "lazy";
            timg.src = `/api/image/${currentSplit}/${encodeURIComponent(img.name)}`;
            twrap.appendChild(timg);

            // Draw bounding boxes — positions relative to thumb-wrap (16:9 cropped)
            if (img.boxes && img.boxes.length > 0) {
                // Once image loads, use its natural dimensions for accurate box placement
                timg.addEventListener("load", () => {
                    img.boxes.forEach(b => {
                        const box = document.createElement("div");
                        box.className = "grid-box";
                        box.style.left = ((b.x - b.w / 2) * 100) + "%";
                        box.style.top = ((b.y - b.h / 2) * 100) + "%";
                        box.style.width = (b.w * 100) + "%";
                        box.style.height = (b.h * 100) + "%";
                        box.style.borderColor = COLORS[b.cls % COLORS.length];
                        twrap.appendChild(box);
                    });
                });
            }

            // Status dot
            const sdot = document.createElement("div");
            sdot.className = "grid-status-dot " + (img.annotated ? "done" : "pending");
            sdot.title = img.annotated ? `Annotated (${img.boxes ? img.boxes.length : 0} boxes)` : "Pending";
            twrap.appendChild(sdot);

            card.appendChild(twrap);
            cont.appendChild(card);
        });
    } catch (err) {
        console.error("Grid load error:", err);
        showToast("Error loading grid: " + err.message, true);
    } finally {
        hideLoading();
    }
}

document.getElementById("grid-filter").addEventListener("change", () => { _healthFilterFiles = null; loadGridPage(1); });
document.getElementById("grid-class-filter").addEventListener("change", () => { _healthFilterFiles = null; loadGridPage(1); });

function openImageModal(imgData) {
    document.getElementById("image-modal").style.display = "flex";
    document.getElementById("im-title").textContent = imgData.name;
    document.getElementById("im-meta").textContent = `Split: ${currentSplit} | Boxes: ${imgData.boxes ? imgData.boxes.length : 0}`;

    const left = document.getElementById("im-left");
    left.innerHTML = "";

    const fullImg = document.createElement("img");
    fullImg.src = `/api/image/${currentSplit}/${encodeURIComponent(imgData.name)}`;
    fullImg.style.maxWidth = "100%";
    fullImg.style.maxHeight = "100%";
    fullImg.style.objectFit = "contain";
    left.appendChild(fullImg);

    // Need to wait for image to load to get aspect ratio for perfect boxes
    fullImg.onload = () => {
        // Calculate actual displayed dimensions of the image inside the container
        const iw = fullImg.naturalWidth, ih = fullImg.naturalHeight;
        const cw = left.clientWidth, ch = left.clientHeight;
        const scale = Math.min(cw / iw, ch / ih);
        const dw = iw * scale, dh = ih * scale;
        const dx = (cw - dw) / 2, dy = (ch - dh) / 2;

        if (imgData.boxes) {
            imgData.boxes.forEach(b => {
                const box = document.createElement("div");
                box.className = "grid-box";
                box.style.left = dx + ((b.x - b.w / 2) * dw) + "px";
                box.style.top = dy + ((b.y - b.h / 2) * dh) + "px";
                box.style.width = (b.w * dw) + "px";
                box.style.height = (b.h * dh) + "px";
                box.style.borderWidth = "2px";
                box.style.borderColor = COLORS[b.cls % COLORS.length];

                // Small class label on box
                const lbl = document.createElement("div");
                lbl.textContent = CLASS_NAMES[b.cls] || b.cls;
                lbl.style.position = "absolute";
                lbl.style.background = COLORS[b.cls % COLORS.length];
                lbl.style.color = "#fff";
                lbl.style.fontSize = "10px";
                lbl.style.padding = "2px 4px";
                lbl.style.top = "-16px";
                lbl.style.left = "-2px";
                lbl.style.whiteSpace = "nowrap";
                box.appendChild(lbl);

                left.appendChild(box);
            });
        }
        document.getElementById("im-meta").textContent = `Size: ${iw}x${ih} | Split: ${currentSplit} | Boxes: ${imgData.boxes ? imgData.boxes.length : 0}`;
    };

    const clsCont = document.getElementById("im-classes");
    clsCont.innerHTML = "";
    if (imgData.boxes && imgData.boxes.length > 0) {
        imgData.boxes.forEach(b => {
            const bge = document.createElement("div");
            bge.className = "badge badge-blue";
            bge.style.background = COLORS[b.cls % COLORS.length] + "22";
            bge.style.border = `1px solid ${COLORS[b.cls % COLORS.length]}44`;
            bge.style.color = COLORS[b.cls % COLORS.length];
            bge.textContent = CLASS_NAMES[b.cls] || b.cls;
            clsCont.appendChild(bge);
        });
    } else {
        clsCont.innerHTML = "<div style='font-size:12px;color:#475569;'>No objects annotated.</div>";
    }

    // Prev/Next buttons logic
    const btnPrev = document.getElementById("btn-modal-prev");
    const btnNext = document.getElementById("btn-modal-next");

    if (btnPrev && btnNext) {
        if (currentGridIndex > 0) {
            btnPrev.disabled = false;
            btnPrev.onclick = () => { currentGridIndex--; openImageModal(currentGridImages[currentGridIndex]); };
        } else {
            btnPrev.disabled = true;
            btnPrev.onclick = null;
        }

        if (currentGridIndex >= 0 && currentGridIndex < currentGridImages.length - 1) {
            btnNext.disabled = false;
            btnNext.onclick = () => { currentGridIndex++; openImageModal(currentGridImages[currentGridIndex]); };
        } else {
            btnNext.disabled = true;
            btnNext.onclick = null;
        }
    }

    // Jump to annotation logic
    document.getElementById("btn-jump-anno").onclick = async () => {
        closeImageModal();
        await setAppMode("annotate");
        // Find index in filteredImages to navigate to
        let idx = filteredImages.findIndex(fi => fi.name === imgData.name);

        if (idx < 0) {
            sidebarFilter = "all";
            document.querySelectorAll(".ftab").forEach(t => t.classList.toggle("active", t.dataset.filter === "all"));
            renderImageList();
            idx = filteredImages.findIndex(fi => fi.name === imgData.name);
        }

        if (idx >= 0) navigateTo(idx);
    };
}

function closeImageModal() {
    document.getElementById("image-modal").style.display = "none";
}

// ─── ADD IMAGES MODAL LOGIC ──────────────────────────────────────────────────
document.getElementById("btn-add-images").addEventListener("click", async () => {
    document.getElementById("add-src-input").value = "";
    document.getElementById("add-ingest-prog").style.display = "none";
    document.getElementById("btn-confirm-add").disabled = false;
    document.getElementById("add-images-modal").classList.remove("hidden");
});

document.getElementById("btn-confirm-add").addEventListener("click", async () => {
    const src = document.getElementById("add-src-input").value;
    if (!src) return;
    const targetSplit = "train";

    document.getElementById("btn-confirm-add").disabled = true;
    document.getElementById("add-ingest-prog").style.display = "block";
    document.getElementById("add-ip-status").textContent = "Starting…";

    const { job_id } = await fetch("/api/ingest_append", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_dir: src, split: targetSplit })
    }).then(r => r.json());

    const es = new EventSource(`/api/progress/${job_id}`);
    es.onmessage = async e => {
        const p = JSON.parse(e.data);
        const pct = p.total > 0 ? Math.round(p.done / p.total * 100) : 0;
        document.getElementById("add-ip-bar").style.width = pct + "%";
        document.getElementById("add-ip-num").textContent = `${p.done} / ${p.total}`;
        document.getElementById("add-ip-status").textContent =
            p.status === "linking" ? `🔗 Linking images… ${pct}%` :
                p.status === "copying" ? `📂 Copying images… ${pct}%` : p.status;

        if (p.status === "done") {
            es.close();
            document.getElementById("add-ip-status").textContent = "✅ Done!";
            setTimeout(() => {
                document.getElementById("add-images-modal").classList.add("hidden");
                // Reload the view so new images appear
                if (appMode === "grid") {
                    loadGridPage(1);
                } else {
                    loadSplitImages(currentSplit);
                }
            }, 800);
        }
        if (String(p.status).startsWith("error")) {
            es.close();
            document.getElementById("add-ip-status").textContent = "❌ " + p.status;
            document.getElementById("btn-confirm-add").disabled = false;
        }
    };
});

// ─── INIT ────────────────────────────────────────────────────────────────────
buildClassDropdowns();
initSetup();

// ─── V3.0 NEW FEATURES ───
// Force toggle draw mode helper
function forceDrawMode(state) {
  if (drawMode !== state) document.getElementById("btn-draw").click();
}

// Right Click Hold to Draw
const ca = document.getElementById("canvas-area");
if (ca) ca.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("mousedown", e => {
  if (e.button === 2 && currentIndex >= 0 && !isPanning) {
    e.preventDefault();
    forceDrawMode(true);
    const { x, y } = getPos(e);
    drag = { on: true, type: "draw", bIdx: -1, sx: x, sy: y, ds: { x, y } };
  }
});
canvas.addEventListener("mouseup", e => {
  if (e.button === 2) {
    e.preventDefault();
    setTimeout(() => forceDrawMode(false), 50);
  }
});

// Minimap
const minimap = document.getElementById("minimap");
const mctx = minimap ? minimap.getContext("2d") : null;
function updateMinimap() {
  if (!minimap || !imgEl.src || !imgEl.complete) return;
  if (zoomScale <= 1.0) { minimap.style.display = "none"; return; }
  minimap.style.display = "block";
  const mw = minimap.width = 120;
  const mh = minimap.height = 120 * (imgNatH / imgNatW);
  mctx.clearRect(0,0,mw,mh);
  mctx.drawImage(imgEl, 0, 0, mw, mh);
  
  const { ox, oy, scale } = getTransform();
  const dW = imgNatW * scale, dH = imgNatH * scale;
  
  const vX = Math.max(0, -ox / dW);
  const vY = Math.max(0, -oy / dH);
  const vW = Math.min(1, canvas.width / dW);
  const vH = Math.min(1, canvas.height / dH);
  
  mctx.strokeStyle = "#3b82f6";
  mctx.lineWidth = 2;
  mctx.fillStyle = "rgba(59,130,246,0.2)";
  mctx.fillRect(vX * mw, vY * mh, vW * mw, vH * mh);
  mctx.strokeRect(vX * mw, vY * mh, vW * mw, vH * mh);
}

const oldRenderCanvas = renderCanvas;
renderCanvas = function() {
  oldRenderCanvas();
  updateMinimap();
};

// Health Dashboard Checks
setInterval(() => {
  const hw = document.getElementById('health-warnings');
  if(hw) {
    let warns = [];
    let nullCount = 0;
    if (typeof allImages !== 'undefined') {
       nullCount = allImages.filter(i => !i.annotated).length;
       if(nullCount > 0) warns.push('⚠️ ' + nullCount + ' pending/unannotated images.');
    }
    if(warns.length > 0) {
      hw.innerHTML = warns.join('<br>');
      hw.style.display = 'block';
    } else {
      hw.style.display = 'none';
    }
  }
}, 3000);

document.getElementById("btn-export-dataset")?.addEventListener("click", () => {
    const format = document.getElementById("export-format").value;
    const btn = document.getElementById("btn-export-dataset");
    const oldText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Exporting...';
    
    // Trigger download
    window.location.href = `/api/export?format=${format}`;
    
    // Re-enable after a short delay since we can't perfectly track when download finishes
    setTimeout(() => {
        btn.disabled = false;
        btn.innerHTML = oldText;
    }, 4000);
});

// ─── HEALTH DASHBOARD ────────────────────────────────────────────────────────
async function openHealthDashboard() {
    const modal = document.getElementById("health-modal");
    const loading = document.getElementById("health-loading");
    const results = document.getElementById("health-results");
    
    modal.style.display = "flex";
    loading.style.display = "block";
    results.style.display = "none";
    
    try {
        const data = await fetch("/api/dataset_health").then(r => r.json());
        
        loading.style.display = "none";
        results.style.display = "block";
        
        document.getElementById("health-score").textContent = data.score;
        document.getElementById("health-score").style.color = data.score > 90 ? "#22c55e" : data.score > 70 ? "#eab308" : "#ef4444";
        document.getElementById("health-total-imgs").textContent = data.total_images;
        document.getElementById("health-total-boxes").textContent = data.total_boxes;
        
        const list = document.getElementById("health-issues-list");
        list.innerHTML = "";
        
        const addIssue = (title, count, icon, color, desc, files = []) => {
            const div = document.createElement("div");
            div.style.background = "#1a1d27";
            div.style.padding = "10px";
            div.style.borderRadius = "6px";
            div.style.display = "flex";
            div.style.justifyContent = "space-between";
            div.style.alignItems = "center";
            div.style.borderLeft = `4px solid ${color}`;
            
            if (count > 0 && files.length > 0) {
                div.style.cursor = "pointer";
                div.title = "Click to filter Grid View to these images";
                div.onmouseover = () => div.style.background = "#2a2d39";
                div.onmouseout = () => div.style.background = "#1a1d27";
                div.onclick = () => {
                    _healthFilterFiles = files;
                    document.getElementById("health-modal").style.display = "none";
                    loadGridPage(1);
                    showToast(`🔍 Filtering to ${files.length} affected images`);
                };
            }
            
            div.innerHTML = `
                <div>
                    <div style="font-weight:700; font-size:13px; color:#e2e8f0;"><i class="${icon}" style="color:${color}; margin-right:6px;"></i> ${title}</div>
                    <div style="font-size:11px; color:#94a3b8; margin-top:2px;">${desc}</div>
                </div>
                <div style="font-weight:800; font-size:16px; color:${count > 0 ? color : '#64748b'};">${count}</div>
            `;
            list.appendChild(div);
        };
        
        addIssue("Corrupt Images", data.corrupt_images.count, "fa-solid fa-file-circle-xmark", "#ef4444", "0 byte or unreadable image files.", data.corrupt_images.files);
        addIssue("Empty Images", data.empty_images.count, "fa-regular fa-image", "#3b82f6", "Images without any annotations.", data.empty_images.files);
        addIssue("Micro Boxes", data.small_boxes.count, "fa-solid fa-down-left-and-up-right-to-center", "#eab308", "Bounding boxes smaller than 0.05% of the image.", data.small_boxes.files);
        addIssue("Out-of-Bounds Boxes", data.oob_boxes.count, "fa-solid fa-expand", "#f97316", "Boxes extending beyond image dimensions.", data.oob_boxes.files);
        
        // Add Class Imbalance Check
        const clsCounts = Object.values(data.classes);
        if (clsCounts.length > 0) {
            const max = Math.max(...clsCounts);
            const min = Math.min(...clsCounts);
            const imbalanceRatio = max / Math.max(min, 1);
            if (imbalanceRatio > 5) {
                addIssue("Class Imbalance", Math.round(imbalanceRatio) + "x", "fa-solid fa-scale-unbalanced", "#eab308", "High variance between most and least frequent classes.");
            } else {
                addIssue("Class Balance", "OK", "fa-solid fa-scale-balanced", "#22c55e", "Classes are relatively balanced.");
            }
        }
        
    } catch (err) {
        console.error("Health check error", err);
        loading.innerHTML = `<span style="color:#ef4444;">❌ Error loading health data: ${err.message}</span>`;
    }
}

document.getElementById("btn-health-close")?.addEventListener("click", () => {
    document.getElementById("health-modal").style.display = "none";
});


