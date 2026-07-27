import os
import io
import json
import shutil
import random
import sqlite3
import threading
import time
import subprocess
import zipfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, Response, after_this_request
from werkzeug.utils import secure_filename
import uuid
import tempfile
import re
import configparser
from dotenv import load_dotenv
import tkinter as tk
from tkinter import filedialog

load_dotenv()

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
config.read(config_path)

# ─── APP ─────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='.')
app.config['TEMPLATES_AUTO_RELOAD'] = True

if config.has_option("Paths", "BASE_DIR"):
    BASE_DIR = config.get("Paths", "BASE_DIR")
else:
    BASE_DIR = os.environ.get("BASE_DIR", r"D:\model_train")

if config.has_option("Paths", "MODELS_DIR"):
    MODELS_DIR = config.get("Paths", "MODELS_DIR")
else:
    MODELS_DIR = BASE_DIR

_current_dataset_name = "default"
if config.has_option("Paths", "OUTPUT_DIR"):
    OUTPUT_DIR = config.get("Paths", "OUTPUT_DIR")
else:
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(BASE_DIR, f"dataset_{_current_dataset_name}"))

if config.has_option("Settings", "CONF_THRESH"):
    CONF_THRESH = float(config.get("Settings", "CONF_THRESH"))
else:
    CONF_THRESH = float(os.environ.get("CONF_THRESH", "0.25"))

VALID_SPLITS = {"train", "valid", "test"}

def validate_split(split):
    if split not in VALID_SPLITS:
        raise ValueError("Invalid split")

def secure_path(base_dir, *paths):
    """Safely join paths and ensure the result is within base_dir."""
    base_abs = os.path.abspath(base_dir)
    final_path = os.path.abspath(os.path.join(base_abs, *paths))
    if os.path.commonpath([base_abs, final_path]) != base_abs:
        raise ValueError("Path traversal attempt detected")
    return final_path
IMG_EXTS    = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
# Supported inference model formats
MODEL_EXTS  = {'.pt', '.onnx', '.engine', '.tflite', '.torchscript'}

# ─── AUTO DEVICE DETECTION ───────────────────────────────────────────────────
def auto_detect_device():
    """Returns 'cuda:0', 'mps', or 'cpu' depending on available hardware."""
    try:
        import torch
        if torch.cuda.is_available():
            # Pick the GPU with most free memory
            best = max(range(torch.cuda.device_count()),
                       key=lambda i: torch.cuda.mem_get_info(i)[0])
            return f"{best}"   # '0', '1', etc. — ultralytics convention
    except Exception:
        pass
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def get_device_info():
    """Returns a dict describing available compute devices."""
    info = {"cpu": True, "cuda": [], "mps": False, "selected": _device}
    try:
        import torch
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            info["cuda"].append({
                "id": i,
                "name": props.name,
                "total_gb": round(total/1e9, 1),
                "free_gb":  round(free/1e9,  1)
            })
        info["mps"] = bool(getattr(getattr(torch, 'backends', None), 'mps', None) and
                            torch.backends.mps.is_available())
    except Exception:
        pass
    return info


# Defaults (overridden at runtime via /api/set_model or DB)
_model_path   = r"D:\model_train\bestv8_p.pt"
_device       = auto_detect_device()   # ← auto GPU on start
_class_names  = ['auto_rickshaw', 'bike', 'bus', 'car', 'mini_bus', 'tractor', 'truck']

# ─── GLOBALS ─────────────────────────────────────────────────────────────────
_model               = None
_model_lock          = threading.RLock()
_annotate_locks      = {}
_annotate_meta_lock  = threading.Lock()
_job_progress        = {}   # job_id -> dict
_train_process       = None
_train_log_buffer    = []
_train_lock          = threading.Lock()

# Pre-annotation background queue
_preanno_queue  = None   # queue.Queue
_preanno_thread = None

DB_PATH = os.path.join(OUTPUT_DIR, "state.db")
_current_session_id = uuid.uuid4().hex

def set_active_dataset(src_dir, is_existing=False):
    global OUTPUT_DIR, DB_PATH, _current_session_id, _current_dataset_name
    
    if is_existing:
        OUTPUT_DIR = src_dir
        _current_dataset_name = os.path.basename(os.path.normpath(src_dir))
    else:
        base = os.path.basename(os.path.normpath(src_dir))
        safe_name = secure_filename(base)
        if not safe_name:
            safe_name = "default"
        
        _current_dataset_name = safe_name
        OUTPUT_DIR = os.path.join(BASE_DIR, f"dataset_{safe_name}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    DB_PATH = os.path.join(OUTPUT_DIR, "state.db")
    init_db()
    _current_session_id = uuid.uuid4().hex


# ─── DATABASE ────────────────────────────────────────────────────────────────
def get_db():
    """Thread-local SQLite connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            name    TEXT NOT NULL,
            src     TEXT,
            split   TEXT NOT NULL DEFAULT 'train'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    
    # Deduplicate before applying the unique index
    conn.execute("""
        DELETE FROM images 
        WHERE id NOT IN (
            SELECT MIN(id) 
            FROM images 
            GROUP BY name, split
        )
    """)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_images_name_split ON images(name, split)")
    
    conn.commit()
    conn.close()

    # Migrate legacy state.json if it exists
    legacy = os.path.join(OUTPUT_DIR, "state.json")
    if os.path.exists(legacy):
        try:
            with open(legacy) as f:
                data = json.load(f)
            conn = get_db()
            for split in ["train", "valid", "test"]:
                for e in data.get(split, []):
                    conn.execute(
                        "INSERT OR IGNORE INTO images (name, src, split) VALUES (?, ?, ?)",
                        (e["name"], e.get("src", ""), split)
                    )
            if data.get("source"):
                conn.execute("INSERT OR REPLACE INTO meta VALUES ('source', ?)", (data["source"],))
            conn.commit()
            conn.close()
            os.rename(legacy, legacy + ".migrated")
            print("[DB] Migrated state.json → state.db")
        except Exception as ex:
            print(f"[DB] Migration failed: {ex}")

    # Load saved classes from DB
    saved_classes = db_get_meta("classes")
    if saved_classes:
        try:
            global _class_names
            _class_names = json.loads(saved_classes)
        except Exception:
            pass


def db_get_meta(key, default=None):
    conn = get_db()
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default


def db_set_meta(key, value):
    conn = get_db()
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))
    conn.commit()
    conn.close()


def db_get_entries(split):
    conn = get_db()
    rows = conn.execute("SELECT name, src FROM images WHERE split=? ORDER BY id", (split,)).fetchall()
    conn.close()
    return [{"name": r["name"], "src": r["src"]} for r in rows]


def db_count(split):
    conn = get_db()
    row = conn.execute("SELECT COUNT(*) AS c FROM images WHERE split=?", (split,)).fetchone()
    conn.close()
    return row["c"]


def db_insert_entries(entries, split):
    conn = get_db()
    conn.executemany(
        "INSERT OR IGNORE INTO images (name, src, split) VALUES (?,?,?)",
        [(e["name"], e.get("src", ""), split) for e in entries]
    )
    conn.commit()
    conn.close()


def db_delete_image(name, split):
    conn = get_db()
    conn.execute("DELETE FROM images WHERE name=? AND split=?", (name, split))
    conn.commit()
    conn.close()


def db_update_split(name, old_split, new_split):
    conn = get_db()
    conn.execute("UPDATE images SET split=? WHERE name=? AND split=?", (new_split, name, old_split))
    conn.commit()
    conn.close()


def db_all_entries():
    conn = get_db()
    rows = conn.execute("SELECT name, src, split FROM images ORDER BY id").fetchall()
    conn.close()
    return [{"name": r["name"], "src": r["src"], "cur_split": r["split"]} for r in rows]


def db_clear():
    conn = get_db()
    conn.execute("DELETE FROM images")
    conn.commit()
    conn.close()


# ─── MODEL ───────────────────────────────────────────────────────────────────
def get_model():
    global _model, _class_names
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            from ultralytics import YOLO
            os.environ.setdefault("YOLO_VERBOSE", "False")
            ext = Path(_model_path).suffix.lower()
            print(f"[MODEL] Loading {_model_path}  format={ext}  device={_device}")
            # YOLO wrapper handles .pt, .onnx, .engine, .tflite, .torchscript
            _model = YOLO(_model_path)
            # Extract class names from the model itself
            if hasattr(_model, "names") and _model.names:
                _class_names = [_model.names[i] for i in sorted(_model.names.keys())]
                print(f"[MODEL] Classes ({len(_class_names)}): {_class_names}")
            print(f"[MODEL] Ready on device={_device}")
    return _model


def reset_model():
    """Clear model from memory (important before starting training on GPU)."""
    global _model
    with _model_lock:
        if _model is not None:
            del _model
            _model = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
    print("[MODEL] Cleared from memory")


def get_img_lock(key):
    with _annotate_meta_lock:
        if key not in _annotate_locks:
            _annotate_locks[key] = threading.Lock()
        return _annotate_locks[key]


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def get_image_dirs():
    skip = {'templates', '.vscode', 'annotated_dataset', 'vedio', '__pycache__', 'Data_annotrator_tool', 'runs'}
    result = []
    for name in sorted(os.listdir(BASE_DIR)):
        if name in skip or name.startswith('.'):
            continue
        full = os.path.join(BASE_DIR, name)
        if os.path.isdir(full):
            count = sum(1 for f in os.listdir(full) if Path(f).suffix.lower() in IMG_EXTS)
            if count > 0:
                result.append({"name": name, "path": full, "count": count})
    return result


def list_images(src_dir):
    return sorted(
        os.path.join(src_dir, f)
        for f in os.listdir(src_dir)
        if Path(f).suffix.lower() in IMG_EXTS
    )


def label_path(split, name):
    validate_split(split)
    return secure_path(OUTPUT_DIR, split, "labels", os.path.splitext(name)[0] + ".txt")


def image_path(split, name):
    validate_split(split)
    return secure_path(OUTPUT_DIR, split, "images", name)


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def annotate_one(img_src_path, lbl_dst_path):
    """Run full-res YOLO on one image → write strict YOLO label file."""
    dataset_type = db_get_meta("dataset_type", "bbox")
    with _model_lock:
        m = get_model()
        results = m(img_src_path, conf=CONF_THRESH, verbose=False, device=_device)
        
    boxes = results[0].boxes
    with open(lbl_dst_path, "w") as f:
        if boxes is not None and len(boxes.cls) > 0:
            if dataset_type == "polygon" and hasattr(results[0], 'masks') and results[0].masks is not None:
                masks = results[0].masks.xyn
                for i in range(len(boxes.cls)):
                    cls_id = int(boxes.cls[i].item())
                    segment = masks[i]
                    if len(segment) >= 3:
                        points_str = " ".join([f"{x} {y}" for x, y in segment])
                        f.write(f"{cls_id} {points_str}\n")
            else:
                # Bounding box logic
                kept_boxes = []
                for i in range(len(boxes.cls)):
                    cls_id = int(boxes.cls[i].item())
                    x, y, w, h = boxes.xywhn[i].tolist()
                    x1, y1, x2, y2 = boxes.xyxyn[i].tolist()
                    
                    # Check for > 90% overlap with already kept boxes
                    is_duplicate = False
                    for kb in kept_boxes:
                        iou = calculate_iou([x1, y1, x2, y2], kb)
                        if iou > 0.90:
                            is_duplicate = True
                            break
                            
                    if is_duplicate:
                        continue
                        
                    kept_boxes.append([x1, y1, x2, y2])
                    
                    x = max(0.0, min(1.0, x))
                    y = max(0.0, min(1.0, y))
                    w = max(0.000001, min(1.0, w))
                    h = max(0.000001, min(1.0, h))
                    f.write(f"{cls_id} {x} {y} {w} {h}\n")


@app.route("/api/dataset_info")
def api_dataset_info():
    path = request.args.get("path", "")
    if not os.path.exists(path):
        return jsonify({"valid": False})
    
    db_file = os.path.join(path, "state.db")
    if not os.path.exists(db_file):
        return jsonify({"valid": False})
    
    try:
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        
        # Get dataset type
        row = conn.execute("SELECT value FROM meta WHERE key='dataset_type'").fetchone()
        ds_type = row["value"] if row else "bbox"
        
        # Get image count
        count = conn.execute("SELECT COUNT(*) as c FROM images").fetchone()["c"]
        conn.close()
        
        return jsonify({"valid": True, "type": ds_type, "count": count})
    except Exception as e:
        return jsonify({"valid": False})

@app.route("/api/source_info")
def api_source_info():
    path = request.args.get("path", "")
    if not os.path.exists(path):
        return jsonify({"valid": False})
    
    count = 0
    try:
        for f in os.listdir(path):
            if Path(f).suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                count += 1
        return jsonify({"valid": True, "count": count})
    except Exception as e:
        return jsonify({"valid": False})

@app.route("/api/datasets")
def api_datasets():
    datasets = []
    try:
        for d in os.listdir(BASE_DIR):
            full_path = os.path.join(BASE_DIR, d)
            if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, "state.db")):
                datasets.append({"name": d, "path": full_path})
        return jsonify(datasets)
    except Exception as e:
        return jsonify([])

def write_yaml():
    with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f:
        f.write("train: ../train/images\nval: ../valid/images\ntest: ../test/images\n\n")
        f.write(f"nc: {len(_class_names)}\nnames: {_class_names}\n")


# ─── BACKGROUND PRE-ANNOTATION ───────────────────────────────────────────────
import queue as _queue_mod

_preanno_queue  = _queue_mod.Queue()
_preanno_active = threading.Event()


def _preanno_worker():
    """Daemon thread: pops image paths from queue and annotates them."""
    while True:
        try:
            job_session, split, name = _preanno_queue.get(timeout=5)
        except _queue_mod.Empty:
            continue
            
        if job_session != _current_session_id:
            _preanno_queue.task_done()
            continue
            
        lbl = label_path(split, name)
        dst = image_path(split, name)
        if not os.path.exists(lbl) and os.path.exists(dst):
            with get_img_lock(f"{split}/{name}"):
                if job_session != _current_session_id:
                    _preanno_queue.task_done()
                    continue
                if not os.path.exists(lbl):
                    try:
                        os.makedirs(os.path.dirname(lbl), exist_ok=True)
                        annotate_one(dst, lbl)
                    except Exception as ex:
                        print(f"[PRE-ANNO] Error on {name}: {ex}")
        _preanno_queue.task_done()


_preanno_thread = threading.Thread(target=_preanno_worker, daemon=True)
_preanno_thread.start()


def enqueue_preanno(split, names):
    """Add a list of image names to the pre-annotation queue."""
    session_id = _current_session_id
    for n in names:
        _preanno_queue.put((session_id, split, n))


# ─── API: DEVICE INFO ────────────────────────────────────────────────────────
@app.route("/api/device_info")
def api_device_info():
    return jsonify(get_device_info())


# ─── API: SET MODEL ──────────────────────────────────────────────────────────
@app.route("/api/set_model", methods=["POST"])
def api_set_model():
    global _model_path, _device, _class_names, _model
    data = request.get_json()
    new_path = data.get("model_path", "")
    explicit_device = data.get("device", "auto")

    if new_path and not os.path.exists(new_path):
        candidate_models = os.path.join(MODELS_DIR, new_path)
        candidate_base = os.path.join(BASE_DIR, new_path)
        if os.path.exists(candidate_models):
            new_path = candidate_models
        elif os.path.exists(candidate_base):
            new_path = candidate_base
        else:
            return jsonify({"error": f"Model not found: {new_path}"}), 400

    ext = Path(new_path).suffix.lower() if new_path else ""
    if new_path and ext not in MODEL_EXTS:
        return jsonify({"error": f"Unsupported format '{ext}'."}), 400

    reset_model() 

    if new_path:
        _model_path = new_path

    if explicit_device == "auto" or not explicit_device:
        _device = auto_detect_device()
    else:
        _device = explicit_device
        
    print(f"[MODEL] Target device assigned: {_device}")

    # Load eagerly to extract class names
    try:
        get_model()
        db_set_meta("classes", json.dumps(_class_names))
        dev_info = get_device_info()
        dev_info["selected"] = _device
        return jsonify({"status": "ok", "classes": _class_names, "device": _device,
                        "device_info": dev_info, "model_path": _model_path})
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500


@app.route("/api/classes")
def api_classes():
    return jsonify({"classes": _class_names})


# ─── API: CLASS STATISTICS ───────────────────────────────────────────────────
@app.route("/api/class_stats")
def api_class_stats():
    """Count per-class box occurrences across the whole annotated dataset."""
    counts = {name: 0 for name in _class_names}
    for split in ["train", "valid", "test"]:
        lbl_dir = os.path.join(OUTPUT_DIR, split, "labels")
        if not os.path.isdir(lbl_dir):
            continue
        for fn in os.listdir(lbl_dir):
            if not fn.endswith(".txt"):
                continue
            try:
                with open(os.path.join(lbl_dir, fn)) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            name = _class_names[cls_id] if cls_id < len(_class_names) else str(cls_id)
                            counts[name] = counts.get(name, 0) + 1
            except (FileNotFoundError, OSError):
                # File might have been moved/deleted by split/cleanup operations
                pass
    return jsonify(counts)


# ─── API: SOURCE DISCOVERY ───────────────────────────────────────────────────
@app.route("/api/browse_folder")
def api_browse_folder():
    try:
        # Tkinter requires to run on the main thread ideally, but for simple dialogs on Windows/Linux this usually works.
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder = filedialog.askdirectory(parent=root, title="Select Directory")
        root.destroy()
        return jsonify({"path": folder})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/source_dirs")
def api_source_dirs():
    return jsonify(get_image_dirs())


# ─── API: EXISTING DATASETS ──────────────────────────────────────────────────
@app.route("/api/existing_datasets")
def api_existing_datasets():
    result = []
    for name in sorted(os.listdir(BASE_DIR)):
        if name.startswith('.'):
            continue
        full = os.path.join(BASE_DIR, name)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, "state.db")):
            result.append({"name": name, "path": full})
    return jsonify(result)


@app.route("/api/load_existing_dataset", methods=["POST"])
def api_load_existing_dataset():
    data = request.get_json()
    dataset_dir = data.get("dataset_dir", "")
    if not dataset_dir or not os.path.isdir(dataset_dir):
        return jsonify({"error": "invalid directory"}), 400
    
    if not os.path.exists(os.path.join(dataset_dir, "state.db")):
        return jsonify({"error": "state.db not found"}), 400

    set_active_dataset(dataset_dir, is_existing=True)
    return jsonify({"status": "ok"})


# ─── API: DATASET STATE ──────────────────────────────────────────────────────
@app.route("/api/stats")
def api_stats():
    result = {}
    for split in ["train", "valid", "test"]:
        entries   = db_get_entries(split)
        annotated = sum(1 for e in entries if os.path.exists(label_path(split, e["name"])))
        result[split] = {"total": len(entries), "annotated": annotated}
    return jsonify(result)


@app.route("/api/state")
def api_state():
    if db_count("train") + db_count("valid") + db_count("test") == 0:
        return jsonify({"ready": False})
    counts = {}
    for split in ["train", "valid", "test"]:
        entries   = db_get_entries(split)
        annotated = sum(1 for e in entries if os.path.exists(label_path(split, e["name"])))
        counts[split] = {"total": len(entries), "annotated": annotated}
    return jsonify({"ready": True, "counts": counts, "source": db_get_meta("source", ""), "dataset_type": db_get_meta("dataset_type", "bbox")})


# ─── API: INGEST — hardlink (or copy) images into train/ ─────────────────────
@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    data = request.get_json()
    src  = data.get("source_dir", "")
    dataset_type = data.get("dataset_type", "bbox")
    if not src or not os.path.isdir(src):
        return jsonify({"error": "invalid source dir"}), 400

    set_active_dataset(src)
    job_id = f"ingest_{int(time.time())}"
    _job_progress[job_id] = {"total": 0, "done": 0, "status": "starting"}

    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

    def run():
        imgs  = list_images(src)
        total = len(imgs)
        _job_progress[job_id].update({"total": total, "status": "linking"})
        entries = []
        for i, img_path in enumerate(imgs):
            name = os.path.basename(img_path)
            dst  = image_path("train", name)
            if not os.path.exists(dst):
                try:
                    os.link(img_path, dst)          # hardlink (no copy!)
                except OSError:
                    shutil.copy2(img_path, dst)     # fallback for cross-device
            entries.append({"name": name, "src": img_path})
            _job_progress[job_id]["done"] = i + 1

        db_clear()
        db_insert_entries(entries, "train")
        db_set_meta("source", src)
        db_set_meta("dataset_type", dataset_type)
        write_yaml()

        # Queue background pre-annotation
        enqueue_preanno("train", [e["name"] for e in entries])

        _job_progress[job_id]["status"] = "done"

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/ingest_append", methods=["POST"])
def api_ingest_append():
    data = request.get_json()
    src  = data.get("source_dir", "")
    target_split = data.get("split", "train")
    try:
        validate_split(target_split)
    except ValueError:
        return jsonify({"error": "Invalid split"}), 400

    if not src or not os.path.isdir(src):
        return jsonify({"error": "invalid source dir"}), 400

    job_id = f"ingest_{int(time.time())}"
    _job_progress[job_id] = {"total": 0, "done": 0, "status": "starting"}

    os.makedirs(os.path.join(OUTPUT_DIR, target_split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, target_split, "labels"), exist_ok=True)

    def run():
        imgs  = list_images(src)
        total = len(imgs)
        _job_progress[job_id].update({"total": total, "status": "linking"})
        entries = []
        for i, img_path in enumerate(imgs):
            name = os.path.basename(img_path)
            dst  = image_path(target_split, name)
            if not os.path.exists(dst):
                try:
                    os.link(img_path, dst)
                except OSError:
                    shutil.copy2(img_path, dst)
                entries.append({"name": name, "src": img_path})
            _job_progress[job_id]["done"] = i + 1

        if entries:
            db_insert_entries(entries, target_split)
            write_yaml()
            enqueue_preanno(target_split, [e["name"] for e in entries])

        _job_progress[job_id]["status"] = "done"

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id})


# ─── API: AUTO-SPLIT (annotated images only) ──────────────────────────────────
@app.route("/api/auto_split", methods=["POST"])
def api_auto_split():
    data   = request.get_json()
    ratios = data.get("ratios", [0.70, 0.20, 0.10])
    
    if (
        not isinstance(ratios, list)
        or len(ratios) != 3
        or not all(isinstance(r, (int, float)) for r in ratios)
        or any(r < 0 or r > 1 for r in ratios)
        or abs(sum(ratios) - 1.0) > 0.001
    ):
        return jsonify({"error": "Invalid ratios. Must be exactly 3 numbers between 0 and 1 that sum to 1."}), 400

    all_entries = db_all_entries()
    if not all_entries:
        return jsonify({"error": "no dataset loaded"}), 400

    # Only split images that are annotated; leave the rest in their current split
    annotated   = [e for e in all_entries if os.path.exists(label_path(e["cur_split"], e["name"]))]
    unannotated = [e for e in all_entries if not os.path.exists(label_path(e["cur_split"], e["name"]))]

    random.seed(None)
    random.shuffle(annotated)

    n  = len(annotated)
    nt = max(0, int(round(n * ratios[0])))
    nv = max(0, int(round(n * ratios[1])))
    ntest = n - nt - nv
    
    # Handle extremely small datasets gracefully
    if n > 0 and nt == 0 and nv == 0 and ntest == 0:
        nt = n
        ntest = 0
    elif ntest < 0:
        # Adjustment if rounding pushed nt+nv > n
        if nv > 0: nv += ntest
        else: nt += ntest
        ntest = 0

    splits = {
        "train": annotated[:nt],
        "valid": annotated[nt:nt + nv],
        "test":  annotated[nt + nv:]
    }

    # Move files only when split changes
    for new_split, entries in splits.items():
        for e in entries:
            cur = e["cur_split"]
            if cur == new_split:
                continue
            for kind in ["images", "labels"]:
                ext   = e["name"] if kind == "images" else os.path.splitext(e["name"])[0] + ".txt"
                src_p = os.path.join(OUTPUT_DIR, cur,       kind, ext)
                dst_p = os.path.join(OUTPUT_DIR, new_split, kind, ext)
                if os.path.exists(src_p):
                    os.makedirs(os.path.dirname(dst_p), exist_ok=True)
                    shutil.move(src_p, dst_p)
            db_update_split(e["name"], cur, new_split)

    counts = {k: len(v) for k, v in splits.items()}
    counts["unannotated_skipped"] = len(unannotated)
    return jsonify({"status": "ok", "counts": counts})


# ─── API: SSE PROGRESS ───────────────────────────────────────────────────────
@app.route("/api/progress/<job_id>")
def api_progress(job_id):
    def generate():
        while True:
            prog = _job_progress.get(job_id, {"status": "unknown"})
            yield f"data: {json.dumps(prog)}\n\n"
            if prog.get("status") in ("done",) or str(prog.get("status", "")).startswith("error"):
                # Clean up old job after reporting completion
                _job_progress.pop(job_id, None)
                break
            time.sleep(0.3)
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─── API: IMAGES LIST ────────────────────────────────────────────────────────
@app.route("/api/images")
def api_images():
    split = request.args.get("split", "train")
    return jsonify([
        {"name": e["name"], "annotated": os.path.exists(label_path(split, e["name"]))}
        for e in db_get_entries(split)
    ])


# ─── API: DATASET PAGE (for Grid View) ───────────────────────────────────────
@app.route("/api/dataset_page")
def api_dataset_page():
    split = request.args.get("split", "train")
    page = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 50))
    filt = request.args.get("filter", "all")  # 'all', 'done', 'pending'
    class_id_str = request.args.get("class_id", "")
    class_id = int(class_id_str) if class_id_str.isdigit() else -1

    entries = db_get_entries(split)
    
    filtered = []
    for e in entries:
        lbl_p = label_path(split, e["name"])
        is_annotated = os.path.exists(lbl_p)
        if filt == "done" and not is_annotated:
            continue
        if filt == "pending" and is_annotated:
            continue
            
        # Optional class filtering
        has_class = False
        if class_id >= 0:
            if not is_annotated:
                continue
            try:
                with open(lbl_p) as f:
                    for line in f:
                        if line.strip().startswith(f"{class_id} "):
                            has_class = True
                            break
            except Exception:
                pass
            if not has_class:
                continue

        filtered.append({"name": e["name"], "annotated": is_annotated, "lbl_path": lbl_p})
        
    total = len(filtered)
    pages = max(1, (total + limit - 1) // limit)
    page = max(1, min(page, pages))
    
    start = (page - 1) * limit
    end = start + limit
    slice_entries = filtered[start:end]
    
    results = []
    for e in slice_entries:
        boxes = []
        if e["annotated"]:
            try:
                with open(e["lbl_path"]) as f:
                    for line in f:
                        p = line.strip().split()
                        if len(p) == 5:
                            boxes.append({
                                "cls": int(p[0]),
                                "x": float(p[1]), "y": float(p[2]),
                                "w": float(p[3]), "h": float(p[4])
                            })
                        elif len(p) > 5:
                            pts = []
                            for j in range(1, len(p) - 1, 2):
                                pts.append({"x": float(p[j]), "y": float(p[j+1])})
                            boxes.append({"cls": int(p[0]), "pts": pts})
            except Exception:
                pass
        results.append({
            "name": e["name"],
            "annotated": e["annotated"],
            "boxes": boxes
        })
        
    return jsonify({
        "total": total,
        "page": page,
        "pages": pages,
        "images": results
    })


# ─── API: SERVE IMAGE ────────────────────────────────────────────────────────
@app.route("/api/image/<split>/<filename>")
def api_image(split, filename):
    dst = image_path(split, filename)
    if not os.path.exists(dst):
        return jsonify({"error": "not found"}), 404
    suffix = Path(filename).suffix.lower()
    mime   = "image/png" if suffix == ".png" else "image/jpeg"
    return send_file(dst, mimetype=mime)


# ─── API: THUMBNAIL ──────────────────────────────────────────────────────────
_thumb_cache = {}   # (split, name) -> bytes
_THUMB_CACHE_MAX = 400   # cap at 400 entries (~170 MB worst case)

@app.route("/api/thumb/<split>/<filename>")
def api_thumb(split, filename):
    key = (split, filename)
    if key in _thumb_cache:
        buf = io.BytesIO(_thumb_cache[key])
        return send_file(buf, mimetype="image/jpeg")

    dst = image_path(split, filename)
    if not os.path.exists(dst):
        return jsonify({"error": "not found"}), 404
    try:
        from PIL import Image as PILImage
        img = PILImage.open(dst)
        img.thumbnail((450, 450))
        # Always convert to RGB to avoid RGBA/P mode JPEG errors
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        data = buf.getvalue()
        # Evict oldest entry if over limit
        if len(_thumb_cache) >= _THUMB_CACHE_MAX:
            _thumb_cache.pop(next(iter(_thumb_cache)))
        _thumb_cache[key] = data
        return send_file(io.BytesIO(data), mimetype="image/jpeg")
    except Exception:
        # If PIL not available, serve original with correct mime type
        suffix = Path(filename).suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        return send_file(dst, mimetype=mime)


# ─── API: LABELS (on-demand annotation) ──────────────────────────────────────
@app.route("/api/labels/<split>/<filename>")
def api_labels(split, filename):
    lbl = label_path(split, filename)
    dst = image_path(split, filename)

    if not os.path.exists(lbl) and os.path.exists(dst):
        with get_img_lock(f"{split}/{filename}"):
            if not os.path.exists(lbl):
                os.makedirs(os.path.dirname(lbl), exist_ok=True)
                annotate_one(dst, lbl)

    boxes = []
    if os.path.exists(lbl):
        with open(lbl) as f:
            for i, line in enumerate(f):
                p = line.strip().split()
                if len(p) == 5:
                    boxes.append({"id": i, "cls": int(p[0]),
                                  "x": float(p[1]), "y": float(p[2]),
                                  "w": float(p[3]), "h": float(p[4])})
                elif len(p) > 5:
                    pts = []
                    for j in range(1, len(p) - 1, 2):
                        pts.append({"x": float(p[j]), "y": float(p[j+1])})
                    boxes.append({"id": i, "cls": int(p[0]), "pts": pts})
    return jsonify(boxes)


# ─── API: SAVE LABELS ────────────────────────────────────────────────────────
@app.route("/api/save", methods=["POST"])
def api_save():
    data  = request.get_json()
    split = data.get("split", "train")
    fname = data.get("filename", "")
    boxes = data.get("boxes", [])
    lbl   = label_path(split, fname)
    os.makedirs(os.path.dirname(lbl), exist_ok=True)
    with open(lbl, "w") as f:
        for b in boxes:
            if "pts" in b:
                points_str = " ".join([f"{pt['x']} {pt['y']}" for pt in b["pts"]])
                f.write(f"{b['cls']} {points_str}\n")
            else:
                f.write(f"{b['cls']} {b['x']} {b['y']} {b['w']} {b['h']}\n")
    return jsonify({"status": "ok"})


# ─── API: DELETE IMAGE ───────────────────────────────────────────────────────
@app.route("/api/delete_image", methods=["POST"])
def api_delete_image():
    data  = request.get_json()
    split = data.get("split", "train")
    fname = data.get("filename", "")
    for p in [image_path(split, fname), label_path(split, fname)]:
        if os.path.exists(p):
            os.remove(p)
    db_delete_image(fname, split)
    return jsonify({"status": "deleted"})


# ─── API: MODEL TRAINING ─────────────────────────────────────────────────────
@app.route("/api/train", methods=["POST"])
def api_train():
    global _train_process, _train_log_buffer
    data       = request.get_json()
    model_file = data.get("model",   "yolov8n.pt")
    epochs     = int(data.get("epochs",    50))
    batch      = int(data.get("batch",     16))
    imgsz      = int(data.get("imgsz",    640))
    lr0        = float(data.get("lr0",   0.01))
    device     = data.get("device",  "cpu")
    project    = data.get("project", "runs/train")
    name       = data.get("name",    "vehicle_detect")
    yaml_path  = os.path.join(OUTPUT_DIR, "data.yaml")

    if device == "cpu":
        return jsonify({"error": "Training on CPU is disabled to prevent system freeze. A CUDA GPU or MPS is required."}), 400

    if not os.path.exists(yaml_path):
        return jsonify({"error": "data.yaml not found — run auto-split first"}), 400

    # ── Release inference model from GPU before training ──
    reset_model()

    cmd = [
        "python", "-m", "ultralytics", "train",
        f"model={model_file}",
        f"data={yaml_path}",
        f"epochs={epochs}",
        f"batch={batch}",
        f"imgsz={imgsz}",
        f"lr0={lr0}",
        f"device={device}",
        f"project={project}",
        f"name={name}",
        "exist_ok=True",
        "verbose=True"
    ]

    with _train_lock:
        if _train_process and _train_process.poll() is None:
            return jsonify({"error": "Training already running"}), 400
        _train_log_buffer = []
        _train_process = subprocess.Popen(
            cmd, cwd=BASE_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )

    def read_logs():
        for line in _train_process.stdout:
            line = line.rstrip()
            with _train_lock:
                _train_log_buffer.append(line)
                if len(_train_log_buffer) > 500:
                    _train_log_buffer.pop(0)

    threading.Thread(target=read_logs, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/train/stop", methods=["POST"])
def api_train_stop():
    global _train_process
    with _train_lock:
        if _train_process and _train_process.poll() is None:
            _train_process.terminate()
            return jsonify({"status": "stopped"})
    return jsonify({"status": "not_running"})


@app.route("/api/train/logs")
def api_train_logs():
    def generate():
        sent = 0
        while True:
            with _train_lock:
                new_lines = _train_log_buffer[sent:]
                for line in new_lines:
                    yield f"data: {json.dumps({'line': line})}\n\n"
                sent += len(new_lines)
                done = _train_process is None or _train_process.poll() is not None
            if done and sent >= len(_train_log_buffer):
                yield f"data: {json.dumps({'line': '--- Training complete ---', 'done': True})}\n\n"
                break
            time.sleep(0.5)
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/train/status")
def api_train_status():
    if _train_process is None:
        return jsonify({"status": "idle"})
    code = _train_process.poll()
    if code is None:
        return jsonify({"status": "running"})
    return jsonify({"status": "done", "exit_code": code})


@app.route("/api/models")
def api_models():
    # Scan MODELS_DIR for all supported model formats
    all_models = []
    if os.path.exists(MODELS_DIR):
        for root, _, files in os.walk(MODELS_DIR):
            for f in files:
                if Path(f).suffix.lower() in MODEL_EXTS:
                    rel_path = os.path.relpath(os.path.join(root, f), MODELS_DIR)
                    all_models.append(rel_path.replace("\\", "/"))
    presets = [
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
        "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"
    ]
    return jsonify({"files": all_models, "presets": presets})


# ─── API: EXPORT DATASET ─────────────────────────────────────────────────────
@app.route("/api/export")
def api_export():
    """Zips the annotated dataset to disk and serves it for download."""
    fmt = request.args.get("format", "yolo")
    temp_fd, temp_path = tempfile.mkstemp(suffix=".zip", prefix=f"{_current_dataset_name}_")
    os.close(temp_fd) 
    
    with zipfile.ZipFile(temp_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        if fmt == "coco":
            for split in ["train", "valid", "test"]:
                split_dir = os.path.join(OUTPUT_DIR, split)
                if not os.path.exists(split_dir): continue
                img_dir = os.path.join(split_dir, "images")
                lbl_dir = os.path.join(split_dir, "labels")
                if not os.path.exists(img_dir): continue
                
                coco = {
                    "info": {"description": "Exported from Data Annotator Tool"},
                    "categories": [{"id": i, "name": n, "supercategory": "none"} for i, n in enumerate(_class_names)],
                    "images": [],
                    "annotations": []
                }
                
                ann_id = 1
                for img_id, img_file in enumerate(os.listdir(img_dir), start=1):
                    img_path = os.path.join(img_dir, img_file)
                    h, w = 1000, 1000
                    try:
                        import cv2
                        im = cv2.imread(img_path)
                        if im is not None:
                            h, w = im.shape[:2]
                    except: pass
                    
                    coco["images"].append({"id": img_id, "file_name": img_file, "width": w, "height": h})
                    zf.write(img_path, f"{split}/images/{img_file}")
                    
                    lbl_file = os.path.splitext(img_file)[0] + ".txt"
                    lbl_path = os.path.join(lbl_dir, lbl_file)
                    if os.path.exists(lbl_path):
                        with open(lbl_path) as f:
                            for line in f:
                                p = line.strip().split()
                                if len(p) == 5:
                                    cls = int(p[0])
                                    cx, cy, bw, bh = map(float, p[1:5])
                                    x = (cx - bw/2) * w
                                    y = (cy - bh/2) * h
                                    width = bw * w
                                    height = bh * h
                                    coco["annotations"].append({
                                        "id": ann_id, "image_id": img_id, "category_id": cls,
                                        "bbox": [x, y, width, height], "area": width*height,
                                        "iscrowd": 0, "segmentation": []
                                    })
                                    ann_id += 1
                                elif len(p) > 5:
                                    cls = int(p[0])
                                    pts = list(map(float, p[1:]))
                                    seg = []
                                    xs, ys = [], []
                                    for i in range(0, len(pts), 2):
                                        px, py = pts[i]*w, pts[i+1]*h
                                        xs.append(px); ys.append(py)
                                        seg.extend([px, py])
                                    if len(xs) > 0:
                                        xmin, ymin = min(xs), min(ys)
                                        xmax, ymax = max(xs), max(ys)
                                        coco["annotations"].append({
                                            "id": ann_id, "image_id": img_id, "category_id": cls,
                                            "bbox": [xmin, ymin, xmax-xmin, ymax-ymin], "area": (xmax-xmin)*(ymax-ymin),
                                            "iscrowd": 0, "segmentation": [seg]
                                        })
                                        ann_id += 1
                
                zf.writestr(f"{split}/_annotations.coco.json", json.dumps(coco))
        else:
            for root, dirs, files in os.walk(OUTPUT_DIR):
                if "state.db" in files: files.remove("state.db")
                for file in files:
                    fpath = os.path.join(root, file)
                    arcname = os.path.relpath(fpath, OUTPUT_DIR)
                    zf.write(fpath, arcname)
                
    @after_this_request
    def remove_file(response):
        try:
            os.remove(temp_path)
        except Exception:
            pass
        return response

    return send_file(temp_path, download_name=f"{_current_dataset_name}_export_{fmt}.zip", as_attachment=True)


# ─── MAIN ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/style.css")
def serve_css():
    return send_file(os.path.join(os.path.dirname(__file__), "style.css"))

@app.route("/app.js")
def serve_js():
    return send_file(os.path.join(os.path.dirname(__file__), "app.js"))


if __name__ == "__main__":
    init_db()
    print("=" * 62)
    print("  Data Annotation Tool")
    print(f"  Output : {OUTPUT_DIR}")
    print(f"  Open   : http://127.0.0.1:8051")
    print("=" * 62)
    app.run(debug=False, host="0.0.0.0", port=8051, threaded=True)
