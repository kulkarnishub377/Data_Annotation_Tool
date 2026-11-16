import os
import io
import sys
import json
import shutil
import random
import sqlite3
import threading
import time
import subprocess
import zipfile
import mimetypes
import collections
import concurrent.futures
import itertools
import ast
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, Response, after_this_request
from werkzeug.utils import secure_filename
import uuid
import tempfile
import re
import configparser
from dotenv import load_dotenv

# ─── WINDOWLESS GUI STDOUT/STDERR GUARDS ──────────────────────────────────────
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

load_dotenv()

# ─── PYTORCH 2.6+ COMPATIBILITY ──────────────────────────────────────────────
try:
    import torch
    _orig_torch_load = torch.load
    def _safe_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _safe_torch_load
except Exception:
    pass

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
if os.path.exists(config_path):
    config.read(config_path)

# ─── APP ─────────────────────────────────────────────────────────────────────
__version__ = "1.1.0"

if getattr(sys, 'frozen', False):
    # Running inside a PyInstaller frozen bundle
    APP_DIR = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    BASE_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = APP_DIR

app = Flask(__name__, template_folder=APP_DIR, static_folder=os.path.join(APP_DIR, 'static'))
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.after_request
def add_no_cache_headers(response):
    if request.path.startswith('/static/') or request.path == '/':
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

def _resolve_dir(path_val, default_base):
    if not path_val:
        return default_base
    if os.path.isabs(path_val):
        return os.path.abspath(path_val)
    return os.path.abspath(os.path.join(APP_DIR, path_val))

# Highest priority: ANNOTATOR_* env overrides (for test/dev instances pointed at a scratch
# folder). Then config.ini, then plain env vars (.env), then auto-detection.
if os.environ.get("ANNOTATOR_BASE_DIR"):
    BASE_DIR = _resolve_dir(os.environ["ANNOTATOR_BASE_DIR"], APP_DIR)
elif config.has_option("Paths", "BASE_DIR"):
    BASE_DIR = _resolve_dir(config.get("Paths", "BASE_DIR"), APP_DIR)
elif "BASE_DIR" in os.environ:
    BASE_DIR = _resolve_dir(os.environ.get("BASE_DIR"), APP_DIR)
else:
    # Auto-detect parent directory as workspace BASE_DIR if parent contains models/ or dataset_*
    parent_dir = os.path.abspath(os.path.join(APP_DIR, ".."))
    if (os.path.exists(os.path.join(parent_dir, "models")) or
        os.path.exists(os.path.join(parent_dir, "dataset_default")) or
        os.path.exists(os.path.join(parent_dir, "best.pt")) or
        os.path.exists(os.path.join(parent_dir, "bestv8_p.pt"))):
        BASE_DIR = parent_dir
    else:
        BASE_DIR = APP_DIR

if os.environ.get("ANNOTATOR_MODELS_DIR"):
    MODELS_DIR = _resolve_dir(os.environ["ANNOTATOR_MODELS_DIR"], os.path.join(BASE_DIR, "models"))
elif config.has_option("Paths", "MODELS_DIR"):
    MODELS_DIR = _resolve_dir(config.get("Paths", "MODELS_DIR"), os.path.join(BASE_DIR, "models"))
elif "MODELS_DIR" in os.environ:
    MODELS_DIR = _resolve_dir(os.environ.get("MODELS_DIR"), os.path.join(BASE_DIR, "models"))
else:
    if os.path.exists(os.path.join(BASE_DIR, "models")):
        MODELS_DIR = os.path.join(BASE_DIR, "models")
    elif os.path.exists(os.path.join(APP_DIR, "models")):
        MODELS_DIR = os.path.join(APP_DIR, "models")
    else:
        MODELS_DIR = os.path.join(BASE_DIR, "models")

_current_dataset_name = "default"
if os.environ.get("ANNOTATOR_OUTPUT_DIR"):
    OUTPUT_DIR = _resolve_dir(os.environ["ANNOTATOR_OUTPUT_DIR"], os.path.join(BASE_DIR, f"dataset_{_current_dataset_name}"))
elif config.has_option("Paths", "OUTPUT_DIR"):
    OUTPUT_DIR = _resolve_dir(config.get("Paths", "OUTPUT_DIR"), os.path.join(BASE_DIR, f"dataset_{_current_dataset_name}"))
else:
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(BASE_DIR, f"dataset_{_current_dataset_name}"))


def dataset_display_name(path):
    """Folder name without the 'dataset_' prefix (dataset_EPE_VIDS -> EPE_VIDS)."""
    base = os.path.basename(os.path.normpath(path))
    return base[len("dataset_"):] if base.startswith("dataset_") and len(base) > len("dataset_") else base


_current_dataset_name = dataset_display_name(OUTPUT_DIR)

if config.has_option("Settings", "CONF_THRESH"):
    CONF_THRESH = float(config.get("Settings", "CONF_THRESH"))
else:
    CONF_THRESH = float(os.environ.get("CONF_THRESH", "0.25"))

VALID_SPLITS = {"train", "valid", "test"}
SPLITS = ["train", "valid", "test"]
DEFAULT_SPLIT_RATIOS = [0.7, 0.2, 0.1]

def validate_split(split):
    if split not in VALID_SPLITS:
        raise ValueError("Invalid split")


def validate_name(name):
    """Image file names are plain basenames (no directories)."""
    if not isinstance(name, str) or not name or name in (".", "..") or \
            "/" in name or "\\" in name or "\x00" in name:
        raise ValueError("Invalid filename")

_secure_path_cache = {}

def secure_path(base_dir, *paths):
    """Safely join paths and ensure the result is within base_dir."""
    if base_dir not in _secure_path_cache:
        _secure_path_cache[base_dir] = os.path.abspath(base_dir)
    base_abs = _secure_path_cache[base_dir]
    
    final_path = os.path.abspath(os.path.join(base_abs, *paths))
    if os.path.commonpath([base_abs, final_path]) != base_abs:
        raise ValueError("Path traversal attempt detected")
    return final_path
IMG_EXTS    = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
# Supported inference model formats
MODEL_EXTS  = {'.pt', '.onnx', '.engine', '.tflite', '.torchscript'}

# Some platforms (notably Windows registry-backed mimetypes) lack or mis-map these.
for _ext, _mt in ((".webp", "image/webp"), (".bmp", "image/bmp"), (".png", "image/png"),
                  (".jpg", "image/jpeg"), (".jpeg", "image/jpeg")):
    mimetypes.add_type(_mt, _ext)


def guess_mimetype(filename):
    mt, _ = mimetypes.guess_type(filename)
    return mt or "application/octet-stream"

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
    info = {"cpu": True, "cuda": [], "mps": False, "selected": _device, "has_gpu": False}
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
        info["has_gpu"] = bool(len(info["cuda"]) > 0 or info["mps"])
    except Exception:
        pass
    return info


# Defaults (overridden at runtime via /api/set_model or DB)
_default_custom = os.path.join(MODELS_DIR, "bestv8_p.pt")
_model_path   = _default_custom if os.path.exists(_default_custom) else os.environ.get("MODEL_PATH", "yolov8n.pt")
_device       = auto_detect_device()   # ← auto GPU on start
# Class list for the active dataset. Load order on dataset open:
# DB meta -> data.yaml names -> loaded model names -> [] (see _load_classes()).
_class_names  = []
# Detection settings (GET/POST /api/settings). Persisted per dataset in state.db meta
# (keys conf / iou / auto_annotate); fallback = config.ini/env CONF_THRESH, 0.7, true.
DEFAULT_IOU    = 0.7
_auto_annotate = True
# True once the user explicitly loads a model via /api/set_model. Until then the default
# model is only used if its file already exists locally - never auto-downloaded.
_model_explicit = False
_conf          = CONF_THRESH
_iou           = DEFAULT_IOU

# Model classes are kept separate from dataset classes (contract section 4/4b).
_model_names        = []   # names of the loaded model, in model index order
_model_class_map    = {}   # model_idx -> dataset target class name (what gets auto-labelled)
_model_map_path     = ""   # model path the in-memory mapping belongs to
_model_map_names    = []   # model names the in-memory mapping belongs to
_model_map_warnings = []   # warnings from the last mapping (re)establishment
_empty_map_logged   = False
COCO_VEHICLES = ["car", "motorcycle", "bus", "truck"]
LARGE_MODEL_CLASSES = 20


def default_model_classes(names):
    """COCO model -> vehicle classes; custom model -> all its classes (mirrors Data Capture Tool)."""
    names = list(names)
    if "person" in names and all(c in names for c in COCO_VEHICLES):
        return list(COCO_VEHICLES)
    return names


def model_kind(names):
    return "coco" if ("person" in names and "traffic light" in names) else "custom"


def default_mapping(model_names, dataset_classes):
    """Default model_idx -> target mapping.

    Dataset without classes: default classes map to their own names.
    Dataset with classes: only default classes that match an existing dataset
    class (case-insensitive) are mapped, so dataset classes are never changed implicitly.
    """
    wanted = set(default_model_classes(model_names))
    lower = {c.lower(): c for c in dataset_classes}
    mapping = {}
    for idx, name in enumerate(model_names):
        if name not in wanted:
            continue
        if not dataset_classes:
            mapping[idx] = name
        elif name.lower() in lower:
            mapping[idx] = lower[name.lower()]
    return mapping


def mapping_targets_in_order(mapping):
    """Unique target names in model-index order."""
    out = []
    for idx in sorted(mapping):
        if mapping[idx] not in out:
            out.append(mapping[idx])
    return out


def build_label_lines(dets, mapping, class_names, dataset_type="bbox", iou_dedupe=0.90):
    """Convert model detections into YOLO label lines for the dataset.

    dets: iterable of (model_cls_id, (x1, y1, x2, y2) normalised, segment or None)
          where segment is a sequence of normalised (x, y) points.
    mapping: {model_idx: dataset target name}; unmapped detections are dropped.
    The written class id is the index of the target in ``class_names``.
    """
    index_of = {n: i for i, n in enumerate(class_names)}
    lines, kept = [], []
    for model_cls, xyxy, segment in dets:
        target = mapping.get(int(model_cls))
        if target is None or target not in index_of:
            continue
        cls_id = index_of[target]
        x1, y1, x2, y2 = (_clamp01(v) for v in xyxy)
        if dataset_type == "polygon" and segment is not None and len(segment) >= 3:
            pts = " ".join(f"{_clamp01(x):.6f} {_clamp01(y):.6f}" for x, y in segment)
            lines.append(f"{cls_id} {pts}")
            continue
        # Drop near-duplicates (> 90% overlap with an already kept box)
        if any(calculate_iou([x1, y1, x2, y2], kb) > iou_dedupe for kb in kept):
            continue
        kept.append([x1, y1, x2, y2])
        if dataset_type == "polygon":
            # Detection-only model on a polygon dataset: box -> 4-point polygon
            lines.append(f"{cls_id} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} "
                         f"{x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}")
        else:
            w = max(0.000001, x2 - x1)
            h = max(0.000001, y2 - y1)
            lines.append(f"{cls_id} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {w:.6f} {h:.6f}")
    return lines


class LogBuffer:
    """Bounded line buffer addressed by a monotonically increasing line counter.

    Readers keep an absolute cursor (number of lines they have consumed). Old
    lines are dropped once more than ``maxlen`` are buffered; a reader that fell
    behind simply skips ahead to the oldest line still available, so a stream
    never stalls regardless of how many lines were produced.
    """

    def __init__(self, maxlen=2000):
        self._lines = collections.deque(maxlen=maxlen)
        self._total = 0
        self._lock = threading.Lock()
        self.closed = False

    def reset(self):
        with self._lock:
            self._lines.clear()
            self._total = 0
            self.closed = False

    def append(self, line):
        with self._lock:
            self._lines.append(line)
            self._total += 1

    def close(self):
        self.closed = True

    @property
    def total(self):
        with self._lock:
            return self._total

    @property
    def start(self):
        """Absolute index of the oldest buffered line."""
        with self._lock:
            return self._total - len(self._lines)

    def read(self, cursor):
        """Return (lines_after_cursor, new_cursor)."""
        with self._lock:
            start = self._total - len(self._lines)
            if cursor < start or cursor > self._total:
                # fell behind (lines evicted) or buffer was reset -> skip ahead / restart
                cursor = start
            lines = list(itertools.islice(self._lines, cursor - start, None))
            return lines, self._total


# ─── GLOBALS ─────────────────────────────────────────────────────────────────
_model               = None
_model_lock          = threading.RLock()
_annotate_locks      = {}
_annotate_meta_lock  = threading.Lock()
_job_progress        = {}   # job_id -> dict
_JOB_TTL_SECONDS     = 600  # finished jobs stay queryable for reconnecting clients
_train_process       = None
_train_log_buffer    = LogBuffer(maxlen=2000)
_train_lock          = threading.Lock()
_labels_write_lock   = threading.RLock()   # held by class merge/delete/restore and label writers
_auto_split_running  = False

# Pre-annotation background queue
_preanno_queue  = None   # queue.Queue
_preanno_thread = None

DB_PATH = os.path.join(OUTPUT_DIR, "state.db")
_current_session_id = uuid.uuid4().hex

def set_active_dataset(src_dir, is_existing=False):
    global OUTPUT_DIR, DB_PATH, _current_session_id, _current_dataset_name
    
    if is_existing:
        OUTPUT_DIR = src_dir
        _current_dataset_name = dataset_display_name(src_dir)
    else:
        base = os.path.basename(os.path.normpath(src_dir))
        safe_name = secure_filename(base)
        if not safe_name:
            safe_name = "default"
        
        _current_dataset_name = safe_name
        OUTPUT_DIR = os.path.join(BASE_DIR, f"dataset_{safe_name}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    DB_PATH = os.path.join(OUTPUT_DIR, "state.db")
    clear_thumb_cache()
    init_db()   # loads this dataset's classes + detection settings
    with _model_lock:
        if _model_names:
            establish_model_mapping()
    _current_session_id = uuid.uuid4().hex


# ─── DATABASE ────────────────────────────────────────────────────────────────
def get_db():
    """Thread-local SQLite connection with WAL enabled."""
    db_dir = os.path.dirname(os.path.abspath(DB_PATH))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            name    TEXT NOT NULL,
            src     TEXT,
            split   TEXT NOT NULL DEFAULT 'train',
            reviewed INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    # Additive migration: datasets created by v1.0 lack the `reviewed` column.
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(images)").fetchall()}
    if "reviewed" not in cols:
        conn.execute("ALTER TABLE images ADD COLUMN reviewed INTEGER NOT NULL DEFAULT 0")
        print("[DB] Added 'reviewed' column to images table")

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

    _load_classes()
    _load_settings()


def _read_yaml_names(yaml_path):
    """Return the `names` list from a YOLO data.yaml, or None."""
    if not os.path.exists(yaml_path):
        return None
    try:
        with open(yaml_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    names = None
    try:
        import yaml  # PyYAML ships with ultralytics
        doc = yaml.safe_load(text)
        if isinstance(doc, dict):
            names = doc.get("names")
    except Exception:
        m = re.search(r"^names:\s*(\[.*\])\s*$", text, re.MULTILINE)
        if m:
            for parser in (json.loads, ast.literal_eval):
                try:
                    names = parser(m.group(1))
                    break
                except Exception:
                    continue
    if isinstance(names, dict):
        try:
            names = [names[k] for k in sorted(names, key=lambda k: int(k))]
        except Exception:
            names = list(names.values())
    if isinstance(names, list) and all(isinstance(n, (str, int, float)) for n in names):
        return [str(n) for n in names]
    return None


def _load_classes():
    """Class load order on dataset open: DB meta -> data.yaml -> model mapping -> [].

    The model fallback (dataset classes = mapped targets of the loaded model) is
    applied by establish_model_mapping(), which set_active_dataset() calls.
    """
    global _class_names
    saved = db_get_meta("classes")
    if saved:
        try:
            loaded = json.loads(saved)
            if isinstance(loaded, list) and loaded:
                _class_names = [str(c) for c in loaded]
                return
        except Exception:
            pass
    names = _read_yaml_names(os.path.join(OUTPUT_DIR, "data.yaml"))
    if names:
        _class_names = names
        return
    _class_names = []


def _load_settings():
    """Detection settings for the active dataset (meta), falling back to config defaults."""
    global _conf, _iou, _auto_annotate
    conf, iou, auto = CONF_THRESH, DEFAULT_IOU, True
    try:
        v = db_get_meta("conf")
        if v is not None and 0.01 <= float(v) <= 0.99:
            conf = float(v)
    except (TypeError, ValueError):
        pass
    try:
        v = db_get_meta("iou")
        if v is not None and 0.1 <= float(v) <= 0.95:
            iou = float(v)
    except (TypeError, ValueError):
        pass
    v = db_get_meta("auto_annotate")
    if v is not None:
        auto = str(v).strip().lower() in ("1", "true", "yes", "on")
    _conf, _iou, _auto_annotate = conf, iou, auto


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
    rows = conn.execute("SELECT name, src, reviewed FROM images WHERE split=? ORDER BY id", (split,)).fetchall()
    conn.close()
    return [{"name": r["name"], "src": r["src"], "reviewed": bool(r["reviewed"])} for r in rows]


def db_all_names():
    """Map image name -> split for every row (first row wins for legacy duplicates)."""
    conn = get_db()
    rows = conn.execute("SELECT name, split FROM images ORDER BY id").fetchall()
    conn.close()
    out = {}
    for r in rows:
        out.setdefault(r["name"], r["split"])
    return out


def db_set_reviewed(name, split, reviewed):
    conn = get_db()
    cur = conn.execute("UPDATE images SET reviewed=? WHERE name=? AND split=?",
                       (1 if reviewed else 0, name, split))
    conn.commit()
    conn.close()
    return cur.rowcount


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


def db_total():
    conn = get_db()
    row = conn.execute("SELECT COUNT(*) AS c FROM images").fetchone()
    conn.close()
    return row["c"]


def dataset_loaded():
    try:
        return db_total() > 0
    except Exception:
        return False


def get_split_ratios():
    raw = db_get_meta("split_ratios")
    if raw:
        try:
            r = json.loads(raw)
            if isinstance(r, list) and len(r) == 3:
                return [float(x) for x in r]
        except Exception:
            pass
    return list(DEFAULT_SPLIT_RATIOS)


def parse_split_ratios(ratios):
    """Validate [train, valid, test] ratios; raises ValueError."""
    if (not isinstance(ratios, list) or len(ratios) != 3
            or not all(isinstance(r, (int, float)) and not isinstance(r, bool) for r in ratios)
            or any(r < 0 or r > 1 for r in ratios)
            or abs(sum(ratios) - 1.0) > 0.001):
        raise ValueError("Invalid ratios. Must be exactly 3 numbers between 0 and 1 that sum to 1.")
    return [float(r) for r in ratios]


# Initialize DB on load (all DB helpers are defined above)
try:
    init_db()
except Exception as _ex:
    print(f"[DB] Initial database setup failed: {_ex}")


# ─── MODEL ───────────────────────────────────────────────────────────────────
def get_model():
    """Load (once) and return the YOLO model.

    Model class names are kept in ``_model_names``; dataset classes are NOT
    overwritten. The model-class -> dataset-class mapping is (re)established here.
    """
    global _model, _model_names
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            from ultralytics import YOLO
            os.environ.setdefault("YOLO_VERBOSE", "False")
            ext = Path(_model_path).suffix.lower()
            print(f"[MODEL] Loading {_model_path}  format={ext}  device={_device}")
            # YOLO wrapper handles .pt, .onnx, .engine, .tflite, .torchscript
            m = YOLO(_model_path)
            _model_names = model_class_names(m)
            _model = m
            print(f"[MODEL] Model classes ({len(_model_names)})")
            try:
                establish_model_mapping()
            except Exception as ex:
                print(f"[MODEL] Class mapping setup failed: {ex}")
            print(f"[MODEL] Ready on device={_device}")
    return _model


def _saved_mapping_for_current_model(warnings):
    """Mapping saved in the active dataset's meta for the current model path, or None."""
    if db_get_meta("model_class_map_model") != _model_path:
        return None
    raw = db_get_meta("model_class_map")
    if not raw:
        return None
    try:
        entries = json.loads(raw)
    except Exception:
        return None
    if not isinstance(entries, list):
        return None
    name_to_idx = {n: i for i, n in enumerate(_model_names)}
    model_label = os.path.basename(_model_path) or _model_path
    mapping = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        name, target, idx = e.get("name"), e.get("target"), e.get("model_idx")
        if not isinstance(target, str) or not target.strip():
            continue
        if isinstance(idx, int) and 0 <= idx < len(_model_names) and _model_names[idx] == name:
            mapping[idx] = target.strip()
        elif name in name_to_idx:
            mapping[name_to_idx[name]] = target.strip()
        else:
            warnings.append(f"Class '{name}' is not in model {model_label} - ignored")
    if entries and not mapping:
        warnings.append("None of the saved classes exist in this model - using its default classes")
        return None
    return mapping or None


def establish_model_mapping():
    """(Re)build the model-class mapping after a model load or dataset switch.

    Priority: mapping saved in the dataset for this model -> current in-memory mapping
    for the same model (if compatible with the dataset classes) -> defaults
    (COCO: car/motorcycle/bus/truck, custom: all classes).
    If the dataset has no classes yet, dataset classes = mapped targets (model order).
    Dataset class indices are never reordered; unknown targets are dropped with a warning.
    """
    global _model_class_map, _model_map_path, _model_map_names, _model_map_warnings
    global _class_names, _empty_map_logged
    warnings = []
    if not _model_names:
        _model_class_map, _model_map_warnings = {}, []
        return
    mapping = _saved_mapping_for_current_model(warnings)
    if mapping is None:
        same_model = (_model_map_path == _model_path and _model_map_names == _model_names
                      and _model_class_map)
        if same_model and (not _class_names or
                           all(t in _class_names for t in _model_class_map.values())):
            mapping = dict(_model_class_map)
        else:
            mapping = default_mapping(_model_names, _class_names)
            if not mapping and _class_names:
                warnings.append("None of this model's default classes match the dataset classes"
                                " - choose classes to auto-label")
    if not _class_names:
        _class_names = mapping_targets_in_order(mapping)
        if _class_names and dataset_loaded():
            db_set_meta("classes", json.dumps(_class_names))
            write_yaml()
    else:
        for idx in sorted(mapping):
            if mapping[idx] not in _class_names:
                warnings.append(f"Class '{mapping[idx]}' is not a dataset class - ignored")
                del mapping[idx]
    _model_class_map = mapping
    _model_map_path = _model_path
    _model_map_names = list(_model_names)
    _model_map_warnings = warnings
    _empty_map_logged = False
    for w in warnings:
        print(f"[MODEL] {w}")


def persist_model_mapping():
    """Save the in-memory mapping into the active dataset's meta."""
    entries = [{"model_idx": i, "name": _model_names[i] if i < len(_model_names) else "",
                "target": t} for i, t in sorted(_model_class_map.items())]
    db_set_meta("model_class_map", json.dumps(entries))
    db_set_meta("model_class_map_model", _model_path or "")


def model_classes_payload():
    names = list(_model_names)
    mapping = [{"model_idx": i, "name": names[i] if i < len(names) else str(i), "target": t}
               for i, t in sorted(_model_class_map.items())]
    return {
        "loaded": _model is not None,
        "model_path": _model_path or "",
        "model_classes": names,
        "mapping": mapping,
        "selected": [m["name"] for m in mapping],
        "default": default_model_classes(names) if names else [],
        "kind": model_kind(names),
        "warnings": list(_model_map_warnings),
        "classes": list(_class_names),
        "is_large": len(names) > LARGE_MODEL_CLASSES,
    }


def model_class_names(m):
    names = getattr(m, "names", None) or {}
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names.keys())]
    return [str(n) for n in names]


def resolve_model_path(new_path):
    """Resolve a model file name against the known model folders.

    Unknown names are returned unchanged so Ultralytics can auto-download presets.
    """
    if new_path and not os.path.exists(new_path):
        candidates = [
            os.path.join(MODELS_DIR, new_path),
            os.path.join(BASE_DIR, new_path),
            os.path.join(BASE_DIR, "models", new_path),
            os.path.join(APP_DIR, new_path),
            os.path.join(APP_DIR, "models", new_path),
            os.path.join(os.path.dirname(APP_DIR), new_path),
            os.path.join(os.path.dirname(APP_DIR), "models", new_path),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                return cand
    return new_path


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
    skip = {'templates', '.vscode', 'annotated_dataset', 'vedio', '__pycache__', 'Data_annotrator_tool', 'runs', 'venv', '.venv', '.git', '.github', 'dist', 'build', 'node_modules', 'models', 'ocr_models', 'static', 'scripts', 'tests'}
    result = []
    scanned_paths = set()

    def _scan_base(b_dir):
        if not b_dir or not os.path.exists(b_dir):
            return
        try:
            with os.scandir(b_dir) as it:
                for entry in it:
                    if entry.name in skip or entry.name.startswith('.'):
                        continue
                    if entry.is_dir() and entry.path not in scanned_paths:
                        scanned_paths.add(entry.path)
                        count = 0
                        try:
                            with os.scandir(entry.path) as sub_it:
                                count = sum(1 for sub in sub_it if sub.is_file() and Path(sub.name).suffix.lower() in IMG_EXTS)
                        except Exception:
                            pass
                        if count > 0:
                            result.append({"name": entry.name, "path": entry.path, "count": count})
        except Exception:
            pass

    _scan_base(BASE_DIR)
    if os.path.abspath(BASE_DIR) != os.path.abspath(APP_DIR):
        _scan_base(APP_DIR)

    result.sort(key=lambda x: x['name'].lower())
    return result


def list_images(src_dir):
    try:
        with os.scandir(src_dir) as it:
            return sorted(
                entry.path for entry in it
                if entry.is_file() and Path(entry.name).suffix.lower() in IMG_EXTS
            )
    except Exception:
        return []


def label_path(split, name):
    validate_split(split)
    validate_name(name)
    return secure_path(OUTPUT_DIR, split, "labels", os.path.splitext(name)[0] + ".txt")


def image_path(split, name):
    validate_split(split)
    validate_name(name)
    return secure_path(OUTPUT_DIR, split, "images", name)


def label_stems(split):
    """Set of label-file stems present for a split (one listdir instead of N stats)."""
    validate_split(split)
    lbl_dir = os.path.join(OUTPUT_DIR, split, "labels")
    try:
        with os.scandir(lbl_dir) as it:
            return {e.name[:-4] for e in it if e.name.endswith(".txt") and e.is_file()}
    except OSError:
        return set()


def is_annotated(name, stems):
    return os.path.splitext(name)[0] in stems


def split_counts(split):
    entries = db_get_entries(split)
    stems = label_stems(split)
    annotated = reviewed = 0
    for e in entries:
        if is_annotated(e["name"], stems):
            annotated += 1
            if e["reviewed"]:
                reviewed += 1
    return {"total": len(entries), "annotated": annotated, "reviewed": reviewed}


def parse_label_line(line):
    """Parse one YOLO label line.

    Returns None for blank lines, a dict for valid lines and raises ValueError for
    malformed ones. Bbox: ``{"cls","x","y","w","h"}``; polygon: ``{"cls","pts":[{x,y}...]}``.
    """
    p = line.strip().split()
    if not p:
        return None
    cls_f = float(p[0])
    if cls_f < 0 or cls_f != int(cls_f):
        raise ValueError("bad class id")
    cls = int(cls_f)
    vals = [float(v) for v in p[1:]]
    if any(v != v for v in vals):  # NaN
        raise ValueError("NaN coordinate")
    if len(vals) == 4:
        x, y, w, h = vals
        return {"cls": cls, "x": x, "y": y, "w": w, "h": h}
    if len(vals) >= 6 and len(vals) % 2 == 0:
        return {"cls": cls, "pts": [{"x": vals[j], "y": vals[j + 1]} for j in range(0, len(vals), 2)]}
    raise ValueError("wrong number of values")


def read_label_file(path):
    """Return (shapes, malformed_line_count) for a label file."""
    shapes, bad = [], 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                s = parse_label_line(line)
            except (ValueError, IndexError):
                bad += 1
                continue
            if s is not None:
                shapes.append(s)
    return shapes, bad


def _clamp01(v):
    return max(0.0, min(1.0, float(v)))


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def sanitize_shapes(boxes):
    """Validate incoming shapes from /api/save. Returns (lines, skipped_count)."""
    lines, skipped = [], 0
    if not isinstance(boxes, list):
        raise ValueError("boxes must be a list")
    for b in boxes:
        try:
            if not isinstance(b, dict):
                raise ValueError
            cls = b.get("cls")
            if isinstance(cls, str) and cls.strip().isdigit():
                cls = int(cls.strip())
            if not _is_num(cls) or cls < 0 or int(cls) != cls:
                raise ValueError
            cls = int(cls)
            if "pts" in b and b["pts"] is not None:
                pts = b["pts"]
                if not isinstance(pts, list):
                    raise ValueError
                coords = []
                for pt in pts:
                    if not isinstance(pt, dict) or not _is_num(pt.get("x")) or not _is_num(pt.get("y")):
                        raise ValueError
                    coords.append((_clamp01(pt["x"]), _clamp01(pt["y"])))
                if len(coords) < 3:
                    raise ValueError
                lines.append(f"{cls} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in coords))
            else:
                vals = [b.get(k) for k in ("x", "y", "w", "h")]
                if not all(_is_num(v) for v in vals):
                    raise ValueError
                x, y, w, h = (float(v) for v in vals)
                # clamp the box edges to the image, then re-derive centre/size
                x1, y1 = _clamp01(x - w / 2), _clamp01(y - h / 2)
                x2, y2 = _clamp01(x + w / 2), _clamp01(y + h / 2)
                w, h = x2 - x1, y2 - y1
                if w <= 1e-6 or h <= 1e-6:
                    raise ValueError
                lines.append(f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {w:.6f} {h:.6f}")
        except (ValueError, TypeError, OverflowError):
            skipped += 1
    return lines, skipped


def atomic_write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{uuid.uuid4().hex[:8]}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


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


NO_MODEL_MSG = "No model loaded — load a model to auto-label"


def model_available():
    """True if auto-labelling may use a model without an implicit network download."""
    return _model is not None or _model_explicit or (bool(_model_path) and os.path.exists(_model_path))


def annotate_one(img_src_path, lbl_dst_path):
    """Run full-res YOLO on one image -> write strict YOLO label file (atomically).

    Only mapped model classes are detected (``classes=``) and each detection is written
    with the dataset index of its mapped target class. Returns False (and writes
    nothing) when no model class is selected for auto-labelling.
    """
    global _empty_map_logged
    dataset_type = db_get_meta("dataset_type", "bbox")
    with _model_lock:
        m = get_model()
        mapping = dict(_model_class_map)
        if not mapping:
            if not _empty_map_logged:
                print("[MODEL] No classes selected for auto-labelling - skipping inference")
                _empty_map_logged = True
            return False
        results = m(img_src_path, conf=_conf, iou=_iou, classes=sorted(mapping),
                    verbose=False, device=_device)

    res = results[0]
    boxes = res.boxes
    dets = []
    if boxes is not None and len(boxes.cls) > 0:
        masks = getattr(res, "masks", None)
        segs = masks.xyn if (dataset_type == "polygon" and masks is not None) else None
        for i in range(len(boxes.cls)):
            seg = segs[i] if segs is not None and i < len(segs) else None
            dets.append((int(boxes.cls[i].item()), boxes.xyxyn[i].tolist(), seg))
    # Resolve target names -> dataset ids under the labels lock, so a concurrent
    # class merge/delete (which shifts ids) can never interleave with this write.
    with _labels_write_lock:
        lines = build_label_lines(dets, mapping, list(_class_names), dataset_type)
        atomic_write_text(lbl_dst_path, "".join(l + "\n" for l in lines))
    return True


@app.route("/api/dataset_info")
def api_dataset_info():
    path = request.args.get("path", "")
    if not os.path.exists(path):
        return jsonify({"valid": False})
    
    db_file = os.path.join(path, "state.db")
    if not os.path.exists(db_file):
        return jsonify({"valid": False})
    
    conn = None
    try:
        # lock-free snapshot: a normal open of a db held by a frozen program would hang
        # this whole process (sqlite's VFS mutex), see _connect_snapshot()
        conn = _connect_snapshot(db_file)
        conn.row_factory = sqlite3.Row
        
        # Get dataset type
        row = conn.execute("SELECT value FROM meta WHERE key='dataset_type'").fetchone()
        ds_type = row["value"] if row else "bbox"
        
        # Get image count
        count = conn.execute("SELECT COUNT(*) as c FROM images").fetchone()["c"]
        return jsonify({"valid": True, "type": ds_type, "count": count})
    except Exception:
        return jsonify({"valid": False})
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

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
    seen = set()
    for search_dir in [BASE_DIR, APP_DIR]:
        if not search_dir or not os.path.exists(search_dir):
            continue
        try:
            for d in os.listdir(search_dir):
                full_path = os.path.abspath(os.path.join(search_dir, d))
                if full_path not in seen and os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, "state.db")):
                    seen.add(full_path)
                    datasets.append({"name": d, "path": full_path})
        except Exception:
            pass
    datasets.sort(key=lambda x: x["name"].lower())
    if request.args.get("stats") in ("1", "true"):
        refresh = request.args.get("refresh") in ("1", "true")
        active = os.path.normcase(os.path.abspath(OUTPUT_DIR))
        for ds in datasets:
            ds["active"] = os.path.normcase(ds["path"]) == active
        # Summaries (cached by file signature) and lock probes (each waiting on its own
        # child process) run concurrently, so the total is ~max(task), not the sum.
        tasks = []
        for ds in datasets:
            tasks.append((ds, "summary", dataset_summary_cached, (ds["path"], ds["active"], refresh)))
            if not ds["active"]:
                tasks.append((ds, "locked", db_locked_cached, (os.path.join(ds["path"], "state.db"), 2.0, 30.0, refresh)))
        if len(tasks) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(tasks))) as pool:
                results = list(pool.map(lambda t: t[2](*t[3]), tasks))
        else:
            results = [t[2](*t[3]) for t in tasks]
        for ds in datasets:
            ds["locked"] = False
        for (ds, kind, _fn, _arg), res in zip(tasks, results):
            if kind == "summary":
                ds.update(res)
            else:
                ds["locked"] = bool(res)
    return jsonify(datasets)


LOCKED_DB_MSG = ("This dataset is locked by another program - usually an old copy of Data Annotation "
                 "Studio that froze. Close it in Task Manager (python.exe) and try again.")


# Lock probing runs in a SEPARATE PROCESS: a sqlite open stuck on another program's
# Windows file lock also holds sqlite's process-wide VFS mutex, which would freeze every
# other database call in this app. A child process can hang alone without harm.
_LOCK_PROBE_CODE = (
    "import sqlite3,sys;"
    "from pathlib import Path;"
    "u=Path(sys.argv[1]).resolve().as_uri();"
    "c=sqlite3.connect(u+'?mode=ro',uri=True,timeout=1.0);"
    "c.execute('SELECT 1 FROM sqlite_master LIMIT 1').fetchone()"
)
_lock_probe = {}        # normcased db path -> (checked_at, responded)
_lock_probe_procs = {}  # normcased db path -> probe Popen still waiting on a lock
_lock_probe_keys = {}   # normcased db path -> RLock serialising probes of that file
_lock_probe_guard = threading.Lock()

# Off switch for the child-process probe (env ANNOTATOR_LOCK_PROBE=0). The test suite turns
# it off because every probe spawns an interpreter; db_responds() then reports "responds".
LOCK_PROBE_ENABLED = os.environ.get("ANNOTATOR_LOCK_PROBE", "1").strip().lower() not in (
    "0", "false", "no", "off")


def _probe_key_lock(key):
    with _lock_probe_guard:
        lk = _lock_probe_keys.get(key)
        if lk is None:
            lk = _lock_probe_keys[key] = threading.RLock()
        return lk


def _kill_probe(proc):
    try:
        proc.kill()   # a probe stuck in the kernel may survive this; it is tracked below
    except Exception:
        pass


def db_responds(db_file, seconds=3.0):
    """True unless another program holds a lock that makes sqlite wait forever.

    Never blocks this process for much longer than `seconds`, and never starts a second
    probe for a file whose earlier probe is still stuck. Thread-safe: concurrent callers
    for the same file share one probe.
    """
    if not LOCK_PROBE_ENABLED or getattr(sys, "frozen", False):
        return True   # disabled, or no separate interpreter available in the frozen build
    key = os.path.normcase(os.path.abspath(db_file))
    with _probe_key_lock(key):
        prev = _lock_probe_procs.get(key)
        if prev is not None:
            if prev.poll() is None:
                return False          # still stuck on the same lock
            _lock_probe_procs.pop(key, None)
        try:
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.Popen([sys.executable, "-I", "-c", _LOCK_PROBE_CODE, db_file],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, creationflags=flags)
        except Exception:
            return True
        try:
            proc.wait(timeout=seconds)
            return True               # finished (even with an sqlite error) -> not hung
        except subprocess.TimeoutExpired:
            _kill_probe(proc)
            _lock_probe_procs[key] = proc
            return False


def db_locked_cached(db_file, seconds=2.0, ttl=30.0, refresh=False):
    """Cached lock check for the home list (results kept `ttl` seconds)."""
    if not LOCK_PROBE_ENABLED:
        return False
    key = os.path.normcase(os.path.abspath(db_file))
    with _probe_key_lock(key):   # a second caller waits for the running probe, then hits the cache
        if not refresh:
            hit = _lock_probe.get(key)
            if hit and time.time() - hit[0] < ttl:
                return not hit[1]
        ok = db_responds(db_file, seconds)
        _lock_probe[key] = (time.time(), ok)
        return not ok


def _connect_snapshot(db_file):
    """Lock-free read-only snapshot of a state.db (never blocks, never writes).

    immutable=1 skips all file locking, so a dataset held open by another program can
    still be summarised on the home screen. Counts may lag a write that is still in an
    un-checkpointed WAL file, which is fine for a summary.
    """
    uri = Path(db_file).resolve().as_uri()
    return sqlite3.connect(f"{uri}?mode=ro&immutable=1", uri=True, timeout=1.0)


def _connect_readonly(db_file):
    """Open a state.db strictly read-only (never creates/migrates anything)."""
    uri = Path(db_file).resolve().as_uri()
    try:
        conn = sqlite3.connect(f"{uri}?mode=ro", uri=True, timeout=2.0)
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
        return conn
    except sqlite3.Error:
        # e.g. WAL database whose -shm cannot be created: read the file as immutable
        return sqlite3.connect(f"{uri}?mode=ro&immutable=1", uri=True, timeout=2.0)


def dataset_summary(path, live=False):
    """Read-only stats for the dataset home list (tolerates v1.0 DBs without `reviewed`).

    live=True (only for the ACTIVE dataset, which this process already opens normally)
    reads through sqlite's WAL; otherwise a lock-free immutable snapshot is used, which
    ignores changes still sitting in an un-checkpointed -wal file.
    """
    out = {"type": "bbox", "total": 0, "reviewed": 0, "annotated": 0, "classes": 0, "modified": 0}
    db_file = os.path.join(path, "state.db")
    try:
        out["modified"] = int(max(os.path.getmtime(p) for p in (db_file, db_file + "-wal")
                                  if os.path.exists(p)))
    except (OSError, ValueError):
        pass
    try:
        conn = _connect_readonly(db_file) if live else _connect_snapshot(db_file)
    except sqlite3.Error:
        out["error"] = "unreadable state.db"
        return out
    try:
        conn.row_factory = sqlite3.Row
        meta = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM meta").fetchall()}
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(images)").fetchall()}
        rev_col = "reviewed" if "reviewed" in cols else "0"
        rows = conn.execute(f"SELECT name, split, {rev_col} AS reviewed FROM images").fetchall()
    except sqlite3.Error as ex:
        out["error"] = f"unreadable state.db: {ex}"
        return out
    finally:
        conn.close()
    out["type"] = meta.get("dataset_type") or "bbox"
    names = None
    try:
        names = json.loads(meta["classes"]) if meta.get("classes") else None
    except Exception:
        names = None
    if not isinstance(names, list) or not names:
        names = _read_yaml_names(os.path.join(path, "data.yaml")) or []
    out["classes"] = len(names)
    stems = {}
    for split in SPLITS:
        try:
            with os.scandir(os.path.join(path, split, "labels")) as it:
                stems[split] = {e.name[:-4] for e in it if e.name.endswith(".txt")}
        except OSError:
            stems[split] = set()
    for r in rows:
        out["total"] += 1
        if os.path.splitext(r["name"])[0] in stems.get(r["split"], ()):
            out["annotated"] += 1
            if r["reviewed"]:
                out["reviewed"] += 1
    return out


_summary_cache = {}   # normcased dataset path -> (signature, summary)
_summary_cache_lock = threading.Lock()


def _stat_sig(path):
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def dataset_signature(path):
    """Cheap change detector for a dataset summary: state.db / -wal / data.yaml (mtime+size)
    and the mtimes of the three labels dirs (a dir's mtime changes when files are added,
    removed or renamed in it — the summary only counts label stems, not their contents)."""
    db_file = os.path.join(path, "state.db")
    return (_stat_sig(db_file), _stat_sig(db_file + "-wal"), _stat_sig(os.path.join(path, "data.yaml")),
            tuple(_stat_sig(os.path.join(path, split, "labels")) for split in SPLITS))


def dataset_summary_cached(path, live=False, refresh=False):
    """dataset_summary(), recomputed only when dataset_signature() changes."""
    key = (os.path.normcase(os.path.abspath(path)), bool(live))
    sig = dataset_signature(path)   # taken BEFORE reading, so a concurrent change re-triggers
    if not refresh:
        with _summary_cache_lock:
            hit = _summary_cache.get(key)
        if hit and hit[0] == sig:
            return dict(hit[1])
    out = dataset_summary(path, live)
    if "error" not in out:
        with _summary_cache_lock:
            _summary_cache[key] = (sig, dict(out))
    return out

def write_yaml():
    """Write data.yaml for the active dataset (names as a JSON list = valid YAML)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    text = ("train: ../train/images\nval: ../valid/images\ntest: ../test/images\n\n"
            f"nc: {len(_class_names)}\n"
            f"names: {json.dumps(list(_class_names), ensure_ascii=False)}\n")
    atomic_write_text(os.path.join(OUTPUT_DIR, "data.yaml"), text)


# ─── THUMBNAIL CACHE ─────────────────────────────────────────────────────────
_thumb_cache = collections.OrderedDict()   # (abs_path, mtime_ns) -> jpeg bytes
_thumb_cache_lock = threading.Lock()
_THUMB_CACHE_MAX = 400   # cap at 400 entries (~170 MB worst case)


def clear_thumb_cache():
    with _thumb_cache_lock:
        _thumb_cache.clear()


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
        try:
            if job_session != _current_session_id or not _auto_annotate or not model_available():
                continue
            lbl = label_path(split, name)
            dst = image_path(split, name)
            if not os.path.exists(lbl) and os.path.exists(dst):
                with get_img_lock(f"{split}/{name}"):
                    if job_session != _current_session_id or not _auto_annotate:
                        continue
                    if not os.path.exists(lbl):
                        os.makedirs(os.path.dirname(lbl), exist_ok=True)
                        annotate_one(dst, lbl)
        except Exception as ex:
            print(f"[PRE-ANNO] Error on {name}: {ex}")
        finally:
            _preanno_queue.task_done()


_preanno_thread = threading.Thread(target=_preanno_worker, daemon=True)
_preanno_thread.start()


def enqueue_preanno(split, names):
    """Add a list of image names to the pre-annotation queue (no-op when auto-annotate is off)."""
    if not _auto_annotate or not model_available():
        return
    session_id = _current_session_id
    for n in names:
        _preanno_queue.put((session_id, split, n))


# ─── API: ERRORS ─────────────────────────────────────────────────────────────
@app.errorhandler(ValueError)
def handle_value_error(ex):
    return jsonify({"error": str(ex) or "Invalid request"}), 400


# ─── API: DEVICE INFO ────────────────────────────────────────────────────────
@app.route("/api/device_info")
def api_device_info():
    return jsonify(get_device_info())


# ─── API: SETTINGS ───────────────────────────────────────────────────────────
def _settings_payload():
    return {
        "auto_annotate": bool(_auto_annotate),
        "conf": float(_conf),
        "iou": float(_iou),
        "model_path": _model_path or "",
        "model_loaded": _model is not None,
        "model_available": model_available(),
        "device": _device,
        "dataset_name": _current_dataset_name,
    }


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    """Detection settings, stored in the active dataset's state.db meta."""
    global _auto_annotate, _conf, _iou
    if request.method == "POST":
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON format"}), 400
        if "auto_annotate" in data and not isinstance(data["auto_annotate"], bool):
            return jsonify({"error": "auto_annotate must be a boolean"}), 400
        if "conf" in data and (not _is_num(data["conf"]) or not (0.01 <= float(data["conf"]) <= 0.99)):
            return jsonify({"error": "conf must be a number between 0.01 and 0.99"}), 400
        if "iou" in data and (not _is_num(data["iou"]) or not (0.1 <= float(data["iou"]) <= 0.95)):
            return jsonify({"error": "iou must be a number between 0.1 and 0.95"}), 400
        if "auto_annotate" in data:
            _auto_annotate = data["auto_annotate"]
            db_set_meta("auto_annotate", "true" if _auto_annotate else "false")
        if "conf" in data:
            _conf = round(float(data["conf"]), 4)
            db_set_meta("conf", str(_conf))
        if "iou" in data:
            _iou = round(float(data["iou"]), 4)
            db_set_meta("iou", str(_iou))
    return jsonify(_settings_payload())


# ─── API: SET MODEL ──────────────────────────────────────────────────────────
@app.route("/api/set_model", methods=["POST"])
def api_set_model():
    global _model_path, _device, _model_explicit
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON format"}), 400

    new_path = data.get("model_path", "")
    if not isinstance(new_path, str):
        return jsonify({"error": "model_path must be a string"}), 400

    explicit_device = data.get("device", "auto")
    if not isinstance(explicit_device, str):
        return jsonify({"error": "device must be a string"}), 400

    # If it doesn't exist locally it is left as-is so Ultralytics can auto-download it.
    new_path = resolve_model_path(new_path.strip())

    ext = Path(new_path).suffix.lower() if new_path else ""
    if new_path and ext not in MODEL_EXTS:
        return jsonify({"error": f"Unsupported format '{ext}'."}), 400

    reset_model()

    if new_path:
        _model_path = new_path
    _model_explicit = True

    if explicit_device == "auto" or not explicit_device:
        _device = auto_detect_device()
    else:
        _device = explicit_device

    print(f"[MODEL] Target device assigned: {_device}")

    # Load eagerly to extract model class names and set up the class mapping.
    try:
        get_model()
        if dataset_loaded() and _class_names:
            db_set_meta("classes", json.dumps(_class_names))
            write_yaml()
        dev_info = get_device_info()
        dev_info["selected"] = _device
        payload = model_classes_payload()
        payload.update({"status": "ok", "classes": _class_names, "device": _device,
                        "device_info": dev_info, "model_path": _model_path})
        return jsonify(payload)
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500


# ─── API: MODEL CLASS SELECTION / MAPPING ────────────────────────────────────
@app.route("/api/model_classes", methods=["GET", "POST"])
def api_model_classes():
    """Which model classes are auto-labelled and which dataset class each is saved as."""
    global _model_class_map, _class_names, _model_map_warnings, _model_map_path, _model_map_names
    global _empty_map_logged
    if request.method == "GET":
        if request.args.get("load") in ("1", "true") and _model is None:
            try:
                get_model()
            except Exception as ex:
                return jsonify({"error": f"Could not load model: {ex}"}), 500
        return jsonify(model_classes_payload())

    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("mapping"), list):
        return jsonify({"error": "Expected {\"mapping\": [{\"model_idx\": int, \"target\": str}, ...]}"}), 400
    if not _model_names:
        return jsonify({"error": "Load a model first"}), 400
    if not data["mapping"]:
        return jsonify({"error": "Keep at least one class selected"}), 400
    new_map, order = {}, []
    for e in data["mapping"]:
        if not isinstance(e, dict):
            return jsonify({"error": "Each mapping entry must be an object"}), 400
        idx = e.get("model_idx")
        if not isinstance(idx, int) or isinstance(idx, bool) or not (0 <= idx < len(_model_names)):
            return jsonify({"error": f"Invalid model_idx: {idx!r}"}), 400
        target = e.get("target", _model_names[idx])
        if not isinstance(target, str) or not target.strip():
            return jsonify({"error": f"Target class for '{_model_names[idx]}' must be a non-empty name"}), 400
        target = target.strip()
        new_map[idx] = target
        order.append(target)
    with _model_lock:
        classes = list(_class_names)
        for t in order:                      # append-only: never reindex dataset classes
            if t not in classes:
                classes.append(t)
        _class_names = classes
        _model_class_map = new_map
        _model_map_path = _model_path
        _model_map_names = list(_model_names)
        _model_map_warnings = []
        _empty_map_logged = False
    if dataset_loaded():
        db_set_meta("classes", json.dumps(_class_names))
        persist_model_mapping()
        write_yaml()
    return jsonify(model_classes_payload())


@app.route("/api/classes", methods=["GET", "POST"])
def api_classes():
    if request.method == "POST":
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not isinstance(data.get("classes"), list):
            return jsonify({"error": "Expected {\"classes\": [str, ...]}"}), 400
        cleaned = []
        for c in data["classes"]:
            if not isinstance(c, str):
                return jsonify({"error": "Class names must be strings"}), 400
            c = c.strip()
            if c:
                cleaned.append(c)
        if not cleaned:
            return jsonify({"error": "At least one class name is required"}), 400
        seen, dups = set(), []
        for c in cleaned:
            if c in seen and c not in dups:
                dups.append(c)
            seen.add(c)
        if dups:
            return jsonify({"error": f"Duplicate class names: {', '.join(dups)}"}), 400
        with _labels_write_lock:
            old = list(_class_names)
            rename_only = len(cleaned) == len(old)
            pure_append = cleaned[:len(old)] == old
            if not (rename_only or pure_append):
                return jsonify({"error": "Use merge/delete to remove classes"}), 400
            renames = {o: n for o, n in zip(old, cleaned) if o != n} if rename_only else {}
            _apply_class_list(cleaned, renames=renames)
    return jsonify({"classes": _class_names})


# ─── API: CLASS MANAGEMENT (usage / rename / add / merge / delete / backups) ─
BACKUP_KEEP = 10
_BACKUP_NAME_RE = re.compile(r"^labels_[0-9]{8}-[0-9]{6}(-[0-9]+)?\.zip$")


def _backup_sort_key(name):
    """Chronological key: (timestamp, same-second counter)."""
    m = _BACKUP_NAME_RE.match(name)
    stamp = name[len("labels_"):len("labels_") + 15]
    return (stamp, int(m.group(1)[1:]) if m and m.group(1) else 0)
CLASS_JOB_KINDS = {"ingest", "relabel_all"}


def class_ops_blocked():
    """Error message if a job that reads/writes labels is running, else None."""
    if _auto_split_running:
        return "Auto-split is running — try again when it finishes"
    for p in list(_job_progress.values()):
        if p.get("_kind") in CLASS_JOB_KINDS and not p.get("_finished_at"):
            return f"A background {p['_kind'].replace('_', ' ')} job is running — try again when it finishes"
    return None


def _iter_label_files():
    for split in SPLITS:
        lbl_dir = os.path.join(OUTPUT_DIR, split, "labels")
        try:
            names = sorted(f for f in os.listdir(lbl_dir) if f.endswith(".txt"))
        except OSError:
            continue
        for fn in names:
            yield split, fn, os.path.join(lbl_dir, fn)


def _line_class_id(line):
    """Class id of a label line, or None for blank / unparseable lines (left untouched)."""
    p = line.split()
    if not p:
        return None
    try:
        v = float(p[0])
    except ValueError:
        return None
    if v < 0 or v != int(v):
        return None
    return int(v)


def _apply_class_list(new_classes, renames=None, removed=None):
    """Store a new dataset class list and make mapping targets follow it.

    renames: {old_name: new_name}; removed: {old_name: new_name_or_None}
    (None = mapping entries pointing at it are dropped).
    """
    global _class_names, _model_class_map
    follow = dict(renames or {})
    follow.update(removed or {})
    if follow and _model_class_map:
        new_map = {}
        for idx, target in _model_class_map.items():
            t = follow.get(target, target)
            if t is not None and t in new_classes:
                new_map[idx] = t
        _model_class_map = new_map
    _class_names = list(new_classes)
    db_set_meta("classes", json.dumps(_class_names))
    saved = db_get_meta("model_class_map")
    if follow and saved:
        # The saved mapping may belong to a model that is not loaded: rewrite it by name.
        try:
            entries = []
            for e in json.loads(saved):
                t = follow.get(e.get("target"), e.get("target"))
                if t is not None and t in _class_names:
                    entries.append(dict(e, target=t))
            db_set_meta("model_class_map", json.dumps(entries))
        except Exception:
            pass
    if dataset_loaded() or os.path.exists(os.path.join(OUTPUT_DIR, "data.yaml")):
        write_yaml()


def backups_dir():
    return os.path.join(OUTPUT_DIR, "backups")


def create_labels_backup():
    """Zip all */labels/*.txt + the class list; keep the newest BACKUP_KEEP. Returns 'backups/<name>'."""
    bdir = backups_dir()
    os.makedirs(bdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # Next counter after any backup from the same second (names freed by pruning are not reused,
    # so the name order stays chronological).
    same = [_backup_sort_key(f)[1] for f in os.listdir(bdir)
            if _BACKUP_NAME_RE.match(f) and _backup_sort_key(f)[0] == stamp]
    name = f"labels_{stamp}.zip" if not same else f"labels_{stamp}-{max(same) + 1}.zip"
    path = os.path.join(bdir, name)
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("classes.json", json.dumps({
            "classes": list(_class_names),
            "model_class_map": db_get_meta("model_class_map"),
            "model_class_map_model": db_get_meta("model_class_map_model"),
            "created": time.time(),
        }))
        for split, fn, full in _iter_label_files():
            zf.write(full, f"{split}/labels/{fn}")
    os.replace(tmp, path)
    existing = sorted((f for f in os.listdir(bdir) if _BACKUP_NAME_RE.match(f)), key=_backup_sort_key)
    for old in existing[:-BACKUP_KEEP]:
        try:
            os.remove(os.path.join(bdir, old))
        except OSError:
            pass
    return f"backups/{name}"


def remap_label_files(id_map):
    """Rewrite class ids in every label file. id_map: {old_id: new_id or None (= delete box)}.

    Ids not in id_map (e.g. unknown ids beyond the class list) and unparseable lines are
    kept verbatim. Returns (changed_files, changed_boxes).
    """
    changed_files = changed_boxes = 0
    for _split, _fn, full in _iter_label_files():
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
        except OSError:
            continue
        out, changed = [], 0
        for line in lines:
            cid = _line_class_id(line)
            if cid is None or cid not in id_map or id_map[cid] == cid:
                if line.strip():
                    out.append(line.strip())
                continue
            new_id = id_map[cid]
            changed += 1
            if new_id is None:
                continue
            out.append(" ".join([str(new_id)] + line.split()[1:]))
        if changed:
            atomic_write_text(full, "".join(l + "\n" for l in out))
            changed_files += 1
            changed_boxes += changed
    return changed_files, changed_boxes


def _shift_map(n_classes, removed_ids, redirect=None):
    """old id -> new id after removing `removed_ids`; removed ids go to redirect (old id) or None."""
    removed = set(removed_ids)
    new_index, k = {}, 0
    for i in range(n_classes):
        if i in removed:
            continue
        new_index[i] = k
        k += 1
    id_map = dict(new_index)
    for r in removed:
        id_map[r] = new_index[redirect] if redirect is not None else None
    return id_map


def _class_change(removed_ids, redirect=None):
    """Remove classes (boxes deleted, or redirected to class `redirect`) with backup + rewrite."""
    old = list(_class_names)
    id_map = _shift_map(len(old), removed_ids, redirect)
    backup = create_labels_backup()
    changed_files, changed_boxes = remap_label_files(id_map)
    new_classes = [c for i, c in enumerate(old) if i not in set(removed_ids)]
    target_name = old[redirect] if redirect is not None else None
    _apply_class_list(new_classes, removed={old[i]: target_name for i in removed_ids})
    return {"classes": list(_class_names), "changed_files": changed_files,
            "changed_boxes": changed_boxes, "backup": backup}


def _class_id_arg(value, label="id"):
    if not isinstance(value, int) or isinstance(value, bool) or not (0 <= value < len(_class_names)):
        raise ValueError(f"Invalid class {label}: {value!r}")
    return value


@app.route("/api/classes/usage")
def api_classes_usage():
    names = list(_class_names)
    boxes = collections.Counter()
    images = collections.Counter()
    for _split, _fn, full in _iter_label_files():
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                ids = [cid for cid in (_line_class_id(l) for l in f) if cid is not None]
        except OSError:
            continue
        boxes.update(ids)
        images.update(set(ids))
    return jsonify({
        "classes": [{"id": i, "name": n, "boxes": boxes.get(i, 0), "images": images.get(i, 0)}
                    for i, n in enumerate(names)],
        "unknown_ids": [{"id": i, "boxes": boxes[i]} for i in sorted(boxes) if i >= len(names)],
    })


@app.route("/api/classes/rename", methods=["POST"])
def api_classes_rename():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON format"}), 400
    with _labels_write_lock:
        cid = _class_id_arg(data.get("id"))
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return jsonify({"error": "Class name must not be empty"}), 400
        name = name.strip()
        if name in _class_names and _class_names.index(name) != cid:
            return jsonify({"error": f"A class named '{name}' already exists"}), 400
        old = _class_names[cid]
        new_classes = list(_class_names)
        new_classes[cid] = name
        _apply_class_list(new_classes, renames={old: name} if old != name else None)
    return jsonify({"classes": list(_class_names)})


@app.route("/api/classes/add", methods=["POST"])
def api_classes_add():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON format"}), 400
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        return jsonify({"error": "Class name must not be empty"}), 400
    name = name.strip()
    with _labels_write_lock:
        if name in _class_names:
            return jsonify({"error": f"A class named '{name}' already exists"}), 400
        _apply_class_list(list(_class_names) + [name])
    return jsonify({"classes": list(_class_names)})


@app.route("/api/classes/merge", methods=["POST"])
def api_classes_merge():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON format"}), 400
    blocked = class_ops_blocked()
    if blocked:
        return jsonify({"error": blocked}), 409
    with _labels_write_lock:
        sources = data.get("sources")
        if not isinstance(sources, list) or not sources:
            return jsonify({"error": "sources must be a non-empty list of class ids"}), 400
        sources = [_class_id_arg(s, "source") for s in sources]
        if len(set(sources)) != len(sources):
            return jsonify({"error": "Duplicate source ids"}), 400
        target = _class_id_arg(data.get("target"), "target")
        if target in sources:
            return jsonify({"error": "target must not be one of the sources"}), 400
        return jsonify(_class_change(sources, redirect=target))


@app.route("/api/classes/delete", methods=["POST"])
def api_classes_delete():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON format"}), 400
    blocked = class_ops_blocked()
    if blocked:
        return jsonify({"error": blocked}), 409
    with _labels_write_lock:
        cid = _class_id_arg(data.get("id"))
        mode = data.get("boxes", "remove")
        if mode == "remove":
            return jsonify(_class_change([cid]))
        if mode == "merge":
            into = _class_id_arg(data.get("merge_into"), "merge_into")
            if into == cid:
                return jsonify({"error": "merge_into must be a different class"}), 400
            return jsonify(_class_change([cid], redirect=into))
        return jsonify({"error": "boxes must be 'remove' or 'merge'"}), 400


@app.route("/api/classes/backups")
def api_classes_backups():
    bdir = backups_dir()
    out = []
    try:
        names = sorted((f for f in os.listdir(bdir) if _BACKUP_NAME_RE.match(f)),
                       key=_backup_sort_key, reverse=True)
    except OSError:
        names = []
    for name in names:
        full = os.path.join(bdir, name)
        item = {"name": name, "path": f"backups/{name}", "size": os.path.getsize(full),
                "created": int(os.path.getmtime(full)), "classes": None, "files": 0}
        try:
            with zipfile.ZipFile(full) as zf:
                meta = json.loads(zf.read("classes.json"))
                item["classes"] = meta.get("classes")
                item["created"] = int(meta.get("created") or item["created"])
                item["files"] = sum(1 for n in zf.namelist() if n.endswith(".txt"))
        except Exception:
            item["error"] = "unreadable backup"
        out.append(item)
    return jsonify({"backups": out})


@app.route("/api/classes/restore", methods=["POST"])
def api_classes_restore():
    """Restore labels + class list from a backup (the current state is backed up first).

    Label files are put back next to wherever each image currently lives; label files
    that are not in the backup are removed so the set matches the restored class list.
    """
    global _model_class_map
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON format"}), 400
    name = os.path.basename(str(data.get("backup") or ""))
    if not _BACKUP_NAME_RE.match(name):
        return jsonify({"error": "Invalid backup name"}), 400
    path = os.path.join(backups_dir(), name)
    if not os.path.isfile(path):
        return jsonify({"error": "Backup not found"}), 404
    blocked = class_ops_blocked()
    if blocked:
        return jsonify({"error": blocked}), 409
    with _labels_write_lock:
        try:
            zf = zipfile.ZipFile(path)
            meta = json.loads(zf.read("classes.json"))
            classes = meta.get("classes")
            if not isinstance(classes, list) or not all(isinstance(c, str) for c in classes):
                raise ValueError("bad class list")
        except Exception as ex:
            return jsonify({"error": f"Backup is unreadable: {ex}"}), 400
        with zf:
            pre_backup = create_labels_backup()
            # where does each image stem live now?
            stem_split = {}
            for split in SPLITS:
                for e in db_get_entries(split):
                    stem_split.setdefault(os.path.splitext(e["name"])[0], split)
            restored = set()
            for member in zf.namelist():
                parts = member.split("/")
                if len(parts) != 3 or parts[1] != "labels" or not parts[2].endswith(".txt"):
                    continue
                fn = parts[2]
                if fn != os.path.basename(fn) or fn in ("", ".txt"):
                    continue
                split = stem_split.get(fn[:-4], parts[0] if parts[0] in VALID_SPLITS else None)
                if split is None:
                    continue
                text = zf.read(member).decode("utf-8", errors="replace")
                atomic_write_text(os.path.join(OUTPUT_DIR, split, "labels", fn), text)
                restored.add((split, fn))
        removed = 0
        for split, fn, full in list(_iter_label_files()):
            if (split, fn) not in restored:
                try:
                    os.remove(full)
                    removed += 1
                except OSError:
                    pass
        if meta.get("model_class_map") is not None:
            db_set_meta("model_class_map", meta["model_class_map"])
            db_set_meta("model_class_map_model", meta.get("model_class_map_model") or "")
        _model_class_map = {}
        _apply_class_list(classes)
        with _model_lock:
            if _model_names:
                establish_model_mapping()
    return jsonify({"classes": list(_class_names), "restored_files": len(restored),
                    "removed_files": removed, "backup": pre_backup})


# ─── API: CLASS STATISTICS ───────────────────────────────────────────────────
@app.route("/api/class_stats")
def api_class_stats():
    """Count per-class box occurrences across the whole annotated dataset."""
    names = list(_class_names)
    counts = {name: 0 for name in names}
    for split in SPLITS:
        lbl_dir = os.path.join(OUTPUT_DIR, split, "labels")
        if not os.path.isdir(lbl_dir):
            continue
        for fn in os.listdir(lbl_dir):
            if not fn.endswith(".txt"):
                continue
            try:
                shapes, _bad = read_label_file(os.path.join(lbl_dir, fn))
            except OSError:
                # File might have been moved/deleted by split/cleanup operations
                continue
            for s in shapes:
                cls_id = s["cls"]
                name = names[cls_id] if cls_id < len(names) else str(cls_id)
                counts[name] = counts.get(name, 0) + 1
    return jsonify(counts)


# ─── API: SOURCE DISCOVERY ───────────────────────────────────────────────────
@app.route("/api/browse_model_file")
def api_browse_model_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        init_dir = MODELS_DIR if os.path.exists(MODELS_DIR) else BASE_DIR
        file_path = filedialog.askopenfilename(
            parent=root,
            title="Select Custom Model File",
            initialdir=init_dir,
            filetypes=[
                ("All Supported Models (*.pt;*.onnx;*.engine;*.tflite;*.torchscript)", "*.pt;*.onnx;*.engine;*.tflite;*.torchscript"),
                ("PyTorch Weights (*.pt)", "*.pt"),
                ("ONNX Model (*.onnx)", "*.onnx"),
                ("TensorRT Engine (*.engine)", "*.engine"),
                ("All Files (*.*)", "*.*")
            ]
        )
        root.destroy()
        return jsonify({"path": file_path or ""})
    except Exception as e:
        return jsonify({"path": "", "error": str(e)})

@app.route("/api/browse_folder")
def api_browse_folder():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder = filedialog.askdirectory(parent=root, title="Select Directory", initialdir=BASE_DIR if os.path.exists(BASE_DIR) else None)
        root.destroy()
        return jsonify({"path": folder or ""})
    except Exception as e:
        return jsonify({"path": "", "error": str(e)})

@app.route("/api/source_dirs")
def api_source_dirs():
    return jsonify(get_image_dirs())


# ─── API: EXISTING DATASETS ──────────────────────────────────────────────────
@app.route("/api/existing_datasets")
def api_existing_datasets():
    result = []
    seen = set()
    for search_dir in [BASE_DIR, APP_DIR]:
        if not search_dir or not os.path.exists(search_dir):
            continue
        try:
            for name in sorted(os.listdir(search_dir)):
                if name.startswith('.'):
                    continue
                full = os.path.abspath(os.path.join(search_dir, name))
                if full not in seen and os.path.isdir(full) and os.path.exists(os.path.join(full, "state.db")):
                    seen.add(full)
                    result.append({"name": name, "path": full})
        except Exception:
            pass
    result.sort(key=lambda x: x["name"].lower())
    return jsonify(result)


@app.route("/api/load_existing_dataset", methods=["POST"])
def api_load_existing_dataset():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON format"}), 400
    dataset_dir = data.get("dataset_dir", "")
    if not isinstance(dataset_dir, str) or not dataset_dir or not os.path.isdir(dataset_dir):
        return jsonify({"error": "invalid directory"}), 400

    if not os.path.exists(os.path.join(dataset_dir, "state.db")):
        return jsonify({"error": "state.db not found"}), 400
    if not db_responds(os.path.join(dataset_dir, "state.db")):
        return jsonify({"error": LOCKED_DB_MSG, "locked": True}), 409

    set_active_dataset(os.path.abspath(dataset_dir), is_existing=True)
    return jsonify({"status": "ok"})


# ─── API: DATASET STATE ──────────────────────────────────────────────────────
@app.route("/api/stats")
def api_stats():
    return jsonify({split: split_counts(split) for split in SPLITS})


@app.route("/api/state")
def api_state():
    info = {
        "dataset_dir": OUTPUT_DIR,
        "dataset_name": _current_dataset_name,
        "base_dir": os.path.abspath(BASE_DIR),
        "split_ratios": get_split_ratios(),
        "classes": list(_class_names),
    }
    if db_total() == 0:
        return jsonify(dict(info, ready=False))
    counts = {split: split_counts(split) for split in SPLITS}
    return jsonify(dict(info, ready=True, counts=counts, source=db_get_meta("source", ""),
                        dataset_type=db_get_meta("dataset_type", "bbox")))


# ─── JOBS (ingest / relabel progress) ────────────────────────────────────────
def _prune_jobs():
    now = time.time()
    for jid in [j for j, p in list(_job_progress.items())
                if p.get("_finished_at") and now - p["_finished_at"] > _JOB_TTL_SECONDS]:
        _job_progress.pop(jid, None)


def new_job(**extra):
    _prune_jobs()
    job_id = uuid.uuid4().hex
    _job_progress[job_id] = dict({"total": 0, "done": 0, "status": "starting"}, **extra)
    return job_id


def finish_job(job_id, status="done", **extra):
    job = _job_progress.setdefault(job_id, {"total": 0, "done": 0})
    job.update(extra)
    job["status"] = status
    job["_finished_at"] = time.time()


def _public_job(prog):
    return {k: v for k, v in prog.items() if not k.startswith("_")}


# ─── API: INGEST — hardlink (or copy) images into a split ────────────────────
def _link_images(src, target_split, job_id):
    """Link new images from ``src`` into ``target_split``; merge with existing rows.

    Images whose name already exists in ANY split are skipped (no duplicates).
    Returns (new_entries, source_names_with_split, skipped_count).
    """
    imgs = list_images(src)
    _job_progress[job_id].update({"total": len(imgs), "status": "linking"})
    existing = db_all_names()
    entries, in_source, skipped = [], [], 0
    for i, img_path in enumerate(imgs):
        name = os.path.basename(img_path)
        if name in existing:
            skipped += 1
            in_source.append((existing[name], name))
        else:
            dst = image_path(target_split, name)
            if not os.path.exists(dst):
                try:
                    os.link(img_path, dst)          # hardlink (no copy!)
                except OSError:
                    shutil.copy2(img_path, dst)     # fallback for cross-device
            entries.append({"name": name, "src": img_path})
            existing[name] = target_split
            in_source.append((target_split, name))
        _job_progress[job_id]["done"] = i + 1
    if entries:
        db_insert_entries(entries, target_split)
    return entries, in_source, skipped


def _enqueue_unlabelled(pairs):
    by_split = collections.defaultdict(list)
    stems = {s: label_stems(s) for s in SPLITS}
    for split, name in pairs:
        if not is_annotated(name, stems[split]):
            by_split[split].append(name)
    for split, names in by_split.items():
        enqueue_preanno(split, names)


@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON format"}), 400

    src = data.get("source_dir", "")
    if not isinstance(src, str):
        return jsonify({"error": "source_dir must be a string"}), 400

    dataset_type = data.get("dataset_type", "bbox")
    if dataset_type not in ["bbox", "polygon"]:
        return jsonify({"error": "dataset_type must be bbox or polygon"}), 400

    if not src or not os.path.isdir(src):
        return jsonify({"error": "invalid source dir"}), 400

    ratios = None
    if data.get("split_ratios") is not None:
        try:
            ratios = parse_split_ratios(data.get("split_ratios"))
        except ValueError as ex:
            return jsonify({"error": str(ex)}), 400

    blocked = class_ops_blocked()
    if blocked:
        return jsonify({"error": blocked}), 409
    set_active_dataset(src)
    job_id = new_job(_kind="ingest")

    for split in SPLITS:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

    session = _current_session_id

    def run():
        try:
            entries, in_source, skipped = _link_images(src, "train", job_id)
            if session != _current_session_id:
                raise RuntimeError("dataset was switched during ingest")
            db_set_meta("source", src)
            db_set_meta("dataset_type", dataset_type)
            if ratios is not None:
                db_set_meta("split_ratios", json.dumps(ratios))
            if _class_names and not db_get_meta("classes"):
                db_set_meta("classes", json.dumps(_class_names))
            if _model_class_map and not db_get_meta("model_class_map"):
                persist_model_mapping()
            write_yaml()
            clear_thumb_cache()
            # Queue background pre-annotation for every unlabelled image from this source
            _enqueue_unlabelled(in_source)
            finish_job(job_id, "done", added=len(entries), skipped=skipped)
        except Exception as ex:
            print(f"[INGEST] Failed: {ex}")
            finish_job(job_id, f"error: {ex}")

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/ingest_append", methods=["POST"])
def api_ingest_append():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON format"}), 400

    src = data.get("source_dir", "")
    if not isinstance(src, str):
        return jsonify({"error": "source_dir must be a string"}), 400

    target_split = data.get("split", "train")
    if not isinstance(target_split, str):
        return jsonify({"error": "split must be a string"}), 400

    try:
        validate_split(target_split)
    except ValueError:
        return jsonify({"error": "Invalid split"}), 400

    if not src or not os.path.isdir(src):
        return jsonify({"error": "invalid source dir"}), 400

    job_id = new_job(_kind="ingest")

    os.makedirs(os.path.join(OUTPUT_DIR, target_split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, target_split, "labels"), exist_ok=True)

    def run():
        try:
            entries, _in_source, skipped = _link_images(src, target_split, job_id)
            if entries:
                write_yaml()
                clear_thumb_cache()
                enqueue_preanno(target_split, [e["name"] for e in entries])
            finish_job(job_id, "done", added=len(entries), skipped=skipped)
        except Exception as ex:
            print(f"[INGEST] Append failed: {ex}")
            finish_job(job_id, f"error: {ex}")

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id})


# ─── API: AUTO-SPLIT (annotated images only) ──────────────────────────────────
@app.route("/api/auto_split", methods=["POST"])
def api_auto_split():
    global _auto_split_running
    with _labels_write_lock:
        _auto_split_running = True
        try:
            return _auto_split_impl()
        finally:
            _auto_split_running = False


def _auto_split_impl():
    data = request.get_json(silent=True) or {}
    try:
        ratios = parse_split_ratios(data.get("ratios", list(DEFAULT_SPLIT_RATIOS)))
    except ValueError as ex:
        return jsonify({"error": str(ex)}), 400

    all_entries = db_all_entries()
    if not all_entries:
        return jsonify({"error": "no dataset loaded"}), 400

    db_set_meta("split_ratios", json.dumps(ratios))

    # Legacy datasets may contain the same name in several splits; never move those.
    name_counts = collections.Counter(e["name"] for e in all_entries)
    stems = {s: label_stems(s) for s in SPLITS}
    annotated, unannotated, dup_skipped = [], [], 0
    for e in all_entries:
        if name_counts[e["name"]] > 1:
            dup_skipped += 1
        elif is_annotated(e["name"], stems.get(e["cur_split"], set())):
            annotated.append(e)
        else:
            unannotated.append(e)

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
            with get_img_lock(f"{cur}/{e['name']}"):
                for kind in ["images", "labels"]:
                    fn = e["name"] if kind == "images" else os.path.splitext(e["name"])[0] + ".txt"
                    src_p = os.path.join(OUTPUT_DIR, cur,       kind, fn)
                    dst_p = os.path.join(OUTPUT_DIR, new_split, kind, fn)
                    if os.path.exists(src_p):
                        os.makedirs(os.path.dirname(dst_p), exist_ok=True)
                        shutil.move(src_p, dst_p)
                db_update_split(e["name"], cur, new_split)

    clear_thumb_cache()
    counts = {k: len(v) for k, v in splits.items()}
    counts["unannotated_skipped"] = len(unannotated)
    if dup_skipped:
        counts["duplicates_skipped"] = dup_skipped
    return jsonify({"status": "ok", "counts": counts})


# ─── API: SSE PROGRESS ───────────────────────────────────────────────────────
@app.route("/api/progress/<job_id>")
def api_progress(job_id):
    def generate():
        while True:
            prog = _job_progress.get(job_id)
            if prog is None:
                yield f"data: {json.dumps({'status': 'error: unknown job'})}\n\n"
                break
            yield f"data: {json.dumps(_public_job(prog))}\n\n"
            status = str(prog.get("status", ""))
            if status == "done" or status.startswith("error"):
                # Finished jobs are kept for a while (pruned later) so a reconnect sees the result
                break
            time.sleep(0.3)
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─── API: IMAGES LIST ────────────────────────────────────────────────────────
@app.route("/api/images")
def api_images():
    split = request.args.get("split", "train")
    validate_split(split)
    stems = label_stems(split)
    out = []
    for e in db_get_entries(split):
        ann = is_annotated(e["name"], stems)
        out.append({"name": e["name"], "annotated": ann, "reviewed": ann and e["reviewed"]})
    return jsonify(out)


# ─── API: DATASET PAGE (for Grid View) ───────────────────────────────────────
DATASET_PAGE_FILTERS = {"all", "pending", "auto", "reviewed", "done"}


@app.route("/api/dataset_page", methods=["GET", "POST"])
def api_dataset_page():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        split = data.get("split", "train")
        page = int(data.get("page", 1))
        limit = int(data.get("limit", 50))
        filt = data.get("filter", "all")
        class_id_str = str(data.get("class_id", ""))
        class_ids_param = data.get("class_ids", None)
        filter_files = data.get("filter_files", None) # list of filenames
    else:
        split = request.args.get("split", "train")
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 50))
        filt = request.args.get("filter", "all")
        class_id_str = request.args.get("class_id", "")
        class_ids_param = request.args.get("class_ids", None)
        filter_files = None

    validate_split(split)
    if filt not in DATASET_PAGE_FILTERS:
        raise ValueError(f"Invalid filter (expected one of {sorted(DATASET_PAGE_FILTERS)})")
    limit = max(1, limit)

    target_class_ids = set()
    if isinstance(class_ids_param, list):
        for cid in class_ids_param:
            try:
                target_class_ids.add(int(cid))
            except (ValueError, TypeError):
                pass
    elif isinstance(class_ids_param, str) and class_ids_param.strip():
        for cid in class_ids_param.split(","):
            if cid.strip().isdigit():
                target_class_ids.add(int(cid.strip()))
    elif class_id_str.isdigit():
        target_class_ids.add(int(class_id_str))

    entries = db_get_entries(split)
    stems = label_stems(split)
    ff_set = set(filter_files) if isinstance(filter_files, list) else None

    filtered = []
    for e in entries:
        if ff_set is not None and e["name"] not in ff_set:
            continue
        ann = is_annotated(e["name"], stems)
        reviewed = ann and e["reviewed"]
        if filt == "done" and not ann:
            continue
        if filt == "pending" and ann:
            continue
        if filt == "auto" and (not ann or reviewed):
            continue
        if filt == "reviewed" and not reviewed:
            continue

        lbl_p = label_path(split, e["name"])
        if target_class_ids:
            if not ann:
                continue
            try:
                shapes, _bad = read_label_file(lbl_p)
            except OSError:
                shapes = []
            if not any(s["cls"] in target_class_ids for s in shapes):
                continue

        filtered.append({"name": e["name"], "annotated": ann, "reviewed": reviewed, "lbl_path": lbl_p})

    total = len(filtered)
    pages = max(1, (total + limit - 1) // limit)
    page = max(1, min(page, pages))
    start = (page - 1) * limit

    results = []
    for e in filtered[start:start + limit]:
        boxes = []
        if e["annotated"]:
            try:
                boxes, _bad = read_label_file(e["lbl_path"])
            except OSError:
                pass
        results.append({
            "name": e["name"],
            "annotated": e["annotated"],
            "reviewed": e["reviewed"],
            "boxes": boxes
        })

    return jsonify({
        "total": total,
        "page": page,
        "pages": pages,
        "images": results
    })


# ─── API: DATASET HEALTH ─────────────────────────────────────────────────────
@app.route("/api/dataset_health")
def api_dataset_health():
    """
    Scans the state.db and checks label files for:
    - Empty images (no boxes)
    - Corrupt images (0 bytes or missing)
    - Very small boxes (w*h < 0.0005)
    - Out of bounds boxes (x,y,w,h outside 0-1)
    - Malformed label lines (unparseable / wrong number of values)
    """
    stats = {
        "total_images": 0,
        "total_boxes": 0,
        "empty_images": {"count": 0, "files": []},
        "corrupt_images": {"count": 0, "files": []},
        "small_boxes": {"count": 0, "files": []},
        "oob_boxes": {"count": 0, "files": []},
        "malformed": {"count": 0, "files": []},
        "classes": {}
    }

    def flag(key, name):
        stats[key]["count"] += 1
        stats[key]["files"].append(name)

    for split in SPLITS:
        stems = label_stems(split)
        for e in db_get_entries(split):
            name = e["name"]
            stats["total_images"] += 1

            img_p = image_path(split, name)
            try:
                corrupt = os.path.getsize(img_p) == 0
            except OSError:
                corrupt = True
            if corrupt:
                flag("corrupt_images", name)
                continue

            if not is_annotated(name, stems):
                flag("empty_images", name)
                continue

            lbl_p = label_path(split, name)
            try:
                shapes, bad = read_label_file(lbl_p)
            except OSError:
                flag("empty_images", name)
                continue

            has_small = has_oob = False
            for s in shapes:
                stats["classes"][s["cls"]] = stats["classes"].get(s["cls"], 0) + 1
                stats["total_boxes"] += 1
                if "pts" in s:
                    if any(v < -0.01 or v > 1.01 for p in s["pts"] for v in (p["x"], p["y"])):
                        has_oob = True
                else:
                    x, y, w, h = s["x"], s["y"], s["w"], s["h"]
                    if w * h < 0.0005:  # Very small
                        has_small = True
                    if (x - w/2) < -0.01 or (x + w/2) > 1.01 or (y - h/2) < -0.01 or (y + h/2) > 1.01:
                        has_oob = True

            if has_small:
                flag("small_boxes", name)
            if has_oob:
                flag("oob_boxes", name)
            if bad:
                flag("malformed", name)
            if not shapes and not bad:
                flag("empty_images", name)

    # Health Score
    total = max(stats["total_images"], 1)
    score = 100
    score -= (stats["corrupt_images"]["count"] / total) * 100 * 2  # Severe penalty
    score -= (stats["empty_images"]["count"] / total) * 100 * 0.5  # Minor penalty
    score -= (stats["small_boxes"]["count"] / total) * 100 * 1
    score -= (stats["oob_boxes"]["count"] / total) * 100 * 1.5
    score -= (stats["malformed"]["count"] / total) * 100 * 1.5

    stats["score"] = max(0, min(100, int(score)))

    return jsonify(stats)


# ─── API: IMAGE ──────────────────────────────────────────────────────────────
@app.route("/api/image/<split>/<filename>")
def api_image(split, filename):
    dst = image_path(split, filename)
    if not os.path.isfile(dst):
        return jsonify({"error": "not found"}), 404
    return send_file(dst, mimetype=guess_mimetype(filename))


# ─── API: THUMBNAIL ──────────────────────────────────────────────────────────
@app.route("/api/thumb/<split>/<filename>")
def api_thumb(split, filename):
    dst = image_path(split, filename)
    try:
        mtime = os.stat(dst).st_mtime_ns
    except OSError:
        return jsonify({"error": "not found"}), 404
    key = (dst, mtime)

    def _respond(data, mimetype="image/jpeg"):
        resp = send_file(io.BytesIO(data), mimetype=mimetype)
        resp.headers["Cache-Control"] = "max-age=3600"
        return resp

    with _thumb_cache_lock:
        data = _thumb_cache.get(key)
        if data is not None:
            _thumb_cache.move_to_end(key)
    if data is not None:
        return _respond(data)

    try:
        from PIL import Image as PILImage, ImageOps
        with PILImage.open(dst) as im:
            try:
                im = ImageOps.exif_transpose(im)
            except Exception:
                pass
            im.thumbnail((450, 450))
            # Always convert to RGB to avoid RGBA/P mode JPEG errors
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
        data = buf.getvalue()
        with _thumb_cache_lock:
            _thumb_cache[key] = data
            while len(_thumb_cache) > _THUMB_CACHE_MAX:
                _thumb_cache.popitem(last=False)
        return _respond(data)
    except Exception:
        # If PIL is unavailable or fails, serve the original with its real mime type
        resp = send_file(dst, mimetype=guess_mimetype(filename))
        resp.headers["Cache-Control"] = "max-age=3600"
        return resp


# ─── API: LABELS (on-demand annotation) ──────────────────────────────────────
def _shapes_with_ids(lbl):
    if not os.path.exists(lbl):
        return []
    shapes, _bad = read_label_file(lbl)
    for i, s in enumerate(shapes):
        s["id"] = i
    return shapes


@app.route("/api/labels/<split>/<filename>")
def api_labels(split, filename):
    lbl = label_path(split, filename)
    dst = image_path(split, filename)

    if _auto_annotate and model_available() and not os.path.exists(lbl) and os.path.exists(dst):
        with get_img_lock(f"{split}/{filename}"):
            if not os.path.exists(lbl):
                try:
                    # Returns False (no file written) when no model class is selected
                    annotate_one(dst, lbl)
                except Exception as ex:
                    print(f"[LABELS] Auto-annotation failed for {filename}: {ex}")
                    return jsonify({"error": f"Auto-annotation failed: {ex}"}), 500

    return jsonify(_shapes_with_ids(lbl))


# ─── API: SAVE LABELS ────────────────────────────────────────────────────────
@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON format"}), 400
    split = data.get("split", "train")
    fname = data.get("filename", "")
    try:
        validate_split(split)
        validate_name(fname)
    except ValueError as ex:
        return jsonify({"error": str(ex)}), 400
    lines, skipped = sanitize_shapes(data.get("boxes", []))
    lbl = label_path(split, fname)
    with get_img_lock(f"{split}/{fname}"), _labels_write_lock:
        atomic_write_text(lbl, "".join(l + "\n" for l in lines))
        db_set_reviewed(fname, split, True)
    return jsonify({"status": "ok", "reviewed": True, "saved": len(lines), "skipped": skipped})


# ─── API: REVIEW STATUS ──────────────────────────────────────────────────────
def db_is_reviewed(name, split):
    conn = get_db()
    row = conn.execute("SELECT reviewed FROM images WHERE name=? AND split=?", (name, split)).fetchone()
    conn.close()
    return None if row is None else bool(row["reviewed"])


@app.route("/api/mark_reviewed", methods=["POST"])
def api_mark_reviewed():
    """Toggle the reviewed flag without rewriting labels.

    Marking an image without a label file as reviewed creates an EMPTY label file
    (a reviewed background image), so it counts as annotated.
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON format"}), 400
    split = data.get("split", "train")
    fname = data.get("filename", "")
    reviewed = data.get("reviewed", True)
    if not isinstance(reviewed, bool):
        return jsonify({"error": "reviewed must be a boolean"}), 400
    validate_split(split)
    validate_name(fname)
    if db_is_reviewed(fname, split) is None:
        return jsonify({"error": "image not found"}), 404
    lbl = label_path(split, fname)
    with get_img_lock(f"{split}/{fname}"):
        if reviewed and not os.path.exists(lbl):
            atomic_write_text(lbl, "")
        db_set_reviewed(fname, split, reviewed)
    return jsonify({"status": "ok", "reviewed": reviewed, "annotated": os.path.exists(lbl)})


# ─── API: RE-RUN AUTO-ANNOTATION ─────────────────────────────────────────────
NO_CLASSES_MSG = "No classes selected for auto-labelling — choose classes"


@app.route("/api/relabel", methods=["POST"])
def api_relabel():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON format"}), 400
    split = data.get("split", "train")
    fname = data.get("filename", "")
    force = data.get("force", False) is True
    validate_split(split)
    validate_name(fname)
    dst = image_path(split, fname)
    lbl = label_path(split, fname)
    if not model_available():
        return jsonify({"error": NO_MODEL_MSG}), 400
    state = db_is_reviewed(fname, split)
    if state is None or not os.path.exists(dst):
        return jsonify({"error": "image not found"}), 404
    with get_img_lock(f"{split}/{fname}"):
        if db_is_reviewed(fname, split) and not force:
            return jsonify({"error": "Image is reviewed — pass force to overwrite"}), 409
        try:
            written = annotate_one(dst, lbl)
        except Exception as ex:
            return jsonify({"error": f"Auto-annotation failed: {ex}"}), 500
        if not written:
            return jsonify({"error": NO_CLASSES_MSG}), 400
        db_set_reviewed(fname, split, False)
    return jsonify(_shapes_with_ids(lbl))


@app.route("/api/relabel_all", methods=["POST"])
def api_relabel_all():
    """Background job: re-annotate every NOT reviewed image of a split (or all splits)."""
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON format"}), 400
    split = data.get("split")
    if split is not None:
        validate_split(split)
    splits = [split] if split else list(SPLITS)
    if not model_available():
        return jsonify({"error": NO_MODEL_MSG}), 400
    if _model is not None and not _model_class_map:
        return jsonify({"error": NO_CLASSES_MSG}), 400

    targets = [(s, e["name"]) for s in splits for e in db_get_entries(s) if not e["reviewed"]]
    job_id = new_job(total=len(targets), _kind="relabel_all")
    session = _current_session_id

    def run():
        try:
            _job_progress[job_id]["status"] = "labelling"
            done = 0
            for s, name in targets:
                if session != _current_session_id:
                    raise RuntimeError("dataset was switched")
                dst, lbl = image_path(s, name), label_path(s, name)
                if os.path.exists(dst):
                    with get_img_lock(f"{s}/{name}"):
                        if not db_is_reviewed(name, s):     # re-check: never touch reviewed images
                            if not annotate_one(dst, lbl):
                                raise RuntimeError(NO_CLASSES_MSG)
                            db_set_reviewed(name, s, False)
                done += 1
                _job_progress[job_id]["done"] = done
            finish_job(job_id, "done")
        except Exception as ex:
            print(f"[RELABEL] Failed: {ex}")
            finish_job(job_id, f"error: {ex}")

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id})


# ─── API: DELETE IMAGE ───────────────────────────────────────────────────────
@app.route("/api/delete_image", methods=["POST"])
def api_delete_image():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON format"}), 400
    split = data.get("split", "train")
    fname = data.get("filename", "")
    validate_split(split)
    validate_name(fname)
    with get_img_lock(f"{split}/{fname}"):
        for p in [image_path(split, fname), label_path(split, fname)]:
            if os.path.exists(p):
                os.remove(p)
        db_delete_image(fname, split)
    clear_thumb_cache()
    return jsonify({"status": "deleted"})


@app.route("/api/delete_images", methods=["POST"])
def api_delete_images():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON format"}), 400
    split = data.get("split", "train")
    filenames = data.get("filenames", [])
    if not isinstance(filenames, list) or not filenames:
        return jsonify({"error": "filenames must be a non-empty list"}), 400
    validate_split(split)
    deleted = 0
    errors = []
    for fname in filenames:
        if not isinstance(fname, str) or not fname:
            continue
        try:
            validate_name(fname)
            with get_img_lock(f"{split}/{fname}"):
                for p in [image_path(split, fname), label_path(split, fname)]:
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except OSError as e:
                            errors.append(f"{fname}: {e}")
                db_delete_image(fname, split)
                deleted += 1
        except Exception as e:
            errors.append(f"{fname}: {e}")
    clear_thumb_cache()
    return jsonify({"status": "ok", "deleted": deleted, "deleted_count": deleted, "errors": errors})


# ─── API: MODEL TRAINING ─────────────────────────────────────────────────────
_TRAIN_DEVICE_RE = re.compile(r"^\d+(,\d+)*$")


def normalize_train_device(device):
    """Accept '0', '0,1', 'mps' (GPU only). Returns the normalised string or None."""
    if not isinstance(device, str):
        return None
    d = device.replace(" ", "").lower()
    if d.startswith("cuda:"):
        d = d[5:]
    if d == "mps" or _TRAIN_DEVICE_RE.match(d):
        return d
    return None


@app.route("/api/train", methods=["POST"])
def api_train():
    global _train_process
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON format"}), 400
    model_file = data.get("model") or "yolov8n.pt"
    custom = data.get("custom_model")
    if isinstance(custom, str) and custom.strip():
        model_file = custom.strip()
    if not isinstance(model_file, str):
        return jsonify({"error": "model must be a string"}), 400
    model_file = resolve_model_path(model_file)
    if Path(model_file).suffix.lower() not in MODEL_EXTS and Path(model_file).suffix.lower() != ".yaml":
        return jsonify({"error": f"Unsupported model format '{Path(model_file).suffix}'."}), 400
    epochs     = int(data.get("epochs",    50))
    batch      = int(data.get("batch",     16))
    imgsz      = int(data.get("imgsz",    640))
    lr0        = float(data.get("lr0",   0.01))
    project    = str(data.get("project") or "runs/train")
    name       = str(data.get("name") or "vehicle_detect")
    yaml_path  = os.path.join(OUTPUT_DIR, "data.yaml")

    # Enforce GPU-only training restriction (CPU training disabled)
    device = normalize_train_device(data.get("device", "cpu"))
    if device is None:
        return jsonify({"error": "Training on CPU is disabled. A dedicated CUDA GPU or Apple Silicon MPS device is required."}), 400

    if dataset_loaded() and _class_names:
        write_yaml()
    if not os.path.exists(yaml_path):
        return jsonify({"error": "data.yaml not found — run auto-split first"}), 400
    if not _class_names:
        return jsonify({"error": "No classes defined — add classes before training"}), 400

    cmd = [
        sys.executable, "-m", "ultralytics", "train",
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

    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    with _train_lock:
        if _train_process and _train_process.poll() is None:
            return jsonify({"error": "Training already running"}), 400
        # Release the inference model from GPU memory before training
        reset_model()
        _train_log_buffer.reset()
        proc = subprocess.Popen(
            cmd, cwd=BASE_DIR, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
        )
        _train_process = proc

    def read_logs():
        try:
            for line in proc.stdout:
                _train_log_buffer.append(line.rstrip())
        except Exception as ex:
            _train_log_buffer.append(f"[log reader error] {ex}")
        finally:
            _train_log_buffer.close()

    threading.Thread(target=read_logs, daemon=True).start()
    return jsonify({"status": "started", "model": model_file, "device": device})


@app.route("/api/train/stop", methods=["POST"])
def api_train_stop():
    with _train_lock:
        if _train_process and _train_process.poll() is None:
            _train_process.terminate()
            return jsonify({"status": "stopped"})
    return jsonify({"status": "not_running"})


@app.route("/api/train/logs")
def api_train_logs():
    """SSE stream of training output; replays the buffered lines first (reconnect-safe)."""
    def generate():
        cursor = 0
        while True:
            lines, cursor = _train_log_buffer.read(cursor)
            for line in lines:
                yield f"data: {json.dumps({'line': line})}\n\n"
            proc = _train_process
            finished = proc is None or (proc.poll() is not None and _train_log_buffer.closed)
            if finished and cursor >= _train_log_buffer.total:
                yield f"data: {json.dumps({'line': '--- Training complete ---', 'done': True})}\n\n"
                break
            time.sleep(0.5)
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/train/status")
def api_train_status():
    lines = _train_log_buffer.total
    if _train_process is None:
        return jsonify({"status": "idle", "lines": lines})
    code = _train_process.poll()
    if code is None:
        return jsonify({"status": "running", "lines": lines})
    return jsonify({"status": "done", "exit_code": code, "lines": lines})


@app.route("/api/models")
def api_models():
    # Scan MODELS_DIR, BASE_DIR, and APP_DIR for all supported model formats
    found_paths = set()
    all_models = []

    def _scan_dir(directory):
        if not directory or not os.path.exists(directory):
            return
        for root, dirs, files in os.walk(directory):
            # Skip non-model/system folders
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'venv', '.venv', '__pycache__', 'runs', 'static', 'templates', 'tests', 'dist', 'build', 'node_modules', 'annotated_dataset', 'vedio'}]
            for f in files:
                if Path(f).suffix.lower() in MODEL_EXTS:
                    full_p = os.path.abspath(os.path.join(root, f))
                    if full_p not in found_paths:
                        found_paths.add(full_p)
                        if directory == MODELS_DIR:
                            rel_p = os.path.relpath(full_p, MODELS_DIR).replace("\\", "/")
                        elif directory == BASE_DIR:
                            rel_p = os.path.relpath(full_p, BASE_DIR).replace("\\", "/")
                        else:
                            rel_p = os.path.basename(full_p)
                        all_models.append(rel_p)

    if os.path.exists(MODELS_DIR):
        _scan_dir(MODELS_DIR)

    # Scan top-level BASE_DIR files
    if os.path.exists(BASE_DIR) and os.path.abspath(BASE_DIR) != os.path.abspath(MODELS_DIR):
        try:
            for item in os.listdir(BASE_DIR):
                full_p = os.path.abspath(os.path.join(BASE_DIR, item))
                if os.path.isfile(full_p) and Path(item).suffix.lower() in MODEL_EXTS:
                    if full_p not in found_paths:
                        found_paths.add(full_p)
                        all_models.append(item)
        except Exception:
            pass

    # Scan parent models folder if distinct
    parent_models = os.path.abspath(os.path.join(APP_DIR, "..", "models"))
    if os.path.exists(parent_models) and parent_models not in (os.path.abspath(MODELS_DIR), os.path.abspath(BASE_DIR)):
        _scan_dir(parent_models)

    all_models.sort(key=lambda x: x.lower())
    presets = [
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
        "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"
    ]
    return jsonify({"files": all_models, "presets": presets})


# ─── API: DATASET TOOLS (RESIZER & COMPRESSOR) ───────────────────────────────
_tool_jobs = {}
_tool_job_lock = threading.Lock()

def _set_tool_job(job_name, **kwargs):
    with _tool_job_lock:
        if job_name not in _tool_jobs:
            _tool_jobs[job_name] = {}
        _tool_jobs[job_name].update(kwargs)

def _get_tool_job(job_name):
    with _tool_job_lock:
        return dict(_tool_jobs.get(job_name, {"running": False, "percent": 0, "status": "Idle"}))


def _import_script(module_name):
    scripts_dir = os.path.join(APP_DIR, "scripts")
    if scripts_dir not in sys.path:
        sys.path.append(scripts_dir)
    return __import__(module_name)


@app.route("/api/tools/resize", methods=["POST"])
def api_tools_resize():
    data = request.get_json(silent=True) or {}
    target_size = int(data.get("size", 640))
    mode = data.get("mode", "letterbox")
    quality = int(data.get("quality", 95))
    workers = int(data.get("workers", 8))
    if not (32 <= target_size <= 8192):
        return jsonify({"error": "size must be between 32 and 8192"}), 400

    # Capture the dataset paths NOW: the worker must not read the global later
    # (the active dataset may change while the job runs).
    input_dir = OUTPUT_DIR
    out_dir = f"{input_dir}_{target_size}"

    with _tool_job_lock:
        current_job = _tool_jobs.get("resize", {})
        if current_job.get("running"):
            return jsonify({"error": "A dataset resizing job is already running."}), 400
        _tool_jobs["resize"] = {
            "running": True,
            "percent": 0,
            "current": 0,
            "total": 0,
            "status": "Initializing...",
            "error": None,
            "out_dir": out_dir
        }

    def run_resize():
        try:
            resize_dataset = _import_script("resize_dataset").resize_dataset

            def on_progress(cur, tot, msg):
                pct = int((cur / max(1, tot)) * 100) if tot > 0 else 0
                _set_tool_job("resize", current=cur, total=tot, percent=pct, status=msg)

            success = resize_dataset(
                input_dir=input_dir,
                output_dir=out_dir,
                target_w=target_size,
                target_h=target_size,
                mode=mode,
                quality=quality,
                workers=workers,
                progress_callback=on_progress
            )
            if success:
                _set_tool_job("resize", running=False, percent=100, status=f"Completed! Saved to {os.path.basename(out_dir)}")
            else:
                _set_tool_job("resize", running=False, error="Resizing failed", status="Failed")
        except Exception as e:
            _set_tool_job("resize", running=False, error=str(e), status=f"Error: {e}")

    threading.Thread(target=run_resize, daemon=True).start()
    return jsonify({"status": "started", "job": "resize"})


@app.route("/api/tools/compress", methods=["POST"])
def api_tools_compress():
    data = request.get_json(silent=True) or {}
    fmt = data.get("format", "zip")
    method = data.get("method", "deflated")
    level = int(data.get("level", 6))
    verify = bool(data.get("verify", True))

    # Capture paths at request time (see resize)
    input_dir = OUTPUT_DIR
    dataset_name = _current_dataset_name

    with _tool_job_lock:
        current_job = _tool_jobs.get("compress", {})
        if current_job.get("running"):
            return jsonify({"error": "A dataset compression job is already running."}), 400
        _tool_jobs["compress"] = {
            "running": True,
            "percent": 0,
            "current": 0,
            "total": 0,
            "status": "Initializing...",
            "error": None,
            "download_url": None,
            "filename": None
        }

    def run_compress():
        try:
            compress_dataset = _import_script("compress_dataset").compress_dataset

            ext_map = {"zip": ".zip", "tar.gz": ".tar.gz", "tar.xz": ".tar.xz", "7z": ".7z"}
            ext = ext_map.get(str(fmt).lower(), f".{fmt}")
            filename = f"{dataset_name}_package{ext}"
            out_path = os.path.join(os.path.dirname(input_dir), filename)

            def on_progress(cur, tot, msg):
                pct = int((cur / max(1, tot)) * 100) if tot > 0 else 0
                _set_tool_job("compress", current=cur, total=tot, percent=pct, status=msg)

            success = compress_dataset(
                input_dir=input_dir,
                output_file=out_path,
                fmt=fmt,
                method=method,
                level=level,
                verify=verify,
                progress_callback=on_progress
            )
            if success:
                _set_tool_job(
                    "compress",
                    running=False,
                    percent=100,
                    status="Archive created successfully!",
                    filename=filename,
                    download_url=f"/api/tools/download?file={filename}"
                )
            else:
                _set_tool_job("compress", running=False, error="Compression failed", status="Failed")
        except Exception as e:
            _set_tool_job("compress", running=False, error=str(e), status=f"Error: {e}")

    threading.Thread(target=run_compress, daemon=True).start()
    return jsonify({"status": "started", "job": "compress"})


@app.route("/api/tools/status")
def api_tools_status():
    job_name = request.args.get("job", "resize")
    return jsonify(_get_tool_job(job_name))


@app.route("/api/tools/download")
def api_tools_download():
    filename = request.args.get("file", "")
    if not filename or ".." in filename or "/" in filename or "\\" in filename:
        return jsonify({"error": "Invalid filename"}), 400
    file_path = os.path.join(os.path.dirname(OUTPUT_DIR), filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, as_attachment=True, download_name=filename)


# ─── API: EXPORT DATASET ─────────────────────────────────────────────────────
def _split_image_files(split):
    img_dir = os.path.join(OUTPUT_DIR, split, "images")
    try:
        return sorted(f for f in os.listdir(img_dir)
                      if Path(f).suffix.lower() in IMG_EXTS and os.path.isfile(os.path.join(img_dir, f)))
    except OSError:
        return []


def _write_yolo_export(zf):
    """YOLO zip: only <split>/images/<img>, <split>/labels/<stem>.txt and data.yaml."""
    for split in SPLITS:
        for img_file in _split_image_files(split):
            zf.write(os.path.join(OUTPUT_DIR, split, "images", img_file), f"{split}/images/{img_file}")
        lbl_dir = os.path.join(OUTPUT_DIR, split, "labels")
        try:
            lbl_files = sorted(f for f in os.listdir(lbl_dir) if f.endswith(".txt"))
        except OSError:
            lbl_files = []
        for lf in lbl_files:
            zf.write(os.path.join(lbl_dir, lf), f"{split}/labels/{lf}")
    yaml_p = os.path.join(OUTPUT_DIR, "data.yaml")
    if os.path.exists(yaml_p):
        zf.write(yaml_p, "data.yaml")


def _write_coco_export(zf):
    for split in SPLITS:
        img_files = _split_image_files(split)
        if not img_files:
            continue
        img_dir = os.path.join(OUTPUT_DIR, split, "images")
        lbl_dir = os.path.join(OUTPUT_DIR, split, "labels")

        coco = {
            "info": {"description": "Exported from Data Annotator Tool"},
            "categories": [{"id": i, "name": n, "supercategory": "none"} for i, n in enumerate(_class_names)],
            "images": [],
            "annotations": []
        }

        ann_id = 1
        for img_id, img_file in enumerate(img_files, start=1):
            img_p = os.path.join(img_dir, img_file)
            h, w = 1000, 1000
            try:
                from PIL import Image as PILImage
                with PILImage.open(img_p) as im:
                    w, h = im.size
            except Exception:
                pass

            coco["images"].append({"id": img_id, "file_name": img_file, "width": w, "height": h})
            zf.write(img_p, f"{split}/images/{img_file}")

            lbl_p = os.path.join(lbl_dir, os.path.splitext(img_file)[0] + ".txt")
            if not os.path.exists(lbl_p):
                continue
            try:
                shapes, _bad = read_label_file(lbl_p)
            except OSError:
                continue
            for s in shapes:
                if "pts" in s:
                    xs = [p["x"] * w for p in s["pts"]]
                    ys = [p["y"] * h for p in s["pts"]]
                    seg = [v for xy in zip(xs, ys) for v in xy]
                    xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
                    coco["annotations"].append({
                        "id": ann_id, "image_id": img_id, "category_id": s["cls"],
                        "bbox": [xmin, ymin, xmax - xmin, ymax - ymin], "area": (xmax - xmin) * (ymax - ymin),
                        "iscrowd": 0, "segmentation": [seg]
                    })
                else:
                    bw, bh = s["w"] * w, s["h"] * h
                    x = (s["x"] - s["w"] / 2) * w
                    y = (s["y"] - s["h"] / 2) * h
                    coco["annotations"].append({
                        "id": ann_id, "image_id": img_id, "category_id": s["cls"],
                        "bbox": [x, y, bw, bh], "area": bw * bh,
                        "iscrowd": 0, "segmentation": []
                    })
                ann_id += 1

        zf.writestr(f"{split}/_annotations.coco.json", json.dumps(coco))


@app.route("/api/export")
def api_export():
    """Zips the annotated dataset to disk and serves it for download."""
    fmt = request.args.get("format", "yolo")
    temp_fd, temp_path = tempfile.mkstemp(suffix=".zip", prefix=f"{_current_dataset_name}_")
    os.close(temp_fd)

    with zipfile.ZipFile(temp_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        if fmt == "coco":
            _write_coco_export(zf)
        else:
            _write_yolo_export(zf)

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
    return send_file(os.path.join(APP_DIR, "index.html"))

@app.route("/style.css")
def serve_css():
    return send_file(os.path.join(APP_DIR, "static", "css", "style.css"))

@app.route("/app.js")
def serve_js():
    return send_file(os.path.join(APP_DIR, "static", "js", "app.js"))


def _run_flask_server(host, port):
    try:
        app.run(debug=False, host=host, port=port, threaded=True, use_reloader=False)
    except Exception as e:
        print(f"[ERROR] Flask server error: {e}")


def _port_is_free(host, port):
    """True if nothing else holds the port.

    Uses SO_EXCLUSIVEADDRUSE on Windows: the dev server sets SO_REUSEADDR, which on
    Windows lets a second process bind the same port, so new connections can land on an
    old, frozen copy of the app. An exclusive bind detects that case.
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _server_responds(url, timeout=3.0):
    import urllib.request
    try:
        with urllib.request.urlopen(url + "/api/state", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _wait_until_ready(url, seconds=20.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if _server_responds(url, timeout=1.0):
            return True
        time.sleep(0.25)
    return False


def _find_app_browser():
    """Edge or Chrome executable for a chromeless --app window (Windows / Linux / macOS)."""
    candidates = []
    if sys.platform == "win32":
        for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
            if base:
                candidates += [os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"),
                               os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")]
    elif sys.platform == "darwin":
        candidates += ["/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                       "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    else:
        for name in ("microsoft-edge", "google-chrome", "chromium", "chromium-browser"):
            found = shutil.which(name)
            if found:
                candidates.append(found)
    return next((c for c in candidates if c and os.path.exists(c)), None)


def _open_app_window(url):
    """Open the UI in its own window without browser tabs/address bar.

    Uses a dedicated browser profile so the window is its own process: closing it shuts
    the app down, like the pywebview window. Returns True if the window ran.
    """
    exe = _find_app_browser()
    if not exe:
        return False
    profile = os.path.join(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir(),
                           "DataAnnotationStudio", "window-profile")
    os.makedirs(profile, exist_ok=True)
    cmd = [exe, f"--app={url}", f"--user-data-dir={profile}", "--window-size=1440,900",
           "--no-first-run", "--no-default-browser-check", "--disable-features=Translate"]
    try:
        print(f"[*] Opening app window ({os.path.basename(exe)} app mode)...")
        proc = subprocess.Popen(cmd)
        proc.wait()
        return True
    except Exception as e:
        print(f"[!] Could not open app window ({e})")
        return False


def _shutdown_now(code=0):
    """Exit immediately.

    Normal interpreter shutdown can stall for a long time on background threads (model
    loading, SSE streams, a sqlite call stuck on another program's lock). All writes are
    already atomic (sqlite transactions, temp-file + os.replace), so a hard exit is safe.
    """
    try:
        proc = _train_process
        if proc is not None and proc.poll() is None:
            print("[*] Stopping training process...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
    except Exception:
        pass
    print("[*] Stopped.")
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(code)


LAUNCH_MODES = ("browser", "window", "app", "none")


def _resolve_launch_mode(args):
    """CLI flag > ANNOTATOR_LAUNCH env > config.ini [App] LAUNCH_MODE > 'browser'."""
    if args.launch:
        return args.launch
    if args.no_gui or os.environ.get("NO_GUI") == "1":
        return "none"
    if args.browser:
        return "browser"
    if args.window:
        return "window"
    mode = (os.environ.get("ANNOTATOR_LAUNCH") or config.get("App", "LAUNCH_MODE", fallback="browser")).strip().lower()
    if mode not in LAUNCH_MODES:
        print(f"[!] Unknown LAUNCH_MODE '{mode}' (use {', '.join(LAUNCH_MODES)}) - using browser.")
        mode = "browser"
    return mode


def _open_browser(url):
    import webbrowser
    try:
        print(f"[*] Opening {url} in your browser...")
        webbrowser.open(url)
    except Exception as e:
        print(f"[!] Could not open the browser ({e}). Open {url} manually.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Data Annotation Studio")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8876, help="Port number (default: 8876)")
    parser.add_argument("--launch", choices=LAUNCH_MODES, help="How to show the UI (overrides config.ini [App] LAUNCH_MODE)")
    parser.add_argument("--browser", action="store_true", help="Open in the default web browser (same as --launch browser)")
    parser.add_argument("--window", action="store_true", help="Open a desktop window (same as --launch window)")
    parser.add_argument("--no-gui", "--headless", action="store_true", help="Server only, open nothing (same as --launch none)")
    args, _ = parser.parse_known_args()
    mode = _resolve_launch_mode(args)

    # ── Port check: never share a port with another (possibly frozen) copy ──
    port = args.port
    if not _port_is_free(args.host, port):
        existing = f"http://{args.host}:{port}"
        if _server_responds(existing):
            print(f"[*] Data Annotation Studio is already running at {existing} - opening it.")
            if mode == "browser" or (mode in ("window", "app") and not _open_app_window(existing)):
                _open_browser(existing)
            return
        print(f"[!] Port {port} is held by another program that is not responding")
        print("    (usually an old copy of this app that froze). Close it in Task Manager")
        print("    (python.exe / pythonw.exe) to use the default port again.")
        for alt in range(port + 1, port + 20):
            if _port_is_free(args.host, alt):
                port = alt
                break
        else:
            print("[ERROR] No free port found. Start with --port <number>.")
            sys.exit(1)
        print(f"[*] Using port {port} instead.")

    if os.path.exists(DB_PATH) and not db_responds(DB_PATH, 4.0):
        print(f"[!] {DB_PATH} is locked by another program (usually an old copy of this app")
        print("    that froze). Close it in Task Manager, otherwise this dataset will not load.")
    init_db()
    url = f"http://{args.host}:{port}"
    print("=" * 64)
    print("  Data Annotation Studio")
    print(f"  Version: {__version__}")
    print(f"  Output : {OUTPUT_DIR}")
    print(f"  Server : {url}")
    print(f"  Launch : {mode}")
    print("=" * 64)

    server_thread = threading.Thread(target=_run_flask_server, args=(args.host, port), daemon=True)
    server_thread.start()
    if not _wait_until_ready(url):
        print("[!] Server did not answer within 20 s - opening the UI anyway.")

    if mode == "window":
        try:
            import webview
            print("[*] Launching native desktop window (pywebview)...")
            webview.create_window(
                title="Data Annotation Studio",
                url=url,
                width=1440,
                height=900,
                min_size=(1024, 700),
                background_color="#0E1013"
            )
            webview.start()
            print("[*] Window closed. Shutting down...")
            _shutdown_now(0)
        except Exception as e:
            print(f"[!] pywebview not available ({e}). Using an app window instead.")
            mode = "app"

    if mode == "app":
        if _open_app_window(url):
            print("[*] Window closed. Shutting down...")
            _shutdown_now(0)
        print("[!] No Edge/Chrome found for an app window. Opening the default browser...")
        mode = "browser"

    if mode == "browser":
        _open_browser(url)
    else:
        print(f"[*] Server only. Open {url} in your browser.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Shutting down server...")
        _shutdown_now(0)


if __name__ == "__main__":
    main()
