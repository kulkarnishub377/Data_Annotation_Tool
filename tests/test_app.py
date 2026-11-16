"""Unit / API tests for the Data Annotation Tool backend.

Isolation: these tests NEVER touch real datasets. Before ``app`` is imported,
OUTPUT_DIR is pointed at a temporary directory; BASE_DIR is then re-pointed at
the same temp root (ingest creates ``dataset_<name>`` there) and every test
works on a freshly generated dataset. Auto-annotation is replaced by a fake so
no YOLO model is ever loaded (no GPU, no download, no network needed).
"""
import io
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest
import zipfile
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

TMP_ROOT = tempfile.mkdtemp(prefix="annot_tool_tests_")
# Highest-priority overrides (beat config.ini / .env), so importing app never opens a real dataset
os.environ["ANNOTATOR_BASE_DIR"] = TMP_ROOT
os.environ["ANNOTATOR_OUTPUT_DIR"] = os.path.join(TMP_ROOT, "dataset_boot")
os.environ["ANNOTATOR_MODELS_DIR"] = os.path.join(TMP_ROOT, "models")

import app as app_module  # noqa: E402
from app import (app, auto_detect_device, calculate_iou, get_device_info,  # noqa: E402
                 validate_split, secure_path)

if os.path.commonpath([os.path.abspath(app_module.OUTPUT_DIR), TMP_ROOT]) != TMP_ROOT:
    raise RuntimeError("Test isolation failure: OUTPUT_DIR is not inside the temp dir "
                       f"({app_module.OUTPUT_DIR}).")

_ORIG = {k: getattr(app_module, k) for k in ("BASE_DIR", "annotate_one", "get_model", "_model_path")}


def _fake_annotate_one(img, lbl):
    """Default stand-in for YOLO: nothing selected -> nothing written."""
    return False


def _no_model():
    raise RuntimeError("tests must not load a YOLO model")


def setUpModule():
    app_module.BASE_DIR = TMP_ROOT
    app_module.annotate_one = _fake_annotate_one
    app_module.get_model = _no_model
    app_module._model_path = "test-model-does-not-exist.pt"


def tearDownModule():
    for k, v in _ORIG.items():
        setattr(app_module, k, v)
    shutil.rmtree(TMP_ROOT, ignore_errors=True)


def make_image(path, size=(64, 48), fmt=None, color=(120, 30, 30)):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, color).save(path, format=fmt)


_counter = [0]


def make_dataset(spec=None, labels=None):
    """Create + activate a temp dataset.

    spec: {split: [image names]}; labels: {(split, name): "label text"}.
    """
    _counter[0] += 1
    ds = os.path.join(TMP_ROOT, f"dataset_t{_counter[0]}")
    spec = spec if spec is not None else {"train": ["a.jpg", "b.png", "c.webp"],
                                          "valid": ["d.jpg"], "test": ["e.jpg"]}
    labels = labels or {}
    for split in app_module.SPLITS:
        os.makedirs(os.path.join(ds, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(ds, split, "labels"), exist_ok=True)
        for name in spec.get(split, []):
            fmt = {".png": "PNG", ".webp": "WEBP", ".bmp": "BMP"}.get(os.path.splitext(name)[1], "JPEG")
            make_image(os.path.join(ds, split, "images", name), fmt=fmt)
    for (split, name), text in labels.items():
        with open(os.path.join(ds, split, "labels", os.path.splitext(name)[0] + ".txt"), "w") as f:
            f.write(text)
    app_module.set_active_dataset(ds, is_existing=True)
    for split, names in spec.items():
        app_module.db_insert_entries([{"name": n, "src": ""} for n in names], split)
    return ds


class BaseCase(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        app.testing = True
        app_module._model_names = []
        app_module._model_class_map = {}
        app_module._model_map_path = ""
        app_module._model_map_names = []
        app_module._model_map_warnings = []
        self.ds = make_dataset()
        app_module._auto_annotate = False
        app_module._model_explicit = True   # behave as if the user loaded a model

    def label_file(self, split, name):
        return os.path.join(self.ds, split, "labels", os.path.splitext(name)[0] + ".txt")


# ─── pure helpers ────────────────────────────────────────────────────────────
class TestHelpers(unittest.TestCase):
    def test_auto_detect_device(self):
        device = auto_detect_device()
        self.assertTrue(device in ("cpu", "mps") or device.isdigit())

    def test_get_device_info(self):
        info = get_device_info()
        for k in ("cpu", "cuda", "mps", "selected", "has_gpu"):
            self.assertIn(k, info)

    def test_calculate_iou(self):
        box1 = [0.1, 0.1, 0.5, 0.5]
        self.assertAlmostEqual(calculate_iou(box1, list(box1)), 1.0, places=4)
        self.assertAlmostEqual(calculate_iou(box1, [0.6, 0.6, 0.9, 0.9]), 0.0, places=4)
        iou = calculate_iou(box1, [0.3, 0.3, 0.7, 0.7])
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)

    def test_validate_split(self):
        for s in ("train", "valid", "test"):
            validate_split(s)
        with self.assertRaises(ValueError):
            validate_split("unknown")

    def test_secure_path_traversal(self):
        base = os.path.join(TMP_ROOT, "fake_base")
        self.assertTrue(secure_path(base, "images", "car.jpg").startswith(os.path.abspath(base)))
        with self.assertRaises(ValueError):
            secure_path(base, "images", "../../../etc/passwd")

    def test_parse_label_line(self):
        p = app_module.parse_label_line
        self.assertIsNone(p("   "))
        self.assertEqual(p("1 0.5 0.5 0.2 0.2")["cls"], 1)
        self.assertEqual(len(p("0 0.1 0.1 0.2 0.1 0.2 0.2")["pts"]), 3)
        for bad in ("x 0.5 0.5 0.2 0.2", "0 0.5 0.5", "0 0.1 0.1 0.2 0.1 0.2", "-1 0.5 0.5 0.1 0.1"):
            with self.assertRaises(ValueError):
                p(bad)

    def test_sanitize_shapes(self):
        lines, skipped = app_module.sanitize_shapes([
            {"cls": 0, "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2},
            {"cls": 1, "x": 0.95, "y": 0.5, "w": 0.2, "h": 0.2},          # clamped at the right edge
            {"cls": 0, "pts": [{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.1}]},  # < 3 points
            {"cls": "a", "x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1},           # bad class
            {"cls": 0, "x": 0.5, "y": None, "w": 0.1, "h": 0.1},            # bad coord
            {"cls": 2, "pts": [{"x": -1, "y": 0.1}, {"x": 2, "y": 0.1}, {"x": 0.5, "y": 0.9}]},
        ])
        self.assertEqual(len(lines), 3)
        self.assertEqual(skipped, 3)
        parts = lines[1].split()
        self.assertAlmostEqual(float(parts[1]) + float(parts[3]) / 2, 1.0, places=5)
        self.assertTrue(lines[2].startswith("2 0.000000 0.100000 1.000000 0.100000"))

    def test_log_buffer_monotonic_counter(self):
        buf = app_module.LogBuffer(maxlen=2000)
        lines, cur = buf.read(0)
        self.assertEqual((lines, cur), ([], 0))
        for i in range(600):
            buf.append(f"line {i}")
        lines, cur = buf.read(0)
        self.assertEqual(len(lines), 600)
        self.assertEqual(cur, 600)
        # A reader past the old 500-line cap keeps receiving new lines (old bug: stalled forever)
        for i in range(600, 700):
            buf.append(f"line {i}")
        lines, cur = buf.read(cur)
        self.assertEqual(lines, [f"line {i}" for i in range(600, 700)])
        self.assertEqual(cur, 700)

    def test_log_buffer_reader_fell_behind_skips_ahead(self):
        buf = app_module.LogBuffer(maxlen=2000)
        for i in range(2500):
            buf.append(str(i))
        self.assertEqual(buf.total, 2500)
        self.assertEqual(buf.start, 500)
        lines, cur = buf.read(10)          # evicted range -> skip to oldest buffered line
        self.assertEqual(lines[0], "500")
        self.assertEqual(len(lines), 2000)
        self.assertEqual(cur, 2500)
        buf.reset()
        buf.append("new run")
        lines, cur = buf.read(2500)        # cursor from a previous run -> restart
        self.assertEqual(lines, ["new run"])
        self.assertEqual(cur, 1)

    def test_normalize_train_device(self):
        n = app_module.normalize_train_device
        self.assertEqual(n("0"), "0")
        self.assertEqual(n("0,1"), "0,1")
        self.assertEqual(n("0, 1"), "0,1")
        self.assertEqual(n("mps"), "mps")
        self.assertEqual(n("cuda:0"), "0")
        for bad in ("cpu", "", "gpu", "0,,1", None, 3):
            self.assertIsNone(n(bad))

    def test_default_model_classes(self):
        coco = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                "boat", "traffic light"]
        self.assertEqual(app_module.default_model_classes(coco), ["car", "motorcycle", "bus", "truck"])
        self.assertEqual(app_module.model_kind(coco), "coco")
        custom = ["auto_rickshaw", "bike", "car"]
        self.assertEqual(app_module.default_model_classes(custom), custom)
        self.assertEqual(app_module.model_kind(custom), "custom")
        # dataset without classes -> identity over defaults
        self.assertEqual(app_module.default_mapping(coco, []), {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"})
        # dataset with classes -> only case-insensitive matches; dataset classes never change
        self.assertEqual(app_module.default_mapping(coco, ["Car", "bike", "bus"]), {2: "Car", 5: "bus"})

    def test_build_label_lines_remaps_classes(self):
        mapping = {2: "vehicle", 7: "vehicle", 3: "bike"}
        classes = ["person", "bike", "vehicle"]
        dets = [
            (2, (0.1, 0.1, 0.3, 0.3), None),   # car -> vehicle (2)
            (7, (0.5, 0.5, 0.9, 0.9), None),   # truck -> vehicle (2)
            (3, (0.0, 0.0, 0.2, 0.2), None),   # motorcycle -> bike (1)
            (0, (0.4, 0.4, 0.6, 0.6), None),   # person: not mapped -> dropped
            (2, (0.1, 0.1, 0.3, 0.3), None),   # exact duplicate -> dropped
        ]
        lines = app_module.build_label_lines(dets, mapping, classes, "bbox")
        self.assertEqual([l.split()[0] for l in lines], ["2", "2", "1"])
        self.assertEqual(lines[0], "2 0.200000 0.200000 0.200000 0.200000")
        # polygon dataset, detection-only model -> 4-point polygon
        poly = app_module.build_label_lines([(3, (0.1, 0.2, 0.3, 0.4), None)], mapping, classes, "polygon")
        self.assertEqual(poly, ["1 0.100000 0.200000 0.300000 0.200000 0.300000 0.400000 0.100000 0.400000"])
        # polygon dataset with a mask segment
        seg = app_module.build_label_lines([(3, (0, 0, 1, 1), [(0.1, 0.1), (0.5, 0.1), (0.3, 0.6)])],
                                           mapping, classes, "polygon")
        self.assertEqual(len(seg[0].split()), 7)
        # empty mapping -> nothing
        self.assertEqual(app_module.build_label_lines(dets, {}, classes), [])

    def test_resize_dataset_transforms(self):
        from resize_dataset import transform_yolo_letterbox
        orig_w, orig_h, target_w, target_h = 1920, 1080, 640, 640
        scale = min(target_w / orig_w, target_h / orig_h)
        pad_x = (target_w - orig_w * scale) / 2.0
        pad_y = (target_h - orig_h * scale) / 2.0
        out = transform_yolo_letterbox(["0 0.5 0.5 0.2 0.2\n"], scale, pad_x, pad_y,
                                       orig_w, orig_h, target_w, target_h)
        parts = out[0].split()
        self.assertEqual(parts[0], "0")
        self.assertAlmostEqual(float(parts[1]), 0.5, places=3)
        self.assertAlmostEqual(float(parts[2]), 0.5, places=3)

    def test_compress_dataset(self):
        from compress_dataset import compress_dataset, verify_zip_archive
        with tempfile.TemporaryDirectory(dir=TMP_ROOT) as tmpdir:
            src_dir = os.path.join(tmpdir, "test_dataset")
            os.makedirs(os.path.join(src_dir, "train", "images"), exist_ok=True)
            with open(os.path.join(src_dir, "train", "images", "sample.txt"), "w") as f:
                f.write("test content")
            out_zip = os.path.join(tmpdir, "output.zip")
            self.assertTrue(compress_dataset(src_dir, out_zip, fmt="zip", threads=2, verify=True))
            self.assertTrue(verify_zip_archive(out_zip))


# ─── API ─────────────────────────────────────────────────────────────────────
class TestBasicApi(BaseCase):
    def test_index_route(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_device_info_shape(self):
        data = self.client.get("/api/device_info").get_json()
        for k in ("cuda", "mps", "selected", "has_gpu"):
            self.assertIn(k, data)

    def test_api_models(self):
        data = self.client.get("/api/models").get_json()
        self.assertIn("files", data)
        self.assertTrue(len(data["presets"]) > 0)

    def test_value_error_returns_400(self):
        res = self.client.get("/api/images?split=bogus")
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())
        res = self.client.get("/api/labels/train/..%5C..%5Csecret.txt")
        self.assertEqual(res.status_code, 400)
        res = self.client.post("/api/dataset_page", json={"split": "train", "page": "x"})
        self.assertEqual(res.status_code, 400)

    def test_state_fields(self):
        data = self.client.get("/api/state").get_json()
        self.assertTrue(data["ready"])
        self.assertEqual(data["dataset_dir"], self.ds)
        self.assertEqual(data["dataset_name"], app_module.dataset_display_name(self.ds))
        self.assertEqual(data["base_dir"], os.path.abspath(app_module.BASE_DIR))
        self.assertEqual(data["split_ratios"], [0.7, 0.2, 0.1])
        self.assertIsInstance(data["classes"], list)
        self.assertEqual(data["counts"]["train"], {"total": 3, "annotated": 0, "reviewed": 0})

    def test_db_migration_adds_reviewed_column(self):
        legacy = os.path.join(TMP_ROOT, "dataset_legacy")
        os.makedirs(legacy, exist_ok=True)
        conn = sqlite3.connect(os.path.join(legacy, "state.db"))
        conn.execute("CREATE TABLE images (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, "
                     "src TEXT, split TEXT NOT NULL DEFAULT 'train')")
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO images (name, src, split) VALUES ('old.jpg', '', 'train')")
        conn.execute("INSERT INTO meta VALUES ('classes', ?)", (json.dumps(["x", "y"]),))
        conn.commit()
        conn.close()
        app_module.set_active_dataset(legacy, is_existing=True)
        conn = sqlite3.connect(os.path.join(legacy, "state.db"))
        cols = [r[1] for r in conn.execute("PRAGMA table_info(images)")]
        rows = conn.execute("SELECT name, reviewed FROM images").fetchall()
        conn.close()
        self.assertIn("reviewed", cols)
        self.assertEqual(rows, [("old.jpg", 0)])
        self.assertEqual(app_module._class_names, ["x", "y"])
        self.assertEqual(self.client.get("/api/images?split=train").get_json(),
                         [{"name": "old.jpg", "annotated": False, "reviewed": False}])


class TestReviewWorkflow(BaseCase):
    def test_save_sets_reviewed(self):
        res = self.client.post("/api/save", json={
            "split": "train", "filename": "a.jpg",
            "boxes": [{"cls": 0, "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2},
                      {"cls": 1, "pts": [{"x": 0.1, "y": 0.1}]}]})
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["reviewed"])
        self.assertEqual(body["skipped"], 1)
        with open(self.label_file("train", "a.jpg")) as f:
            self.assertEqual(f.read().strip().splitlines(), ["0 0.500000 0.500000 0.200000 0.200000"])
        imgs = {i["name"]: i for i in self.client.get("/api/images?split=train").get_json()}
        self.assertEqual(imgs["a.jpg"], {"name": "a.jpg", "annotated": True, "reviewed": True})
        self.assertEqual(imgs["b.png"], {"name": "b.png", "annotated": False, "reviewed": False})
        stats = self.client.get("/api/stats").get_json()
        self.assertEqual(stats["train"], {"total": 3, "annotated": 1, "reviewed": 1})

    def test_save_bad_split_400(self):
        res = self.client.post("/api/save", json={"split": "nope", "filename": "a.jpg", "boxes": []})
        self.assertEqual(res.status_code, 400)
        res = self.client.post("/api/save", json={"split": "train", "filename": "", "boxes": []})
        self.assertEqual(res.status_code, 400)

    def test_mark_reviewed_toggle(self):
        with open(self.label_file("train", "a.jpg"), "w") as f:
            f.write("0 0.5 0.5 0.1 0.1\n")
        res = self.client.post("/api/mark_reviewed", json={"split": "train", "filename": "a.jpg", "reviewed": True})
        self.assertEqual(res.get_json()["reviewed"], True)
        with open(self.label_file("train", "a.jpg")) as f:
            self.assertEqual(f.read(), "0 0.5 0.5 0.1 0.1\n")   # labels not rewritten
        res = self.client.post("/api/mark_reviewed", json={"split": "train", "filename": "a.jpg", "reviewed": False})
        self.assertFalse(res.get_json()["reviewed"])
        # pending image marked reviewed -> empty label file (reviewed background image)
        res = self.client.post("/api/mark_reviewed", json={"split": "train", "filename": "b.png", "reviewed": True})
        self.assertTrue(res.get_json()["annotated"])
        self.assertEqual(os.path.getsize(self.label_file("train", "b.png")), 0)
        res = self.client.post("/api/mark_reviewed", json={"split": "train", "filename": "zzz.jpg", "reviewed": True})
        self.assertEqual(res.status_code, 404)

    def test_dataset_page_filters(self):
        with open(self.label_file("train", "a.jpg"), "w") as f:
            f.write("0 0.5 0.5 0.1 0.1\nbroken line\n")
        with open(self.label_file("train", "b.png"), "w") as f:
            f.write("1 0.5 0.5 0.1 0.1\n")
        app_module.db_set_reviewed("b.png", "train", True)

        def names(filt, **kw):
            body = dict({"split": "train", "filter": filt}, **kw)
            return [i["name"] for i in self.client.post("/api/dataset_page", json=body).get_json()["images"]]
        self.assertEqual(names("all"), ["a.jpg", "b.png", "c.webp"])
        self.assertEqual(names("pending"), ["c.webp"])
        self.assertEqual(names("auto"), ["a.jpg"])
        self.assertEqual(names("reviewed"), ["b.png"])
        self.assertEqual(names("done"), ["a.jpg", "b.png"])
        self.assertEqual(names("all", class_id=1), ["b.png"])
        self.assertEqual(names("all", class_ids=[0]), ["a.jpg"])
        self.assertEqual(names("all", class_ids=[0, 1]), ["a.jpg", "b.png"])
        self.assertEqual(names("all", class_ids=[]), ["a.jpg", "b.png", "c.webp"])
        page = self.client.post("/api/dataset_page", json={"split": "train", "filter": "reviewed"}).get_json()
        self.assertTrue(page["images"][0]["reviewed"])
        self.assertEqual(len(page["images"][0]["boxes"]), 1)
        self.assertEqual(self.client.post("/api/dataset_page", json={"filter": "weird"}).status_code, 400)

    def test_delete_images_batch(self):
        res = self.client.post("/api/delete_images", json={"split": "train", "filenames": ["a.jpg", "b.png"]})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["deleted_count"], 2)
        self.assertFalse(os.path.exists(os.path.join(self.ds, "train", "images", "a.jpg")))
        self.assertFalse(os.path.exists(self.label_file("train", "a.jpg")))
        self.assertFalse(os.path.exists(os.path.join(self.ds, "train", "images", "b.png")))
        self.assertEqual(self.client.post("/api/delete_images", json={"split": "train", "filenames": []}).status_code, 400)
        self.assertEqual(self.client.post("/api/delete_images", json={"split": "invalid", "filenames": ["c.webp"]}).status_code, 400)


class TestClassesAndSettings(BaseCase):
    def test_classes_post(self):
        res = self.client.post("/api/classes", json={"classes": [" car ", "bike", "", "bus"]})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), {"classes": ["car", "bike", "bus"]})
        self.assertEqual(self.client.get("/api/classes").get_json()["classes"], ["car", "bike", "bus"])
        self.assertEqual(json.loads(app_module.db_get_meta("classes")), ["car", "bike", "bus"])
        import yaml
        with open(os.path.join(self.ds, "data.yaml"), encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        self.assertEqual(doc["names"], ["car", "bike", "bus"])
        self.assertEqual(doc["nc"], 3)
        for bad in ({"classes": []}, {"classes": ["a", "a"]}, {"classes": ["a", 3]}, {"x": 1}, {"classes": ["  "]}):
            self.assertEqual(self.client.post("/api/classes", json=bad).status_code, 400, bad)

    def test_classes_load_order_yaml_then_empty(self):
        ds = os.path.join(TMP_ROOT, "dataset_yaml_only")
        os.makedirs(ds, exist_ok=True)
        with open(os.path.join(ds, "data.yaml"), "w") as f:   # legacy python-repr names
            f.write("train: ../train/images\nnc: 2\nnames: ['alpha', 'beta']\n")
        app_module.set_active_dataset(ds, is_existing=True)
        self.assertEqual(app_module._class_names, ["alpha", "beta"])
        make_dataset()
        self.assertEqual(app_module._class_names, [])

    def test_settings_roundtrip_and_auto_annotate_off(self):
        data = self.client.get("/api/settings").get_json()
        for k in ("auto_annotate", "conf", "iou", "model_path", "model_loaded", "device", "dataset_name"):
            self.assertIn(k, data)
        self.assertFalse(data["model_loaded"])
        res = self.client.post("/api/settings", json={"auto_annotate": False, "conf": 0.4, "iou": 0.55})
        self.assertEqual(res.status_code, 200)
        data = self.client.get("/api/settings").get_json()
        self.assertEqual((data["auto_annotate"], data["conf"], data["iou"]), (False, 0.4, 0.55))
        self.assertEqual(data["dataset_name"], app_module.dataset_display_name(self.ds))

        called = []
        with mock.patch.object(app_module, "annotate_one", side_effect=lambda *a: called.append(a)):
            res = self.client.get("/api/labels/train/a.jpg")
        self.assertEqual(res.get_json(), [])
        self.assertEqual(called, [])
        self.assertFalse(os.path.exists(self.label_file("train", "a.jpg")))

    def test_settings_validation(self):
        for bad in ({"conf": 0}, {"conf": 1.5}, {"conf": "0.3"}, {"iou": 0.05}, {"iou": 0.99},
                    {"auto_annotate": "yes"}, {"conf": True}):
            self.assertEqual(self.client.post("/api/settings", json=bad).status_code, 400, bad)
        self.assertEqual(self.client.post("/api/settings", data="nope",
                                          content_type="application/json").status_code, 400)

    def test_settings_persist_per_dataset(self):
        first = self.ds
        self.client.post("/api/settings", json={"auto_annotate": False, "conf": 0.33, "iou": 0.45})
        make_dataset()   # switch to another dataset -> defaults
        data = self.client.get("/api/settings").get_json()
        self.assertEqual((data["auto_annotate"], data["conf"], data["iou"]),
                         (True, app_module.CONF_THRESH, app_module.DEFAULT_IOU))
        app_module.set_active_dataset(first, is_existing=True)   # switch back -> restored
        data = self.client.get("/api/settings").get_json()
        self.assertEqual((data["auto_annotate"], data["conf"], data["iou"]), (False, 0.33, 0.45))
        app_module._auto_annotate = False


class TestLaunchMode(unittest.TestCase):
    def _args(self, **kw):
        import argparse
        base = dict(launch=None, no_gui=False, browser=False, window=False)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_precedence(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANNOTATOR_LAUNCH", None); os.environ.pop("NO_GUI", None)
            with mock.patch.object(app_module.config, "get", return_value="browser"):
                self.assertEqual(app_module._resolve_launch_mode(self._args()), "browser")
                self.assertEqual(app_module._resolve_launch_mode(self._args(window=True)), "window")
                self.assertEqual(app_module._resolve_launch_mode(self._args(no_gui=True)), "none")
                self.assertEqual(app_module._resolve_launch_mode(self._args(launch="app", no_gui=True)), "app")
            with mock.patch.object(app_module.config, "get", return_value="window"):
                self.assertEqual(app_module._resolve_launch_mode(self._args()), "window")
                self.assertEqual(app_module._resolve_launch_mode(self._args(browser=True)), "browser")
                os.environ["ANNOTATOR_LAUNCH"] = "none"
                self.assertEqual(app_module._resolve_launch_mode(self._args()), "none")
                os.environ["ANNOTATOR_LAUNCH"] = "bogus"
                self.assertEqual(app_module._resolve_launch_mode(self._args()), "browser")
            os.environ.pop("ANNOTATOR_LAUNCH", None)

    def test_port_helpers(self):
        import socket
        s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen(1)
        busy = s.getsockname()[1]
        try:
            self.assertFalse(app_module._port_is_free("127.0.0.1", busy))
            self.assertFalse(app_module._server_responds(f"http://127.0.0.1:{busy}", timeout=0.5))
        finally:
            s.close()


class TestNoImplicitModelDownload(BaseCase):
    def test_no_model_means_no_auto_label_and_no_download(self):
        app_module._model_explicit = False
        app_module._auto_annotate = True
        called = []
        with mock.patch.object(app_module, "annotate_one", side_effect=lambda *a: called.append(a)):
            res = self.client.get("/api/labels/train/a.jpg")
            self.assertEqual(res.get_json(), [])
            self.assertFalse(os.path.exists(self.label_file("train", "a.jpg")))
            r1 = self.client.post("/api/relabel", json={"split": "train", "filename": "a.jpg"})
            r2 = self.client.post("/api/relabel_all", json={"split": "train"})
        self.assertEqual(called, [])
        self.assertEqual((r1.status_code, r2.status_code), (400, 400))
        self.assertIn("load a model", r1.get_json()["error"])
        settings = self.client.get("/api/settings").get_json()
        self.assertFalse(settings["model_available"])


class TestModelClasses(BaseCase):
    COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light"]

    def load_fake_model(self, names, classes=None):
        app_module._class_names = list(classes or [])
        app_module._model_names = list(names)
        app_module.establish_model_mapping()

    def test_get_shape_and_defaults(self):
        self.load_fake_model(self.COCO)
        data = self.client.get("/api/model_classes").get_json()
        for k in ("loaded", "model_path", "model_classes", "mapping", "selected", "default", "kind",
                  "warnings", "classes", "is_large"):
            self.assertIn(k, data)
        self.assertEqual(data["selected"], ["car", "motorcycle", "bus", "truck"])
        self.assertEqual(data["default"], ["car", "motorcycle", "bus", "truck"])
        self.assertEqual(data["kind"], "coco")
        self.assertEqual(data["classes"], ["car", "motorcycle", "bus", "truck"])
        self.assertEqual(data["mapping"][0], {"model_idx": 2, "name": "car", "target": "car"})
        self.assertFalse(data["is_large"])

    def test_post_appends_and_merges(self):
        self.load_fake_model(self.COCO, classes=["bike"])
        res = self.client.post("/api/model_classes", json={"mapping": [
            {"model_idx": 3, "target": "bike"},
            {"model_idx": 2, "target": " vehicle "},
            {"model_idx": 7, "target": "vehicle"},
            {"model_idx": 0, "target": "person"},
        ]})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["classes"], ["bike", "vehicle", "person"])   # appended, never reindexed
        self.assertEqual(app_module._model_class_map, {3: "bike", 2: "vehicle", 7: "vehicle", 0: "person"})
        self.assertEqual(json.loads(app_module.db_get_meta("classes")), ["bike", "vehicle", "person"])
        saved = json.loads(app_module.db_get_meta("model_class_map"))
        self.assertIn({"model_idx": 7, "name": "truck", "target": "vehicle"}, saved)
        # reopening the dataset with the same model restores the saved mapping
        app_module._model_class_map = {}
        app_module.set_active_dataset(self.ds, is_existing=True)
        self.assertEqual(app_module._model_class_map, {3: "bike", 2: "vehicle", 7: "vehicle", 0: "person"})

    def test_post_validation(self):
        self.load_fake_model(self.COCO)
        res = self.client.post("/api/model_classes", json={"mapping": []})
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.get_json()["error"], "Keep at least one class selected")
        for bad in ([{"model_idx": 99, "target": "x"}], [{"model_idx": -1, "target": "x"}],
                    [{"model_idx": "2", "target": "x"}], [{"model_idx": 2, "target": "  "}]):
            self.assertEqual(self.client.post("/api/model_classes", json={"mapping": bad}).status_code, 400, bad)
        app_module._model_names = []
        self.assertEqual(self.client.post("/api/model_classes",
                                          json={"mapping": [{"model_idx": 0, "target": "x"}]}).status_code, 400)

    def test_saved_mapping_warns_on_unknown_names(self):
        app_module.db_set_meta("model_class_map", json.dumps([{"model_idx": 0, "name": "ghost", "target": "car"},
                                                              {"model_idx": 1, "name": "car", "target": "car"}]))
        app_module.db_set_meta("model_class_map_model", app_module._model_path)
        self.load_fake_model(["person", "car"], classes=["car"])
        self.assertEqual(app_module._model_class_map, {1: "car"})
        self.assertTrue(any("ghost" in w for w in app_module._model_map_warnings))


class TestRelabel(BaseCase):
    def fake_annotate(self, img, lbl):
        with open(lbl, "w") as f:
            f.write("0 0.5 0.5 0.3 0.3\n")
        return True

    def test_relabel_refuses_reviewed_unless_forced(self):
        self.client.post("/api/save", json={"split": "train", "filename": "a.jpg",
                                            "boxes": [{"cls": 1, "x": 0.2, "y": 0.2, "w": 0.1, "h": 0.1}]})
        with mock.patch.object(app_module, "annotate_one", side_effect=self.fake_annotate) as m:
            res = self.client.post("/api/relabel", json={"split": "train", "filename": "a.jpg"})
            self.assertEqual(res.status_code, 409)
            self.assertIn("reviewed", res.get_json()["error"])
            m.assert_not_called()
            res = self.client.post("/api/relabel", json={"split": "train", "filename": "a.jpg", "force": True})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), [{"id": 0, "cls": 0, "x": 0.5, "y": 0.5, "w": 0.3, "h": 0.3}])
        self.assertFalse(app_module.db_is_reviewed("a.jpg", "train"))

    def test_relabel_all_skips_reviewed(self):
        self.client.post("/api/save", json={"split": "train", "filename": "a.jpg", "boxes": []})
        with mock.patch.object(app_module, "annotate_one", side_effect=self.fake_annotate) as m:
            res = self.client.post("/api/relabel_all", json={"split": "train"})
            job = res.get_json()["job_id"]
            stream = self.client.get(f"/api/progress/{job}").get_data(as_text=True)
        self.assertIn('"status": "done"', stream)
        touched = sorted(os.path.basename(c.args[0]) for c in m.call_args_list)
        self.assertEqual(touched, ["b.png", "c.webp"])
        self.assertEqual(os.path.getsize(self.label_file("train", "a.jpg")), 0)   # reviewed untouched


class TestIngestAndProgress(BaseCase):
    def run_job(self, res):
        self.assertEqual(res.status_code, 200, res.get_data(as_text=True))
        job = res.get_json()["job_id"]
        events = [json.loads(l[6:]) for l in self.client.get(f"/api/progress/{job}").get_data(as_text=True)
                  .splitlines() if l.startswith("data: ")]
        return events[-1]

    def test_ingest_merge_does_not_duplicate(self):
        src = os.path.join(TMP_ROOT, "src_frames")
        for n in ("f1.jpg", "f2.jpg", "f3.png"):
            make_image(os.path.join(src, n))
        last = self.run_job(self.client.post("/api/ingest", json={"source_dir": src,
                                                                  "split_ratios": [0.6, 0.3, 0.1]}))
        self.assertEqual(last["status"], "done")
        self.assertEqual(app_module.OUTPUT_DIR, os.path.join(TMP_ROOT, "dataset_src_frames"))
        self.assertEqual(self.client.get("/api/state").get_json()["split_ratios"], [0.6, 0.3, 0.1])
        # simulate an auto-split moving one image to valid, then re-ingest the same folder
        out = app_module.OUTPUT_DIR
        shutil.move(os.path.join(out, "train", "images", "f1.jpg"), os.path.join(out, "valid", "images", "f1.jpg"))
        app_module.db_update_split("f1.jpg", "train", "valid")
        make_image(os.path.join(src, "f4.jpg"))
        last = self.run_job(self.client.post("/api/ingest", json={"source_dir": src}))
        self.assertEqual((last["status"], last["added"], last["skipped"]), ("done", 1, 3))
        rows = app_module.db_all_entries()
        self.assertEqual(sorted(r["name"] for r in rows), ["f1.jpg", "f2.jpg", "f3.png", "f4.jpg"])
        self.assertFalse(os.path.exists(os.path.join(out, "train", "images", "f1.jpg")))

    def test_ingest_append_skips_names_in_other_splits(self):
        src = os.path.join(TMP_ROOT, "src_append")
        make_image(os.path.join(src, "d.jpg"))       # already in valid
        make_image(os.path.join(src, "new.jpg"))
        last = self.run_job(self.client.post("/api/ingest_append", json={"source_dir": src, "split": "test"}))
        self.assertEqual((last["added"], last["skipped"]), (1, 1))
        names = [e["name"] for e in app_module.db_get_entries("test")]
        self.assertEqual(names, ["e.jpg", "new.jpg"])

    def test_ingest_invalid_input(self):
        self.assertEqual(self.client.post("/api/ingest", json={"source_dir": "Z:/nope/missing"}).status_code, 400)
        self.assertEqual(self.client.post("/api/ingest", json={"source_dir": TMP_ROOT,
                                                               "split_ratios": [0.5, 0.5, 0.5]}).status_code, 400)
        self.assertEqual(self.client.post("/api/ingest_append", json={"source_dir": TMP_ROOT,
                                                                      "split": "bad"}).status_code, 400)

    def test_ingest_worker_error_sets_status(self):
        src = os.path.join(TMP_ROOT, "src_err")
        make_image(os.path.join(src, "x.jpg"))
        with mock.patch.object(app_module, "db_insert_entries", side_effect=RuntimeError("disk full")):
            last = self.run_job(self.client.post("/api/ingest", json={"source_dir": src}))
        self.assertEqual(last["status"], "error: disk full")

    def test_progress_unknown_job_terminates(self):
        body = self.client.get("/api/progress/does-not-exist").get_data(as_text=True)
        self.assertEqual(body.strip(), 'data: {"status": "error: unknown job"}')


class TestHealthExportMedia(BaseCase):
    def test_health_reports_malformed_lines(self):
        with open(self.label_file("train", "a.jpg"), "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\nnot a label\n1 0.5\n")
        with open(self.label_file("train", "b.png"), "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")
        res = self.client.get("/api/dataset_health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["malformed"], {"count": 1, "files": ["a.jpg"]})
        self.assertEqual(data["total_boxes"], 2)
        for k in ("score", "total_images", "corrupt_images", "empty_images", "small_boxes", "oob_boxes"):
            self.assertIn(k, data)
        self.assertEqual(self.client.get("/api/class_stats").status_code, 200)
        labels = self.client.get("/api/labels/train/a.jpg").get_json()
        self.assertEqual(len(labels), 1)

    def test_yolo_export_excludes_state_files(self):
        app_module.write_yaml()
        for junk in ("state.db-wal", "state.db-shm", "state.json.migrated", "notes.tmp"):
            with open(os.path.join(self.ds, junk), "w") as f:
                f.write("x")
        with open(os.path.join(self.ds, "train", "images", "readme.txt"), "w") as f:
            f.write("not an image")
        with open(self.label_file("train", "a.jpg"), "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")
        res = self.client.get("/api/export?format=yolo")
        self.assertEqual(res.status_code, 200)
        names = zipfile.ZipFile(io.BytesIO(res.get_data())).namelist()
        res.close()
        self.assertIn("data.yaml", names)
        self.assertIn("train/images/a.jpg", names)
        self.assertIn("train/labels/a.txt", names)
        self.assertFalse([n for n in names if "state" in n or n.endswith(".tmp") or n.endswith("readme.txt")])

    def test_coco_export_only_images(self):
        with open(os.path.join(self.ds, "train", "images", "readme.txt"), "w") as f:
            f.write("not an image")
        res = self.client.get("/api/export?format=coco")
        zf = zipfile.ZipFile(io.BytesIO(res.get_data()))
        res.close()
        coco = json.loads(zf.read("train/_annotations.coco.json"))
        self.assertEqual(sorted(i["file_name"] for i in coco["images"]), ["a.jpg", "b.png", "c.webp"])

    def test_image_and_thumb_mimetypes(self):
        for name, mt in (("a.jpg", "image/jpeg"), ("b.png", "image/png"), ("c.webp", "image/webp")):
            res = self.client.get(f"/api/image/train/{name}")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.mimetype, mt)
            res.close()
            res = self.client.get(f"/api/thumb/train/{name}")
            self.assertEqual(res.mimetype, "image/jpeg")
            self.assertIn("max-age=3600", res.headers.get("Cache-Control", ""))
            res.close()
        self.assertEqual(self.client.get("/api/image/train/missing.jpg").status_code, 404)

    def test_delete_image(self):
        res = self.client.post("/api/delete_image", json={"split": "train", "filename": "a.jpg"})
        self.assertEqual(res.get_json(), {"status": "deleted"})
        self.assertNotIn("a.jpg", [e["name"] for e in app_module.db_get_entries("train")])
        self.assertEqual(self.client.post("/api/delete_image", json={"split": "x", "filename": "a"}).status_code, 400)


class TestClassManagement(BaseCase):
    """Class merge / delete / rename / add / restore with label rewriting."""

    def setUp(self):
        super().setUp()
        self.client.post("/api/classes", json={"classes": ["car", "bike", "bus", "truck"]})
        self.write("train", "a.jpg", "0 0.1 0.1 0.1 0.1\n1 0.2 0.2 0.1 0.1\n2 0.3 0.3 0.1 0.1\n"
                                     "3 0.4 0.4 0.1 0.1\n9 0.5 0.5 0.1 0.1\ngarbage\n")
        self.write("train", "b.png", "3 0.1 0.2 0.3 0.1 0.2 0.3\n")          # polygon, class truck
        self.write("valid", "d.jpg", "1 0.5 0.5 0.2 0.2\n")

    def write(self, split, name, text):
        with open(self.label_file(split, name), "w") as f:
            f.write(text)

    def read(self, split, name):
        with open(self.label_file(split, name)) as f:
            return f.read().splitlines()

    def ids(self, split, name):
        return [l.split()[0] for l in self.read(split, name)]

    def test_usage(self):
        data = self.client.get("/api/classes/usage").get_json()
        self.assertEqual(data["classes"], [
            {"id": 0, "name": "car", "boxes": 1, "images": 1},
            {"id": 1, "name": "bike", "boxes": 2, "images": 2},
            {"id": 2, "name": "bus", "boxes": 1, "images": 1},
            {"id": 3, "name": "truck", "boxes": 2, "images": 2},
        ])
        self.assertEqual(data["unknown_ids"], [{"id": 9, "boxes": 1}])

    def test_rename_and_add(self):
        res = self.client.post("/api/classes/rename", json={"id": 1, "name": " motorbike "})
        self.assertEqual(res.get_json()["classes"], ["car", "motorbike", "bus", "truck"])
        self.assertEqual(self.ids("train", "a.jpg"), ["0", "1", "2", "3", "9", "garbage"])   # untouched
        self.assertFalse(os.path.exists(os.path.join(self.ds, "backups")))                  # no backup
        self.assertEqual(self.client.post("/api/classes/rename", json={"id": 1, "name": "car"}).status_code, 400)
        self.assertEqual(self.client.post("/api/classes/rename", json={"id": 1, "name": " "}).status_code, 400)
        self.assertEqual(self.client.post("/api/classes/rename", json={"id": 7, "name": "x"}).status_code, 400)
        res = self.client.post("/api/classes/add", json={"name": "van"})
        self.assertEqual(res.get_json()["classes"], ["car", "motorbike", "bus", "truck", "van"])
        self.assertEqual(self.client.post("/api/classes/add", json={"name": "van"}).status_code, 400)

    def test_merge_shifts_ids(self):
        # merge bus(2) + car(0) into truck(3) -> classes [bike, truck]; bike 1->0, truck 3->1
        res = self.client.post("/api/classes/merge", json={"sources": [2, 0], "target": 3})
        self.assertEqual(res.status_code, 200, res.get_data(as_text=True))
        body = res.get_json()
        self.assertEqual(body["classes"], ["bike", "truck"])
        self.assertEqual((body["changed_files"], body["changed_boxes"]), (3, 6))
        self.assertTrue(body["backup"].startswith("backups/labels_"))
        self.assertTrue(os.path.isfile(os.path.join(self.ds, body["backup"])))
        self.assertEqual(self.ids("train", "a.jpg"), ["1", "0", "1", "1", "9", "garbage"])
        self.assertEqual(self.read("train", "a.jpg")[0], "1 0.1 0.1 0.1 0.1")   # coords kept verbatim
        self.assertEqual(self.read("train", "b.png"), ["1 0.1 0.2 0.3 0.1 0.2 0.3"])
        self.assertEqual(self.ids("valid", "d.jpg"), ["0"])
        self.assertEqual(json.loads(app_module.db_get_meta("classes")), ["bike", "truck"])
        import yaml
        with open(os.path.join(self.ds, "data.yaml"), encoding="utf-8") as f:
            self.assertEqual(yaml.safe_load(f)["names"], ["bike", "truck"])

    def test_merge_validation(self):
        for bad in ({"sources": [], "target": 1}, {"sources": [1], "target": 1}, {"sources": [9], "target": 1},
                    {"sources": [0, 0], "target": 1}, {"sources": [0], "target": 8}, {"target": 1}):
            self.assertEqual(self.client.post("/api/classes/merge", json=bad).status_code, 400, bad)

    def test_merge_updates_model_mapping_targets(self):
        app_module._model_names = ["person", "car", "truck", "bus"]
        app_module._model_class_map = {1: "car", 2: "truck", 3: "bus"}
        self.client.post("/api/classes/merge", json={"sources": [2], "target": 3})   # bus -> truck
        self.assertEqual(app_module._model_class_map, {1: "car", 2: "truck", 3: "truck"})
        self.client.post("/api/classes/delete", json={"id": 0, "boxes": "remove"})    # car removed
        self.assertEqual(app_module._model_class_map, {2: "truck", 3: "truck"})

    def test_delete_remove_and_merge(self):
        res = self.client.post("/api/classes/delete", json={"id": 1, "boxes": "remove"})   # drop bike
        body = res.get_json()
        self.assertEqual(body["classes"], ["car", "bus", "truck"])
        # a.jpg: bike box removed + bus/truck shifted (3); b.png: truck shifted (1); d.jpg: bike removed (1)
        self.assertEqual((body["changed_files"], body["changed_boxes"]), (3, 5))
        self.assertEqual(self.ids("train", "a.jpg"), ["0", "1", "2", "9", "garbage"])
        self.assertEqual(self.read("valid", "d.jpg"), [])
        self.assertEqual(self.ids("train", "b.png"), ["2"])
        res = self.client.post("/api/classes/delete", json={"id": 0, "boxes": "merge", "merge_into": 2})
        self.assertEqual(res.get_json()["classes"], ["bus", "truck"])
        self.assertEqual(self.ids("train", "a.jpg"), ["1", "0", "1", "9", "garbage"])
        for bad in ({"id": 0, "boxes": "merge", "merge_into": 0}, {"id": 0, "boxes": "zap"}, {"id": 5}):
            self.assertEqual(self.client.post("/api/classes/delete", json=bad).status_code, 400, bad)

    def test_backup_and_restore(self):
        original = {k: self.read(*k) for k in (("train", "a.jpg"), ("train", "b.png"), ("valid", "d.jpg"))}
        backup = self.client.post("/api/classes/merge", json={"sources": [0, 1], "target": 2}).get_json()["backup"]
        self.write("train", "c.webp", "0 0.5 0.5 0.1 0.1\n")            # label created after the backup
        listing = self.client.get("/api/classes/backups").get_json()["backups"]
        self.assertEqual(listing[0]["name"], os.path.basename(backup))
        self.assertEqual(listing[0]["classes"], ["car", "bike", "bus", "truck"])
        self.assertEqual(listing[0]["files"], 3)
        res = self.client.post("/api/classes/restore", json={"backup": os.path.basename(backup)})
        self.assertEqual(res.status_code, 200, res.get_data(as_text=True))
        body = res.get_json()
        self.assertEqual(body["classes"], ["car", "bike", "bus", "truck"])
        self.assertEqual(app_module._class_names, ["car", "bike", "bus", "truck"])
        self.assertEqual((body["restored_files"], body["removed_files"]), (3, 1))
        for k, lines in original.items():
            self.assertEqual(self.read(*k), lines)
        self.assertFalse(os.path.exists(self.label_file("train", "c.webp")))
        self.assertTrue(os.path.isfile(os.path.join(self.ds, body["backup"])))   # pre-restore backup
        self.assertEqual(self.client.post("/api/classes/restore", json={"backup": "../../x.zip"}).status_code, 400)
        self.assertEqual(self.client.post("/api/classes/restore",
                                          json={"backup": "labels_20000101-000000.zip"}).status_code, 404)

    def test_restore_follows_split_moves(self):
        backup = self.client.post("/api/classes/delete", json={"id": 3, "boxes": "remove"}).get_json()["backup"]
        # image a.jpg moves to test after the backup (e.g. auto-split)
        shutil.move(os.path.join(self.ds, "train", "images", "a.jpg"), os.path.join(self.ds, "test", "images", "a.jpg"))
        shutil.move(self.label_file("train", "a.jpg"), self.label_file("test", "a.jpg"))
        app_module.db_update_split("a.jpg", "train", "test")
        self.client.post("/api/classes/restore", json={"backup": os.path.basename(backup)})
        self.assertEqual(self.ids("test", "a.jpg"), ["0", "1", "2", "3", "9", "garbage"])
        self.assertFalse(os.path.exists(self.label_file("train", "a.jpg")))

    def test_backups_keep_last_10(self):
        for i in range(12):
            self.client.post("/api/classes/add", json={"name": f"tmp{i}"})
            self.client.post("/api/classes/delete", json={"id": 4, "boxes": "remove"})
        names = [b for b in os.listdir(os.path.join(self.ds, "backups")) if b.endswith(".zip")]
        self.assertEqual(len(names), 10)
        listing = [b["name"] for b in self.client.get("/api/classes/backups").get_json()["backups"]]
        self.assertEqual(listing, sorted(names, key=app_module._backup_sort_key, reverse=True))
        # the newest backup reflects the latest state (13 classes added/removed -> last add was tmp11)
        self.assertIn("tmp11", self.client.get("/api/classes/backups").get_json()["backups"][0]["classes"])

    def test_conflict_while_job_running(self):
        job = app_module.new_job(_kind="relabel_all")
        try:
            res = self.client.post("/api/classes/merge", json={"sources": [0], "target": 1})
            self.assertEqual(res.status_code, 409)
            self.assertEqual(self.client.post("/api/classes/delete", json={"id": 0}).status_code, 409)
        finally:
            app_module.finish_job(job)
        self.assertEqual(self.client.post("/api/classes/merge", json={"sources": [0], "target": 1}).status_code, 200)

    def test_legacy_classes_post_rename_or_append_only(self):
        ok = self.client.post("/api/classes", json={"classes": ["car", "motorbike", "bus", "truck"]})
        self.assertEqual(ok.status_code, 200)
        ok = self.client.post("/api/classes", json={"classes": ["car", "motorbike", "bus", "truck", "van"]})
        self.assertEqual(ok.status_code, 200)
        bad = self.client.post("/api/classes", json={"classes": ["car", "bus"]})
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(bad.get_json()["error"], "Use merge/delete to remove classes")


class TestDatasetList(BaseCase):
    def test_datasets_stats_readonly_and_old_db(self):
        with open(self.label_file("train", "a.jpg"), "w") as f:
            f.write("0 0.5 0.5 0.1 0.1\n")
        self.client.post("/api/save", json={"split": "train", "filename": "b.png", "boxes": []})
        self.client.post("/api/classes", json={"classes": ["x", "y"]})
        # a v1.0-style dataset (no `reviewed` column), must not be modified by listing
        old = os.path.join(TMP_ROOT, "dataset_oldschema")
        os.makedirs(os.path.join(old, "train", "labels"), exist_ok=True)
        db = os.path.join(old, "state.db")
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE images (id INTEGER PRIMARY KEY, name TEXT, src TEXT, split TEXT)")
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.executemany("INSERT INTO images (name, src, split) VALUES (?, '', 'train')", [("p.jpg",), ("q.jpg",)])
        conn.execute("INSERT INTO meta VALUES ('dataset_type', 'polygon')")
        conn.commit()
        conn.close()
        with open(os.path.join(old, "train", "labels", "p.txt"), "w") as f:
            f.write("0 0.1 0.1 0.2 0.1 0.2 0.2\n")
        with open(db, "rb") as f:
            before = f.read()
        mtime = os.path.getmtime(db)

        data = {d["name"]: d for d in self.client.get("/api/datasets?stats=1").get_json()}
        mine = data[os.path.basename(self.ds)]
        self.assertEqual({k: mine[k] for k in ("type", "total", "annotated", "reviewed", "classes", "active")},
                         {"type": "bbox", "total": 5, "annotated": 2, "reviewed": 1, "classes": 2, "active": True})
        self.assertGreater(mine["modified"], 0)
        o = data["dataset_oldschema"]
        self.assertEqual({k: o[k] for k in ("type", "total", "annotated", "reviewed", "classes", "active")},
                         {"type": "polygon", "total": 2, "annotated": 1, "reviewed": 0, "classes": 0, "active": False})
        with open(db, "rb") as f:
            self.assertEqual(f.read(), before)
        self.assertEqual(os.path.getmtime(db), mtime)
        conn = sqlite3.connect(db)
        self.assertNotIn("reviewed", [r[1] for r in conn.execute("PRAGMA table_info(images)")])
        conn.close()
        # without stats=1 the old shape is kept
        plain = self.client.get("/api/datasets").get_json()
        self.assertEqual(set(plain[0].keys()), {"name", "path"})

        # stats=1 with refresh=1 works and includes locked field
        refreshed = {d["name"]: d for d in self.client.get("/api/datasets?stats=1&refresh=1").get_json()}
        self.assertIn("locked", refreshed[os.path.basename(self.ds)])
        self.assertFalse(refreshed[os.path.basename(self.ds)]["locked"])
        self.assertIn("locked", refreshed["dataset_oldschema"])
        self.assertFalse(refreshed["dataset_oldschema"]["locked"])


class TestTrainAndTools(BaseCase):
    def test_train_cpu_restriction(self):
        res = self.client.post("/api/train", json={"model": "yolov8n.pt", "device": "cpu"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())

    def test_train_custom_model_multi_gpu(self):
        app_module._class_names = ["car"]
        app_module.write_yaml()
        fake = mock.MagicMock()
        fake.stdout = iter(["epoch 1\n", "done\n"])
        fake.poll.return_value = 0
        with mock.patch.object(app_module.subprocess, "Popen", return_value=fake) as popen, \
                mock.patch.object(app_module, "reset_model") as reset:
            res = self.client.post("/api/train", json={"model": "yolov8n.pt", "custom_model": "my_custom.pt",
                                                       "device": "0,1", "epochs": 1})
            self.assertEqual(res.status_code, 200, res.get_data(as_text=True))
            reset.assert_called_once()
            cmd = popen.call_args.args[0]
            kwargs = popen.call_args.kwargs
        self.assertIn("model=my_custom.pt", cmd)
        self.assertIn("device=0,1", cmd)
        self.assertEqual(kwargs["encoding"], "utf-8")
        self.assertEqual(kwargs["env"]["PYTHONIOENCODING"], "utf-8")
        logs = self.client.get("/api/train/logs").get_data(as_text=True)
        self.assertIn("epoch 1", logs)
        self.assertIn('"done": true', logs)
        status = self.client.get("/api/train/status").get_json()
        self.assertEqual(status, {"status": "done", "exit_code": 0, "lines": 2})
        app_module._train_process = None

    def test_tools_status(self):
        data = self.client.get("/api/tools/status?job=resize").get_json()
        for k in ("running", "percent", "status"):
            self.assertIn(k, data)

    def test_tools_resize_uses_temp_dataset(self):
        res = self.client.post("/api/tools/resize", json={"size": 64, "mode": "letterbox", "workers": 1})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), {"status": "started", "job": "resize"})
        expected = f"{self.ds}_64"
        deadline = time.time() + 60
        while time.time() < deadline:
            st = self.client.get("/api/tools/status?job=resize").get_json()
            if not st.get("running"):
                break
            time.sleep(0.1)
        self.assertFalse(st.get("running"), st)
        self.assertEqual(st["out_dir"], expected)
        self.assertTrue(os.path.commonpath([expected, TMP_ROOT]) == TMP_ROOT)


if __name__ == "__main__":
    unittest.main()
