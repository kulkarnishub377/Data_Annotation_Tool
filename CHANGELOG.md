# Changelog

All notable changes to the **Data Annotation & YOLO Studio** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-11-05

### Added
- **Reviewed status**: images are Pending / Auto-labelled / Reviewed. Saving marks an image reviewed;
  `POST /api/mark_reviewed` toggles it. Counts in `/api/stats` and `/api/state` include `reviewed`;
  `/api/images` and `/api/dataset_page` items include `reviewed`; grid filters `pending | auto | reviewed | done`.
- **Detection settings per dataset**: `GET/POST /api/settings` (`auto_annotate`, `conf`, `iou`), stored in the
  dataset's `state.db`. With auto-labelling off, opening an image no longer creates a label file.
- **Re-run detection**: `POST /api/relabel` (single image, refuses reviewed images unless `force`) and
  `POST /api/relabel_all` (background job over all images that are not reviewed).
- **Editable classes**: `POST /api/classes`; classes load from dataset meta → `data.yaml` → model.
  No hard-coded default class list any more.
- **Model class selection & mapping**: `GET/POST /api/model_classes`. COCO models default to
  car / motorcycle / bus / truck, custom models to all classes; model classes can be saved as other dataset
  classes (e.g. `motorcycle` → `bike`). Inference only runs for selected classes and labels are remapped to
  dataset class indices. Mapping is saved per dataset and model.
- **Class management**: `GET /api/classes/usage`, `POST /api/classes/{rename,add,merge,delete,restore}`,
  `GET /api/classes/backups`. Merge/delete rewrite label files atomically, shift ids to stay contiguous, update
  `data.yaml` and model-class mapping targets, and save a labels backup zip first (last 10 kept). Refused (409)
  while ingest / re-label / auto-split runs. `POST /api/classes` is now limited to renames and appends.
- `GET /api/datasets?stats=1`: per-dataset type, image / annotated / reviewed counts, class count, last
  modified and active flag, read from each `state.db` strictly read-only (works with v1.0 databases).
- `ANNOTATOR_BASE_DIR` / `ANNOTATOR_MODELS_DIR` / `ANNOTATOR_OUTPUT_DIR` env overrides (beat `config.ini`)
  for pointing test/dev instances at scratch folders.
- `/api/state` returns `dataset_dir`, `dataset_name`, `split_ratios`, `classes`; ingest accepts `split_ratios`.
- `/api/train/status` reports the number of log lines; `/api/train/logs` replays buffered output on reconnect.
- New shortcuts and review UI (class bar, status bar, shortcut sheet, inline class picker) – see README.

### Fixed
- Training log stream stalled after 500 lines (buffer now uses a monotonic line counter, keeps 2000 lines).
- `/api/train` ignored `custom_model`, rejected multi-GPU `"0,1"` and never released the inference model;
  training output is decoded as UTF-8.
- Re-ingesting a folder wiped the database and duplicated images across splits; ingest now merges.
- Ingest worker errors left jobs stuck at "linking"; unknown progress job ids streamed forever.
- `.webp` / `.bmp` images and thumbnails were served as `image/jpeg`; thumbnail cache now keyed by file mtime.
- YOLO export included `state.db-wal` / `-shm`; exports now contain only images, labels and `data.yaml`.
- `data.yaml` `names` written as a Python repr; now a JSON list (valid YAML).
- Dataset health crashed on malformed label lines (now reported under `malformed`).
- Dataset resize/compress jobs read the active dataset path late; paths are captured at request time.
- Polygon datasets with detection-only models now receive 4-point polygons.
- Invalid splits / file names return HTTP 400 JSON instead of 500.

### Changed
- Label files are written atomically; saved boxes are validated and clamped to the image.
- Tests run entirely on temporary datasets (the old resize test modified the real active dataset).

---

## [1.0.0] - 2026-08-22

### 🚀 Initial Production Release (v1.0.0)

#### Core Features
- **High-Performance Web Annotation Tool**: Single Page Application (SPA) architecture with fast HTML5 Canvas rendering, zooming, panning, drag-to-draw, and polygon point manipulation.
- **YOLO & Custom Weights Auto-Inference**: Auto-loads YOLO models (YOLO26, YOLO11, YOLOv8) and local `.pt` weights for automatic real-time pre-annotation with multi-threaded background processing.
- **PyTorch 2.6+ Checkpoint Compatibility**: Automatic safe serialization patch for Ultralytics models under PyTorch 2.6+ `weights_only` defaults.
- **SQLite State Management (WAL Mode)**: Thread-safe high-throughput database backing for image metadata, annotation history, and split records.
- **Multi-Split Dataset Management**: Interactive auto-splitting across `train`, `valid`, and `test` with custom percentage controls.
- **Dataset Health Dashboard**: Comprehensive anomaly detection scanning for corrupt 0-byte images, micro-boxes, out-of-bounds annotations, empty images, and class distribution imbalances.

#### Train & Tools Suite
- **GPU-Accelerated Model Training**: Integrated Ultralytics training with real-time SSE log streaming and strict GPU-only enforcement (CUDA / Apple Silicon MPS).
- **Multi-Threaded Dataset Resizer (`scripts/resize_dataset.py`)**: Parallel dataset dimension conversion (e.g. `640x640`, `1280x1280`) with pixel-accurate letterbox padding and bounding box / polygon coordinate recalculation.
- **Multi-Threaded Dataset Compressor (`scripts/compress_dataset.py`)**: Parallel multi-format dataset archiving (`.zip`, `.7z`, `.tar.gz`, `.tar.xz`) with smart runtime temp file filtering and CRC integrity verification.
- **Dataset Frame Cleanup Utility (`scripts/cleanup.py`)**: Command-line frame range cleanup from database and disk.

#### Web Interface & UI
- **Train & Tools Sub-Tabs**: Integrated sub-tabs for Training, Resizing, and Compression with real-time progress bars.
- **Modal Interactivity**: Add More Images modal, Shortcuts & Help modal, and Dataset Health Dashboard with universal backdrop click dismissals.
- **Zero-Emoji Clean Terminal Standards**: Pure ASCII cross-platform terminal logging across Windows `cp1252` and Linux/macOS `UTF-8`.

#### CI & DevOps
- **Automated Test Matrix**: 14 unit tests passing in 0.08s across Ubuntu, Windows, and macOS on Python 3.10, 3.11, and 3.12.
- **GitHub Community Standards**: Bug report and feature request issue templates, PR template, Code of Conduct, Contributing guide, and Security policy.

---

## [2.0.0] - Planned / Upcoming Roadmap
- AI-assisted polygon auto-segmentation (SAM / MobileSAM).
- Multi-user collaborative annotation sessions.
- Video stream auto-frame extraction and active learning pipeline.
- Export support for COCO, Pascal VOC, and YOLOv10/YOLO26 format extensions.
