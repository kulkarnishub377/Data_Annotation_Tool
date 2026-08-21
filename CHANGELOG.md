# Changelog

All notable changes to the **Data Annotation & YOLO Studio** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
