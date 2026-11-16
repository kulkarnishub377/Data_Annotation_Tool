# Free Open Source Data Annotation Tool for Computer Vision 🚀

[![Release](https://img.shields.io/badge/Release-v1.1.0-blue?style=for-the-badge&logo=github)](https://github.com/kulkarnishub377/Data_Annotation_Tool/releases/tag/v1.1.0)
[![CI Tests](https://github.com/kulkarnishub377/Data_Annotation_Tool/actions/workflows/ci.yml/badge.svg)](https://github.com/kulkarnishub377/Data_Annotation_Tool/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0+-black?style=for-the-badge&logo=flask)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Open Source](https://img.shields.io/badge/Open_Source-100%25-orange?style=for-the-badge)

**Data Annotation Tool** is a professional-grade, 100% free and open-source web application designed for fast, accurate, and private computer vision labeling. Whether you are building bounding boxes for Object Detection or drawing polygons for Instance Segmentation, this tool provides an end-to-end pipeline: from auto-annotating datasets using YOLO models to reviewing, splitting, and directly training models right from a sleek, intuitive UI.

> **Why choose this over cloud-based tools?** Zero cloud lock-in, zero subscription fees, and absolute privacy. Your datasets and weights never leave your local machine.

---

## 🌟 Key Highlights

- **Dataset Health Dashboard:** Instantly scans your entire dataset for empty annotations, corrupt files, microscopic bounding boxes, and out-of-bounds coordinates to catch training errors early. Filter visually directly from the dashboard!
- **Zero Configuration Setup:** Load datasets instantly via web UI without editing paths in code.
- **In-Browser Training & SSE Live Logs:** Train YOLOv8, YOLO11, and YOLO26 directly from the UI with real-time log streaming.
- **Dynamic Class Shortcuts & Popups:** Instant class assignment via keys `1-9` or inline popup when drawing.
- **High-Precision Export:** Export fully validated datasets in YOLO and COCO formats with exact image dimensions.

---

## 🚀 Key Features

### Powerful Backend (Flask & YOLO Integration)
*   **Multi-Format Inference**: Natively loads `.pt`, `.onnx`, `.engine`, `.tflite`, and `.torchscript` models.
*   **100% Offline Support**: Works entirely locally on your machine. Once dependencies are installed, no internet connection is required.
*   **Bounding Box & Polygon Segmentation**: Fully supports creating and exporting standard YOLO Object Detection (Bounding Box) or YOLO Instance Segmentation (Polygon) datasets.
*   **Integrated Model Downloading**: Easily select from standard YOLOv8, YOLO11, or the next-gen **YOLO26** pre-trained models.
*   **Intelligent Hardware Profiling**: Automatically profiles system hardware and selects the fastest available inference device (CUDA GPU, Apple MPS, or fallback CPU).
*   **Background Pre-Annotation Engine**: Ingests images instantly using fast, space-saving OS hardlinks, while parallel background threads run YOLO inference to auto-annotate ahead of your workflow.
*   **Dataset Export**: Zips and exports your fully annotated dataset and generated `data.yaml` in either YOLO or COCO formats with a single click.

### Advanced Frontend (Vanilla HTML5 / CSS3 / ES6+)
*   **Sleek Setup & Grid View**: A beautiful dark-themed setup screen auto-detects classes and tracks dataset formats. A comprehensive "Grid View" allows dataset-wide inspection, dynamic pagination (20, 50, 100, 200, 500 limits), and status filtering.
*   **Perfect Zoom, Pan & Minimap**: Refined scroll-wheel math for butter-smooth zooming centered on the cursor, middle-click drag for panning, and a dynamic heads-up minimap.
*   **Image Enhancement Tools**: Real-time slider controls for adjusting image brightness, contrast, and extra filters (grayscale, invert, sepia) to spot difficult objects.
*   **Session Tracking**: Topbar session timer tracks your active annotation time.

---

## 🏗️ Architecture & Directory Structure

```text
Data_Annotation_Tool/
├── .github/
│   ├── workflows/
│   │   └── ci.yml               # Automated CI test workflow matrix
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md        # Standardized Bug Report template
│   │   └── feature_request.md   # Standardized Feature Request template
│   ├── PULL_REQUEST_TEMPLATE.md # PR checklist and guidelines
│   ├── CODE_OF_CONDUCT.md       # Contributor Covenant Code of Conduct
│   ├── CONTRIBUTING.md          # Comprehensive contributing guide
│   └── SECURITY.md              # Security & vulnerability reporting policy
├── static/
│   ├── css/
│   │   ├── style.css            # Dark theme styles and design tokens
│   │   └── features.css         # Styles for newer components (class bar, status bar, pickers)
│   ├── js/
│   │   └── app.js               # Frontend canvas math, hotkeys, API & SSE
│   └── assets/                  # Icons and static media
├── scripts/
│   ├── build_exe.py             # PyInstaller desktop build helper
│   ├── cleanup.py               # Dataset frame-range cleanup tool
│   ├── compress_dataset.py      # Multi-threaded dataset archiver
│   └── resize_dataset.py        # Multi-threaded dataset resizer (letterbox / stretch)
├── tests/
│   └── test_app.py              # Automated unit/API test suite (runs on temp data only)
├── .env.example                 # Environment variables template
├── .gitattributes               # Cross-platform line endings & binary formatting
├── .gitignore                   # Comprehensive ignore rules
├── app.py                       # Core Flask backend & YOLO engine
├── CHANGELOG.md                 # Release notes
├── config.example.ini           # System configuration template
├── index.html                   # Main UI Single Page Application markup
├── LICENSE                      # MIT License
├── README.md                    # Project documentation
└── requirements.txt             # Pinned project dependencies
```

---

## 🛠️ Setup & Installation

### Prerequisites
*   **Python 3.10+** (Python 3.12 recommended)
*   **Modern Web Browser** (Chrome / Edge / Firefox / Brave)
*   **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/kulkarnishub377/Data_Annotation_Tool.git
cd Data_Annotation_Tool
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configuration (Optional)

The application automatically defaults to local portable folders. To customize default paths or thresholds, copy `config.example.ini`:

```bash
cp config.example.ini config.ini
```

Path resolution order (first match wins):

| Setting | 1. Override env var | 2. `config.ini` `[Paths]` | 3. Env / `.env` | 4. Default |
| :--- | :--- | :--- | :--- | :--- |
| Workspace (datasets) | `ANNOTATOR_BASE_DIR` | `BASE_DIR` | `BASE_DIR` | parent folder if it has `models/` or `dataset_default/`, else the app folder |
| Models | `ANNOTATOR_MODELS_DIR` | `MODELS_DIR` | `MODELS_DIR` | `<workspace>/models` |
| Startup dataset | `ANNOTATOR_OUTPUT_DIR` | `OUTPUT_DIR` | `OUTPUT_DIR` | `<workspace>/dataset_default` |

The `ANNOTATOR_*` variables are meant for test / development instances: they beat `config.ini` and `.env`, so
you can point a second copy of the app at a scratch folder without touching your real datasets, e.g.

```bash
# PowerShell
$env:ANNOTATOR_BASE_DIR = "D:\scratch\annot"; python app.py --browser --port 8052
# bash
ANNOTATOR_BASE_DIR=/tmp/annot python app.py --browser --port 8052
```

Detection thresholds (confidence, IoU) and the auto-label toggle are saved per dataset from the UI;
`CONF_THRESH` in `config.ini` / `.env` only sets the default for datasets that have not saved their own value.

---

## 🚦 Running the Application

With your virtual environment activated, start the server:

```bash
python app.py
```

It opens **[http://127.0.0.1:8051](http://127.0.0.1:8051)** in your default browser. On Windows you can also double-click `start.bat` (closing its console window stops the app).

How the UI opens is set in `config.ini`:

```ini
[App]
# browser (default) | window (desktop window, needs pywebview) |
# app (Edge/Chrome window without tabs) | none (server only)
LAUNCH_MODE = browser
```

A command-line flag overrides it for one run: `--launch browser|window|app|none` (or the shortcuts `--browser`, `--window`, `--no-gui`). The `ANNOTATOR_LAUNCH` environment variable works too.

If the port is already used by a running copy, the app just opens that copy. If it is held by a frozen program, the app warns you and starts on the next free port.

---

## 🔁 Review Workflow

Every image has one of three states, shown as a coloured dot in the image list and a badge in the grid:

| State | Meaning |
| :--- | :--- |
| **Pending** (grey) | No label file yet. |
| **Auto-labelled** (amber) | A label file exists (written by the model or imported) but nobody has checked it. |
| **Reviewed** (green) | A person saved the image (or pressed `R` to accept it unchanged). |

- The left sidebar has `All · To review · Reviewed` tabs with counts and a `reviewed / total` progress bar.
- `Enter` saves and moves on; `J` jumps to the next image that is not reviewed yet.
- Accepting an image with no objects (`R` on a pending image) stores an empty label file, i.e. a reviewed background image.
- Existing datasets are upgraded automatically: a `reviewed` column is added to `state.db` the first time it is opened (nothing is removed or rewritten).

## ⚙️ Detection Settings, Classes & Model Class Selection

- **Detection settings** (setup Model card and the canvas toolbar `Detection` popover): confidence threshold, overlap (IoU) threshold and an *Auto-label new images* toggle. They are stored per dataset in its `state.db`, so every dataset remembers its own thresholds. Turn auto-labelling off to annotate by hand without a model.
- **Re-run detection** on the current image or on all images that are not reviewed yet (reviewed images are never overwritten).
- **Manage classes**: add, rename, merge (e.g. `truck` + `bus` → `vehicle`) or delete classes, with box/image counts per class. You can label without loading any model. Merge and delete rewrite every label file and renumber the remaining classes so ids stay contiguous; a backup of all labels and the class list is saved first in `<dataset>/backups/` (last 10 kept) and can be restored from the `Backups` list. Classes are saved in the dataset and written to `data.yaml`.
- **Choose classes** for pretrained models: a COCO model (80 classes) defaults to `car, motorcycle, bus, truck`; a custom model defaults to all its classes. Each model class can be saved as a different dataset class (e.g. COCO `motorcycle` → `bike`, `car` + `truck` → `vehicle`). New names are appended to the dataset classes; existing class indices are never renumbered.

## ⌨️ Keyboard Shortcuts

Press `?` in the annotator to see the full, always up-to-date list. Highlights:

| Key | Action |
| :--- | :--- |
| `D` | Toggle draw mode |
| `Ctrl` (hold) | Draw box / polygon |
| `1` – `9` | Set the active class, or reassign the selected shapes |
| `Q` / `E` | Cycle class of the selected shape |
| `Tab` / `Shift+Tab` | Select next / previous shape |
| `Ctrl+A` | Select all shapes |
| `Ctrl+C` / `Ctrl+V` | Copy / paste selected shapes (also across images) |
| `Ctrl+D` | Duplicate selection |
| `C` | Copy all shapes from the previous image |
| `Shift+Arrows` | Nudge selection 1 px (`Alt+Shift` = 10 px) |
| `Del` | Delete selected shape |
| `Ctrl+Z` / `Ctrl+Y` | Undo / Redo |
| `Ctrl+S` | Save |
| `Enter` | Save & next image |
| `R` | Mark current image reviewed without changes |
| `J` | Jump to next image that is not reviewed |
| `←` / `→` | Previous / next image |
| `H` / `L` | Hide annotations / toggle label text |
| `F` or `0` | Fit image to screen |
| `Space` (hold) | Pan with the left mouse button |
| `Wheel` / `Mid-drag` | Zoom at cursor / pan |
| `Esc` | Close popup, cancel drawing, or deselect |
| `?` | Shortcut sheet |

Polygon datasets: click to add vertices, click the first vertex / double-click / `Enter` to close, `Backspace` removes the last vertex, `Alt+click` deletes a vertex, double-click an edge to insert one.

---

## 🛠️ Dataset Utilities & Resizing

Included in the `scripts/` directory are production utilities for dataset maintenance and image dimension conversion:

### Resizing Datasets to 640x640 (or Multiple Sizes)
Convert an entire dataset to standard YOLO input resolutions with mathematical bounding box and polygon transformation:

```bash
# Standard 640x640 letterbox conversion
python scripts/resize_dataset.py --input ./dataset_default --size 640

# Batch export to multiple sizes simultaneously (640, 1280, 416)
python scripts/resize_dataset.py --input ./dataset_default --sizes 640 1280 416

# Custom dimensions & resize mode (letterbox, stretch, max_edge)
python scripts/resize_dataset.py --input ./dataset_default --width 640 --height 640 --mode stretch
```

### High-Speed Multi-Threaded Dataset Compression (.zip, .7z, .tar.gz)
Quickly archive large datasets using parallel worker threads with automatic filtering and integrity verification:

```bash
# Multi-threaded ZIP compression with integrity check
python scripts/compress_dataset.py --input ./dataset_default --output ./dataset.zip --verify

# High-compression LZMA or 7Z
python scripts/compress_dataset.py --input ./dataset_default --format 7z --threads 8

# TAR.GZ format
python scripts/compress_dataset.py --input ./dataset_default --format tar.gz
```

### Dataset Frame Cleanup
Remove specific corrupted frame ranges from database and disk:

```bash
python scripts/cleanup.py --dir ./dataset_default --start 6601 --end 8200
```

---

## 🧪 Testing

Run the automated unit test suite:

```bash
python -m unittest discover tests -v
```

The tests build small throw-away datasets in a temporary folder; they never read or modify your real
`dataset_*` folders, never load a YOLO model and need no GPU or network.

---

## 👨‍💻 Author & Maintainer

**Shubham Kulkarni**
- **Email:** [kulkarnishub377@gmail.com](mailto:kulkarnishub377@gmail.com)
- **LinkedIn:** [linkedin.com/in/shubhkulk21](https://linkedin.com/in/shubhkulk21)
- **GitHub:** [github.com/kulkarnishub377](https://github.com/kulkarnishub377)
- **Portfolio:** [kulkarnishub377.github.io](https://kulkarnishub377.github.io)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
