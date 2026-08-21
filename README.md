# Free Open Source Data Annotation Tool for Computer Vision 🚀

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
│   │   └── style.css            # Dark theme styles, tokens, and animations
│   ├── js/
│   │   └── app.js               # Frontend canvas math, hotkeys, API & SSE
│   └── assets/                  # Icons and static media
├── scripts/
│   ├── cleanup.py               # Dataset maintenance & cleanup tool
│   └── refactor.py              # Legacy dataset upgrade utility
├── tests/
│   ├── __init__.py
│   └── test_app.py              # Automated unit test suite
├── .env.example                 # Environment variables template
├── .gitattributes               # Cross-platform line endings & binary formatting
├── .gitignore                   # Comprehensive ignore rules
├── app.py                       # Core Flask backend & YOLO engine
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

---

## 🚦 Running the Application

With your virtual environment activated, start the server:

```bash
python app.py
```

Then open your browser and navigate to: **[http://127.0.0.1:8051](http://127.0.0.1:8051)**

---

## ⌨️ Essential Keyboard Shortcuts

| Key | Action |
| :--- | :--- |
| `D` | Toggle Draw mode |
| `Ctrl` (Hold) | Draw box / polygon |
| `F` | Fit image to screen |
| `C` | Copy all boxes from previous image |
| `J` | Jump to next unannotated image |
| `Q` / `E` | Cycle class of currently selected box |
| `1` – `9` | Set class 1-9 for selected box |
| `←` / `→` | Previous / Next image |
| `Enter` | Save annotations & jump to next image |
| `Del` | Delete selected box |
| `Ctrl + Z` / `Ctrl + Y` | Undo / Redo annotation actions |
| `Wheel` | Zoom in/out (centered on cursor) |
| `Mid-drag` | Pan canvas |

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

### Dataset Frame Cleanup
Remove specific corrupted frame ranges from database and disk:

```bash
python scripts/cleanup.py --dir ./dataset_default --start 6601 --end 8200
```

---

## 🧪 Testing

Run the automated unit test suite:

```bash
python -m unittest discover tests
```

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
