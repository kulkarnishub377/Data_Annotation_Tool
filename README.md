# Data Annotation Tool

![App Overview](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python) ![Flask](https://img.shields.io/badge/Flask-Backend-black?style=for-the-badge&logo=flask) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Integration-yellow?style=for-the-badge)

A complete, professional-grade, end-to-end web application for auto-annotating vehicle datasets, reviewing and editing annotations, splitting datasets, and directly training YOLO models from a sleek, intuitive, and highly responsive UI.

> **🎯 Perfect for Fine-Tuning:** Seamlessly load your old custom models to continue training and fine-tuning them on entirely new datasets directly from the browser!

## 🚀 Key Features

### Powerful Backend (Flask & YOLO Integration)
*   **Multi-Format Inference**: Natively loads `.pt`, `.onnx`, `.engine`, `.tflite`, and `.torchscript` models.
*   **Intelligent Auto-GPU Detection**: Automatically profiles system hardware and selects the fastest available inference device (CUDA GPU, Apple MPS, or fallback CPU) to maximize performance.
*   **Background Pre-Annotation Engine**: Ingests images instantly using fast, space-saving OS hardlinks, while parallel background threads run YOLO inference to auto-annotate ahead of your workflow.
*   **Dataset Export**: Zips and exports your fully annotated dataset and generated `data.yaml` with a single click.
*   **"Null Image" Support & Max Precision**: Retains unannotated background images (Roboflow standard) and saves bounding box coordinates to maximum floating-point precision (16+ decimal places) to preserve absolute mathematical accuracy.

### Advanced Frontend (Vanilla HTML/CSS/JS)
*   **Sleek Setup & Grid View**: A beautiful dark-themed setup screen auto-detects classes. A comprehensive "Grid View" allows dataset-wide inspection, pagination, and status filtering.
*   **Perfect Zoom, Pan & Minimap**: Refined scroll-wheel math for butter-smooth zooming exactly centered on the cursor, a middle-click drag for panning, and a dynamic heads-up minimap for navigation.
*   **Image Enhancement Tools**: Real-time slider controls for adjusting image brightness and contrast on-the-fly to spot difficult objects.
*   **In-Browser Training & SSE Logging**: Start, stop, and configure YOLO training directly from the UI (epochs, batch size, imgsz, lr, device). Monitors progress via Server-Sent Events (SSE) live streaming logs.
*   **Dynamic Class Popups & Analytics**: Inline class selection popups when drawing new boxes. Dynamic visual progress ring and real-time class distribution bar charts tracking dataset balance.
*   **Session Tracking**: Topbar session timer tracks your active annotation time.

## 🏗️ System Architecture

*   **Backend (Flask)**: Serves the REST API and manages application state via a lightweight thread-safe SQLite database (`state.db`).
*   **Background Processing**: Uses Python `threading.Thread` and `queue.Queue` to run YOLO inference asynchronously, ensuring the main UI thread never blocks during pre-annotation or heavy file ingestion.
*   **Inference Engine (YOLO)**: Leverages the `ultralytics` package. The system dynamically queries PyTorch/CUDA/MPS at startup to allocate the most efficient compute device.
*   **Frontend**: Pure HTML5 Canvas, CSS flexbox/grid, and vanilla JavaScript. It manages complex coordinate math for panning/zooming and uses Server-Sent Events (SSE) to receive real-time training logs from the backend subprocess.

## 📂 Directory Structure

```text
Data_annotrator_tool/
├── app.py              # Backend server (Flask, YOLO inference, SQLite state, Threads)
├── app.js              # Core frontend logic (Canvas math, SSE listeners, API calls)
├── index.html          # Frontend UI (Data Annotation Tool markup)
├── style.css           # Custom dark-theme styling, animations, flexbox layouts
├── .env.example        # Environment variables template
├── refactor.py         # Development script for legacy dataset upgrades
├── requirements.txt    # Python dependencies
├── .gitignore          # Git exclusion rules
├── LICENSE             # MIT License
├── tests/              # Unit testing suite
└── README.md           # Documentation
```

## 🛠️ Setup & Installation

### Prerequisites
*   **Python 3.8+**
*   **Modern Web Browser** (Chrome / Edge / Firefox recommended)
*   **Git** (for cloning the repository)

### 1. Clone the Repository

Clone the project to your local machine:

```bash
git clone https://github.com/kulkarnishub377/Data_annotrator_tool.git
cd Data_annotrator_tool
```

### 2. Create a Virtual Environment (Recommended)

Keep your global Python environment clean by using a virtual environment:

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages using the provided requirements file:

```bash
pip install -r requirements.txt
```
> *Note: `pillow` is strictly required for generating optimized, heavily cached thumbnails in the Grid View.*

### 4. Configuration

Configure your environment variables by copying the example file:

```bash
cp .env.example .env
```

Then open `.env` and edit the paths to point to your local directories (this avoids hard-coded paths in the code!):

```env
BASE_DIR=D:\your_raw_images
OUTPUT_DIR=D:\your_raw_images\annotated_dataset
```

## 🚦 Running the Application

With your virtual environment activated, start the Flask server:

```bash
python app.py
```

Then, open your browser and navigate to: **[http://127.0.0.1:8051](http://127.0.0.1:8051)**

> [!CAUTION]
> **Security Notice:** The application has no authentication and may provide filesystem/model-management functionality. Do not expose the server directly to the public internet. Use only on trusted networks.

## 🔄 End-to-End Workflow

1.  **Select Model**: Pick any supported model file (`.pt`, `.onnx`, etc.). The tool auto-detects your GPU and extracts model classes.
2.  **Ingest Images**: Select a source folder and click "Load Images". Hardlinking prevents duplicating image data on your drive.
3.  **Review & Annotate**: YOLO will pre-annotate images in the background. Use the draw mode, hotkeys, and brightness adjustments to perfect your dataset.
4.  **Auto-Split Dataset**: Check the real-time Class Distribution chart, configure your Train/Valid/Test ratios, and execute "Auto-Split Dataset".
5.  **Train Model**: Go to the Training panel, adjust hyperparameters, and click "Start Training". Watch the live CLI logs stream directly into your browser. *(Note: This works perfectly with old custom models, allowing you to easily fine-tune them on entirely new datasets!)*
6.  **Export**: Return to the Grid View and click "Export Dataset" to download the finalized `.zip` package.

## ⌨️ Essential Keyboard Shortcuts

| Key | Action |
| :--- | :--- |
| `D` | Toggle Draw mode |
| `F` | Fit image to screen |
| `C` | Copy all boxes from the previous image |
| `J` | Jump to the next unannotated image |
| `Q` / `E`| Cycle class of currently selected box |
| `1` – `9` | Set class of currently selected box |
| `←` / `→` | Previous / Next image |
| `Enter` | Save annotations & jump to next image |
| `Del` | Delete selected box |
| `Ctrl + S` | Save annotations manually |
| `Wheel` | Zoom in/out (centered on cursor) |
| `Mid-drag`| Pan canvas |

## 🧪 Testing

To verify your environment is correctly configured, run the included test suite:
```bash
python -m unittest discover tests
```

---
*Data Annotation Tool — Built for high-speed, professional computer vision workflows.*
