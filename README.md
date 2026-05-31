# Car Damage Assessment AI

Car Damage Assessment AI is an end-to-end computer vision and data mining solution designed to automate the detection, classification, and financial evaluation of physical vehicle damage. Leveraging a fine-tuned YOLOv8s object detection architecture and an interactive Streamlit framework, the system transitions vehicle inspections from subjective manual logs to auditable, data-driven insurance workflows.

---

## 🚀 Core Features

* **Multiclass Bounding-Box Detection:** Localizes target impact zones across four distinct architectural damage types: `scratch`, `dent`, `crack`, and `broken_part`.
* **Proportional Severity Estimation:** Evaluates a flexible ratio comparing total bounding box pixel counts against holistic image footprints to classify damage states into `Light`, `Moderate`, or `Severe` without hardcoding camera distance assumptions.
* **Deterministic Financial Simulation:** Translates dynamic geometric boundaries and object classes into standardized cost approximations according to tiered pricing matrix rules.
* **Triage Automated Rules:** Runs a downstream rule engine that flags evaluations with status tags (`AUTO_APPROVE`, `HUMAN_REVIEW`, or `REJECT`) depending on structural damage weights.
* **Rich Diagnostic UI:** Provides interactive telemetry displays featuring class distribution pie charts, severity statistics bar charts, before/after repair simulations, and downloadable structured JSON logs.

---

## 🏗️ System Architecture

The analytical pipeline is built on a decoupled three-tier workflow:
1. **Ingestion Layer (`app.py`):** A responsive Streamlit engine accepting user-uploaded images (`JPEG`/`PNG`). It orchestrates background visual state renderings and houses data dashboards.
2. **Vision Execution Layer (`detector.py`):** Mounts a fine-tuned **YOLOv8s** weights file trained on over 4,000 images from the public CarDD dataset. It implements accelerated inference routing optimized natively across available hardware backends (`CUDA` ➔ `MPS` ➔ `CPU`).
3. **Operational Heuristics Layer (`detector.py`):** Feeds probabilistic bounding box dimensions into rule-based conditional loops to perform cost modeling and status tracking.

---

## 🛠️ Tech Stack & Libraries

* **Framework & UI:** Streamlit
* **Computer Vision Backbone:** Ultralytics YOLOv8s, OpenCV
* **Data Diagnostics:** Pandas, NumPy, Matplotlib, Plotly
* **Core Environment:** Python 3.10+

---

## 📁 Repository Structure

```text
├── src/
│   ├── train.py                # Backbone training script using Ultralytics API
│   ├── detector.py             # CarDamageDetector implementation class
│   └── app.py                  # Streamlit-based web dashboard interface
│
├── data/
│   ├── data.yaml               # YOLO dataset configurations and class mapping
│   ├── train/                  # CarDD training images and bounding box labels
│   └── val/                    # CarDD validation subset images and labels
│
├── models/
│   └── best.pt                 # High-resolution fine-tuned YOLOv8s weights
│
└── package.json / requirements.txt
