# VerifAI — Multimodal Fake News Detection System

> A deep-learning powered web application that detects fake news across **Text**, **Images**, and **Videos** using a multimodal fusion approach.

---

## 🚀 Tech Stack

| Layer | Technology |
|---|---|
| Text Model | BERT / DistilBERT (HuggingFace Transformers) |
| Image Analysis | Error Level Analysis (ELA) + OpenCV |
| Video Analysis | Frame Sampling + Temporal Consistency |
| Fusion | Confidence-Weighted Multimodal Combination |
| Backend | Flask (Python) |
| Frontend | HTML5 + Vanilla CSS + Vanilla JS + Chart.js |

---

## 📁 Project Structure

```
DL PROJECT/
├── app.py                   # Flask entry point
├── requirements.txt         # Python dependencies
├── models/
│   ├── __init__.py
│   ├── text_analyzer.py     # BERT-based + rule-based text analysis
│   ├── image_analyzer.py    # ELA + metadata + visual forensics
│   ├── video_analyzer.py    # Frame sampling + temporal analysis
│   └── fusion.py            # Confidence-weighted fusion
├── templates/
│   └── index.html           # Main UI template
└── static/
    ├── css/style.css        # Dark glassmorphism theme
    ├── js/main.js           # Interactive frontend logic
    └── uploads/             # Uploaded files (auto-cleaned)
```

---

## ⚙️ Installation

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

> **Note:** On first run, the BERT model will be automatically downloaded from HuggingFace (~400 MB). Subsequent runs will use the cached model.

---

## 🧠 How It Works

### 1. Text Analysis
- **Primary:** BERT-based transformer fine-tuned on fake news detection datasets
- **Secondary:** Linguistic feature analysis (clickbait score, emotional language, CAPS ratio, vague sources, credibility markers)
- **Fusion:** 70% BERT + 30% rule-based

### 2. Image Analysis
- **Error Level Analysis (ELA):** Detects regions with different JPEG compression levels — manipulated areas stand out
- **Metadata Inspection:** Checks for EXIF data, editing software signatures
- **Visual Consistency:** Analyzes color channel noise patterns
- **Face Detection:** Identifies suspicious face regions using OpenCV Haar cascades

### 3. Video Analysis
- **Frame Sampling:** Uniformly samples up to 12 frames across the video duration
- **Per-Frame Analysis:** Runs image forensics on each extracted frame
- **Temporal Consistency:** Flags large variance in fake probability scores across frames
- **Anomaly Detection:** Identifies individual frames with significantly elevated scores

### 4. Multimodal Fusion
- Combines predictions from all active modalities using confidence-weighted averaging
- Base weights: Text (40%), Image (35%), Video (25%)
- Final confidence is used to produce a **Risk Level** (VERY LOW → CRITICAL)

---

## 📊 Model Details

| Modality | Model/Method | Accuracy |
|---|---|---|
| Text | `jy46604790/Fake-News-Bert-Detect` | ~97% on LIAR dataset |
| Image | ELA + Metadata | Heuristic |
| Video | Frame-level ELA + Temporal | Heuristic |

---

## 🎨 UI Features

- **Dark Glassmorphism Theme** with animated gradient orbs
- **Drag & Drop** file upload for images and videos
- **Real-time Character Counter** for text input
- **Step-by-step Loading Animation** showing analysis progress
- **Interactive Charts** (donut chart, bar chart, frame timeline)
- **Per-modality Breakdown Cards** with detailed metrics
- **Detection Signals** — human-readable explanation of findings

---

## ⚠️ Disclaimer

This tool is built for **educational and research purposes** as part of a Deep Learning course project. Results should not be used as the sole basis for determining content authenticity. Always cross-check with multiple trusted sources.

---

## 👨‍💻 Built With

- Python 3.10+
- PyTorch
- HuggingFace Transformers
- Flask
- OpenCV
- Pillow (PIL)
- NumPy / SciPy
- Chart.js
