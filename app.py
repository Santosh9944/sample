# -*- coding: utf-8 -*-
"""
VerifAI - Flask Application Entry Point
Multimodal fake news detection system: Text + Image + Video
"""

import os
import sys
import uuid
import logging
from typing import Optional

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Force UTF-8 output on Windows to avoid charmap codec errors
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# -------------------------------------------------------------------------
# App setup
# -------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    stream=sys.stdout
)
log = logging.getLogger("verifai")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 150 * 1024 * 1024   # 150 MB

ALLOWED_IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}
ALLOWED_VIDEO_EXTS = {"mp4", "avi", "mov", "webm", "mkv"}

# -------------------------------------------------------------------------
# Lazy model singletons (loaded on first request to speed up startup)
# -------------------------------------------------------------------------

_text_analyzer = None
_image_analyzer = None
_video_analyzer = None
_fusion = None


def get_text_analyzer():
    global _text_analyzer
    if _text_analyzer is None:
        log.info("Initialising TextAnalyzer...")
        from models.text_analyzer import TextAnalyzer
        _text_analyzer = TextAnalyzer()
    return _text_analyzer


def get_image_analyzer():
    global _image_analyzer
    if _image_analyzer is None:
        log.info("Initialising ImageAnalyzer...")
        from models.image_analyzer import ImageAnalyzer
        _image_analyzer = ImageAnalyzer()
    return _image_analyzer


def get_video_analyzer():
    global _video_analyzer
    if _video_analyzer is None:
        log.info("Initialising VideoAnalyzer...")
        from models.video_analyzer import VideoAnalyzer
        _video_analyzer = VideoAnalyzer()
    return _video_analyzer


def get_fusion():
    global _fusion
    if _fusion is None:
        from models.fusion import FusionModel
        _fusion = FusionModel()
    return _fusion


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _allowed(filename: str, extensions: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in extensions


def _save_upload(file_storage, extensions: set) -> Optional[str]:
    """Validate and save an uploaded file; return absolute path or None."""
    if not file_storage or not file_storage.filename:
        return None
    if not _allowed(file_storage.filename, extensions):
        return None
    unique_name = "{}_{}".format(uuid.uuid4().hex, secure_filename(file_storage.filename))
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file_storage.save(save_path)
    return save_path


# -------------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accepts multipart/form-data with optional fields:
        text  - plain text to analyse
        image - image file
        video - video file

    Returns JSON with the fused verdict and per-modality breakdown.
    """
    results = {}
    saved_files = []

    try:
        # ---- Text ----
        text = request.form.get("text", "").strip()
        if text:
            log.info("Analysing text (%d chars)...", len(text))
            results["text"] = get_text_analyzer().analyze(text)

        # ---- Image ----
        image_file = request.files.get("image")
        if image_file and image_file.filename:
            path = _save_upload(image_file, ALLOWED_IMAGE_EXTS)
            if path:
                saved_files.append(path)
                log.info("Analysing image: %s", os.path.basename(path))
                results["image"] = get_image_analyzer().analyze(path)
            else:
                results["image"] = {
                    "error": "Unsupported image format",
                    "verdict": "UNCERTAIN",
                    "fake_probability": 0.5,
                    "real_probability": 0.5,
                    "confidence": 0.5,
                    "manipulation_indicators": [],
                }

        # ---- Video ----
        video_file = request.files.get("video")
        if video_file and video_file.filename:
            path = _save_upload(video_file, ALLOWED_VIDEO_EXTS)
            if path:
                saved_files.append(path)
                log.info("Analysing video: %s", os.path.basename(path))
                results["video"] = get_video_analyzer().analyze(path)
            else:
                results["video"] = {
                    "error": "Unsupported video format",
                    "verdict": "UNCERTAIN",
                    "fake_probability": 0.5,
                    "real_probability": 0.5,
                    "confidence": 0.5,
                    "manipulation_indicators": [],
                }

        if not results:
            return jsonify({"error": "No input provided. Please add text, image, or video."}), 400

        final = get_fusion().combine(results)
        log.info("Fusion verdict: %s (fake=%.3f)", final.get("verdict"), final.get("fake_probability", 0))
        return jsonify(final)

    except Exception as exc:
        log.exception("Analysis failed")
        return jsonify({"error": "Analysis failed: {}".format(str(exc))}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "VerifAI"})


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    log.info("Starting VerifAI server on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
