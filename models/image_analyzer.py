# -*- coding: utf-8 -*-
"""
VerifAI - Image Analyzer Module
Uses Error Level Analysis (ELA), metadata inspection, visual consistency,
and face forensics to detect image manipulation / misinformation.
"""

import io
import os
import numpy as np
from typing import Dict, Any, Optional


class ImageAnalyzer:
    """
    Multi-technique image authenticity analyzer.
    Techniques:
    - Error Level Analysis (ELA) for recompression / splicing artifacts
    - EXIF metadata forensics (software tags, camera model, timestamps)
    - Color channel distribution & visual consistency
    - Multi-environment Face detection (Haar cascade with YCrCb chrominance fallback)
    - JPEG 8x8 block grid compression artifact analysis
    """

    def __init__(self):
        self.face_cascade = None
        self._load_face_detector()
        print("  [OK] Image analyzer initialized")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_face_detector(self):
        """Attempt to load OpenCV Haar cascade if available."""
        try:
            import cv2
            if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades") and hasattr(cv2, "CascadeClassifier"):
                path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                if os.path.exists(path):
                    self.face_cascade = cv2.CascadeClassifier(path)
                    print("  [OK] Haar face cascade detector ready")
        except Exception:
            self.face_cascade = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, image_path, is_video_frame=False):
        # type: (str, bool) -> Dict[str, Any]
        """Return full analysis dict for the image at *image_path*."""
        try:
            from PIL import Image

            # Keep raw format before color conversion
            raw_img = Image.open(image_path)
            orig_format = (raw_img.format or "").upper()
            width, height = raw_img.size

            # Working RGB image
            img = raw_img.convert("RGB")

            ela = self._error_level_analysis(image_path, img)
            metadata = self._extract_metadata(image_path, raw_img, is_video_frame, orig_format)
            visual = self._visual_analysis(img)
            faces = self._face_detection(image_path, img)
            compression = self._compression_artifacts(img)

            manip_score = self._manipulation_score(ela, metadata, visual, compression, is_video_frame)

            indicators = []
            if ela.get("ela_score", 0) > 0.35:
                indicators.append(
                    "High ELA inconsistency ({:.2f}) -- image regions show re-compression or splicing".format(
                        ela["ela_score"]
                    )
                )
            if not is_video_frame and orig_format in ("JPEG", "JPG", "TIFF") and not metadata.get("has_exif"):
                indicators.append("Missing camera EXIF metadata in photo file -- possible re-export after editing")
            if metadata.get("software_edit"):
                indicators.append(
                    "Editing software signature detected: {}".format(metadata.get("software", "Unknown"))
                )
            if compression.get("artifact_score", 0) > 0.45:
                indicators.append("Irregular 8x8 block compression artifacts detected")
            if visual.get("color_inconsistency", 0) > 0.40:
                indicators.append("Significant color-channel distribution discrepancy -- possible compositing")
            if faces.get("faces_detected") and faces.get("face_count", 0) > 4:
                indicators.append("Crowd / multi-face scene detected ({} face candidates)".format(faces["face_count"]))

            fake_prob = float(np.clip(manip_score, 0.03, 0.97))
            real_prob = 1.0 - fake_prob
            verdict = "FAKE" if fake_prob > 0.50 else "REAL"

            return {
                "verdict": verdict,
                "fake_probability": round(fake_prob, 4),
                "real_probability": round(real_prob, 4),
                "confidence": round(max(fake_prob, real_prob), 4),
                "ela_analysis": ela,
                "metadata": metadata,
                "visual_analysis": visual,
                "face_detection": faces,
                "compression_analysis": compression,
                "manipulation_indicators": indicators,
                "image_size": "{}x{}".format(width, height),
                "image_format": orig_format or "JPEG",
            }

        except FileNotFoundError:
            return self._error_result("Image file not found")
        except Exception as e:
            return self._error_result(str(e))

    # ------------------------------------------------------------------
    # ELA -- Error Level Analysis
    # ------------------------------------------------------------------

    def _error_level_analysis(self, image_path, original_rgb):
        # type: (str, Any) -> Dict[str, Any]
        """
        Compare original image with JPEG re-compressed versions.
        Evaluates both global error and localized block-level discrepancy (splicing/cloning).
        """
        try:
            from PIL import Image, ImageChops

            scores_by_quality = []
            block_ratios = []
            block_maxes = []

            for quality in (75, 85, 90):
                buf = io.BytesIO()
                original_rgb.save(buf, format="JPEG", quality=quality)
                buf.seek(0)
                recomp = Image.open(buf).convert("RGB")

                diff = ImageChops.difference(original_rgb, recomp)
                diff_arr = np.asarray(diff, dtype=float)
                # 2D error intensity map
                err_map = diff_arr.mean(axis=2)

                # Localized block variance (16x16 blocks)
                h, w = err_map.shape
                bs = 16
                b_means = []
                for y in range(0, max(h - bs, 1), bs):
                    for x in range(0, max(w - bs, 1), bs):
                        b_means.append(float(err_map[y: y + bs, x: x + bs].mean()))

                if b_means:
                    b_arr = np.array(b_means)
                    b_mean = float(b_arr.mean())
                    b_max = float(b_arr.max())
                    reg_ratio = float((b_max - b_mean) / (b_mean + 10.0))
                    block_ratios.append(reg_ratio)
                    block_maxes.append(b_max)

                scores_by_quality.append(
                    {
                        "quality": quality,
                        "mean": float(diff_arr.mean()),
                        "std": float(diff_arr.std()),
                        "max": float(diff_arr.max()),
                    }
                )

            avg_mean = float(np.mean([s["mean"] for s in scores_by_quality]))
            avg_std = float(np.mean([s["std"] for s in scores_by_quality]))
            avg_ratio = float(np.mean(block_ratios)) if block_ratios else 0.0
            avg_block_max = float(np.mean(block_maxes)) if block_maxes else avg_mean

            # Spliced regions create high block contrast (> 0.35) and elevated peak error (> 6.0)
            localized_anomaly = max(0.0, (avg_ratio - 0.25) / 0.50)
            base_err = (avg_mean / 255.0) * 8.0 + (avg_std / 255.0) * 3.0

            ela_score = float(np.clip(base_err + localized_anomaly * 0.35, 0.0, 1.0))

            return {
                "ela_score": round(ela_score, 4),
                "ela_mean": round(avg_mean, 4),
                "ela_std": round(avg_std, 4),
                "block_ratio": round(avg_ratio, 3),
                "block_max": round(avg_block_max, 2),
                "quality_scores": scores_by_quality,
                "suspicious": ela_score > 0.35 or avg_ratio > 0.50,
            }
        except Exception as e:
            return {"ela_score": 0.25, "error": str(e), "suspicious": False}

    # ------------------------------------------------------------------
    # Metadata Inspection
    # ------------------------------------------------------------------

    def _extract_metadata(self, image_path, raw_img, is_video_frame, orig_format):
        # type: (str, Any, bool, str) -> Dict[str, Any]
        try:
            from PIL.ExifTags import TAGS

            has_exif = False
            software_edit = False
            software = ""
            exif_fields = {}

            if not is_video_frame:
                raw_exif_fn = getattr(raw_img, "_getexif", None)
                raw_exif = raw_exif_fn() if raw_exif_fn else None

                if raw_exif:
                    has_exif = True
                    for tag_id, value in raw_exif.items():
                        tag_name = TAGS.get(tag_id, str(tag_id))
                        if tag_name == "Software":
                            software = str(value)
                            edit_sw = [
                                "photoshop", "gimp", "lightroom", "premiere",
                                "aftereffects", "davinci", "pixlr", "canva",
                                "snapseed", "vsco", "paint.net", "picsart",
                            ]
                            if any(sw in software.lower() for sw in edit_sw):
                                software_edit = True
                        if tag_name in ("Make", "Model", "Software", "DateTime"):
                            exif_fields[tag_name] = str(value)[:80]

            try:
                file_size_kb = round(os.path.getsize(image_path) / 1024, 1)
            except Exception:
                file_size_kb = 0.0

            return {
                "has_exif": has_exif,
                "exif_data": exif_fields,
                "software": software,
                "software_edit": software_edit,
                "file_size_kb": file_size_kb,
            }
        except Exception as e:
            return {"has_exif": False, "error": str(e), "software_edit": False, "file_size_kb": 0.0}

    # ------------------------------------------------------------------
    # Visual Consistency Analysis
    # ------------------------------------------------------------------

    def _visual_analysis(self, img):
        # type: (Any) -> Dict[str, Any]
        try:
            arr = np.asarray(img, dtype=float)
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

            stds = [float(np.std(r)), float(np.std(g)), float(np.std(b))]
            mean_std = float(np.mean(stds))
            color_inconsistency = float(np.std(stds)) / (mean_std + 1e-10)
            color_inconsistency = min(color_inconsistency, 1.0)

            brightness = float(arr.mean())
            brightness_std = float(arr.std())

            return {
                "brightness": round(brightness, 2),
                "brightness_std": round(brightness_std, 2),
                "color_inconsistency": round(color_inconsistency, 4),
                "channel_stds": {
                    "r": round(stds[0], 2),
                    "g": round(stds[1], 2),
                    "b": round(stds[2], 2),
                },
            }
        except Exception as e:
            return {"color_inconsistency": 0.1, "error": str(e)}

    # ------------------------------------------------------------------
    # Face Detection (Multi-Strategy)
    # ------------------------------------------------------------------

    def _face_detection(self, image_path, pil_img):
        # type: (str, Any) -> Dict[str, Any]
        """
        Detect face regions using Haar cascade if available,
        or robust skin-chrominance blob analysis as a universal fallback.
        """
        # Strategy 1: OpenCV Haar Cascade if loaded
        if self.face_cascade is not None:
            try:
                import cv2
                img = cv2.imread(image_path)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
                    )
                    count = int(len(faces)) if hasattr(faces, "__len__") else 0
                    return {
                        "face_count": count,
                        "faces_detected": count > 0,
                        "method": "HaarCascade",
                        "available": True,
                    }
            except Exception:
                pass

        # Strategy 2: Universal skin-chrominance (YCrCb) face candidate detector
        try:
            arr = np.asarray(pil_img, dtype=float)
            # YCrCb conversion:
            # Y  = 0.299R + 0.587G + 0.114B
            # Cr = (R - Y) * 0.713 + 128
            # Cb = (B - Y) * 0.564 + 128
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            y = 0.299 * r + 0.587 * g + 0.114 * b
            cr = (r - y) * 0.713 + 128
            cb = (b - y) * 0.564 + 128

            # Human skin locus in YCrCb: Cr in [133, 173], Cb in [77, 127]
            skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
            total_pixels = float(skin_mask.size)
            skin_ratio = float(np.sum(skin_mask)) / total_pixels

            # Estimate portrait/face presence based on skin cluster density
            estimated_faces = 0
            if skin_ratio > 0.04:
                # Moderate/large skin clusters indicate 1 or more portrait subjects
                estimated_faces = 1 if skin_ratio < 0.28 else 2

            return {
                "face_count": estimated_faces,
                "faces_detected": estimated_faces > 0,
                "method": "ChrominanceAnalysis",
                "available": True,
            }
        except Exception as e:
            return {"face_count": 0, "faces_detected": False, "method": "None", "error": str(e)}

    # ------------------------------------------------------------------
    # Compression Artifacts
    # ------------------------------------------------------------------

    def _compression_artifacts(self, img):
        # type: (Any) -> Dict[str, Any]
        try:
            arr = np.asarray(img.convert("L"), dtype=float)
            h, w = arr.shape
            diffs = []

            for y in range(0, min(h - 8, 160), 8):
                for x in range(0, min(w - 8, 160), 8):
                    if y + 8 < h:
                        diffs.append(
                            float(np.abs(arr[y + 7, x: x + 8] - arr[y + 8, x: x + 8]).mean())
                        )
                    if x + 8 < w:
                        diffs.append(
                            float(np.abs(arr[y: y + 8, x + 7] - arr[y: y + 8, x + 8]).mean())
                        )

            if diffs:
                avg_diff = float(np.mean(diffs))
                artifact_score = float(np.clip(avg_diff / 30.0, 0.0, 1.0))
            else:
                avg_diff = 0.0
                artifact_score = 0.0

            return {
                "artifact_score": round(artifact_score, 4),
                "avg_block_diff": round(avg_diff, 2),
            }
        except Exception as e:
            return {"artifact_score": 0.15, "error": str(e)}

    # ------------------------------------------------------------------
    # Score Fusion
    # ------------------------------------------------------------------

    def _manipulation_score(self, ela, metadata, visual, compression, is_video_frame=False):
        # type: (Dict, Dict, Dict, Dict, bool) -> float
        score = 0.0
        ela_val = float(ela.get("ela_score", 0.2))
        score += ela_val * 0.50

        if ela.get("suspicious"):
            score += 0.20

        if not is_video_frame:
            if metadata.get("software_edit"):
                score += 0.35
            elif not metadata.get("has_exif"):
                # Minor clue for standalone photos
                score += 0.05

        score += float(min(visual.get("color_inconsistency", 0.0), 1.0)) * 0.15
        score += float(compression.get("artifact_score", 0.0)) * 0.15

        return float(np.clip(score, 0.04, 0.96))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _error_result(message):
        # type: (str) -> Dict[str, Any]
        return {
            "verdict": "UNCERTAIN",
            "fake_probability": 0.5,
            "real_probability": 0.5,
            "confidence": 0.5,
            "error": message,
            "manipulation_indicators": ["Analysis error: {}".format(message)],
        }
