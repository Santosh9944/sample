# -*- coding: utf-8 -*-
"""
VerifAI - Image Analyzer Module
Uses Error Level Analysis (ELA), metadata inspection, and visual consistency
checks to detect image manipulation / misinformation.
"""

import io
import os
import numpy as np
from typing import Dict, Any


class ImageAnalyzer:
    """
    Multi-technique image authenticity analyzer.
    Primary technique: Error Level Analysis (ELA) -- no model download required.
    """

    def __init__(self):
        self.face_cascade = None
        self._load_face_detector()
        print("  [OK] Image analyzer initialized")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_face_detector(self):
        try:
            import cv2
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(path):
                self.face_cascade = cv2.CascadeClassifier(path)
                print("  [OK] Face detector ready")
        except Exception as e:
            print("  [WARN] Face detector unavailable: {}".format(e))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, image_path):
        # type: (str) -> Dict[str, Any]
        """Return full analysis dict for the image at *image_path*."""
        try:
            from PIL import Image

            img = Image.open(image_path).convert("RGB")

            ela = self._error_level_analysis(image_path)
            metadata = self._extract_metadata(image_path, img)
            visual = self._visual_analysis(img)
            faces = self._face_detection(image_path)
            compression = self._compression_artifacts(img)

            manip_score = self._manipulation_score(ela, metadata, visual, compression)

            indicators = []
            if ela.get("ela_score", 0) > 0.35:
                indicators.append(
                    "High ELA inconsistency ({:.2f}) -- region(s) may be edited".format(
                        ela["ela_score"]
                    )
                )
            if not metadata.get("has_exif") and img.format != "PNG":
                indicators.append("Missing EXIF metadata -- possible re-save after editing")
            if metadata.get("software_edit"):
                indicators.append(
                    "Editing software detected: {}".format(metadata.get("software", "Unknown"))
                )
            if compression.get("artifact_score", 0) > 0.5:
                indicators.append("Unusual JPEG compression artifacts")
            if visual.get("color_inconsistency", 0) > 0.4:
                indicators.append("Color-channel inconsistency -- possible compositing")

            fake_prob = float(np.clip(manip_score, 0.02, 0.98))

            return {
                "verdict": "FAKE" if fake_prob > 0.5 else "REAL",
                "fake_probability": round(fake_prob, 4),
                "real_probability": round(1.0 - fake_prob, 4),
                "confidence": round(max(fake_prob, 1.0 - fake_prob), 4),
                "ela_analysis": ela,
                "metadata": metadata,
                "visual_analysis": visual,
                "face_detection": faces,
                "compression_analysis": compression,
                "manipulation_indicators": indicators,
                "image_size": "{}x{}".format(img.width, img.height),
                "image_format": str(img.format or "unknown"),
            }

        except FileNotFoundError:
            return self._error_result("Image file not found")
        except Exception as e:
            return self._error_result(str(e))

    # ------------------------------------------------------------------
    # ELA -- Error Level Analysis
    # ------------------------------------------------------------------

    def _error_level_analysis(self, image_path):
        # type: (str) -> Dict[str, Any]
        """
        Compare the original image with a re-compressed version.
        Edited regions typically show higher error levels.
        """
        try:
            from PIL import Image, ImageChops

            original = Image.open(image_path).convert("RGB")
            scores_by_quality = []

            for quality in (75, 85, 90):
                buf = io.BytesIO()
                original.save(buf, format="JPEG", quality=quality)
                buf.seek(0)
                recomp = Image.open(buf).convert("RGB")

                diff = ImageChops.difference(original, recomp)
                diff_arr = np.asarray(diff, dtype=float)

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

            # Normalise to [0, 1]
            ela_score = float(np.clip(avg_mean / 255.0 * 12 + avg_std / 255.0 * 4, 0, 1))

            return {
                "ela_score": round(ela_score, 4),
                "ela_mean": round(avg_mean, 4),
                "ela_std": round(avg_std, 4),
                "quality_scores": scores_by_quality,
                "suspicious": ela_score > 0.3,
            }
        except Exception as e:
            return {"ela_score": 0.5, "error": str(e), "suspicious": False}

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _extract_metadata(self, image_path, img):
        # type: (str, Any) -> Dict[str, Any]
        try:
            from PIL.ExifTags import TAGS

            has_exif = False
            software_edit = False
            software = ""
            exif_fields = {}   # type: Dict[str, str]

            raw_exif_fn = getattr(img, "_getexif", None)
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
                            "snapseed", "vsco", "paint",
                        ]
                        if any(sw in software.lower() for sw in edit_sw):
                            software_edit = True
                    if tag_name in ("Make", "Model", "Software", "DateTime"):
                        exif_fields[tag_name] = str(value)[:80]

            try:
                file_size_kb = round(os.path.getsize(image_path) / 1024, 1)
            except Exception:
                file_size_kb = 0

            return {
                "has_exif": has_exif,
                "exif_data": exif_fields,
                "software": software,
                "software_edit": software_edit,
                "file_size_kb": file_size_kb,
            }
        except Exception as e:
            return {"has_exif": False, "error": str(e), "software_edit": False}

    # ------------------------------------------------------------------
    # Visual Analysis
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
    # Face Detection
    # ------------------------------------------------------------------

    def _face_detection(self, image_path):
        # type: (str) -> Dict[str, Any]
        if self.face_cascade is None:
            return {"face_count": 0, "faces_detected": False, "available": False}
        try:
            import cv2

            img = cv2.imread(image_path)
            if img is None:
                return {"face_count": 0, "faces_detected": False, "error": "unreadable"}
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
            )
            count = int(len(faces)) if hasattr(faces, "__len__") else 0
            return {
                "face_count": count,
                "faces_detected": count > 0,
                "available": True,
                "note": (
                    "Multiple faces detected - verify context"
                    if count > 3
                    else ""
                ),
            }
        except Exception as e:
            return {"face_count": 0, "faces_detected": False, "error": str(e)}

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
                artifact_score = float(np.clip(avg_diff / 30.0, 0, 1))
            else:
                avg_diff = 0.0
                artifact_score = 0.0

            return {
                "artifact_score": round(artifact_score, 4),
                "avg_block_diff": round(avg_diff, 2),
            }
        except Exception as e:
            return {"artifact_score": 0.2, "error": str(e)}

    # ------------------------------------------------------------------
    # Score Fusion
    # ------------------------------------------------------------------

    def _manipulation_score(self, ela, metadata, visual, compression):
        # type: (Dict, Dict, Dict, Dict) -> float
        score = 0.0
        score += float(ela.get("ela_score", 0.5)) * 0.40
        if not metadata.get("has_exif"):
            score += 0.08
        if metadata.get("software_edit"):
            score += 0.15
        score += float(min(visual.get("color_inconsistency", 0), 1.0)) * 0.17
        score += float(compression.get("artifact_score", 0)) * 0.20
        return score

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
