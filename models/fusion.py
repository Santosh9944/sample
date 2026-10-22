# -*- coding: utf-8 -*-
"""
VerifAI - Multimodal Fusion Module
Confidence-weighted combination of text, image, and video predictions.
"""

import numpy as np
from typing import Dict, Any, List


class FusionModel:
    """
    Weighted fusion of modality-level fake-news probabilities.
    Weights are scaled by each modality's reported confidence.
    """

    # Base weights (normalised over active modalities)
    BASE_WEIGHTS = {
        "text": 0.40,
        "image": 0.35,
        "video": 0.25,
    }  # type: Dict[str, float]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def combine(self, results):
        # type: (Dict[str, Any]) -> Dict[str, Any]
        """Combine per-modality result dicts into a single verdict."""

        modality_info = {}  # type: Dict[str, Any]
        weighted_fake = 0.0
        total_weight = 0.0
        modalities_used = []  # type: List[str]

        for modality, result in results.items():
            if modality not in self.BASE_WEIGHTS:
                continue

            is_uncertain = result.get("verdict") == "UNCERTAIN" or "error" in result
            base_w = self.BASE_WEIGHTS[modality] * (0.5 if is_uncertain else 1.0)

            fake_prob = float(result.get("fake_probability", 0.5))
            confidence = float(result.get("confidence", 0.5))

            eff_w = base_w * confidence
            weighted_fake += fake_prob * eff_w
            total_weight += eff_w
            modalities_used.append(modality)

            modality_info[modality] = {
                "verdict": result.get("verdict", "UNCERTAIN"),
                "fake_probability": result.get("fake_probability", 0.5),
                "real_probability": result.get("real_probability", 0.5),
                "confidence": result.get("confidence", 0.5),
                "base_weight": round(base_w, 3),
                "effective_weight": round(eff_w, 3),
                "key_details": self._key_details(modality, result),
            }

        if total_weight == 0:
            return {"error": "No valid modality results to fuse."}

        final_fake = float(np.clip(weighted_fake / total_weight, 0.01, 0.99))
        final_real = 1.0 - final_fake
        verdict = "FAKE" if final_fake > 0.5 else "REAL"

        risk_level, risk_color = self._risk(final_fake)

        # Collect all manipulation indicators (deduped)
        all_indicators = []  # type: List[str]
        for res in results.values():
            all_indicators.extend(res.get("manipulation_indicators", []))
        all_indicators = list(dict.fromkeys(all_indicators))[:10]

        return {
            "verdict": verdict,
            "fake_probability": round(final_fake, 4),
            "real_probability": round(final_real, 4),
            "confidence": round(max(final_fake, final_real), 4),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "modalities_used": modalities_used,
            "modality_results": modality_info,
            "manipulation_indicators": all_indicators,
            "fusion_method": "Confidence-weighted average",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _risk(fake_prob):
        # type: (float) -> tuple
        if fake_prob > 0.82:
            return "CRITICAL", "#ef4444"
        if fake_prob > 0.65:
            return "HIGH", "#f97316"
        if fake_prob > 0.50:
            return "MEDIUM", "#f59e0b"
        if fake_prob > 0.35:
            return "LOW", "#84cc16"
        return "VERY LOW", "#10b981"

    @staticmethod
    def _key_details(modality, result):
        # type: (str, Dict[str, Any]) -> Dict[str, Any]
        if modality == "text":
            f = result.get("features", {})
            return {
                "model": result.get("model_used", "Rule-based"),
                "word_count": f.get("word_count", 0),
                "clickbait_score": f.get("clickbait_score", 0),
                "emotional_score": f.get("emotional_score", 0),
                "credibility_score": f.get("credibility_score", 0),
            }
        if modality == "image":
            ela = result.get("ela_analysis", {})
            meta = result.get("metadata", {})
            faces = result.get("face_detection", {})
            return {
                "ela_score": ela.get("ela_score", 0),
                "has_exif": meta.get("has_exif", False),
                "software_edit": meta.get("software_edit", False),
                "face_count": faces.get("face_count", 0),
                "image_size": result.get("image_size", "N/A"),
            }
        if modality == "video":
            vi = result.get("video_info", {})
            ta = result.get("temporal_analysis", {})
            return {
                "duration_sec": vi.get("duration_sec", 0),
                "analyzed_frames": vi.get("analyzed_frames", 0),
                "consistency_score": ta.get("consistency_score", 0),
                "anomalous_frames": ta.get("anomalous_frames", 0),
                "resolution": vi.get("resolution", "N/A"),
            }
        return {}
