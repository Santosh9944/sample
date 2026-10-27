# -*- coding: utf-8 -*-
"""
VerifAI - Video Analyzer Module
Samples frames from a video and runs image-level forensics on each frame.
Temporal consistency and frame anomalies are used to detect manipulation / deepfakes.
"""

import os
import tempfile
import numpy as np
from typing import Dict, Any, List


class VideoAnalyzer:
    """
    Frame-sampled video authenticity analyzer.
    Analyzes temporal continuity, frame-level ELA, and anomalous frame spikes.
    """

    MAX_FRAMES = 12          # Max frames to sample
    ANOMALY_THRESHOLD = 1.5  # Std-dev multiplier for flagging anomalous frames

    def __init__(self):
        from models.image_analyzer import ImageAnalyzer
        self.image_analyzer = ImageAnalyzer()
        print("  [OK] Video analyzer initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Return full analysis dict for the video at *video_path*."""
        try:
            import cv2
        except ImportError:
            return self._error_result("OpenCV not installed. Cannot analyse video.")

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return self._error_result("Cannot open video file.")

            # Video metadata
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = max(total_frames / fps, 0.1) if total_frames > 0 else 0.1

            # Sample frames uniformly
            sample_count = min(self.MAX_FRAMES, max(1, total_frames))
            frame_indices = np.linspace(0, max(total_frames - 1, 0), sample_count, dtype=int)

            frame_results = []   # type: List[Dict]
            temp_files = []      # type: List[str]

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                try:
                    cv2.imwrite(tmp.name, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    tmp.close()
                    temp_files.append(tmp.name)

                    # Pass is_video_frame=True to avoid false-positive EXIF penalties
                    result = self.image_analyzer.analyze(tmp.name, is_video_frame=True)
                    frame_results.append(
                        {
                            "frame_index": int(idx),
                            "timestamp": round(int(idx) / fps, 2),
                            "fake_probability": float(result.get("fake_probability", 0.5)),
                            "ela_score": float(
                                result.get("ela_analysis", {}).get("ela_score", 0)
                            ),
                            "manipulation_indicators": result.get(
                                "manipulation_indicators", []
                            ),
                        }
                    )
                except Exception as frame_err:
                    print("  [WARN] Frame {} analysis error: {}".format(idx, frame_err))

            cap.release()

            # Cleanup temp files
            for f in temp_files:
                try:
                    if os.path.exists(f):
                        os.unlink(f)
                except Exception:
                    pass

            if not frame_results:
                return self._error_result("No frames could be extracted from the video.")

            return self._aggregate(frame_results, total_frames, fps, duration, width, height)

        except Exception as e:
            return self._error_result(str(e))

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        frame_results,  # type: List[Dict]
        total_frames,   # type: int
        fps,            # type: float
        duration,       # type: float
        width,          # type: int
        height,         # type: int
    ):
        # type: (...) -> Dict[str, Any]

        fake_probs = np.array([r["fake_probability"] for r in frame_results])
        avg_fake = float(fake_probs.mean())
        fake_std = float(fake_probs.std())
        peak_fake = float(np.max(fake_probs)) if len(fake_probs) else avg_fake

        # Temporal consistency: low std -> stable frames; high std -> possible spliced edits
        temporal_consistency = float(np.clip(1.0 - fake_std * 2.5, 0.05, 1.0))

        # Anomalous frames: must be both statistical outliers AND above moderate fake threshold
        anomaly_threshold = max(avg_fake + 0.15, avg_fake + self.ANOMALY_THRESHOLD * fake_std)
        anomalous = [r for r in frame_results if r["fake_probability"] >= anomaly_threshold and r["fake_probability"] > 0.40]

        # Collect unique indicators
        all_indicators = []  # type: List[str]
        for r in frame_results:
            all_indicators.extend(r["manipulation_indicators"])
        all_indicators = list(dict.fromkeys(all_indicators))[:6]

        if fake_std > 0.18:
            all_indicators.append(
                "High temporal variance between frames -- possible selective segment tampering"
            )
        if len(anomalous) > 0:
            all_indicators.append(
                "{} of {} sampled frames show elevated manipulation signals".format(
                    len(anomalous), len(frame_results)
                )
            )

        # Blended fake probability: 70% average frame score + 30% peak frame score
        # plus temporal variance penalty
        blended_fake = 0.70 * avg_fake + 0.30 * peak_fake
        if fake_std > 0.18:
            blended_fake += min((fake_std - 0.18) * 0.4, 0.12)

        fake_prob = float(np.clip(blended_fake, 0.04, 0.96))
        real_prob = 1.0 - fake_prob
        verdict = "FAKE" if fake_prob > 0.50 else "REAL"

        return {
            "verdict": verdict,
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(real_prob, 4),
            "confidence": round(max(fake_prob, real_prob), 4),
            "video_info": {
                "duration_sec": round(duration, 2),
                "fps": round(fps, 2),
                "resolution": "{}x{}".format(width, height),
                "total_frames": total_frames,
                "analyzed_frames": len(frame_results),
            },
            "temporal_analysis": {
                "consistency_score": round(temporal_consistency, 4),
                "fake_prob_mean": round(avg_fake, 4),
                "fake_prob_std": round(fake_std, 4),
                "anomalous_frames": len(anomalous),
            },
            "frame_analysis": frame_results,
            "manipulation_indicators": all_indicators,
        }

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
            "manipulation_indicators": ["Video error: {}".format(message)],
        }
