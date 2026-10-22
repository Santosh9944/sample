# -*- coding: utf-8 -*-
"""
VerifAI - Video Analyzer Module
Samples frames from a video and runs image-level analysis on each frame.
Temporal consistency is used as an additional fake signal.
"""

import os
import tempfile
import numpy as np
from typing import Dict, Any, List


class VideoAnalyzer:
    """
    Frame-sampled video authenticity analyzer.
    Re-uses ImageAnalyzer under the hood.
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
            duration = total_frames / fps

            # Sample frames uniformly
            sample_count = min(self.MAX_FRAMES, max(1, total_frames))
            frame_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

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

                    result = self.image_analyzer.analyze(tmp.name)
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

        # Temporal consistency: low std -> consistent; high std -> suspicious
        temporal_consistency = float(max(0.0, 1.0 - fake_std * 2.0))

        # Anomalous frames
        threshold = avg_fake + self.ANOMALY_THRESHOLD * fake_std
        anomalous = [r for r in frame_results if r["fake_probability"] > threshold]

        # Collect unique indicators
        all_indicators = []  # type: List[str]
        for r in frame_results:
            all_indicators.extend(r["manipulation_indicators"])
        all_indicators = list(dict.fromkeys(all_indicators))[:6]

        if fake_std > 0.25:
            all_indicators.append(
                "High variance between frames - possible selective editing"
            )
        if len(anomalous) > len(frame_results) * 0.3:
            all_indicators.append(
                "{} of {} frames show elevated fake signals".format(
                    len(anomalous), len(frame_results)
                )
            )

        fake_prob = float(np.clip(avg_fake, 0.02, 0.98))

        return {
            "verdict": "FAKE" if fake_prob > 0.5 else "REAL",
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(1.0 - fake_prob, 4),
            "confidence": round(max(fake_prob, 1.0 - fake_prob), 4),
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
            "manipulation_indicators": [],
        }
