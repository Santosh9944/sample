# -*- coding: utf-8 -*-
"""
VerifAI - Text Analyzer Module
Uses HuggingFace Inference API for fake news detection (no local model needed).
Falls back to rule-based linguistic analysis if API is unavailable.
"""

import re
import os
import json
import urllib.request
import urllib.error
import numpy as np
from typing import Dict, Any, Optional


class TextAnalyzer:
    """
    Fake news text analyzer using HuggingFace Inference API + rule-based fallback.
    No PyTorch or local model download required.
    """

    API_URL = "https://api-inference.huggingface.co/models/jy46604790/Fake-News-Bert-Detect"
    HF_TOKEN = os.environ.get("HF_TOKEN", "")   # Optional: set for higher rate limits

    def __init__(self):
        print("  [OK] TextAnalyzer initialized (HuggingFace Inference API mode)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text):
        # type: (str) -> Dict[str, Any]
        """Return full analysis dict for the supplied text."""
        if not text or not text.strip():
            return {
                "error": "Empty text provided",
                "verdict": "UNCERTAIN",
                "confidence": 0.0,
                "fake_probability": 0.5,
                "real_probability": 0.5,
                "manipulation_indicators": [],
            }

        rule_result = self._rule_based_analyze(text)
        bert_result = self._bert_analyze(text)

        if bert_result:
            # Weighted blend: 70% BERT + 30% rules
            fake_prob = (
                bert_result["fake_probability"] * 0.70
                + rule_result["fake_probability"] * 0.30
            )
            model_used = "BERT API + Rule-based"
        else:
            fake_prob = rule_result["fake_probability"]
            model_used = "Rule-based (API unavailable)"

        fake_prob = float(np.clip(fake_prob, 0.02, 0.98))
        real_prob = 1.0 - fake_prob
        verdict = "FAKE" if fake_prob > 0.5 else "REAL"

        return {
            "verdict": verdict,
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(real_prob, 4),
            "confidence": round(max(fake_prob, real_prob), 4),
            "model_used": model_used,
            "bert_result": bert_result,
            "features": rule_result["features"],
            "manipulation_indicators": rule_result.get("indicators", []),
        }

    # ------------------------------------------------------------------
    # HuggingFace Inference API
    # ------------------------------------------------------------------

    def _bert_analyze(self, text):
        # type: (str) -> Optional[Dict[str, Any]]
        """Call HuggingFace Inference API for BERT classification."""
        try:
            payload = json.dumps({"inputs": text[:512]}).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if self.HF_TOKEN:
                headers["Authorization"] = "Bearer " + self.HF_TOKEN

            req = urllib.request.Request(self.API_URL, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode("utf-8")

            data = json.loads(raw)

            # HF API returns: [[{"label": "LABEL_0", "score": 0.99}, ...]]
            if isinstance(data, list) and len(data) > 0:
                inner = data[0] if isinstance(data[0], list) else data
                # Sort by score descending, pick top label
                best = max(inner, key=lambda x: x.get("score", 0))
                label = str(best.get("label", "")).upper()
                score = float(best.get("score", 0.5))

                if "LABEL_1" in label or "FAKE" in label:
                    fake_prob = score
                elif "LABEL_0" in label or "REAL" in label:
                    fake_prob = 1.0 - score
                else:
                    fake_prob = 1.0 - score

                return {
                    "raw_label": label,
                    "raw_score": round(score, 4),
                    "fake_probability": round(float(fake_prob), 4),
                    "real_probability": round(float(1.0 - fake_prob), 4),
                }

        except urllib.error.HTTPError as e:
            if e.code == 503:
                print("  [WARN] HF model is loading (503). Using rule-based only.")
            else:
                print("  [WARN] HF API HTTP error {}: {}".format(e.code, e.reason))
        except Exception as e:
            print("  [WARN] HF API error: {}".format(e))

        return None

    # ------------------------------------------------------------------
    # Rule-based Analysis
    # ------------------------------------------------------------------

    def _rule_based_analyze(self, text):
        # type: (str) -> Dict[str, Any]
        text_lower = text.lower()
        words = text.split()
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

        # --- Clickbait ---
        clickbait_phrases = [
            "you won't believe", "shocking", "unbelievable", "breaking", "exclusive",
            "explosive", "bombshell", "stunning", "incredible", "amazing", "urgent",
            "alert", "warning", "secret", "revealed", "exposed", "banned", "censored",
            "they don't want you to know", "mainstream media won't tell you", "viral",
        ]
        clickbait_hits = sum(1 for p in clickbait_phrases if p in text_lower)
        clickbait_score = min(clickbait_hits / 5.0, 1.0)

        # --- Emotional language ---
        emotional_words = [
            "outrage", "fury", "rage", "disgusting", "horrifying", "terrible", "awful",
            "devastating", "dangerous", "corrupt", "evil", "criminal", "fraud", "lies",
            "hoax", "conspiracy", "agenda", "puppet", "invasion", "destroy",
        ]
        emotional_hits = sum(1 for w in emotional_words if w in text_lower)
        emotional_score = min(emotional_hits / 8.0, 1.0)

        # --- CAPS ratio ---
        letters = [c for c in text if c.isalpha()]
        caps_ratio = (
            sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0
        )
        caps_score = min(caps_ratio * 3.0, 1.0)

        # --- Exclamation marks ---
        exclamation_count = text.count("!")
        exclamation_score = min(exclamation_count / 5.0, 1.0)

        # --- Vague sources ---
        vague_sources = [
            "sources say", "people are saying", "many believe", "some experts",
            "insiders reveal", "rumor has it", "allegedly", "reportedly",
            "claims to show", "appears to show",
        ]
        vague_hits = sum(1 for p in vague_sources if p in text_lower)
        vague_score = min(vague_hits / 3.0, 1.0)

        # --- Credibility signals ---
        credibility_phrases = [
            "according to", "research shows", "study published", "data shows",
            "statistics show", "peer-reviewed", "scientists say", "official statement",
            "confirmed by", "verified by", "fact-checked",
        ]
        cred_hits = sum(1 for p in credibility_phrases if p in text_lower)
        credibility_bonus = min(cred_hits / 5.0, 0.5)

        # --- Writing quality ---
        avg_word_len = (
            sum(len(w) for w in words) / len(words) if words else 0.0
        )
        avg_sent_len = len(words) / len(sentences) if sentences else 0.0

        writing_quality_score = 0.0
        if avg_word_len < 3.5:
            writing_quality_score += 0.15
        if avg_sent_len > 60:
            writing_quality_score += 0.10
        if avg_sent_len < 4:
            writing_quality_score += 0.10

        raw_score = (
            clickbait_score * 0.25
            + emotional_score * 0.20
            + caps_score * 0.15
            + exclamation_score * 0.10
            + vague_score * 0.20
            + writing_quality_score * 0.10
        ) - credibility_bonus * 0.30

        fake_prob = float(np.clip(raw_score, 0.05, 0.95))

        indicators = []
        if clickbait_score > 0.3:
            indicators.append("Clickbait language detected (score: {:.2f})".format(clickbait_score))
        if emotional_score > 0.3:
            indicators.append("High emotional manipulation language (score: {:.2f})".format(emotional_score))
        if caps_score > 0.3:
            indicators.append("Excessive use of capital letters (ratio: {:.2f})".format(caps_ratio))
        if vague_score > 0.3:
            indicators.append("Vague or unverifiable source citations")
        if credibility_bonus > 0.2:
            indicators.append("Contains credibility markers (citations, studies)")

        return {
            "fake_probability": fake_prob,
            "real_probability": 1.0 - fake_prob,
            "indicators": indicators,
            "features": {
                "word_count": len(words),
                "sentence_count": len(sentences),
                "clickbait_score": round(clickbait_score, 3),
                "emotional_score": round(emotional_score, 3),
                "caps_ratio": round(caps_ratio, 3),
                "exclamation_count": exclamation_count,
                "vague_source_score": round(vague_score, 3),
                "credibility_score": round(credibility_bonus, 3),
                "avg_word_length": round(avg_word_len, 1),
                "avg_sentence_length": round(avg_sent_len, 1),
            },
        }
