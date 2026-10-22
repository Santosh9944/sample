# -*- coding: utf-8 -*-
"""
VerifAI - Text Analyzer Module
Uses BERT-based transformer for fake news detection with rule-based fallback.
"""

import re
import numpy as np
from typing import Dict, Any, Optional


class TextAnalyzer:
    """
    BERT-based fake news text analyzer.
    Falls back to rule-based linguistic analysis if model unavailable.
    """

    MODEL_OPTIONS = [
        "jy46604790/Fake-News-Bert-Detect",
        "hamzab/bert-fake-news-detection",
        "mrm8488/bert-tiny-finetuned-fake_news_detection",
    ]

    def __init__(self):
        self.pipeline = None
        self.fake_label = None  # type: Optional[str]
        self._load_model()

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Try to load a pre-trained HuggingFace fake-news classifier."""
        try:
            from transformers import pipeline as hf_pipeline

            for model_name in self.MODEL_OPTIONS:
                try:
                    print("  Loading text model: {} ...".format(model_name))
                    self.pipeline = hf_pipeline(
                        "text-classification",
                        model=model_name,
                        device=-1,       # CPU
                        truncation=True,
                        max_length=512,
                    )
                    self._detect_fake_label()
                    print("  [OK] Text model loaded [{}] fake_label={}".format(
                        model_name, self.fake_label
                    ))
                    return
                except Exception as e:
                    print("  [FAIL] Could not load {}: {}".format(model_name, e))

            print("  [WARN] No BERT model loaded - using rule-based analysis only.")

        except ImportError:
            print("  [WARN] transformers not installed - using rule-based analysis only.")

    def _detect_fake_label(self):
        """Determine which output label corresponds to FAKE news."""
        cfg = getattr(self.pipeline.model, "config", None)
        if cfg and hasattr(cfg, "id2label"):
            for label_id, label in cfg.id2label.items():
                if "fake" in label.lower() or label.upper() in ("LABEL_1", "1"):
                    self.fake_label = label
                    return
            # Default: assume higher id == FAKE
            max_id = max(cfg.id2label.keys())
            self.fake_label = cfg.id2label[max_id]
        else:
            self.fake_label = "FAKE"

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

        bert_result = None
        if self.pipeline:
            bert_result = self._bert_analyze(text)

        if bert_result:
            # Weighted blend: 70% BERT + 30% rules
            fake_prob = (
                bert_result["fake_probability"] * 0.70
                + rule_result["fake_probability"] * 0.30
            )
            model_used = "BERT + Rule-based"
        else:
            fake_prob = rule_result["fake_probability"]
            model_used = "Rule-based" if not self.pipeline else "Rule-based (BERT error)"

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
    # BERT Inference
    # ------------------------------------------------------------------

    def _bert_analyze(self, text):
        # type: (str) -> Optional[Dict[str, Any]]
        try:
            truncated = text[:2000]
            out = self.pipeline(truncated, truncation=True, max_length=512)
            result = out[0] if isinstance(out, list) else out

            label = str(result["label"])
            score = float(result["score"])

            # Interpret label -> fake probability
            label_upper = label.upper()
            if self.fake_label and label == self.fake_label:
                fake_prob = score
            elif "FAKE" in label_upper:
                fake_prob = score
            elif "REAL" in label_upper:
                fake_prob = 1.0 - score
            elif label_upper in ("LABEL_1", "1"):
                fake_prob = score      # Convention: LABEL_1 = FAKE
            else:
                fake_prob = 1.0 - score  # Convention: LABEL_0 = FAKE

            return {
                "raw_label": label,
                "raw_score": round(score, 4),
                "fake_probability": round(float(fake_prob), 4),
                "real_probability": round(float(1.0 - fake_prob), 4),
            }
        except Exception as e:
            print("  [WARN] BERT inference error: {}".format(e))
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

        # --- Credibility signals (reduce fake score) ---
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

        # Human-readable indicators
        indicators = []
        if clickbait_score > 0.3:
            indicators.append(
                "Clickbait language detected (score: {:.2f})".format(clickbait_score)
            )
        if emotional_score > 0.3:
            indicators.append(
                "High emotional manipulation language (score: {:.2f})".format(emotional_score)
            )
        if caps_score > 0.3:
            indicators.append(
                "Excessive use of capital letters (ratio: {:.2f})".format(caps_ratio)
            )
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
