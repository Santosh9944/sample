# -*- coding: utf-8 -*-
"""
VerifAI - Text Analyzer Module
Hybrid Fake News Detection combining:
1. Local trained Machine Learning model (TF-IDF + Logistic Regression) on news dataset
2. Deep Linguistic & Stylometric NLP Forensics (Clickbait, Conspiracy, Emotional, Credibility)
3. Optional HuggingFace Inference API with automatic fallback
"""

import os
import re
import math
import json
import urllib.request
import urllib.error
import numpy as np
from typing import Dict, Any, Optional

try:
    import joblib
except ImportError:
    joblib = None


class TextAnalyzer:
    """
    Fake news text analyzer using local ML pipeline + deep linguistic NLP heuristics,
    with optional HuggingFace Inference API support.
    """

    API_URL = "https://api-inference.huggingface.co/models/jy46604790/Fake-News-Bert-Detect"
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "text_model.joblib")
    DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_news.csv")

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.local_model = None
        self.feature_names = set()
        self.fake_class_idx = 0

        self._init_local_model()
        self._init_patterns()
        print("  [OK] TextAnalyzer initialized (Hybrid Local ML + Linguistic NLP)")

    # ------------------------------------------------------------------
    # Model Loading / Bootstrapping
    # ------------------------------------------------------------------

    def _init_local_model(self):
        """Load pre-trained model or train on the local dataset if needed."""
        if joblib is None:
            print("  [WARN] joblib not installed; running in linguistic NLP mode.")
            return

        if os.path.exists(self.model_path):
            try:
                self.local_model = joblib.load(self.model_path)
                self._cache_model_info()
                print("  [OK] Loaded local text model from {}".format(os.path.basename(self.model_path)))
                return
            except Exception as e:
                print("  [WARN] Failed to load {}: {}".format(self.model_path, e))

        # If model file does not exist, try auto-training on test_news.csv
        if os.path.exists(self.DATASET_PATH):
            try:
                print("  [INFO] Training text model on {}...".format(os.path.basename(self.DATASET_PATH)))
                import pandas as pd
                from sklearn.pipeline import Pipeline
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.linear_model import LogisticRegression

                df = pd.read_csv(self.DATASET_PATH).dropna(subset=["text", "label"])
                df["label"] = df["label"].str.strip().str.upper()
                df = df[df["label"].isin(["FAKE", "REAL"])]

                pipe = Pipeline([
                    ("tfidf", TfidfVectorizer(
                        max_features=30000,
                        stop_words="english",
                        ngram_range=(1, 2),
                        sublinear_tf=True
                    )),
                    ("clf", LogisticRegression(
                        max_iter=1000,
                        C=2.0,
                        class_weight="balanced"
                    ))
                ])
                pipe.fit(df["text"], df["label"])
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                joblib.dump(pipe, self.model_path)
                self.local_model = pipe
                self._cache_model_info()
                print("  [OK] Trained & saved local model to {}".format(os.path.basename(self.model_path)))
            except Exception as e:
                print("  [WARN] Auto-training failed: {}. Continuing with linguistic NLP.".format(e))

    def _cache_model_info(self):
        """Cache feature vocabulary and class indices for fast inference."""
        if self.local_model is None:
            return
        try:
            tfidf = self.local_model.named_steps.get("tfidf")
            if tfidf and hasattr(tfidf, "vocabulary_"):
                self.feature_names = set(tfidf.vocabulary_.keys())

            clf = self.local_model.named_steps.get("clf")
            if clf and hasattr(clf, "classes_"):
                classes = list(clf.classes_)
                self.fake_class_idx = classes.index("FAKE") if "FAKE" in classes else 0
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Linguistic & Forensic Pattern Lexicons
    # ------------------------------------------------------------------

    def _init_patterns(self):
        """Compile regex patterns for forensic stylometric evaluation."""
        self.clickbait_regexes = [
            re.compile(r"\byou won'?t believe\b", re.I),
            re.compile(r"\bshocking\b", re.I),
            re.compile(r"\bunbelievable\b", re.I),
            re.compile(r"\bmind[- ]blowing\b", re.I),
            re.compile(r"\bexplosive\b", re.I),
            re.compile(r"\bbombshell\b", re.I),
            re.compile(r"\bsecret revealed\b", re.I),
            re.compile(r"\bexposed\b", re.I),
            re.compile(r"\bwhat happened next\b", re.I),
            re.compile(r"\bthey don'?t want you to know\b", re.I),
            re.compile(r"\bmainstream media won'?t tell you\b", re.I),
            re.compile(r"\bshare before (it'?s|its) (banned|deleted|taken down)\b", re.I),
            re.compile(r"\bwill shock you\b", re.I),
            re.compile(r"\bmiracle cure\b", re.I),
            re.compile(r"\binstant cure\b", re.I),
            re.compile(r"\bbanned from\b", re.I),
            re.compile(r"\bcensored\b", re.I),
            re.compile(r"\bunprecedented revelation\b", re.I),
            re.compile(r"\bbreaking bombshell\b", re.I),
            re.compile(r"\bthis changes everything\b", re.I),
            re.compile(r"\bdon'?t let them fool you\b", re.I),
        ]

        self.conspiracy_regexes = [
            re.compile(r"\bdeep state\b", re.I),
            re.compile(r"\bbig pharma\b", re.I),
            re.compile(r"\billuminati\b", re.I),
            re.compile(r"\bcover[- ]?up\b", re.I),
            re.compile(r"\bbioweapon\b", re.I),
            re.compile(r"\bdepopulation\b", re.I),
            re.compile(r"\bsecret agenda\b", re.I),
            re.compile(r"\bfalse flag\b", re.I),
            re.compile(r"\bglobalist\b", re.I),
            re.compile(r"\bpuppet master\b", re.I),
            re.compile(r"\bplandemic\b", re.I),
            re.compile(r"\bpoisoning the population\b", re.I),
            re.compile(r"\bmicrochip\b", re.I),
            re.compile(r"\b5g (towers|radiation|outbreak)\b", re.I),
            re.compile(r"\bcabal\b", re.I),
            re.compile(r"\bsecret society\b", re.I),
            re.compile(r"\bchemtrail\b", re.I),
            re.compile(r"\bstaged event\b", re.I),
        ]

        self.vague_sources_regexes = [
            re.compile(r"\bsources say\b", re.I),
            re.compile(r"\bpeople are saying\b", re.I),
            re.compile(r"\bmany believe\b", re.I),
            re.compile(r"\bsome experts\b", re.I),
            re.compile(r"\binsiders reveal\b", re.I),
            re.compile(r"\brumor has it\b", re.I),
            re.compile(r"\ballegedly\b", re.I),
            re.compile(r"\breportedly\b", re.I),
            re.compile(r"\bclaims to show\b", re.I),
            re.compile(r"\bword is that\b", re.I),
            re.compile(r"\beveryone knows\b", re.I),
            re.compile(r"\banonymous source\b", re.I),
        ]

        self.credibility_regexes = [
            re.compile(r"\baccording to\b", re.I),
            re.compile(r"\breuters\b", re.I),
            re.compile(r"\bassociated press\b", re.I),
            re.compile(r"\bspokesperson (for|said|stated)\b", re.I),
            re.compile(r"\bin a statement\b", re.I),
            re.compile(r"\bpress release\b", re.I),
            re.compile(r"\bpress conference\b", re.I),
            re.compile(r"\bpublished in\b", re.I),
            re.compile(r"\bpeer[- ]reviewed\b", re.I),
            re.compile(r"\bjournal of\b", re.I),
            re.compile(r"\bofficial data\b", re.I),
            re.compile(r"\bdepartment of\b", re.I),
            re.compile(r"\bministry of\b", re.I),
            re.compile(r"\bconfirmed by\b", re.I),
            re.compile(r"\bverified by\b", re.I),
            re.compile(r"\bstatistically significant\b", re.I),
            re.compile(r"\bdata shows\b", re.I),
            re.compile(r"\bstudy conducted by\b", re.I),
            re.compile(r"\bresearchers at\b", re.I),
            re.compile(r"\bclinical trial\b", re.I),
            re.compile(r"\bofficial statement\b", re.I),
            re.compile(r"\bfact[- ]checked\b", re.I),
        ]

        self.emotional_words = {
            "outrage", "fury", "rage", "disgusting", "horrifying", "horrific", "terrible", "awful",
            "devastating", "dangerous", "corrupt", "evil", "criminal", "fraud", "lies", "treason",
            "insanity", "nightmare", "disaster", "catastrophe", "vicious", "scandalous",
            "pathetic", "scam", "shameful", "monstrous", "atrocity", "sinister", "corrupt",
            "hoax", "conspiracy", "agenda", "puppet", "invasion", "destroy", "annihilate"
        }

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
                "features": {},
            }

        text = text.strip()

        # 1. Forensic Linguistic / Stylometric Analysis
        ling_result = self._linguistic_analyze(text)

        # 2. Local Machine Learning Model Analysis
        ml_result = self._ml_analyze(text)

        # 3. Optional HuggingFace API Call (with quick timeout)
        bert_result = self._bert_analyze(text)

        # --------------------------------------------------------------
        # Ensemble Blending
        # --------------------------------------------------------------
        if bert_result:
            # If external BERT API responded, combine it with local ML and linguistics
            if ml_result:
                fake_prob = (
                    bert_result["fake_probability"] * 0.40
                    + ml_result["fake_probability"] * 0.35
                    + ling_result["fake_probability"] * 0.25
                )
                model_used = "BERT API + Local ML + Linguistic NLP"
            else:
                fake_prob = (
                    bert_result["fake_probability"] * 0.60
                    + ling_result["fake_probability"] * 0.40
                )
                model_used = "BERT API + Linguistic NLP"
        elif ml_result:
            # Primary mode: Local ML Model + Linguistic Forensics
            # Weight is dynamically adjusted based on vocabulary coverage
            coverage = ml_result.get("coverage", 0.0)
            # If input has good overlap with trained vocabulary, ML gets up to 65% weight
            # Otherwise linguistic rules get more weight
            ml_weight = float(np.clip(0.35 + 0.35 * coverage, 0.35, 0.70))
            ling_weight = 1.0 - ml_weight

            fake_prob = (
                ml_result["fake_probability"] * ml_weight
                + ling_result["fake_probability"] * ling_weight
            )
            model_used = "TF-IDF ML Model + Linguistic Forensics"
        else:
            # Fallback when no ML model is available
            fake_prob = ling_result["fake_probability"]
            model_used = "Linguistic & Forensic NLP"

        # Safe bounds to reflect natural probabilistic uncertainty
        fake_prob = float(np.clip(fake_prob, 0.03, 0.97))
        real_prob = 1.0 - fake_prob

        verdict = "FAKE" if fake_prob > 0.50 else "REAL"
        confidence = float(max(fake_prob, real_prob))

        # Merge manipulation indicators
        indicators = list(ling_result.get("indicators", []))
        if ml_result and ml_result.get("fake_probability", 0) > 0.75:
            indicators.append("ML Classifier identified high similarity to known disinformation patterns")

        return {
            "verdict": verdict,
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(real_prob, 4),
            "confidence": round(confidence, 4),
            "model_used": model_used,
            "bert_result": bert_result,
            "ml_result": ml_result,
            "features": ling_result["features"],
            "manipulation_indicators": indicators,
        }

    # ------------------------------------------------------------------
    # Local Machine Learning Classification
    # ------------------------------------------------------------------

    def _ml_analyze(self, text):
        # type: (str) -> Optional[Dict[str, Any]]
        """Run text through the local trained TF-IDF + Classifier model."""
        if self.local_model is None:
            return None

        try:
            words = set(re.findall(r"\b[a-zA-Z0-9_\']+\b", text.lower()))
            if not words:
                return None

            # Calculate vocabulary coverage (how familiar the model is with this text)
            matching_tokens = len(words.intersection(self.feature_names))
            coverage = min(matching_tokens / max(len(words), 1), 1.0)

            # Predict probabilities
            probs = self.local_model.predict_proba([text])[0]
            fake_p = float(probs[self.fake_class_idx])
            real_p = float(1.0 - fake_p)

            return {
                "fake_probability": round(fake_p, 4),
                "real_probability": round(real_p, 4),
                "coverage": round(coverage, 3),
            }
        except Exception as e:
            print("  [WARN] Local ML prediction error: {}".format(e))
            return None

    # ------------------------------------------------------------------
    # Comprehensive Linguistic & Forensic Analyzer
    # ------------------------------------------------------------------

    def _linguistic_analyze(self, text):
        # type: (str) -> Dict[str, Any]
        """
        Evaluate text using linguistic forensics, stylometry, and credibility signals.
        Centered around 0.50 neutral baseline.
        """
        text_lower = text.lower()
        words = re.findall(r"[a-zA-Z0-9_\']+", text)
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 3]
        num_words = max(len(words), 1)

        # 1. Clickbait & Sensationalism
        cb_hits = sum(len(p.findall(text_lower)) for p in self.clickbait_regexes)
        cb_score = min(cb_hits / max(1.0, math.sqrt(num_words / 20.0)), 1.0)

        # 2. Conspiracy & Pseudoscience
        consp_hits = sum(len(p.findall(text_lower)) for p in self.conspiracy_regexes)
        consp_score = min(consp_hits / max(1.0, math.sqrt(num_words / 25.0)), 1.0)

        # 3. Emotional / Outrage Density
        emo_hits = sum(1 for w in words if w.lower() in self.emotional_words)
        emo_density = (emo_hits / num_words) * 100.0
        emo_score = min(emo_density / 3.0, 1.0)

        # 4. Vague Sources
        vague_hits = sum(len(p.findall(text_lower)) for p in self.vague_sources_regexes)
        vague_score = min(vague_hits / max(1.0, num_words / 35.0), 1.0)

        # 5. Credibility Markers
        cred_hits = sum(len(p.findall(text_lower)) for p in self.credibility_regexes)
        has_dates = len(re.findall(r"\b(january|february|march|april|may|june|july|august|september|october|november|december|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{4})\b", text_lower)) > 0
        has_numbers = len(re.findall(r"\b\d+(\.\d+)?%?\b", text)) > 0
        has_quotes = bool(re.search(r'["\u201c\u201d\'][^"\u201c\u201d\']{8,}["\u201c\u201d\']', text))

        cred_signals = (
            cred_hits * 1.5
            + (0.4 if has_dates else 0.0)
            + (0.4 if has_numbers else 0.0)
            + (0.6 if has_quotes else 0.0)
        )
        cred_score = min(cred_signals / max(1.8, math.sqrt(num_words / 15.0) * 1.2), 1.0)

        # 6. Formatting / Shouting / Anomalous Punctuation
        letters = [c for c in text if c.isalpha()]
        caps_ratio = (
            sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0
        )
        excess_caps = max(0.0, (caps_ratio - 0.12) / 0.25)
        caps_score = min(excess_caps, 1.0)

        exclamation_count = text.count("!")
        excl_score = min(exclamation_count / max(2.0, num_words / 25.0), 1.0)

        multiple_punct = len(re.findall(r"[!?]{2,}", text))
        punct_score = min(multiple_punct / 2.0, 1.0)

        # 7. Writing Complexity Metrics
        avg_word_len = sum(len(w) for w in words) / num_words
        avg_sent_len = num_words / max(len(sentences), 1)

        # --------------------------------------------------------------
        # Balanced Continuous Logit Calculation (Centered at 0.50)
        # --------------------------------------------------------------
        fake_push = (
            cb_score * 0.90
            + consp_score * 1.00
            + emo_score * 0.70
            + vague_score * 0.50
            + caps_score * 0.45
            + excl_score * 0.35
            + punct_score * 0.40
        )
        real_push = (
            cred_score * 1.30
            + (0.25 if has_dates and has_numbers else 0.0)
            + (0.20 if has_quotes else 0.0)
        )

        net_signal = fake_push - real_push
        # Logistic sigmoid centered at 0.50
        fake_prob = 1.0 / (1.0 + math.exp(-2.6 * net_signal))
        fake_prob = float(np.clip(fake_prob, 0.05, 0.95))

        # Manipulation Indicators list
        indicators = []
        if cb_score > 0.25:
            indicators.append("Sensational or clickbait language detected (score: {:.2f})".format(cb_score))
        if consp_score > 0.20:
            indicators.append("Conspiracy theory / unverified claims rhetoric (score: {:.2f})".format(consp_score))
        if emo_score > 0.25:
            indicators.append("High emotional manipulation / outrage tone (density: {:.1f}%)".format(emo_density))
        if caps_score > 0.25:
            indicators.append("Excessive capitalization / shouting detected ({:.1f}% uppercase)".format(caps_ratio * 100))
        if punct_score > 0.25:
            indicators.append("Unusual emphatic punctuation detected ('!!', '!?')")
        if vague_score > 0.25:
            indicators.append("Vague or unverifiable source attribution ('sources say', 'reportedly')")
        if cred_score > 0.30:
            indicators.append("Contains journalistic credibility markers and attributed citations")

        return {
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(1.0 - fake_prob, 4),
            "indicators": indicators,
            "features": {
                "word_count": num_words,
                "sentence_count": len(sentences),
                "clickbait_score": round(cb_score, 3),
                "emotional_score": round(emo_score, 3),
                "caps_ratio": round(caps_ratio, 3),
                "exclamation_count": exclamation_count,
                "vague_source_score": round(vague_score, 3),
                "credibility_score": round(cred_score, 3),
                "avg_word_length": round(avg_word_len, 1),
                "avg_sentence_length": round(avg_sent_len, 1),
            },
        }

    # ------------------------------------------------------------------
    # HuggingFace Inference API (Optional, Fast Timeout)
    # ------------------------------------------------------------------

    def _bert_analyze(self, text):
        # type: (str) -> Optional[Dict[str, Any]]
        """Call HuggingFace Inference API with a short timeout if configured."""
        try:
            payload = json.dumps({"inputs": text[:512]}).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if self.HF_TOKEN:
                headers["Authorization"] = "Bearer " + self.HF_TOKEN

            req = urllib.request.Request(self.API_URL, data=payload, headers=headers)
            # Short 2.5s timeout so local app is never held up if HF is offline
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                raw = resp.read().decode("utf-8")

            data = json.loads(raw)
            if isinstance(data, list) and len(data) > 0:
                inner = data[0] if isinstance(data[0], list) else data
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
        except Exception:
            pass  # Expected when offline or without token

        return None
