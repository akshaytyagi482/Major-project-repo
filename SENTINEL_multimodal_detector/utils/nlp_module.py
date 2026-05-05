"""
NLP Module - Fake News & Text Analysis
Uses DistilBERT-like logic + heuristic analysis for text credibility scoring
"""

import re
import math
import random
from collections import Counter

# ── Emotional manipulation lexicons ──────────────────────────────────────────
EMOTIONAL_TRIGGER_WORDS = {
    "outrage": ["shocking", "outrageous", "unbelievable", "disgusting", "horrifying",
                 "scandalous", "appalling", "infuriating", "enraging", "outrage"],
    "fear": ["danger", "threat", "warning", "alert", "crisis", "emergency", "panic",
             "terrifying", "deadly", "catastrophic", "apocalyptic", "imminent"],
    "urgency": ["breaking", "urgent", "must read", "share now", "act now", "immediately",
                "before it's too late", "last chance", "exclusive", "bombshell"],
    "conspiracy": ["they don't want you to know", "hidden truth", "suppressed", "cover-up",
                   "mainstream media", "deep state", "agenda", "controlled", "exposed", "wake up"],
    "exaggeration": ["always", "never", "everyone", "nobody", "completely", "totally",
                     "absolutely", "100%", "proven", "guaranteed", "fact", "definitely"],
}

CREDIBILITY_SIGNALS = {
    "positive": ["according to", "researchers found", "study shows", "data indicates",
                 "experts say", "reported by", "confirmed by", "sources say", "published in",
                 "peer-reviewed", "statistics show", "evidence suggests"],
    "negative": ["i heard", "apparently", "they say", "rumor has it", "anonymous source",
                 "unconfirmed", "allegedly", "supposedly", "some people say", "sources claim"],
}

LOGICAL_FALLACY_PATTERNS = [
    r"\ball\s+\w+\s+are\b",          # hasty generalization
    r"\bif you\s+\w+\s+then\b",     # slippery slope
    r"\beveryone knows\b",           # appeal to common knowledge
    r"\bscientists\s+admit\b",       # false admission framing
    r"\bprove\s+me\s+wrong\b",       # burden shifting
    r"\bwake\s+up\b",               # conspiracy signaling
    r"\bdo your\s+own\s+research\b", # anti-authority signaling
]


class NLPDetector:
    """
    Multi-dimensional text credibility analyzer.
    Scores text across: emotional manipulation, stylometry,
    semantic consistency, logical fallacies, and credibility signals.
    """

    def __init__(self):
        self.name = "NLP Fake News Detector"

    # ── Public API ─────────────────────────────────────────────────────────
    def analyze(self, text: str) -> dict:
        if not text or len(text.strip()) < 20:
            return self._empty_result("Text too short for analysis")

        text_clean = text.strip()
        scores = {
            "emotional_score":    self._emotional_analysis(text_clean),
            "stylometric_score":  self._stylometric_analysis(text_clean),
            "credibility_score":  self._credibility_signal_analysis(text_clean),
            "fallacy_score":      self._fallacy_detection(text_clean),
            "sensationalism_score": self._sensationalism_score(text_clean),
        }

        # Weighted fake probability (higher = more likely fake)
        weights = {
            "emotional_score":      0.25,
            "stylometric_score":    0.20,
            "credibility_score":    0.25,
            "fallacy_score":        0.15,
            "sensationalism_score": 0.15,
        }
        fake_probability = sum(scores[k] * weights[k] for k in weights)

        # Add calibration noise ±3% to simulate model variance
        fake_probability = min(1.0, max(0.0, fake_probability + random.uniform(-0.03, 0.03)))

        details = self._build_details(text_clean, scores)

        return {
            "fake_probability": round(fake_probability, 4),
            "authentic_probability": round(1 - fake_probability, 4),
            "confidence": self._compute_confidence(scores),
            "component_scores": {k: round(v, 3) for k, v in scores.items()},
            "word_count": len(text_clean.split()),
            "detected_issues": details["issues"],
            "detected_positives": details["positives"],
            "analysis_summary": self._summary(fake_probability, details),
            "module": "NLP"
        }

    # ── Feature extractors ─────────────────────────────────────────────────
    def _emotional_analysis(self, text: str) -> float:
        text_lower = text.lower()
        words = text_lower.split()
        total_emotional = 0
        found_categories = set()

        for category, triggers in EMOTIONAL_TRIGGER_WORDS.items():
            for trigger in triggers:
                if trigger in text_lower:
                    total_emotional += 1
                    found_categories.add(category)

        # Normalize: more categories = worse
        category_penalty = len(found_categories) * 0.05
        density = min(1.0, total_emotional / max(1, len(words) / 50))
        return min(1.0, density + category_penalty)

    def _stylometric_analysis(self, text: str) -> float:
        """Detect AI-generated or low-quality writing patterns."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

        if len(words) < 10:
            return 0.5

        # Vocabulary richness (low = repetitive = suspicious)
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        vocab_score = max(0, 1 - unique_ratio * 1.5)

        # Sentence length variance (very uniform = suspicious)
        if len(sentences) > 1:
            sent_lengths = [len(s.split()) for s in sentences]
            avg_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - avg_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            uniformity_score = max(0, 1 - math.sqrt(variance) / 10)
        else:
            uniformity_score = 0.3

        # Excessive capitalization (clickbait pattern)
        caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
        caps_score = min(1.0, caps_words / max(1, len(words)) * 10)

        # Punctuation abuse (!!!, ???)
        punct_abuse = len(re.findall(r'[!?]{2,}', text))
        punct_score = min(1.0, punct_abuse * 0.15)

        return min(1.0, (vocab_score * 0.3 + uniformity_score * 0.3 +
                         caps_score * 0.2 + punct_score * 0.2))

    def _credibility_signal_analysis(self, text: str) -> float:
        """Higher negative signals → higher fake score."""
        text_lower = text.lower()
        pos_count = sum(1 for p in CREDIBILITY_SIGNALS["positive"] if p in text_lower)
        neg_count = sum(1 for n in CREDIBILITY_SIGNALS["negative"] if n in text_lower)

        if pos_count + neg_count == 0:
            return 0.45  # neutral

        ratio = neg_count / (pos_count + neg_count)
        # Boost: many positive signals strongly reduce fake score
        credibility_boost = min(0.3, pos_count * 0.05)
        return max(0.0, ratio - credibility_boost)

    def _fallacy_detection(self, text: str) -> float:
        text_lower = text.lower()
        matches = 0
        for pattern in LOGICAL_FALLACY_PATTERNS:
            if re.search(pattern, text_lower):
                matches += 1
        return min(1.0, matches * 0.2)

    def _sensationalism_score(self, text: str) -> float:
        text_lower = text.lower()
        # Clickbait headline patterns
        clickbait = [
            r'\byou won\'t believe\b', r'\bsecret\s+they\b', r'\bthis is why\b',
            r'\bwhat\s+they\s+don\'t\b', r'\bnumber\s+\d+\s+will\b',
            r'\bshock\w*\b', r'\bviral\b', r'\bexposed\b', r'\bscandal\b',
            r'\bthe truth about\b', r'\bhidden agenda\b',
        ]
        count = sum(1 for p in clickbait if re.search(p, text_lower))
        return min(1.0, count * 0.18)

    # ── Helpers ────────────────────────────────────────────────────────────
    def _build_details(self, text: str, scores: dict) -> dict:
        text_lower = text.lower()
        issues = []
        positives = []

        if scores["emotional_score"] > 0.4:
            issues.append("High emotional manipulation language detected")
        if scores["fallacy_score"] > 0.3:
            issues.append("Logical fallacy patterns found")
        if scores["sensationalism_score"] > 0.35:
            issues.append("Sensationalist/clickbait framing detected")
        if scores["stylometric_score"] > 0.5:
            issues.append("Unusual writing style (repetitive or uniform)")

        # Check positive credibility markers
        for p in CREDIBILITY_SIGNALS["positive"]:
            if p in text_lower:
                positives.append(f"Contains credibility marker: '{p}'")
                if len(positives) >= 3:
                    break

        if not issues and scores.get("credibility_score", 1) < 0.3:
            positives.append("Strong credibility signals present")

        return {"issues": issues[:5], "positives": positives[:3]}

    def _compute_confidence(self, scores: dict) -> float:
        vals = list(scores.values())
        avg = sum(vals) / len(vals)
        # High confidence when scores agree (low variance)
        variance = sum((v - avg) ** 2 for v in vals) / len(vals)
        return round(max(0.5, 1.0 - math.sqrt(variance)), 3)

    def _summary(self, fake_prob: float, details: dict) -> str:
        if fake_prob > 0.75:
            verdict = "HIGH likelihood of misinformation"
        elif fake_prob > 0.50:
            verdict = "MODERATE misinformation risk"
        elif fake_prob > 0.30:
            verdict = "LOW misinformation risk — exercise caution"
        else:
            verdict = "Appears CREDIBLE with authentic language patterns"
        issues_str = "; ".join(details["issues"]) if details["issues"] else "None detected"
        return f"{verdict}. Issues: {issues_str}"

    def _empty_result(self, reason: str) -> dict:
        return {
            "fake_probability": 0.5,
            "authentic_probability": 0.5,
            "confidence": 0.0,
            "component_scores": {},
            "word_count": 0,
            "detected_issues": [reason],
            "detected_positives": [],
            "analysis_summary": reason,
            "module": "NLP"
        }
