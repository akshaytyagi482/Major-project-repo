import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from app.schemas.analysis import ModalityResult


SENSATIONAL_TERMS = {
    "shocking",
    "exposed",
    "breaking",
    "unbelievable",
    "secret",
    "conspiracy",
    "banned",
    "viral",
    "must watch",
    "urgent",
    "traitor",
    "destroyed",
}

EMOTIONAL_TERMS = {
    "outrage",
    "fear",
    "panic",
    "anger",
    "hate",
    "betrayal",
    "threat",
    "chaos",
    "catastrophe",
    "disaster",
}


@dataclass
class TextSignal:
    feature: str
    score: float
    detail: str


_MODEL = None
_VECTORIZER = None


def _bootstrap_training_data() -> tuple[list[str], list[int]]:
    real_samples = [
        "The city council approved a new public transportation budget after a 6-hour meeting.",
        "Researchers published peer-reviewed findings on climate adaptation in coastal regions.",
        "The central bank maintained interest rates and released quarterly inflation estimates.",
        "A local hospital inaugurated a new emergency wing with 120 additional beds.",
        "The election commission released official voter turnout statistics by district.",
        "The university announced scholarship programs for students in engineering and medicine.",
        "The ministry confirmed the policy changes through an official press release.",
        "The report includes methodology, data sources, and limitations for transparency.",
        "International observers documented the ceasefire talks and participant statements.",
        "The company filed audited financial statements with the securities regulator.",
    ]

    fake_samples = [
        "Shocking truth exposed! Government secret finally leaked and everyone is in danger!",
        "Breaking: unbelievable conspiracy will destroy the nation by midnight!!!",
        "Urgent viral alert: banned cure hidden by elites and traitors in power.",
        "Must watch now! Catastrophic betrayal uncovered in secret underground lab.",
        "Panic and chaos everywhere as hidden agenda takes over your city tonight.",
        "Outrage grows after unbelievable fake evidence shows shocking scandal.",
        "This secret report proves everything they told you is a lie!!!",
        "Fear spreads as conspiracy insiders reveal total collapse is guaranteed.",
        "Destroyed in minutes: traitor network exposed by anonymous viral source.",
        "Emergency warning: chaos, threat, panic, betrayal, and disaster incoming.",
    ]

    texts = real_samples + fake_samples
    labels = [0] * len(real_samples) + [1] * len(fake_samples)
    return texts, labels


def _ensure_model() -> tuple[TfidfVectorizer, LogisticRegression]:
    global _MODEL, _VECTORIZER

    if _MODEL is not None and _VECTORIZER is not None:
        return _VECTORIZER, _MODEL

    texts, labels = _bootstrap_training_data()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=2500)
    x = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(x, labels)

    _VECTORIZER = vectorizer
    _MODEL = model
    return vectorizer, model


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def analyze_text(text: str) -> ModalityResult:
    tokens = _tokenize(text)
    total_words = max(len(tokens), 1)
    counts = Counter(tokens)

    cap_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    exclamations = text.count("!")
    questions = text.count("?")
    sensational_hits = sum(counts[t] for t in SENSATIONAL_TERMS)
    emotional_hits = sum(counts[t] for t in EMOTIONAL_TERMS)

    repeated_words = sum(1 for _, v in counts.items() if v >= 4)
    unique_words = max(len(counts), 1)

    sensational_rate = sensational_hits / total_words
    emotional_rate = emotional_hits / total_words
    repetition_rate = repeated_words / unique_words

    signals: List[TextSignal] = [
        TextSignal("caps_ratio", min(cap_ratio * 3.0, 1.0), f"Uppercase ratio {cap_ratio:.2f}"),
        TextSignal(
            "punctuation_pressure",
            min((exclamations + questions) / max(total_words * 0.2, 4.0), 1.0),
            f"Punctuation bursts: !={exclamations}, ?={questions}",
        ),
        TextSignal("sensational_language", min(sensational_rate * 14.0, 1.0), f"Sensational term hits: {sensational_hits}"),
        TextSignal("emotional_manipulation", min(emotional_rate * 16.0, 1.0), f"Emotional term hits: {emotional_hits}"),
        TextSignal("repetition_pattern", min(repetition_rate * 2.5, 1.0), f"Repeated words >=4: {repeated_words}"),
    ]

    weighted_score = (
        signals[0].score * 0.14
        + signals[1].score * 0.2
        + signals[2].score * 0.29
        + signals[3].score * 0.22
        + signals[4].score * 0.15
    )

    length_factor = min(math.log(total_words + 1, 200), 1.0)
    confidence = max(0.3, min(0.95, 0.45 + 0.5 * length_factor))

    vectorizer, model = _ensure_model()
    model_score = float(model.predict_proba(vectorizer.transform([text]))[0][1])

    score_fake = max(0.0, min(weighted_score * 0.4 + model_score * 0.6, 1.0))
    score_real = 1.0 - score_fake

    if score_fake >= 0.68:
        verdict = "fake"
    elif score_fake >= 0.45:
        verdict = "suspicious"
    else:
        verdict = "real"

    evidence = [f"{s.feature}: {s.detail}" for s in signals if s.score >= 0.25]
    evidence.append(f"ml_classifier_probability_fake: {model_score:.2f}")
    if not evidence:
        evidence = ["No high-risk linguistic signals detected."]

    return ModalityResult(
        modality="text",
        score_fake=round(score_fake, 4),
        score_real=round(score_real, 4),
        confidence=round(confidence, 4),
        verdict=verdict,
        evidence=evidence,
        metadata={
            "word_count": total_words,
            "exclamation_count": exclamations,
            "question_count": questions,
        },
    )
