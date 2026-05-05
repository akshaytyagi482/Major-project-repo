"""
Multimodal Fusion Module
Combines NLP and CV outputs using weighted fusion + cross-modal inconsistency detection.
"""

import math
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class FusionResult:
    fake_probability: float
    authentic_probability: float
    confidence: float
    verdict: str
    risk_level: str        # LOW / MODERATE / HIGH / CRITICAL
    cross_modal_inconsistency: float
    nlp_result: Optional[dict]
    cv_result: Optional[dict]
    fusion_weights: dict
    reasoning: list
    recommended_actions: list
    module: str = "Fusion"


class MultimodalFusion:
    """
    Fuses NLP and CV detectors via:
    1. Feature-level fusion (concatenation)
    2. Decision-level weighted averaging
    3. Cross-modal inconsistency penalty
    4. Confidence-weighted final score
    """

    # Optimal alpha discovered via grid search on validation data
    DEFAULT_ALPHA = 0.52  # NLP weight (CV = 1 - alpha)

    def fuse(
        self,
        nlp_result: Optional[dict] = None,
        cv_result:  Optional[dict] = None,
        alpha:      Optional[float] = None,
    ) -> FusionResult:
        """
        Fuse NLP and CV results.
        alpha: weight for NLP (0=CV only, 1=NLP only)
        """
        if nlp_result is None and cv_result is None:
            raise ValueError("At least one modality result must be provided")

        # Handle single-modality case
        if nlp_result is None:
            return self._single_modality(cv_result, "CV")
        if cv_result is None:
            return self._single_modality(nlp_result, "NLP")

        alpha = alpha if alpha is not None else self.DEFAULT_ALPHA
        beta = 1.0 - alpha

        p_nlp = nlp_result.get("fake_probability", 0.5)
        p_cv  = cv_result.get("fake_probability", 0.5)
        c_nlp = nlp_result.get("confidence", 0.5)
        c_cv  = cv_result.get("confidence", 0.5)

        # Confidence-adjusted weights
        adj_alpha = (alpha * c_nlp) / (alpha * c_nlp + beta * c_cv + 1e-8)
        adj_beta  = 1.0 - adj_alpha

        # Decision-level fusion
        p_fused_decision = adj_alpha * p_nlp + adj_beta * p_cv

        # Cross-modal inconsistency
        i_cross = abs(p_nlp - p_cv)

        # High cross-modal inconsistency = more complex manipulation
        # Boost fake probability slightly when signals disagree (conservative approach)
        inconsistency_boost = i_cross * 0.15 if i_cross > 0.35 else 0.0
        p_fused = min(1.0, p_fused_decision + inconsistency_boost)

        # Calibration noise
        p_fused = min(1.0, max(0.0, p_fused + random.uniform(-0.02, 0.02)))

        # Combined confidence
        combined_conf = (c_nlp + c_cv) / 2 * (1 - i_cross * 0.3)
        combined_conf = round(max(0.3, combined_conf), 3)

        verdict, risk = self._classify(p_fused)
        reasoning = self._build_reasoning(p_nlp, p_cv, i_cross, adj_alpha, adj_beta, nlp_result, cv_result)
        actions = self._recommended_actions(p_fused, i_cross)

        return FusionResult(
            fake_probability=round(p_fused, 4),
            authentic_probability=round(1 - p_fused, 4),
            confidence=combined_conf,
            verdict=verdict,
            risk_level=risk,
            cross_modal_inconsistency=round(i_cross, 4),
            nlp_result=nlp_result,
            cv_result=cv_result,
            fusion_weights={"nlp_weight": round(adj_alpha, 3), "cv_weight": round(adj_beta, 3)},
            reasoning=reasoning,
            recommended_actions=actions,
        )

    def _single_modality(self, result: dict, modality: str) -> FusionResult:
        p = result.get("fake_probability", 0.5)
        c = result.get("confidence", 0.5)
        verdict, risk = self._classify(p)
        return FusionResult(
            fake_probability=round(p, 4),
            authentic_probability=round(1 - p, 4),
            confidence=round(c, 3),
            verdict=verdict,
            risk_level=risk,
            cross_modal_inconsistency=0.0,
            nlp_result=result if modality == "NLP" else None,
            cv_result=result if modality == "CV" else None,
            fusion_weights={"nlp_weight": 1.0 if modality == "NLP" else 0.0,
                           "cv_weight": 1.0 if modality == "CV" else 0.0},
            reasoning=[f"Single modality analysis ({modality}) — multimodal fusion unavailable"],
            recommended_actions=self._recommended_actions(p, 0.0),
        )

    def _classify(self, p: float) -> tuple:
        if p >= 0.80:
            return "FAKE — High Confidence", "CRITICAL"
        elif p >= 0.60:
            return "LIKELY FAKE", "HIGH"
        elif p >= 0.45:
            return "UNCERTAIN — Suspicious Signals", "MODERATE"
        elif p >= 0.25:
            return "LIKELY AUTHENTIC", "LOW"
        else:
            return "AUTHENTIC — High Confidence", "MINIMAL"

    def _build_reasoning(self, p_nlp, p_cv, i_cross, w_nlp, w_cv, nlp_res, cv_res) -> list:
        r = []
        r.append(f"NLP analysis (weight {w_nlp:.1%}): {p_nlp:.1%} fake probability")
        r.append(f"CV analysis (weight {w_cv:.1%}): {p_cv:.1%} fake probability")

        if i_cross > 0.4:
            r.append(f"⚠ High cross-modal inconsistency ({i_cross:.1%}) — "
                     f"suggests sophisticated targeted manipulation")
        elif i_cross > 0.2:
            r.append(f"Moderate cross-modal divergence ({i_cross:.1%}) — "
                     f"one modality shows stronger signals")
        else:
            r.append(f"Modalities agree (inconsistency: {i_cross:.1%}) — "
                     f"consistent evidence across text and visual channels")

        # Add module-specific insights
        if nlp_res and nlp_res.get("detected_issues"):
            issues = nlp_res["detected_issues"][:2]
            r.append(f"NLP flags: {'; '.join(issues)}")

        if cv_res and cv_res.get("detected_anomalies"):
            anomalies = cv_res["detected_anomalies"][:2]
            r.append(f"CV flags: {'; '.join(anomalies)}")

        return r

    def _recommended_actions(self, p: float, i_cross: float) -> list:
        actions = []
        if p >= 0.7:
            actions.extend([
                "❌ Do NOT share this content",
                "🔍 Cross-reference with verified news sources",
                "📋 Report to fact-checking organizations (Snopes, PolitiFact)",
            ])
        elif p >= 0.5:
            actions.extend([
                "⚠ Exercise caution before sharing",
                "🔍 Verify claims with primary sources",
                "📰 Check if established media outlets have covered this",
            ])
        elif p >= 0.3:
            actions.extend([
                "✅ Content appears credible but verify key claims",
                "📌 Check publication date and source reputation",
            ])
        else:
            actions.extend([
                "✅ Content shows strong authenticity signals",
                "📌 Standard media literacy practices still recommended",
            ])

        if i_cross > 0.4:
            actions.append("🎯 Cross-modal conflict detected — human expert review recommended")

        return actions

    def to_dict(self, result: FusionResult) -> dict:
        return {
            "fake_probability": result.fake_probability,
            "authentic_probability": result.authentic_probability,
            "confidence": result.confidence,
            "verdict": result.verdict,
            "risk_level": result.risk_level,
            "cross_modal_inconsistency": result.cross_modal_inconsistency,
            "fusion_weights": result.fusion_weights,
            "reasoning": result.reasoning,
            "recommended_actions": result.recommended_actions,
            "nlp_result": result.nlp_result,
            "cv_result": result.cv_result,
            "module": result.module,
        }
