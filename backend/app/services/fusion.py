from typing import List

from app.schemas.analysis import AnalysisResponse, ModalityResult


def fuse_results(results: List[ModalityResult]) -> AnalysisResponse:
    if not results:
        return AnalysisResponse(
            verdict="suspicious",
            confidence=0.0,
            risk_score=0.5,
            summary="No content was analyzed.",
            modality_results=[],
            recommendations=["Submit text, image, video, or URL content for analysis."],
        )

    total_weight = 0.0
    weighted_fake = 0.0
    weighted_conf = 0.0

    modality_weights = {"text": 0.35, "image": 0.3, "video": 0.35}

    for result in results:
        weight = modality_weights.get(result.modality, 0.2)
        total_weight += weight
        weighted_fake += result.score_fake * weight
        weighted_conf += result.confidence * weight

    risk_score = weighted_fake / total_weight if total_weight else 0.5
    confidence = weighted_conf / total_weight if total_weight else 0.0

    if risk_score >= 0.68:
        verdict = "fake"
    elif risk_score >= 0.45:
        verdict = "suspicious"
    else:
        verdict = "real"

    summary = (
        f"Multimodal analysis indicates {verdict.upper()} content "
        f"with risk score {risk_score:.2f} and confidence {confidence:.2f}."
    )

    recommendations = [
        "Cross-check against reputable fact-checking portals.",
        "Inspect original source metadata and publication timeline.",
        "Use reverse image/video search for provenance verification.",
    ]

    if verdict == "fake":
        recommendations.insert(0, "Avoid sharing this content until verified by trusted sources.")

    return AnalysisResponse(
        verdict=verdict,
        confidence=round(confidence, 4),
        risk_score=round(risk_score, 4),
        summary=summary,
        modality_results=results,
        recommendations=recommendations,
    )
