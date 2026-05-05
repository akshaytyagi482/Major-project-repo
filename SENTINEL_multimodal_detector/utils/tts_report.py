"""
TTS & Report Generation
Text-to-speech for analysis results + PDF/JSON report generation.
"""

import json
import os
import datetime
import hashlib


class TTSEngine:
    """Text-to-Speech for analysis summaries using gTTS."""

    def __init__(self, output_dir: str = "static/audio"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def synthesize(self, text: str, filename_hint: str = "result") -> dict:
        """Generate audio from text. Returns file path or error."""
        try:
            from gtts import gTTS
            clean_text = self._clean_for_speech(text)
            tts = gTTS(text=clean_text, lang="en", slow=False)
            filename = f"{filename_hint}_{hashlib.md5(text.encode()).hexdigest()[:8]}.mp3"
            filepath = os.path.join(self.output_dir, filename)
            tts.save(filepath)
            return {"success": True, "path": filepath, "url": f"/audio/{filename}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def generate_analysis_speech(self, fusion_result: dict) -> dict:
        """Convert fusion analysis to natural speech."""
        verdict = fusion_result.get("verdict", "Unknown")
        confidence = fusion_result.get("confidence", 0)
        fake_prob = fusion_result.get("fake_probability", 0)
        risk = fusion_result.get("risk_level", "unknown")

        speech_text = (
            f"Analysis complete. "
            f"Verdict: {verdict}. "
            f"Risk level: {risk}. "
            f"The content has a {fake_prob:.0%} probability of being fake or manipulated. "
            f"Confidence in this assessment: {confidence:.0%}. "
        )

        reasoning = fusion_result.get("reasoning", [])
        if reasoning:
            speech_text += "Key findings: " + ". ".join(reasoning[:2]) + ". "

        actions = fusion_result.get("recommended_actions", [])
        if actions:
            clean_actions = [a.replace("❌", "").replace("⚠", "").replace("✅", "")
                             .replace("🔍", "").replace("📋", "").replace("🎯", "")
                             .replace("📌", "").replace("📰", "").strip()
                             for a in actions[:2]]
            speech_text += "Recommended actions: " + ". ".join(clean_actions) + "."

        return self.synthesize(speech_text, "analysis")

    def _clean_for_speech(self, text: str) -> str:
        """Remove markdown, emojis, and symbols for clean TTS output."""
        import re
        # Remove emoji unicode ranges
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F1FF"
            u"\U00002600-\U000027BF"
            u"\U0001F900-\U0001F9FF"
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub("", text)
        text = re.sub(r"[*_`#]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


class ReportGenerator:
    """Generate analysis reports in JSON and human-readable formats."""

    def generate_json_report(self, fusion_result: dict, media_info: dict) -> dict:
        """Full structured JSON report."""
        return {
            "report_id": self._generate_id(),
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "system": "Multimodal Fake News & Deepfake Detection System v2.0",
            "media_analyzed": media_info,
            "executive_summary": {
                "verdict": fusion_result.get("verdict"),
                "risk_level": fusion_result.get("risk_level"),
                "fake_probability": fusion_result.get("fake_probability"),
                "confidence": fusion_result.get("confidence"),
                "cross_modal_inconsistency": fusion_result.get("cross_modal_inconsistency"),
            },
            "detailed_analysis": {
                "fusion_weights": fusion_result.get("fusion_weights"),
                "reasoning": fusion_result.get("reasoning"),
                "nlp_analysis": fusion_result.get("nlp_result"),
                "cv_analysis": fusion_result.get("cv_result"),
            },
            "recommendations": fusion_result.get("recommended_actions"),
            "methodology": {
                "nlp_model": "DistilBERT-based linguistic analyzer",
                "cv_model": "ResNet-50 + ELA forensics pipeline",
                "fusion": "Confidence-weighted decision fusion with cross-modal inconsistency detection",
                "threshold": 0.5,
            }
        }

    def generate_text_report(self, fusion_result: dict) -> str:
        """Human-readable text report."""
        lines = [
            "=" * 60,
            "MULTIMODAL DETECTION ANALYSIS REPORT",
            f"Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 60,
            "",
            f"VERDICT:     {fusion_result.get('verdict', 'N/A')}",
            f"RISK LEVEL:  {fusion_result.get('risk_level', 'N/A')}",
            f"FAKE PROB:   {fusion_result.get('fake_probability', 0):.1%}",
            f"CONFIDENCE:  {fusion_result.get('confidence', 0):.1%}",
            f"CROSS-MODAL: {fusion_result.get('cross_modal_inconsistency', 0):.1%}",
            "",
            "REASONING:",
        ]
        for r in fusion_result.get("reasoning", []):
            lines.append(f"  • {r}")

        lines += ["", "RECOMMENDED ACTIONS:"]
        for a in fusion_result.get("recommended_actions", []):
            lines.append(f"  {a}")

        if fusion_result.get("nlp_result"):
            nlp = fusion_result["nlp_result"]
            lines += [
                "",
                "NLP ANALYSIS:",
                f"  Fake probability:  {nlp.get('fake_probability', 0):.1%}",
                f"  Word count:        {nlp.get('word_count', 0)}",
                f"  Confidence:        {nlp.get('confidence', 0):.1%}",
            ]
            for issue in nlp.get("detected_issues", []):
                lines.append(f"  ⚠ {issue}")

        if fusion_result.get("cv_result"):
            cv = fusion_result["cv_result"]
            lines += [
                "",
                "CV ANALYSIS:",
                f"  Fake probability:  {cv.get('fake_probability', 0):.1%}",
                f"  Confidence:        {cv.get('confidence', 0):.1%}",
            ]
            for a in cv.get("detected_anomalies", []):
                lines.append(f"  ⚠ {a}")

        lines += ["", "=" * 60]
        return "\n".join(lines)

    def _generate_id(self) -> str:
        import time
        return hashlib.md5(str(time.time()).encode()).hexdigest()[:12].upper()
