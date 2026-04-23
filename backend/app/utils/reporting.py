from datetime import datetime
from io import BytesIO

from fpdf import FPDF

from app.schemas.analysis import AnalysisResponse


class PDF(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 10, "Multimodal Fake News and Deepfake Detection Report", ln=True)
        self.ln(1)


def build_report_pdf(result: AnalysisResponse, source_label: str) -> bytes:
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Generated: {datetime.utcnow().isoformat()} UTC", ln=True)
    pdf.cell(0, 8, f"Source: {source_label}", ln=True)
    pdf.cell(0, 8, f"Final Verdict: {result.verdict.upper()}", ln=True)
    pdf.cell(0, 8, f"Risk Score: {result.risk_score:.2f}", ln=True)
    pdf.cell(0, 8, f"Confidence: {result.confidence:.2f}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, result.summary)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Modality Breakdown", ln=True)
    pdf.set_font("Helvetica", "", 10)

    for idx, modality in enumerate(result.modality_results, start=1):
        pdf.multi_cell(
            0,
            7,
            (
                f"{idx}. {modality.modality.upper()} | Verdict: {modality.verdict.upper()} | "
                f"Fake Score: {modality.score_fake:.2f} | Confidence: {modality.confidence:.2f}"
            ),
        )
        for ev in modality.evidence:
            pdf.multi_cell(0, 6, f"   - {ev}")
        pdf.ln(1)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Recommendations", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for rec in result.recommendations:
        pdf.multi_cell(0, 7, f"- {rec}")

    raw_pdf = pdf.output(dest="S")
    if isinstance(raw_pdf, (bytes, bytearray)):
        return bytes(raw_pdf)

    buffer = BytesIO()
    buffer.write(raw_pdf.encode("latin-1"))
    return buffer.getvalue()
