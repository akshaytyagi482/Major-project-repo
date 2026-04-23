from __future__ import annotations

from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.core.config import settings
from app.schemas.analysis import AnalysisResponse, TextAnalysisRequest, UrlAnalysisRequest
from app.services.fusion import fuse_results
from app.services.text_detector import analyze_text
from app.services.url_extractor import extract_main_text
from app.services.vision_detector import analyze_image, analyze_video
from app.utils.reporting import build_report_pdf

router = APIRouter(prefix="/api/v1", tags=["analysis"])


def _is_image(filename: str) -> bool:
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def _is_video(filename: str) -> bool:
    return filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))


@router.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text_endpoint(payload: TextAnalysisRequest) -> AnalysisResponse:
    text_result = analyze_text(payload.text)
    return fuse_results([text_result])


@router.post("/analyze/url", response_model=AnalysisResponse)
async def analyze_url_endpoint(payload: UrlAnalysisRequest) -> AnalysisResponse:
    if not payload.url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Only HTTP(S) URLs are supported.")

    try:
        extracted = extract_main_text(payload.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    text_result = analyze_text(extracted.text)
    text_result.metadata["url"] = extracted.url
    text_result.metadata["title"] = extracted.title
    text_result.metadata["extracted_chars"] = len(extracted.text)
    return fuse_results([text_result])


@router.post("/analyze/file", response_model=AnalysisResponse)
async def analyze_file_endpoint(file: UploadFile = File(...)) -> AnalysisResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb}MB upload limit.")

    results = []
    filename = file.filename or "uploaded_file"

    if _is_image(filename):
        results.append(analyze_image(content, filename=filename))
    elif _is_video(filename):
        results.append(analyze_video(content, filename=filename))
    elif filename.lower().endswith(".txt"):
        try:
            decoded = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="Text file must be UTF-8 encoded.") from exc
        results.append(analyze_text(decoded))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    return fuse_results(results)


@router.post("/analyze/multimodal", response_model=AnalysisResponse)
async def analyze_multimodal_endpoint(
    text: str | None = Form(default=None),
    files: List[UploadFile] | None = File(default=None),
) -> AnalysisResponse:
    results = []

    if text:
        results.append(analyze_text(text))

    for upload in files or []:
        data = await upload.read()
        if not data:
            continue
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"One of the files exceeds {settings.max_upload_mb}MB upload limit.")
        filename = upload.filename or "upload"
        if _is_image(filename):
            results.append(analyze_image(data, filename=filename))
        elif _is_video(filename):
            results.append(analyze_video(data, filename=filename))
        elif filename.lower().endswith(".txt"):
            results.append(analyze_text(data.decode("utf-8", errors="ignore")))

    if not results:
        raise HTTPException(status_code=400, detail="Submit at least text or one file.")

    return fuse_results(results)


@router.post("/report/pdf", response_class=Response)
async def generate_pdf_report(payload: AnalysisResponse) -> Response:
    pdf_bytes = build_report_pdf(payload, source_label="User submitted content")
    headers = {"Content-Disposition": "attachment; filename=detection_report.pdf"}
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
