"""
Multimodal Fake News & Deepfake Detection System
Flask Backend — REST API
"""

# Disable CUDA and set up environment BEFORE importing PyTorch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_LOGGING_LEVEL'] = '3'

import json
import uuid
import time
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Local modules
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.nlp_module import NLPDetector
from utils.cv_module import CVDetector
from utils.fusion_module import MultimodalFusion
from utils.tts_report import TTSEngine, ReportGenerator

# ── App Setup ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
AUDIO_DIR  = BASE_DIR / "static" / "audio"
UPLOAD_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True, parents=True)

ALLOWED_IMAGE = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}
ALLOWED_VIDEO = {"mp4", "avi", "mov", "mkv", "webm", "flv"}
ALLOWED_ALL   = ALLOWED_IMAGE | ALLOWED_VIDEO
MAX_FILE_MB   = 50

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024
CORS(app)

# ── Module Instances ───────────────────────────────────────────────────────
nlp_detector = NLPDetector()
cv_detector  = CVDetector()
fusion       = MultimodalFusion()
tts_engine   = TTSEngine(str(AUDIO_DIR))
reporter     = ReportGenerator()

# ── Helpers ────────────────────────────────────────────────────────────────
def allowed_file(filename: str, allowed: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed

def save_upload(file, allowed: set) -> tuple[str | None, str | None]:
    """Save uploaded file and return (path, error)."""
    if not file or file.filename == "":
        return None, "No file selected"
    if not allowed_file(file.filename, allowed):
        return None, f"Unsupported file type. Allowed: {', '.join(sorted(allowed))}"
    ext = file.filename.rsplit(".", 1)[1].lower()
    name = f"{uuid.uuid4().hex}.{ext}"
    path = str(UPLOAD_DIR / name)
    file.save(path)
    return path, None

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/audio/<path:filename>")
def serve_audio(filename):
    return send_from_directory(str(AUDIO_DIR), filename)

@app.route("/api/health")
def health():
    return jsonify({"status": "online", "modules": ["NLP", "CV", "Fusion", "TTS"]})

# ── Text Analysis ──────────────────────────────────────────────────────────
@app.route("/api/analyze/text", methods=["POST"])
def analyze_text():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) > 50000:
        return jsonify({"error": "Text too long (max 50,000 chars)"}), 400

    t0 = time.time()
    nlp_result = nlp_detector.analyze(text)
    fusion_result = fusion.to_dict(fusion.fuse(nlp_result=nlp_result))

    # TTS
    tts = tts_engine.generate_analysis_speech(fusion_result)

    report = reporter.generate_json_report(
        fusion_result, {"type": "text", "length": len(text)}
    )

    return jsonify({
        "status": "ok",
        "elapsed_ms": round((time.time() - t0) * 1000),
        "analysis": fusion_result,
        "report": report,
        "tts": tts,
    })

# ── Image Analysis ─────────────────────────────────────────────────────────
@app.route("/api/analyze/image", methods=["POST"])
def analyze_image():
    text = request.form.get("text", "").strip()
    file = request.files.get("file")
    if not file and not text:
        return jsonify({"error": "Provide an image file and/or text"}), 400

    t0 = time.time()
    nlp_result = nlp_detector.analyze(text) if text else None
    cv_result  = None

    if file:
        path, err = save_upload(file, ALLOWED_IMAGE)
        if err:
            return jsonify({"error": err}), 400
        try:
            cv_result = cv_detector.analyze_image(path)
        finally:
            try: os.remove(path)
            except: pass

    fusion_result = fusion.to_dict(fusion.fuse(nlp_result=nlp_result, cv_result=cv_result))
    tts = tts_engine.generate_analysis_speech(fusion_result)
    report = reporter.generate_json_report(
        fusion_result, {"type": "image+text" if text else "image",
                        "filename": file.filename if file else None}
    )

    return jsonify({
        "status": "ok",
        "elapsed_ms": round((time.time() - t0) * 1000),
        "analysis": fusion_result,
        "report": report,
        "tts": tts,
    })

# ── Video Analysis ─────────────────────────────────────────────────────────
@app.route("/api/analyze/video", methods=["POST"])
def analyze_video():
    text = request.form.get("text", "").strip()
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "Provide a video file"}), 400

    path, err = save_upload(file, ALLOWED_VIDEO)
    if err:
        return jsonify({"error": err}), 400

    t0 = time.time()
    nlp_result = nlp_detector.analyze(text) if text else None
    cv_result  = None
    try:
        cv_result = cv_detector.analyze_video(path)
    finally:
        try: os.remove(path)
        except: pass

    fusion_result = fusion.to_dict(fusion.fuse(nlp_result=nlp_result, cv_result=cv_result))
    tts = tts_engine.generate_analysis_speech(fusion_result)
    report = reporter.generate_json_report(
        fusion_result, {"type": "video", "filename": file.filename}
    )

    return jsonify({
        "status": "ok",
        "elapsed_ms": round((time.time() - t0) * 1000),
        "analysis": fusion_result,
        "report": report,
        "tts": tts,
    })

# ── URL Scrape + Analyze ───────────────────────────────────────────────────
@app.route("/api/analyze/url", methods=["POST"])
def analyze_url():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    t0 = time.time()
    text = _scrape_url(url)
    if not text:
        return jsonify({"error": "Could not extract text from URL"}), 422

    nlp_result = nlp_detector.analyze(text)
    fusion_result = fusion.to_dict(fusion.fuse(nlp_result=nlp_result))
    tts = tts_engine.generate_analysis_speech(fusion_result)
    report = reporter.generate_json_report(
        fusion_result, {"type": "url", "url": url, "extracted_chars": len(text)}
    )

    return jsonify({
        "status": "ok",
        "elapsed_ms": round((time.time() - t0) * 1000),
        "extracted_text_preview": text[:300] + "..." if len(text) > 300 else text,
        "analysis": fusion_result,
        "report": report,
        "tts": tts,
    })

# ── Report Download ────────────────────────────────────────────────────────
@app.route("/api/report/text", methods=["POST"])
def text_report():
    data = request.get_json(silent=True) or {}
    fusion_result = data.get("fusion_result")
    if not fusion_result:
        return jsonify({"error": "No fusion result provided"}), 400
    text = reporter.generate_text_report(fusion_result)
    return jsonify({"report": text})

# ── Utility ────────────────────────────────────────────────────────────────
def _scrape_url(url: str) -> str:
    """Simple URL text extractor."""
    try:
        import urllib.request
        import html
        import re
        headers = {"User-Agent": "Mozilla/5.0 (compatible; FakeNewsDetector/1.0)"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")

        # Strip HTML tags
        raw = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
        raw = re.sub(r"<style[^>]*>.*?</style>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
        raw = re.sub(r"<[^>]+>", " ", raw)
        raw = html.unescape(raw)
        raw = re.sub(r"\s+", " ", raw).strip()
        return raw[:8000]  # Cap at 8000 chars
    except Exception:
        return ""

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🛡  Multimodal Fake News & Deepfake Detector — Starting...")
    print("   NLP Module:    DistilBERT-based linguistic analyzer")
    print("   CV Module:     ResNet-50 + ELA forensics pipeline")
    print("   Fusion:        Confidence-weighted multimodal fusion")
    print("   TTS:           gTTS audio synthesis")
    print("   Server:        http://localhost:5000")
    print("─" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)
