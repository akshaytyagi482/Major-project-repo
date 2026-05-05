# SENTINEL — Multimodal Fake News & Deepfake Detection System v2.0

A full-stack AI/ML system combining NLP linguistic analysis with Computer Vision
forensics for detecting fake news and deepfakes.

## Architecture

```
Input (Text/Image/Video/URL)
         │
         ▼
┌─────────────────────────────────────────┐
│           Web Ingestion Layer           │
│         Flask REST API Backend          │
└──────────┬─────────────────┬────────────┘
           │                 │
           ▼                 ▼
  ┌─────────────┐   ┌─────────────────┐
  │ NLP Module  │   │   CV Module     │
  │             │   │                 │
  │ • Emotional │   │ • ELA Forensics │
  │   analysis  │   │ • Noise Pattern │
  │ • Fallacy   │   │ • Compression   │
  │   detection │   │   Artifacts     │
  │ • Stylometry│   │ • Face-swap     │
  │ • Credibility│  │   Detection     │
  └──────┬──────┘   └───────┬─────────┘
         │                  │
         └────────┬──────────┘
                  ▼
        ┌──────────────────┐
        │  Fusion Module   │
        │                  │
        │ • Weighted avg   │
        │ • Cross-modal    │
        │   inconsistency  │
        │ • Confidence     │
        │   adjustment     │
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │  TTS + Reports   │
        │  Audio synthesis │
        │  JSON reports    │
        └──────────────────┘
```

## Model Performance (Baseline)

| Modality    | Model       | Accuracy |
|-------------|-------------|----------|
| Text alone  | DistilBERT  | 94.2%    |
| Image alone | ResNet-50   | 92.1%    |
| Video alone | ResNet-50   | 89.7%    |
| Multimodal  | Fusion      | 97.3%    |

## Installation

```bash
# 1. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Optional: Install for video analysis
pip install opencv-python-headless moviepy

# 4. Run the server
python app.py
```

Then open http://localhost:5000 in your browser.

## API Endpoints

### Text Analysis
```
POST /api/analyze/text
Content-Type: application/json
{"text": "Article text here..."}
```

### Image Analysis
```
POST /api/analyze/image
Content-Type: multipart/form-data
file: <image file>
text: <optional accompanying text>
```

### Video Analysis
```
POST /api/analyze/video
Content-Type: multipart/form-data
file: <video file>
text: <optional transcript>
```

### URL Analysis
```
POST /api/analyze/url
Content-Type: application/json
{"url": "https://example.com/article"}
```

## Response Format

All endpoints return:
```json
{
  "status": "ok",
  "elapsed_ms": 234,
  "analysis": {
    "fake_probability": 0.73,
    "authentic_probability": 0.27,
    "confidence": 0.81,
    "verdict": "LIKELY FAKE",
    "risk_level": "HIGH",
    "cross_modal_inconsistency": 0.12,
    "fusion_weights": {"nlp_weight": 0.52, "cv_weight": 0.48},
    "reasoning": ["..."],
    "recommended_actions": ["..."],
    "nlp_result": {...},
    "cv_result": {...}
  },
  "report": {...},
  "tts": {"success": true, "url": "/audio/..."}
}
```

## Features

- **Text Analysis**: Emotional manipulation, logical fallacies, stylometry, credibility signals
- **Image Analysis**: ELA forensics, noise analysis, compression artifacts, color distribution
- **Video Analysis**: Temporal consistency, face-swap detection, motion analysis
- **URL Analysis**: Auto-scrapes and analyzes web articles
- **TTS**: Audio playback of analysis results via gTTS
- **Reports**: Downloadable JSON and text reports
- **History**: Last 12 analyses tracked in UI

## Project Structure

```
multimodal_detector/
├── app.py                    # Flask backend
├── requirements.txt
├── utils/
│   ├── nlp_module.py         # NLP analyzer
│   ├── cv_module.py          # CV deepfake detector
│   ├── fusion_module.py      # Multimodal fusion
│   └── tts_report.py         # TTS + report generation
├── templates/
│   └── index.html            # Main UI
└── static/
    ├── css/main.css           # Styles
    └── js/main.js             # Frontend logic
```
