# Multimodal Fake News and Deepfake Detection System

A full-stack web platform for detecting fake news and deepfake content using multimodal analysis of text, images, and videos.

## What this project includes

- FastAPI backend for text, URL, file, and multimodal detection.
- Hybrid ML + heuristic NLP analysis for fake-news signals in text.
- Computer vision analysis for image/video artifact and consistency signals.
- Multimodal fusion engine that combines modality-specific outputs.
- PDF report generation for evidence and archival.
- Next.js frontend dashboard for non-technical users.

## Architecture

- Frontend: Next.js (TypeScript), Recharts, Framer Motion
- Backend: FastAPI (Python)
- AI/ML libs (extensible): NLTK, spaCy, Transformers, Torch, OpenCV, Pillow, scikit-learn
- Reporting: FPDF2

## Project structure

```text
pro project/
  backend/
    app/
      api/
      core/
      schemas/
      services/
      utils/
    tests/
    requirements.txt
  frontend/
    app/
    components/
    lib/
    package.json
```

## Setup and run

### 1) Backend setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend docs: http://127.0.0.1:8000/docs

### 2) Frontend setup

In a new terminal:

```bash
cd frontend
copy .env.local.example .env.local
npm install
npm run dev
```

Frontend app: http://127.0.0.1:3000

## API endpoints

- `GET /health`
- `POST /api/v1/analyze/text`
- `POST /api/v1/analyze/url`
- `POST /api/v1/analyze/file`
- `POST /api/v1/analyze/multimodal`
- `POST /api/v1/report/pdf`

## Notes on model quality

Current implementation provides a practical working baseline with real URL content extraction, explainable multimodal signals, and a trainable text classifier.

## Current Reality and Next Steps

This project is functionally working end-to-end, but it is still an MVP baseline and not yet a production-grade ML system.

### What is currently implemented

- Text analysis uses a hybrid approach in which heuristic linguistic signals are combined with a lightweight TF-IDF + Logistic Regression classifier.
- URL analysis is real (content is fetched and parsed), then routed through the text analysis pipeline.
- Image and video analysis currently rely on explainable computer-vision heuristics (artifact, lighting, and temporal-consistency signals), not a deepfake neural model.
- Frontend, backend APIs, multimodal fusion, and PDF reporting are all operational.

### What still needs to be built (high priority)

- Replace the current text baseline with a trained transformer model (for example, DistilBERT/RoBERTa) using a real misinformation dataset.
- Add true deepfake model inference for image/video (for example, Xception/EfficientNet/ViT-based detector).
- Add model artifact loading/versioning so inference uses saved trained weights, not runtime toy training.
- Add benchmark/evaluation scripts with Accuracy, Precision, Recall, F1, ROC-AUC, and confusion matrix outputs.
- Add confidence calibration and decision-threshold management per model version.

### Suggested implementation order

1. Build offline training pipeline and save model artifacts.
2. Replace backend inference path to load trained artifacts.
3. Integrate deepfake vision model and frame-level video aggregation.
4. Add evaluation reports and reproducible validation splits.
5. Add production hardening (auth, rate limiting, monitoring, and async heavy-job handling).

To move toward production-grade accuracy:

- Replace baseline text/image/video detectors with fine-tuned transformer and deepfake-specific models.
- Add source credibility scoring and article extraction for URL analysis.
- Add benchmark dataset evaluation and confusion-matrix reporting.
- Add authentication, rate limiting, and model versioning.
