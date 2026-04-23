from __future__ import annotations

import io
import os
import tempfile
from statistics import mean
from typing import List

import cv2
import numpy as np
from PIL import Image

from app.schemas.analysis import ModalityResult


def _jpeg_artifact_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray) / 255.0)
    high_freq = np.mean(np.abs(dct[gray.shape[0] // 4 :, gray.shape[1] // 4 :]))
    return float(min(high_freq * 1.8, 1.0))


def _lighting_inconsistency_score(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    std_v = float(np.std(v_channel) / 255.0)
    return min(std_v * 1.6, 1.0)


def analyze_image(image_bytes: bytes, filename: str = "uploaded_image") -> ModalityResult:
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    h, w = np_img.shape[:2]
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5)

    artifact = _jpeg_artifact_score(np_img)
    lighting = _lighting_inconsistency_score(np_img)
    face_count = len(faces)
    face_signal = 0.35 if face_count == 0 else min(face_count / 10.0, 0.2)

    score_fake = min(artifact * 0.5 + lighting * 0.3 + face_signal * 0.2, 1.0)
    score_real = 1.0 - score_fake

    if score_fake >= 0.7:
        verdict = "fake"
    elif score_fake >= 0.45:
        verdict = "suspicious"
    else:
        verdict = "real"

    evidence = [
        f"Compression artifact signal: {artifact:.2f}",
        f"Lighting inconsistency signal: {lighting:.2f}",
        f"Detected faces: {face_count}",
    ]

    confidence = 0.72 if face_count > 0 else 0.6

    return ModalityResult(
        modality="image",
        score_fake=round(score_fake, 4),
        score_real=round(score_real, 4),
        confidence=round(confidence, 4),
        verdict=verdict,
        evidence=evidence,
        metadata={"filename": filename, "width": w, "height": h, "face_count": face_count},
    )


def analyze_video(video_bytes: bytes, filename: str = "uploaded_video") -> ModalityResult:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        temp.write(video_bytes)
        temp_path = temp.name

    try:
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return ModalityResult(
                modality="video",
                score_fake=0.5,
                score_real=0.5,
                confidence=0.3,
                verdict="suspicious",
                evidence=["Unable to decode video stream."],
                metadata={"filename": filename},
            )

        frame_scores: List[float] = []
        frame_count = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            if frame_count % 8 != 0:
                continue
            artifact = _jpeg_artifact_score(frame)
            lighting = _lighting_inconsistency_score(frame)
            temporal_proxy = min(abs(artifact - lighting) * 1.4, 1.0)
            frame_scores.append(min(artifact * 0.5 + lighting * 0.25 + temporal_proxy * 0.25, 1.0))

        cap.release()

        if not frame_scores:
            return ModalityResult(
                modality="video",
                score_fake=0.5,
                score_real=0.5,
                confidence=0.35,
                verdict="suspicious",
                evidence=["Video did not contain enough readable frames."],
                metadata={"filename": filename, "frames_read": frame_count},
            )

        score_fake = float(mean(frame_scores))
        score_real = 1.0 - score_fake
        confidence = 0.65 if frame_count > 40 else 0.52

        if score_fake >= 0.7:
            verdict = "fake"
        elif score_fake >= 0.45:
            verdict = "suspicious"
        else:
            verdict = "real"

        return ModalityResult(
            modality="video",
            score_fake=round(score_fake, 4),
            score_real=round(score_real, 4),
            confidence=round(confidence, 4),
            verdict=verdict,
            evidence=[
                f"Sampled frames: {len(frame_scores)}",
                f"Total frames read: {frame_count}",
                f"Temporal consistency drift: {np.std(frame_scores):.3f}",
            ],
            metadata={"filename": filename, "frames_read": frame_count, "sampled_frames": len(frame_scores)},
        )
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
