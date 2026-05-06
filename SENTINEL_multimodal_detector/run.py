#!/usr/bin/env python3
"""
Wrapper script to run Flask app with CUDA disabled
This prevents torch from hanging during initialization
"""
import os
import sys

# Disable CUDA to prevent torch hanging issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_LOGGING_LEVEL'] = '3'  # Suppress TensorFlow logging

# Now import and run the Flask app
from app import app

if __name__ == "__main__":
    print("🛡  Multimodal Fake News & Deepfake Detector — Starting...")
    print("   NLP Module:    DistilBERT + Transformers (Real AI)")
    print("   CV Module:     ResNet-50 + PyTorch (Real AI)")
    print("   Fusion:        Neural multimodal fusion")
    print("   TTS:           gTTS audio synthesis")
    print("   Server:        http://localhost:5000")
    print("─" * 70)
    app.run(debug=True, host="0.0.0.0", port=5000)
