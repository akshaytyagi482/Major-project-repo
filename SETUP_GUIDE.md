# 🛠️ SENTINEL Setup & Deployment Guide

## 📋 Quick Reference

- **Frontend URL**: http://localhost:3000 (React/Next.js)
- **Backend URL**: http://localhost:5000 (Flask API)
- **Python Version**: 3.12+
- **Node Version**: 16+ (optional, if using Next.js)
- **RAM Required**: 8GB minimum
- **Storage**: ~5GB (for models cache)

---

## 💻 Windows Setup (Recommended)

### **Step 1: Install Python 3.12+**

Download from: https://www.python.org/downloads/

**Verify Installation:**
```bash
python --version
# Output: Python 3.12.10
```

### **Step 2: Clone & Navigate**

```bash
cd "c:\Users\aksha\OneDrive\Desktop\pro project\Major-project-repo"
```

### **Step 3: Create Virtual Environment**

```bash
# Create venv
python -m venv venv

# Activate venv
venv\Scripts\activate

# Verify activation (should see "venv" in terminal)
```

### **Step 4: Install Dependencies**

```bash
cd SENTINEL_multimodal_detector
pip install -r requirements.txt

# Verify key packages
python -c "import torch; import transformers; print('✓ All packages OK')"
```

### **Step 5: Download Pre-trained Models**

```bash
# Models auto-download on first run, but you can pre-download:
python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

print('Downloading BART...')
AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')

print('Downloading MiniLM...')
SentenceTransformer('all-MiniLM-L6-v2')

print('✓ Models downloaded!')
"
```

### **Step 6: Run Application**

```bash
# Option A: Direct (simplest)
python app.py

# Option B: With CUDA disabled (recommended)
$env:CUDA_VISIBLE_DEVICES=""; python app.py

# Option C: Background (for development)
python app.py > app.log 2>&1 &
```

**Server should output:**
```
🛡  Multimodal Fake News & Deepfake Detector — Starting...
   NLP Module:    DistilBERT + Transformers (Real AI)
   CV Module:     ResNet-50 + PyTorch (Real AI)
   ...
   Running on http://127.0.0.1:5000
```

### **Step 7: Verify Installation**

```bash
# In another terminal window
curl http://localhost:5000/api/health

# Should return:
# {"status": "online", "modules": ["NLP", "CV", "Fusion", "TTS"]}
```

---

## 🐧 Mac/Linux Setup

### **Prerequisites**

```bash
# Install Homebrew (Mac) or apt (Linux)
# Mac:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.12

# Linux (Ubuntu/Debian):
sudo apt-get update
sudo apt-get install python3.12 python3.12-venv python3-pip
```

### **Setup Steps**

```bash
# Navigate to project
cd ~/Desktop/pro\ project/Major-project-repo/SENTINEL_multimodal_detector

# Create virtual environment
python3.12 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download models
python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
SentenceTransformer('all-MiniLM-L6-v2')
print('✓ Models ready!')
"

# Run
export CUDA_VISIBLE_DEVICES=""
python app.py
```

---

## 🎨 Frontend Setup (Optional - Next.js)

If you want to run the Next.js frontend separately:

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev

# Open http://localhost:3000
```

**Build for production:**
```bash
npm run build
npm start
```

---

## 🔧 Troubleshooting Setup Issues

### **Issue 1: ModuleNotFoundError**

```bash
# Verify virtual environment is activated
# You should see "(venv)" in terminal

# If not, activate:
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Reinstall packages
pip install --upgrade -r requirements.txt
```

### **Issue 2: "No module named 'transformers'"**

```bash
pip install transformers==5.7.0 torch==2.10.0

# Verify
python -c "import transformers; print(transformers.__version__)"
```

### **Issue 3: Out of Memory**

```bash
# Windows - Increase virtual memory
# Settings → System → Advanced → Performance → Virtual Memory

# Or use CPU-only inference
$env:CUDA_VISIBLE_DEVICES=""
python app.py
```

### **Issue 4: Models not downloading**

```bash
# Manual download
mkdir ~/.cache/huggingface/hub
python -c "
import os
os.environ['HF_HOME'] = './.cache/huggingface'
from transformers import AutoModelForSequenceClassification
AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
"
```

### **Issue 5: Port 5000 already in use**

```bash
# Find process using port 5000
# Windows:
netstat -ano | findstr :5000

# Kill process (replace PID with actual number)
taskkill /PID <PID> /F

# Or use different port:
python app.py --port 8000
```

---

## 📊 System Requirements Checklist

| Component | Requirement | Check |
|-----------|------------|-------|
| **Python** | 3.12+ | `python --version` |
| **pip** | Latest | `pip --version` |
| **RAM** | 8GB minimum | Task Manager |
| **GPU** | Optional (NVIDIA) | `nvidia-smi` |
| **Disk** | 5GB free | Check C: drive |
| **Internet** | For model downloads | Ping huggingface.co |

---

## 🚀 Deployment (Production)

### **Using Gunicorn (Linux/Mac)**

```bash
# Install Gunicorn
pip install gunicorn

# Run with 4 workers
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### **Using Waitress (Windows)**

```bash
# Install Waitress
pip install waitress

# Create run_production.py
from waitress import serve
from app import app
serve(app, host='0.0.0.0', port=5000)

# Run
python run_production.py
```

### **Docker (Optional)**

```dockerfile
# Dockerfile
FROM python:3.12

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

**Build and run:**
```bash
docker build -t sentinel .
docker run -p 5000:5000 sentinel
```

---

## 🔐 Environment Variables

Create `.env` file in project root:

```bash
# Model cache location
HF_HOME=/path/to/models
TRANSFORMERS_CACHE=/path/to/cache

# API settings
FLASK_ENV=production
FLASK_DEBUG=False
MAX_FILE_MB=50

# CUDA settings
CUDA_VISIBLE_DEVICES=""  # Use CPU only
TF_CPP_LOGGING_LEVEL=3   # Suppress TensorFlow logs

# Security
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=http://localhost:3000
```

Load in Flask:

```python
from dotenv import load_dotenv
import os

load_dotenv()
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key')
```

---

## 📈 Performance Optimization

### **1. Model Quantization**

```python
# Use smaller models instead of large ones
# Instead of: facebook/bart-large-mnli (1.6GB)
# Use: facebook/bart-base-mnli (500MB)

classifier = pipeline("zero-shot-classification",
                     model="facebook/bart-base-mnli",
                     device=0)  # Use GPU
```

### **2. Batch Processing**

```python
# Process multiple requests at once
def analyze_batch(texts: List[str]):
    results = []
    for text in texts:
        results.append(nlp_detector.analyze(text))
    return results
```

### **3. Caching**

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def analyze_cached(text):
    return nlp_detector.analyze(text)
```

### **4. GPU Acceleration**

```bash
# Install GPU-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📱 API Testing

### **Using cURL**

```bash
# Test health endpoint
curl http://localhost:5000/api/health

# Analyze text
curl -X POST http://localhost:5000/api/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking news about X"}'

# Upload image
curl -X POST http://localhost:5000/api/analyze/image \
  -F "file=@image.jpg"
```

### **Using Python**

```python
import requests

# Test health
response = requests.get('http://localhost:5000/api/health')
print(response.json())

# Analyze text
data = {"text": "Fake news example"}
response = requests.post('http://localhost:5000/api/analyze/text', json=data)
print(response.json())

# Upload image
with open('image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/analyze/image', files=files)
print(response.json())
```

### **Using Postman**

1. Open Postman
2. Create new request
3. Set method to POST
4. Enter URL: `http://localhost:5000/api/analyze/text`
5. Headers: `Content-Type: application/json`
6. Body (raw JSON):
   ```json
   {
     "text": "Breaking: Scientists announce shocking discovery!"
   }
   ```
7. Click Send

---

## 📊 Monitoring & Logging

### **Enable Detailed Logging**

```python
import logging

# In app.py
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Application started")
```

### **Monitor Performance**

```bash
# Linux/Mac
top -p $(pgrep -f "python app.py")

# Windows
# Use Task Manager → Performance tab
```

### **View Logs**

```bash
# Real-time logs
tail -f app.log

# Search logs
grep "error" app.log
```

---

## 🔄 Continuous Deployment

### **GitHub Actions (CI/CD)**

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy SENTINEL

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - run: |
          pip install -r requirements.txt
          python -m pytest tests/
      - run: |
          # Deploy to production
          # e.g., push to Heroku, AWS, etc.
```

---

## 🆘 Getting Help

### **Check Logs**
```bash
tail -f app.log
```

### **Debug Mode**
```python
# In app.py
app.run(debug=True)  # Auto-reload, better error messages
```

### **Test Endpoints**
```bash
python test_api.py
```

### **Common Issues**

| Error | Cause | Fix |
|-------|-------|-----|
| 404 Not Found | Wrong endpoint | Check `/api/` prefix |
| 400 Bad Request | Missing field | Check JSON format |
| 422 Unprocessable | Wrong file type | Use .jpg, .txt, .mp4 |
| 500 Internal Error | Model error | Check logs, restart server |
| Timeout | Slow inference | Use GPU or smaller model |

---

## ✅ Post-Installation Checklist

- [ ] Python 3.12+ installed
- [ ] Virtual environment created & activated
- [ ] requirements.txt installed
- [ ] Models downloaded successfully
- [ ] `python app.py` starts without errors
- [ ] Health endpoint returns 200 OK
- [ ] Can upload files without errors
- [ ] Frontend loads at http://localhost:5000
- [ ] Can analyze sample text/image
- [ ] Results appear correctly

---

## 📚 Additional Resources

- **Flask Documentation**: https://flask.palletsprojects.com/
- **Transformers Tutorial**: https://huggingface.co/docs/transformers/
- **PyTorch Setup**: https://pytorch.org/get-started/locally/
- **Deployment Guides**: https://flask.palletsprojects.com/en/2.3.x/deploying/

---

**Last Updated:** May 6, 2026  
**Version:** 2.0  
**Status:** Production Ready ✅
