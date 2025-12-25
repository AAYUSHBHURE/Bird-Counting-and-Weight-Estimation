# Model Files

## 📦 Required Models

This project uses two detection models:

### 1. YOLOv8n (Close-up Footage)
- **File**: `best.pt` or `yolov8n.pt`
- **Purpose**: COCO pretrained model for bird detection (class 14)
- **Size**: ~6 MB
- **Auto-downloaded** by Ultralytics on first run

### 2. YOLOv11s Chicken Detection (Distant Footage)
- **File**: `chicken_yolov11s.pt`
- **Purpose**: Specialized chicken detection for overhead cameras
- **Size**: ~19 MB
- **Source**: Hugging Face - `IceKhoffi/chicken-object-detection-yolov11s`

## 📥 Download Instructions

### Option 1: Automatic Download (Recommended)

The `chicken_yolov11s.pt` model will be automatically downloaded when you run:
```python
python process_distant_footage.py your_video.mp4
```

### Option 2: Manual Download

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id='IceKhoffi/chicken-object-detection-yolov11s',
    filename='yolov11s.pt'
)
# Copy to models/chicken_yolov11s.pt
```

### Option 3: Direct Download

Download from: https://huggingface.co/IceKhoffi/chicken-object-detection-yolov11s

Save as: `models/chicken_yolov11s.pt`

## ⚠️ Important Notes

- Model files are **excluded from Git** due to size (`.gitignore`)
- Models are downloaded automatically on first use
- Internet connection required for first run
- Models are cached in `models/` directory

## 📂 Directory Structure

```
models/
├── best.pt                  # Fine-tuned model (if available)
├── chicken_yolov11s.pt     # Chicken detection model
└── MODELS_README.md        # This file
```

## 🔄 Model Updates

To update the chicken detection model:
1. Download new version from Hugging Face
2. Replace `models/chicken_yolov11s.pt`
3. No code changes needed

---

**Current Model Status:** ✅ Updated to latest chicken_yolov11s.pt
