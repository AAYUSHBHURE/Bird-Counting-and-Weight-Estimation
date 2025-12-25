# 🐔 Poultry CCTV Analysis System

Advanced ML-powered system for bird counting, tracking, and weight estimation in poultry farms using computer vision.

## 🎬 Demo Files (via Git LFS)

- **📹 Annotated Video**: [`outputs/tracked_2025_12_15_15_24_16_4_dQCiGf.mp4`](outputs/tracked_2025_12_15_15_24_16_4_dQCiGf.mp4) (1.9 GB)
  - Shows Bird #IDs, bounding boxes, count overlays
- **🤖 Chicken Detection Model**: [`models/chicken_yolov11s.pt`](models/chicken_yolov11s.pt) (19 MB)
  - YOLOv11s trained on chicken detection

> **Note**: Large files hosted via Git LFS. Clone with `git lfs pull` to download.

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [API Usage](#-api-usage)
- [Project Structure](#-project-structure)
- [Approach & Methodology](#-approach--methodology)
- [Results](#-results)
- [Submission Deliverables](#-submission-deliverables)

---

## ✨ Features

### Mandatory Requirements ✅

- **🐦 Bird Detection & Counting**
  - **YOLOv8n** for close-up footage (COCO bird class)
  - **YOLOv11s** for distant footage (chicken-specific model)
  - Real-time bounding box visualization
  - Count over time tracking
  - Support for both close-up and distant footage

- **🎯 Stable Tracking**
  - ByteTrack algorithm for persistent IDs
  - Occlusion handling (60-frame buffer)
  - Minimal ID switching
  - Individual bird identification

- **⚖️ Weight Estimation**
  - Area-based proxy calculation
  - Per-bird weight indices
  - Calibration framework ready
  - Uncertainty quantification (±15%)

- **🎬 Annotated Output**
  - Bounding boxes with bird IDs
  - Count overlays
  - JSON results export
  - MP4 video output

### Optional Enhancements ✅

- **📊 Interactive UI** - Streamlit web interface
- **🚀 FastAPI Service** - RESTful API with async processing
- **🎨 Professional Annotations** - Color-coded tracking
- **📈 Advanced Analytics** - Aggregate statistics

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip
```

### Installation

```bash
# 1. Clone repository
git clone https://github.com/AAYUSHBHURE/Bird-Counting-and-Weight-Estimation.git
cd Bird-Counting-and-Weight-Estimation

# 2. Install dependencies
pip install -r requirements.txt
```

### Run API Server

```bash
uvicorn main:app --port 8000
```

API available at: `http://localhost:8000`  
Docs at: `http://localhost:8000/docs`

### Run Demo App

```bash
streamlit run streamlit_app.py
```

Demo opens at: `http://localhost:8501`

### Process Video (Command Line)

```bash
# Close-up footage
python process_video.py your_video.mp4

# Distant footage (optimized)
python process_distant_footage.py your_video.mp4
```

---

## 📖 API Usage

### Analyze Video Endpoint

**POST** `/analyze_video`

```bash
curl -X POST "http://localhost:8000/analyze_video" \
  -F "video=@video.mp4" \
  -F "fps_sample=5" \
  -F "conf_thresh=0.3" \
  -F "generate_annotated=true"
```

### Response Format

```json
{
  "video_info": {
    "filename": "video.mp4",
    "duration_seconds": 13.8,
    "total_frames": 345,
    "source_fps": 25.0,
    "resolution": "1920x1080"
  },
  "count_summary": {
    "total_unique_birds": 324,
    "max_simultaneous_count": 187,
    "average_count": 156.3
  },
  "counts_over_time": {
    "00:00.000": 4,
    "00:00.200": 3
  },
  "weight_estimates": [
    {
      "bird_id": 1,
      "weight_index": 151.87,
      "uncertainty": 3.45,
      "sample_count": 42
    }
  ],
  "tracking_metrics": {
    "total_tracks": 324,
    "frames_processed": 69
  },
  "artifacts": {
    "annotated_video": "outputs/annotated_video.mp4",
    "results_json": "outputs/results.json"
  }
}
```

---

## 📁 Project Structure

```
poultry-cctv-analysis/
├── main.py                      # FastAPI server
├── streamlit_app.py             # Interactive demo
├── process_video.py             # CLI video processor
├── process_distant_footage.py   # Optimized for distant cameras
├── config.py                    # Configuration settings
├── requirements.txt             # Dependencies
│
├── src/
│   ├── detector.py              # YOLOv8 detection
│   ├── tracker.py               # ByteTrack tracking
│   ├── weight_estimator.py      # Weight calculation
│   ├── annotator.py             # Video annotation
│   └── video_processor.py       # Main pipeline
│
├── models/
│   ├── best.pt                  # Fine-tuned model
│   └── chicken_yolov11s.pt      # Chicken detection model
│
├── outputs/                     # Generated results
├── tests/                       # Unit tests
└── notebooks/                   # Training notebooks
```

---

## 🧠 Approach & Methodology

### 1. Detection Model

**Close-up Footage:**
- Model: YOLOv8n (COCO pretrained)
- Class: 14 (bird)
- Confidence: 0.3
- Resolution: 640x640

**Distant Footage:**
- Model: YOLOv11s (chicken-trained from Hugging Face)
- Source: `IceKhoffi/chicken-object-detection-yolov11s`
- Confidence: 0.35
- Resolution: 1280x1280 (higher for small objects)

### 2. Tracking Algorithm

**ByteTrack** with optimizations:
- Track activation: 0.30-0.50
- Lost track buffer: 30-60 frames
- Matching threshold: 0.70-0.80
- Kalman filter for motion prediction

### 3. Weight Estimation

**Area-based Proxy Method:**
```python
weight_index = (bbox_area / frame_height²) × 1000
```

- Uses median across detections (stable)
- Normalized by frame size
- Ready for calibration: `weight_grams = slope × index + intercept`

### 4. Optimizations for Distant Footage

**Challenges:**
- Small bird size (~20-40 pixels)
- High density & occlusions
- Background clutter (feeders, watchers)

**Solutions:**
- Higher resolution processing (1280px)
- Specialized chicken detection model
- Red object filtering (removes feeders)
- Longer track buffer (60 frames)
- Lower confidence threshold (0.35)

---

## 📊 Results

### Close-up Footage (Test Video)

| Metric | Value |
|--------|-------|
| Birds Detected | 4/4 (100%) |
| ID Switch Rate | 0.0% |
| Processing Speed | ~60 FPS |
| Weight Stability | <5% variance |

### Distant Footage (Assignment Video)

| Metric | Value |
|--------|-------|
| Max Simultaneous Count | 187 birds |
| Average Count | 156.3 birds |
| Total Unique Tracks | 324 |
| Model | YOLOv11s Chicken |

---

## 📦 Submission Deliverables

### 1. Complete Source Code ✅
- All Python files
- Configuration
- Requirements
- Documentation

### 2. README.md ✅
- Comprehensive setup guide
- API documentation
- Methodology explanation

### 3. Annotated Output Video ✅
- Bounding boxes with bird IDs
- Count overlays
- MP4 format
- Located in: `outputs/tracked_*.mp4`

### 4. Sample JSON Response ✅
- Complete analysis results
- All required fields
- Located in: `outputs/results_*.json`

---

## 🛠️ Configuration

Edit `config.py` to customize:

```python
# Detection settings
DETECTION_CONFIG = {
    "conf_thresh": 0.3,
    "iou_thresh": 0.5,
    "classes": [14],  # Bird class
}

# Tracking settings
TRACKING_CONFIG = {
    "track_buffer": 30,
    "match_thresh": 0.8,
}

# For distant footage
CHICKEN_DETECTION_CONFIG = {
    "conf_thresh": 0.35,
    "imgsz": 1280,
}
```

---

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Test API endpoints
pytest tests/test_api.py

# Test video processing
python process_video.py test_data/sample.mp4
```

---

## 📝 Notes

- **Model Selection**: Close-up uses COCO bird class, distant uses specialized chicken model
- **Weight Calibration**: Requires reference data (grams) for linear regression
- **Performance**: Distant footage processes at ~3-5 FPS due to higher resolution
- **Accuracy**: Best results with clear, well-lit footage

---

## 👨‍💻 Author

**Aayush Bhure**

GitHub: [AAYUSHBHURE/Bird-Counting-and-Weight-Estimation](https://github.com/AAYUSHBHURE/Bird-Counting-and-Weight-Estimation)

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- YOLOv8/YOLOv11 by Ultralytics
- ByteTrack by Zhang et al.
- Supervision library
- Hugging Face chicken detection model
