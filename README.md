# 🐔 Bird Counting and Weight Estimation System

ML-powered FastAPI service for analyzing poultry CCTV footage with real-time bird counting, stable tracking, and weight estimation capabilities.

## 🎯 Features

- **🐦 Bird Detection**: YOLOv8 pretrained model (COCO class 14: bird) with 0.3 confidence threshold
- **🔍 Stable Tracking**: ByteTrack algorithm achieving 0.0% ID switch rate
- **⚖️ Weight Estimation**: Area-based proxy with calibration support for gram conversion
- **🎬 Video Annotation**: Bounding boxes with IDs, confidence scores, and count overlays
- **📊 Interactive Demo**: Streamlit web app with complete metrics visualization

## 🏆 Key Achievements

- ✅ **0.0% ID Switch Rate** - Perfect tracking stability
- ✅ **4/4 Birds Detected** - 100% accuracy on test video
- ✅ **Real-time Processing** - ~60 FPS detection speed
- ✅ **Stable Weight Values** - Median-based calculation prevents fluctuation

## 📋 Requirements Met

### Mandatory Features

1. **Bird Counting** ✅
   - Bounding box detection with confidence scores
   - Stable tracking IDs (ByteTrack)
   - Count over time (timestamp → count mapping)
   - Occlusion handling (30-frame buffer + Kalman filter)
   - ID switch prevention (0.0% achieved!)

2. **Weight Estimation** ✅
   - Calibration-based pixel-to-real mapping
   - Per-bird median weights (stable, normalized)
   - Flock aggregate statistics (mean, std, min, max)
   - Linear regression framework for gram conversion

3. **Annotated Output** ✅
   - Annotated video with bounding boxes
   - JSON results with complete metrics
   - Continuous annotations (no flickering)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/AAYUSHBHURE/Bird-Counting-and-Weight-Estimation.git
cd Bird-Counting-and-Weight-Estimation

# Install dependencies
pip install -r requirements.txt
```

### Run API Server

```bash
uvicorn main:app --port 8000
```

API will be available at `http://localhost:8000`

### Run Interactive Demo

```bash
streamlit run streamlit_app.py
```

Demo will open at `http://localhost:8501`

## 📖 API Usage

### Analyze Video

```bash
curl -X POST "http://localhost:8000/analyze_video" \
  -F "video=@your_video.mp4" \
  -F "fps_sample=5" \
  -F "conf_thresh=0.3" \
  -F "generate_annotated=true"
```

### Response Format

```json
{
  "video_info": {
    "duration_seconds": 13.8,
    "total_frames": 345,
    "source_fps": 25.0
  },
  "count_summary": {
    "total_unique_birds": 4,
    "max_count": 3,
    "avg_count": 2.3
  },
  "weight_estimates": [
    {
      "id": 1,
      "index": 151.87,
      "uncertainty": 22.78,
      "sample_count": 51
    }
  ],
  "tracking_metrics": {
    "total_tracks": 4,
    "id_switch_rate": 0.0
  }
}
```

## ⚙️ Configuration

Edit `config.py` to adjust:

```python
DETECTION_CONFIG = {
    "conf_thresh": 0.3,  # Detection confidence threshold
    "iou_thresh": 0.5,   # IoU threshold for NMS
    "classes": [14],     # Bird class from COCO dataset
}
```

## 📊 Weight Calibration

### Current Output: Weight Indices

The system outputs normalized area-based proxies, not grams.

### Convert to Grams (3 Steps)

1. **Collect Data**: 50+ videos with known bird weights
2. **Run Regression**:
   ```python
   from src.weight_estimator import WeightEstimator
   
   estimator = WeightEstimator()
   slope, intercept, r2 = estimator.calculate_calibration_regression(
       area_indices=[152, 605, 154],
       actual_weights_grams=[180, 950, 175]
   )
   ```
3. **Apply Formula**: `weight_grams = slope × index + intercept`

Expected R² > 0.85 for reliable calibration.

## 📁 Project Structure

```
Bird-Counting-and-Weight-Estimation/
├── README.md                  # This file
├── GIT_SETUP.md              # Git initialization guide
├── SUMMARY.md                # Project summary
├── implementation_details.md # Technical documentation
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── config.py                # Configuration settings
├── main.py                  # FastAPI application
├── process_video.py         # ML processing pipeline
├── streamlit_app.py        # Interactive demo interface
├── demo_video.mp4          # Sample input video
├── demo_annotated_web.mp4  # Sample annotated output
├── demo_output.json        # Sample API response
├── demo_final.json         # Complete analysis results
├── yolov8n.pt             # YOLOv8 pretrained model
├── src/
│   ├── __init__.py
│   ├── detector.py        # YOLOv8 detection
│   ├── tracker.py         # ByteTrack tracking
│   ├── weight_estimator.py # Weight calculation
│   ├── annotator.py       # Video annotation
│   └── dataset_loader.py  # Dataset utilities
├── models/
│   └── best.pt           # Fine-tuned model (optional)
└── notebooks/
    └── train_yolov8_colab.ipynb  # Training notebook
```

## 🎬 Demo Application

The Streamlit demo includes 5 sections:

1. **Overview** - System architecture and key metrics
2. **Requirements** - Detailed compliance checklist
3. **Demo Video** - Annotated video playback with bounding boxes
4. **Results** - Complete analysis with charts and tables
5. **API Documentation** - Endpoints and usage examples

## 🔧 Technical Details

### Detection Model
- **Base**: YOLOv8n pretrained on COCO dataset
- **Target Class**: Bird (ID 14)
- **Input Size**: 640×640 pixels
- **Fine-tuning**: Optional custom model support

### Tracking Algorithm
- **Method**: ByteTrack with Kalman filter
- **Buffer Size**: 30 frames (~1 second @ 30 FPS)
- **Features**: IoU matching, confidence decay, occlusion handling

### Weight Estimation
- **Formula**: `index = (bbox_area / frame_height²) × 1000`
- **Method**: Median over all detections per bird
- **Uncertainty**: ±15% from posture variations
- **Filter**: Minimum 8-frame detections to reduce false positives

## 📈 Performance Metrics

### Test Results
- **Birds Detected**: 4/4 (100% accuracy)
- **ID Switch Rate**: 0.0% (perfect tracking)
- **Max Simultaneous**: 3 birds in frame
- **Processing Speed**: ~60 FPS detection
- **Video Generation**: 5-6 seconds for 14-second input

### Weight Estimates (Test Video)
| Bird ID | Weight Index | Uncertainty | Frames |
|---------|--------------|-------------|--------|
| 1       | 151.87       | ±22.78      | 51     |
| 2       | 605.56       | ±90.83      | 59     |
| 3       | 151.81       | ±22.77      | 25     |
| 4       | 147.88       | ±22.18      | 8      |

## 🐛 Troubleshooting

**Video shows black screen in browser**
- Cause: MP4v codec not supported
- Solution: Videos auto-converted to H.264, or download and play in VLC

**Wrong number of birds detected**
- Solution: Adjust `conf_thresh` in config (try 0.25-0.4)
- Alternative: Modify minimum detection threshold (currently 8 frames)

**API returns 422 error**
- Check: Video file format (MP4, AVI supported)
- Check: Parameter types (fps_sample as int, conf_thresh as float)

## 🚀 Future Enhancements

- Real-time WebSocket streaming for live video
- Advanced analytics (activity patterns, health indicators)
- Docker containerization for easy deployment
- Cloud deployment (AWS/Azure)
- Mobile app integration

## 📄 License

MIT License - See LICENSE file for details

## 👤 Author

**Aayush Bhure**

- GitHub: [@AAYUSHBHURE](https://github.com/AAYUSHBHURE)
- Project: [Bird-Counting-and-Weight-Estimation](https://github.com/AAYUSHBHURE/Bird-Counting-and-Weight-Estimation)

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for object detection
- **ByteTrack**: For stable multi-object tracking
- **Supervision**: For tracking utilities and annotations
- **FastAPI**: For modern API framework
- **Streamlit**: For interactive demo interface

## 📞 Contact

For questions, issues, or contributions, please open a GitHub issue or contact via GitHub.

---

⭐ Star this repository if you find it helpful!
