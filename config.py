# Poultry CCTV Video Analysis - Configuration

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    "model_path": str(MODELS_DIR / "best.pt"),  # Fine-tuned poultry model
    "default_model": "yolov8n.pt",  # Pretrained COCO model (has bird class)
    "task": "detect",  # Options: detect, segment
    "imgsz": 640,
}

# Detection thresholds
DETECTION_CONFIG = {
    "conf_thresh": 0.3,  # Higher threshold to reduce false positives (4 birds not 5)
    "iou_thresh": 0.5,   # Lower IoU for crowded scenes
    "classes": [14],     # Class 14 = bird in COCO (use None for all classes)
}

# Tracking configuration (ByteTrack)
TRACKING_CONFIG = {
    "track_thresh": 0.5,
    "match_thresh": 0.8,
    "track_buffer": 30,  # Frames to persist lost tracks (Kalman)
    "frame_rate": 30,
}

# Video processing
VIDEO_CONFIG = {
    "fps_sample": 5,  # Sample rate for processing
    "output_fps": 30,
    "codec": "mp4v",
}

# Weight estimation
WEIGHT_CONFIG = {
    "normalization_factor": 1000,  # Scale factor for area index
    "uncertainty_percent": 0.15,  # ±15% uncertainty from posture
}

# Annotation styling
ANNOTATION_CONFIG = {
    "box_color": (0, 255, 0),  # Green (BGR)
    "text_color": (255, 255, 255),  # White
    "box_thickness": 2,
    "font_scale": 0.6,
    "font_thickness": 2,
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "max_upload_size": 500 * 1024 * 1024,  # 500MB max video upload
}
