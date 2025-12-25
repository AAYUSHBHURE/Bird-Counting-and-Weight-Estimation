"""
Distant Footage Processor - Using Chicken Detection Model

Optimized for processing distant/overhead CCTV footage where birds appear small.
Uses pre-trained YOLOv11s chicken model with ByteTrack for persistent tracking.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
import json
from tqdm import tqdm
import argparse

from config import (
    MODEL_CONFIG,
    CHICKEN_DETECTION_CONFIG,
    DISTANT_TRACKING_CONFIG,
    OUTPUTS_DIR,
    WEIGHT_CONFIG,
)


def filter_red_objects(detections, frame, width, height):
    """Filter out red feeders/waterers from detections."""
    if len(detections) == 0:
        return detections
    
    keep_mask = []
    for box in detections.xyxy:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            avg_color = roi.mean(axis=(0, 1))
            b, g, r = avg_color
            
            # Filter red feeders: high red, low blue/green
            is_red = r > 120 and r > g + 40 and r > b + 40
            keep_mask.append(not is_red)
        else:
            keep_mask.append(True)
    
    return detections[np.array(keep_mask)]


def process_distant_video(
    video_path: str,
    output_dir: str = None,
    model_path: str = None,
    conf_threshold: float = None,
    fps_sample: int = 3,
):
    """
    Process distant footage video with chicken detection model.
    
    Args:
        video_path: Path to input video
        output_dir: Output directory (default: ./outputs)
        model_path: Path to chicken model (default: from config)
        conf_threshold: Detection confidence (default: 0.35)
        fps_sample: Process every Nth frame (default: 3)
    
    Returns:
        dict: Results including counts, weights, and output paths
    """
    # Setup
    output_dir = Path(output_dir) if output_dir else OUTPUTS_DIR
    output_dir.mkdir(exist_ok=True)
    
    model_path = model_path or MODEL_CONFIG["chicken_model"]
    conf_threshold = conf_threshold or CHICKEN_DETECTION_CONFIG["conf_thresh"]
    img_size = CHICKEN_DETECTION_CONFIG["imgsz"]
    
    print(f"📦 Loading chicken model from: {model_path}")
    model = YOLO(model_path)
    
    # Initialize tracker
    tracker = sv.ByteTrack(**DISTANT_TRACKING_CONFIG)
    
    # Annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.8, text_thickness=2)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"📹 Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    
    # Output video
    video_name = Path(video_path).stem
    output_video = output_dir / f"tracked_{video_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
    # Data collection
    counts_by_frame = {}
    weights_by_id = defaultdict(list)
    unique_ids = set()
    frame_idx = 0
    
    print(f"\n🚀 Processing with chicken model (conf={conf_threshold}, img_size={img_size})...\n")
    
    # Process video
    with tqdm(total=total_frames, desc="Processing") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % fps_sample == 0:
                # Detect chickens
                results = model(frame, conf=conf_threshold, imgsz=img_size, verbose=False)
                detections = sv.Detections.from_ultralytics(results[0])
                
                # Filter red feeders
                detections = filter_red_objects(detections, frame, width, height)
                
                # Track
                detections = tracker.update_with_detections(detections)
                
                count = len(detections)
                counts_by_frame[frame_idx] = count
                
                # Collect tracking data
                if detections.tracker_id is not None:
                    for box, tid in zip(detections.xyxy, detections.tracker_id):
                        unique_ids.add(int(tid))
                        area = (box[2] - box[0]) * (box[3] - box[1])
                        weights_by_id[int(tid)].append(float(area))
                
                # Annotate
                annotated = frame.copy()
                annotated = box_annotator.annotate(annotated, detections)
                
                # Labels: "Bird #ID"
                if detections.tracker_id is not None and len(detections) > 0:
                    labels = [f"Bird #{int(tid)}" for tid in detections.tracker_id]
                    annotated = label_annotator.annotate(annotated, detections, labels)
                
                # Overlay stats
                timestamp = frame_idx / fps
                cv2.putText(
                    annotated,
                    f"Count: {count} | Unique: {len(unique_ids)}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3,
                )
                cv2.putText(
                    annotated,
                    f"Time: {timestamp:.1f}s",
                    (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                
                out.write(annotated)
            else:
                out.write(frame)
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    out.release()
    
    # Calculate statistics
    count_values = list(counts_by_frame.values())
    max_count = max(count_values) if count_values else 0
    avg_count = np.mean(count_values) if count_values else 0
    
    # Weight estimates
    weight_estimates = []
    for tid, areas in weights_by_id.items():
        if len(areas) >= 5:  # Minimum 5 detections
            normalized = [(a / (height ** 2)) * WEIGHT_CONFIG["normalization_factor"] for a in areas]
            weight_estimates.append({
                "bird_id": tid,
                "weight_index": round(float(np.median(normalized)), 2),
                "uncertainty": round(float(np.std(normalized)) * WEIGHT_CONFIG["uncertainty_percent"], 2),
                "sample_count": len(areas),
            })
    
    # Sort by ID
    weight_estimates.sort(key=lambda x: x["bird_id"])
    
    # Counts over time
    counts_over_time = {}
    for frame, count in counts_by_frame.items():
        t = frame / fps
        ts = f"{int(t//60):02d}:{int(t%60):02d}.{int((t%1)*1000):03d}"
        counts_over_time[ts] = count
    
    # Compile results
    results = {
        "video_info": {
            "filename": Path(video_path).name,
            "duration_seconds": round(duration, 2),
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "resolution": f"{width}x{height}",
        },
        "count_summary": {
            "total_unique_birds": len(unique_ids),
            "max_simultaneous_count": max_count,
            "average_count": round(avg_count, 2),
        },
        "counts_over_time": counts_over_time,
        "weight_estimates": weight_estimates,
        "weight_metadata": {
            "unit": "index",
            "note": "Area-based weight proxy. Requires calibration for grams.",
            "calibration_needed": "Linear regression: weight_grams = slope × index + intercept",
        },
        "tracking_metrics": {
            "total_tracks": len(unique_ids),
            "birds_with_weight_data": len(weight_estimates),
            "frames_processed": len(counts_by_frame),
        },
        "processing_config": {
            "model": "YOLOv11s Chicken Detection",
            "source": "Hugging Face - IceKhoffi/chicken-object-detection-yolov11s",
            "confidence_threshold": conf_threshold,
            "image_size": img_size,
            "fps_sample": fps_sample,
        },
        "artifacts": {
            "annotated_video": str(output_video),
        },
    }
    
    # Save JSON
    results_json_path = output_dir / f"results_{video_name}.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    results["artifacts"]["results_json"] = str(results_json_path)
    
    print(f"\n✅ Processing complete!")
    print(f"   Total unique birds: {len(unique_ids)}")
    print(f"   Max count: {max_count}")
    print(f"   Average count: {avg_count:.1f}")
    print(f"   Birds with weight data: {len(weight_estimates)}")
    print(f"\n📁 Outputs:")
    print(f"   Video: {output_video}")
    print(f"   JSON: {results_json_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Process distant poultry CCTV footage")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--output-dir", help="Output directory", default=None)
    parser.add_argument("--model", help="Path to chicken model", default=None)
    parser.add_argument("--conf", type=float, help="Confidence threshold", default=None)
    parser.add_argument("--fps-sample", type=int, help="Process every Nth frame", default=3)
    
    args = parser.parse_args()
    
    process_distant_video(
        video_path=args.video,
        output_dir=args.output_dir,
        model_path=args.model,
        conf_threshold=args.conf,
        fps_sample=args.fps_sample,
    )


if __name__ == "__main__":
    main()
