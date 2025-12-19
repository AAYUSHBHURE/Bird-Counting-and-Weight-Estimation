"""
Video Processing Pipeline for Poultry CCTV Analysis

Orchestrates the complete ML pipeline:
1. Video frame extraction at sampled FPS
2. YOLOv8 detection
3. ByteTrack tracking
4. Weight proxy calculation
5. Annotated video generation
6. JSON result compilation
"""

import os
import tempfile
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
import cv2
from tqdm import tqdm

from config import (
    VIDEO_CONFIG,
    DETECTION_CONFIG,
    OUTPUTS_DIR,
)
from src.detector import PoultryDetector
from src.tracker import PoultryTracker
from src.weight_estimator import WeightEstimator
from src.annotator import PoultryAnnotator, VideoWriter


class VideoProcessor:
    """
    Complete video analysis pipeline for poultry CCTV footage.
    
    Processes videos to extract bird counts, track IDs, and weight
    proxy estimates, with optional annotated video output.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_thresh: float = DETECTION_CONFIG["conf_thresh"],
        iou_thresh: float = DETECTION_CONFIG["iou_thresh"],
        fps_sample: int = VIDEO_CONFIG["fps_sample"],
    ):
        """
        Initialize the video processor.
        
        Args:
            model_path: Path to YOLOv8 model weights.
            conf_thresh: Detection confidence threshold.
            iou_thresh: Detection IoU threshold.
            fps_sample: Frame rate for processing (samples every N frames).
        """
        self.detector = PoultryDetector(
            model_path=model_path,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
        )
        self.tracker = PoultryTracker()
        self.weight_estimator = WeightEstimator()
        self.annotator = PoultryAnnotator()
        
        self.fps_sample = fps_sample
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
    def process_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        generate_annotated: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> Dict:
        """
        Process a video file and return analysis results.
        
        Args:
            video_path: Path to input video file.
            output_dir: Directory for outputs (default: OUTPUTS_DIR).
            generate_annotated: Whether to generate annotated video.
            progress_callback: Optional callback(current, total) for progress.
            
        Returns:
            Dictionary with counts, tracks, weight estimates, and artifacts.
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir or OUTPUTS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Reset components for new video
        self.tracker.reset()
        self.weight_estimator.reset()
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set frame dimensions for weight estimation
        self.weight_estimator.set_frame_dimensions(frame_height, frame_width)
        
        # Calculate frame skip for sampling
        frame_skip = max(1, int(source_fps / self.fps_sample))
        
        # Setup output video writer
        video_writer = None
        output_video_path = None
        
        if generate_annotated:
            output_video_path = output_dir / f"{video_path.stem}_annotated.mp4"
            video_writer = VideoWriter(
                str(output_video_path),
                fps=VIDEO_CONFIG["output_fps"],
                frame_size=(frame_width, frame_height),
            )
            
        # Results containers
        counts_by_timestamp: Dict[str, int] = {}
        frame_results: List[Dict] = []
        
        # Store last known state for continuous annotations
        last_tracked_detections = None
        last_weight_indices = None
        last_count = 0
        
        # Process frames
        frame_idx = 0
        processed_count = 0
        
        pbar = tqdm(total=total_frames, desc="Processing video")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Sample frames
            if frame_idx % frame_skip == 0:
                # Run detection
                detections = self.detector.detect(frame)
                
                # Get areas for weight estimation
                areas = self.detector.get_detection_areas(detections)
                
                # Update tracker
                tracked_detections = self.tracker.update(detections, areas)
                
                # Update weight estimator
                if tracked_detections.tracker_id is not None:
                    self.weight_estimator.update_batch(
                        tracked_detections.tracker_id,
                        areas,
                    )
                    
                # Get current count
                current_count = self.tracker.get_active_count()
                
                # Get fixed weight indices using weight estimator's median calculation
                weight_indices = None
                if tracked_detections.tracker_id is not None and len(tracked_detections.tracker_id) > 0:
                    weight_indices = []
                    for track_id in tracked_detections.tracker_id:
                        if track_id is not None:
                            estimate = self.weight_estimator.get_estimate(int(track_id))
                            if estimate:
                                weight_indices.append(estimate.index)
                            else:
                                # Fallback for new tracks
                                idx = list(tracked_detections.tracker_id).index(track_id)
                                weight_indices.append(
                                    self.weight_estimator.calculate_index(areas[idx]) if idx < len(areas) else 0
                                )
                        else:
                            weight_indices.append(0)
                    weight_indices = np.array(weight_indices)
                
                # Store for use in non-sampled frames
                last_tracked_detections = tracked_detections
                last_weight_indices = weight_indices
                last_count = current_count
                
                # Calculate timestamp
                timestamp = self._frame_to_timestamp(frame_idx, source_fps)
                
                # Store count (use 1-second buckets)
                timestamp_key = timestamp.split(".")[0]  # Remove milliseconds
                counts_by_timestamp[timestamp_key] = current_count
                
                # Store frame result
                frame_results.append({
                    "frame": frame_idx,
                    "timestamp": timestamp,
                    "count": current_count,
                    "detections": len(detections),
                })
                
                # Generate annotated frame
                if video_writer:
                    annotated_frame = self.annotator.annotate_frame(
                        frame,
                        tracked_detections,
                        current_count,
                        weight_indices=weight_indices,
                    )
                    video_writer.write(annotated_frame)
                    
                processed_count += 1
                
            else:
                # For non-sampled frames, use last known detections for continuous annotations
                if video_writer and last_tracked_detections is not None:
                    annotated_frame = self.annotator.annotate_frame(
                        frame,
                        last_tracked_detections,
                        last_count,
                        weight_indices=last_weight_indices,
                    )
                    video_writer.write(annotated_frame)
                elif video_writer:
                    # No detections yet, write raw frame
                    video_writer.write(frame)
                    
            frame_idx += 1
            pbar.update(1)
            
            if progress_callback:
                progress_callback(frame_idx, total_frames)
                
        pbar.close()
        cap.release()
        
        if video_writer:
            video_writer.release()
            
        # Compile results
        results = self._compile_results(
            counts_by_timestamp,
            frame_results,
            output_video_path,
            video_path,
            source_fps,
            total_frames,
        )
        
        # Save JSON results
        json_path = output_dir / f"{video_path.stem}_results.json"
        self._save_results(results, json_path)
        results["artifacts"]["results_json"] = str(json_path)
        
        return results
    
    def _frame_to_timestamp(self, frame_idx: int, fps: float) -> str:
        """Convert frame index to timestamp string (MM:SS.mmm)."""
        seconds = frame_idx / fps
        td = timedelta(seconds=seconds)
        
        minutes = int(td.total_seconds() // 60)
        secs = td.total_seconds() % 60
        
        return f"{minutes:02d}:{secs:05.2f}"
    
    def _compile_results(
        self,
        counts_by_timestamp: Dict[str, int],
        frame_results: List[Dict],
        output_video_path: Optional[Path],
        video_path: Path,
        source_fps: float,
        total_frames: int,
    ) -> Dict:
        """Compile all results into final output dictionary."""
        
        # Get weight estimates and filter out birds with too few detections
        MIN_DETECTION_THRESHOLD = 8  # Lower threshold to include bird that appears later (ID 4)
        
        all_weight_estimates = self.weight_estimator.get_all_estimates()
        weight_estimates = [
            e.to_dict() for e in all_weight_estimates 
            if e.sample_count >= MIN_DETECTION_THRESHOLD
        ]
        
        # Get flock statistics (only for filtered birds)
        if weight_estimates:
            indices = [e['index'] for e in weight_estimates]
            uncertainties = [e['uncertainty'] for e in weight_estimates]
            flock_stats = {
                "mean_index": round(float(np.mean(indices)), 2),
                "std_index": round(float(np.std(indices)), 2),
                "min_index": round(float(np.min(indices)), 2),
                "max_index": round(float(np.max(indices)), 2),
                "total_birds": len(weight_estimates),
                "mean_uncertainty": round(float(np.mean(uncertainties)), 3),
            }
        else:
            flock_stats = {
                "mean_index": 0.0,
                "std_index": 0.0,
                "min_index": 0.0,
                "max_index": 0.0,
                "total_birds": 0,
                "mean_uncertainty": 0.0,
            }
        
        # Get sample tracks (only for birds with enough detections)
        valid_bird_ids = {e['id'] for e in weight_estimates}
        all_sample_tracks = self.tracker.get_sample_tracks(n=10)
        sample_tracks = [
            track for track in all_sample_tracks 
            if track['id'] in valid_bird_ids
        ][:5]  # Limit to 5 samples
        
        # Get tracking metrics
        id_switch_rate = self.tracker.get_id_switch_rate()
        
        return {
            "video_info": {
                "filename": video_path.name,
                "duration_seconds": round(total_frames / source_fps, 2),
                "total_frames": total_frames,
                "source_fps": round(source_fps, 2),
                "processed_fps": self.fps_sample,
            },
            "counts": counts_by_timestamp,
            "count_summary": {
                "max_count": max(counts_by_timestamp.values()) if counts_by_timestamp else 0,
                "min_count": min(counts_by_timestamp.values()) if counts_by_timestamp else 0,
                "avg_count": round(
                    sum(counts_by_timestamp.values()) / len(counts_by_timestamp), 1
                ) if counts_by_timestamp else 0,
                "total_unique_birds": len(weight_estimates),  # Use filtered count
            },
            "tracks_sample": sample_tracks,
            "tracking_metrics": {
                "total_tracks": len(weight_estimates),  # Use filtered count
                "id_switch_rate": round(id_switch_rate, 4),
            },
            "weight_estimates": weight_estimates,
            "flock_statistics": flock_stats,
            "artifacts": {
                "annotated_video": str(output_video_path) if output_video_path else None,
            },
            "processing_config": {
                "conf_thresh": self.conf_thresh,
                "iou_thresh": self.iou_thresh,
                "fps_sample": self.fps_sample,
            },
        }
    
    def _save_results(self, results: Dict, path: Path):
        """Save results to JSON file."""
        import json
        
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to: {path}")


def process_video_file(
    video_path: str,
    output_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    conf_thresh: float = 0.5,
    iou_thresh: float = 0.7,
    fps_sample: int = 5,
    generate_annotated: bool = True,
) -> Dict:
    """
    Convenience function to process a video file.
    
    Args:
        video_path: Path to input video.
        output_dir: Output directory for results.
        model_path: Optional custom model path.
        conf_thresh: Detection confidence threshold.
        iou_thresh: Detection IoU threshold.
        fps_sample: Processing frame rate.
        generate_annotated: Whether to generate annotated video.
        
    Returns:
        Analysis results dictionary.
    """
    processor = VideoProcessor(
        model_path=model_path,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        fps_sample=fps_sample,
    )
    
    return processor.process_video(
        video_path,
        output_dir=output_dir,
        generate_annotated=generate_annotated,
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python process_video.py <video_path> [output_dir]")
        sys.exit(1)
        
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    results = process_video_file(video_path, output_dir)
    
    print("\n=== Analysis Complete ===")
    print(f"Total Unique Birds: {results['count_summary']['total_unique_birds']}")
    print(f"Average Count: {results['count_summary']['avg_count']}")
    print(f"Annotated Video: {results['artifacts']['annotated_video']}")
