"""
Video Annotation Module for Poultry CCTV Analysis

Provides utilities for drawing bounding boxes, track IDs, confidence
scores, and count overlays on video frames.
"""

from typing import Optional, Tuple
import numpy as np
import cv2
import supervision as sv

from config import ANNOTATION_CONFIG


class PoultryAnnotator:
    """
    Video annotator for poultry detection visualization.
    
    Draws bounding boxes, labels, and count overlays using
    Supervision annotators with custom styling.
    """
    
    def __init__(
        self,
        box_color: Tuple[int, int, int] = ANNOTATION_CONFIG["box_color"],
        text_color: Tuple[int, int, int] = ANNOTATION_CONFIG["text_color"],
        box_thickness: int = ANNOTATION_CONFIG["box_thickness"],
        font_scale: float = ANNOTATION_CONFIG["font_scale"],
    ):
        """
        Initialize the annotator.
        
        Args:
            box_color: BGR color for bounding boxes.
            text_color: BGR color for text labels.
            box_thickness: Line thickness for boxes.
            font_scale: Scale factor for text.
        """
        self.box_color = box_color
        self.text_color = text_color
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        
        # Create Supervision annotators
        self.box_annotator = sv.BoxAnnotator(
            color=sv.Color(*box_color[::-1]),  # BGR to RGB
            thickness=box_thickness,
        )
        
        self.label_annotator = sv.LabelAnnotator(
            color=sv.Color(*box_color[::-1]),
            text_color=sv.Color(*text_color[::-1]),
            text_scale=font_scale,
            text_thickness=1,
        )
        
    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        count: int,
        show_confidence: bool = True,
        show_ids: bool = True,
        weight_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Annotate a frame with detections and count overlay.
        
        Args:
            frame: BGR image as numpy array.
            detections: Supervision Detections object.
            count: Current bird count to display.
            show_confidence: Whether to show confidence scores.
            show_ids: Whether to show tracker IDs.
            weight_indices: Optional array of weight indices per detection.
            
        Returns:
            Annotated frame.
        """
        annotated = frame.copy()
        
        # Draw bounding boxes
        annotated = self.box_annotator.annotate(annotated, detections)
        
        # Generate labels
        if len(detections) > 0:
            labels = self._generate_labels(
                detections,
                show_confidence=show_confidence,
                show_ids=show_ids,
                weight_indices=weight_indices,
            )
            annotated = self.label_annotator.annotate(annotated, detections, labels)
        
        # Draw count overlay
        annotated = self._draw_count_overlay(annotated, count)
        
        return annotated
    
    def _generate_labels(
        self,
        detections: sv.Detections,
        show_confidence: bool = True,
        show_ids: bool = True,
        weight_indices: Optional[np.ndarray] = None,
    ) -> list[str]:
        """Generate label strings for each detection."""
        labels = []
        
        for i in range(len(detections)):
            parts = []
            
            # Add tracker ID
            if show_ids and detections.tracker_id is not None:
                track_id = detections.tracker_id[i]
                if track_id is not None:
                    parts.append(f"ID:{int(track_id)}")
                    
            # Add confidence
            if show_confidence and detections.confidence is not None:
                conf = detections.confidence[i]
                parts.append(f"{conf:.2f}")
            
            # Weight removed from labels - will be shown in summary only
                
            labels.append(" ".join(parts) if parts else "")
            
        return labels
    
    def _draw_count_overlay(
        self,
        frame: np.ndarray,
        count: int,
        position: Tuple[int, int] = (10, 40),
    ) -> np.ndarray:
        """Draw count text overlay on frame."""
        text = f"Count: {count}"
        
        # Draw background rectangle for readability
        (text_w, text_h), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale * 1.5,
            2,
        )
        
        x, y = position
        cv2.rectangle(
            frame,
            (x - 5, y - text_h - 10),
            (x + text_w + 10, y + baseline + 5),
            (0, 0, 0),  # Black background
            -1,  # Filled
        )
        
        # Draw text
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale * 1.5,
            self.text_color,
            2,
            cv2.LINE_AA,
        )
        
        return frame
    
    def draw_statistics_overlay(
        self,
        frame: np.ndarray,
        stats: dict,
        position: Tuple[int, int] = (10, 80),
    ) -> np.ndarray:
        """
        Draw additional statistics overlay.
        
        Args:
            frame: BGR image.
            stats: Dictionary with statistics (e.g., weight, fps).
            position: Top-left position for overlay.
            
        Returns:
            Annotated frame.
        """
        y_offset = position[1]
        
        for key, value in stats.items():
            text = f"{key}: {value}"
            
            cv2.putText(
                frame,
                text,
                (position[0], y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.text_color,
                1,
                cv2.LINE_AA,
            )
            
            y_offset += int(25 * self.font_scale)
            
        return frame


class VideoWriter:
    """Helper class for writing annotated video output."""
    
    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: Tuple[int, int],
        codec: str = "mp4v",
    ):
        """
        Initialize video writer.
        
        Args:
            output_path: Path to output video file.
            fps: Output frame rate.
            frame_size: (width, height) of frames.
            codec: FourCC codec code.
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            frame_size,
        )
        
        self.frame_count = 0
        
    def write(self, frame: np.ndarray):
        """Write a frame to the video."""
        self.writer.write(frame)
        self.frame_count += 1
        
    def release(self):
        """Release the video writer."""
        self.writer.release()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
