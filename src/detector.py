"""
YOLOv8 Detection Module for Poultry CCTV Analysis

Provides bird detection using Ultralytics YOLOv8 with configurable
confidence and IoU thresholds.
"""

from typing import Optional, Tuple, List
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import supervision as sv

from config import MODEL_CONFIG, DETECTION_CONFIG


class PoultryDetector:
    """YOLOv8-based detector for poultry/birds in CCTV footage."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_thresh: float = DETECTION_CONFIG["conf_thresh"],
        iou_thresh: float = DETECTION_CONFIG["iou_thresh"],
        task: str = MODEL_CONFIG["task"],
    ):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to YOLOv8 model weights (.pt file).
                        Uses yolov8n.pt (pretrained) if not provided.
            conf_thresh: Confidence threshold for detections (0-1).
            iou_thresh: IoU threshold for NMS (0-1).
            task: Detection task - 'detect' or 'segment'.
        """
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.task = task
        
        # Load model - use provided path or default to pretrained yolov8n
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            # Use pretrained yolov8n from Ultralytics (80 COCO classes)
            default_model = "yolov8n.pt"
            if task == "segment":
                default_model = "yolov8n-seg.pt"
            self.model = YOLO(default_model)
            
        self.class_names = self.model.names
        
    def detect(
        self,
        frame: np.ndarray,
        classes: Optional[List[int]] = None,
    ) -> sv.Detections:
        """
        Run detection on a single frame.
        
        Args:
            frame: BGR image as numpy array.
            classes: List of class IDs to filter (None = all classes).
            
        Returns:
            sv.Detections object with boxes, confidence, and class IDs.
        """
        # Run inference
        results = self.model(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=classes,
            verbose=False,
        )[0]
        
        # Convert to Supervision Detections
        detections = sv.Detections.from_ultralytics(results)
        
        return detections
    
    def detect_with_masks(
        self,
        frame: np.ndarray,
        classes: Optional[List[int]] = None,
    ) -> Tuple[sv.Detections, Optional[np.ndarray]]:
        """
        Run segmentation detection to get masks for weight estimation.
        
        Args:
            frame: BGR image as numpy array.
            classes: List of class IDs to filter.
            
        Returns:
            Tuple of (Detections, masks array or None).
        """
        results = self.model(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=classes,
            verbose=False,
        )[0]
        
        detections = sv.Detections.from_ultralytics(results)
        
        # Extract masks if available (segmentation model)
        masks = None
        if hasattr(results, 'masks') and results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            
        return detections, masks
    
    def get_detection_areas(self, detections: sv.Detections) -> np.ndarray:
        """
        Calculate bounding box areas for each detection.
        
        Args:
            detections: Supervision Detections object.
            
        Returns:
            Array of areas for each detection.
        """
        if len(detections) == 0:
            return np.array([])
            
        boxes = detections.xyxy  # [x1, y1, x2, y2]
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        
        return widths * heights
    
    @staticmethod
    def fine_tune(
        data_yaml: str,
        epochs: int = 10,
        imgsz: int = 640,
        model_name: str = "yolov8n.pt",
        project: str = "runs/train",
        name: str = "poultry_finetune",
    ) -> str:
        """
        Fine-tune YOLOv8 on custom poultry dataset.
        
        Args:
            data_yaml: Path to dataset YAML configuration.
            epochs: Number of training epochs.
            imgsz: Input image size.
            model_name: Base model to fine-tune.
            project: Output project directory.
            name: Experiment name.
            
        Returns:
            Path to best weights.
        """
        model = YOLO(model_name)
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            project=project,
            name=name,
            patience=5,
            save=True,
            plots=True,
        )
        
        return str(Path(project) / name / "weights" / "best.pt")
