"""
Chicks4FreeID Dataset Loader for Poultry CCTV Analysis

Provides utilities to load and prepare the Chicks4FreeID dataset
from Hugging Face for fine-tuning YOLOv8 on poultry detection.

Dataset: https://huggingface.co/datasets/dariakern/Chicks4FreeID
- 677 top-down coop images
- 1,270 bird instances (chickens, roosters, ducks)
- Re-ID labels for tracking evaluation
- Instance segmentation masks
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm

from config import BASE_DIR


class Chicks4FreeIDLoader:
    """
    Loader for Chicks4FreeID dataset from Hugging Face.
    
    Exports to YOLO format for fine-tuning detection/segmentation models.
    """
    
    DATASET_NAME = "dariakern/Chicks4FreeID"
    CLASS_NAMES = ["chicken"]  # Simplified to single class for detection
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the dataset loader.
        
        Args:
            output_dir: Directory to save exported dataset.
                       Defaults to BASE_DIR / "datasets" / "chicks4freeid"
        """
        self.output_dir = output_dir or (BASE_DIR / "datasets" / "chicks4freeid")
        self.output_dir = Path(self.output_dir)
        
        self._dataset = None
        
    def load_dataset(self):
        """
        Load dataset from Hugging Face.
        
        Returns:
            The loaded dataset.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Please install datasets: pip install datasets"
            )
            
        print(f"Loading {self.DATASET_NAME} from Hugging Face...")
        self._dataset = load_dataset(self.DATASET_NAME, split="train")
        print(f"Loaded {len(self._dataset)} images")
        
        return self._dataset
    
    def get_dataset_info(self) -> Dict:
        """Get basic information about the dataset."""
        if self._dataset is None:
            self.load_dataset()
            
        return {
            "name": self.DATASET_NAME,
            "num_images": len(self._dataset),
            "features": list(self._dataset.features.keys()),
            "description": "Top-down poultry coop images with re-ID annotations",
        }
    
    def export_to_yolo(
        self,
        train_split: float = 0.8,
        include_masks: bool = False,
    ) -> str:
        """
        Export dataset to YOLO format for training.
        
        Args:
            train_split: Fraction of data for training (rest for validation).
            include_masks: Whether to export segmentation masks (for YOLOv8-seg).
            
        Returns:
            Path to generated data.yaml file.
        """
        if self._dataset is None:
            self.load_dataset()
            
        # Create directory structure
        images_train = self.output_dir / "images" / "train"
        images_val = self.output_dir / "images" / "val"
        labels_train = self.output_dir / "labels" / "train"
        labels_val = self.output_dir / "labels" / "val"
        
        for d in [images_train, images_val, labels_train, labels_val]:
            d.mkdir(parents=True, exist_ok=True)
            
        # Split dataset
        n_train = int(len(self._dataset) * train_split)
        indices = np.random.permutation(len(self._dataset))
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        print(f"Exporting {n_train} train and {len(val_indices)} val images...")
        
        # Export training set
        self._export_split(
            train_indices, 
            images_train, 
            labels_train,
            include_masks,
        )
        
        # Export validation set
        self._export_split(
            val_indices,
            images_val,
            labels_val,
            include_masks,
        )
        
        # Create data.yaml
        yaml_path = self.output_dir / "data.yaml"
        yaml_content = f"""# Chicks4FreeID Dataset - YOLO Format
# Auto-generated for poultry detection fine-tuning

path: {self.output_dir.as_posix()}
train: images/train
val: images/val

nc: {len(self.CLASS_NAMES)}
names: {self.CLASS_NAMES}
"""
        
        yaml_path.write_text(yaml_content)
        print(f"Created {yaml_path}")
        
        return str(yaml_path)
    
    def _export_split(
        self,
        indices: np.ndarray,
        images_dir: Path,
        labels_dir: Path,
        include_masks: bool = False,
    ):
        """Export a subset of the dataset."""
        for idx in tqdm(indices, desc=f"Exporting to {images_dir.parent.name}"):
            sample = self._dataset[int(idx)]
            
            # Get image
            image = sample.get("image")
            if image is None:
                continue
                
            # Save image
            image_name = f"img_{idx:05d}.jpg"
            image_path = images_dir / image_name
            
            if isinstance(image, Image.Image):
                image.save(image_path)
                img_w, img_h = image.size
            else:
                # Assume numpy array
                img = Image.fromarray(image)
                img.save(image_path)
                img_h, img_w = image.shape[:2]
                
            # Generate YOLO labels
            label_lines = self._generate_yolo_labels(
                sample, 
                img_w, 
                img_h,
                include_masks,
            )
            
            # Save labels
            label_name = f"img_{idx:05d}.txt"
            label_path = labels_dir / label_name
            label_path.write_text("\n".join(label_lines))
            
    def _generate_yolo_labels(
        self,
        sample: Dict,
        img_w: int,
        img_h: int,
        include_masks: bool = False,
    ) -> List[str]:
        """
        Generate YOLO format label lines from sample annotations.
        
        YOLO format: class_id x_center y_center width height
        All values normalized to 0-1.
        """
        labels = []
        
        # Try different annotation formats
        bboxes = sample.get("bboxes", sample.get("bbox", []))
        
        if not bboxes:
            # Try to extract from other fields
            annotations = sample.get("annotations", [])
            if annotations:
                for ann in annotations:
                    if isinstance(ann, dict) and "bbox" in ann:
                        bboxes.append(ann["bbox"])
                        
        # Convert bboxes to YOLO format
        for bbox in bboxes:
            if len(bbox) >= 4:
                # Assume [x, y, w, h] or [x1, y1, x2, y2]
                if bbox[2] > 1 and bbox[3] > 1:
                    # Likely pixel coordinates
                    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                        # [x1, y1, x2, y2] format
                        x1, y1, x2, y2 = bbox[:4]
                        w, h = x2 - x1, y2 - y1
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                    else:
                        # [x, y, w, h] format
                        x, y, w, h = bbox[:4]
                        x_center = x + w / 2
                        y_center = y + h / 2
                        
                    # Normalize
                    x_center /= img_w
                    y_center /= img_h
                    w /= img_w
                    h /= img_h
                else:
                    # Already normalized
                    x_center, y_center, w, h = bbox[:4]
                    
                # Class 0 = chicken
                labels.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                
        return labels
    
    def get_sample_images(self, n: int = 5) -> List[Tuple[Image.Image, Dict]]:
        """
        Get sample images with annotations for preview.
        
        Args:
            n: Number of samples to return.
            
        Returns:
            List of (image, annotation_dict) tuples.
        """
        if self._dataset is None:
            self.load_dataset()
            
        samples = []
        for i in range(min(n, len(self._dataset))):
            sample = self._dataset[i]
            image = sample.get("image")
            samples.append((image, sample))
            
        return samples


def prepare_dataset_for_training(
    output_dir: Optional[str] = None,
    train_split: float = 0.8,
) -> str:
    """
    Convenience function to download and prepare dataset.
    
    Args:
        output_dir: Optional output directory.
        train_split: Train/val split ratio.
        
    Returns:
        Path to data.yaml for YOLO training.
    """
    loader = Chicks4FreeIDLoader(output_dir)
    loader.load_dataset()
    yaml_path = loader.export_to_yolo(train_split=train_split)
    
    print(f"\nDataset ready for training!")
    print(f"Use: yolo train data={yaml_path} model=yolov8n.pt epochs=10")
    
    return yaml_path


if __name__ == "__main__":
    # Run standalone to prepare dataset
    prepare_dataset_for_training()
