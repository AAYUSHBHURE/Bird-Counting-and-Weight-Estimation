"""
Weight Estimation Module for Poultry CCTV Analysis

Provides area-based weight proxy calculation using bounding box
or segmentation mask areas, normalized by frame dimensions.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from config import WEIGHT_CONFIG


@dataclass
class WeightEstimate:
    """Weight estimation result for a single bird."""
    track_id: int
    index: float  # Normalized area-based weight proxy
    uncertainty: float  # ±15% uncertainty
    sample_count: int  # Number of area samples used
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.track_id,
            "index": round(self.index, 2),
            "uncertainty": round(self.uncertainty, 3),
            "sample_count": self.sample_count,
        }


class WeightEstimator:
    """
    Area-based weight proxy estimator for poultry.
    
    Uses normalized bounding box or mask area as a relative weight index:
        index = (area / frame_height²) × normalization_factor
        
    This provides a scale-invariant proxy that correlates with actual weight.
    For absolute gram estimates, calibration with 50+ paired measurements
    (video area → scale weight) is required.
    """
    
    def __init__(
        self,
        normalization_factor: float = WEIGHT_CONFIG["normalization_factor"],
        uncertainty_percent: float = WEIGHT_CONFIG["uncertainty_percent"],
    ):
        """
        Initialize the weight estimator.
        
        Args:
            normalization_factor: Multiplier for area index (default 1000).
            uncertainty_percent: Uncertainty range (±15% from posture).
        """
        self.normalization_factor = normalization_factor
        self.uncertainty_percent = uncertainty_percent
        self.frame_height: Optional[int] = None
        
        # Track history for averaging
        self.track_areas: Dict[int, List[float]] = {}
        
    def set_frame_dimensions(self, height: int, width: int):
        """Set frame dimensions for normalization."""
        self.frame_height = height
        
    def reset(self):
        """Reset estimator state for new video."""
        self.track_areas.clear()
        
    def update(
        self,
        track_id: int,
        area: float,
    ):
        """
        Update area history for a tracked bird.
        
        Args:
            track_id: Bird track ID.
            area: Detected area (pixels²).
        """
        if track_id not in self.track_areas:
            self.track_areas[track_id] = []
        self.track_areas[track_id].append(area)
        
    def update_batch(
        self,
        tracker_ids: np.ndarray,
        areas: np.ndarray,
    ):
        """
        Update areas for multiple tracks at once.
        
        Args:
            tracker_ids: Array of track IDs.
            areas: Array of corresponding areas.
        """
        for track_id, area in zip(tracker_ids, areas):
            if track_id is not None:
                self.update(int(track_id), float(area))
                
    def calculate_index(self, area: float) -> float:
        """
        Calculate normalized weight index from area.
        
        Args:
            area: Pixel area of bounding box or mask.
            
        Returns:
            Normalized weight index.
        """
        if self.frame_height is None or self.frame_height == 0:
            # Fallback: return raw scaled area
            return area / 1000.0
            
        # Normalize by frame height squared for scale invariance
        normalized = area / (self.frame_height ** 2)
        
        return normalized * self.normalization_factor
    
    def get_estimate(self, track_id: int) -> Optional[WeightEstimate]:
        """
        Get weight estimate for a specific track.
        
        Args:
            track_id: Bird track ID.
            
        Returns:
            WeightEstimate or None if track not found.
        """
        if track_id not in self.track_areas or not self.track_areas[track_id]:
            return None
            
        areas = self.track_areas[track_id]
        
        # Use median for robustness against outliers (posture changes)
        median_area = np.median(areas)
        index = self.calculate_index(median_area)
        uncertainty = index * self.uncertainty_percent
        
        return WeightEstimate(
            track_id=track_id,
            index=index,
            uncertainty=uncertainty,
            sample_count=len(areas),
        )
    
    def get_all_estimates(self) -> List[WeightEstimate]:
        """
        Get weight estimates for all tracked birds.
        
        Returns:
            List of WeightEstimate objects.
        """
        estimates = []
        for track_id in self.track_areas:
            estimate = self.get_estimate(track_id)
            if estimate:
                estimates.append(estimate)
        return estimates
    
    def get_flock_statistics(self) -> Dict:
        """
        Calculate aggregate flock weight statistics.
        
        Returns:
            Dictionary with mean, std, min, max indices.
        """
        estimates = self.get_all_estimates()
        
        if not estimates:
            return {
                "mean_index": 0.0,
                "std_index": 0.0,
                "min_index": 0.0,
                "max_index": 0.0,
                "total_birds": 0,
            }
            
        indices = [e.index for e in estimates]
        
        return {
            "mean_index": round(float(np.mean(indices)), 2),
            "std_index": round(float(np.std(indices)), 2),
            "min_index": round(float(np.min(indices)), 2),
            "max_index": round(float(np.max(indices)), 2),
            "total_birds": len(estimates),
            "mean_uncertainty": round(float(np.mean(indices)) * self.uncertainty_percent, 3),
        }
    
    @staticmethod
    def calculate_calibration_regression(
        area_indices: List[float],
        actual_weights_grams: List[float],
    ) -> Tuple[float, float, float]:
        """
        Calculate linear regression for calibrating proxy to actual weight.
        
        Args:
            area_indices: List of computed area indices.
            actual_weights_grams: Corresponding actual weights in grams.
            
        Returns:
            Tuple of (slope, intercept, r_squared).
        """
        if len(area_indices) < 2:
            raise ValueError("Need at least 2 paired measurements for calibration")
            
        x = np.array(area_indices)
        y = np.array(actual_weights_grams)
        
        # Linear regression: y = mx + b
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs
        
        # Calculate R²
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return float(slope), float(intercept), float(r_squared)
    
    def apply_calibration(
        self,
        slope: float,
        intercept: float,
    ) -> List[Dict]:
        """
        Apply calibration to convert indices to grams.
        
        Args:
            slope: Calibration slope from regression.
            intercept: Calibration intercept.
            
        Returns:
            List of calibrated weight estimates.
        """
        estimates = self.get_all_estimates()
        calibrated = []
        
        for estimate in estimates:
            weight_grams = slope * estimate.index + intercept
            uncertainty_grams = slope * estimate.uncertainty
            
            calibrated.append({
                "id": estimate.track_id,
                "weight_grams": round(weight_grams, 1),
                "uncertainty_grams": round(uncertainty_grams, 1),
                "index": estimate.index,
            })
            
        return calibrated
