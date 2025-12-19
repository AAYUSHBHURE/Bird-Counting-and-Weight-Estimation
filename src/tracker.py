"""
ByteTrack Tracking Module for Poultry CCTV Analysis

Provides multi-object tracking with stable IDs and occlusion handling
using Supervision's ByteTrack implementation.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import supervision as sv

from config import TRACKING_CONFIG


@dataclass
class TrackInfo:
    """Information about a tracked bird."""
    track_id: int
    boxes: List[np.ndarray] = field(default_factory=list)
    areas: List[float] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    frames: List[int] = field(default_factory=list)
    last_seen: int = 0
    active: bool = True


class PoultryTracker:
    """ByteTrack-based multi-object tracker for poultry."""
    
    def __init__(
        self,
        track_thresh: float = TRACKING_CONFIG["track_thresh"],
        match_thresh: float = TRACKING_CONFIG["match_thresh"],
        track_buffer: int = TRACKING_CONFIG["track_buffer"],
        frame_rate: int = TRACKING_CONFIG["frame_rate"],
    ):
        """
        Initialize the tracker.
        
        Args:
            track_thresh: Detection confidence threshold for tracking.
            match_thresh: IoU threshold for track association.
            track_buffer: Frames to keep lost tracks (Kalman persistence).
            frame_rate: Video frame rate for velocity estimation.
        """
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=frame_rate,
        )
        
        self.track_buffer = track_buffer
        self.tracks: Dict[int, TrackInfo] = {}
        self.frame_count = 0
        self.active_ids: set = set()
        
    def reset(self):
        """Reset tracker state for new video."""
        self.tracker.reset()
        self.tracks.clear()
        self.frame_count = 0
        self.active_ids.clear()
        
    def update(
        self,
        detections: sv.Detections,
        areas: Optional[np.ndarray] = None,
    ) -> sv.Detections:
        """
        Update tracker with new detections.
        
        Args:
            detections: Current frame detections.
            areas: Optional pre-computed areas for weight estimation.
            
        Returns:
            Detections with assigned tracker IDs.
        """
        self.frame_count += 1
        
        # Update ByteTrack
        tracked_detections = self.tracker.update_with_detections(detections)
        
        # Get tracker IDs
        tracker_ids = (
            tracked_detections.tracker_id 
            if tracked_detections.tracker_id is not None 
            else np.array([])
        )
        
        # Update internal track history
        self._update_track_history(tracked_detections, areas)
        
        return tracked_detections
    
    def _update_track_history(
        self,
        detections: sv.Detections,
        areas: Optional[np.ndarray] = None,
    ):
        """Update internal track history for all detections."""
        if detections.tracker_id is None:
            return
            
        current_ids = set()
        
        for i, track_id in enumerate(detections.tracker_id):
            if track_id is None:
                continue
                
            track_id = int(track_id)
            current_ids.add(track_id)
            
            if track_id not in self.tracks:
                self.tracks[track_id] = TrackInfo(track_id=track_id)
                
            track = self.tracks[track_id]
            track.boxes.append(detections.xyxy[i].copy())
            track.frames.append(self.frame_count)
            track.last_seen = self.frame_count
            track.active = True
            
            if detections.confidence is not None:
                track.confidences.append(float(detections.confidence[i]))
                
            if areas is not None and i < len(areas):
                track.areas.append(float(areas[i]))
            else:
                # Compute area from box
                box = detections.xyxy[i]
                area = (box[2] - box[0]) * (box[3] - box[1])
                track.areas.append(float(area))
                
        # Mark lost tracks as inactive
        for track_id, track in self.tracks.items():
            if track_id not in current_ids:
                frames_since_seen = self.frame_count - track.last_seen
                if frames_since_seen > self.track_buffer:
                    track.active = False
                    
        self.active_ids = current_ids
        
    def get_active_count(self) -> int:
        """Get count of currently active (visible) tracks."""
        return len(self.active_ids)
    
    def get_unique_count(self) -> int:
        """Get total unique birds seen (active + recently lost)."""
        return sum(1 for t in self.tracks.values() if t.active)
    
    def get_total_unique_count(self) -> int:
        """Get total unique birds ever tracked."""
        return len(self.tracks)
    
    def get_track_info(self, track_id: int) -> Optional[TrackInfo]:
        """Get detailed info for a specific track."""
        return self.tracks.get(track_id)
    
    def get_all_tracks(self) -> Dict[int, TrackInfo]:
        """Get all track information."""
        return self.tracks.copy()
    
    def get_sample_tracks(self, n: int = 5) -> List[Dict]:
        """
        Get sample track data for API response.
        
        Args:
            n: Number of tracks to sample.
            
        Returns:
            List of track dictionaries with id, box, timestamp.
        """
        samples = []
        sorted_tracks = sorted(
            self.tracks.items(),
            key=lambda x: x[1].last_seen,
            reverse=True
        )
        
        for track_id, track in sorted_tracks[:n]:
            if track.boxes:
                last_box = track.boxes[-1].tolist()
                samples.append({
                    "id": track_id,
                    "box": last_box,
                    "frame": track.last_seen,
                    "avg_confidence": (
                        np.mean(track.confidences) 
                        if track.confidences else 0.0
                    ),
                })
                
        return samples
    
    def get_id_switch_rate(self) -> float:
        """
        Estimate ID switch rate based on track fragmentation.
        
        Returns:
            Estimated switch rate (0-1).
        """
        if not self.tracks:
            return 0.0
            
        # Heuristic: short tracks relative to video length suggest switches
        avg_track_length = np.mean([
            len(t.frames) for t in self.tracks.values()
        ])
        
        expected_length = self.frame_count / max(len(self.tracks), 1)
        
        if expected_length == 0:
            return 0.0
            
        fragmentation = 1 - min(avg_track_length / expected_length, 1)
        
        return fragmentation
