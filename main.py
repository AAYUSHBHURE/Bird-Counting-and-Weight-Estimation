"""
FastAPI Application for Poultry CCTV Video Analysis

Provides REST API endpoints for video upload and analysis:
- GET /health: Health check
- POST /analyze_video: Video analysis with configurable parameters
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional
import uuid

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from config import API_CONFIG, OUTPUTS_DIR
from process_video import VideoProcessor


# Initialize FastAPI app
app = FastAPI(
    title="Poultry CCTV Video Analysis API",
    description="""
    ML-powered API for analyzing poultry CCTV footage.
    
    Features:
    - Bird counting with detection and tracking
    - Weight estimation via area-based proxy
    - Annotated video output with bounding boxes and IDs
    
    Built with YOLOv8 + ByteTrack + FastAPI.
    """,
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        {"status": "OK"} if service is running.
    """
    return {"status": "OK"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Poultry CCTV Video Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/analyze_video": "POST - Analyze video for bird counting and weight estimation",
            "/outputs/{filename}": "GET - Download output files",
        },
        "documentation": "/docs",
    }


@app.post("/analyze_video")
async def analyze_video(
    video: UploadFile = File(..., description="Video file to analyze (MP4, AVI, etc.)"),
    fps_sample: int = Form(5, description="Frame rate for processing (1-30)"),
    conf_thresh: float = Form(0.5, description="Detection confidence threshold (0-1)"),
    iou_thresh: float = Form(0.7, description="IoU threshold for NMS (0-1)"),
    generate_annotated: bool = Form(True, description="Generate annotated output video"),
    model_path: Optional[str] = Form(None, description="Optional custom model path"),
):
    """
    Analyze a video file for bird counting and weight estimation.
    
    Accepts multipart form data with video file and optional parameters.
    
    Returns JSON with:
    - counts: Timestamp → bird count mapping
    - tracks_sample: Sample of tracked birds with IDs
    - weight_estimates: Per-bird weight proxy indices
    - flock_statistics: Aggregate flock weight statistics
    - artifacts: Paths to generated output files
    """
    # Validate parameters
    if fps_sample < 1 or fps_sample > 30:
        raise HTTPException(
            status_code=400,
            detail="fps_sample must be between 1 and 30",
        )
        
    if conf_thresh < 0 or conf_thresh > 1:
        raise HTTPException(
            status_code=400,
            detail="conf_thresh must be between 0 and 1",
        )
        
    if iou_thresh < 0 or iou_thresh > 1:
        raise HTTPException(
            status_code=400,
            detail="iou_thresh must be between 0 and 1",
        )
        
    # Create unique output directory for this request
    request_id = str(uuid.uuid4())[:8]
    output_dir = OUTPUTS_DIR / request_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded video to temp file
    temp_video_path = None
    try:
        # Get original filename extension
        ext = Path(video.filename).suffix or ".mp4"
        
        # Create temp file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=ext,
            dir=str(output_dir),
        ) as temp_file:
            temp_video_path = temp_file.name
            
            # Copy uploaded file
            shutil.copyfileobj(video.file, temp_file)
            
        # Initialize processor
        processor = VideoProcessor(
            model_path=model_path,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            fps_sample=fps_sample,
        )
        
        # Process video
        results = processor.process_video(
            temp_video_path,
            output_dir=str(output_dir),
            generate_annotated=generate_annotated,
        )
        
        # Update artifact paths to be relative for API response
        if results["artifacts"]["annotated_video"]:
            video_filename = Path(results["artifacts"]["annotated_video"]).name
            results["artifacts"]["annotated_video"] = f"/outputs/{request_id}/{video_filename}"
            
        if results["artifacts"].get("results_json"):
            json_filename = Path(results["artifacts"]["results_json"]).name
            results["artifacts"]["results_json"] = f"/outputs/{request_id}/{json_filename}"
            
        # Add request metadata
        results["request_id"] = request_id
        results["original_filename"] = video.filename
        
        return JSONResponse(content=results)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}",
        )
        
    finally:
        # Clean up temp video file
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
            except Exception:
                pass


@app.get("/outputs/{request_id}/{filename}")
async def get_output_file(request_id: str, filename: str):
    """
    Download an output file.
    
    Args:
        request_id: Request ID from analyze_video response.
        filename: Name of the file to download.
        
    Returns:
        The requested file.
    """
    file_path = OUTPUTS_DIR / request_id / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {filename}",
        )
        
    # Determine media type
    media_type = "application/octet-stream"
    if filename.endswith(".mp4"):
        media_type = "video/mp4"
    elif filename.endswith(".json"):
        media_type = "application/json"
        
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type=media_type,
    )


@app.get("/outputs")
async def list_outputs():
    """List all output directories."""
    outputs = []
    
    if OUTPUTS_DIR.exists():
        for item in OUTPUTS_DIR.iterdir():
            if item.is_dir():
                files = [f.name for f in item.iterdir() if f.is_file()]
                outputs.append({
                    "request_id": item.name,
                    "files": files,
                })
                
    return {"outputs": outputs}


# Sample JSON response for documentation
SAMPLE_RESPONSE = {
    "video_info": {
        "filename": "sample.mp4",
        "duration_seconds": 30.0,
        "total_frames": 900,
        "source_fps": 30.0,
        "processed_fps": 5,
    },
    "counts": {
        "00:01": 45,
        "00:02": 47,
        "00:03": 46,
    },
    "count_summary": {
        "max_count": 47,
        "min_count": 45,
        "avg_count": 46.0,
        "total_unique_birds": 52,
    },
    "tracks_sample": [
        {"id": 1, "box": [100, 150, 180, 230], "frame": 150, "avg_confidence": 0.92},
        {"id": 2, "box": [250, 200, 330, 280], "frame": 148, "avg_confidence": 0.88},
    ],
    "tracking_metrics": {
        "total_tracks": 52,
        "id_switch_rate": 0.023,
    },
    "weight_estimates": [
        {"id": 1, "index": 12.5, "uncertainty": 1.875, "sample_count": 30},
        {"id": 2, "index": 11.8, "uncertainty": 1.77, "sample_count": 28},
    ],
    "flock_statistics": {
        "mean_index": 12.15,
        "std_index": 1.23,
        "min_index": 9.5,
        "max_index": 15.2,
        "total_birds": 52,
        "mean_uncertainty": 1.82,
    },
    "artifacts": {
        "annotated_video": "/outputs/abc12345/sample_annotated.mp4",
        "results_json": "/outputs/abc12345/sample_results.json",
    },
    "request_id": "abc12345",
}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=True,
    )
