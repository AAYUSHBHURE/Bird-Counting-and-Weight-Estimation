"""
Poultry CCTV Video Analysis - Streamlit Demo App

Interactive showcase of bird counting and weight estimation system.
"""

import streamlit as st
import json
from pathlib import Path
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Poultry CCTV Analysis Demo",
    page_icon="🐔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .requirement-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #1f77b4;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🐔 Navigation")
page = st.sidebar.radio(
    "Select Section",
    ["🏠 Overview", "📊 Requirements", "🎬 Demo Video", "📈 Results", "🔧 API Documentation"]
)

# Load demo results
demo_json_path = Path("demo_final.json")
demo_results = None
if demo_json_path.exists():
    with open(demo_json_path, 'r') as f:
        demo_results = json.load(f)

# Main content
if page == "🏠 Overview":
    st.markdown('<div class="main-header">🐔 Poultry CCTV Video Analysis System</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Project Overview
    
    ML-powered API for analyzing poultry CCTV footage with:
    - **Bird Counting**: Accurate detection and tracking with stable IDs
    - **Weight Estimation**: Area-based proxy with calibration support
    - **Video Annotation**: Visual output with bounding boxes and metrics
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Detection</h3>
            <p>YOLOv8 Pretrained</p>
            <h2>COCO Class 14</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Tracking</h3>
            <p>ByteTrack Algorithm</p>
            <h2>30-Frame Buffer</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⚖️ Weight</h3>
            <p>Area-based Proxy</p>
            <h2>±15% Uncertainty</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🚀 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Detection & Tracking:**
        - Individual bird detection with bounding boxes
        - Stable tracking IDs across frames
        - Occlusion handling with Kalman filter
        - 30-frame track persistence
        """)
    
    with col2:
        st.markdown("""
        **Weight Estimation:**
        - Normalized area-based index
        - Per-bird median weights (stable)
        - Flock aggregate statistics
        - Calibration support for grams
        """)

elif page == "📊 Requirements":
    st.markdown('<div class="main-header">📊 Requirements Compliance</div>', unsafe_allow_html=True)
    
    st.markdown("## ✅ Mandatory Requirements Met")
    
    # Bird Counting Requirement
    st.markdown("""
    <div class="requirement-box">
        <h3>1️⃣ Bird Counting</h3>
        <div class="success-box">
            <strong>✅ FULLY IMPLEMENTED</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Detection:**
    - ✅ Bounding boxes with confidence scores
    - ✅ YOLOv8 pretrained (COCO class 14: bird)
    - ✅ Configurable thresholds (conf: 0.2, IoU: 0.5)
    
    **Stable Tracking IDs:**
    - ✅ ByteTrack algorithm for ID assignment
    - ✅ 30-frame track buffer for persistence
    - ✅ Kalman filter for position prediction
    
    **Count Over Time:**
    - ✅ Timestamp → count mapping (1-second buckets)
    - ✅ Total unique birds tracked
    - ✅ Average/max/min counts
    
    **Avoid Double-Counting:**
    - ✅ IoU-based matching prevents duplicates
    - ✅ Track history maintains unique IDs
    - ✅ Active count excludes lost tracks
    
    **Occlusion Handling:**
    - ✅ 30-frame buffer keeps IDs during occlusions
    - ✅ Kalman filter predicts hidden positions
    - ✅ Confidence decay for lost tracks
    
    **ID Switch Prevention:**
    - ✅ High match threshold (0.8)
    - ✅ ID switch rate metric reported
    - ✅ Conservative re-assignment policy
    """)
    
    st.markdown("---")
    
    # Weight Estimation Requirement
    st.markdown("""
    <div class="requirement-box">
        <h3>2️⃣ Weight Estimation</h3>
        <div class="success-box">
            <strong>✅ FULLY IMPLEMENTED</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Approach: (b) Calibration-based Pixel-to-Real Mapping**
    
    **Weight Proxy/Index:**
    ```python
    weight_index = (bbox_area / frame_height²) × 1000
    ```
    
    **Per-Bird Weights:**
    - ✅ Median-based stable weight per bird ID
    - ✅ Normalized by frame dimensions (scale-invariant)
    - ✅ ±15% uncertainty from posture variations
    
    **Aggregate Weights:**
    - ✅ Flock mean, std, min, max indices
    - ✅ Total birds and mean uncertainty
    
    **Calibration for Grams:**
    - ✅ Linear regression method implemented
    - ✅ Requires 50+ paired measurements (video → scale)
    - ✅ Formula: `weight_grams = slope × index + intercept`
    - ✅ R² metric for calibration quality
    
    **What's Needed for Gram Conversion:**
    1. Collect 50+ videos with known bird weights
    2. Run `WeightEstimator.calculate_calibration_regression()`
    3. Apply slope/intercept to convert indices to grams
    4. Expected R² > 0.85 for good calibration
    """)
    
    st.markdown("---")
    
    # Artifacts Requirement
    st.markdown("""
    <div class="requirement-box">
        <h3>3️⃣ Annotated Output</h3>
        <div class="success-box">
            <strong>✅ FULLY IMPLEMENTED</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Generated Artifacts:**
    - ✅ Annotated video with bounding boxes
    - ✅ Tracking IDs displayed on each bird
    - ✅ Count overlay at top of frame
    - ✅ Weight indices shown per bird
    - ✅ Continuous annotations (no flickering)
    - ✅ JSON results with all metrics
    """)

elif page == "🎬 Demo Video":
    st.markdown('<div class="main-header">🎬 Demo Video Showcase</div>', unsafe_allow_html=True)
    
    # Use browser-compatible H.264 encoded video
    demo_video_path = Path(__file__).parent / "demo_browser_compatible.mp4"
    annotated_video_path = Path(__file__).parent / "outputs" / "e3565aa3" / "tmpbv1l3hsq_annotated.mp4"
    
    if demo_video_path.exists():
        st.markdown("## 📹 Annotated Output Video")
        
        st.markdown("""
        **What you'll see in the video:**
        - 🔲 Individual bounding boxes around each bird
        - 🏷️ Labels showing: `ID:X 0.XX W:YYY` (ID, confidence, weight)
        - 📊 Count overlay at the top
        - 🎯 Stable weight values (not changing)
        - ✨ Continuous annotations (no flickering)
        """)
        
        # Display annotated video
        try:
            with open(demo_video_path, 'rb') as video_file:
                video_bytes = video_file.read()
            st.video(video_bytes)
            st.success(f"✅ Annotated video loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading video: {e}")
        
        st.markdown("### 📥 Download Annotated Video")
        if annotated_video_path.exists():
            st.info(f"""
            **Annotated video location:**  
            `{annotated_video_path.absolute()}`
            
            **To view:** Download and open in VLC Player or Windows Media Player
            """)
            
            try:
                with open(annotated_video_path, 'rb') as f:
                    annotated_bytes = f.read()
                st.download_button(
                    label="⬇️ Download Annotated Video",
                    data=annotated_bytes,
                    file_name="poultry_analysis_annotated.mp4",
                    mime="video/mp4"
                )
            except:
                pass
        
        if demo_results:
            st.markdown("### 📊 Video Analysis Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Duration", f"{demo_results['video_info']['duration_seconds']}s")
            
            with col2:
                st.metric("Total Birds", demo_results['count_summary']['total_unique_birds'])
            
            with col3:
                st.metric("Max Count", demo_results['count_summary']['max_count'])
            
            with col4:
                st.metric("ID Switches", f"{demo_results['tracking_metrics']['id_switch_rate']:.1%}")
            
            # Add per-bird weight estimates table
            st.markdown("### ⚖️ Per-Bird Weight Estimates")
            weights_df = pd.DataFrame(demo_results['weight_estimates'])
            st.dataframe(weights_df, use_container_width=True)
            
            st.info("""
            **Weight Index Explanation:**  
            - Values are normalized area-based proxies (not grams)
            - Higher index = larger bird
            - To convert to grams: need calibration with 50+ known weights
            - Formula: `weight_grams = slope × index + intercept`
            """)
    else:
        st.error(f"❌ Sample video not found at: `{demo_video_path}`")

elif page == "📈 Results":
    st.markdown('<div class="main-header">📈 Analysis Results</div>', unsafe_allow_html=True)
    
    if demo_results:
        # Video Info
        st.markdown("## 📹 Video Information")
        video_info = demo_results['video_info']
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Duration", f"{video_info['duration_seconds']}s")
        col2.metric("Total Frames", video_info['total_frames'])
        col3.metric("Source FPS", video_info['source_fps'])
        col4.metric("Processing FPS", video_info['processed_fps'])
        
        st.markdown("---")
        
        # Counting Results
        st.markdown("## 🐔 Bird Counting Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Count Summary")
            summary = demo_results['count_summary']
            st.metric("Total Unique Birds", summary['total_unique_birds'])
            st.metric("Average Count", f"{summary['avg_count']:.1f}")
            st.metric("Max Count", summary['max_count'])
            st.metric("Min Count", summary['min_count'])
        
        with col2:
            st.markdown("### Count Over Time")
            counts_df = pd.DataFrame(
                list(demo_results['counts'].items()),
                columns=['Timestamp', 'Count']
            )
            st.line_chart(counts_df.set_index('Timestamp'))
        
        st.markdown("---")
        
        # Weight Estimates
        st.markdown("## ⚖️ Weight Estimation Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Per-Bird Weights")
            weights_df = pd.DataFrame(demo_results['weight_estimates'])
            st.dataframe(weights_df, use_container_width=True)
        
        with col2:
            st.markdown("### Flock Statistics")
            flock = demo_results['flock_statistics']
            st.metric("Mean Weight Index", f"{flock['mean_index']:.2f}")
            st.metric("Std Deviation", f"{flock['std_index']:.2f}")
            st.metric("Min Weight", f"{flock['min_index']:.2f}")
            st.metric("Max Weight", f"{flock['max_index']:.2f}")
            st.metric("Mean Uncertainty", f"±{flock['mean_uncertainty']:.2f}")
        
        st.markdown("---")
        
        # Tracking Metrics
        st.markdown("## 🔍 Tracking Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Tracks", demo_results['tracking_metrics']['total_tracks'])
        
        with col2:
            id_switch_rate = demo_results['tracking_metrics']['id_switch_rate']
            st.metric("ID Switch Rate", f"{id_switch_rate:.1%}", 
                     delta="Perfect!" if id_switch_rate == 0 else None)
        
        # Sample Tracks
        st.markdown("### 📋 Sample Tracked Birds")
        tracks_df = pd.DataFrame(demo_results['tracks_sample'])
        if not tracks_df.empty:
            st.dataframe(tracks_df, use_container_width=True)
    
    else:
        st.warning("⚠️ No demo results found. Please run the analysis first.")

elif page == "🔧 API Documentation":
    st.markdown('<div class="main-header">🔧 API Documentation</div>', unsafe_allow_html=True)
    
    st.markdown("## 🚀 FastAPI Endpoints")
    
    # Health endpoint
    st.markdown("""
    ### GET `/health`
    Health check endpoint.
    
    **Response:**
    ```json
    {"status": "OK"}
    ```
    
    **Example:**
    ```bash
    curl http://localhost:8000/health
    ```
    """)
    
    st.markdown("---")
    
    # Analyze video endpoint
    st.markdown("""
    ### POST `/analyze_video`
    Analyze video for bird counting and weight estimation.
    
    **Parameters (multipart/form-data):**
    - `video` (file): Video file to analyze
    - `fps_sample` (int, optional): Frame sampling rate (default: 5)
    - `conf_thresh` (float, optional): Detection confidence threshold (default: 0.2)
    - `iou_thresh` (float, optional): IoU threshold for NMS (default: 0.5)
    - `generate_annotated` (bool, optional): Generate annotated video (default: true)
    
    **Response:**
    ```json
    {
      "video_info": {...},
      "counts": {"00:00": 3, "00:01": 3, ...},
      "count_summary": {...},
      "tracks_sample": [...],
      "weight_estimates": [...],
      "flock_statistics": {...},
      "artifacts": {"annotated_video": "..."},
      "processing_config": {...}
    }
    ```
    
    **Example:**
    ```bash
    curl -X POST "http://localhost:8000/analyze_video" \\
      -F "video=@chicken_farm.mp4" \\
      -F "fps_sample=5" \\
      -F "conf_thresh=0.2"
    ```
    """)
    
    st.markdown("---")
    
    st.markdown("## 🛠️ Setup & Run")
    
    st.code("""
# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn main:app --reload --port 8000

# Start Streamlit demo
streamlit run streamlit_app.py
    """, language="bash")
    
    st.markdown("---")
    
    st.markdown("## 📦 Project Structure")
    
    st.code("""
poultry-cctv-analysis/
├── main.py                  # FastAPI application
├── process_video.py         # ML pipeline
├── streamlit_app.py        # This demo app
├── config.py               # Configuration
├── requirements.txt        # Dependencies
├── src/
│   ├── detector.py         # YOLOv8 detection
│   ├── tracker.py          # ByteTrack tracking
│   ├── weight_estimator.py # Weight proxy
│   └── annotator.py        # Video annotation
├── models/
│   └── best.pt            # Fine-tuned model
└── outputs/               # Analysis results
    """, language="text")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Info")
st.sidebar.info("""
**Model:** YOLOv8n Pretrained  
**Tracker:** ByteTrack  
**Framework:** FastAPI + Streamlit  
**Version:** 1.0.0
""")
