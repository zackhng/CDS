"""
Visual (Image / Video) Emotion Analysis Page
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import streamlit as st
import numpy as np
from PIL import Image as PILImage
from util.emotion_utils import process_image, process_video, EMOTION_EMOJIS, EMOTION_COLORS
from util.viz_utils import (
    confidence_bar_chart, emotion_result_card,
    frame_emotion_timeline, radar_chart,
)


st.set_page_config(page_title="E-Motion · Visual", page_icon="🖼️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .frame-grid { display: flex; flex-wrap: wrap; gap: 8px; }
    .info-badge {
        display: inline-block; background: #1e2a3a;
        border: 1px solid #2a3140; border-radius: 8px;
        padding: 8px 14px; font-size: 0.88rem; color: #a8c4e0; margin: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🖼️ Visual Emotion Analysis")
st.markdown("Upload an image or video file for face detection and emotion classification.")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_img, tab_vid, tab_cam = st.tabs(["🖼️ Image Upload", "🎬 Video Upload", "📷 Webcam Snapshot"])

result = None
input_type = None

# ─── Image upload ─────────────────────────────────────────────────────────────
with tab_img:
    img_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "webp"],
        help="Faces are detected automatically. Works best with frontal, well-lit faces.",
    )
    if img_file:
        img_pil = PILImage.open(img_file).convert("RGB")
        col_orig, col_info = st.columns([2, 1])
        with col_orig:
            st.image(img_pil, caption="Uploaded Image", use_container_width=True)
        with col_info:
            w, h = img_pil.size
            st.markdown(f"""
            <div style="background:#161b27;border:1px solid #2a3140;
                        border-radius:10px;padding:16px;margin-top:10px;">
                <b>📐 Image Info</b><br><br>
                <span class="info-badge">Size: {w}×{h}</span>
                <span class="info-badge">Mode: {img_pil.mode}</span>
                <span class="info-badge">{img_file.size/1024:.1f} KB</span>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🔍 Analyse Image", type="primary"):
            with st.spinner("Running face detection and emotion model…"):
                result = process_image(img_pil)
            input_type = "image"
            st.session_state["img_result"] = result
            st.session_state["img_pil"] = img_pil

# ─── Video upload ─────────────────────────────────────────────────────────────
with tab_vid:
    vid_file = st.file_uploader(
        "Upload a video file", type=["mp4", "avi", "mov", "mkv"],
        help="Frames are extracted and processed individually. Keep videos under 30 seconds for best performance.",
    )
    if vid_file:
        st.video(vid_file)
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            max_frames = st.slider("Max frames to analyse", 5, 50, 20)
        with col_v2:
            st.markdown(f"""
            <div style="background:#161b27;border:1px solid #2a3140;
                        border-radius:10px;padding:14px;">
                <b>📹 Video Info</b><br>
                <span class="info-badge">{vid_file.size/1024/1024:.1f} MB</span>
                <span class="info-badge">{max_frames} frames</span>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🔍 Analyse Video", type="primary"):
            with st.spinner(f"Extracting and analysing up to {max_frames} frames…"):
                result = process_video(vid_file.read(), max_frames=max_frames)
            input_type = "video"
            st.session_state["vid_result"] = result

# ─── Webcam ───────────────────────────────────────────────────────────────────
with tab_cam:
    st.info("Take a snapshot using your webcam. Make sure your face is visible.")
    cam_img = st.camera_input("Take a photo")
    if cam_img:
        img_pil = PILImage.open(cam_img).convert("RGB")
        if st.button("🔍 Analyse Snapshot", type="primary"):
            with st.spinner("Running face detection on snapshot…"):
                result = process_image(img_pil, label_hint="webcam_snapshot")
            input_type = "image"
            st.session_state["img_result"] = result
            st.session_state["img_pil"] = img_pil

# ── Show stored results ───────────────────────────────────────────────────────
if result is None and "img_result" in st.session_state and input_type != "video":
    result = st.session_state["img_result"]
    input_type = "image"
if result is None and "vid_result" in st.session_state:
    result = st.session_state["vid_result"]
    input_type = "video"

# ── Display Results ───────────────────────────────────────────────────────────
if result:
    st.markdown("---")
    st.markdown("## 📊 Results")

    if input_type == "image":
        col_card, col_chart = st.columns([1, 2])
        with col_card:
            emotion_result_card(result["emotion"], result["confidence"], modality="Visual Model")
            faces = result.get("faces_detected", 0)
            st.markdown(f"""
            <div style="background:#161b27;border:1px solid #2a3140;
                        border-radius:10px;padding:14px;margin-top:10px;">
                👤 <b>{faces} face(s)</b> detected
            </div>
            """, unsafe_allow_html=True)

        with col_chart:
            sorted_scores = dict(sorted(result["all_scores"].items(), key=lambda x: -x[1]))
            st.plotly_chart(confidence_bar_chart(sorted_scores), use_container_width=True)

        # Annotated image
        annotated = result.get("annotated_image")
        if annotated is not None:
            st.markdown("### 🔍 Annotated Image (Face Detections)")
            st.image(annotated, caption="Detected faces highlighted in green",
                     use_container_width=True, clamp=True)

    elif input_type == "video":
        col_card, col_chart = st.columns([1, 2])
        with col_card:
            emotion_result_card(result["emotion"], result["confidence"], modality="Visual Model (Video)")
            frame_results = result.get("frame_results", [])
            st.markdown(f"""
            <div style="background:#161b27;border:1px solid #2a3140;
                        border-radius:10px;padding:14px;margin-top:10px;">
                🎞️ <b>{len(frame_results)} frames</b> analysed
            </div>
            """, unsafe_allow_html=True)

        with col_chart:
            sorted_scores = dict(sorted(result["all_scores"].items(), key=lambda x: -x[1]))
            st.plotly_chart(confidence_bar_chart(sorted_scores, title="Aggregated Emotion Scores"),
                            use_container_width=True)

        # Timeline
        frame_results = result.get("frame_results", [])
        if frame_results:
            st.markdown("### 📈 Frame-Level Emotion Timeline")
            fig_tl = frame_emotion_timeline(frame_results)
            if fig_tl:
                st.plotly_chart(fig_tl, use_container_width=True)

            # Dominant emotion per frame table
            with st.expander("📋 Per-Frame Breakdown"):
                import pandas as pd
                df = pd.DataFrame([
                    {
                        "Frame": r["frame"],
                        "Emotion": f"{EMOTION_EMOJIS.get(r['emotion'],'')} {r['emotion'].capitalize()}",
                        "Confidence (%)": f"{r['confidence']*100:.1f}",
                    }
                    for r in frame_results
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)

        # Sample frames
        sample_frames = result.get("sample_frames", [])
        if sample_frames:
            st.markdown("### 🎞️ Sample Frames")
            cols = st.columns(min(len(sample_frames), 3))
            for i, (col, frame) in enumerate(zip(cols, sample_frames[:3])):
                with col:
                    st.image(frame, caption=f"Frame {frame_results[i]['frame'] if i < len(frame_results) else i}",
                             use_container_width=True, clamp=True)

    # Radar for all types
    st.markdown("### 🕸️ Emotion Radar")
    st.plotly_chart(radar_chart(result["all_scores"]), use_container_width=True)

else:
    st.markdown("""
    <div style="
        background:#161b27; border:1px dashed #2a3140;
        border-radius:14px; padding:40px; text-align:center; color:#8892a4;
    ">
        <div style="font-size:3rem;">🖼️</div><br>
        Upload an image, video, or take a webcam snapshot to get started.
    </div>
    """, unsafe_allow_html=True)
