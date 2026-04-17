"""
E-Motion · Multimodal Emotion Detection
Combines visual, audio, and text analysis with late fusion
"""

import sys
import os
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import streamlit as st
import numpy as np
from PIL import Image as PILImage
from util.emotion_utils import (
    process_image, process_video, process_audio, process_text,
    fuse_predictions, transcribe_audio, extract_audio_from_video,
    EMOTION_EMOJIS, EMOTION_COLORS, EMOTIONS
)
from util.viz_utils import (
    confidence_bar_chart, emotion_result_card,
    frame_emotion_timeline, radar_chart,
)

st.set_page_config(
    page_title="E-Motion · Multimodal Fusion",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0e1117; }

    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3498db, #8e44ad, #e74c3c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
        margin-bottom: 0.5rem;
    }
    .hero-sub {
        font-size: 1.15rem;
        color: #8892a4;
        margin-bottom: 2rem;
    }
    
    .pipeline-step {
        background: #161b27; border: 1px solid #2a3140;
        border-radius: 12px; padding: 16px; margin: 10px 0;
        border-left: 4px solid #3498db;
    }
    .pipeline-step.done { border-left-color: #2ecc71; }
    .pipeline-step.error { border-left-color: #e74c3c; }
    
    .modality-card {
        background: #161b27; border: 1px solid #2a3140;
        border-radius: 14px; padding: 16px; margin: 8px;
        text-align: center; flex: 1;
    }
    .modality-emoji { font-size: 2rem; margin-bottom: 8px; }
    .modality-emotion { font-size: 1.2rem; font-weight: 600; color: #e0e6f0; margin: 8px 0; }
    .modality-confidence { font-size: 0.9rem; color: #8892a4; }
    
    .fusion-box {
        background: linear-gradient(135deg, #1a1f35, #161b27);
        border: 2px solid #3498db44;
        border-radius: 16px; padding: 28px; text-align: center; margin: 16px 0;
    }
    .fusion-emotion { 
        font-size: 2.5rem; font-weight: 700; margin: 8px 0;
        background: linear-gradient(135deg, #3498db, #8e44ad);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .info-badge {
        display: inline-block; background: #1e2a3a;
        border: 1px solid #2a3140; border-radius: 8px;
        padding: 8px 14px; font-size: 0.88rem; color: #a8c4e0; margin: 4px;
    }
    
    div[data-testid="stSidebar"] { background: #161b27; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🎭 E-Motion Multimodal</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Visual + Audio + Text Emotion Fusion</p>', unsafe_allow_html=True)
st.markdown("Upload an image, video, or webcam snapshot. Videos automatically extract audio and transcribe speech for full multimodal analysis.")
st.markdown("---")

# ── Fusion Settings Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Fusion Weights")
    w_visual = st.slider("🖼️ Visual", 0.0, 1.0, 0.40)
    w_audio = st.slider("🔊 Audio", 0.0, 1.0, 0.35)
    w_text = st.slider("🔤 Text", 0.0, 1.0, 0.25)
    
    fusion_weights = {"visual": w_visual, "audio": w_audio, "text": w_text}
    total_w = sum(fusion_weights.values())
    
    if abs(total_w - 1.0) > 0.01:
        st.warning(f"⚠️ Weights sum to {total_w:.2f} (will be normalized)")
    else:
        st.success(f"✓ Weights normalized")

# ── Input Tabs ──────────────────────────────────────────────────────────────────
st.markdown("## 📥 Input")

tab_img, tab_vid, tab_cam = st.tabs(["🖼️ Image", "🎬 Video", "📷 Webcam"])

result = None
input_type = None
video_bytes = None

# ─── Image upload ───────────────────────────────────────────────────────────────
with tab_img:
    img_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "webp"],
        help="Faces are detected automatically.",
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

        if st.button("🔍 Analyse Image", type="primary", key="btn_img"):
            with st.spinner("Running face detection and emotion model…"):
                result = process_image(img_pil)
            input_type = "image"
            st.session_state["img_result"] = result
            st.session_state["img_pil"] = img_pil

# ─── Video upload ───────────────────────────────────────────────────────────────
with tab_vid:
    vid_file = st.file_uploader(
        "Upload a video file", type=["mp4", "avi", "mov", "mkv"],
        help="Videos must be under 100MB. Audio will be extracted for multimodal analysis.",
    )
    if vid_file:
        st.video(vid_file)
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            max_frames = st.slider("Max frames to analyse", 5, 50, 20, key="max_frames_video")
        with col_v2:
            st.markdown(f"""
            <div style="background:#161b27;border:1px solid #2a3140;
                        border-radius:10px;padding:14px;">
                <b>📹 Video Info</b><br>
                <span class="info-badge">{vid_file.size/1024/1024:.1f} MB</span>
                <span class="info-badge">{max_frames} frames</span>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🔍 Analyse Video", type="primary", key="btn_vid"):
            video_bytes = vid_file.read()
            input_type = "video"

# ─── Webcam ─────────────────────────────────────────────────────────────────────
with tab_cam:
    st.info("Take a snapshot using your webcam.")
    cam_img = st.camera_input("Take a photo")
    if cam_img:
        img_pil = PILImage.open(cam_img).convert("RGB")
        if st.button("🔍 Analyse Snapshot", type="primary", key="btn_cam"):
            with st.spinner("Running face detection on snapshot…"):
                result = process_image(img_pil)
            input_type = "image"
            st.session_state["img_result"] = result
            st.session_state["img_pil"] = img_pil

# ── Show stored results (for images) ─────────────────────────────────────────────
if result is None and "img_result" in st.session_state and input_type != "video":
    result = st.session_state["img_result"]
    input_type = "image"

# ── Multimodal Video Analysis ───────────────────────────────────────────────────
if video_bytes and input_type == "video":
    st.markdown("---")
    st.markdown("## 🔄 Multimodal Pipeline Execution")
    
    progress = st.progress(0)
    status = st.empty()
    
    text_result = None
    audio_result = None
    visual_result = None
    transcript = ""
    extracted_frames = []
    tmp_vid_path = None
    wav_path = None
    audio_bytes_extracted = None
    
    # ── Step 1: Visual Analysis (Frame Extraction) ────────────────────────
    status.markdown("""
    <div class="pipeline-step">
        <b>1️⃣ Extracting frames from video…</b>
    </div>""", unsafe_allow_html=True)
    progress.progress(15)
    
    visual_result = process_video(video_bytes, max_frames=max_frames)
    em = visual_result["emotion"]
    n_frames = len(visual_result.get("frame_results", []))
    extracted_frames = visual_result.get("sample_frames", [])
    
    status.markdown(f"""
    <div class="pipeline-step done">
        <b>✓ Extracted {n_frames} frames for visual analysis</b><br>
        <span style="color:#8892a4;">Visual Emotion: {EMOTION_EMOJIS.get(em,'')} {em.capitalize()} ({visual_result['confidence']*100:.1f}%)</span>
    </div>""", unsafe_allow_html=True)
    
    # ── Step 2: Audio Extraction ─────────────────────────────────────────────
    status.markdown("""
    <div class="pipeline-step">
        <b>2️⃣ Extracting audio track from video…</b>
    </div>""", unsafe_allow_html=True)
    progress.progress(35)
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid:
        tmp_vid.write(video_bytes)
        tmp_vid_path = tmp_vid.name
    
    wav_path = tmp_vid_path.replace(".mp4", ".wav")
    audio_ok = extract_audio_from_video(tmp_vid_path, wav_path)
    
    if audio_ok and os.path.exists(wav_path):
        with open(wav_path, "rb") as f:
            audio_bytes_extracted = f.read()
        status.markdown(f"""
        <div class="pipeline-step done">
            <b>✓ Audio extracted successfully ({len(audio_bytes_extracted)/1024/1024:.2f} MB)</b>
        </div>""", unsafe_allow_html=True)
        st.session_state["extracted_wav"] = audio_bytes_extracted
        st.session_state["wav_filename"] = "extracted_audio.wav"
    else:
        status.markdown("""
        <div class="pipeline-step error">
            <b>⚠️ Audio extraction failed (will use visual + mock audio)</b>
        </div>""", unsafe_allow_html=True)
        audio_bytes_extracted = None
    
    # ── Step 3: Speech-to-Text ───────────────────────────────────────────────
    status.markdown("""
    <div class="pipeline-step">
        <b>3️⃣ Converting speech to text…</b>
    </div>""", unsafe_allow_html=True)
    progress.progress(50)
    
    if audio_ok and os.path.exists(wav_path):
        transcript = transcribe_audio(wav_path)
    else:
        transcript = "Could not extract audio from video."
    
    status.markdown(f"""
    <div class="pipeline-step done">
        <b>✓ Transcription complete</b><br>
        <span style="color:#a8c4e0; font-size:0.9rem;">{transcript[:200]}{"…" if len(transcript)>200 else ""}</span>
    </div>""", unsafe_allow_html=True)
    st.session_state["transcript"] = transcript
    
    # ── Step 4: Text Model ───────────────────────────────────────────────────
    status.markdown("""
    <div class="pipeline-step">
        <b>4️⃣ Running text emotion detection…</b>
    </div>""", unsafe_allow_html=True)
    progress.progress(65)
    
    if transcript and transcript != "Could not extract audio from video.":
        text_result = process_text(transcript)
    else:
        text_result = None
    
    if text_result:
        em = text_result["emotion"]
        status.markdown(f"""
        <div class="pipeline-step done">
            <b>✓ Text Emotion: {EMOTION_EMOJIS.get(em,'')} {em.capitalize()}</b>
            ({text_result['confidence']*100:.1f}%)
        </div>""", unsafe_allow_html=True)
    
    # ── Step 5: Audio Model ──────────────────────────────────────────────────
    status.markdown("""
    <div class="pipeline-step">
        <b>5️⃣ Running audio emotion detection…</b>
    </div>""", unsafe_allow_html=True)
    progress.progress(80)
    
    if audio_bytes_extracted:
        audio_result = process_audio(audio_bytes_extracted, filename="extracted_audio.wav")
    else:
        audio_result = None
    
    if audio_result:
        em = audio_result["emotion"]
        status.markdown(f"""
        <div class="pipeline-step done">
            <b>✓ Audio Emotion: {EMOTION_EMOJIS.get(em,'')} {em.capitalize()}</b>
            ({audio_result['confidence']*100:.1f}%)
        </div>""", unsafe_allow_html=True)
    
    # ── Step 6: Late Fusion ──────────────────────────────────────────────────
    status.markdown("""
    <div class="pipeline-step">
        <b>6️⃣ Fusing all modalities…</b>
    </div>""", unsafe_allow_html=True)
    progress.progress(95)
    
    fused = fuse_predictions(text_result, audio_result, visual_result, weights=fusion_weights)
    
    progress.progress(100)
    status.empty()
    
    # ── Store extracted data in session ──────────────────────────────────────
    st.session_state["extracted_frames"] = extracted_frames
    st.session_state["visual_result"] = visual_result
    
    # ── Cleanup temp video file (but keep wav for download) ──────────────────
    try:
        if tmp_vid_path and os.path.exists(tmp_vid_path):
            os.unlink(tmp_vid_path)
    except Exception:
        pass
    
    result = fused
    input_type = "multimodal"

# ── Display Results ──────────────────────────────────────────────────────────────
if result:
    st.markdown("---")
    st.markdown("## 📊 Results")

    if input_type == "image":
        # ─ Single Image ──────────────────────────────────────────────────────
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

        st.markdown("### 🕸️ Emotion Radar")
        st.plotly_chart(radar_chart(result["all_scores"]), use_container_width=True)

    elif input_type == "multimodal":
        # ─ Multimodal Fusion ─────────────────────────────────────────────────
        
        # Final Fusion Result
        st.markdown("### 🎯 Final Fused Prediction")
        final_color = EMOTION_COLORS.get(result["emotion"], "#888")
        final_emoji = EMOTION_EMOJIS.get(result["emotion"], "🎭")
        final_pct = result["confidence"] * 100
        
        st.markdown(f"""
        <div class="fusion-box">
            <div style="font-size:3rem;">{final_emoji}</div>
            <div class="fusion-emotion">{result["emotion"].capitalize()}</div>
            <div style="font-size:1.2rem;color:#8892a4;">{final_pct:.1f}% Confidence</div>
        </div>
        """, unsafe_allow_html=True)
        
        # ─ Extracted Media ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 📥 Extracted Media")
        
        # Tab for extracted content
        tab_frames, tab_audio, tab_text = st.tabs(["🖼️ Frames", "🔊 Audio", "🔤 Text"])
        
        # ─ Extracted Frames ─────────────────────────────────────────────────
        with tab_frames:
            extracted_frames = st.session_state.get("extracted_frames", [])
            if extracted_frames:
                st.success(f"✓ Extracted {len(extracted_frames)} frames from video")
                
                # Display frames in a grid
                n_cols = 3
                cols = st.columns(n_cols)
                for idx, frame in enumerate(extracted_frames):
                    col = cols[idx % n_cols]
                    with col:
                        st.image(frame, caption=f"Frame {idx + 1}", use_container_width=True)
            else:
                st.info("No frames extracted yet. Run the analysis first.")
        
        # ─ Extracted Audio ──────────────────────────────────────────────────
        with tab_audio:
            audio_bytes_data = st.session_state.get("extracted_wav")
            if audio_bytes_data:
                st.success("✓ Audio successfully extracted from video")
                
                # Display audio player
                st.audio(audio_bytes_data, format="audio/wav")
                
                # Download button
                st.download_button(
                    label="📥 Download WAV File",
                    data=audio_bytes_data,
                    file_name="extracted_audio.wav",
                    mime="audio/wav"
                )
                
                file_size_mb = len(audio_bytes_data) / 1024 / 1024
                st.caption(f"File size: {file_size_mb:.2f} MB")
            else:
                st.warning("⚠️ No audio extracted. Try re-running the analysis.")
        
        # ─ Speech-to-Text Transcript ────────────────────────────────────────
        with tab_text:
            transcript = st.session_state.get("transcript", "")
            if transcript:
                st.success("✓ Speech-to-text transcription complete")
                
                # Display full transcript
                st.markdown("**Full Transcript:**")
                st.markdown(f"""
                <div style="background:#0d1117;border:1px solid #2a3140;border-radius:8px;
                            padding:16px;font-family:monospace;font-size:0.95rem;
                            color:#a8c4e0;line-height:1.6;">
                {transcript}
                </div>
                """, unsafe_allow_html=True)
                
                # Copy button
                st.text_area("Copy transcript from here:", value=transcript, height=100, disabled=True)
                
            else:
                st.warning("⚠️ No transcript available. Check audio extraction.")
        
        st.markdown("---")
        
        # Modality Breakdown
        st.markdown("### 📊 Modality Breakdown")
        cols = st.columns(3)
        
        modality_data = [
            ("🖼️", "Visual", visual_result, w_visual if visual_result else 0),
            ("🔊", "Audio", audio_result, w_audio if audio_result else 0),
            ("🔤", "Text", text_result, w_text if text_result else 0),
        ]
        
        for col, (emoji, name, res, weight) in zip(cols, modality_data):
            with col:
                if res:
                    em = res["emotion"]
                    conf = res["confidence"] * 100
                    st.markdown(f"""
                    <div class="modality-card">
                        <div class="modality-emoji">{emoji}</div>
                        <div style="font-size:0.9rem;color:#8892a4;margin-bottom:8px">{name}</div>
                        <div class="modality-emotion">{em.capitalize()}</div>
                        <div class="modality-confidence">{conf:.1f}% conf</div>
                        <div style="font-size:0.8rem;color:#555;margin-top:8px;">Weight: {weight:.0%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="modality-card" style="opacity:0.5;">
                        <div class="modality-emoji">{emoji}</div>
                        <div style="font-size:0.9rem;color:#8892a4;">Not available</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Fusion Weights Used
        st.markdown("### ⚙️ Fusion Configuration")
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            st.metric("Visual Weight", f"{w_visual:.0%}")
        with col_w2:
            st.metric("Audio Weight", f"{w_audio:.0%}")
        with col_w3:
            st.metric("Text Weight", f"{w_text:.0%}")
        
        # Overall Fused Scores
        st.markdown("### 🕸️ Fused Emotion Scores")
        col_chart1, col_chart2 = st.columns([2, 1])
        
        with col_chart1:
            sorted_scores = dict(sorted(result["all_scores"].items(), key=lambda x: -x[1]))
            st.plotly_chart(confidence_bar_chart(sorted_scores, title="Fused Confidence Scores"),
                           use_container_width=True)
        
        with col_chart2:
            st.plotly_chart(radar_chart(result["all_scores"]), use_container_width=True)
        
        # Frame Timeline (if available)
        if visual_result.get("frame_results"):
            st.markdown("### 📈 Frame-Level Emotion Timeline")
            fig_tl = frame_emotion_timeline(visual_result.get("frame_results", []))
            if fig_tl:
                st.plotly_chart(fig_tl, use_container_width=True)

else:
    st.markdown("""
    <div style="
        background:#161b27; border:1px dashed #2a3140;
        border-radius:14px; padding:40px; text-align:center; color:#8892a4;
    ">
        <div style="font-size:3rem;">🎭</div><br>
        Upload an image, video, or take a webcam snapshot to start multimodal emotion analysis.
    </div>
    """, unsafe_allow_html=True)
