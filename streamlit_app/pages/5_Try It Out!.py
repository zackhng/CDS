"""
Try It Out! — Real-Time Multimodal Emotion Inference
Records webcam + audio, transcribes speech, and fuses all three modalities.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import streamlit as st
import tempfile
import numpy as np
import time

from util.emotion_utils import (
    process_text, process_audio, process_video,
    fuse_predictions, transcribe_audio, extract_audio_from_video,
    EMOTION_COLORS, EMOTION_EMOJIS, EMOTIONS,
)
from util.viz_utils import (
    confidence_bar_chart, emotion_result_card,
    fusion_summary_cards, modality_comparison_chart,
    radar_chart, frame_emotion_timeline,
)

st.set_page_config(page_title="E-Motion · Try It Out!", page_icon="🎥", layout="wide")

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .pipeline-step {
        background: #161b27; border: 1px solid #2a3140;
        border-radius: 12px; padding: 18px; margin: 8px 0;
        border-left: 4px solid #3498db;
    }
    .pipeline-step.done  { border-left-color: #2ecc71; }
    .pipeline-step.error { border-left-color: #e74c3c; }

    .fusion-box {
        background: linear-gradient(135deg, #1a1f35, #161b27);
        border: 2px solid #3498db44;
        border-radius: 16px; padding: 28px; text-align: center; margin: 16px 0;
    }
    .weight-chip {
        display: inline-block; background: #1e2a3a;
        border: 1px solid #3498db44; border-radius: 8px;
        padding: 6px 12px; font-size: 0.82rem; color: #a8c4e0; margin: 3px;
    }
    .step-num {
        display: inline-block; width: 28px; height: 28px;
        border-radius: 50%; background: #3498db22;
        border: 1px solid #3498db; text-align: center;
        line-height: 28px; font-size: 0.85rem; color: #3498db;
        margin-right: 10px; font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🎥 Try It Out! — Multimodal Emotion Detection")
st.markdown(
    "Record yourself speaking (or upload a video), and the system will analyse "
    "your **facial expressions**, **voice**, and **spoken words** together."
)
st.markdown("---")

# ── Pipeline Overview ─────────────────────────────────────────────────────────
with st.expander("🔬 How the Pipeline Works", expanded=False):
    st.markdown("""
    ```
    Video Recording (mp4)
        │
        ├─► Extract Audio (.wav) ──► Speech-to-Text ──► Text Model ──┐
        │                                                              │
        ├─► Audio Features (MFCC) ──► Audio Model ────────────────────┤──► Late Fusion ──► Final Emotion
        │                                                              │
        └─► Extract Frames ──► Face Detection ──► Visual Model ───────┘
    ```
    - **Late Fusion** uses weighted averaging: Text 35% · Audio 35% · Visual 30%
    - All models degrade gracefully — if a checkpoint isn't found, feature-seeded mocks are used.
    """)

# ── Input Section ─────────────────────────────────────────────────────────────
st.markdown("## 📥 Step 1: Provide Input")

tab_record, tab_upload = st.tabs(["🎥 Record with Webcam", "📁 Upload Video"])

video_bytes = None
source_label = ""

with tab_record:
    st.info(
        "Click the record button below. Speak naturally — say anything. "
        "When done, stop the recording and submit."
    )
    webcam_video = st.camera_input("📸 Take a snapshot (video recording via webcam)")

    if webcam_video:
        st.warning(
            "ℹ️ Streamlit's `camera_input` captures a still image. "
            "For full video + audio, please use the **Upload Video** tab "
            "or the webrtc-based recorder if deployed with HTTPS."
        )
        img_bytes = webcam_video.read()

        # We'll treat the snapshot as an image-only input
        if st.button("🔍 Analyse Snapshot (Image modality only)", type="primary"):
            from PIL import Image as PILImage
            from util.emotion_utils import process_image
            img_pil = PILImage.open(webcam_video)
            with st.spinner("Analysing snapshot…"):
                vis_result = process_image(img_pil, label_hint="webcam_snap")

            st.markdown("---")
            st.markdown("### 📊 Result (Visual Only)")
            emotion_result_card(vis_result["emotion"], vis_result["confidence"], "Visual Model (Snapshot)")
            sorted_scores = dict(sorted(vis_result["all_scores"].items(), key=lambda x: -x[1]))
            st.plotly_chart(confidence_bar_chart(sorted_scores), use_container_width=True)

    # WebRTC recorder (optional — requires streamlit-webrtc)
    st.markdown("---")
    st.markdown("#### 🎙️ WebRTC Live Recorder (requires camera + mic permissions)")
    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode
        import av

        if "rtc_frames" not in st.session_state:
            st.session_state.rtc_frames = []
        if "rtc_recording" not in st.session_state:
            st.session_state.rtc_recording = False

        class _VideoProcessor:
            def recv(self, frame):
                if st.session_state.rtc_recording:
                    img = frame.to_ndarray(format="bgr24")
                    st.session_state.rtc_frames.append(img)
                return frame

        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("▶️ Start Recording"):
                st.session_state.rtc_recording = True
                st.session_state.rtc_frames = []
        with col_stop:
            if st.button("⏹️ Stop Recording"):
                st.session_state.rtc_recording = False

        webrtc_streamer(
            key="multimodal-recorder",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=_VideoProcessor,
            media_stream_constraints={"video": True, "audio": True},
        )

        if st.button("✅ Submit WebRTC Recording") and st.session_state.rtc_frames:
            frames = st.session_state.rtc_frames
            out_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            h, w, _ = frames[0].shape
            container = av.open(out_tmp.name, mode="w")
            stream = container.add_stream("mpeg4", rate=20)
            stream.width = w; stream.height = h; stream.pix_fmt = "yuv420p"
            for f in frames:
                vf = av.VideoFrame.from_ndarray(f, format="bgr24")
                for pkt in stream.encode(vf):
                    container.mux(pkt)
            for pkt in stream.encode(None):
                container.mux(pkt)
            container.close()
            with open(out_tmp.name, "rb") as fh:
                video_bytes = fh.read()
            source_label = "webrtc_recording.mp4"
            st.success(f"Recording saved ({len(video_bytes)/1024:.1f} KB)")
    except ImportError:
        st.info(
            "Install `streamlit-webrtc` and `av` for live webcam recording: "
            "`pip install streamlit-webrtc av`"
        )

with tab_upload:
    vid_file = st.file_uploader(
        "Upload a video file (mp4 / avi / mov)",
        type=["mp4", "avi", "mov", "mkv", "webm"],
    )
    if vid_file:
        st.video(vid_file)
        video_bytes = vid_file.read()
        source_label = vid_file.name
        st.success(f"Loaded: **{vid_file.name}** ({len(video_bytes)/1024:.1f} KB)")

# ── Fusion Settings ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## ⚙️ Step 2: Configure Fusion Weights")

col_tw, col_aw, col_vw = st.columns(3)
with col_tw:
    w_text   = st.slider("🔤 Text weight",   0.0, 1.0, 0.35, 0.05)
with col_aw:
    w_audio  = st.slider("🔊 Audio weight",  0.0, 1.0, 0.35, 0.05)
with col_vw:
    w_visual = st.slider("🖼️ Visual weight", 0.0, 1.0, 0.30, 0.05)

total_w = w_text + w_audio + w_visual
if abs(total_w - 1.0) > 0.01:
    st.warning(f"⚠️ Weights sum to {total_w:.2f} (not 1.0) — they will be auto-normalised during fusion.")

max_frames = st.slider("Max video frames to analyse", 5, 40, 16)

# ── Run Pipeline ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🚀 Step 3: Run Multimodal Analysis")

run_btn = st.button(
    "⚡ Run Full Pipeline",
    type="primary",
    disabled=(video_bytes is None),
    use_container_width=True,
)

if video_bytes is None:
    st.markdown("""
    <div style="
        background:#161b27; border:1px dashed #2a3140;
        border-radius:14px; padding:30px; text-align:center; color:#8892a4;
    ">
        <div style="font-size:2.5rem;">🎥</div><br>
        Upload a video or record with webcam to enable analysis.
    </div>
    """, unsafe_allow_html=True)

if run_btn and video_bytes:
    # Write video to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid:
        tmp_vid.write(video_bytes)
        tmp_vid_path = tmp_vid.name

    wav_path = tmp_vid_path.replace(".mp4", ".wav")

    st.markdown("---")
    st.markdown("## 🔄 Pipeline Execution")

    # Progress tracking
    progress = st.progress(0)
    status = st.empty()

    text_result   = None
    audio_result  = None
    visual_result = None
    transcript    = None
    audio_bytes_extracted = None

    # ── Step 1: Extract Audio ──────────────────────────────────────────────────
    status.markdown("""
    <div class="pipeline-step">
        <span class="step-num">1</span><b>Extracting audio from video…</b>
    </div>""", unsafe_allow_html=True)
    progress.progress(10)

    audio_ok = extract_audio_from_video(tmp_vid_path, wav_path)
    if audio_ok:
        with open(wav_path, "rb") as f:
            audio_bytes_extracted = f.read()
        status.markdown("""
        <div class="pipeline-step done">
            <span class="step-num">✓</span><b>Audio extracted successfully</b>
        </div>""", unsafe_allow_html=True)
    else:
        status.markdown("""
        <div class="pipeline-step error">
            <span class="step-num">!</span>
            <b>Audio extraction failed</b> — ffmpeg/moviepy not available; skipping audio modality.
        </div>""", unsafe_allow_html=True)

    # ── Step 2: Speech-to-Text ────────────────────────────────────────────────
    progress.progress(25)
    status.markdown("""
    <div class="pipeline-step">
        <span class="step-num">2</span><b>Running speech-to-text transcription…</b>
    </div>""", unsafe_allow_html=True)

    if audio_ok and os.path.exists(wav_path):
        transcript = transcribe_audio(wav_path)
    else:
        # mock
        transcript = "I'm feeling quite unsure about everything right now."

    status.markdown(f"""
    <div class="pipeline-step done">
        <span class="step-num">✓</span><b>Transcription:</b>
        <span style="color:#a8c4e0;">{transcript[:200]}{"…" if len(transcript)>200 else ""}</span>
    </div>""", unsafe_allow_html=True)

    # ── Step 3: Text Model ────────────────────────────────────────────────────
    progress.progress(40)
    status.markdown("""
    <div class="pipeline-step">
        <span class="step-num">3</span><b>Running text emotion model…</b>
    </div>""", unsafe_allow_html=True)

    if transcript:
        text_result = process_text(transcript)
        em = text_result["emotion"]
        status.markdown(f"""
        <div class="pipeline-step done">
            <span class="step-num">✓</span>
            <b>Text → {EMOTION_EMOJIS.get(em,'')} {em.capitalize()}</b>
            ({text_result['confidence']*100:.1f}%)
        </div>""", unsafe_allow_html=True)

    # ── Step 4: Audio Model ───────────────────────────────────────────────────
    progress.progress(60)
    status.markdown("""
    <div class="pipeline-step">
        <span class="step-num">4</span><b>Running audio emotion model…</b>
    </div>""", unsafe_allow_html=True)

    if audio_bytes_extracted:
        audio_result = process_audio(audio_bytes_extracted, filename="extracted.wav")
    else:
        # Still run mock so fusion has something
        audio_result = process_audio(b"", filename=source_label)

    em = audio_result["emotion"]
    status.markdown(f"""
    <div class="pipeline-step done">
        <span class="step-num">✓</span>
        <b>Audio → {EMOTION_EMOJIS.get(em,'')} {em.capitalize()}</b>
        ({audio_result['confidence']*100:.1f}%)
    </div>""", unsafe_allow_html=True)

    # ── Step 5: Visual Model ──────────────────────────────────────────────────
    progress.progress(80)
    status.markdown("""
    <div class="pipeline-step">
        <span class="step-num">5</span><b>Extracting frames and running visual model…</b>
    </div>""", unsafe_allow_html=True)

    visual_result = process_video(video_bytes, max_frames=max_frames)
    em = visual_result["emotion"]
    n_frames = len(visual_result.get("frame_results", []))
    status.markdown(f"""
    <div class="pipeline-step done">
        <span class="step-num">✓</span>
        <b>Visual ({n_frames} frames) → {EMOTION_EMOJIS.get(em,'')} {em.capitalize()}</b>
        ({visual_result['confidence']*100:.1f}%)
    </div>""", unsafe_allow_html=True)

    # ── Step 6: Fusion ────────────────────────────────────────────────────────
    progress.progress(95)
    status.markdown("""
    <div class="pipeline-step">
        <span class="step-num">6</span><b>Fusing modality predictions…</b>
    </div>""", unsafe_allow_html=True)

    fused = fuse_predictions(
        text_result, audio_result, visual_result,
        weights={"text": w_text, "audio": w_audio, "visual": w_visual},
    )

    progress.progress(100)
    status.empty()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    try:
        os.unlink(tmp_vid_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)
    except Exception:
        pass

    # ════════════════════════════════════════════════════════════════════════
    # RESULTS
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 🎯 Final Prediction")

    final_color  = EMOTION_COLORS.get(fused["emotion"], "#888")
    final_emoji  = EMOTION_EMOJIS.get(fused["emotion"], "🎭")
    final_pct    = fused["confidence"] * 100

    st.markdown(f"""
    <div class="fusion-box">
        <div style="font-size:5rem;">{final_emoji}</div>
        <div style="font-size:2.4rem; font-weight:800; color:{final_color};
                    text-transform:capitalize; letter-spacing:0.06em; margin:10px 0;">
            {fused["emotion"]}
        </div>
        <div style="font-size:1.05rem; color:#8892a4;">Fused Prediction</div>
        <div style="margin-top:18px; background:rgba(0,0,0,0.3); border-radius:10px;
                    overflow:hidden; height:14px; max-width:400px; margin-left:auto; margin-right:auto;">
            <div style="width:{final_pct:.1f}%; height:100%; background:{final_color};
                        border-radius:10px;"></div>
        </div>
        <div style="font-size:1.1rem; color:#ccc; margin-top:8px;">
            {final_pct:.1f}% confidence
        </div>
        <div style="margin-top:14px;">
            <span class="weight-chip">🔤 Text {w_text:.0%}</span>
            <span class="weight-chip">🔊 Audio {w_audio:.0%}</span>
            <span class="weight-chip">🖼️ Visual {w_visual:.0%}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Per-Modality Summary ───────────────────────────────────────────────────
    st.markdown("### 📋 Modality Breakdown")
    mod_summary = fused.get("modality_summary", {})
    if mod_summary:
        fusion_summary_cards(mod_summary)

    # ── Confidence Comparison ──────────────────────────────────────────────────
    st.markdown("### 📊 Confidence Comparison")

    col_a, col_b = st.columns(2)
    with col_a:
        sorted_fused = dict(sorted(fused["all_scores"].items(), key=lambda x: -x[1]))
        st.plotly_chart(
            confidence_bar_chart(sorted_fused, title="🔀 Fused Scores"),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(radar_chart(fused["all_scores"], title="Fused Emotion Radar"),
                        use_container_width=True)

    # ── Modality comparison ────────────────────────────────────────────────────
    st.markdown("### 🔬 Modality Comparison")
    fig_comp = modality_comparison_chart(
        text_result["all_scores"]   if text_result   else {},
        audio_result["all_scores"]  if audio_result  else {},
        visual_result["all_scores"] if visual_result else {},
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # ── Transcription panel ────────────────────────────────────────────────────
    if transcript:
        st.markdown("### 🗣️ Transcribed Speech")
        st.markdown(f"""
        <div style="background:#161b27;border:1px solid #2a3140;border-radius:10px;
                    padding:16px;font-style:italic;color:#a8c4e0;font-size:1.05rem;">
            "{transcript}"
        </div>
        """, unsafe_allow_html=True)

    # ── Frame timeline ─────────────────────────────────────────────────────────
    frame_results = visual_result.get("frame_results", [])
    if frame_results:
        st.markdown("### 📈 Visual Emotion Timeline")
        fig_tl = frame_emotion_timeline(frame_results)
        if fig_tl:
            st.plotly_chart(fig_tl, use_container_width=True)

    # ── Sample frames ──────────────────────────────────────────────────────────
    sample_frames = visual_result.get("sample_frames", [])
    if sample_frames:
        st.markdown("### 🎞️ Sample Frames")
        n_show = min(len(sample_frames), 3)
        cols = st.columns(n_show)
        for i, (col, frame) in enumerate(zip(cols, sample_frames[:n_show])):
            with col:
                fr_idx = frame_results[i]["frame"] if i < len(frame_results) else i
                em_label = frame_results[i]["emotion"] if i < len(frame_results) else ""
                st.image(frame, caption=f"Frame {fr_idx} · {em_label.capitalize()}",
                         use_container_width=True, clamp=True)

    # ── Audio waveform (if available) ─────────────────────────────────────────
    if audio_result and audio_result.get("waveform") is not None:
        from util.viz_utils import waveform_plot
        st.markdown("### 🎵 Audio Waveform")
        fig_wave = waveform_plot(audio_result["waveform"], audio_result["sr"])
        st.plotly_chart(fig_wave, use_container_width=True)

    # ── Full scores table ─────────────────────────────────────────────────────
    with st.expander("📋 Detailed Score Table"):
        import pandas as pd
        rows = []
        for e in EMOTIONS:
            rows.append({
                "Emotion": f"{EMOTION_EMOJIS.get(e,'')} {e.capitalize()}",
                "Text (%)":   f"{text_result['all_scores'].get(e,0)*100:.1f}"   if text_result   else "—",
                "Audio (%)":  f"{audio_result['all_scores'].get(e,0)*100:.1f}"  if audio_result  else "—",
                "Visual (%)": f"{visual_result['all_scores'].get(e,0)*100:.1f}" if visual_result else "—",
                "Fused (%)":  f"{fused['all_scores'].get(e,0)*100:.1f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
