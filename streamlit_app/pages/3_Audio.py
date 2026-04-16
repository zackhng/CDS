"""
Audio Emotion Analysis Page
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import streamlit as st
import numpy as np
from util.emotion_utils import process_audio, EMOTION_EMOJIS, EMOTION_COLORS
from util.viz_utils import (
    confidence_bar_chart, waveform_plot, mfcc_heatmap,
    emotion_result_card, radar_chart,
)

st.set_page_config(page_title="E-Motion · Audio", page_icon="🔊", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .info-card {
        background: #161b27; border: 1px solid #2a3140;
        border-radius: 12px; padding: 18px; margin: 10px 0;
    }
    .feature-row { display: flex; gap: 12px; flex-wrap: wrap; }
    .feat-chip {
        background: #1e2a3a; border: 1px solid #3498db44;
        border-radius: 8px; padding: 8px 14px;
        font-size: 0.85rem; color: #a8c4e0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🔊 Audio Emotion Analysis")
st.markdown("Upload a `.wav` audio file or record directly from your microphone.")
st.markdown("---")

# ── Input tabs ────────────────────────────────────────────────────────────────
tab_upload, tab_record = st.tabs(["📁 Upload WAV File", "🎙️ Record Audio"])

audio_bytes = None
source_label = ""

with tab_upload:
    uploaded = st.file_uploader(
        "Choose a WAV audio file",
        type=["wav", "mp3", "ogg", "flac"],
        help="Recommended: short clips of 3–15 seconds work best.",
    )
    if uploaded:
        st.audio(uploaded, format="audio/wav")
        audio_bytes = uploaded.read()
        source_label = uploaded.name
        st.success(f"Loaded: **{uploaded.name}** ({len(audio_bytes)/1024:.1f} KB)")

with tab_record:
    st.info("Click the microphone icon below to record. When finished, the audio will appear for playback.")
    recorded = st.audio_input("Record audio here")
    if recorded:
        audio_bytes = recorded.read()
        source_label = "live_recording.wav"
        st.success(f"Recording captured ({len(audio_bytes)/1024:.1f} KB)")

# ── Analysis ──────────────────────────────────────────────────────────────────
st.markdown("---")
col_btn, _ = st.columns([1, 4])
with col_btn:
    analyse = st.button("🔍 Analyse Audio", type="primary",
                        disabled=(audio_bytes is None), use_container_width=True)

if analyse and audio_bytes:
    with st.spinner("Extracting audio features and running emotion model…"):
        result = process_audio(audio_bytes, filename=source_label)

    st.markdown("## 📊 Results")

    # Main layout
    col_card, col_chart = st.columns([1, 2])

    with col_card:
        emotion_result_card(result["emotion"], result["confidence"], modality="Audio Model")

        # Feature summary
        feats = result.get("features", {})
        if feats:
            rms_val = feats.get("rms", 0)
            zcr_val = feats.get("zcr", 0)
            sc_val  = feats.get("spectral_centroid", 0)
            st.markdown(f"""
            <div class="info-card">
                <b>🎵 Extracted Features</b><br><br>
                <div class="feature-row">
                    <div class="feat-chip">RMS Energy<br><b>{rms_val:.4f}</b></div>
                    <div class="feat-chip">Zero Cross Rate<br><b>{zcr_val:.4f}</b></div>
                    <div class="feat-chip">Spectral Centroid<br><b>{sc_val:.1f} Hz</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_chart:
        sorted_scores = dict(sorted(result["all_scores"].items(), key=lambda x: -x[1]))
        fig = confidence_bar_chart(sorted_scores, title="Confidence Scores")
        st.plotly_chart(fig, use_container_width=True)

    # ── Waveform ──────────────────────────────────────────────────────────────
    waveform = result.get("waveform")
    sr = result.get("sr")

    if waveform is not None and sr is not None:
        st.markdown("### 📈 Waveform")
        fig_wave = waveform_plot(waveform, sr)
        st.plotly_chart(fig_wave, use_container_width=True)

        # ── MFCC ──────────────────────────────────────────────────────────────
        mfcc_mean = result.get("features", {}).get("mfcc_mean")
        if mfcc_mean:
            st.markdown("### 🌈 MFCC Coefficients")
            st.caption(
                "Mel-Frequency Cepstral Coefficients capture the spectral "
                "envelope of the audio signal — key features for emotion recognition."
            )
            fig_mfcc = mfcc_heatmap(mfcc_mean, title="Mean MFCC (40 coefficients)")
            st.plotly_chart(fig_mfcc, use_container_width=True)

            # Bar chart of MFCC values
            import plotly.graph_objects as go
            coeff_idx = list(range(1, len(mfcc_mean) + 1))
            fig_bar = go.Figure(go.Bar(
                x=coeff_idx,
                y=mfcc_mean,
                marker=dict(
                    color=mfcc_mean,
                    colorscale="RdBu",
                    showscale=False,
                ),
            ))
            fig_bar.update_layout(
                title="MFCC Coefficient Values",
                xaxis_title="Coefficient Index",
                yaxis_title="Value",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0e0"),
                height=220,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Install **librosa** for waveform and MFCC visualizations: `pip install librosa`")

    # ── Radar ─────────────────────────────────────────────────────────────────
    st.markdown("### 🕸️ Emotion Radar")
    fig_radar = radar_chart(result["all_scores"])
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Explanation ───────────────────────────────────────────────────────────
    with st.expander("ℹ️ How Audio Emotion Detection Works"):
        st.markdown("""
        The audio pipeline works in three stages:

        1. **Feature Extraction** — `librosa` computes:
           - **MFCCs** (40 coefficients): compact representation of the vocal tract shape
           - **RMS Energy**: average loudness / emotional intensity
           - **Zero-Crossing Rate**: related to voiced/unvoiced segments
           - **Spectral Centroid**: brightness of the audio signal

        2. **Model Inference** — the feature vector is passed through a CNN/LSTM trained
           on RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song).

        3. **Emotion Output** — softmax probabilities across 8 emotions:
           *angry, calm, disgust, fearful, happy, neutral, sad, surprised*

        > **Note:** If a trained model checkpoint is not found, the app falls back to
        > feature-seeded mock predictions so you can still explore the full pipeline.
        """)

    # ── Raw scores ────────────────────────────────────────────────────────────
    with st.expander("📋 Raw Confidence Scores"):
        import pandas as pd
        df = pd.DataFrame([
            {
                "Emotion": f"{EMOTION_EMOJIS.get(e,'')}{e.capitalize()}",
                "Confidence (%)": f"{v*100:.2f}",
            }
            for e, v in sorted(result["all_scores"].items(), key=lambda x: -x[1])
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

elif audio_bytes is None:
    st.markdown("""
    <div style="
        background:#161b27; border:1px dashed #2a3140;
        border-radius:14px; padding:40px; text-align:center; color:#8892a4;
    ">
        <div style="font-size:3rem;">🔊</div>
        <br>Upload a WAV file or record audio to get started.
    </div>
    """, unsafe_allow_html=True)
