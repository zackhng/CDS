"""
Visualization helpers for the Emotion Detection Streamlit app.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from .emotion_utils import EMOTIONS, EMOTION_COLORS, EMOTION_EMOJIS


# ---------------------------------------------------------------------------
# Confidence Bar Chart
# ---------------------------------------------------------------------------
def confidence_bar_chart(scores: dict, title: str = "Emotion Confidence", highlight: str = None):
    """Horizontal bar chart of emotion confidence scores."""
    labels = list(scores.keys())
    values = [scores[e] * 100 for e in labels]
    colors = [
        EMOTION_COLORS.get(e, "#888") if e != highlight else "#fff"
        for e in labels
    ]
    border_colors = [
        "#ffffff" if e == highlight else EMOTION_COLORS.get(e, "#888")
        for e in labels
    ]
    emoji_labels = [f"{EMOTION_EMOJIS.get(e, '')} {e.capitalize()}" for e in labels]

    fig = go.Figure(go.Bar(
        x=values,
        y=emoji_labels,
        orientation="h",
        marker=dict(
            color=[EMOTION_COLORS.get(e, "#888") for e in labels],
            line=dict(color='rgba(255, 255, 255, 0.13)', width=1),
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(range=[0, 105], showgrid=False, ticksuffix="%"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0", size=13),
        margin=dict(l=10, r=60, t=40, b=10),
        height=340,
    )
    return fig


# ---------------------------------------------------------------------------
# Radar / Spider Chart
# ---------------------------------------------------------------------------
def radar_chart(scores: dict, title: str = "Emotion Radar"):
    """Radar chart for multimodal comparison."""
    categories = [f"{EMOTION_EMOJIS.get(e,'')} {e.capitalize()}" for e in EMOTIONS]
    values = [scores.get(e, 0) * 100 for e in EMOTIONS]
    values += values[:1]  # close loop
    cats = categories + categories[:1]

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=cats,
        fill="toself",
        fillcolor="rgba(52,152,219,0.2)",
        line=dict(color="#3498db", width=2),
        marker=dict(size=6, color="#3498db"),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100],
                            ticksuffix="%", gridcolor="#333"),
            angularaxis=dict(gridcolor="#333"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        title=title,
        margin=dict(l=40, r=40, t=50, b=40),
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# Modality Comparison (grouped bar)
# ---------------------------------------------------------------------------
def modality_comparison_chart(text_scores: dict, audio_scores: dict, visual_scores: dict):
    """Side-by-side comparison of all three modalities."""
    fig = go.Figure()

    modalities = {
        "🔤 Text": (text_scores, "#e74c3c"),
        "🔊 Audio": (audio_scores, "#f39c12"),
        "🖼️ Visual": (visual_scores, "#2ecc71"),
    }

    for name, (scores, color) in modalities.items():
        if scores:
            fig.add_trace(go.Bar(
                name=name,
                x=[e.capitalize() for e in EMOTIONS],
                y=[scores.get(e, 0) * 100 for e in EMOTIONS],
                marker_color=color,
                opacity=0.85,
            ))

    fig.update_layout(
        barmode="group",
        title="Modality Comparison",
        xaxis_title="Emotion",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 105]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0", size=12),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="#444"),
        margin=dict(l=10, r=10, t=50, b=10),
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# Waveform Plot
# ---------------------------------------------------------------------------
def waveform_plot(y: np.ndarray, sr: int, title: str = "Audio Waveform"):
    """Plot audio waveform using plotly."""
    duration = len(y) / sr
    time_axis = np.linspace(0, duration, num=min(len(y), 4000))
    # Downsample for performance
    step = max(1, len(y) // 4000)
    y_down = y[::step][:4000]

    fig = go.Figure(go.Scatter(
        x=time_axis[:len(y_down)],
        y=y_down,
        line=dict(color="#3498db", width=0.8),
        fill="tozeroy",
        fillcolor="rgba(52,152,219,0.1)",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=200,
    )
    return fig


# ---------------------------------------------------------------------------
# MFCC Heatmap
# ---------------------------------------------------------------------------
def mfcc_heatmap(mfcc_mean: list, title: str = "MFCC Feature Vector"):
    """Display MFCC coefficients as a horizontal heatmap."""
    z = np.array([mfcc_mean])
    fig = go.Figure(go.Heatmap(
        z=z,
        colorscale="RdBu",
        showscale=True,
        colorbar=dict(len=0.5),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="MFCC Coefficient",
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        margin=dict(l=10, r=10, t=40, b=30),
        height=160,
    )
    return fig


# ---------------------------------------------------------------------------
# Frame-level timeline
# ---------------------------------------------------------------------------
def frame_emotion_timeline(frame_results: list):
    """Line chart of per-frame emotion over time."""
    if not frame_results:
        return None

    frames = [r["frame"] for r in frame_results]
    confidences = [r["confidence"] * 100 for r in frame_results]
    emotions = [r["emotion"] for r in frame_results]
    colors = [EMOTION_COLORS.get(e, "#888") for e in emotions]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frames,
        y=confidences,
        mode="lines+markers",
        line=dict(color="#3498db", width=2),
        marker=dict(color=colors, size=10, line=dict(color="#fff", width=1)),
        hovertemplate="Frame %{x}<br>Emotion: %{text}<br>Confidence: %{y:.1f}%<extra></extra>",
        text=emotions,
    ))

    fig.update_layout(
        title="Frame-Level Emotion Timeline",
        xaxis_title="Frame Index",
        yaxis_title="Confidence (%)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=260,
    )
    return fig


# ---------------------------------------------------------------------------
# Emotion result card
# ---------------------------------------------------------------------------
def emotion_result_card(emotion: str, confidence: float, modality: str = ""):
    """Display a styled emotion result card using Streamlit markdown."""
    color = EMOTION_COLORS.get(emotion, "#888")
    emoji = EMOTION_EMOJIS.get(emotion, "🎭")
    pct = confidence * 100

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}22, {color}44);
        border: 2px solid {color};
        border-radius: 16px;
        padding: 24px 28px;
        text-align: center;
        margin: 12px 0;
    ">
        <div style="font-size: 3.2rem; margin-bottom: 6px;">{emoji}</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: {color};
                    text-transform: capitalize; letter-spacing: 0.05em;">
            {emotion}
        </div>
        {"<div style='font-size:0.85rem; color:#aaa; margin-top:4px;'>" + modality + "</div>" if modality else ""}
        <div style="
            margin-top: 14px;
            background: rgba(0,0,0,0.25);
            border-radius: 8px;
            overflow: hidden;
            height: 10px;
        ">
            <div style="
                width: {pct:.1f}%;
                height: 100%;
                background: {color};
                border-radius: 8px;
                transition: width 0.6s ease;
            "></div>
        </div>
        <div style="font-size: 1rem; color: #ccc; margin-top: 6px;">
            {pct:.1f}% confidence
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Fusion result summary
# ---------------------------------------------------------------------------
def fusion_summary_cards(modality_summary: dict):
    """Show small emotion badges for each modality."""
    cols = st.columns(len(modality_summary))
    icons = {"text": "🔤", "audio": "🔊", "visual": "🖼️"}
    for col, (mod, info) in zip(cols, modality_summary.items()):
        with col:
            color = EMOTION_COLORS.get(info["emotion"], "#888")
            emoji = EMOTION_EMOJIS.get(info["emotion"], "🎭")
            icon = icons.get(mod, "🔵")
            st.markdown(f"""
            <div style="
                background: {color}22;
                border: 1px solid {color};
                border-radius: 12px;
                padding: 14px;
                text-align: center;
            ">
                <div style="font-size:1.4rem;">{icon}</div>
                <div style="font-size:0.8rem; color:#aaa; margin:4px 0;">
                    {mod.capitalize()}
                </div>
                <div style="font-size:1.05rem; color:{color}; font-weight:600;">
                    {emoji} {info['emotion'].capitalize()}
                </div>
                <div style="font-size:0.8rem; color:#888;">
                    {info['confidence']*100:.0f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
