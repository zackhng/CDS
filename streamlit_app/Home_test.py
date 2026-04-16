"""
E-Motion · Home Page
Multi-Modal Emotion Detection System
"""

import streamlit as st
import time
from src.util.emotion_utils import EMOTIONS, EMOTION_EMOJIS, EMOTION_COLORS

st.set_page_config(
    page_title="E-Motion · Multimodal Emotion Detection",
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
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3498db, #8e44ad, #e74c3c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
        margin-bottom: 0.25rem;
    }
    .hero-sub {
        font-size: 1.25rem;
        color: #8892a4;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #161b27;
        border: 1px solid #2a3140;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: border-color 0.2s;
    }
    .feature-card:hover { border-color: #3498db; }
    .feature-icon { font-size: 2.6rem; margin-bottom: 10px; }
    .feature-title { font-size: 1.1rem; font-weight: 600; color: #e0e6f0; margin-bottom: 6px; }
    .feature-desc { font-size: 0.88rem; color: #8892a4; line-height: 1.5; }

    .stat-box {
        background: #161b27;
        border: 1px solid #2a3140;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .stat-num { font-size: 2rem; font-weight: 700; color: #3498db; }
    .stat-label { font-size: 0.82rem; color: #8892a4; margin-top: 4px; }

    .emotion-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
    }
    div[data-testid="stSidebar"] { background: #161b27; }
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🎭 E-Motion</h1>', unsafe_allow_html=True)

tagline = "See, Hear, Read — Detect Emotion Intelligently."
placeholder = st.empty()
built = ""
for ch in tagline:
    built += ch
    placeholder.markdown(f'<p class="hero-sub">{built}▌</p>', unsafe_allow_html=True)
    time.sleep(0.03)
placeholder.markdown(f'<p class="hero-sub">{tagline}</p>', unsafe_allow_html=True)

st.markdown("---")

# ── Stats Row ─────────────────────────────────────────────────────────────────
s1, s2, s3, s4 = st.columns(4)
stats = [
    ("3", "Modalities", "🔤🔊🖼️"),
    ("8", "Emotion Classes", "🎭"),
    ("Late Fusion", "Strategy", "⚗️"),
    ("Real-Time", "Inference", "⚡"),
]
for col, (num, label, icon) in zip([s1, s2, s3, s4], stats):
    with col:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-num">{icon} {num}</div>
            <div class="stat-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Feature Cards ─────────────────────────────────────────────────────────────
st.markdown("### What You Can Do")
c1, c2, c3, c4 = st.columns(4)
features = [
    ("🔤", "Text Analysis", "Paste any text and get emotion predictions with token-level confidence breakdown.", "pages/4_Text"),
    ("🔊", "Audio Analysis", "Upload a WAV file or record live audio. Features MFCC extraction, waveform visualisation, and pitch analysis.", "pages/3_Audio"),
    ("🖼️", "Visual Analysis", "Upload an image or video. Detects faces, runs per-frame predictions, and shows a timeline.", "pages/2_Image"),
    ("🎥", "Try It Out!", "Record yourself with webcam + mic. All three modalities are fused into one final emotion prediction.", "pages/5_Try_It_Out!"),
]
for col, (icon, title, desc, _) in zip([c1, c2, c3, c4], features):
    with col:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Emotion Set ───────────────────────────────────────────────────────────────
st.markdown("### Detectable Emotions")
pills_html = "".join([
    f'<span class="emotion-pill" style="background:{EMOTION_COLORS[e]}33; '
    f'color:{EMOTION_COLORS[e]}; border:1px solid {EMOTION_COLORS[e]}55;">'
    f'{EMOTION_EMOJIS[e]} {e.capitalize()}</span>'
    for e in EMOTIONS
])
st.markdown(f"<div>{pills_html}</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Architecture ──────────────────────────────────────────────────────────────
st.markdown("### System Architecture")
st.markdown("""
```
Input ──┬── Text  ──→ [Clean → Transformer]  ──→ Emotion Scores ─┐
        │                                                          │
        ├── Audio ──→ [MFCC / Librosa → CNN/LSTM] ──→ Scores ─────┤──→ Late Fusion ──→ Final Emotion
        │                                                          │
        └── Video ──→ [Frames → Face Detection → ResNet] → Scores ─┘
                 └── [Audio track → STT → Text pipeline]
```
""")

# ── Quick Start ───────────────────────────────────────────────────────────────
with st.expander("📖 Quick Start Guide"):
    st.markdown("""
    1. **Text page** — paste a sentence or paragraph and click *Analyse*.
    2. **Audio page** — upload a `.wav` file and click *Analyse Audio*.
    3. **Image page** — upload an image (`.jpg`/`.png`) or video (`.mp4`) and click *Analyse*.
    4. **Try It Out!** — grant camera & microphone access, record yourself, submit for multimodal fusion.

    > **Tip:** All models degrade gracefully — if a pretrained checkpoint is unavailable, the app
    > uses feature-seeded mock predictions so the full pipeline is always exercisable.
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555;font-size:0.8rem;'>"
    "Built with Streamlit · PyTorch · Librosa · OpenCV · Transformers"
    "</div>",
    unsafe_allow_html=True,
)
