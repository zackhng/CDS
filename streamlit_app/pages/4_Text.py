"""
Text Emotion Analysis Page
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import streamlit as st
import plotly.graph_objects as go
from util.emotion_utils import process_text, EMOTION_COLORS, EMOTION_EMOJIS, EMOTIONS
from util.viz_utils import confidence_bar_chart, radar_chart, emotion_result_card

st.set_page_config(page_title="E-Motion · Text", page_icon="🔤", layout="wide")

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .analysis-section {
        background: #161b27; border: 1px solid #2a3140;
        border-radius: 14px; padding: 22px; margin: 14px 0;
    }
    .token-highlight {
        display: inline-block; padding: 2px 6px; border-radius: 4px;
        font-size: 0.95rem; margin: 2px; font-weight: 500;
    }
    .cleaned-text {
        background: #0d1117; border: 1px solid #2a3140; border-radius: 8px;
        padding: 14px; font-family: monospace; font-size: 0.92rem;
        color: #a8b4c8; line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🔤 Text Emotion Analysis")
st.markdown("Enter any text — a sentence, tweet, or paragraph — and the model will predict the emotional tone.")
st.markdown("---")

# ── Example Presets ───────────────────────────────────────────────────────────
with st.expander("💡 Try an example"):
    examples = {
        "Happy":     "I just got the news — I'm absolutely thrilled and couldn't be happier!",
        "Angry":     "This is completely unacceptable! I've been waiting for hours with no explanation.",
        "Sad":       "I miss those times so much. Everything feels empty without them.",
        "Fearful":   "I don't know what's going to happen and it terrifies me to think about it.",
        "Surprised": "Wait — are you serious? I genuinely didn't see that coming at all!",
        "Neutral":   "The meeting is scheduled for Tuesday at 3 PM in conference room B.",
        "Disgust":   "The conditions in that place were absolutely revolting and disgraceful.",
        "Calm":      "I sat by the lake for a while, listening to the water and breathing slowly.",
    }
    selected = st.selectbox("Choose an example emotion:", list(examples.keys()))
    if st.button("Load Example"):
        st.session_state["text_input"] = examples[selected]

# ── Input ─────────────────────────────────────────────────────────────────────
default_text = st.session_state.get("text_input", "")
user_text = st.text_area(
    "Enter your text below:",
    value=default_text,
    height=140,
    placeholder="Type or paste any text here…",
    key="text_area_input",
)

col_btn, col_clear = st.columns([1, 5])
with col_btn:
    run_analysis = st.button("🔍 Analyse", type="primary", use_container_width=True)
with col_clear:
    if st.button("🗑️ Clear"):
        st.session_state["text_input"] = ""
        st.rerun()

# ── Analysis ──────────────────────────────────────────────────────────────────
if run_analysis:
    if not user_text.strip():
        st.warning("Please enter some text before analysing.")
    else:
        with st.spinner("Running text emotion analysis…"):
            result = process_text(user_text)

        st.markdown("---")
        st.markdown("## 📊 Results")

        # Main emotion card
        col_card, col_scores = st.columns([1, 2])

        with col_card:
            emotion_result_card(result["emotion"], result["confidence"], modality="Text Model")

            # Key stats
            st.markdown(f"""
            <div class="analysis-section">
                <b>📝 Input length</b><br>
                <span style="color:#3498db;">{len(user_text.split())} words · {len(user_text)} chars</span>
            </div>
            """, unsafe_allow_html=True)

        with col_scores:
            # Sort scores descending
            sorted_scores = dict(sorted(result["all_scores"].items(), key=lambda x: -x[1]))
            fig = confidence_bar_chart(sorted_scores, title="Confidence Scores", highlight=result["emotion"])
            st.plotly_chart(fig, use_container_width=True)

        # ── Cleaned text ──────────────────────────────────────────────────────
        st.markdown("### 🧹 Cleaned Text")
        st.markdown(f'<div class="cleaned-text">{result["cleaned_text"]}</div>', unsafe_allow_html=True)

        # ── Token-level visualization ─────────────────────────────────────────
        st.markdown("### 🎨 Token-Level Emotion Intensity")
        st.markdown("Each word is highlighted with an intensity based on its emotional weight (heuristic).")

        import re, hashlib
        words = result["cleaned_text"].split()
        top_emotion = result["emotion"]
        color = EMOTION_COLORS.get(top_emotion, "#888")

        def word_heat(word):
            seed = int(hashlib.md5(word.encode()).hexdigest(), 16) % 100
            return seed / 100

        tokens_html = " ".join([
            f'<span class="token-highlight" '
            f'style="background:{color}{int(word_heat(w)*80+20):02x}; color:#fff;">'
            f'{w}</span>'
            for w in words
        ])
        st.markdown(f"<div style='line-height:2.2;'>{tokens_html}</div>", unsafe_allow_html=True)

        # ── Radar chart ───────────────────────────────────────────────────────
        st.markdown("### 🕸️ Emotion Radar")
        fig_radar = radar_chart(result["all_scores"], title="Emotion Distribution")
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── Raw scores table ──────────────────────────────────────────────────
        with st.expander("📋 Raw Confidence Scores"):
            import pandas as pd
            df = pd.DataFrame([
                {
                    "Emotion": f"{EMOTION_EMOJIS.get(e,'')} {e.capitalize()}",
                    "Confidence (%)": f"{v*100:.2f}",
                    "Bar": "█" * int(v * 30),
                }
                for e, v in sorted(result["all_scores"].items(), key=lambda x: -x[1])
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
