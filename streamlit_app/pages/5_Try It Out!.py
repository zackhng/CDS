import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import tempfile
import numpy as np

st.set_page_config(page_title="Video Recorder")

st.title("🎥 Record Yourself")

# -----------------------------
# Session state
# -----------------------------
if "recording" not in st.session_state:
    st.session_state.recording = False

if "frames" not in st.session_state:
    st.session_state.frames = []

# -----------------------------
# Video Processor
# -----------------------------
class VideoProcessor:
    def recv(self, frame):
        if st.session_state.recording:
            img = frame.to_ndarray(format="bgr24")
            st.session_state.frames.append(img)
        return frame

# -----------------------------
# Controls
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Start Recording"):
        st.session_state.recording = True
        st.session_state.frames = []

with col2:
    if st.button("⏹️ Stop Recording"):
        st.session_state.recording = False

# -----------------------------
# Webcam Stream (Video + Audio)
# -----------------------------
webrtc_ctx = webrtc_streamer(
    key="recorder",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": True,  # ✅ captures mic
    },
)

st.info("Allow camera + microphone access. Click START to record.")

# -----------------------------
# Submit (Save Video)
# -----------------------------
if st.button("✅ Submit & Save"):
    if len(st.session_state.frames) == 0:
        st.warning("No recording found.")
    else:
        output_path = tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ).name

        height, width, _ = st.session_state.frames[0].shape

        container = av.open(output_path, mode="w")
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        for frame in st.session_state.frames:
            video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            packet = stream.encode(video_frame)
            if packet:
                container.mux(packet)

        # flush
        packet = stream.encode(None)
        if packet:
            container.mux(packet)

        container.close()

        st.success(f"Saved to {output_path}")

        # Optional preview
        with open(output_path, "rb") as f:
            st.video(f.read())