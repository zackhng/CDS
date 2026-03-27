import streamlit as st
import time

st.title("E-Motion")
# Your tagline as a single string
tag_line = "See, Hear, Read — Detect Emotion Intelligently."

# Placeholder for streaming
tag_placeholder = st.empty()

# Accumulate text character by character
accumulated_text = ""

for char in tag_line:
    accumulated_text += char
    tag_placeholder.markdown(accumulated_text)
    time.sleep(0.05)  # Adjust typing speed (seconds)

st.markdown("""
Welcome to the platform! This app allows you to explore different types of content:

- **Audio**: Record and analyze audio
- **Image**: Upload or generate images
- **Text**: Explore text-based features
- **Try it Out!**: Record yourself talking with video + audio

Use the sidebar on the left to navigate between sections.
""")

# st.image("https://images.unsplash.com/photo-1612832021204-49c142da4393?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=60", caption="Welcome to the App")

st.info("Click on the menu on the left to get started!")