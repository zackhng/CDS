"""
Multimodal Emotion Detection Pipeline
High-level orchestrator used by the Try It Out! page.
"""

import os
import tempfile
from typing import Optional
from util.emotion_utils import (
    process_text,
    process_audio,
    process_video,
    fuse_predictions,
    transcribe_audio,
    extract_audio_from_video,
)


class MultimodalPipeline:
    """
    End-to-end pipeline: video bytes → fused emotion prediction.

    Usage:
        pipeline = MultimodalPipeline()
        result = pipeline.run(video_bytes=..., weights={...})
    """

    def __init__(
        self,
        max_video_frames: int = 20,
        weights: Optional[dict] = None,
    ):
        self.max_video_frames = max_video_frames
        self.weights = weights or {"text": 0.35, "audio": 0.35, "visual": 0.30}

    def run(
        self,
        video_bytes: bytes,
        source_label: str = "input.mp4",
        on_step: Optional[callable] = None,
    ) -> dict:
        """
        Execute the full pipeline.

        Args:
            video_bytes: Raw bytes of the input video file.
            source_label: Display name for logging.
            on_step: Optional callback(step_name, message) for progress updates.

        Returns:
            dict with keys:
                - fused: final fusion result
                - text_result, audio_result, visual_result: per-modality results
                - transcript: transcribed speech
        """
        def log(name, msg):
            if on_step:
                on_step(name, msg)

        # Write video to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            video_path = tmp.name

        wav_path = video_path.replace(".mp4", ".wav")
        audio_bytes = None
        transcript = None
        text_result = None

        try:
            # ── 1. Extract audio ───────────────────────────────────────────────
            log("audio_extract", "Extracting audio from video…")
            audio_ok = extract_audio_from_video(video_path, wav_path)

            if audio_ok and os.path.exists(wav_path):
                with open(wav_path, "rb") as f:
                    audio_bytes = f.read()
                log("audio_extract", "✓ Audio extracted")
            else:
                log("audio_extract", "⚠ Audio extraction failed; using mock")

            # ── 2. Speech-to-text ──────────────────────────────────────────────
            log("stt", "Transcribing speech…")
            transcript = (
                transcribe_audio(wav_path)
                if audio_ok and os.path.exists(wav_path)
                else "Could not transcribe audio."
            )
            log("stt", f"✓ Transcript: {transcript[:80]}…")

            # ── 3. Text model ──────────────────────────────────────────────────
            log("text_model", "Running text emotion model…")
            text_result = process_text(transcript)
            log("text_model", f"✓ Text → {text_result['emotion']} ({text_result['confidence']:.1%})")

            # ── 4. Audio model ─────────────────────────────────────────────────
            log("audio_model", "Running audio emotion model…")
            audio_result = process_audio(
                audio_bytes or b"",
                filename=source_label,
            )
            log("audio_model", f"✓ Audio → {audio_result['emotion']} ({audio_result['confidence']:.1%})")

            # ── 5. Visual model ────────────────────────────────────────────────
            log("visual_model", f"Running visual model on {self.max_video_frames} frames…")
            visual_result = process_video(video_bytes, max_frames=self.max_video_frames)
            log("visual_model", f"✓ Visual → {visual_result['emotion']} ({visual_result['confidence']:.1%})")

            # ── 6. Fusion ──────────────────────────────────────────────────────
            log("fusion", "Fusing modality predictions…")
            fused = fuse_predictions(
                text_result, audio_result, visual_result,
                weights=self.weights,
            )
            log("fusion", f"✓ Fused → {fused['emotion']} ({fused['confidence']:.1%})")

        finally:
            try:
                os.unlink(video_path)
            except Exception:
                pass
            try:
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
            except Exception:
                pass

        return {
            "fused": fused,
            "text_result": text_result,
            "audio_result": audio_result,
            "visual_result": visual_result,
            "transcript": transcript,
        }
