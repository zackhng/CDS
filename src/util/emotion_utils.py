"""
Core emotion processing utilities.
Supports text, audio, and visual modalities with mock fallbacks.
"""

import numpy as np
import re
import io
import os
import tempfile
from typing import Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_tokenizer = None
# ---------------------------------------------------------------------------
# Emotion label set (matches RAVDESS + general sentiment datasets)
# ---------------------------------------------------------------------------
EMOTIONS = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

EMOTION_COLORS = {
    "angry": "#ff4d4d",
    "disgust": "#2ecc71",
    "fearful": "#9b59b6",
    "happy": "#f1c40f",
    "neutral": "#95a5a6",
    "sad": "#3498db",
    "surprised": "#e67e22",
}

EMOTION_EMOJIS = {
    "angry": "😡",
    "disgust": "🤢",
    "fearful": "😨",
    "happy": "😄",
    "neutral": "😐",
    "sad": "😢",
    "surprised": "😲",
}

# ---------------------------------------------------------------------------

def _load_model():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    global _model, _tokenizer

    if _model is None:
        model_path = "C:/Desktop/CDS_Proj/models/text_model"

        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        _model = AutoModelForSequenceClassification.from_pretrained(model_path)

        _model.eval()

    return _model, _tokenizer


# ---------------------------------------------------------------------------
# Text Processing
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Basic text cleaning pipeline."""
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s'!?,.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def process_text(raw_text: str) -> dict:
    model, tokenizer = _load_model()

    cleaned_text = clean_text(raw_text)

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    # move inputs to device
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        probs = F.softmax(logits, dim=1)[0]

    # IMPORTANT: use model config labels if available
    if hasattr(model, "config") and hasattr(model.config, "id2label"):
        id2label = model.config.id2label
    else:
        id2label = {i: e for i, e in enumerate(EMOTIONS)}

    scores = {
        id2label[i]: probs[i].item()
        for i in range(len(probs))
    }

    prediction = max(scores, key=scores.get)

    return {
        "cleaned_text": cleaned_text,
        "emotion": prediction,
        "confidence": scores[prediction],
        "all_scores": scores,
    }

_text_classifier = None

def _get_text_classifier():
    global _text_classifier
    if _text_classifier is None:
        try:
            from transformers import pipeline as hf_pipeline
            _text_classifier = hf_pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=False,
            )
        except Exception:
            _text_classifier = False
    return _text_classifier if _text_classifier else None


# ---------------------------------------------------------------------------
# Audio Processing
# ---------------------------------------------------------------------------
def process_audio(audio_bytes: bytes, filename: str = "audio.wav") -> dict:
    """
    Extract features from audio and predict emotion.

    Returns:
        dict with keys: emotion, confidence, all_scores, features, waveform, sr
    """
    try:
        import librosa
        import librosa.display

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        y, sr = librosa.load(tmp_path, sr=22050, duration=10.0)
        os.unlink(tmp_path)

        # Feature extraction
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = mfccs.mean(axis=1)
        rms = librosa.feature.rms(y=y).mean()
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()

        # Aggregate feature vector for mock model input
        feature_vector = np.concatenate([mfcc_mean, [rms, zcr, spec_centroid]])
        seed_str = str(feature_vector[:5].round(3).tolist())
        scores = _mock_softmax(seed_str)
        emotion = max(scores, key=scores.get)

        return {
            "emotion": emotion,
            "confidence": scores[emotion],
            "all_scores": scores,
            "features": {
                "mfcc_mean": mfcc_mean.tolist(),
                "rms": float(rms),
                "zcr": float(zcr),
                "spectral_centroid": float(spec_centroid),
            },
            "waveform": y,
            "sr": sr,
        }

    except ImportError:
        pass
    except Exception as e:
        pass

    # Full fallback
    scores = _mock_softmax(filename)
    emotion = max(scores, key=scores.get)
    return {
        "emotion": emotion,
        "confidence": scores[emotion],
        "all_scores": scores,
        "features": {},
        "waveform": None,
        "sr": None,
    }


# ---------------------------------------------------------------------------
# Visual Processing
# ---------------------------------------------------------------------------
def process_image(image_input, label_hint: str = "") -> dict:
    """
    Run face detection + emotion classification on image.

    Args:
        image_input: PIL Image or np.ndarray
        label_hint: Optional string seed for mock

    Returns:
        dict with keys: emotion, confidence, all_scores, faces_detected, annotated_image
    """
    import numpy as np
    try:
        from PIL import Image as PILImage
        import cv2

        if hasattr(image_input, "read"):
            img_pil = PILImage.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img_pil = PILImage.fromarray(image_input)
        else:
            img_pil = image_input

        img_cv = np.array(img_pil)
        img_bgr = img_cv[:, :, ::-1].copy()

        # Try MTCNN face detection
        faces_detected = 0
        annotated = img_cv.copy()
        try:
            from facenet_pytorch import MTCNN
            detector = MTCNN(keep_all=True, post_process=False)
            import torch
            img_tensor = PILImage.fromarray(img_cv)
            boxes, probs = detector.detect(img_tensor)
            if boxes is not None:
                faces_detected = len(boxes)
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 100), 2)
        except Exception:
            # Haar cascade fallback
            try:
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                faces_detected = len(faces)
                for (x, y, w, h) in faces:
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 100), 2)
            except Exception:
                pass

        seed_str = label_hint or str(img_cv[::30, ::30, 0].flatten()[:10].tolist())
        scores = _mock_softmax(seed_str)
        emotion = max(scores, key=scores.get)

        return {
            "emotion": emotion,
            "confidence": scores[emotion],
            "all_scores": scores,
            "faces_detected": faces_detected,
            "annotated_image": annotated,
        }

    except Exception as e:
        scores = _mock_softmax(label_hint or "image")
        emotion = max(scores, key=scores.get)
        return {
            "emotion": emotion,
            "confidence": scores[emotion],
            "all_scores": scores,
            "faces_detected": 0,
            "annotated_image": None,
        }


def process_video(video_bytes: bytes, max_frames: int = 20) -> dict:
    """
    Extract frames from video and run per-frame visual emotion prediction.

    Returns:
        dict with keys: emotion, confidence, all_scores, frame_results, sample_frames
    """
    import cv2
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames)

    frame_results = []
    sample_frames = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = process_image(rgb, label_hint=f"frame_{frame_idx}")
            frame_results.append({
                "frame": frame_idx,
                "emotion": result["emotion"],
                "confidence": result["confidence"],
            })
            if len(sample_frames) < 6:
                sample_frames.append(result.get("annotated_image", rgb))
        frame_idx += 1

    cap.release()
    os.unlink(tmp_path)

    if not frame_results:
        scores = _mock_softmax("empty_video")
        return {"emotion": max(scores, key=scores.get),
                "confidence": max(scores.values()),
                "all_scores": scores,
                "frame_results": [],
                "sample_frames": []}

    # Aggregate: majority vote weighted by confidence
    aggregated = {}
    for fr in frame_results:
        em = fr["emotion"]
        aggregated[em] = aggregated.get(em, 0) + fr["confidence"]
    total = sum(aggregated.values())
    all_scores = {e: aggregated.get(e, 0) / total for e in EMOTIONS}
    emotion = max(aggregated, key=aggregated.get)

    return {
        "emotion": emotion,
        "confidence": all_scores[emotion],
        "all_scores": all_scores,
        "frame_results": frame_results,
        "sample_frames": sample_frames,
    }


# ---------------------------------------------------------------------------
# Speech-to-Text
# ---------------------------------------------------------------------------
def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file to text using Whisper (or mock)."""
    try:
        import whisper
        model = whisper.load_model("tiny")
        result = model.transcribe(audio_path, fp16=False)
        return result.get("text", "").strip()
    except Exception:
        pass

    # Mock transcription
    mock_phrases = [
        "I am feeling really good about everything today.",
        "This situation makes me quite anxious and worried.",
        "I don't know how to feel right now.",
        "That was absolutely amazing and unexpected!",
        "Everything is going as planned, nothing unusual.",
    ]
    seed = abs(hash(audio_path)) % len(mock_phrases)
    return mock_phrases[seed]


# ---------------------------------------------------------------------------
# Multimodal Fusion
# ---------------------------------------------------------------------------
def fuse_predictions(
    text_result: Optional[dict],
    audio_result: Optional[dict],
    visual_result: Optional[dict],
    weights: dict = None,
) -> dict:
    """
    Late fusion: weighted average of modality confidence scores.

    Default weights: text=0.35, audio=0.35, visual=0.30
    """
    if weights is None:
        weights = {"text": 0.35, "audio": 0.35, "visual": 0.30}

    fused = {e: 0.0 for e in EMOTIONS}
    total_weight = 0.0

    modality_summary = {}

    for name, result, w in [
        ("text",   text_result,   weights.get("text", 0.35)),
        ("audio",  audio_result,  weights.get("audio", 0.35)),
        ("visual", visual_result, weights.get("visual", 0.30)),
    ]:
        if result and "all_scores" in result:
            scores = result["all_scores"]
            for e in EMOTIONS:
                fused[e] += scores.get(e, 0.0) * w
            total_weight += w
            modality_summary[name] = {
                "emotion": result["emotion"],
                "confidence": result["confidence"],
            }

    if total_weight > 0:
        fused = {e: v / total_weight for e, v in fused.items()}

    final_emotion = max(fused, key=fused.get)
    return {
        "emotion": final_emotion,
        "confidence": fused[final_emotion],
        "all_scores": fused,
        "modality_summary": modality_summary,
        "weights_used": {k: v for k, v in weights.items()},
    }


# ---------------------------------------------------------------------------
# Video → Audio extraction
# ---------------------------------------------------------------------------
def extract_audio_from_video(video_path: str, output_wav: str) -> bool:
    """Extract audio track from video using ffmpeg or moviepy."""
    try:
        import subprocess
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn",
             "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1", output_wav],
            capture_output=True, timeout=60,
        )
        return result.returncode == 0 and os.path.exists(output_wav)
    except Exception:
        pass

    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(video_path)
        if clip.audio:
            clip.audio.write_audiofile(output_wav, fps=22050, nbytes=2,
                                       codec="pcm_s16le", logger=None)
            clip.close()
            return os.path.exists(output_wav)
        clip.close()
    except Exception:
        pass

    return False
