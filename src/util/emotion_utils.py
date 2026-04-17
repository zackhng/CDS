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

_text_model = None
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

def _load_text_model():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    global _text_model, _tokenizer

    if _text_model is None:
        model_path = "C:/Users/zhaoh/Desktop/CDS/models/text_model"

        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        _text_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        _text_model.to(_device)
        _text_model.eval()

    return _text_model, _tokenizer

_visual_model = None

def _load_visual_model():
    global _visual_model

    if _visual_model is None:
        import torch
        import torch.nn as nn
        from transformers import AutoModel

        class DINOv2Classifier(nn.Module):
            def __init__(
                self,
                num_classes: int,
                backbone: str = "facebook/dinov2-base",
                freeze_backbone: bool = True
            ):
                super().__init__()

                self.backbone = AutoModel.from_pretrained(backbone)

                hidden_size = self.backbone.config.hidden_size

                self.classifier = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, num_classes)
                )

                if freeze_backbone:
                    for p in self.backbone.parameters():
                        p.requires_grad = False

            def forward(self, pixel_values):
                outputs = self.backbone(pixel_values=pixel_values)
                cls_token = outputs.last_hidden_state[:, 0]
                return self.classifier(cls_token)

        # 1. Instantiate model FIRST
        _visual_model = DINOv2Classifier(
            num_classes=7,
            freeze_backbone=True
        ).to(_device)

        # 2. Load weights (IMPORTANT: state_dict)
        model_path = "C:/Users/zhaoh/Desktop/CDS/models/image_dino.pt"
        state_dict = torch.load(model_path, map_location=_device)

        _visual_model.load_state_dict(state_dict)

        # 3. Eval mode
        _visual_model.eval()

    return _visual_model

_audio_model = None

def _load_audio_model():
    global _audio_model

    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Model

    device = _device
    MODEL_NAME = "facebook/wav2vec2-base"

    class Wav2Vec2EmotionClassifier(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(MODEL_NAME)

            hidden_size = self.wav2vec2.config.hidden_size

            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )

        def forward(self, input_values):
            outputs = self.wav2vec2(input_values)
            pooled = outputs.last_hidden_state.mean(dim=1)
            return self.classifier(pooled)

    if _audio_model is None:
        path = r"C:/Users/zhaoh/Desktop/CDS/models/audio_wave2vec2.pt"

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        model = Wav2Vec2EmotionClassifier(
            num_classes=checkpoint["num_classes"]
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        _audio_model = model

    return _audio_model

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
    model, tokenizer = _load_text_model()

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
    import torch
    import tempfile
    import os
    import librosa
    from transformers import Wav2Vec2Processor

    model = _load_audio_model()

    # IMPORTANT: MUST match training model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    # ---- Save temp file ----
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # MUST be 16kHz (same as training)
        y, sr = librosa.load(tmp_path, sr=16000, mono=True)
    finally:
        os.unlink(tmp_path)

    # ---- Correct Wav2Vec2 input ----
    input_values = processor(
        y,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_values.to(_device)

    # ---- Inference ----
    with torch.no_grad():
        logits = model(input_values)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    scores = {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))}
    emotion = max(scores, key=scores.get)

    return {
        "emotion": emotion,
        "confidence": scores[emotion],
        "all_scores": scores,
        "waveform": y,
        "sr": sr
    }

# ---------------------------------------------------------------------------
# Visual Processing
# ---------------------------------------------------------------------------
def process_image(image_input) -> dict:
    """
    Run face detection + emotion classification on image.

    Args:
        image_input: PIL Image or np.ndarray

    Returns:
        dict with keys: emotion, confidence, all_scores, faces_detected, annotated_image
    """
    import numpy as np
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

    model = _load_visual_model()

    # ---- Preprocess ----
    face_img = img_cv

    # If faces detected → crop first face
    if faces_detected > 0:
        try:
            if 'boxes' in locals() and boxes is not None:
                x1, y1, x2, y2 = [int(b) for b in boxes[0]]
                face_img = img_cv[y1:y2, x1:x2]
        except:
            pass

    import cv2
    face_img = cv2.resize(face_img, (224, 224))  # ⚠️ CHANGE if your model uses different size
    face_img = face_img / 255.0

    # HWC → CHW
    face_img = np.transpose(face_img, (2, 0, 1))

    import torch
    input_tensor = torch.tensor(face_img, dtype=torch.float32).unsqueeze(0).to(_device)

    # ---- Inference ----
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    # ---- Convert to dict ----
    scores = {
        EMOTIONS[i]: float(probs[i])
        for i in range(len(EMOTIONS))
    }

    emotion = max(scores, key=scores.get)

    return {
        "emotion": emotion,
        "confidence": scores[emotion],
        "all_scores": scores,
        "faces_detected": faces_detected,
        "annotated_image": annotated,
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
            result = process_image(rgb)
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

from moviepy import VideoFileClip

def extract_audio_from_video(video_path, wav_path):
    try:
        clip = VideoFileClip(video_path)

        # if no audio stream
        if clip.audio is None:
            clip.close()
            return False

        clip.audio.write_audiofile(
            wav_path,
            codec="pcm_s16le",
            logger=None  # suppress verbose logs (good for Streamlit)
        )

        clip.close()
        return True

    except Exception as e:
        print("Audio extraction error:", e)
        return False