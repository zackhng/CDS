import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import Tensor
from PIL import Image
from torchvision import transforms
import tqdm
import cv2
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from facenet_pytorch import MTCNN

def get_device():
    """Return GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_video_data(metadata_csv: str, frames_root_dir: str)->tuple[list[str], list[int], dict[str,int]]:
    """
    Load video directories and aligned labels from CSV metadata.

    Args:
        metadata_csv (str): Path to metadata CSV file.
        frames_root_dir (str): Root directory containing video frame folders.

    Returns:
        video_dirs (list of str): List of video frame folder paths.
        labels (list of int): Corresponding integer emotion labels.
        emotion_to_idx (dict): Mapping of emotion string → integer label.
    """
    df = pd.read_csv(metadata_csv)

    # map emotion → integer label
    emotion_to_idx = {e: i for i, e in enumerate(sorted(df["emotion"].unique()))}

    # list all subfolders (each folder = one video)
    video_dirs = [
        os.path.join(frames_root_dir, folder)
        for folder in sorted(os.listdir(frames_root_dir))
        if os.path.isdir(os.path.join(frames_root_dir, folder))
    ]

    # Build a dict: video_id → label
    video_label_map = {
        row["video_id"]: emotion_to_idx[row["emotion"]]
        for _, row in df.iterrows()
    }

    # align labels with folder order
    labels = [video_label_map[os.path.basename(d)] for d in video_dirs]

    return video_dirs, labels, emotion_to_idx

def evaluate(all_labels: Tensor, all_preds:Tensor)->None:
    '''
    Print evaluation matrix.
    Args:
        all_labels (Tensor): True Labels.
        all_preds (Tensor): Predicted Labels.
    '''
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {accuracy:.4f}")

    print("Classification Report:")
    print(classification_report(all_labels, all_preds))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

def extract_faces_preserve_structure_mtcnn(
    input_root,
    output_root,
    resize_to=(224, 224),
    min_confidence=0.5,
    blur_threshold=40,
    margin_ratio=0.2
):
    """
    Extract faces from images using MTCNN, preserve folder structure,
    filter blurry faces, apply margin, resize, and save.
    """

    # -----------------------
    # Initialize MTCNN
    # -----------------------
    detector = MTCNN(keep_all=True)

    # -----------------------
    # Blur check function
    # -----------------------
    def is_blurry(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return score < blur_threshold

    # -----------------------
    # Counters
    # -----------------------
    total_images = 0
    saved_faces = 0
    skipped_no_face = 0
    skipped_low_conf = 0
    skipped_blurry = 0

    # -----------------------
    # Walk through folders
    # -----------------------
    for root, _, files in os.walk(input_root):
        rel_path = os.path.relpath(root, input_root)
        output_subfolder = os.path.join(output_root, rel_path)
        os.makedirs(output_subfolder, exist_ok=True)

        for img_file in files:
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            total_images += 1
            img_path = os.path.join(root, img_file)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            # Convert to RGB for MTCNN
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Detect faces
            boxes, probs = detector.detect(img_rgb)

            if boxes is None or len(boxes) == 0:
                skipped_no_face += 1
                continue

            # Take first face
            box = boxes[0]
            confidence = probs[0]

            if confidence < min_confidence:
                skipped_low_conf += 1
                continue

            x1, y1, x2, y2 = box.astype(int)

            # Apply margin
            w, h = x2 - x1, y2 - y1
            mx, my = int(w * margin_ratio), int(h * margin_ratio)

            x1 = max(0, x1 - mx)
            y1 = max(0, y1 - my)
            x2 = min(img_bgr.shape[1], x2 + mx)
            y2 = min(img_bgr.shape[0], y2 + my)

            face_img = img_bgr[y1:y2, x1:x2]

            if face_img.size == 0:
                continue

            # Blur filtering
            if is_blurry(face_img):
                skipped_blurry += 1
                continue

            # Resize
            if resize_to is not None:
                face_img = cv2.resize(face_img, resize_to, interpolation=cv2.INTER_AREA)

            # Save face
            face_filename = f"{os.path.splitext(img_file)[0]}_face.jpg"
            save_path = os.path.join(output_subfolder, face_filename)
            cv2.imwrite(save_path, face_img)
            saved_faces += 1

    # -----------------------
    # Summary
    # -----------------------
    print("\n===== Extraction Summary =====")
    print(f"Total images processed : {total_images}")
    print(f"Faces saved            : {saved_faces}")
    print(f"No face detected       : {skipped_no_face}")
    print(f"Low confidence skipped : {skipped_low_conf}")
    print(f"Blurry faces skipped   : {skipped_blurry}")
    print(f"Saved to               : {output_root}")