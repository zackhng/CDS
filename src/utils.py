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

def extract_faces_preserve_structure(input_root, output_root, resize_to=(224, 224)):
    """
    Extract faces from images in a folder of folders and save them to a new folder,
    preserving the same internal subfolder structure.
    
    Args:
        input_root (str): Path to the root folder containing subfolders with images.
        output_root (str): Path to the root folder where output will be saved.
        resize_to (tuple, optional): Resize cropped faces to this size. Set None to skip resizing.
    """
    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Loop through all subfolders
    for root, dirs, files in os.walk(input_root):
        # Compute relative path to maintain folder structure
        rel_path = os.path.relpath(root, input_root)
        output_subfolder = os.path.join(output_root, rel_path)
        os.makedirs(output_subfolder, exist_ok=True)

        # Process images
        for img_file in files:
            if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(root, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            # Save each face
            for i, (x, y, w, h) in enumerate(faces):
                face_img = img[y:y+h, x:x+w]
                if resize_to is not None:
                    face_img = cv2.resize(face_img, resize_to)

                face_filename = f"{os.path.splitext(img_file)[0]}_face{i}.jpg"
                cv2.imwrite(os.path.join(output_subfolder, face_filename), face_img)

    print(f"Face extraction complete! Faces saved to: {output_root}")