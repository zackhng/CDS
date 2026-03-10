import os
import pandas as pd
from pathlib import Path

# Dictionaries for mapping filename numbers to labels
emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

intensity_dict = {
    "01": "normal",
    "02": "strong"
}

statement_dict = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door"
}

def get_gender(actor_num):
    return "female" if int(actor_num) % 2 == 0 else "male" # odd num: male, even num: female

# Paths
ROOT = Path(__file__).resolve().parents[3]  # project root
audio_dir = ROOT / "raw_data" / "ravdess_audio"
metadata_file = ROOT / "raw_data" / "metadata.csv"

# Build metadata
data = []

for file in os.listdir(audio_dir):
    if file.endswith(".wav"):
        parts = file.replace(".wav", "").split("-")
        
        emotion = emotion_dict[parts[2]]
        intensity = intensity_dict[parts[3]]
        statement = statement_dict[parts[4]]
        actor = int(parts[6])
        gender = get_gender(actor)
        
        data.append({
            "file": file,
            "emotion": emotion,
            "intensity": intensity,
            "statement": statement,
            "gender": gender
        })

# Convert to DataFrame
df = pd.DataFrame(data)
df.to_csv(metadata_file, index=False)

print(f"metadata.csv created at {metadata_file} with {len(df)} entries")