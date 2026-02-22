import os
from typing import Dict
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, folder_path: str, labels: Dict[str,int]=None):
        """
        folder_path: folder containing text files (.txt)
        labels: optional dict {text_id: label}, where text_id = filename without extension
        """
        super().__init__()
        self.folder_path = folder_path
        self.labels = labels

        # List all text files
        self.text_paths = [os.path.join(folder_path, f)
                           for f in os.listdir(folder_path)
                           if f.lower().endswith(".txt")]

    def __len__(self):
        return len(self.text_paths)

    def __getitem__(self, idx):
        text_path = self.text_paths[idx]
        text_id = os.path.splitext(os.path.basename(text_path))[0]

        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if self.labels:
            label = self.labels.get(text_id, -1)
            return text, label

        return text