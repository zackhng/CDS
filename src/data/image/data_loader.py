import os
from typing import String
from torch.utils.data import Dataset
from .process import ImageProcessor

class ImageDataset(Dataset):
    def __init__(self, folder_path: String, processor: ImageProcessor, labels=None):
        super().__init__()
        self.folder_path = folder_path
        self.processor = processor
        self.image_paths = [os.path.join(folder_path, f) 
                            for f in os.listdir(folder_path) 
                            if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.labels = labels  # optional, list of same length as image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.processor.process(self.image_paths[idx])
        if self.labels:
            label = self.labels[idx]
            return img, label
        return img
