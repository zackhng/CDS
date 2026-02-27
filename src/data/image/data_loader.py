import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, video_dirs, labels, transform=None, n_frames=8):
        super().__init__()
        valid_video_dirs = []
        valid_labels = []

        for folder, label in zip(video_dirs, labels):
            if os.path.exists(folder) and len(os.listdir(folder)) > 0:
                valid_video_dirs.append(folder)
                valid_labels.append(label)
            else:
                continue

        self.video_dirs = valid_video_dirs
        self.labels = valid_labels
        self.transform = transform
        self.n_frames = n_frames

    def __len__(self):
        return len(self.video_dirs)

    def sample_indices(self, total_frames):
        """
        Always return exactly n_frames indices.
        Automatically repeats frames if video is short.
        """

        if total_frames == 0:
            return []

        # uniform sampling (works for both long & short videos)
        indices = np.linspace(
            0,
            total_frames - 1,
            self.n_frames
        )

        indices = np.clip(indices.astype(int), 0, total_frames - 1)

        return indices

    def __getitem__(self, idx):
        folder = self.video_dirs[idx]
        label = self.labels[idx]

        # get frame list
        frames = sorted(os.listdir(folder))

        # ✅ sample indices safely
        chosen_indices = self.sample_indices(len(frames))

        if len(chosen_indices) == 0:
            # you can raise an IndexError to let DataLoader skip it
            raise IndexError(f"Video folder '{folder}' is empty")

        images = []
        for i in chosen_indices:
            img_path = os.path.join(folder, frames[i])
            img = Image.open(img_path).convert("RGB")

            if self.transform:
                img = self.transform(img)

            images.append(img)

        # shape: (T, C, H, W)
        images = torch.stack(images)

        return images, label