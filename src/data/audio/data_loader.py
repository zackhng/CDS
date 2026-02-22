import os
from typing import List, Dict
from torch.utils.data import Dataset
import torch
import torchaudio

class AudioDataset(Dataset):
    def __init__(self, folder_path: str, labels: Dict[str,int]=None, sample_rate: int =16000, n_mels: int=64):
        """
        folder_path: folder containing .wav audio files
        labels: optional dict {audio_id: label}, where audio_id = filename without extension
        sample_rate: resample all audio to this
        n_mels: number of Mel filterbanks
        """
        super().__init__()
        self.folder_path = folder_path
        self.labels = labels
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        # List all .wav files
        self.audio_paths = [os.path.join(folder_path, f) 
                            for f in os.listdir(folder_path) 
                            if f.lower().endswith(".wav")]

        # Define Mel-spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels
        )

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        audio_id = os.path.splitext(os.path.basename(audio_path))[0]

        waveform, sr = torchaudio.load(audio_path)
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if multi-channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Apply Mel-spectrogram
        mel_spec = self.mel_transform(waveform)  # shape: [1, n_mels, time]

        if self.labels:
            label = self.labels.get(audio_id, -1)
            return mel_spec, torch.tensor(label, dtype=torch.long)

        return mel_spec