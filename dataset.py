import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, clean_folder, noisy_folder, sr=16000, target_length=32000):
        self.sr = sr
        self.target_length = target_length
        self.file_pairs = []
        
        noisy_files = [f for f in os.listdir(noisy_folder) if f.endswith('.wav')]
        for noisy_file in noisy_files:
            clean_file = noisy_file
            if os.path.exists(os.path.join(clean_folder, clean_file)):
                self.file_pairs.append({
                    'noisy': os.path.join(noisy_folder, noisy_file),
                    'clean': os.path.join(clean_folder, clean_file)
                })

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        pair = self.file_pairs[idx]
        noisy, _ = librosa.load(pair['noisy'], sr=self.sr)
        clean, _ = librosa.load(pair['clean'], sr=self.sr)

        if len(noisy) < self.target_length:
            pad_length = self.target_length - len(noisy)
            noisy = np.pad(noisy, (0, pad_length), mode='constant')
            clean = np.pad(clean, (0, pad_length), mode='constant')
        elif len(noisy) > self.target_length:
            start = np.random.randint(0, len(noisy) - self.target_length)
            noisy = noisy[start:start + self.target_length]
            clean = clean[start:start + self.target_length]

        return (torch.tensor(noisy, dtype=torch.float32).unsqueeze(0),
                torch.tensor(clean, dtype=torch.float32).unsqueeze(0))