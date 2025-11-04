import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple

class AudioDataset(Dataset):
    def __init__(self, clean_folder: str, noisy_folder: str, sr: int = 16000, target_length: int = 32000):
        """
        Args:
            clean_folder: Path to clean audio files
            noisy_folder: Path to noisy audio files
            sr: Sample rate for audio loading
            target_length: Target length in samples (must be positive)
        """
        self.sr = sr
        self.target_length = max(1, target_length)  # Ensure positive length
        
        # Validate folders exist
        clean_path = Path(clean_folder)
        noisy_path = Path(noisy_folder)
        if not clean_path.exists():
            raise ValueError(f"Clean folder does not exist: {clean_folder}")
        if not noisy_path.exists():
            raise ValueError(f"Noisy folder does not exist: {noisy_folder}")
        
        # Build file pairs with validation
        self.file_pairs = []
        for noisy_file in noisy_path.glob("*.wav"):
            clean_file = clean_path / noisy_file.name
            if clean_file.exists():
                self.file_pairs.append({'noisy': str(noisy_file), 'clean': str(clean_file)})
        
        if not self.file_pairs:
            raise ValueError("No matching audio pairs found")
    
    def __len__(self) -> int:
        return len(self.file_pairs)
    
    def _process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Pad or crop audio to target length"""
        audio_len = len(audio)
        if audio_len < self.target_length:
            return np.pad(audio, (0, self.target_length - audio_len), mode='constant')
        elif audio_len > self.target_length:
            start = np.random.randint(0, audio_len - self.target_length + 1)
            return audio[start:start + self.target_length]
        return audio
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self.file_pairs):
            raise IndexError(f"Index {idx} out of range [0, {len(self.file_pairs)})")
        
        pair = self.file_pairs[idx]
        
        try:
            # Load audio with error handling
            noisy, _ = librosa.load(pair['noisy'], sr=self.sr, mono=True)
            clean, _ = librosa.load(pair['clean'], sr=self.sr, mono=True)
            
            # Process to target length
            noisy = self._process_audio(noisy)
            clean = self._process_audio(clean)
            
            # Convert to tensors
            return (torch.from_numpy(noisy).float().unsqueeze(0),
                    torch.from_numpy(clean).float().unsqueeze(0))
        
        except Exception as e:
            raise RuntimeError(f"Failed to load audio pair at index {idx}: {pair['noisy']}") from e
