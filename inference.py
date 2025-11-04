import torch
import librosa
import soundfile as sf
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
from model import UNetANC

def record_audio(filename: str = "recorded_noisy.wav", duration: int = 15, sr: int = 16000) -> None:
    """Record audio from microphone and save to file."""
    print(f"Recording for {duration} seconds...")
    try:
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        sf.write(filename, audio.squeeze(), sr)
        print(f"✓ Recording saved as {filename}")
    except Exception as e:
        raise RuntimeError(f"Failed to record audio: {e}") from e

def plot_audio_comparison(noisy_audio: np.ndarray, denoised_audio: np.ndarray, 
                         sr: int = 16000, save_path: str = 'audio_comparison.png') -> None:
    """Generate and save comparison plots of noisy vs denoised audio."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Waveforms
    axes[0, 0].plot(noisy_audio)
    axes[0, 0].set_title("Noisy Audio Waveform")
    axes[0, 0].set_xlabel("Sample")
    axes[0, 0].set_ylabel("Amplitude")
    
    axes[0, 1].plot(denoised_audio)
    axes[0, 1].set_title("Denoised Audio Waveform")
    axes[0, 1].set_xlabel("Sample")
    axes[0, 1].set_ylabel("Amplitude")
    
    # Spectrograms
    for idx, (audio, title) in enumerate([(noisy_audio, "Noisy"), (denoised_audio, "Denoised")]):
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[1, idx])
        fig.colorbar(img, ax=axes[1, idx], format='%+2.0f dB')
        axes[1, idx].set_title(f"{title} Audio Spectrogram")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Comparison plot saved as {save_path}")

def process_audio(model: torch.nn.Module, audio_file: str, device: torch.device, 
                 chunk_size: int = 32000, sr: int = 16000) -> np.ndarray:
    """Process audio file through denoising model in chunks."""
    if not Path(audio_file).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    audio, _ = librosa.load(audio_file, sr=sr, mono=True)
    
    # Process in chunks
    denoised_chunks = []
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            
            # Pad last chunk if needed
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
            
            # Process chunk
            chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0).to(device)
            denoised_chunk = model(chunk_tensor).squeeze().cpu().numpy()
            denoised_chunks.append(denoised_chunk)
    
    # Concatenate and trim to original length
    return np.concatenate(denoised_chunks)[:len(audio)]

def load_model(model_path: str = 'best_model.pth', device: Optional[torch.device] = None) -> Tuple[UNetANC, torch.device]:
    """Load trained model from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    model = UNetANC().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded on {device}")
    
    return model, device

def main():
    """Main pipeline: record, denoise, and visualize audio."""
    # Configuration
    RECORD_FILE = "recorded_noisy.wav"
    OUTPUT_FILE = "denoised_output.wav"
    MODEL_PATH = "best_model.pth"
    SR = 16000
    
    # Load model
    model, device = load_model(MODEL_PATH)
    
    # Record audio
    record_audio(RECORD_FILE, duration=15, sr=SR)
    
    # Denoise audio
    print("Processing audio...")
    denoised_audio = process_audio(model, RECORD_FILE, device, sr=SR)
    sf.write(OUTPUT_FILE, denoised_audio, SR)
    print(f"✓ Denoised audio saved as {OUTPUT_FILE}")
    
    # Visualize comparison
    noisy_audio, _ = librosa.load(RECORD_FILE, sr=SR)
    plot_audio_comparison(noisy_audio, denoised_audio, sr=SR)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"✗ Error: {e}")
        raise
