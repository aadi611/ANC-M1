import torch
import librosa
import soundfile as sf
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from model import UNetANC

def record_audio(filename="recorded_noisy.wav", duration=15, sr=16000):
    print("Recording... (Duration: 15 seconds)")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    write(filename, sr, np.squeeze(audio))
    print(f"Recording saved as {filename}")

def plot_audio_comparison(noisy_audio, denoised_audio, sr=16000):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(noisy_audio)
    plt.title("Noisy Audio Waveform")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    
    plt.subplot(2, 2, 2)
    plt.plot(denoised_audio)
    plt.title("Denoised Audio Waveform")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    
    plt.subplot(2, 2, 3)
    D_noisy = librosa.amplitude_to_db(np.abs(librosa.stft(noisy_audio)), ref=np.max)
    librosa.display.specshow(D_noisy, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Noisy Audio Spectrogram")
    
    plt.subplot(2, 2, 4)
    D_denoised = librosa.amplitude_to_db(np.abs(librosa.stft(denoised_audio)), ref=np.max)
    librosa.display.specshow(D_denoised, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Denoised Audio Spectrogram")
    
    plt.tight_layout()
    plt.savefig('audio_comparison.png')
    plt.show()

def process_audio(model, audio_file, device, chunk_size=32000):
    audio, sr = librosa.load(audio_file, sr=16000)
    chunks = []
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
        chunks.append(torch.tensor(chunk).float().unsqueeze(0).unsqueeze(0))
    
    denoised_chunks = []
    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            chunk = chunk.to(device)
            denoised_chunk = model(chunk).squeeze().cpu().numpy()
            denoised_chunks.append(denoised_chunk)
    
    denoised_audio = np.concatenate(denoised_chunks)[:len(audio)]
    return denoised_audio

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetANC().to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device)['model_state_dict'])
    
    record_audio()
    denoised_audio = process_audio(model, "recorded_noisy.wav", device)
    sf.write("denoised_output.wav", denoised_audio, 16000)
    
    noisy_audio, _ = librosa.load("recorded_noisy.wav", sr=16000)
    plot_audio_comparison(noisy_audio, denoised_audio)