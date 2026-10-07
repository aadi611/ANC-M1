import torch
import librosa
import librosa.display
import soundfile as sf
import numpy as np
import sounddevice as sd


from pathlib import Path
from typing import Optional, Tuple

from model import UNetANC


# =========================================================
# CONFIGURATION
# =========================================================

SR = 16_000
RECORD_DURATION = 15

RECORD_FILE = "recorded_noisy.wav"
OUTPUT_FILE = "denoised_output.wav"
MODEL_PATH = "best_model.pth"

# Keep this equal to the input size used while training.
CHUNK_SIZE = 32_000

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# AUDIO RECORDING
# =========================================================

def record_audio(
    filename: str = RECORD_FILE,
    duration: int = RECORD_DURATION,
    sr: int = SR,
) -> None:
    """
    Record mono audio from the default microphone
    and save it as a WAV file.
    """

    print(f"\n🎙 Recording for {duration} seconds...")

    try:
        audio = sd.rec(
            int(duration * sr),
            samplerate=sr,
            channels=1,
            dtype=np.float32,
        )

        sd.wait()

        # Convert [samples, 1] -> [samples]
        sf.write(
            filename,
            audio[:, 0],
            sr,
        )

        print(f"✓ Recording saved: {filename}")

    except Exception as exc:
        raise RuntimeError(
            f"Failed to record audio: {exc}"
        ) from exc


# =========================================================
# MODEL LOADING
# =========================================================

def load_model(
    model_path: str = MODEL_PATH,
    device: Optional[torch.device] = None,
) -> Tuple[UNetANC, torch.device]:
    """
    Load the trained UNetANC model.
    """

    device = device or DEVICE

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    print(f"\nLoading model from: {model_path}")

    model = UNetANC().to(device)

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    # Disable training-specific layers such as dropout.
    model.eval()

    # CUDA inference optimization.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"✓ Model loaded successfully")
    print(f"✓ Device: {device}")

    if device.type == "cuda":
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

    return model, device


# =========================================================
# AUDIO PROCESSING
# =========================================================

def process_audio(
    model: torch.nn.Module,
    audio: np.ndarray,
    device: torch.device,
    chunk_size: int = CHUNK_SIZE,
) -> np.ndarray:
    """
    Process audio through the denoising model.

    Audio is split into fixed-size chunks and processed
    together as a batch for better inference performance.
    """

    original_length = len(audio)

    # -----------------------------------------------------
    # Pad audio to a multiple of chunk_size
    # -----------------------------------------------------

    padded_length = (
        (original_length + chunk_size - 1)
        // chunk_size
    ) * chunk_size

    if padded_length != original_length:

        audio = np.pad(
            audio,
            (
                0,
                padded_length - original_length,
            ),
            mode="constant",
        )

    # -----------------------------------------------------
    # Convert audio into chunks
    # -----------------------------------------------------

    chunks = audio.reshape(
        -1,
        chunk_size,
    )

    print(
        f"\nProcessing {len(chunks)} chunks "
        f"of {chunk_size} samples..."
    )

    # -----------------------------------------------------
    # NumPy -> PyTorch
    # -----------------------------------------------------

    audio_tensor = torch.from_numpy(
        chunks.astype(
            np.float32,
            copy=False,
        )
    )

    # Shape:
    #
    # [num_chunks, chunk_size]
    #
    # becomes:
    #
    # [num_chunks, 1, chunk_size]

    audio_tensor = audio_tensor.unsqueeze(1)

    audio_tensor = audio_tensor.to(
        device,
        non_blocking=True,
    )

    # -----------------------------------------------------
    # Model inference
    # -----------------------------------------------------

    with torch.inference_mode():

        if device.type == "cuda":

            # Mixed precision can significantly speed up
            # inference on modern NVIDIA GPUs.

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                output = model(audio_tensor)

        else:
            output = model(audio_tensor)

    # -----------------------------------------------------
    # Tensor -> NumPy
    # -----------------------------------------------------

    denoised = (
        output
        .squeeze(1)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    # Remove padding.
    denoised = denoised[:original_length]

    return denoised


# =========================================================
# AUDIO VISUALIZATION
# =========================================================

def plot_audio_comparison(
    noisy_audio: np.ndarray,
    denoised_audio: np.ndarray,
    sr: int = SR,
    save_path: str = "audio_comparison.png",
) -> None:
    """
    Generate waveform and spectrogram comparisons.
    """

    print("\nGenerating comparison plot...")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 9),
    )

    # -----------------------------------------------------
    # Time axes
    # -----------------------------------------------------

    noisy_time = (
        np.arange(len(noisy_audio)) / sr
    )

    denoised_time = (
        np.arange(len(denoised_audio)) / sr
    )

    # -----------------------------------------------------
    # Noisy waveform
    # -----------------------------------------------------

    axes[0, 0].plot(
        noisy_time,
        noisy_audio,
        linewidth=0.8,
    )

    axes[0, 0].set_title(
        "Noisy Audio Waveform"
    )

    axes[0, 0].set_xlabel(
        "Time (seconds)"
    )

    axes[0, 0].set_ylabel(
        "Amplitude"
    )

    axes[0, 0].grid(
        alpha=0.3
    )

    # -----------------------------------------------------
    # Denoised waveform
    # -----------------------------------------------------

    axes[0, 1].plot(
        denoised_time,
        denoised_audio,
        linewidth=0.8,
    )

    axes[0, 1].set_title(
        "Denoised Audio Waveform"
    )

    axes[0, 1].set_xlabel(
        "Time (seconds)"
    )

    axes[0, 1].set_ylabel(
        "Amplitude"
    )

    axes[0, 1].grid(
        alpha=0.3
    )

    # -----------------------------------------------------
    # Spectrograms
    # -----------------------------------------------------

    noisy_stft = librosa.stft(
        noisy_audio
    )

    denoised_stft = librosa.stft(
        denoised_audio
    )

    noisy_db = librosa.amplitude_to_db(
        np.abs(noisy_stft),
        ref=np.max,
    )

    denoised_db = librosa.amplitude_to_db(
        np.abs(denoised_stft),
        ref=np.max,
    )

    # -----------------------------------------------------
    # Noisy spectrogram
    # -----------------------------------------------------

    noisy_img = librosa.display.specshow(
        noisy_db,
        sr=sr,
        x_axis="time",
        y_axis="hz",
        ax=axes[1, 0],
    )

    axes[1, 0].set_title(
        "Noisy Audio Spectrogram"
    )

    fig.colorbar(
        noisy_img,
        ax=axes[1, 0],
        format="%+2.0f dB",
    )

    # -----------------------------------------------------
    # Denoised spectrogram
    # -----------------------------------------------------

    denoised_img = librosa.display.specshow(
        denoised_db,
        sr=sr,
        x_axis="time",
        y_axis="hz",
        ax=axes[1, 1],
    )

    axes[1, 1].set_title(
        "Denoised Audio Spectrogram"
    )

    fig.colorbar(
        denoised_img,
        ax=axes[1, 1],
        format="%+2.0f dB",
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    plt.tight_layout()

    fig.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    # Prevent matplotlib from keeping the figure
    # in memory.
    plt.close(fig)

    print(
        f"✓ Comparison plot saved: {save_path}"
    )


# =========================================================
# MAIN PIPELINE
# =========================================================

def main() -> None:
    """
    Complete pipeline:

    1. Load trained model
    2. Record noisy audio
    3. Load audio
    4. Run denoising
    5. Save denoised audio
    6. Generate comparison visualization
    """

    print("\n" + "=" * 60)
    print("        AI AUDIO DENOISING PIPELINE")
    print("=" * 60)

    # -----------------------------------------------------
    # 1. Load model
    # -----------------------------------------------------

    model, device = load_model(
        MODEL_PATH,
        DEVICE,
    )

    # -----------------------------------------------------
    # 2. Record audio
    # -----------------------------------------------------

    record_audio(
        filename=RECORD_FILE,
        duration=RECORD_DURATION,
        sr=SR,
    )

    # -----------------------------------------------------
    # 3. Load audio
    # -----------------------------------------------------

    print("\nLoading recorded audio...")

    noisy_audio, _ = librosa.load(
        RECORD_FILE,
        sr=SR,
        mono=True,
    )

    print(
        f"✓ Audio loaded"
    )

    print(
        f"✓ Duration: "
        f"{len(noisy_audio) / SR:.2f} seconds"
    )

    print(
        f"✓ Samples: {len(noisy_audio):,}"
    )

    # -----------------------------------------------------
    # 4. Denoise
    # -----------------------------------------------------

    print("\nRunning denoising inference...")

    denoised_audio = process_audio(
        model=model,
        audio=noisy_audio,
        device=device,
        chunk_size=CHUNK_SIZE,
    )

    print("✓ Denoising completed")

    # -----------------------------------------------------
    # 5. Save output
    # -----------------------------------------------------

    sf.write(
        OUTPUT_FILE,
        denoised_audio,
        SR,
    )

    print(
        f"✓ Denoised audio saved: "
        f"{OUTPUT_FILE}"
    )

    # -----------------------------------------------------
    # 6. Generate visualization
    # -----------------------------------------------------

    plot_audio_comparison(
        noisy_audio=noisy_audio,
        denoised_audio=denoised_audio,
        sr=SR,
    )

    # -----------------------------------------------------
    # Complete
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("                 COMPLETE")
    print("=" * 60)

    print(
        f"Input  : {RECORD_FILE}"
    )

    print(
        f"Output : {OUTPUT_FILE}"
    )

    print(
        f"Device : {device}"
    )

    print("=" * 60 + "\n")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print("\n\n⚠ Process interrupted by user.")

    except Exception as exc:
        print(
            f"\n✗ Error: {exc}"
        )
        raise
```
