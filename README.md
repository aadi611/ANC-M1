# ANC-M1: Active Noise Cancellation Model

A deep learning-based active noise cancellation system using UNet architecture for real-time audio denoising.

## Features

- Real-time audio recording and denoising
- Custom UNet architecture optimized for audio processing
- Support for both CPU and GPU inference
- Processes audio in overlapping chunks for seamless denoising
- Includes training pipeline with customizable parameters

## Prerequisites

```bash
pip install torch librosa soundfile numpy sounddevice scipy
```

## Project Structure

```
ANC-M1/
├── dataset.py           # Dataset handling for training
├── training_final.py    # Training script
├── denosiedaudiofinal.py # Inference and audio processing
├── unet_anc_model.py    # Model architecture (required)
└── README.md
```

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ANC-M1.git
cd ANC-M1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For inference (denoising audio):
```bash
python denosiedaudiofinal.py
```
This will:
- Record 10 seconds of audio
- Process it through the model
- Save the denoised version as 'denoised_audio.wav'

4. For training a new model:
```bash
python training_final.py
```

## Training

To train the model on your own dataset:

1. Prepare your dataset in the following structure:
```
dataset/
├── clean_trainset_28spk_wav/  # Clean audio files
└── pnoisy_trainset_28spk_wav/ # Noisy audio files
```

2. Update the paths in `training_final.py`:
```python
noisy_folder = "path/to/noisy/files"
clean_folder = "path/to/clean/files"
```

3. Run the training script:
```bash
python training_final.py
```

## Model Architecture

The model uses a UNet architecture optimized for audio processing. Key features:
- Input: Raw audio waveform
- Output: Denoised audio waveform
- Processing: Performed in chunks of 32000 samples with 1600 sample overlap

## Performance

- Sampling rate: 16kHz
- Chunk size: 32000 samples (~2 seconds)
- Overlap: 1600 samples
- Supports both CPU and GPU inference

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/new-feature`
5. Submit a pull request


