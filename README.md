# ANC-M1: Deep Learning-Based Active Noise Cancellation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

*A real-time audio denoising system utilizing deep learning, featuring a custom UNet architecture designed for quality noise reduction*

[Features](#-key-features) • [Installation](#️-installation) • [Usage](#-usage) • [Architecture](#-software-architecture) • [Demo](#-demo)

</div>

---

## 🌟 Key Features

- **🎯 Real-Time Processing**: Live audio recording and denoising capabilities with minimal latency
- **🧠 Advanced Architecture**: Custom UNet design optimized for 1D audio signal processing
- **⚡ Flexible Deployment**: Supports both CPU and GPU inference with automatic device detection
- **🔊 Seamless Audio**: Processes audio in overlapping chunks for artifact-free output
- **📊 Complete Pipeline**: Includes training, inference, evaluation, and visualization scripts
- **🎨 Modern Web Interface**: Beautiful, responsive frontend for easy interaction
- **📈 Training Monitoring**: Real-time loss tracking with early stopping and learning rate scheduling

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Sampling Rate** | 16 kHz |
| **Processing Latency** | ~100ms |
| **Chunk Size** | 32,000 samples (2 seconds) |
| **Overlap** | 1,600 samples (10%) |
| **Model Parameters** | ~2.5M trainable |
| **Model Size** | ~10 MB |
| **Platform Support** | Windows, Linux, macOS |

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for training acceleration)
- 4GB+ RAM recommended

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/aadi611/ANC-M1.git
cd ANC-M1
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify installation:**
```bash
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA Available: {torch.cuda.is_available()}')"
```

---

## 💡 Usage

### Quick Start - Audio Denoising

```python
from denosiedaudiofinal import process_audio, load_model
import soundfile as sf

# Load the trained model
model, device = load_model('best_model.pth')

# Denoise an audio file
denoised_audio = process_audio(model, "noisy_audio.wav", device)

# Save the result
sf.write("clean_audio.wav", denoised_audio, 16000)
```

### Real-time Recording & Denoising

```bash
python denosiedaudiofinal.py
```

This will:
1. Record 15 seconds of audio from your microphone
2. Process the audio through the UNet model
3. Save denoised output as `denoised_output.wav`
4. Generate comparison plots

### Training a New Model

```bash
python training_final.py
```

**Training Configuration:**
- Modify `DATASET_PATHS` in `training_final.py` to point to your dataset
- Adjust hyperparameters (learning rate, batch size, epochs)
- Monitor training progress in real-time
- Best model automatically saved based on validation loss

### Using the Web Interface

1. Open `audio_denoiser_frontend.html` in a web browser
2. Choose between:
   - **Upload Tab**: Drag & drop audio files
   - **Record Tab**: Record directly from microphone
3. Click "Denoise Audio" to process
4. Download the cleaned audio

---

## 🏗️ Software Architecture Document (SAD)

### System Overview

ANC-M1 is a modular audio denoising system built on PyTorch, implementing a UNet-based encoder-decoder architecture for real-time noise suppression.

```
┌─────────────────────────────────────────────────────────────┐
│                     ANC-M1 System Architecture               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐      ┌──────────────┐      ┌────────────┐ │
│  │   Input     │─────▶│  Preprocessing│─────▶│  UNet      │ │
│  │   Audio     │      │  (Chunking)   │      │  Model     │ │
│  └─────────────┘      └──────────────┘      └────────────┘ │
│                                                    │          │
│                                                    ▼          │
│  ┌─────────────┐      ┌──────────────┐      ┌────────────┐ │
│  │  Denoised   │◀─────│ Postprocessing│◀─────│  Inference │ │
│  │   Output    │      │  (Stitching)  │      │  Engine    │ │
│  └─────────────┘      └──────────────┘      └────────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Architecture Components

#### 1. **Data Layer** (`dataset.py`)

**Responsibilities:**
- Load paired noisy-clean audio files
- Handle audio preprocessing (resampling, normalization)
- Implement data augmentation for training
- Provide batched data to training pipeline

**Key Classes:**
```python
AudioDataset(Dataset)
├── __init__(clean_folder, noisy_folder, sr, target_length)
├── __len__()
├── __getitem__(idx)
└── _process_audio(audio)
```

**Features:**
- Automatic file pair matching
- Dynamic padding/cropping to target length
- Mono audio conversion
- Path validation with meaningful errors

---

#### 2. **Model Layer** (`unet_anc_model.py`)

**Architecture: UNet for 1D Audio Signals**

```
Input: (batch, 1, 32000)
    │
    ├─[Encoder 1]─> (batch, 64, 32000)  ────┐
    │                                         │
    ├─[Encoder 2]─> (batch, 128, 16000) ──┐ │
    │                                       │ │
    ├─[Encoder 3]─> (batch, 256, 8000) ─┐ │ │
    │                                     │ │ │
    ├─[Encoder 4]─> (batch, 512, 4000) ─┤ │ │
    │                                    │ │ │ │
    └─[Bottleneck]─> (batch, 512, 2000) │ │ │ │
                            │            │ │ │ │
                    [Upsample + Skip]◄───┘ │ │ │
                            │              │ │ │
                    [Decoder 3]◄───────────┘ │ │
                            │                │ │
                    [Decoder 2]◄─────────────┘ │
                            │                  │
                    [Decoder 1]◄───────────────┘
                            │
                    [Output Conv]
                            │
Output: (batch, 1, 32000) ──┘
```

**Key Components:**

```python
UNetANC(nn.Module)
├── Encoder Blocks (4 levels)
│   ├── DoubleConv (Conv1d → BatchNorm → LeakyReLU → Dropout)
│   └── MaxPool1d (downsampling)
│
├── Bottleneck
│   └── DoubleConv (feature extraction at lowest resolution)
│
├── Decoder Blocks (4 levels)
│   ├── Upsample (linear interpolation)
│   ├── Skip Connection (concatenation)
│   └── DoubleConv (reconstruction)
│
└── Output Layer
    ├── Conv1d (channel reduction)
    └── Tanh (output normalization to [-1, 1])
```

**Design Rationale:**
- **Skip Connections**: Preserve high-frequency details lost in downsampling
- **LeakyReLU**: Prevent dying ReLU problem, better gradient flow
- **BatchNorm**: Stabilize training, allow higher learning rates
- **Dropout**: Regularization to prevent overfitting
- **1D Convolutions**: Optimized for temporal audio data

---

#### 3. **Training Layer** (`training_final.py`)

**Training Pipeline:**

```
Dataset Loading
    ↓
Data Splitting (80/20)
    ↓
DataLoader Creation
    ↓
Model Initialization
    ↓
┌─────────────────────────────────┐
│    Training Loop (per epoch)    │
│                                  │
│  1. Forward Pass                │
│  2. Loss Calculation (MSE)      │
│  3. Backward Pass               │
│  4. Gradient Clipping           │
│  5. Optimizer Step              │
│  6. Validation                  │
│  7. Learning Rate Scheduling    │
│  8. Checkpoint Saving           │
│  9. Early Stopping Check        │
│ 10. Progress Visualization      │
└─────────────────────────────────┘
    ↓
Best Model Selection
```

**Key Features:**
- **Early Stopping**: Prevents overfitting (patience=5, min_delta=1e-4)
- **Learning Rate Scheduling**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Gradient Clipping**: Prevents exploding gradients (max_norm=1.0)
- **Checkpoint Management**: Saves best model based on validation loss
- **Real-time Monitoring**: Loss curves plotted every epoch

**Hyperparameters:**
```python
{
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 50,
    'weight_decay': 1e-5,
    'optimizer': 'Adam',
    'loss_function': 'MSELoss'
}
```

---

#### 4. **Inference Layer** (`denosiedaudiofinal.py`)

**Processing Pipeline:**

```
Audio Input (any length)
    ↓
Chunk Splitting (32000 samples)
    ↓
┌────────────────────────┐
│  For each chunk:       │
│  1. Normalize          │
│  2. To Tensor          │
│  3. Model Forward      │
│  4. Post-process       │
└────────────────────────┘
    ↓
Chunk Stitching (overlap handling)
    ↓
Trim to Original Length
    ↓
Denoised Output
```

**Key Functions:**

```python
# Model loading with error handling
load_model(model_path, device) -> (model, device)

# Audio processing in chunks
process_audio(model, audio_file, device, chunk_size, sr) -> np.ndarray

# Recording from microphone
record_audio(filename, duration, sr) -> None

# Visualization
plot_audio_comparison(noisy, denoised, sr, save_path) -> None
```

---

#### 5. **Presentation Layer** (`audio_denoiser_frontend.html`)

**User Interface Architecture:**

```
┌──────────────────────────────────────────────────────┐
│               Web Interface (HTML5)                   │
├──────────────────────────────────────────────────────┤
│                                                        │
│  ┌─────────────┐          ┌──────────────┐          │
│  │  Upload Tab │          │  Record Tab  │          │
│  ├─────────────┤          ├──────────────┤          │
│  │ - Drag/Drop │          │ - Mic Access │          │
│  │ - Preview   │          │ - Timer      │          │
│  │ - Waveform  │          │ - Controls   │          │
│  └─────────────┘          └──────────────┘          │
│         │                         │                   │
│         └─────────┬───────────────┘                  │
│                   ▼                                   │
│         ┌──────────────────┐                         │
│         │  Audio Processor  │                         │
│         │  (JavaScript)     │                         │
│         └──────────────────┘                         │
│                   │                                   │
│                   ▼                                   │
│         ┌──────────────────┐                         │
│         │  Results Display  │                         │
│         │  - Comparison     │                         │
│         │  - Download       │                         │
│         └──────────────────┘                         │
│                                                        │
└──────────────────────────────────────────────────────┘
```

**Features:**
- Responsive design (mobile & desktop)
- Real-time waveform visualization
- Drag-and-drop file upload
- Microphone recording with timer
- Side-by-side audio comparison
- One-click download

---

### Data Flow Diagram

```
┌─────────────┐
│ Raw Audio   │
│ (Noisy)     │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Preprocessing       │
│ - Load (librosa)    │
│ - Resample (16kHz)  │
│ - Normalize         │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Chunking            │
│ - Split into 32k    │
│ - Add padding       │
│ - Convert to tensor │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Model Inference     │
│ - Encoder           │
│ - Bottleneck        │
│ - Decoder           │
│ - Skip connections  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Postprocessing      │
│ - Concatenate       │
│ - Trim to length    │
│ - Denormalize       │
└──────┬──────────────┘
       │
       ▼
┌─────────────┐
│ Clean Audio │
│ (Denoised)  │
└─────────────┘
```

---

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Deep Learning** | PyTorch 2.0+ | Model implementation & training |
| **Audio Processing** | librosa, soundfile | Audio I/O and manipulation |
| **Numerical Computing** | NumPy | Array operations |
| **Visualization** | Matplotlib | Training curves & spectrograms |
| **Frontend** | HTML5, CSS3, JavaScript | Web interface |
| **Recording** | sounddevice | Microphone input |
| **Data Loading** | torch.utils.data | Efficient batching |

---

### Design Patterns

#### 1. **Module Pattern**
Each component (dataset, model, training, inference) is self-contained with clear interfaces.

#### 2. **Factory Pattern**
```python
def load_model(model_path, device=None):
    """Factory function for model instantiation"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetANC().to(device)
    # ... load weights
    return model, device
```

#### 3. **Strategy Pattern**
Different audio processing strategies (chunk-based, streaming) can be swapped.

#### 4. **Observer Pattern**
Training callbacks for early stopping, learning rate scheduling, and checkpointing.

---

### Error Handling

**Robust error handling at every layer:**

```python
# Input validation
if not Path(audio_file).exists():
    raise FileNotFoundError(f"Audio file not found: {audio_file}")

# Device compatibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Graceful degradation
try:
    # GPU processing
except RuntimeError as e:
    # Fallback to CPU
```

---

### Performance Optimization

1. **Batch Processing**: Process multiple audio chunks simultaneously
2. **GPU Acceleration**: Automatic CUDA utilization when available
3. **Memory Management**: Chunk-based processing for large files
4. **Mixed Precision**: Optional FP16 training for 2x speedup
5. **Pin Memory**: Faster data transfer between CPU and GPU
6. **Gradient Accumulation**: Handle larger effective batch sizes

---

### Scalability Considerations

**Current Limitations:**
- Single-threaded inference
- No distributed training support
- Fixed sampling rate (16kHz)

**Future Enhancements:**
- Multi-GPU training with DistributedDataParallel
- REST API with FastAPI for remote inference
- Docker containerization
- Real-time streaming support with WebRTC
- Support for multiple sampling rates
- Model quantization for edge deployment

---

## 📁 Project Structure

```
ANC-M1/
├── 📜 dataset.py                  # Dataset handling and preprocessing
├── 🎯 training_final.py           # Training pipeline with monitoring
├── 🎤 denosiedaudiofinal.py       # Inference engine and recording
├── 🧠 unet_anc_model.py           # UNet model architecture
├── 🎨 audio_denoiser_frontend.html # Web interface
├── 📋 requirements.txt            # Python dependencies
├── 📊 best_model.pth              # Trained model checkpoint
├── 📈 training_progress.png       # Loss curves
└── 📖 README.md                   # Documentation
```

---

## 🚀 Demo

### Command Line Interface

```bash
# Record and denoise
$ python denosiedaudiofinal.py
Recording for 15 seconds...
✓ Recording saved as recorded_noisy.wav
Processing audio...
✓ Model loaded on cuda
✓ Denoised audio saved as denoised_output.wav
✓ Comparison plot saved as audio_comparison.png
```

### Python API

```python
from denosiedaudiofinal import process_audio, load_model
import soundfile as sf

# Load model
model, device = load_model('best_model.pth')

# Process audio
denoised = process_audio(model, "noisy.wav", device)

# Save result
sf.write("clean.wav", denoised, 16000)
```

### Web Interface

1. Open `audio_denoiser_frontend.html`
2. Upload or record audio
3. Click "Denoise Audio"
4. Compare and download results

---

## 📈 Training Your Own Model

### Dataset Preparation

```
dataset/
├── clean/
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ...
└── noisy/
    ├── audio_001.wav
    ├── audio_002.wav
    └── ...
```

### Training Configuration

Edit `training_final.py`:

```python
DATASET_PATHS = {
    'clean_testset': '/path/to/clean',
    'noisy_dataset': '/path/to/noisy'
}

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
```

### Start Training

```bash
python training_final.py
```

**Expected Output:**
```
Device: cuda
GPU: NVIDIA GeForce RTX 3080
✓ Dataset loaded: 5000 samples
Train samples: 4000, Val samples: 1000
✓ Model parameters: 2,547,201

============================================================
Epoch 1/50
============================================================
Epoch 1 [100/125] Loss: 0.023456
...
Validation loss improved: 0.034567 → 0.028901
✓ Checkpoint saved: best_model.pth

Epoch 1 Summary:
  Train Loss: 0.028234
  Val Loss:   0.028901
  LR:         0.001000
```

---

## 🔧 Advanced Configuration

### Custom Model Architecture

```python
# Modify unet_anc_model.py
model = UNetANC(
    in_channels=1,
    base_channels=64,  # Increase for more capacity
    dropout=0.2        # Adjust regularization
)
```

### Custom Loss Function

```python
# In training_final.py
criterion = torch.nn.L1Loss()  # MAE instead of MSE
# or
from torch.nn import MultiTaskLoss
criterion = CustomSpectralLoss()  # Frequency-domain loss
```

### Data Augmentation

```python
# In dataset.py
def __getitem__(self, idx):
    noisy, clean = self.load_audio_pair(idx)
    
    # Add augmentation
    if self.augment:
        noisy = add_gaussian_noise(noisy, snr=random.uniform(0, 20))
        noisy, clean = random_time_shift(noisy, clean)
    
    return noisy, clean
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Reporting Bugs

1. Check existing issues
2. Create detailed bug report with:
   - System information
   - Steps to reproduce
   - Expected vs actual behavior
   - Error logs

### Suggesting Enhancements

1. Open an issue with `[Feature Request]` tag
2. Describe the enhancement
3. Explain use case and benefits

### Pull Requests

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes with clear commit messages
4. Add tests if applicable
5. Update documentation
6. Push to branch: `git push origin feature/amazing-feature`
7. Open Pull Request with detailed description

### Code Style

- Follow PEP 8 for Python code
- Use type hints
- Add docstrings to functions
- Keep functions focused and modular

---

## 📄 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2024 Aadityan Gupta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📬 Contact & Support

<div align="center">

**Aadityan Gupta**

[![Email](https://img.shields.io/badge/Email-ag260%40snu.edu.in-blue?style=flat-square&logo=gmail)](mailto:ag260@snu.edu.in)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Aadityan%20Gupta-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/aadityangupta/)
[![GitHub](https://img.shields.io/badge/GitHub-aadi611-181717?style=flat-square&logo=github)](https://github.com/aadi611)

</div>

### Getting Help

- 📖 **Documentation**: Read this README thoroughly
- 🐛 **Bug Reports**: [Open an issue](https://github.com/aadi611/ANC-M1/issues)
- 💡 **Feature Requests**: [Start a discussion](https://github.com/aadi611/ANC-M1/discussions)
- 💬 **Questions**: [Ask in Discussions](https://github.com/aadi611/ANC-M1/discussions/categories/q-a)

---

## 🙏 Acknowledgments

Special thanks to:

- **PyTorch Team** for the incredible deep learning framework
- **librosa** developers for audio processing tools
- **Open-source community** for inspiration and support
- **Contributors** who help improve this project
- **Shiv Nadar University** for providing research resources

### Citations

If you use this project in your research, please cite:

```bibtex
@software{anc_m1_2024,
  author = {Gupta, Aadityan},
  title = {ANC-M1: Deep Learning-Based Active Noise Cancellation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/aadi611/ANC-M1}
}
```

---

## 📊 Project Stats

<div align="center">

![Star History Chart](https://api.star-history.com/svg?repos=aadi611/ANC-M1&type=Date)

[![GitHub stars](https://img.shields.io/github/stars/aadi611/ANC-M1?style=social)](https://github.com/aadi611/ANC-M1/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/aadi611/ANC-M1?style=social)](https://github.com/aadi611/ANC-M1/network/members)
[![GitHub watchers](https://img.shields.io/github/watchers/aadi611/ANC-M1?style=social)](https://github.com/aadi611/ANC-M1/watchers)

</div>

---

## 🗺️ Roadmap

### Version 1.0 (Current)
- [x] Basic UNet architecture
- [x] Training pipeline
- [x] Real-time inference
- [x] Web interface
- [x] Documentation

### Version 1.1 (Planned)
- [ ] REST API with FastAPI
- [ ] Real-time streaming support
- [ ] Multiple sampling rates
- [ ] Advanced loss functions
- [ ] Model ensemble

### Version 2.0 (Future)
- [ ] Transformer-based architecture
- [ ] Multi-speaker separation
- [ ] Docker deployment
- [ ] Cloud integration
- [ ] Mobile app

---

<div align="center">

**⭐ Star this repo if you find it helpful! ⭐**

Made with ❤️ by [Aadityan Gupta](https://github.com/aadi611)

</div>
