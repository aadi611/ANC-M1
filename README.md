# ANC-M1: Deep Learning-Based Active Noise Cancellation



A real-time audio denoising system utilizing deep learning, featuring a custom UNet architecture designed for high-quality noise reduction

## 🌟 Key Features

- **Real-Time Processing**: Live audio recording and denoising capabilities
- **Advanced Architecture**: Custom UNet design optimized for audio signal processing
- **Flexible Deployment**: Supports both CPU and GPU inference
- **Seamless Audio**: Processes audio in overlapping chunks for artifact-free output
- **Complete Pipeline**: Includes training, inference, and evaluation scripts

## 🚀 Quick Demo

```python
from denosiedaudiofinal import denoise_audio
import soundfile as sf

# Load and denoise audio
denoised_audio = denoise_audio("input.wav")
sf.write("denoised.wav", denoised_audio, 16000)
```

## 📊 Performance

- **Sampling Rate**: 16kHz
- **Latency**: ~100ms for real-time processing
- **Chunk Processing**: 32000 samples with 1600 sample overlap
- **Platform Support**: Tested on Windows, Linux, and macOS

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/aadi611/ANC-M1.git
cd ANC-M1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💡 Usage

### Real-time Denoising
```bash
python denosiedaudiofinal.py
```

### Training New Model
```bash
python training_final.py
```

## 📁 Project Structure

```
ANC-M1/
├── 📜 dataset.py           # Dataset handling
├── 🎯 training_final.py    # Training pipeline
├── 🎤 denosiedaudiofinal.py # Inference engine
├── 🧠 unet_anc_model.py    # Model architecture
└── 📖 README.md
```

## 🔧 Model Architecture

The ANC-M1 uses a specialized UNet architecture:

- **Input Layer**: Raw audio waveform processing
- **Encoder**: Multi-scale feature extraction
- **Decoder**: Progressive upsampling with skip connections
- **Output**: Clean audio reconstruction

## 📈 Training Process

1. Prepare paired noisy-clean audio dataset
2. Configure training parameters in `training_final.py`
3. Run training script
4. Monitor progress through console outputs

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -am 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📬 Contact

- **Author**: [Aadityan Gupta]
- **Email**: [ag260@snu.edu.in]
- **LinkedIn**: [https://www.linkedin.com/in/aadityangupta/]

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=aadi611/ANC-M1&type=Date)](https://star-history.com/#aadi611/ANC-M1&Date)

## 🙏 Acknowledgments

Special thanks to all contributors and the open-source community.
