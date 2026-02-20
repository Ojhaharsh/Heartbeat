# 🎤 Heartbeat

**Native C++17 Inference Engine for Kokoro-82M Text-to-Speech**

Heartbeat is a high-performance, standalone TTS engine that runs Kokoro-82M entirely in C++ with no Python runtime dependencies. Built on GGML for tensor operations and custom ISTFT for audio synthesis.

## ✨ Features

- ⚡ **Fast**: <200ms latency for 5-second sentences on AVX2 CPUs
- 🎯 **Portable**: Single GGUF model file, no external dependencies at runtime
- 🔊 **High Quality**: 24kHz audio output using ISTFTNet vocoder
- 🌍 **Multi-Voice**: American English, Indian English, and more

## 🚀 Quick Start

### 1. Set Up Dependencies

```powershell
# Windows (PowerShell as Administrator)
.\scripts\setup_dependencies.ps1
```

This installs:
- **espeak-ng** - Text-to-phoneme conversion
- **GGML** - Tensor operations library
- **KissFFT** - Fast Fourier Transform
- **Python packages** - For model export

### 2. Download & Export Model

```bash
# Download Kokoro-82M from Hugging Face
python scripts/download_model.py

# Convert to GGUF format
python scripts/export_kokoro.py
```

### 3. Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### 4. Run

```bash
./heartbeat --text "Hello, world!" --voice af --output hello.wav
```

## 🎭 Available Voices

| Voice Code | Description |
|------------|-------------|
| `af` | American Female |
| `am` | American Male |
| `bf` | British Female |
| `bm` | British Male |
| `in_f` | Indian Female |
| `in_m` | Indian Male |

## 📖 Usage

```bash
# Basic synthesis
heartbeat --text "Welcome to Heartbeat!" --output welcome.wav

# Specify voice
heartbeat --text "नमस्ते" --voice in_f --output namaste.wav

# Benchmark mode
heartbeat --benchmark --text "Performance test sentence."
```

## 🏗️ Architecture

```
Text → Phonemizer (espeak-ng) → PL-BERT Encoder → Duration Predictor
                                      ↓
    WAV ← ISTFT ← ISTFTNet Decoder ← Length Regulator ← Style Vector
```

## 📁 Project Structure

```
Heartbeat/
├── extern/           # Third-party libraries
│   ├── ggml/         # Tensor operations
│   └── kissfft/      # FFT library
├── models/           # Model files (.pth, .gguf)
├── scripts/          # Python utilities
├── include/          # C++ headers
├── src/              # C++ implementation
└── tests/            # Unit tests
```

## 🤝 Credits

- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) - The original model
- [GGML](https://github.com/ggerganov/ggml) - Tensor library by Georgi Gerganov
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) - Text-to-phoneme engine
- [StyleTTS2](https://github.com/yl4579/StyleTTS2) - Original architecture

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.
