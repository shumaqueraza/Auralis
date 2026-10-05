# 🎧 Auralis

**A 96 KB neural net that knows when you clap, snap, type, knock, whistle, or talk. Runs on your CPU, nothing leaves your machine.**

> "Wherever we are, what we hear is mostly noise. When we ignore it, it disturbs us. When we listen to it, we find it fascinating."
>
> John Cage

---

## ❓ What is this?

Auralis is a small Python library that listens to your microphone and tells you what it hears. Ten sound classes, one tiny CNN exported to ONNX, zero cloud.

It watches a 0.4 second window of audio on a background thread and calls your function every time it recognizes something. Stop it and the thread is gone. No system service, no leftover process, no network calls.

The model is 96 KB. The whole package is a folder you can copy into any project.

---

## 👂 What can it hear?

- 😶 **background** (quiet room, fan noise, silence)
- 👏 **clap**
- 👆 **finger_tap** (tapping a desk or a screen)
- ⌨️ **keyboard_mouse** (typing and clicking)
- 🔑 **keys** (keychain jingle)
- 🚪 **knock**
- 🖊️ **pen_click** (clicky pen)
- 🫰 **snaps**
- 🗣️ **speech**
- 🎵 **whistle**

---

## 🤔 Why?

Because "always listening" usually means "always uploading". Auralis goes the other way. Audio is processed on your machine and thrown away, and the weights are open, so you can inspect the model instead of trusting it.

> "The quieter you become, the more you are able to hear."
>
> Rumi

Use it for sound reactive UIs, desktop toys, accessibility tools, automations, or to find out how much of your day is keyboard clicks.

---

## 🧰 Tech Stack

| Layer | What |
|---|---|
| Model | TinyCNN, 10 classes, exported to ONNX (96 KB) |
| Inference | onnxruntime, CPU provider |
| Audio input | sounddevice for the microphone, soundfile for files |
| Preprocessing | librosa (resampling, mel spectrogram) |
| Math | numpy |
| Dashboard demo | `test.py`, standard library HTTP server with server sent events |
| Runtime | Python 3.11+ |

---

## 🧠 The Model

`auralis/model/auralis.onnx` is the TinyCNN checkpoint exported for inference. `auralis/model/metadata.json` sits next to it and describes everything the library needs to preprocess audio the same way the model was trained:

| Setting | Value |
|---|---|
| Sample rate | 16,000 Hz |
| Window | 0.4 s (6,400 samples) |
| Mel bins | 64 |
| n_fft / win_length / hop_length | 512 / 400 / 160 |
| f_min / f_max | 50 / 8,000 Hz |
| top_db | 80 |
| Input shape | 1 x 1 x 64 x 41 |
| Output shape | 1 x 10 |

If you feed it a file or a numpy array, `predict()` handles resampling, cropping, the mel spectrogram, dB scaling, and normalization for you.

---

## 📦 Drop It In Your Project

Auralis is meant to be copied. Clone the repo or download the ZIP, then move the `auralis/` folder (code plus `model/`) into your project.

### Prerequisites

- Python 3.11 or newer
- A microphone if you want live audio
- numpy, onnxruntime, librosa, sounddevice, soundfile

### Quick Start

```bash
# 1. clone the repo (or download the ZIP and unpack it)
git clone https://github.com/shumaqueraza/Auralis.git

# 2. copy the package into your project
cp -r Auralis/auralis your-project/

# 3. install the dependencies
pip install numpy onnxruntime librosa sounddevice soundfile

# or skip the copying and install straight from git
pip install "git+https://github.com/shumaqueraza/Auralis.git"
```

### 🎤 Live microphone

```python
from auralis import Detector

detector = Detector()

def on_result(result):
    print(result.class_name, result.confidence)

detector.start(on_result)
```

The callback runs on a background thread, so your app keeps doing its thing. When you are done:

```python
detector.stop()
```

`example.py` in this repo is exactly that, 15 lines total.

### 📁 External audio

```python
result = detector.predict_file("clap.wav")
print(result.class_name, result.confidence)

# or pass raw samples you already have
result = detector.predict(samples, sample_rate=44100)
```

Both return an `AudioResult` with `class_name` and `confidence`.

### 🎚️ Tuning

```python
detector = Detector(confidence=0.85)        # ignore predictions below 85%
detector = Detector(stride_ms=100)          # predict every 100 ms instead of 50 ms
detector = Detector(microphone_device=2)    # pick a specific input device
```

### 🖥️ Live dashboard

`test.py` is a single file demo that serves a live web dashboard with a big class readout, a confidence trace, and a detection log. It opens your browser for you.

```bash
python test.py                  # microphone mode
python test.py --demo           # fake detections, no mic needed
python test.py --confidence 0.7
python test.py --port 9000 --no-browser
```

Ctrl+C stops the mic and the server together.

---

## ⚖️ License

MIT, and the weights are open. See [LICENSE](LICENSE).

---

## ⚠️ Disclaimer

Auralis classifies live sound. It does not record, save, or upload your audio. That said, if you point it at other people, be upfront about it.

> "There is no such thing as silence. Something is always happening that makes a sound."
>
> John Cage
