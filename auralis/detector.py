from dataclasses import dataclass
from pathlib import Path
import queue
import threading

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

from .model import AudioModel


@dataclass(frozen=True)
class AudioResult:
    class_name: str
    confidence: float


class Detector:
    def __init__(self, model_path=None, confidence=None, microphone_device=None, stride_ms=50):
        self.model = AudioModel(model_path)
        self.classes = self.model.classes
        self.sample_rate = self.model.sample_rate
        self.window_samples = self.model.window_samples
        self.confidence = confidence
        self.microphone_device = microphone_device
        self.stride_samples = max(1, int(self.sample_rate * stride_ms / 1000))

        self._stream = None
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._queue = queue.Queue(maxsize=2)
        self._buffer = np.zeros(self.window_samples, dtype=np.float32)
        self._received = 0
        self._since_prediction = 0

    def _center_crop_or_pad(self, audio):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        if audio.size < self.window_samples:
            total = self.window_samples - audio.size
            left = total // 2
            right = total - left
            return np.pad(audio, (left, right))

        if audio.size == self.window_samples:
            return audio

        kernel = max(1, int(self.sample_rate * 0.005))
        usable = (audio.size // kernel) * kernel
        energy = np.mean(np.abs(audio[:usable]).reshape(-1, kernel), axis=1)
        center = int(np.argmax(energy) * kernel + kernel // 2)
        start = center - self.window_samples // 2
        start = max(0, min(start, audio.size - self.window_samples))
        return audio[start:start + self.window_samples]

    def _resample(self, audio, sample_rate):
        if sample_rate == self.sample_rate:
            return audio
        return librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=self.sample_rate,
            res_type="kaiser_best",
        ).astype(np.float32, copy=False)

    def _mel(self, audio):
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.model.n_fft,
            win_length=self.model.win_length,
            hop_length=self.model.hop_length,
            n_mels=self.model.n_mels,
            fmin=self.model.f_min,
            fmax=self.model.f_max,
            power=2.0,
            center=True,
            pad_mode="reflect",
            window="hann",
            htk=True,
            norm="slaney",
        )

        reference = np.max(mel)
        mel = np.maximum(mel, 1e-10)
        if reference <= 1e-10:
            db = 10.0 * np.log10(mel / 1e-10)
        else:
            db = 10.0 * np.log10(mel / reference)
        db = np.maximum(db, db.max() - self.model.top_db)

        mean = db.mean()
        std = max(float(db.std()), 1e-6)
        normalized = (db - mean) / std
        return normalized.astype(np.float32)[None, None, :, :]

    def _preprocess(self, audio, sample_rate):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        audio = self._resample(audio, sample_rate)
        audio = self._center_crop_or_pad(audio)
        return self._mel(audio)

    def predict(self, audio, sample_rate=16000):
        model_input = self._preprocess(audio, sample_rate)
        index, probabilities = self.model.predict(model_input)
        return AudioResult(
            class_name=self.classes[index],
            confidence=float(probabilities[index]),
        )

    def predict_file(self, path):
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        return self.predict(audio, sample_rate)

    def _enqueue(self, audio):
        try:
            self._queue.put_nowait(audio)
            return
        except queue.Full:
            pass

        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._queue.put_nowait(audio)
        except queue.Full:
            pass

    def _audio_callback(self, indata, frames, time_info, status):
        chunk = np.asarray(indata[:, 0], dtype=np.float32).copy()
        length = len(chunk)
        if length == 0:
            return

        if length >= self.window_samples:
            self._buffer[:] = chunk[-self.window_samples:]
        else:
            self._buffer[:-length] = self._buffer[length:]
            self._buffer[-length:] = chunk

        self._received += length
        self._since_prediction += length

        if self._received >= self.window_samples and self._since_prediction >= self.stride_samples:
            self._since_prediction = 0
            self._enqueue(self._buffer.copy())

    def _worker(self, callback):
        while not self._stop_event.is_set():
            try:
                audio = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue

            try:
                result = self.predict(audio, self.sample_rate)
                if self.confidence is None or result.confidence >= self.confidence:
                    callback(result)
            except Exception:
                continue

    def start(self, callback):
        if not callable(callback):
            raise TypeError("callback must be callable")

        if self._stream is not None:
            return

        self._stop_event.clear()
        self._queue = queue.Queue(maxsize=2)
        self._buffer.fill(0)
        self._received = 0
        self._since_prediction = 0

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=512,
            device=self.microphone_device,
            callback=self._audio_callback,
        )

        try:
            self._stream.start()
            self._worker_thread = threading.Thread(
                target=self._worker,
                args=(callback,),
                daemon=True,
            )
            self._worker_thread.start()
        except Exception:
            self._stream.close()
            self._stream = None
            raise

    def stop(self):
        self._stop_event.set()

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._worker_thread is not None:
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None
