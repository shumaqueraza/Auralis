from pathlib import Path
import json
import numpy as np
import onnxruntime as ort


class AudioModel:
    def __init__(self, model_path=None):
        package_dir = Path(__file__).resolve().parent
        model_dir = package_dir / "model"
        self.model_path = Path(model_path) if model_path else model_dir / "auralis.onnx"
        self.metadata_path = model_dir / "metadata.json"

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Auralis model not found: {self.model_path}\n"
                "Place auralis.onnx and metadata.json in auralis/model/."
            )

        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"Auralis metadata not found: {self.metadata_path}"
            )

        with self.metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.classes = tuple(metadata["classes"])
        self.sample_rate = int(metadata["sample_rate"])
        self.window_samples = int(metadata["window_samples"])
        self.n_mels = int(metadata["n_mels"])
        self.n_fft = int(metadata["n_fft"])
        self.win_length = int(metadata["win_length"])
        self.hop_length = int(metadata["hop_length"])
        self.f_min = float(metadata["f_min"])
        self.f_max = float(metadata["f_max"])
        self.top_db = float(metadata["top_db"])

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Auralis expects an ONNX model with one input and one output.")

        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

    def predict(self, model_input):
        logits = np.asarray(
            self.session.run(
                [self.output_name],
                {self.input_name: np.asarray(model_input, dtype=np.float32)},
            )[0][0],
            dtype=np.float32,
        )

        logits -= np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities)
        index = int(np.argmax(probabilities))
        return index, probabilities
