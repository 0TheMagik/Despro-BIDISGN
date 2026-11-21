from __future__ import annotations

import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "deployment_model"

# Urutan label HARUS sama dengan urutan folder dataset (sorted) saat training.
# Model .pt hasil training menyimpan daftarnya sendiri; daftar ini hanya fallback.
FALLBACK_LABELS = [
    "A", "Adik", "Ajak", "Aku", "Anak", "Anda",
    "Apa", "B", "Baik", "Bantu", "Bapak", "Bayi",
    "Bekerja", "Belajar", "Berapa", "Berdiri", "Bodoh", "Boros",
    "C", "Cerewet", "D", "Darimana", "Datang", "Dimana",
    "E", "Egois", "F", "G", "Ganteng", "H",
    "Hallo", "Hari", "Hati-hati", "I", "Ibu", "Ingat",
    "J", "K", "Kakak", "Kakek", "Kalian", "Kamu",
    "Kapan", "Kemana", "Kenapa", "Keren", "Kereta", "Kerja",
    "Kita", "Kuliah", "L", "Lupa", "M", "Maaf",
    "Makan", "Mau", "Menonton", "Minum", "Mobil", "Motor",
    "N", "Nakal", "Nama", "Nenek", "Nunggu", "O",
    "Om", "P", "Pergi", "Perkenalkan", "Pintar", "Q",
    "R", "Ramah", "S", "Sabar", "Sama-sama", "Saudara",
    "Saya", "Sederhana", "Semua", "Senang", "Siapa", "Sombong",
    "Sudah", "T", "Terima Kasih", "Tidak", "Tidak Mau", "Tolong",
    "Tuli", "U", "V", "W", "X", "Y", "Z",
]


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def resolve_model_path() -> Path:
    explicit = os.getenv("MODEL_PATH")
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise RuntimeError(f"MODEL_PATH menunjuk ke file yang tidak ada: {path}")
        return path

    candidates = [
        MODEL_DIR / "model_cnn_lstm_isyarat.pt",
        MODEL_DIR / "model_cnn_lstm_isyarat.keras",
        Path("/app/model/deployment_model/model_cnn_lstm_isyarat.pt"),
        Path("/app/model/deployment_model/model_cnn_lstm_isyarat.keras"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError(
        "Model tidak ditemukan. Letakkan model_cnn_lstm_isyarat.pt (97 kelas) "
        f"di {MODEL_DIR} atau set env MODEL_PATH."
    )


def load_labels_sidecar(model_path: Path, expected: int | None = None) -> list[str] | None:
    candidates = []
    explicit = os.getenv("LABELS_PATH")
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(model_path.parent / "labels.json")
    candidates.append(model_path.with_suffix(".json"))

    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        labels = payload.get("labels") if isinstance(payload, dict) else payload
        if not labels:
            continue
        labels = [str(label) for label in labels]
        if expected is not None and len(labels) != expected:
            raise RuntimeError(
                f"labels.json berisi {len(labels)} label, tetapi model punya {expected} kelas. "
                "Pastikan isi labels.json sama dengan dataset training."
            )
        return labels
    return None


def resample_sequence(sequence: np.ndarray, target_frames: int) -> np.ndarray:
    """Resample linier, sama seperti resample_sequence() pada notebook training."""
    if sequence.shape[0] == target_frames:
        return sequence.astype(np.float32, copy=False)
    if sequence.shape[0] == 1:
        return np.repeat(sequence.astype(np.float32), target_frames, axis=0)

    source_positions = np.linspace(0.0, 1.0, sequence.shape[0])
    target_positions = np.linspace(0.0, 1.0, target_frames)
    resampled = np.empty((target_frames, sequence.shape[1]), dtype=np.float32)
    for feature_index in range(sequence.shape[1]):
        resampled[:, feature_index] = np.interp(
            target_positions,
            source_positions,
            sequence[:, feature_index],
        )
    return resampled


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """Normalisasi per frame (wrist-relative + skala), sama seperti training notebook."""
    normalized = sequence.reshape(sequence.shape[0], 2, 21, 3).copy()
    for hand_index in range(2):
        hand = normalized[:, hand_index]
        valid_frames = np.abs(hand).sum(axis=(1, 2)) > 0
        if not np.any(valid_frames):
            continue

        wrist = hand[:, :1, :].copy()
        hand -= wrist
        scale = np.linalg.norm(hand, axis=2).max(axis=1, keepdims=True)
        scale = np.where(scale > 1e-6, scale, 1.0)
        hand /= scale[:, :, None]
        normalized[~valid_frames, hand_index] = 0.0

    return normalized.reshape(sequence.shape).astype(np.float32, copy=False)


class CNNLSTM:
    """Arsitektur PyTorch dari notebook training (dibangun saat runtime)."""

    @staticmethod
    def build(torch, num_classes: int, features: int, **config: Any):
        nn = torch.nn

        class _CNNLSTM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv1d(features, config.get("filters_1", 64), kernel_size=3, padding="same"),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(config.get("dropout", 0.2)),
                    nn.Conv1d(
                        config.get("filters_1", 64),
                        config.get("filters_2", 64),
                        kernel_size=3,
                        padding="same",
                    ),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(config.get("dropout", 0.2)),
                )
                self.lstm = nn.LSTM(config.get("filters_2", 64), config.get("lstm_1", 64), batch_first=True)
                self.lstm_2 = nn.LSTM(config.get("lstm_1", 64), config.get("lstm_2", 64), batch_first=True)
                self.lstm_3 = nn.LSTM(config.get("lstm_2", 64), config.get("lstm_3", 64), batch_first=True)
                self.temporal_dropout = nn.Dropout(config.get("dropout", 0.2))
                self.classifier = nn.Sequential(
                    nn.Linear(config.get("lstm_3", 64), config.get("dense_1", 64)),
                    nn.ReLU(),
                    nn.Dropout(config.get("dropout", 0.2)),
                    nn.Linear(config.get("dense_1", 64), config.get("dense_2", 32)),
                    nn.ReLU(),
                    nn.Linear(config.get("dense_2", 32), num_classes),
                )

            def forward(self, inputs):
                features_seq = self.cnn(inputs.transpose(1, 2)).transpose(1, 2)
                features_seq, _ = self.lstm(features_seq)
                features_seq = self.temporal_dropout(torch.relu(features_seq))
                features_seq, _ = self.lstm_2(features_seq)
                features_seq = self.temporal_dropout(torch.relu(features_seq))
                features_seq, _ = self.lstm_3(features_seq)
                features_seq = self.temporal_dropout(torch.relu(features_seq))
                return self.classifier(features_seq[:, -1])

        return _CNNLSTM()


class Predictor:
    def __init__(self) -> None:
        self.ready = False
        self.backend = "-"
        self.model_path = "-"
        self.labels: list[str] = []
        self.seq_len = 90
        self.features = 126
        self.normalize = True
        self.torch = None
        self.model = None
        self._lock = threading.Lock()

    def load(self) -> None:
        path = resolve_model_path()
        self.model_path = str(path)
        if path.suffix.lower() == ".pt":
            self._load_torch(path)
        else:
            self._load_keras(path)
        self.ready = True
        print(
            f"[model] {self.backend} | {self.seq_len}x{self.features} | "
            f"{len(self.labels)} kelas | normalize={self.normalize} | {path}"
        )

    def _load_torch(self, path: Path) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "File model .pt memerlukan PyTorch. Install dengan `pip install torch` "
                "atau jalankan lewat Docker image yang sudah disiapkan."
            ) from exc

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            config = checkpoint.get("model_config") or {}
            labels = checkpoint.get("actions")
            seq_len = int(checkpoint.get("video_frames", 90))
            feats = int(checkpoint.get("features", 126))
            normalize = bool(checkpoint.get("normalize", env_bool("NORMALIZE_SEQUENCE", True)))
        else:  # state_dict polos
            state_dict = checkpoint
            config = {}
            labels = None
            seq_len = 90
            feats = 126
            normalize = env_bool("NORMALIZE_SEQUENCE", True)

        if not labels:
            labels = load_labels_sidecar(path)
        if not labels:
            labels = FALLBACK_LABELS

        num_classes = len(labels)
        self.torch = torch
        self.model = CNNLSTM.build(torch, num_classes=num_classes, features=feats, **config)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        threads = int(os.getenv("TORCH_NUM_THREADS", "0"))
        if threads > 0:
            torch.set_num_threads(threads)

        self.backend = "torch"
        self.labels = [str(label) for label in labels]
        self.seq_len = seq_len
        self.features = feats
        self.normalize = normalize

    def _load_keras(self, path: Path) -> None:
        try:
            import keras
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "File model .keras memerlukan TensorFlow/Keras. Install `pip install tensorflow` "
                "atau gunakan model .pt terbaru."
            ) from exc

        model = keras.models.load_model(str(path), compile=False)
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        output_shape = model.output_shape
        if isinstance(output_shape, list):
            output_shape = output_shape[0]

        seq_len = int(input_shape[1] or 90)
        feats = int(input_shape[2] or 126)
        num_classes = int(output_shape[-1])

        labels = load_labels_sidecar(path, expected=num_classes)
        if labels is None:
            if num_classes != len(FALLBACK_LABELS):
                raise RuntimeError(
                    f"Model .keras punya {num_classes} kelas tetapi tidak ada labels.json yang cocok. "
                    f"Buat {path.parent / 'labels.json'} berisi {{\"labels\": [...]}} sesuai urutan training."
                )
            labels = FALLBACK_LABELS

        self.backend = "keras"
        self.model = model
        self.labels = [str(label) for label in labels]
        self.seq_len = seq_len
        self.features = feats
        self.normalize = env_bool("NORMALIZE_SEQUENCE", False)

    def prepare(self, instance: list[list[float]]) -> np.ndarray:
        array = np.asarray(instance, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[0] == 0:
            raise HTTPException(status_code=400, detail="Setiap instance harus sequence 2D yang tidak kosong")

        if array.shape[1] != self.features:
            if array.shape[1] > self.features:
                array = array[:, : self.features]
            else:
                padding = np.zeros((array.shape[0], self.features - array.shape[1]), dtype=np.float32)
                array = np.concatenate([array, padding], axis=1)

        array = resample_sequence(array, self.seq_len)
        if self.normalize:
            array = normalize_sequence(array)
        return np.ascontiguousarray(array, dtype=np.float32)

    def predict(self, instances: list[list[list[float]]]) -> list[dict[str, Any]]:
        if not self.ready or self.model is None:
            raise HTTPException(status_code=503, detail="Model belum siap")

        prepared: list[np.ndarray] = []
        empty_flags: list[bool] = []
        for instance in instances:
            sequence = self.prepare(instance)
            empty_flags.append(bool(np.abs(sequence).sum() < 1e-6))
            prepared.append(sequence)

        batch = np.stack(prepared, axis=0)

        with self._lock:
            if self.backend == "torch":
                torch = self.torch
                with torch.inference_mode():
                    logits = self.model(torch.from_numpy(batch))
                    probabilities = torch.softmax(logits, dim=1).numpy()
            else:
                probabilities = np.asarray(self.model.predict(batch, verbose=0), dtype=np.float32)

        results: list[dict[str, Any]] = []
        for index, is_empty in enumerate(empty_flags):
            scores = probabilities[index]
            if is_empty:
                results.append(
                    {
                        "label": "-",
                        "label_index": -1,
                        "confidence": 0.0,
                        "top_k": [],
                        "scores": [],
                        "empty": True,
                    }
                )
                continue

            best_index = int(np.argmax(scores))
            top_indices = np.argsort(scores)[::-1][: min(5, len(scores))]
            top_k = [
                {
                    "label": self.labels[int(i)] if int(i) < len(self.labels) else str(i),
                    "confidence": float(scores[int(i)]),
                }
                for i in top_indices
            ]
            results.append(
                {
                    "label": self.labels[best_index] if best_index < len(self.labels) else str(best_index),
                    "label_index": best_index,
                    "confidence": float(scores[best_index]),
                    "top_k": top_k,
                    "scores": [float(value) for value in scores],
                    "empty": False,
                }
            )
        return results


predictor = Predictor()


@asynccontextmanager
async def lifespan(_: FastAPI):
    predictor.load()
    yield


app = FastAPI(title="BISINDO Sign Language Predictor", lifespan=lifespan)


class PredictRequest(BaseModel):
    instances: list[list[list[float]]] = Field(
        ..., description="Batch sequence berbentuk [frames, features], misalnya 45x126"
    )


@app.get("/")
def root():
    return {"service": "bisindo-model", "endpoints": ["/health", "/labels", "/predict"]}


@app.get("/health")
def health():
    return {
        "status": "ok" if predictor.ready else "loading",
        "backend": predictor.backend,
        "num_classes": len(predictor.labels),
        "seq_len": predictor.seq_len,
        "features": predictor.features,
        "normalize": predictor.normalize,
    }


@app.get("/labels")
def labels():
    return {
        "labels": predictor.labels,
        "num_classes": len(predictor.labels),
        "seq_len": predictor.seq_len,
        "features": predictor.features,
        "normalize": predictor.normalize,
        "backend": predictor.backend,
        "model_path": predictor.model_path,
    }


@app.post("/predict")
def predict(request: PredictRequest):
    if not request.instances:
        raise HTTPException(status_code=400, detail="instances tidak boleh kosong")
    return {"predictions": predictor.predict(request.instances)}
