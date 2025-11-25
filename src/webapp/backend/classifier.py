import logging
import os
from pathlib import Path
try:
    from typing import Final
except ImportError:  # Python 3.6 fallback
    from typing_extensions import Final

import numpy as np
import tensorflow as tf

# Input constraints replicated from training (50 lines, 305 chars each)
MAX_LINES: Final[int] = 50
MAX_CHARS: Final[int] = 305


def _normalize_line_endings(text: str) -> str:
    """Unify all line endings to '\n' to match dataset preprocessing."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def code_to_matrix(code: str, max_lines: int = MAX_LINES, max_chars: int = MAX_CHARS) -> np.ndarray:
    """
    Convert a raw code snippet into the structure matrix expected by the CNN.

    The dataset encodes each line as ASCII/Unicode code points, padded with -1.
    """
    processed = _normalize_line_endings(code or "")
    # Preserve empty trailing line if the text ends with a newline
    raw_lines = processed.split("\n")

    matrix = np.full((max_lines, max_chars), -1, dtype=np.int16)
    for row_idx in range(min(len(raw_lines), max_lines)):
        line = raw_lines[row_idx].replace("\t", "    ")
        # Append newline to mimic original file iteration behaviour
        line_with_newline = f"{line}\n"
        char_codes = [ord(ch) for ch in line_with_newline[:max_chars]]
        matrix[row_idx, : len(char_codes)] = char_codes
    return matrix


def build_structure_model() -> tf.keras.Model:
    """Recreate the fine-tuned CNN used during training."""
    regularizer = tf.keras.regularizers.l2(0.001)

    structure_input = tf.keras.Input(shape=(MAX_LINES, MAX_CHARS), name="structure")
    x = tf.keras.layers.Reshape((MAX_LINES, MAX_CHARS, 1), name="reshape")(structure_input)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", name="conv1")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, name="pool1")(x)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", name="conv2")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, name="pool2")(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", name="conv3")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=3, name="pool3")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer, name="dense1")(x)
    x = tf.keras.layers.Dropout(0.5, name="drop")(x)
    x = tf.keras.layers.Dense(16, activation="relu", name="random_detail")(x)
    dense3 = tf.keras.layers.Dense(1, activation="sigmoid", name="dense3")(x)

    return tf.keras.Model(inputs=structure_input, outputs=dense3, name="structure_readability")


class StructureReadabilityClassifier:
    """Load the CNN once and expose a simple prediction API."""

    def __init__(self, weights_path: Path):
        self.weights_path = weights_path
        resolved_path = self._resolve_weights_path()
        if not resolved_path.exists():
            raise FileNotFoundError(f"未找到模型权重文件: {resolved_path}")
        self.model = self._load_model(resolved_path)

    def _resolve_weights_path(self) -> Path:
        if self.weights_path.is_file():
            return self.weights_path
        # Allow users to pass directories; use default filename if so
        candidate = self.weights_path / "T_BEST.hdf5"
        return candidate

    def _load_model(self, resolved_path: Path) -> tf.keras.Model:
        try:
            model = tf.keras.models.load_model(str(resolved_path), compile=False)
            logging.info("已作为完整模型加载权重: %s", resolved_path)
            return model
        except Exception as exc:
            logging.warning("直接加载模型失败，尝试按结构加载权重: %s", exc)

        model = build_structure_model()
        model.load_weights(str(resolved_path), by_name=True, skip_mismatch=True)
        return model

    def predict_probability(self, code: str) -> float:
        matrix = code_to_matrix(code)
        batch = np.expand_dims(matrix.astype(np.float32), axis=0)
        logits = self.model(batch, training=False)
        return float(tf.reshape(logits, []).numpy())

    def predict_label(self, code: str, threshold: float = 0.5) -> dict:
        probability = self.predict_probability(code)
        label = "Readable" if probability >= threshold else "Unreadable"
        return {"probability": probability, "label": label}


def default_weights_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    default_path = repo_root / "code" / "Experimental output" / "T_BEST.hdf5"
    env_override = os.getenv("MODEL_WEIGHTS_PATH")
    if env_override:
        return Path(env_override).expanduser().resolve()
    return default_path

