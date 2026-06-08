"""SageMaker entry point for the RiskSequencer real-time endpoint.

Implements the four SageMaker inference handlers (`model_fn`, `input_fn`,
`predict_fn`, `output_fn`). The deployed `model.tar.gz` is expected to bundle:

    model.pt      — torch.save of {"state_dict", "model_config"}
    scaler.pkl    — the fitted RobustScaler
    threshold.json— {"threshold": <float>} chosen at train time

The handler accepts already-feature-engineered sequences so the latency
budget (p99 < 50ms) covers only scaling + the forward pass. Raw feature
rows are scaled here using the bundled scaler to guarantee train/serve
parity.
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any

import numpy as np
import torch

from config import FEATURE_COLUMNS, SEQUENCE_LENGTH
from models.lstm_model import RiskSequencer

JSON = "application/json"


def model_fn(model_dir: str) -> dict[str, Any]:
    """Load model + scaler + threshold from the unpacked model.tar.gz."""
    ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    cfg = ckpt["model_config"]
    model = RiskSequencer(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(model_dir, "threshold.json")) as f:
        threshold = float(json.load(f)["threshold"])

    return {"model": model, "scaler": scaler, "threshold": threshold}


def input_fn(request_body: str | bytes, content_type: str = JSON) -> dict[str, np.ndarray]:
    """Parse a JSON request into raw feature arrays.

    Expected body:
        {"sequences": [[[f1..fF] x up-to-50] , ...]}   # one entry per user
    Variable-length inner sequences are accepted; they are left-padded here.
    """
    if content_type != JSON:
        raise ValueError(f"Unsupported content type: {content_type}")
    payload = json.loads(request_body)
    return {"sequences": payload["sequences"]}


def _pad_and_mask(seq: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(seq, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"each sequence must be (t, {len(FEATURE_COLUMNS)}); got {arr.shape}"
        )
    t = arr.shape[0]
    if t >= SEQUENCE_LENGTH:
        return arr[-SEQUENCE_LENGTH:], np.zeros(SEQUENCE_LENGTH, dtype=bool)
    pad = np.zeros((SEQUENCE_LENGTH - t, len(FEATURE_COLUMNS)), dtype=np.float32)
    mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
    mask[: SEQUENCE_LENGTH - t] = True
    return np.vstack([pad, arr]), mask


def predict_fn(inputs: dict[str, np.ndarray], artifacts: dict[str, Any]) -> dict[str, list]:
    """Scale, pad, run the model, apply the threshold."""
    model, scaler, threshold = artifacts["model"], artifacts["scaler"], artifacts["threshold"]

    X, masks = [], []
    for seq in inputs["sequences"]:
        padded, mask = _pad_and_mask(seq)
        # scale only the real rows; padding stays zero
        real = ~mask
        scaled = padded.copy()
        scaled[real] = scaler.transform(padded[real].astype(np.float64)).astype(np.float32)
        X.append(scaled)
        masks.append(mask)

    xb = torch.from_numpy(np.stack(X))
    mb = torch.from_numpy(np.stack(masks))
    probs = model.predict_proba(xb, mb).numpy()
    flags = (probs >= threshold).astype(int)
    return {"probabilities": probs.tolist(), "is_fraud": flags.tolist()}


def output_fn(prediction: dict[str, list], accept: str = JSON) -> str:
    """Serialise the prediction dict to JSON."""
    if accept != JSON:
        raise ValueError(f"Unsupported accept type: {accept}")
    return json.dumps(prediction)
