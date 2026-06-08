"""Tests for the SageMaker inference handlers (local, no AWS).

Builds a tiny model.tar.gz layout in a temp dir, runs the four handlers
end-to-end, and asserts the output contract (valid probabilities, latency
budget headroom on CPU).
"""

from __future__ import annotations

import json
import pickle
import time

import numpy as np
import torch

from config import FEATURE_COLUMNS, SEQUENCE_LENGTH, TrainConfig
from models.lstm_model import RiskSequencer
from serving import inference

F = len(FEATURE_COLUMNS)


def _make_artifacts(tmp_path):
    from sklearn.preprocessing import RobustScaler

    cfg = TrainConfig().model
    model = RiskSequencer(input_size=F)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {
                "input_size": F,
                "hidden_size": cfg.hidden_size,
                "num_layers": cfg.num_layers,
                "dropout": cfg.dropout,
            },
        },
        tmp_path / "model.pt",
    )
    scaler = RobustScaler().fit(np.random.randn(500, F))
    with open(tmp_path / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(tmp_path / "threshold.json", "w") as f:
        json.dump({"threshold": 0.5}, f)
    return str(tmp_path)


def test_inference_roundtrip(tmp_path):
    model_dir = _make_artifacts(tmp_path)
    artifacts = inference.model_fn(model_dir)

    # one full-length sequence + one short (must be left-padded)
    body = json.dumps(
        {
            "sequences": [
                np.random.randn(SEQUENCE_LENGTH, F).tolist(),
                np.random.randn(7, F).tolist(),
            ]
        }
    )
    parsed = inference.input_fn(body, "application/json")
    preds = inference.predict_fn(parsed, artifacts)
    out = json.loads(inference.output_fn(preds, "application/json"))

    assert len(out["probabilities"]) == 2
    assert len(out["is_fraud"]) == 2
    for p in out["probabilities"]:
        assert 0.0 <= p <= 1.0
    for flag in out["is_fraud"]:
        assert flag in (0, 1)


def test_latency_headroom(tmp_path):
    model_dir = _make_artifacts(tmp_path)
    artifacts = inference.model_fn(model_dir)
    seq = np.random.randn(SEQUENCE_LENGTH, F).tolist()
    parsed = {"sequences": [seq]}

    # warmup
    inference.predict_fn(parsed, artifacts)
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        inference.predict_fn(parsed, artifacts)
        times.append((time.perf_counter() - t0) * 1000)
    p99 = float(np.percentile(times, 99))
    # Generous CPU bound; the ml.m5/c5 SLA is < 50ms. This guards against
    # accidental O(n) blowups in the handler, not the real SLA.
    assert p99 < 200.0, f"single-request p99 too slow on CPU: {p99:.1f}ms"
