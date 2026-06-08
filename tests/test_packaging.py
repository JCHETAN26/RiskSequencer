"""Tests for SageMaker model packaging (save artifacts -> tar -> reload)."""

from __future__ import annotations

import json
import tarfile

import numpy as np
from sklearn.preprocessing import RobustScaler

from config import FEATURE_COLUMNS, ModelConfig
from models.lstm_model import RiskSequencer
from serving import inference
from serving.package_model import build_model_tar, save_artifacts

F = len(FEATURE_COLUMNS)


def _fitted_scaler() -> RobustScaler:
    return RobustScaler().fit(np.random.randn(200, F))


def test_save_artifacts_writes_three_files(tmp_path):
    cfg = ModelConfig()
    model = RiskSequencer(input_size=F)
    save_artifacts(model, _fitted_scaler(), 0.73, cfg, tmp_path)
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "scaler.pkl").exists()
    assert json.loads((tmp_path / "threshold.json").read_text())["threshold"] == 0.73


def test_build_tar_has_sagemaker_layout(tmp_path):
    cfg = ModelConfig()
    save_artifacts(RiskSequencer(input_size=F), _fitted_scaler(), 0.5, cfg, tmp_path)
    out = build_model_tar(tmp_path, tmp_path / "model.tar.gz")
    with tarfile.open(out) as tar:
        names = set(tar.getnames())
    for expected in (
        "model.pt", "scaler.pkl", "threshold.json",
        "code/inference.py", "code/config.py", "code/models/lstm_model.py",
        "code/requirements.txt",
    ):
        assert expected in names, f"{expected} missing from archive"


def test_artifacts_reload_through_inference_handler(tmp_path):
    cfg = ModelConfig()
    save_artifacts(RiskSequencer(input_size=F), _fitted_scaler(), 0.42, cfg, tmp_path)
    # model_fn reads the unpacked artifact directory (what SageMaker hands it).
    artifacts = inference.model_fn(str(tmp_path))
    assert artifacts["threshold"] == 0.42
    assert hasattr(artifacts["model"], "predict_proba")
    # end-to-end predict on a single short sequence
    body = json.dumps({"sequences": [np.random.randn(8, F).tolist()]})
    preds = inference.predict_fn(inference.input_fn(body), artifacts)
    assert 0.0 <= preds["probabilities"][0] <= 1.0
