"""Regression guard: torch + LightGBM must coexist in one process.

Both libraries ship their own OpenMP runtime; loading both in a single process
segfaults on macOS unless OpenMP is constrained. The `02_modeling` notebook
trains both models in one kernel, so this locks in the workaround
(OMP_NUM_THREADS=1 + KMP_DUPLICATE_LIB_OK=TRUE + single-threaded libs).

It runs in a **subprocess** on purpose: a crash here must not take down the
whole pytest process, and the env vars have to be set *before* the libraries
import (impossible to do reliably inside an already-running interpreter).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# The exact pattern the modeling notebook uses: train both models together.
_SNIPPET = """
import lightgbm, torch
torch.set_num_threads(1)
from config import TrainConfig
from data.synthetic import generate_transactions
from features.feature_pipeline import build_features
from data.sequence_builder import build_sequences, fit_scaler, time_based_split
from models.lgbm_baseline import last_txn_table, train_lgbm
from training.train import run_training

raw = generate_transactions(n_users=200, seed=1)
feats = build_features(raw)
tr, va, te = time_based_split(feats)
Xtr, ytr = last_txn_table(tr); Xva, yva = last_txn_table(va)
train_lgbm(Xtr, ytr, Xva, yva, params={"num_threads": 1})
sc = fit_scaler(tr)
run_training(build_sequences(tr, sc), build_sequences(va, sc), TrainConfig(max_epochs=1))
print("COEXIST_OK")
"""


def _run(env_extra: dict) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(ROOT), **env_extra}
    return subprocess.run(
        [sys.executable, "-c", _SNIPPET],
        env=env, capture_output=True, text=True, timeout=300,
    )


def test_torch_and_lightgbm_coexist_with_guard():
    """With the OpenMP guard set, training both models succeeds."""
    proc = _run({"OMP_NUM_THREADS": "1", "KMP_DUPLICATE_LIB_OK": "TRUE"})
    assert proc.returncode == 0, (
        f"torch+lightgbm crashed even with the OpenMP guard "
        f"(rc={proc.returncode}).\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-2000:]}"
    )
    assert "COEXIST_OK" in proc.stdout
