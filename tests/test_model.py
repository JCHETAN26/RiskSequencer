"""Tests for the RiskSequencer LSTM: shapes, masking, gradient flow."""

from __future__ import annotations

import torch

from config import FEATURE_COLUMNS, SEQUENCE_LENGTH
from models.lstm_model import RiskSequencer

F = len(FEATURE_COLUMNS)


def test_forward_shapes():
    model = RiskSequencer(input_size=F)
    x = torch.randn(4, SEQUENCE_LENGTH, F)
    logits, attn = model(x)
    assert logits.shape == (4, 1)
    assert attn.shape == (4, SEQUENCE_LENGTH, 1)


def test_attention_ignores_padding():
    model = RiskSequencer(input_size=F)
    x = torch.randn(3, SEQUENCE_LENGTH, F)
    mask = torch.zeros(3, SEQUENCE_LENGTH, dtype=torch.bool)
    mask[:, :12] = True  # first 12 are padding
    _, attn = model(x, mask)
    assert torch.allclose(attn[:, :12].sum(), torch.tensor(0.0), atol=1e-4)
    assert torch.allclose(attn.sum(dim=1), torch.ones(3, 1), atol=1e-4)


def test_predict_proba_in_unit_interval():
    model = RiskSequencer(input_size=F)
    x = torch.randn(5, SEQUENCE_LENGTH, F)
    p = model.predict_proba(x)
    assert p.shape == (5,)
    assert (p >= 0).all() and (p <= 1).all()


def test_gradients_flow():
    model = RiskSequencer(input_size=F)
    x = torch.randn(4, SEQUENCE_LENGTH, F)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    logits, _ = model(x)
    loss = torch.nn.BCEWithLogitsLoss()(logits.squeeze(-1), y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients computed"
    assert any(g.abs().sum() > 0 for g in grads)
