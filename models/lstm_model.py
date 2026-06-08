"""RiskSequencer — 2-layer LSTM with additive attention.

Matches the reference architecture in the system prompt. Attention is
**mandatory** (explainability is a hard requirement): the forward pass
returns per-timestep attention weights alongside the fraud probability so
flagged sequences can be explained ("these past events drove the flag").

Masking convention matches `data.sequence_builder`: `mask[b, t] == True`
means position t is *padding* and must be excluded from attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RiskSequencer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.attention = nn.Linear(hidden_size, 1)  # additive attention score
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        # NOTE: the final Sigmoid from the reference snippet is intentionally
        # omitted. We emit raw *logits* so training can use the numerically
        # stable BCEWithLogitsLoss(pos_weight=...). `predict_proba` applies
        # the sigmoid for inference.

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, attn_weights).

        x    : (B, T, F)
        mask : (B, T) bool, True where padding (optional)
        logits       : (B, 1)
        attn_weights : (B, T, 1) — softmax over real (non-pad) timesteps
        """
        lstm_out, _ = self.lstm(x)                  # (B, T, H)
        attn_scores = self.attention(lstm_out)      # (B, T, 1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = (attn_weights * lstm_out).sum(dim=1)  # (B, H)
        logits = self.classifier(context)           # (B, 1)
        return logits, attn_weights

    @torch.no_grad()
    def predict_proba(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Fraud probability in [0, 1], shape (B,)."""
        logits, _ = self.forward(x, mask)
        return torch.sigmoid(logits).squeeze(-1)


if __name__ == "__main__":
    from config import FEATURE_COLUMNS, SEQUENCE_LENGTH

    B, T, F = 8, SEQUENCE_LENGTH, len(FEATURE_COLUMNS)
    model = RiskSequencer(input_size=F)
    x = torch.randn(B, T, F)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, :10] = True  # first 10 positions are padding
    logits, attn = model(x, mask)
    assert logits.shape == (B, 1)
    assert attn.shape == (B, T, 1)
    # attention over padded positions must be ~0
    assert torch.allclose(attn[:, :10].sum(), torch.tensor(0.0), atol=1e-4)
    # weights over real positions sum to 1
    assert torch.allclose(attn.sum(dim=1), torch.ones(B, 1), atol=1e-4)
    print("OK: forward pass, masking, and attention normalisation verified.")
