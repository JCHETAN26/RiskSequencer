"""Slack webhook alerting for RiskSequencer drift & retraining events.

Reads the webhook URL from the SLACK_WEBHOOK_URL env var (never hard-code
secrets). If the var is unset the functions log to stdout instead of
failing, so local/test runs don't need a real webhook.
"""

from __future__ import annotations

import json
import os
import urllib.request

WEBHOOK_ENV = "SLACK_WEBHOOK_URL"


def _post(text: str) -> bool:
    url = os.environ.get(WEBHOOK_ENV)
    if not url:
        print(f"[slack:noop] {text}")
        return False
    data = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status == 200


def alert_drift(feature: str, psi: float, threshold: float = 0.20) -> bool:
    return _post(
        "🚨 *RiskSequencer Drift Alert*\n"
        f"Feature: `{feature}`\n"
        f"PSI: {psi:.2f} (threshold: {threshold:.2f})\n"
        "Retraining triggered automatically."
    )


def alert_retrain_complete(new_auc: float, gate: float) -> bool:
    return _post(
        "✅ *RiskSequencer Retraining Complete*\n"
        f"New test AUC: {new_auc:.3f} (gate: {gate:.2f})\n"
        "Model promoted to production."
    )


def alert_gate_failed(new_auc: float, gate: float) -> bool:
    return _post(
        "⛔ *RiskSequencer Retraining Blocked*\n"
        f"New test AUC: {new_auc:.3f} is below the gate ({gate:.2f}).\n"
        "Champion retained; challenger discarded."
    )


def alert_task_failure(task_id: str, error: str) -> bool:
    return _post(f"❌ *RiskSequencer DAG failure* in `{task_id}`\n```{error[:500]}```")
