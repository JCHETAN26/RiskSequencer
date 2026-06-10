"""Deploy RiskSequencer to a SageMaker real-time endpoint, benchmark, tear down.

One-shot deployment to verify the p99 inference latency target. Trains+packages
the model, uploads to S3, deploys a PyTorch real-time endpoint, sends a burst of
requests, then reads the TRUE inference latency from CloudWatch `ModelLatency`
(which excludes client/network round-trip) and reports p50/p95/p99.

Teardown runs in a `finally` block, so the endpoint is always deleted even if the
benchmark fails — nothing is left billing.

Usage:
    export SAGEMAKER_ROLE_ARN=arn:aws:iam::<acct>:role/SageMakerExecutionRole
    python -m serving.deploy_sagemaker --instance ml.c5.large --requests 300

Verified run (2026-06-09, ml.c5.large): CloudWatch ModelLatency p99 = 45.14 ms
(< 50 ms target). Endpoint auto-torn-down.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# SageMaker execution role: set SAGEMAKER_ROLE_ARN, or pass --role.
DEFAULT_ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", "")
PYTORCH_VERSION = "2.1.0"
PY_VERSION = "py310"
FRAMETAG = f"{PYTORCH_VERSION}-cpu-{PY_VERSION}"


def _build_source_dir(dst: Path) -> Path:
    """Assemble the SageMaker source_dir (entry point + imported modules)."""
    import shutil

    from config import PROJECT_ROOT

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "models").mkdir(exist_ok=True)
    shutil.copy2(PROJECT_ROOT / "serving/inference.py", dst / "inference.py")
    shutil.copy2(PROJECT_ROOT / "config.py", dst / "config.py")
    shutil.copy2(PROJECT_ROOT / "models/__init__.py", dst / "models/__init__.py")
    shutil.copy2(PROJECT_ROOT / "models/lstm_model.py", dst / "models/lstm_model.py")
    # torch is preinstalled in the DLC; only sklearn (for the scaler) is needed.
    (dst / "requirements.txt").write_text("scikit-learn>=1.3\n")
    return dst


def _sample_payload(n_seq: int = 1, seq_len: int = 50) -> dict:
    from config import FEATURE_COLUMNS

    F = len(FEATURE_COLUMNS)
    return {"sequences": [np.random.randn(seq_len, F).tolist() for _ in range(n_seq)]}


def _model_latency_p99(endpoint_name: str, start, end, region: str) -> dict:
    """Query CloudWatch ModelLatency (microseconds) percentiles for the window."""
    import boto3

    cw = boto3.client("cloudwatch", region_name=region)
    resp = cw.get_metric_statistics(
        Namespace="AWS/SageMaker",
        MetricName="ModelLatency",
        Dimensions=[
            {"Name": "EndpointName", "Value": endpoint_name},
            {"Name": "VariantName", "Value": "AllTraffic"},
        ],
        StartTime=start,
        EndTime=end,
        Period=300,
        ExtendedStatistics=["p50", "p95", "p99"],
    )
    points = resp.get("Datapoints", [])
    if not points:
        return {}
    latest = sorted(points, key=lambda p: p["Timestamp"])[-1]["ExtendedStatistics"]
    # CloudWatch reports microseconds -> convert to ms
    return {k: v / 1000.0 for k, v in latest.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy + benchmark + teardown.")
    parser.add_argument("--instance", default="ml.c5.large")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--role", default=DEFAULT_ROLE_ARN,
                        help="SageMaker execution role ARN (or set SAGEMAKER_ROLE_ARN)")
    parser.add_argument("--keep", action="store_true", help="skip teardown (NOT recommended)")
    args = parser.parse_args()
    if not args.role:
        parser.error("provide --role <arn> or set SAGEMAKER_ROLE_ARN")

    import sagemaker
    from sagemaker.deserializers import JSONDeserializer
    from sagemaker.pytorch import PyTorchModel
    from sagemaker.serializers import JSONSerializer

    from config import ARTIFACTS_DIR, TrainConfig, ensure_dirs
    from data.sequence_builder import build_sequences, fit_scaler, time_based_split
    from data.synthetic import generate_transactions
    from features.feature_pipeline import build_features
    from serving.package_model import build_model_tar, save_artifacts
    from training.evaluate import tune_threshold
    from training.train import _scores, run_training

    sess = sagemaker.Session()
    region = sess.boto_region_name
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    endpoint_name = f"risksequencer-{ts}"

    # --- 1. train + package -------------------------------------------------
    print("[1/5] training + packaging model...", flush=True)
    ensure_dirs()
    raw = generate_transactions(n_users=2000, seed=42)
    feats = build_features(raw)
    tr, va, te = time_based_split(feats)
    scaler = fit_scaler(tr)
    train_ds, val_ds = build_sequences(tr, scaler), build_sequences(va, scaler)
    cfg = TrainConfig(max_epochs=args.max_epochs)
    out = run_training(train_ds, val_ds, cfg)
    threshold = tune_threshold(val_ds.y, _scores(out.model, val_ds, device="cpu"))
    save_artifacts(out.model, scaler, threshold, cfg.model, ARTIFACTS_DIR)
    tar = build_model_tar(ARTIFACTS_DIR, ARTIFACTS_DIR / "model.tar.gz", include_code=False)

    # --- 2. upload ----------------------------------------------------------
    print("[2/5] uploading model.tar.gz to S3...", flush=True)
    model_data = sess.upload_data(str(tar), key_prefix=f"risksequencer/{ts}")
    print(f"      {model_data}", flush=True)

    source_dir = _build_source_dir(ARTIFACTS_DIR / "src")
    predictor = None
    model = None
    try:
        # --- 3. deploy ------------------------------------------------------
        print(f"[3/5] deploying to {args.instance} (5-8 min)...", flush=True)
        model = PyTorchModel(
            model_data=model_data,
            role=args.role,
            entry_point="inference.py",
            source_dir=str(source_dir),
            framework_version=PYTORCH_VERSION,
            py_version=PY_VERSION,
            sagemaker_session=sess,
        )
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=args.instance,
            endpoint_name=endpoint_name,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
        )
        print(f"      endpoint live: {endpoint_name}", flush=True)

        # --- 4. benchmark ---------------------------------------------------
        print(f"[4/5] benchmarking {args.requests} requests...", flush=True)
        bench_start = datetime.now(timezone.utc)
        for _ in range(10):  # warmup
            predictor.predict(_sample_payload())
        client_ms = []
        for i in range(args.requests):
            payload = _sample_payload()
            t0 = time.perf_counter()
            predictor.predict(payload)
            client_ms.append((time.perf_counter() - t0) * 1000)
        bench_end = datetime.now(timezone.utc)
        client = {p: float(np.percentile(client_ms, q))
                  for p, q in (("p50", 50), ("p95", 95), ("p99", 99))}

        # --- 5. CloudWatch ModelLatency (the honest inference latency) ------
        print("[5/5] waiting ~150s for CloudWatch ModelLatency...", flush=True)
        time.sleep(150)
        model_lat = _model_latency_p99(
            endpoint_name, bench_start - timedelta(minutes=1),
            bench_end + timedelta(minutes=5), region,
        )

        print("\n==== LATENCY RESULTS ====")
        print(f"instance: {args.instance}  requests: {args.requests}")
        print("client round-trip (incl. network from laptop):")
        for k in ("p50", "p95", "p99"):
            print(f"   {k}: {client[k]:.1f} ms")
        if model_lat:
            print("CloudWatch ModelLatency (TRUE inference latency, no network):")
            for k in ("p50", "p95", "p99"):
                if k in model_lat:
                    print(f"   {k}: {model_lat[k]:.2f} ms")
            p99 = model_lat.get("p99")
            if p99 is not None:
                print(f"\nTARGET p99 < 50ms : {'PASS ✅' if p99 < 50 else 'MISS'} ({p99:.2f} ms)")
        else:
            print("ModelLatency not yet in CloudWatch; check the endpoint's metrics console.")
    finally:
        if not args.keep:
            print("\n[teardown] deleting endpoint, config, and model...", flush=True)
            try:
                if predictor is not None:
                    predictor.delete_endpoint()  # endpoint + endpoint config
                if model is not None:
                    model.delete_model()
            except Exception as e:
                print(f"[teardown] WARNING: {e} — verify in console!", flush=True)
            print("[teardown] done. Verify nothing remains: aws sagemaker list-endpoints", flush=True)


if __name__ == "__main__":
    main()
