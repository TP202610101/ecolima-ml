"""Build a simulated training dataset and train a development model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

from ml.config import DEFAULT_SERVING_MODEL, DEFAULT_SIMULATED_TRAINING_DATASET, METRICS_DIR, MODELS_DIR
from ml.dataset_builder import build_simulated_training_dataset
from ml.evaluator import evaluate_classifier, find_optimal_threshold
from ml.preprocessing import load_and_split
from ml.trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simulated EcoLima training")
    parser.add_argument("--n-per-district", type=int, default=100)
    parser.add_argument("--positive-rate", type=float, default=0.35)
    parser.add_argument("--model-name", default=DEFAULT_SERVING_MODEL)
    parser.add_argument("--output", default=str(DEFAULT_SIMULATED_TRAINING_DATASET))
    parser.add_argument("--hpo", action="store_true", help="Enable Optuna HPO. Disabled by default for fast runs.")
    parser.add_argument("--metrics-out", default=str(METRICS_DIR / "simulated_v0_3_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    dataset, dataset_summary = build_simulated_training_dataset(
        n_per_district=args.n_per_district,
        positive_rate=args.positive_rate,
        output_path=args.output,
    )

    X_train, X_test, y_train, y_test, groups_train, groups_test = load_and_split(df=dataset)
    result = train(
        X_train,
        y_train,
        groups_train,
        run_hpo=args.hpo,
        save=True,
        model_name=args.model_name,
    )
    threshold = find_optimal_threshold(
        result["model"],
        result["preprocessor"],
        result["feature_engineer"],
        X_test,
        y_test,
        metric="f1",
        feature_names=result["feature_names"],
    )
    metrics = evaluate_classifier(
        result["model"],
        result["preprocessor"],
        result["feature_engineer"],
        X_test,
        y_test,
        threshold=threshold,
        feature_names=result["feature_names"],
    )


    metadata_path = MODELS_DIR / f"{args.model_name}_metadata.json"
    if metadata_path.exists():
        model_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_metadata["serving_threshold"] = threshold
        model_metadata["training_dataset"] = dataset_summary["output"]
        model_metadata["training_dataset_version"] = dataset_summary["dataset_version"]
        model_metadata["test_metrics"] = metrics
        metadata_path.write_text(json.dumps(model_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    payload = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_summary,
        "model_name": args.model_name,
        "run_hpo": bool(args.hpo),
        "cv_auc_mean": result["cv_auc_mean"],
        "cv_auc_std": result["cv_auc_std"],
        "threshold": threshold,
        "metrics": metrics,
        "elapsed_seconds": round(time.time() - started, 2),
        "warning": "Entrenamiento sobre datos simulados; no reportar como evidencia real de tesis.",
    }

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[run_simulated_training] Metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
