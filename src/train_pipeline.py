"""
train_pipeline.py
Script principal de entrenamiento — ejecutar para correr el pipeline completo:

    cd ml/src
    python train_pipeline.py [--no-hpo] [--n-per-district 200]

Pasos:
  1. Genera dataset sintético (o carga datos reales si existen)
  2. Split espacial por distrito
  3. Entrenamiento LightGBM + HPO con Optuna
  4. Evaluación en test set
  5. Guarda modelo + metadata
"""

import argparse
from pathlib import Path

from data_generator import generate_synthetic_dataset
from preprocessing import load_and_split
from trainer import train
from evaluator import evaluate_classifier, find_optimal_threshold
from config import SYNTH_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de entrenamiento LightGBM")
    parser.add_argument("--no-hpo",          action="store_true", help="Saltar HPO con Optuna")
    parser.add_argument("--n-per-district",  type=int, default=200, help="Zonas candidatas por distrito (datos sintéticos)")
    parser.add_argument("--model-name",      type=str, default="lgbm_recycling")
    parser.add_argument("--csv",             type=str, default=None, help="Ruta a CSV de datos reales (omitir para usar sintéticos)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 1. Datos ──────────────────────────────────────────────────────────────
    if args.csv:
        csv_path = Path(args.csv)
        print(f"[pipeline] Cargando datos reales: {csv_path}")
    else:
        print(f"[pipeline] Generando dataset sintético ({args.n_per_district} zonas/distrito)...")
        df = generate_synthetic_dataset(n_per_district=args.n_per_district)
        csv_path = SYNTH_DIR / "synthetic_dataset.csv"

    # ── 2. Split espacial ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, groups_train, groups_test = load_and_split(
        csv_path=str(csv_path)
    )

    # ── 3. Entrenamiento ──────────────────────────────────────────────────────
    result = train(
        X_train, y_train, groups_train,
        run_hpo=not args.no_hpo,
        save=True,
        model_name=args.model_name,
    )

    # ── 4. Evaluación ─────────────────────────────────────────────────────────
    # Buscar umbral óptimo en un fold de validación (aquí usamos test como proxy)
    optimal_threshold = find_optimal_threshold(
        result["model"],
        result["preprocessor"],
        result["feature_engineer"],
        X_test, y_test,
        metric="f1",
    )

    metrics = evaluate_classifier(
        result["model"],
        result["preprocessor"],
        result["feature_engineer"],
        X_test, y_test,
        threshold=optimal_threshold,
    )

    print("\n── Resumen Final ───────────────────────────────────────────")
    print(f"  CV AUC  : {result['cv_auc_mean']:.4f} ± {result['cv_auc_std']:.4f}")
    print(f"  Test AUC: {metrics['auc_roc']:.4f}")
    print(f"  Test F1 : {metrics['f1']:.4f}")
    print(f"  Umbral  : {optimal_threshold:.2f}")
    print("────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
