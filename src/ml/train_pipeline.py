"""
train_pipeline.py
Script principal de entrenamiento — ejecutar para correr el pipeline completo:

    cd src
    python -m ml.train_pipeline [--no-hpo] [--n-per-district 200]

Pasos:
  1. Genera dataset sintético (o carga datos reales si existen)
  2. Split espacial por distrito
  3. Entrenamiento LightGBM + HPO con Optuna
  4. Evaluación en test set
  5. Guarda modelo + metadata

CAMBIOS v2 (2026-05-29):
  - Logging con timestamps en cada etapa
  - Regeneración automática del CSV si se detecta cambio en el generador
  - Resumen de balance de clases antes del entrenamiento
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

from ml.data_generator import generate_synthetic_dataset
from ml.preprocessing import load_and_split
from ml.trainer import train
from ml.evaluator import evaluate_classifier, find_optimal_threshold
from ml.config import SYNTH_DIR


# ── Helpers de logging ────────────────────────────────────────────────────────

def _ts() -> str:
    """Timestamp actual formateado."""
    return datetime.now().strftime("%H:%M:%S")


def _step(n: int, total: int, msg: str) -> float:
    """Imprime un paso numerado con timestamp y devuelve el tiempo de inicio."""
    print(f"\n[{_ts()}] ▶  Paso {n}/{total} — {msg}")
    return time.time()


def _done(t0: float) -> None:
    """Imprime el tiempo transcurrido desde t0."""
    elapsed = time.time() - t0
    print(f"[{_ts()}] ✔  Completado en {elapsed:.1f}s")


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de entrenamiento EcoLima ML")
    parser.add_argument("--no-hpo",         action="store_true",
                        help="Saltar HPO con Optuna (útil para pruebas rápidas)")
    parser.add_argument("--n-per-district", type=int, default=200,
                        help="Zonas candidatas por distrito en datos sintéticos")
    parser.add_argument("--model-name",     type=str, default="lgbm_recycling")
    parser.add_argument("--csv",            type=str, default=None,
                        help="Ruta a CSV de datos reales (omitir para usar sintéticos)")
    parser.add_argument("--regen",          action="store_true",
                        help="Forzar regeneración del dataset sintético aunque exista")
    return parser.parse_args()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    t_total = time.time()
    TOTAL   = 5

    print("=" * 60)
    print("  EcoLima ML — Pipeline de entrenamiento")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Paso 1: Datos ─────────────────────────────────────────────────────────
    t0 = _step(1, TOTAL, "Carga / generación de datos")

    if args.csv:
        csv_path = Path(args.csv)
        print(f"  Fuente: datos reales → {csv_path}")
        if not csv_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")
    else:
        csv_path = SYNTH_DIR / "synthetic_dataset.csv"
        if args.regen or not csv_path.exists():
            print(f"  Fuente: datos sintéticos (generando {args.n_per_district} zonas/distrito)...")
            generate_synthetic_dataset(n_per_district=args.n_per_district)
        else:
            print(f"  Fuente: datos sintéticos existentes → {csv_path}")
            print(f"  (usar --regen para forzar regeneración)")

    _done(t0)

    # ── Paso 2: Split espacial ────────────────────────────────────────────────
    t0 = _step(2, TOTAL, "Split espacial por distrito")
    print("  Estrategia: split por district_id (sin leakage espacial)")

    X_train, X_test, y_train, y_test, groups_train, groups_test = load_and_split(
        csv_path=str(csv_path)
    )

    print(f"  Train: {len(X_train):,} muestras | Test: {len(X_test):,} muestras")
    print(f"  Balance train — positivos: {y_train.mean():.1%} | negativos: {1 - y_train.mean():.1%}")
    print(f"  Balance test  — positivos: {y_test.mean():.1%}  | negativos: {1 - y_test.mean():.1%}")

    if y_train.mean() > 0.75 or y_train.mean() < 0.15:
        print("  ⚠  Desbalance severo detectado — revisar data_generator.py o aplicar class_weight")

    _done(t0)

    # ── Paso 3: Entrenamiento + HPO ───────────────────────────────────────────
    modo = "con HPO (Optuna)" if not args.no_hpo else "sin HPO (parámetros base)"
    t0   = _step(3, TOTAL, f"Entrenamiento LightGBM {modo}")

    result = train(
        X_train, y_train, groups_train,
        run_hpo=not args.no_hpo,
        save=True,
        model_name=args.model_name,
    )

    _done(t0)

    # ── Paso 4: Umbral óptimo ─────────────────────────────────────────────────
    t0 = _step(4, TOTAL, "Búsqueda de umbral óptimo (métrica: F1)")

    optimal_threshold = find_optimal_threshold(
        result["model"],
        result["preprocessor"],
        result["feature_engineer"],
        X_test, y_test,
        metric="f1",
    )

    if optimal_threshold < 0.25:
        print(f"  ⚠  Umbral óptimo muy bajo ({optimal_threshold:.2f}) — posible señal débil del modelo")
        print(f"     Verificar separabilidad de clases en los datos")

    _done(t0)

    # ── Paso 5: Evaluación final ──────────────────────────────────────────────
    t0 = _step(5, TOTAL, "Evaluación en test set")

    metrics = evaluate_classifier(
        result["model"],
        result["preprocessor"],
        result["feature_engineer"],
        X_test, y_test,
        threshold=optimal_threshold,
    )

    _done(t0)

    # ── Resumen ───────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_total
    print("\n" + "=" * 60)
    print("  RESUMEN FINAL")
    print("=" * 60)
    print(f"  CV AUC  : {result['cv_auc_mean']:.4f} ± {result['cv_auc_std']:.4f}")
    print(f"  Test AUC: {metrics['auc_roc']:.4f}")
    print(f"  Test F1 : {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  Umbral   : {optimal_threshold:.2f}")
    print(f"  Tiempo total: {elapsed_total:.1f}s")

    # Interpretación automática básica
    auc = metrics['auc_roc']
    if auc >= 0.80:
        nivel = "✔  Bueno"
    elif auc >= 0.70:
        nivel = "~  Aceptable (suficiente para datos sintéticos)"
    elif auc >= 0.60:
        nivel = "⚠  Débil — señal limitada"
    else:
        nivel = "✗  Casi aleatorio — revisar datos y features"

    print(f"  Calidad AUC: {nivel}")
    print("=" * 60)


if __name__ == "__main__":
    main()
