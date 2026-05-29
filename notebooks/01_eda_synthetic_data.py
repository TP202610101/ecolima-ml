"""
01_eda_synthetic_data.py
EDA completo del dataset sintético de EcoLima ML.

Ejecutar desde la raíz del repo:
    cd C:\\Users\\nikole.garcia\\Repositorios\\ecolima-ml
    python notebooks/01_eda_synthetic_data.py

O copiar las celdas a un Jupyter Notebook (.ipynb) en Google Colab.

Secciones:
    1. Carga y resumen general
    2. Distribución del target (balance de clases)
    3. Distribuciones univariadas por tipo de feature
    4. Correlación entre features (heatmap)
    5. Correlación de cada feature con el target
    6. Separabilidad de clases (boxplots por is_optimal)
    7. Detección de outliers (IQR)
    8. Análisis por distrito
    9. Matriz de dispersión (top features)
   10. Conclusiones automáticas
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# ── Config de estilo ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})
PALETTE   = {"0 — No óptima": "#E07B54", "1 — Óptima": "#4A90C4"}
COLOR_POS = "#4A90C4"
COLOR_NEG = "#E07B54"
COLOR_NEU = "#6BAF92"

# ── Rutas ─────────────────────────────────────────────────────────────────────
# Ajusta esta ruta si corres el script desde otro directorio
REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_PATH  = REPO_ROOT / "data" / "synthetic" / "synthetic_dataset.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Listas de features por categoría (deben coincidir con config.py)
DEMOGRAPHIC  = ["population_density", "num_households", "nse_ab_pct",
                "nse_c_pct", "nse_de_pct", "urbanization_rate"]
GEOSPATIAL   = ["dist_nearest_road_m", "walkability_score",
                "poi_commercial_500m", "poi_educational_500m",
                "poi_parks_500m", "land_use_encoded", "area_m2"]
OPERATIONAL  = ["dist_nearest_recycling_m", "recycling_density_1km",
                "waste_per_capita_kg", "coverage_gap_index"]
ENGINEERED   = ["accessibility_composite", "nse_high_ratio", "recycling_deficit"]
ALL_FEATURES = DEMOGRAPHIC + GEOSPATIAL + OPERATIONAL + ENGINEERED
TARGET       = "is_optimal"
GROUP        = "district_id"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CARGA Y RESUMEN GENERAL
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  SECCIÓN 1 — Carga y resumen general")
print("="*60)

df = pd.read_csv(DATA_PATH)
print(f"\nShape: {df.shape[0]:,} filas × {df.shape[1]} columnas")
print(f"Distritos: {df['district_name'].unique().tolist()}")
print(f"Filas por distrito: {df.groupby('district_name').size().to_dict()}")
print(f"\nTipos de datos:")
print(df[ALL_FEATURES + [TARGET]].dtypes.value_counts())
print(f"\nValores nulos:")
nulls = df[ALL_FEATURES + [TARGET]].isnull().sum()
print(nulls[nulls > 0] if nulls.any() else "  Sin valores nulos ✔")

print(f"\nEstadísticas descriptivas:")
print(df[ALL_FEATURES].describe().round(3).T[
    ["mean", "std", "min", "25%", "50%", "75%", "max"]
].to_string())


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DISTRIBUCIÓN DEL TARGET
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  SECCIÓN 2 — Balance de clases")
print("="*60)

balance = df[TARGET].value_counts()
balance_pct = df[TARGET].value_counts(normalize=True)
print(f"\n  is_optimal=0 (No óptima): {balance[0]:,}  ({balance_pct[0]:.1%})")
print(f"  is_optimal=1 (Óptima)   : {balance[1]:,}  ({balance_pct[1]:.1%})")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Balance de clases — is_optimal", fontsize=13, fontweight="bold", y=1.01)

# Barplot
axes[0].bar(["No óptima (0)", "Óptima (1)"], [balance[0], balance[1]],
            color=[COLOR_NEG, COLOR_POS], edgecolor="white", linewidth=1.5)
for i, (v, p) in enumerate(zip(balance, balance_pct)):
    axes[0].text(i, v + 5, f"{v:,}\n({p:.1%})", ha="center", va="bottom", fontsize=10)
axes[0].set_ylabel("Cantidad de zonas")
axes[0].set_title("Conteo absoluto")

# Por distrito
district_balance = df.groupby(["district_name", TARGET]).size().unstack(fill_value=0)
district_balance.plot(kind="bar", ax=axes[1], color=[COLOR_NEG, COLOR_POS],
                      edgecolor="white", linewidth=0.8)
axes[1].set_xlabel("")
axes[1].set_ylabel("Zonas")
axes[1].set_title("Balance por distrito")
axes[1].legend(["No óptima", "Óptima"], frameon=False)
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_balance_clases.png", bbox_inches="tight")
plt.show()
print(f"  Guardado: outputs/eda/01_balance_clases.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DISTRIBUCIONES UNIVARIADAS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  SECCIÓN 3 — Distribuciones univariadas")
print("="*60)

numeric_features = [f for f in ALL_FEATURES if f != "land_use_encoded"]
n_cols = 4
n_rows = int(np.ceil(len(numeric_features) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
axes = axes.flatten()

for i, feat in enumerate(numeric_features):
    ax = axes[i]
    ax.hist(df.loc[df[TARGET] == 0, feat].dropna(), bins=30,
            alpha=0.6, color=COLOR_NEG, label="No óptima", density=True)
    ax.hist(df.loc[df[TARGET] == 1, feat].dropna(), bins=30,
            alpha=0.6, color=COLOR_POS, label="Óptima", density=True)
    ax.set_title(feat, fontsize=9, fontweight="bold")
    ax.set_xlabel("")
    ax.tick_params(labelsize=7)
    if i == 0:
        ax.legend(fontsize=7, frameon=False)

# Ocultar ejes sobrantes
for j in range(len(numeric_features), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Distribuciones univariadas — por clase", fontsize=13,
             fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_distribuciones_univariadas.png", bbox_inches="tight")
plt.show()
print(f"  Guardado: outputs/eda/02_distribuciones_univariadas.png")
print("  Interpretar: si las distribuciones de clase 0 y 1 se solapan mucho,")
print("  la feature tiene poco poder discriminativo.")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CORRELACIÓN ENTRE FEATURES (HEATMAP)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  SECCIÓN 4 — Correlación entre features")
print("="*60)

corr_matrix = df[ALL_FEATURES].corr(method="pearson")

# Pares con alta correlación (posible redundancia)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        val = corr_matrix.iloc[i, j]
        if abs(val) >= 0.70:
            high_corr_pairs.append({
                "feature_a": corr_matrix.columns[i],
                "feature_b": corr_matrix.columns[j],
                "correlacion": round(val, 3),
            })

if high_corr_pairs:
    print(f"\n  Pares con |correlación| ≥ 0.70 (posible multicolinealidad):")
    for p in sorted(high_corr_pairs, key=lambda x: abs(x["correlacion"]), reverse=True):
        print(f"    {p['feature_a']} ↔ {p['feature_b']}: {p['correlacion']:.3f}")
else:
    print("\n  Sin pares con correlación ≥ 0.70 ✔")

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    linewidths=0.5, ax=ax,
    annot_kws={"size": 7},
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Matriz de correlación de Pearson — features", fontsize=13, fontweight="bold")
ax.tick_params(axis="x", rotation=45, labelsize=8)
ax.tick_params(axis="y", rotation=0, labelsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_correlacion_features.png", bbox_inches="tight")
plt.show()
print(f"  Guardado: outputs/eda/03_correlacion_features.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CORRELACIÓN CON EL TARGET
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  SECCIÓN 5 — Correlación de features con el target")
print("="*60)

target_corr = df[ALL_FEATURES + [TARGET]].corr(method="pearson")[TARGET].drop(TARGET)
target_corr_sorted = target_corr.abs().sort_values(ascending=False)

print(f"\n  Ranking de features por |correlación con is_optimal|:")
for feat, val in target_corr_sorted.items():
    direction = "↑" if target_corr[feat] > 0 else "↓"
    bar = "█" * int(abs(val) * 30)
    print(f"  {direction} {feat:<32} {val:.3f}  {bar}")

fig, ax = plt.subplots(figsize=(10, 7))
colors = [COLOR_POS if target_corr[f] > 0 else COLOR_NEG for f in target_corr_sorted.index]
ax.barh(target_corr_sorted.index, target_corr_sorted.values, color=colors, edgecolor="white")
ax.axvline(0.1, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="|r|=0.10")
ax.set_xlabel("|Correlación de Pearson con is_optimal|")
ax.set_title("Importancia univariada (correlación con target)", fontsize=13, fontweight="bold")
ax.legend(frameon=False, fontsize=9)

# Etiquetas de valor
for i, (feat, val) in enumerate(target_corr_sorted.items()):
    real_val = target_corr[feat]
    sign = "+" if real_val >= 0 else ""
    ax.text(val + 0.003, i, f"{sign}{real_val:.3f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_correlacion_target.png", bbox_inches="tight")
plt.show()
print(f"  Guardado: outputs/eda/04_correlacion_target.png")

# Features por debajo del umbral mínimo
umbral = 0.05
debiles = target_corr_sorted[target_corr_sorted < umbral]
if not debiles.empty:
    print(f"\n  ⚠  Features con |r| < {umbral} (señal muy débil con el target):")
    for f, v in debiles.items():
        print(f"     {f}: {v:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SEPARABILIDAD DE CLASES — BOXPLOTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  SECCIÓN 6 — Separabilidad de clases (boxplots)")
print("="*60)

# Top 12 features por correlación con target
top_features = target_corr_sorted.head(12).index.tolist()

n_cols = 4
n_rows = int(np.ceil(len(top_features) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
axes = axes.flatten()

for i, feat in enumerate(top_features):
    ax = axes[i]
    data_0 = df.loc[df[TARGET] == 0, feat].dropna()
    data_1 = df.loc[df[TARGET] == 1, feat].dropna()
    bp = ax.boxplot([data_0, data_1], patch_artist=True, notch=True,
                    widths=0.5, showfliers=True,
                    flierprops={"marker": ".", "markersize": 3, "alpha": 0.3})
    bp["boxes"][0].set_facecolor(COLOR_NEG + "99")
    bp["boxes"][1].set_facecolor(COLOR_POS + "99")
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)
    ax.set_xticklabels(["No óptima", "Óptima"], fontsize=8)
    ax.set_title(feat, fontsize=9, fontweight="bold")
    r = target_corr[feat]
    ax.set_xlabel(f"r={r:.3f}", fontsize=7, color="gray")

for j in range(len(top_features), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Separabilidad de clases — Top 12 features por correlación",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_boxplots_separabilidad.png", bbox_inches="tight")
plt.show()
print(f"  Guardado: outputs/eda/05_boxplots_separabilidad.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DETECCIÓN DE OUTLIERS (IQR)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  SECCIÓN 7 — Detección de outliers (método IQR)")
print("="*60)

outlier_summary = []
for feat in ALL_FEATURES:
    if feat == "land_use_encoded":
        continue
    col = df[feat].dropna()
    Q1, Q3 = col.quantile(0.25), col.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_out = ((col < lower) | (col > upper)).sum()
    pct   = n_out / len(col)
    outlier_summary.append({
        "feature": feat, "n_outliers": n_out,
        "pct_outliers": round(pct * 100, 2),
        "lower_fence": round(lower, 3),
        "upper_fence": round(upper, 3),
    })

df_out = pd.DataFrame(outlier_summary).sort_values("pct_outliers", ascending=False)
print(f"\n  Features con outliers IQR:")
print(df_out[df_out["n_outliers"] > 0].to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
df_out_plot = df_out[df_out["pct_outliers"] > 0].sort_values("pct_outliers")
ax.barh(df_out_plot["feature"], df_out_plot["pct_outliers"], color=COLOR_NEU)
ax.axvline(5, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label="5%")
ax.set_xlabel("% de outliers (IQR × 1.5)")
ax.set_title("Porcentaje de outliers por feature", fontsize=12, fontweight="bold")
ax.legend(frameon=False)
for i, (_, row) in enumerate(df_out_plot.iterrows()):
    ax.text(row["pct_outliers"] + 0.05, i, f"{row['pct_outliers']:.1f}%", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_outliers.png", bbox_inches="tight")
plt.show()
print(f"  Guardado: outputs/eda/06_outliers.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. ANÁLISIS POR DISTRITO
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  SECCIÓN 8 — Análisis por distrito")
print("="*60)

district_stats = df.groupby("district_name").agg(
    n_zonas         = ("zone_id", "count"),
    pct_optimas     = (TARGET, "mean"),
    pop_density_med = ("population_density", "median"),
    walkability_med = ("walkability_score", "median"),
    dist_recycling_med = ("dist_nearest_recycling_m", "median"),
    coverage_gap_med   = ("coverage_gap_index", "median"),
    nse_ab_med      = ("nse_ab_pct", "median"),
).round(3)

print(f"\n{district_stats.to_string()}")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Perfil por distrito", fontsize=13, fontweight="bold")

metrics_to_plot = [
    ("pct_optimas",       "% zonas óptimas",           COLOR_POS),
    ("pop_density_med",   "Densidad pob. mediana",      COLOR_NEU),
    ("dist_recycling_med","Dist. punto reciclaje (med).",COLOR_NEG),
    ("nse_ab_med",        "NSE A+B mediano",            "#9B59B6"),
]

for ax, (col, label, color) in zip(axes.flatten(), metrics_to_plot):
    vals = district_stats[col].sort_values(ascending=True)
    ax.barh(vals.index, vals.values, color=color, edgecolor="white")
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_perfil_distritos.png", bbox_inches="tight")
plt.show()
print(f"  Guardado: outputs/eda/07_perfil_distritos.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SCATTER MATRIX — TOP FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  SECCIÓN 9 — Matriz de dispersión (top 6 features)")
print("="*60)

top6 = target_corr_sorted.head(6).index.tolist()
df_plot = df[top6 + [TARGET]].copy()
df_plot["Clase"] = df_plot[TARGET].map({0: "0 — No óptima", 1: "1 — Óptima"})

g = sns.pairplot(
    df_plot, vars=top6, hue="Clase",
    palette={"0 — No óptima": COLOR_NEG, "1 — Óptima": COLOR_POS},
    plot_kws={"alpha": 0.3, "s": 10},
    diag_kind="kde",
    corner=True,
)
g.figure.suptitle("Scatter matrix — Top 6 features por correlación con target",
                   fontsize=12, fontweight="bold", y=1.01)
g.figure.savefig(OUTPUT_DIR / "08_scatter_matrix.png", bbox_inches="tight")
plt.show()
print(f"  Guardado: outputs/eda/08_scatter_matrix.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. CONCLUSIONES AUTOMÁTICAS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  SECCIÓN 10 — Conclusiones automáticas")
print("="*60)

balance_1 = df[TARGET].mean()
balance_0 = 1 - balance_1

print(f"""
BALANCE DE CLASES
  Clase 0 (no óptima): {balance_0:.1%}
  Clase 1 (óptima)   : {balance_1:.1%}
  {'✔ Balance aceptable (entre 30% y 70%)' if 0.30 <= balance_1 <= 0.70
   else '⚠ Desbalance — considerar scale_pos_weight o SMOTE'}

TOP 5 FEATURES POR CORRELACIÓN CON TARGET:
""")

for i, (feat, val) in enumerate(target_corr_sorted.head(5).items(), 1):
    r = target_corr[feat]
    direction = "positiva" if r > 0 else "negativa"
    print(f"  {i}. {feat}: r={r:.3f} ({direction})")

print(f"""
MULTICOLINEALIDAD:
  Pares con |r| ≥ 0.70: {len(high_corr_pairs)}
  {'✔ Sin multicolinealidad severa' if not high_corr_pairs
   else '⚠ Revisar: ' + ', '.join([f"{p['feature_a']}↔{p['feature_b']}" for p in high_corr_pairs[:3]])}

FEATURES DÉBILES (|r con target| < 0.05):
  {', '.join(debiles.index.tolist()) if not debiles.empty else '✔ Todas las features tienen señal mínima'}

SEPARABILIDAD GENERAL:
  AUC esperado mínimo con estas features: {'> 0.65 (señal presente)' if target_corr_sorted.iloc[0] > 0.20
                                            else '< 0.65 (señal débil — revisar generador)'}

TODOS LOS PLOTS guardados en: outputs/eda/
""")
