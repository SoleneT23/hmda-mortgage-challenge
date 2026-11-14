"""Trace l'AUC du mélange en fonction du poids attribué à CatBoost."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = (
    RESULTS_DIR / "comparison_catboost_lightgbm_grid_best_blends.csv"
)
PDF_PATH = FIGURES_DIR / "courbe_auc_poids_catboost.pdf"
PNG_PATH = FIGURES_DIR / "courbe_auc_poids_catboost.png"


# Chargement des résultats de la recherche des poids
if not RESULTS_PATH.exists():
    raise FileNotFoundError(
        f"Le fichier de résultats est introuvable : {RESULTS_PATH}. "
        "Exécutez d'abord V20_select_catboost_lightgbm_weight.py."
    )

results = pd.read_csv(RESULTS_PATH)

required_columns = {"catboost_weight", "validation_auc"}
missing_columns = required_columns.difference(results.columns)

if missing_columns:
    raise ValueError(
        "Colonnes absentes du fichier de résultats : "
        f"{sorted(missing_columns)}"
    )

results = results.sort_values("catboost_weight")


# Identification du meilleur poids
best_index = results["validation_auc"].idxmax()
best_row = results.loc[best_index]

best_weight = best_row["catboost_weight"]
best_auc = best_row["validation_auc"]


# AUC de CatBoost seul
catboost_auc = results.loc[
    results["catboost_weight"] == 1.0,
    "validation_auc"
].iloc[0]


# Création de la figure
fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 5)
)


# Courbe complète
axes[0].plot(
    results["catboost_weight"],
    results["validation_auc"],
    color="steelblue",
    linewidth=2
)

axes[0].scatter(
    best_weight,
    best_auc,
    color="darkred",
    s=60,
    zorder=3,
    label=(
        f"Maximum : poids = {best_weight:.2f}, "
        f"AUC = {best_auc:.6f}"
    )
)

axes[0].axhline(
    catboost_auc,
    color="darkorange",
    linestyle="--",
    linewidth=1.5,
    label=f"CatBoost seul : AUC = {catboost_auc:.6f}"
)

axes[0].set_xlabel("Poids attribué à CatBoost")
axes[0].set_ylabel("AUC sur l'ensemble de validation")
axes[0].set_title("Ensemble des poids testés")
axes[0].grid(alpha=0.25)
axes[0].legend(fontsize=8)


# Agrandissement autour du maximum
zoom_results = results[
    results["catboost_weight"].between(0.75, 1.00)
]

axes[1].plot(
    zoom_results["catboost_weight"],
    zoom_results["validation_auc"],
    color="steelblue",
    marker="o",
    markersize=3,
    linewidth=2
)

axes[1].scatter(
    best_weight,
    best_auc,
    color="darkred",
    s=60,
    zorder=3,
    label=f"Maximum : {best_weight:.2f}"
)

axes[1].axhline(
    catboost_auc,
    color="darkorange",
    linestyle="--",
    linewidth=1.5,
    label="CatBoost seul"
)

axes[1].set_xlabel("Poids attribué à CatBoost")
axes[1].set_ylabel("AUC sur l'ensemble de validation")
axes[1].set_title("Agrandissement autour du maximum")
axes[1].grid(alpha=0.25)
axes[1].legend(fontsize=8)


fig.suptitle(
    "Influence du poids de CatBoost sur l'AUC du mélange",
    fontsize=14
)

fig.tight_layout()


# PDF vectoriel pour le rapport
fig.savefig(
    PDF_PATH,
    bbox_inches="tight"
)

# Version PNG pour visualiser facilement la figure
fig.savefig(
    PNG_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


print("Meilleur poids CatBoost :", best_weight)
print("Poids LightGBM associé :", 1 - best_weight)
print("Meilleure AUC :", best_auc)
print("Fichiers créés :")
print("-", PDF_PATH)
print("-", PNG_PATH)
