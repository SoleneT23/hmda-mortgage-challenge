"""Sélectionne le poids du mélange entre CatBoost et LightGBM."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# Les deux fichiers de prédictions doivent provenir de la même séparation
# entraînement-validation. Ils sont produits respectivement par V10 et V19.
PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
INTERMEDIATE_DIR = PROJECT_DIR / "intermediate"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

CATBOOST_FILE = (
    INTERMEDIATE_DIR
    / "validation_predictions_catboost_depth7_max1500.csv"
)
LIGHTGBM_FILE = (
    INTERMEDIATE_DIR
    / "validation_predictions_lightgbm_grid_best.csv"
)

RESULTS_FILE = (
    RESULTS_DIR / "comparison_catboost_lightgbm_grid_best_blends.csv"
)
PREDICTIONS_FILE = (
    INTERMEDIATE_DIR
    / "validation_predictions_best_catboost_lightgbm_grid_blend.csv"
)
PARAMETERS_FILE = (
    RESULTS_DIR / "best_catboost_lightgbm_blend_parameters.json"
)


def check_prediction_file(data, required_columns, file_name):
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"Colonnes absentes dans {file_name}: {sorted(missing_columns)}"
        )

    if data["row_index"].duplicated().any():
        raise ValueError(f"Des row_index sont dupliqués dans {file_name}.")


def find_probability_column(data, expected_name, model_name, file_name):
    if expected_name in data.columns:
        return expected_name

    excluded_columns = {"row_index", "accepted"}
    keywords = ("prob", "predict", "score")

    candidates = [
        column
        for column in data.columns
        if column not in excluded_columns
        and any(keyword in column.lower() for keyword in keywords)
    ]

    if not candidates:
        numeric_candidates = [
            column
            for column in data.columns
            if column not in excluded_columns
            and pd.api.types.is_numeric_dtype(data[column])
        ]

        if len(numeric_candidates) == 1:
            candidates = numeric_candidates

    if len(candidates) != 1:
        raise ValueError(
            f"Impossible d'identifier de manière unique la colonne de "
            f"probabilités {model_name} dans {file_name}. "
            f"Colonnes disponibles : {data.columns.tolist()}. "
            f"Colonnes candidates : {candidates}."
        )

    detected_column = candidates[0]
    print(
        f"Colonne de probabilités {model_name} détectée : "
        f"{detected_column}"
    )
    return detected_column


if not CATBOOST_FILE.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {CATBOOST_FILE}. "
        "Exécutez d'abord V10_catboost_depth7_max1500_validation.py."
    )

if not LIGHTGBM_FILE.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {LIGHTGBM_FILE}. "
        "Exécutez d'abord V19_lightgbm_grid_54_validation.py."
    )


catboost_predictions = pd.read_csv(CATBOOST_FILE)
lightgbm_predictions = pd.read_csv(LIGHTGBM_FILE)

check_prediction_file(
    catboost_predictions,
    {"row_index", "accepted"},
    CATBOOST_FILE.name,
)
check_prediction_file(
    lightgbm_predictions,
    {"row_index", "accepted"},
    LIGHTGBM_FILE.name,
)

catboost_probability_column = find_probability_column(
    catboost_predictions,
    expected_name="catboost_probability",
    model_name="CatBoost",
    file_name=CATBOOST_FILE.name,
)
lightgbm_probability_column = find_probability_column(
    lightgbm_predictions,
    expected_name="lightgbm_probability",
    model_name="LightGBM",
    file_name=LIGHTGBM_FILE.name,
)

catboost_predictions = catboost_predictions.rename(
    columns={catboost_probability_column: "catboost_probability"}
)
lightgbm_predictions = lightgbm_predictions.rename(
    columns={lightgbm_probability_column: "lightgbm_probability"}
)

print("Dimensions des prédictions CatBoost :", catboost_predictions.shape)
print("Dimensions des prédictions LightGBM :", lightgbm_predictions.shape)


# La fusion par row_index garantit que les deux probabilités correspondent
# au même dossier de validation, même si les lignes ne sont pas dans le même ordre.
validation_predictions = catboost_predictions.merge(
    lightgbm_predictions,
    on="row_index",
    how="inner",
    suffixes=("_catboost", "_lightgbm"),
    validate="one_to_one",
)

if len(validation_predictions) != len(catboost_predictions):
    raise ValueError(
        "Certains dossiers CatBoost n'ont pas de prédiction LightGBM correspondante."
    )

if len(validation_predictions) != len(lightgbm_predictions):
    raise ValueError(
        "Certains dossiers LightGBM n'ont pas de prédiction CatBoost correspondante."
    )

targets_are_identical = np.array_equal(
    validation_predictions["accepted_catboost"].to_numpy(),
    validation_predictions["accepted_lightgbm"].to_numpy(),
)

print("Dimensions après fusion :", validation_predictions.shape)
print("Cibles identiques dans les deux fichiers :", targets_are_identical)

if not targets_are_identical:
    raise ValueError("Les cibles ne sont pas identiques dans les deux fichiers.")


y_validation = validation_predictions["accepted_catboost"].to_numpy()
catboost_probability = validation_predictions["catboost_probability"].to_numpy()
lightgbm_probability = validation_predictions["lightgbm_probability"].to_numpy()

for model_name, probabilities in {
    "CatBoost": catboost_probability,
    "LightGBM": lightgbm_probability,
}.items():
    if not np.isfinite(probabilities).all():
        raise ValueError(f"Les prédictions {model_name} contiennent NaN ou infini.")

    if not np.logical_and(probabilities >= 0, probabilities <= 1).all():
        raise ValueError(
            f"Les prédictions {model_name} ne sont pas toutes comprises entre 0 et 1."
        )


catboost_auc = roc_auc_score(y_validation, catboost_probability)
lightgbm_auc = roc_auc_score(y_validation, lightgbm_probability)

pearson_correlation = np.corrcoef(
    catboost_probability,
    lightgbm_probability,
)[0, 1]

spearman_correlation = pd.Series(catboost_probability).corr(
    pd.Series(lightgbm_probability),
    method="spearman",
)

print("\nModèles utilisés séparément")
print("AUC CatBoost :", catboost_auc)
print("AUC LightGBM optimisé :", lightgbm_auc)
print("Corrélation de Pearson :", pearson_correlation)
print("Corrélation de Spearman :", spearman_correlation)


# w représente le poids de CatBoost. Les valeurs 0 et 1 correspondent
# respectivement à LightGBM seul et à CatBoost seul.
catboost_weights = np.round(np.arange(0.0, 1.0001, 0.01), 2)
blend_results = []

for catboost_weight in catboost_weights:
    lightgbm_weight = 1.0 - catboost_weight

    blended_probability = (
        catboost_weight * catboost_probability
        + lightgbm_weight * lightgbm_probability
    )

    blend_auc = roc_auc_score(y_validation, blended_probability)

    blend_results.append(
        {
            "catboost_weight": catboost_weight,
            "lightgbm_weight": lightgbm_weight,
            "validation_auc": blend_auc,
            "difference_vs_catboost": blend_auc - catboost_auc,
            "difference_vs_lightgbm": blend_auc - lightgbm_auc,
        }
    )


results = pd.DataFrame(blend_results)
results["rank"] = (
    results["validation_auc"]
    .rank(method="min", ascending=False)
    .astype(int)
)

maximum_auc = results["validation_auc"].max()

# Plusieurs poids peuvent produire exactement la même AUC. Dans ce cas,
# on retient celui qui donne le plus de poids au meilleur modèle individuel.
best_candidates = results[
    np.isclose(
        results["validation_auc"],
        maximum_auc,
        rtol=0.0,
        atol=1e-12,
    )
]

best_result = best_candidates.sort_values(
    "catboost_weight",
    ascending=False,
).iloc[0]

best_catboost_weight = float(best_result["catboost_weight"])
best_lightgbm_weight = float(best_result["lightgbm_weight"])
best_blend_auc = float(best_result["validation_auc"])


print("\nDix meilleurs poids")
print(
    results.sort_values(
        ["validation_auc", "catboost_weight"],
        ascending=[False, False],
    )
    .head(10)
    .to_string(index=False)
)

print("\nMeilleur mélange")
print("Poids CatBoost :", best_catboost_weight)
print("Poids LightGBM :", best_lightgbm_weight)
print("AUC du mélange :", best_blend_auc)
print("AUC CatBoost seul :", catboost_auc)
print("Différence par rapport à CatBoost :", best_blend_auc - catboost_auc)
print(
    "Différence sur une échelle de 0 à 100 :",
    100 * (best_blend_auc - catboost_auc),
)


results.sort_values("catboost_weight").to_csv(
    RESULTS_FILE,
    index=False,
)

best_blended_probability = (
    best_catboost_weight * catboost_probability
    + best_lightgbm_weight * lightgbm_probability
)

best_validation_predictions = pd.DataFrame(
    {
        "row_index": validation_predictions["row_index"],
        "accepted": y_validation,
        "catboost_probability": catboost_probability,
        "lightgbm_probability": lightgbm_probability,
        "blended_probability": best_blended_probability,
    }
)

best_validation_predictions.to_csv(
    PREDICTIONS_FILE,
    index=False,
)

best_parameters = {
    "catboost_weight": best_catboost_weight,
    "lightgbm_weight": best_lightgbm_weight,
    "catboost_auc": float(catboost_auc),
    "lightgbm_auc": float(lightgbm_auc),
    "blend_auc": best_blend_auc,
    "difference_vs_catboost": float(best_blend_auc - catboost_auc),
    "weight_step": 0.01,
}

with open(PARAMETERS_FILE, "w", encoding="utf-8") as file:
    json.dump(best_parameters, file, indent=4, ensure_ascii=False)

print("\nFichiers créés :")
print("-", RESULTS_FILE)
print("-", PREDICTIONS_FILE)
print("-", PARAMETERS_FILE)
