"""Teste une grille de 54 configurations LightGBM sur une validation fixe.

La séparation entraînement-validation est identique à celle des expériences
précédentes. Le nombre d'arbres est choisi par arrêt anticipé pour chaque
configuration.
"""

import gc
import importlib.metadata
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
NUMBER_OF_THREADS = 2
MAXIMUM_NUMBER_OF_TREES = 5000
EARLY_STOPPING_ROUNDS = 100

INITIAL_LIGHTGBM_VALIDATION_AUC = 0.9011402452453896
CATBOOST_REFERENCE_VALIDATION_AUC = 0.9118183305741654

# Les chemins sont construits à partir de l'emplacement du script afin que
# celui-ci puisse être lancé depuis n'importe quel dossier.
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
INTERMEDIATE_DIR = PROJECT_DIR / "intermediate"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"

RESULTS_FILE = RESULTS_DIR / "lightgbm_grid_54_results.csv"
PARAMETERS_FILE = RESULTS_DIR / "best_lightgbm_grid_parameters.json"
PREDICTIONS_FILE = (
    INTERMEDIATE_DIR / "validation_predictions_lightgbm_grid_best.csv"
)
IMPORTANCE_FILE = RESULTS_DIR / "feature_importance_lightgbm_grid_best.csv"
MODEL_FILE = INTERMEDIATE_DIR / "lightgbm_grid_best_validation.txt"


def add_features(data):
    """Ajoute les trois variables utilisées dans les modèles précédents."""

    data = data.copy()

    data["ltv_missing"] = (
        data["loan_to_value_ratio"].isna().astype(int)
    )

    data["ltv_above_100"] = (
        data["loan_to_value_ratio"] > 100
    ).astype(int)

    data["loan_property_ratio"] = (
        data["loan_amount"]
        / data["property_value"]
        * 100
    )

    data["loan_property_ratio"] = (
        data["loan_property_ratio"]
        .replace([np.inf, -np.inf], np.nan)
    )

    return data


def prepare_lightgbm_data(
    training_data,
    validation_data,
    numeric_columns,
    categorical_columns,
):
    """Prépare les catégories sans utiliser les valeurs de validation."""

    prepared_training = training_data.copy()
    prepared_validation = validation_data.copy()

    prepared_training[numeric_columns] = (
        prepared_training[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    prepared_validation[numeric_columns] = (
        prepared_validation[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    for column in categorical_columns:
        training_values = (
            prepared_training[column]
            .astype("string")
            .fillna("__MISSING__")
        )

        validation_values = (
            prepared_validation[column]
            .astype("string")
            .fillna("__MISSING__")
        )

        prepared_training[column] = training_values.astype("category")

        prepared_validation[column] = pd.Categorical(
            validation_values,
            categories=prepared_training[column].cat.categories,
        )

    return prepared_training, prepared_validation


def build_model(parameters):
    """Construit une configuration de la grille."""

    return LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=MAXIMUM_NUMBER_OF_TREES,
        learning_rate=parameters["learning_rate"],
        num_leaves=parameters["num_leaves"],
        max_depth=-1,
        min_child_samples=parameters["min_child_samples"],
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=parameters["reg_lambda"],
        random_state=RANDOM_STATE,
        n_jobs=NUMBER_OF_THREADS,
        importance_type="gain",
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def evaluate_configuration(
    configuration_number,
    total_configurations,
    parameters,
    X_training,
    y_training,
    X_validation,
    y_validation,
):
    """Entraîne une configuration et calcule ses AUC."""

    print(
        f"\nConfiguration {configuration_number} / {total_configurations}"
    )
    print(parameters)

    model = build_model(parameters)
    starting_time = time.perf_counter()

    # Les colonnes pandas de type category sont détectées automatiquement
    # par LightGBM ; il n'est pas nécessaire de les redéclarer ici.
    model.fit(
        X_training,
        y_training,
        eval_set=[(X_validation, y_validation)],
        eval_metric="auc",
        callbacks=[
            early_stopping(
                stopping_rounds=EARLY_STOPPING_ROUNDS,
                first_metric_only=True,
                verbose=False,
            ),
            log_evaluation(period=0),
        ],
    )

    training_seconds = time.perf_counter() - starting_time
    best_iteration = int(model.best_iteration_)

    training_probabilities = model.predict_proba(
        X_training,
        num_iteration=best_iteration,
    )[:, 1]

    validation_probabilities = model.predict_proba(
        X_validation,
        num_iteration=best_iteration,
    )[:, 1]

    training_auc = roc_auc_score(
        y_training,
        training_probabilities,
    )

    validation_auc = roc_auc_score(
        y_validation,
        validation_probabilities,
    )

    result = {
        "configuration_number": configuration_number,
        **parameters,
        "best_iteration": best_iteration,
        "train_auc": training_auc,
        "validation_auc": validation_auc,
        "train_validation_gap": training_auc - validation_auc,
        "difference_vs_initial_lightgbm": (
            validation_auc - INITIAL_LIGHTGBM_VALIDATION_AUC
        ),
        "difference_vs_catboost": (
            validation_auc - CATBOOST_REFERENCE_VALIDATION_AUC
        ),
        "training_seconds": training_seconds,
    }

    print("Meilleure itération :", best_iteration)
    print("AUC entraînement :", training_auc)
    print("AUC validation :", validation_auc)
    print("Écart entraînement - validation :", training_auc - validation_auc)
    print("Durée en secondes :", training_seconds)

    return model, validation_probabilities, result


# Chargement et préparation des données
if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Le fichier de données est introuvable : {TRAIN_PATH}"
    )

train = pd.read_csv(TRAIN_PATH)
train = add_features(train)

X = train.drop(columns=["id", "accepted"])
y = train["accepted"]

numeric_columns = [
    "loan_amount",
    "loan_to_value_ratio",
    "property_value",
    "income",
    "loan_term",
    "tract_population",
    "tract_minority_population_percent",
    "ffiec_msa_md_median_family_income",
    "tract_to_msa_income_percentage",
    "tract_owner_occupied_units",
    "tract_one_to_four_family_homes",
    "tract_median_age_of_housing_units",
    "ltv_missing",
    "ltv_above_100",
    "loan_property_ratio",
]

categorical_columns = [
    column
    for column in X.columns
    if column not in numeric_columns
]

X_training_raw, X_validation_raw, y_training, y_validation = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
)

X_training, X_validation = prepare_lightgbm_data(
    X_training_raw,
    X_validation_raw,
    numeric_columns,
    categorical_columns,
)

print("Version de LightGBM :", importlib.metadata.version("lightgbm"))
print("Dimensions de X :", X.shape)
print("Dossiers d'entraînement :", len(X_training))
print("Dossiers de validation :", len(X_validation))
print("Variables numériques :", len(numeric_columns))
print("Variables catégorielles :", len(categorical_columns))


parameter_grid = list(
    itertools.product(
        [0.02, 0.03],
        [15, 31, 63],
        [20, 50, 100],
        [0.0, 1.0, 5.0],
    )
)

total_configurations = len(parameter_grid)

if total_configurations != 54:
    raise ValueError(
        "La grille doit contenir exactement 54 configurations."
    )

print("Nombre de configurations :", total_configurations)

search_results = []
best_model = None
best_validation_probabilities = None
best_result = None


for configuration_number, values in enumerate(
    parameter_grid,
    start=1,
):
    learning_rate, num_leaves, min_child_samples, reg_lambda = values

    parameters = {
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "reg_lambda": reg_lambda,
    }

    model, validation_probabilities, result = evaluate_configuration(
        configuration_number=configuration_number,
        total_configurations=total_configurations,
        parameters=parameters,
        X_training=X_training,
        y_training=y_training,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    search_results.append(result)

    if (
        best_result is None
        or result["validation_auc"] > best_result["validation_auc"]
    ):
        best_model = model
        best_validation_probabilities = validation_probabilities
        best_result = result
    else:
        del model

    # Le fichier est mis à jour après chaque configuration afin de conserver
    # les résultats déjà calculés en cas d'interruption du programme.
    pd.DataFrame(search_results).to_csv(
        RESULTS_FILE,
        index=False,
    )

    gc.collect()


results = pd.DataFrame(search_results).sort_values(
    by="validation_auc",
    ascending=False,
).reset_index(drop=True)

results.insert(0, "rank", np.arange(1, len(results) + 1))
results.to_csv(RESULTS_FILE, index=False)

best_parameters = {
    "learning_rate": float(best_result["learning_rate"]),
    "num_leaves": int(best_result["num_leaves"]),
    "min_child_samples": int(best_result["min_child_samples"]),
    "reg_lambda": float(best_result["reg_lambda"]),
    "best_iteration": int(best_result["best_iteration"]),
    "validation_auc": float(best_result["validation_auc"]),
}

with open(PARAMETERS_FILE, "w", encoding="utf-8") as file:
    json.dump(
        best_parameters,
        file,
        ensure_ascii=False,
        indent=4,
    )

validation_predictions = pd.DataFrame({
    "row_index": X_validation.index,
    "accepted": y_validation.to_numpy(),
    "lightgbm_probability": best_validation_probabilities,
})

validation_predictions.to_csv(
    PREDICTIONS_FILE,
    index=False,
)

best_iteration = int(best_result["best_iteration"])

gain_importances = best_model.booster_.feature_importance(
    importance_type="gain",
    iteration=best_iteration,
)

total_gain = gain_importances.sum()

if total_gain > 0:
    normalized_importances = gain_importances / total_gain * 100
else:
    normalized_importances = gain_importances

feature_importance = pd.DataFrame({
    "variable": X_training.columns,
    "importance_gain": gain_importances,
    "importance_percent": normalized_importances,
}).sort_values(
    by="importance_percent",
    ascending=False,
).reset_index(drop=True)

feature_importance.to_csv(
    IMPORTANCE_FILE,
    index=False,
)

best_model.booster_.save_model(
    MODEL_FILE,
    num_iteration=best_iteration,
)

print("\nDix meilleures configurations")
print(
    results[
        [
            "rank",
            "learning_rate",
            "num_leaves",
            "min_child_samples",
            "reg_lambda",
            "best_iteration",
            "train_auc",
            "validation_auc",
            "train_validation_gap",
        ]
    ].head(10).to_string(index=False)
)

print("\nMeilleure configuration")
print(json.dumps(best_parameters, ensure_ascii=False, indent=4))
print(
    "Amélioration par rapport au LightGBM initial :",
    best_result["difference_vs_initial_lightgbm"],
)
print(
    "Différence par rapport au CatBoost de référence :",
    best_result["difference_vs_catboost"],
)

print("\nFichiers créés :")
print("-", RESULTS_FILE)
print("-", PARAMETERS_FILE)
print("-", PREDICTIONS_FILE)
print("-", IMPORTANCE_FILE)
print("-", MODEL_FILE)
