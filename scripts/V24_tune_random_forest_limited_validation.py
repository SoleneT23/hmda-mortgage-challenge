"""Compare six configurations de forêt aléatoire sur une validation fixe.

Le prétraitement est ajusté une seule fois sur l'ensemble d'entraînement.
Les configurations sont sélectionnées sur la même validation et devront être
confirmées ultérieurement si elles deviennent compétitives.
"""

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42
NUMBER_OF_THREADS = 2
NUMBER_OF_TREES = 300
CATBOOST_REFERENCE_VALIDATION_AUC = 0.9118183305741654

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
INTERMEDIATE_DIR = PROJECT_DIR / "intermediate"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = (
    RESULTS_DIR / "random_forest_limited_search_results.csv"
)
PARAMETERS_FILE = (
    RESULTS_DIR / "best_random_forest_limited_parameters.json"
)
PREDICTIONS_FILE = (
    INTERMEDIATE_DIR
    / "validation_predictions_best_random_forest.csv"
)


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


# Préparation des données
train = pd.read_csv(DATA_DIR / "train.csv")
train = add_features(train)

row_ids = train["id"].copy()
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

for column in categorical_columns:
    X[column] = (
        X[column]
        .astype("string")
        .fillna("__MISSING__")
    )


training_indices, validation_indices = train_test_split(
    np.arange(len(X)),
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE,
)

X_training = X.iloc[training_indices].copy()
X_validation = X.iloc[validation_indices].copy()
y_training = y.iloc[training_indices]
y_validation = y.iloc[validation_indices]


# Prétraitement ajusté uniquement sur l'ensemble d'entraînement
numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median"),
    ),
])

categorical_pipeline = Pipeline([
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float32,
        ),
    ),
])

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_columns,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns,
        ),
    ],
    sparse_threshold=1.0,
)


print("Dimensions de X :", X.shape)
print("Dossiers d'entraînement :", len(X_training))
print("Dossiers de validation :", len(X_validation))
print("Variables numériques :", len(numeric_columns))
print("Variables catégorielles :", len(categorical_columns))

print("\nPrétraitement et encodage one-hot")

preprocessing_start = time.perf_counter()

X_training_encoded = preprocessor.fit_transform(X_training)
X_validation_encoded = preprocessor.transform(X_validation)

preprocessing_seconds = time.perf_counter() - preprocessing_start

print("Dimensions du train encodé :", X_training_encoded.shape)
print("Dimensions de la validation encodée :", X_validation_encoded.shape)
print("Durée du prétraitement en secondes :", preprocessing_seconds)


# Configurations comparées sur la même validation
configurations = [
    {
        "configuration": "baseline",
        "max_depth": None,
        "min_samples_leaf": 2,
    },
    {
        "configuration": "depth_25_leaf_2",
        "max_depth": 25,
        "min_samples_leaf": 2,
    },
    {
        "configuration": "depth_20_leaf_5",
        "max_depth": 20,
        "min_samples_leaf": 5,
    },
    {
        "configuration": "depth_15_leaf_5",
        "max_depth": 15,
        "min_samples_leaf": 5,
    },
    {
        "configuration": "depth_20_leaf_10",
        "max_depth": 20,
        "min_samples_leaf": 10,
    },
    {
        "configuration": "depth_15_leaf_10",
        "max_depth": 15,
        "min_samples_leaf": 10,
    },
]

search_results = []
best_result = None
best_validation_probability = None


for configuration_number, parameters in enumerate(
    configurations,
    start=1,
):
    print()
    print(
        f"Configuration {configuration_number} / {len(configurations)} : "
        f"{parameters['configuration']}"
    )
    print("max_depth :", parameters["max_depth"])
    print("min_samples_leaf :", parameters["min_samples_leaf"])

    model = RandomForestClassifier(
        n_estimators=NUMBER_OF_TREES,
        max_depth=parameters["max_depth"],
        min_samples_leaf=parameters["min_samples_leaf"],
        max_features="sqrt",
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=NUMBER_OF_THREADS,
        verbose=0,
    )

    training_start = time.perf_counter()
    model.fit(X_training_encoded, y_training)
    training_seconds = time.perf_counter() - training_start

    training_probability = model.predict_proba(
        X_training_encoded
    )[:, 1]

    validation_probability = model.predict_proba(
        X_validation_encoded
    )[:, 1]

    if not np.isfinite(training_probability).all():
        raise ValueError(
            "Les prédictions d'entraînement ne sont pas finies."
        )

    if not np.isfinite(validation_probability).all():
        raise ValueError(
            "Les prédictions de validation ne sont pas finies."
        )

    training_auc = roc_auc_score(
        y_training,
        training_probability,
    )

    validation_auc = roc_auc_score(
        y_validation,
        validation_probability,
    )

    result = {
        "configuration_number": configuration_number,
        "configuration": parameters["configuration"],
        "n_estimators": NUMBER_OF_TREES,
        "max_depth": (
            "None"
            if parameters["max_depth"] is None
            else parameters["max_depth"]
        ),
        "min_samples_leaf": parameters["min_samples_leaf"],
        "max_features": "sqrt",
        "bootstrap": True,
        "training_seconds": training_seconds,
        "train_auc": training_auc,
        "validation_auc": validation_auc,
        "train_validation_gap": training_auc - validation_auc,
        "difference_vs_catboost": (
            validation_auc - CATBOOST_REFERENCE_VALIDATION_AUC
        ),
    }

    search_results.append(result)

    if (
        best_result is None
        or validation_auc > best_result["validation_auc"]
    ):
        best_result = result.copy()
        best_validation_probability = validation_probability.copy()

    print("Durée d'entraînement :", training_seconds)
    print("AUC entraînement :", training_auc)
    print("AUC validation :", validation_auc)
    print("Écart entraînement - validation :", training_auc - validation_auc)
    print(
        "Différence par rapport à CatBoost :",
        validation_auc - CATBOOST_REFERENCE_VALIDATION_AUC,
    )

    del model
    del training_probability
    del validation_probability
    gc.collect()


# Sauvegarde des résultats et des prédictions de la meilleure configuration
results = pd.DataFrame(search_results)
results["rank"] = (
    results["validation_auc"]
    .rank(method="min", ascending=False)
    .astype(int)
)

results = results.sort_values(
    ["validation_auc", "configuration_number"],
    ascending=[False, True],
).reset_index(drop=True)

results.to_csv(RESULTS_FILE, index=False)


best_parameters = {
    "configuration": best_result["configuration"],
    "n_estimators": NUMBER_OF_TREES,
    "max_depth": (
        None
        if best_result["max_depth"] == "None"
        else int(best_result["max_depth"])
    ),
    "min_samples_leaf": int(best_result["min_samples_leaf"]),
    "max_features": "sqrt",
    "bootstrap": True,
    "train_auc": float(best_result["train_auc"]),
    "validation_auc": float(best_result["validation_auc"]),
    "train_validation_gap": float(
        best_result["train_validation_gap"]
    ),
    "difference_vs_catboost": float(
        best_result["difference_vs_catboost"]
    ),
}

with open(PARAMETERS_FILE, "w", encoding="utf-8") as file:
    json.dump(
        best_parameters,
        file,
        indent=4,
        ensure_ascii=False,
    )


validation_predictions = pd.DataFrame({
    "row_index": validation_indices,
    "id": row_ids.iloc[validation_indices].to_numpy(),
    "accepted": y_validation.to_numpy(),
    "random_forest_probability": best_validation_probability,
})

validation_predictions.to_csv(
    PREDICTIONS_FILE,
    index=False,
)


print("\nComparaison des configurations")
print(results.to_string(index=False))

print("\nMeilleure configuration :", best_result["configuration"])
print("Meilleure AUC validation :", best_result["validation_auc"])
print("AUC entraînement associée :", best_result["train_auc"])
print(
    "Écart entraînement - validation :",
    best_result["train_validation_gap"],
)
print(
    "Différence par rapport à CatBoost :",
    best_result["difference_vs_catboost"],
)

print("\nFichiers créés :")
print("-", RESULTS_FILE)
print("-", PARAMETERS_FILE)
print("-", PREDICTIONS_FILE)
