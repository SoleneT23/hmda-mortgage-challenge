"""Confirme par validation croisée le mélange CatBoost--LightGBM 86/14.

Les hyperparamètres et les poids sont fixés avant la validation croisée.
Chaque dossier reçoit ainsi une prédiction hors échantillon provenant d'un
modèle qui n'a pas été entraîné sur ce dossier.
"""

import gc
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


RANDOM_STATE = 42
NUMBER_OF_FOLDS = 5
NUMBER_OF_THREADS = 2

CATBOOST_WEIGHT = 0.86
LIGHTGBM_WEIGHT = 0.14

CATBOOST_NUMBER_OF_TREES = 1302
LIGHTGBM_NUMBER_OF_TREES = 125

# Les chemins sont construits à partir de l'emplacement du script.
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
INTERMEDIATE_DIR = PROJECT_DIR / "intermediate"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
FOLD_RESULTS_FILE = (
    RESULTS_DIR / "comparison_catboost_lightgbm_blend_5fold.csv"
)
OOF_PREDICTIONS_FILE = (
    INTERMEDIATE_DIR
    / "oof_predictions_catboost_lightgbm_blend_5fold.csv"
)


def add_features(data):
    """Ajoute les trois variables utilisées par les modèles précédents."""

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


def prepare_catboost_data(
    data,
    numeric_columns,
    categorical_columns,
):
    """Prépare un sous-ensemble pour CatBoost."""

    prepared_data = data.copy()

    prepared_data[numeric_columns] = (
        prepared_data[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    for column in categorical_columns:
        prepared_data[column] = (
            prepared_data[column]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    return prepared_data


def prepare_lightgbm_fold(
    training_data,
    validation_data,
    numeric_columns,
    categorical_columns,
):
    """Définit les catégories à partir du fold d'entraînement uniquement."""

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

        categories = pd.Index(training_values.unique())

        prepared_training[column] = pd.Categorical(
            training_values,
            categories=categories,
        )

        # Une modalité absente de l'entraînement devient une valeur manquante
        # dans la validation, que LightGBM sait traiter.
        prepared_validation[column] = pd.Categorical(
            validation_values,
            categories=categories,
        )

    return prepared_training, prepared_validation


def build_catboost_model():
    """Construit la configuration CatBoost retenue."""

    return CatBoostClassifier(
        iterations=CATBOOST_NUMBER_OF_TREES,
        depth=7,
        learning_rate=0.05,
        l2_leaf_reg=3,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        thread_count=NUMBER_OF_THREADS,
        verbose=200,
        allow_writing_files=False,
    )


def build_lightgbm_model():
    """Construit la meilleure configuration de la grille LightGBM."""

    return LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=LIGHTGBM_NUMBER_OF_TREES,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=NUMBER_OF_THREADS,
        importance_type="gain",
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


# Chargement des données
if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Le fichier de données est introuvable : {TRAIN_PATH}"
    )

train = pd.read_csv(TRAIN_PATH)
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

print("Version de CatBoost :", importlib.metadata.version("catboost"))
print("Version de LightGBM :", importlib.metadata.version("lightgbm"))
print("Dimensions de X :", X.shape)
print("Dimensions de y :", y.shape)
print("Variables numériques :", len(numeric_columns))
print("Variables catégorielles :", len(categorical_columns))
print("Nombre de folds :", NUMBER_OF_FOLDS)
print("Poids CatBoost fixé avant la CV :", CATBOOST_WEIGHT)
print("Poids LightGBM fixé avant la CV :", LIGHTGBM_WEIGHT)
print("Arbres CatBoost :", CATBOOST_NUMBER_OF_TREES)
print("Arbres LightGBM :", LIGHTGBM_NUMBER_OF_TREES)


cross_validation = StratifiedKFold(
    n_splits=NUMBER_OF_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

catboost_oof_probability = np.full(len(X), np.nan)
lightgbm_oof_probability = np.full(len(X), np.nan)
blend_oof_probability = np.full(len(X), np.nan)
fold_assignment = np.full(len(X), -1, dtype=int)

fold_results = []


for fold, (training_indices, validation_indices) in enumerate(
    cross_validation.split(X, y),
    start=1,
):
    print(f"\nFold {fold} / {NUMBER_OF_FOLDS}")

    X_training_raw = X.iloc[training_indices].copy()
    X_validation_raw = X.iloc[validation_indices].copy()
    y_training = y.iloc[training_indices]
    y_validation = y.iloc[validation_indices]

    print("Dossiers d'entraînement :", len(training_indices))
    print("Dossiers de validation :", len(validation_indices))
    print(
        "Proportion de prêts accordés dans la validation :",
        y_validation.mean(),
    )

    X_training_catboost = prepare_catboost_data(
        X_training_raw,
        numeric_columns,
        categorical_columns,
    )

    X_validation_catboost = prepare_catboost_data(
        X_validation_raw,
        numeric_columns,
        categorical_columns,
    )

    catboost_model = build_catboost_model()
    catboost_model.fit(
        X_training_catboost,
        y_training,
        cat_features=categorical_columns,
    )

    catboost_probability = catboost_model.predict_proba(
        X_validation_catboost
    )[:, 1]

    X_training_lightgbm, X_validation_lightgbm = (
        prepare_lightgbm_fold(
            X_training_raw,
            X_validation_raw,
            numeric_columns,
            categorical_columns,
        )
    )

    lightgbm_model = build_lightgbm_model()
    lightgbm_model.fit(
        X_training_lightgbm,
        y_training,
    )

    lightgbm_probability = lightgbm_model.predict_proba(
        X_validation_lightgbm
    )[:, 1]

    blend_probability = (
        CATBOOST_WEIGHT * catboost_probability
        + LIGHTGBM_WEIGHT * lightgbm_probability
    )

    catboost_auc = roc_auc_score(
        y_validation,
        catboost_probability,
    )

    lightgbm_auc = roc_auc_score(
        y_validation,
        lightgbm_probability,
    )

    blend_auc = roc_auc_score(
        y_validation,
        blend_probability,
    )

    auc_difference = blend_auc - catboost_auc

    pearson_correlation = np.corrcoef(
        catboost_probability,
        lightgbm_probability,
    )[0, 1]

    spearman_correlation = pd.Series(
        catboost_probability
    ).corr(
        pd.Series(lightgbm_probability),
        method="spearman",
    )

    catboost_oof_probability[validation_indices] = catboost_probability
    lightgbm_oof_probability[validation_indices] = lightgbm_probability
    blend_oof_probability[validation_indices] = blend_probability
    fold_assignment[validation_indices] = fold

    fold_results.append({
        "fold": fold,
        "number_of_training_rows": len(training_indices),
        "number_of_validation_rows": len(validation_indices),
        "catboost_auc": catboost_auc,
        "lightgbm_auc": lightgbm_auc,
        "blend_86_14_auc": blend_auc,
        "blend_minus_catboost": auc_difference,
        "pearson_correlation": pearson_correlation,
        "spearman_correlation": spearman_correlation,
    })

    print("AUC CatBoost :", catboost_auc)
    print("AUC LightGBM :", lightgbm_auc)
    print("AUC du mélange 86/14 :", blend_auc)
    print("Différence mélange - CatBoost :", auc_difference)

    del catboost_model
    del lightgbm_model
    del X_training_catboost
    del X_validation_catboost
    del X_training_lightgbm
    del X_validation_lightgbm
    gc.collect()


if (
    np.isnan(catboost_oof_probability).any()
    or np.isnan(lightgbm_oof_probability).any()
    or np.isnan(blend_oof_probability).any()
    or (fold_assignment == -1).any()
):
    raise ValueError(
        "Chaque observation doit recevoir une prédiction hors échantillon."
    )

for model_name, probabilities in {
    "CatBoost": catboost_oof_probability,
    "LightGBM": lightgbm_oof_probability,
    "mélange": blend_oof_probability,
}.items():
    if not np.isfinite(probabilities).all():
        raise ValueError(
            f"Les prédictions {model_name} contiennent NaN ou infini."
        )

    if not np.logical_and(
        probabilities >= 0,
        probabilities <= 1,
    ).all():
        raise ValueError(
            f"Les prédictions {model_name} ne sont pas toutes entre 0 et 1."
        )


results = pd.DataFrame(fold_results)
results.to_csv(FOLD_RESULTS_FILE, index=False)

mean_catboost_auc = results["catboost_auc"].mean()
mean_lightgbm_auc = results["lightgbm_auc"].mean()
mean_blend_auc = results["blend_86_14_auc"].mean()

auc_differences = results["blend_minus_catboost"]
mean_auc_difference = auc_differences.mean()
standard_deviation_of_differences = auc_differences.std(ddof=1)
improved_folds = int((auc_differences > 0).sum())

catboost_oof_auc = roc_auc_score(
    y,
    catboost_oof_probability,
)

lightgbm_oof_auc = roc_auc_score(
    y,
    lightgbm_oof_probability,
)

blend_oof_auc = roc_auc_score(
    y,
    blend_oof_probability,
)

oof_auc_difference = blend_oof_auc - catboost_oof_auc

oof_predictions = pd.DataFrame({
    "row_index": np.arange(len(X)),
    "id": row_ids.to_numpy(),
    "fold": fold_assignment,
    "accepted": y.to_numpy(),
    "catboost_probability": catboost_oof_probability,
    "lightgbm_probability": lightgbm_oof_probability,
    "blend_86_14_probability": blend_oof_probability,
})

oof_predictions.to_csv(
    OOF_PREDICTIONS_FILE,
    index=False,
)


print("\nRésultats de la validation croisée")
print(results.to_string(index=False))

print("\nMoyenne des AUC CatBoost :", mean_catboost_auc)
print("Moyenne des AUC LightGBM :", mean_lightgbm_auc)
print("Moyenne des AUC du mélange :", mean_blend_auc)
print("Moyenne des différences :", mean_auc_difference)
print(
    "Écart-type des différences :",
    standard_deviation_of_differences,
)
print(
    "Nombre de folds où le mélange améliore CatBoost :",
    f"{improved_folds} / {NUMBER_OF_FOLDS}",
)

print("\nAUC OOF CatBoost :", catboost_oof_auc)
print("AUC OOF LightGBM :", lightgbm_oof_auc)
print("AUC OOF du mélange 86/14 :", blend_oof_auc)
print("Différence OOF mélange - CatBoost :", oof_auc_difference)

print("\nFichiers créés :")
print("-", FOLD_RESULTS_FILE)
print("-", OOF_PREDICTIONS_FILE)

if improved_folds == NUMBER_OF_FOLDS:
    print(
        "Conclusion : le mélange améliore CatBoost sur les cinq folds."
    )
elif improved_folds == 0:
    print(
        "Conclusion : le mélange n'améliore CatBoost sur aucun fold."
    )
else:
    print(
        "Conclusion : l'amélioration n'est pas constante selon les folds."
    )
