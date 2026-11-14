"""Entraîne CatBoost et LightGBM sur toutes les données puis soumet leur mélange."""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


RANDOM_STATE = 42
NUMBER_OF_THREADS = 2

CATBOOST_WEIGHT = 0.86
LIGHTGBM_WEIGHT = 0.14

CATBOOST_NUMBER_OF_TREES = 1302
LIGHTGBM_NUMBER_OF_TREES = 125

# Le script peut être lancé depuis n'importe quel dossier.
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
SUBMISSIONS_DIR = PROJECT_DIR / "submissions"

SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_FILE = (
    SUBMISSIONS_DIR / "submission_catboost86_lightgbm14_optimized.csv"
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


def prepare_catboost_data(
    data,
    numeric_columns,
    categorical_columns,
):
    """Prépare les variables numériques et catégorielles pour CatBoost."""

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


def prepare_lightgbm_data(
    training_data,
    test_data,
    numeric_columns,
    categorical_columns,
):
    """Définit les catégories de LightGBM à partir du train uniquement."""

    prepared_training = training_data.copy()
    prepared_test = test_data.copy()

    prepared_training[numeric_columns] = (
        prepared_training[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    prepared_test[numeric_columns] = (
        prepared_test[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    for column in categorical_columns:
        training_values = (
            prepared_training[column]
            .astype("string")
            .fillna("__MISSING__")
        )

        test_values = (
            prepared_test[column]
            .astype("string")
            .fillna("__MISSING__")
        )

        categories = pd.Index(training_values.unique())

        prepared_training[column] = pd.Categorical(
            training_values,
            categories=categories,
        )

        # Une catégorie du test absente du train devient une valeur manquante.
        prepared_test[column] = pd.Categorical(
            test_values,
            categories=categories,
        )

    return prepared_training, prepared_test


def build_catboost_model():
    """Construit la configuration CatBoost retenue après validation."""

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
required_files = [TRAIN_PATH, TEST_PATH, SAMPLE_SUBMISSION_PATH]
missing_files = [path for path in required_files if not path.exists()]

if missing_files:
    missing_names = ", ".join(str(path) for path in missing_files)
    raise FileNotFoundError(
        "Fichiers de données introuvables : " + missing_names
    )

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

print("Dimensions du train avant feature engineering :", train.shape)
print("Dimensions du test avant feature engineering :", test.shape)

test_ids = test["id"].copy()

train = add_features(train)
test = add_features(test)

X = train.drop(columns=["id", "accepted"])
y = train["accepted"]
X_test = test.drop(columns=["id"])

if list(X.columns) != list(X_test.columns):
    raise ValueError(
        "Les variables du train et du test ne sont pas identiques."
    )

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

print("Dimensions de X :", X.shape)
print("Dimensions de X_test :", X_test.shape)
print("Variables numériques :", len(numeric_columns))
print("Variables catégorielles :", len(categorical_columns))


print("\nEntraînement de CatBoost sur les 105 000 dossiers")

X_catboost = prepare_catboost_data(
    X,
    numeric_columns,
    categorical_columns,
)

X_test_catboost = prepare_catboost_data(
    X_test,
    numeric_columns,
    categorical_columns,
)

catboost_model = build_catboost_model()
catboost_model.fit(
    X_catboost,
    y,
    cat_features=categorical_columns,
)

catboost_test_probability = catboost_model.predict_proba(
    X_test_catboost
)[:, 1]

print("Nombre d'arbres CatBoost :", catboost_model.tree_count_)

del catboost_model
del X_catboost
del X_test_catboost
gc.collect()


print("\nEntraînement de LightGBM sur les 105 000 dossiers")

X_lightgbm, X_test_lightgbm = prepare_lightgbm_data(
    X,
    X_test,
    numeric_columns,
    categorical_columns,
)

lightgbm_model = build_lightgbm_model()
lightgbm_model.fit(X_lightgbm, y)

lightgbm_test_probability = lightgbm_model.predict_proba(
    X_test_lightgbm
)[:, 1]

print("Nombre d'arbres LightGBM :", lightgbm_model.n_estimators)

del lightgbm_model
del X_lightgbm
del X_test_lightgbm
gc.collect()


blend_test_probability = (
    CATBOOST_WEIGHT * catboost_test_probability
    + LIGHTGBM_WEIGHT * lightgbm_test_probability
)

print("\nVérification des prédictions")
print("Nombre de prédictions :", len(blend_test_probability))
print("Probabilité minimale :", blend_test_probability.min())
print("Probabilité maximale :", blend_test_probability.max())
print("Valeurs manquantes :", np.isnan(blend_test_probability).sum())
print(
    "Corrélation entre les deux modèles :",
    np.corrcoef(
        catboost_test_probability,
        lightgbm_test_probability,
    )[0, 1],
)

predictions_by_id = pd.DataFrame({
    "id": test_ids,
    "accepted": blend_test_probability,
})

submission = (
    sample_submission[["id"]]
    .merge(
        predictions_by_id,
        on="id",
        how="left",
        validate="one_to_one",
    )
)

if list(submission.columns) != ["id", "accepted"]:
    raise ValueError("Les colonnes de la soumission sont incorrectes.")

if len(submission) != len(sample_submission):
    raise ValueError("Le nombre de lignes de la soumission est incorrect.")

if submission["id"].duplicated().any():
    raise ValueError("La soumission contient des identifiants dupliqués.")

if submission["accepted"].isna().any():
    raise ValueError("La soumission contient des prédictions manquantes.")

if not np.isfinite(submission["accepted"]).all():
    raise ValueError("La soumission contient une valeur non finie.")

if not submission["accepted"].between(0, 1).all():
    raise ValueError("Certaines probabilités ne sont pas entre 0 et 1.")

if not submission["id"].equals(sample_submission["id"]):
    raise ValueError(
        "L'ordre des identifiants diffère de sample_submission.csv."
    )

submission.to_csv(OUTPUT_FILE, index=False)

submission_check = pd.read_csv(OUTPUT_FILE)

print("\nDimensions de la soumission :", submission.shape)
print("\nPremières prédictions :")
print(submission.head())
print("\nFichier créé :", OUTPUT_FILE)
print("Vérification du fichier enregistré :", submission_check.shape)
