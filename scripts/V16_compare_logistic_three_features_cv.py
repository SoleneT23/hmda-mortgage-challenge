"""Compare deux régressions logistiques et crée leurs soumissions Kaggle.

Modèle A : variables d'origine uniquement.
Modèle B : variables d'origine et trois variables créées à partir du LTV.

La comparaison repose sur une validation croisée stratifiée à trois folds.
Les deux modèles utilisent exactement les mêmes folds et le même pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Le script peut être lancé depuis n'importe quel dossier. Les chemins sont
# construits à partir de son emplacement dans le dépôt.
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
SUBMISSIONS_DIR = PROJECT_DIR / "submissions"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

N_SPLITS = 3
RANDOM_STATE = 42
MAX_ITER = 2000


def add_engineered_features(data):
    """Ajoute les trois variables étudiées."""

    data_with_features = data.copy()

    data_with_features["ltv_missing"] = (
        data_with_features["loan_to_value_ratio"]
        .isna()
        .astype(int)
    )

    data_with_features["ltv_above_100"] = (
        data_with_features["loan_to_value_ratio"] > 100
    ).astype(int)

    data_with_features["loan_property_ratio"] = (
        data_with_features["loan_amount"]
        / data_with_features["property_value"]
        * 100
    )

    data_with_features["loan_property_ratio"] = (
        data_with_features["loan_property_ratio"]
        .replace([np.inf, -np.inf], np.nan)
    )

    return data_with_features


def create_logistic_pipeline(numeric_columns, categorical_columns):
    """Construit le pipeline de prétraitement et de régression logistique."""

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    return Pipeline([
        ("preprocessing", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=MAX_ITER,
                solver="saga",
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def build_submission(
    sample_submission,
    test_ids,
    probabilities,
    output_path,
):
    """Construit et enregistre une soumission Kaggle."""

    predictions_by_id = pd.DataFrame({
        "id": test_ids,
        "accepted": probabilities,
    })

    submission = sample_submission[["id"]].merge(
        predictions_by_id,
        on="id",
        how="left",
        validate="one_to_one",
    )

    if submission["accepted"].isna().any():
        raise ValueError("La soumission contient des prédictions manquantes.")

    if not submission["accepted"].between(0, 1).all():
        raise ValueError(
            "Certaines prédictions ne sont pas comprises entre 0 et 1."
        )

    submission.to_csv(output_path, index=False)
    return submission


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

y = train["accepted"].copy()
test_ids = test["id"].copy()

X_original = train.drop(columns=["id", "accepted"]).copy()
X_test_original = test.drop(columns=["id"]).copy()

if set(X_original.columns) != set(X_test_original.columns):
    raise ValueError("Les variables de train et de test ne correspondent pas.")

X_engineered = add_engineered_features(X_original)
X_test_engineered = add_engineered_features(X_test_original)

original_numeric_columns = [
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
]

original_categorical_columns = [
    column
    for column in X_original.columns
    if column not in original_numeric_columns
]

engineered_numeric_columns = original_numeric_columns + [
    "ltv_missing",
    "ltv_above_100",
    "loan_property_ratio",
]

engineered_categorical_columns = [
    column
    for column in X_engineered.columns
    if column not in engineered_numeric_columns
]

print("Dimensions du train :", train.shape)
print("Dimensions du test :", test.shape)
print("Variables du modèle A :", X_original.shape[1])
print("Variables du modèle B :", X_engineered.shape[1])


# Les deux modèles utilisent exactement les mêmes folds.
cross_validation = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

fold_results = []

for fold, (training_indices, validation_indices) in enumerate(
    cross_validation.split(X_original, y),
    start=1,
):
    y_training = y.iloc[training_indices]
    y_validation = y.iloc[validation_indices]

    model_original = create_logistic_pipeline(
        original_numeric_columns,
        original_categorical_columns,
    )
    model_original.fit(
        X_original.iloc[training_indices],
        y_training,
    )
    probabilities_original = model_original.predict_proba(
        X_original.iloc[validation_indices]
    )[:, 1]
    auc_original = roc_auc_score(
        y_validation,
        probabilities_original,
    )

    model_engineered = create_logistic_pipeline(
        engineered_numeric_columns,
        engineered_categorical_columns,
    )
    model_engineered.fit(
        X_engineered.iloc[training_indices],
        y_training,
    )
    probabilities_engineered = model_engineered.predict_proba(
        X_engineered.iloc[validation_indices]
    )[:, 1]
    auc_engineered = roc_auc_score(
        y_validation,
        probabilities_engineered,
    )

    difference = auc_engineered - auc_original

    fold_results.append({
        "fold": fold,
        "auc_original_features": auc_original,
        "auc_original_plus_three_features": auc_engineered,
        "auc_difference": difference,
    })

    print(f"\nFold {fold}")
    print("AUC sans les nouvelles variables :", auc_original)
    print("AUC avec les nouvelles variables :", auc_engineered)
    print("Différence :", difference)


fold_results = pd.DataFrame(fold_results)

mean_auc_original = fold_results["auc_original_features"].mean()
mean_auc_engineered = fold_results[
    "auc_original_plus_three_features"
].mean()
mean_difference = fold_results["auc_difference"].mean()
std_difference = fold_results["auc_difference"].std(ddof=1)
improved_folds = int((fold_results["auc_difference"] > 0).sum())

print("\nRésultats de la validation croisée")
print(fold_results.to_string(index=False))
print("\nAUC moyenne sans les nouvelles variables :", mean_auc_original)
print("AUC moyenne avec les nouvelles variables :", mean_auc_engineered)
print("Moyenne des différences :", mean_difference)
print("Écart-type des différences :", std_difference)
print("Nombre de folds améliorés :", improved_folds, "/", N_SPLITS)

# Enregistrement de la comparaison fold par fold
comparison_path = (
    RESULTS_DIR / "comparison_logistic_three_features_3fold.csv"
)

fold_results.to_csv(
    comparison_path,
    index=False,
)


# Réentraînement sur les 105 000 dossiers pour Kaggle
print("\nEntraînement final sans les nouvelles variables")

final_model_original = create_logistic_pipeline(
    original_numeric_columns,
    original_categorical_columns,
)
final_model_original.fit(X_original, y)

test_probabilities_original = final_model_original.predict_proba(
    X_test_original
)[:, 1]

submission_original_path = (
    SUBMISSIONS_DIR / "submission_logistic_original_features.csv"
)

submission_original = build_submission(
    sample_submission=sample_submission,
    test_ids=test_ids,
    probabilities=test_probabilities_original,
    output_path=submission_original_path,
)

print("Fichier créé :", submission_original_path)
print(submission_original.head())


print("\nEntraînement final avec les trois nouvelles variables")

final_model_engineered = create_logistic_pipeline(
    engineered_numeric_columns,
    engineered_categorical_columns,
)
final_model_engineered.fit(X_engineered, y)

test_probabilities_engineered = final_model_engineered.predict_proba(
    X_test_engineered
)[:, 1]

submission_engineered_path = (
    SUBMISSIONS_DIR
    / "submission_logistic_original_plus_three_features.csv"
)

submission_engineered = build_submission(
    sample_submission=sample_submission,
    test_ids=test_ids,
    probabilities=test_probabilities_engineered,
    output_path=submission_engineered_path,
)

print("Fichier créé :", submission_engineered_path)
print(submission_engineered.head())

print("\nFichiers produits :")
print("-", comparison_path)
print("-", submission_original_path)
print("-", submission_engineered_path)
