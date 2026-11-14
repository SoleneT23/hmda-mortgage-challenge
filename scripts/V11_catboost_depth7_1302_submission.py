"""Entraîne le CatBoost final et crée la soumission Kaggle."""

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


# Chemins et paramètres du modèle final
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
INTERMEDIATE_DIR = PROJECT_DIR / "intermediate"
SUBMISSIONS_DIR = PROJECT_DIR / "submissions"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

MODEL_PATH = INTERMEDIATE_DIR / "catboost_depth7_1302.cbm"
IMPORTANCE_PATH = (
    RESULTS_DIR / "feature_importance_catboost_depth7_1302.csv"
)
SUBMISSION_PATH = (
    SUBMISSIONS_DIR / "submission_catboost_depth7_1302.csv"
)

RANDOM_STATE = 42
NUMBER_OF_TREES = 1302
NUMBER_OF_THREADS = 2


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
print("Dimensions de sample_submission :", sample_submission.shape)

# On conserve les identifiants avant de retirer la colonne id.
test_ids = test["id"].copy()


# Création des trois variables supplémentaires
def add_features(data):
    """Ajoute les trois variables utilisées par le modèle final."""

    data = data.copy()

    # 1 si le LTV est manquant, 0 sinon.
    data["ltv_missing"] = (
        data["loan_to_value_ratio"].isna().astype(int)
    )

    # 1 si le LTV est strictement supérieur à 100, 0 sinon.
    data["ltv_above_100"] = (
        data["loan_to_value_ratio"] > 100
    ).astype(int)

    # Rapport entre le montant du prêt et la valeur du bien, en pourcentage.
    data["loan_property_ratio"] = (
        data["loan_amount"]
        / data["property_value"]
        * 100
    )

    # CatBoost accepte les NaN numériques, mais pas les valeurs infinies.
    data["loan_property_ratio"] = (
        data["loan_property_ratio"]
        .replace([np.inf, -np.inf], np.nan)
    )

    return data


train = add_features(train)
test = add_features(test)


# Séparation des variables explicatives et de la cible
X = train.drop(columns=["id", "accepted"])
y = train["accepted"]

X_test = test.drop(columns=["id"])

print("\nDimensions de X :", X.shape)
print("Dimensions de y :", y.shape)
print("Dimensions de X_test :", X_test.shape)

# Le train et le test doivent contenir les mêmes variables dans le même ordre.
if list(X.columns) != list(X_test.columns):
    raise ValueError(
        "Les colonnes de X et X_test ne sont pas identiques ou "
        "ne sont pas dans le même ordre."
    )


# Définition des variables numériques et catégorielles
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

print("Nombre de variables numériques :", len(numeric_columns))
print("Nombre de variables catégorielles :", len(categorical_columns))
print("Nombre total de variables :", len(X.columns))


# Préparation des données pour CatBoost
# Les valeurs manquantes catégorielles deviennent une catégorie explicite.
# Toutes les variables catégorielles sont converties en chaînes de caractères.
for column in categorical_columns:
    X[column] = X[column].fillna("__MISSING__").astype(str)
    X_test[column] = X_test[column].fillna("__MISSING__").astype(str)

# Par sécurité, les éventuelles valeurs infinies numériques deviennent des NaN.
X[numeric_columns] = X[numeric_columns].replace(
    [np.inf, -np.inf],
    np.nan,
)

X_test[numeric_columns] = X_test[numeric_columns].replace(
    [np.inf, -np.inf],
    np.nan,
)


# Modèle final : profondeur 7 et 1 302 arbres
# Dans V10, la meilleure itération était 1301.
# Comme les itérations commencent à zéro, cela correspond à 1 302 arbres.
final_model = CatBoostClassifier(
    iterations=NUMBER_OF_TREES,
    depth=7,
    learning_rate=0.05,
    l2_leaf_reg=3,
    loss_function="Logloss",
    eval_metric="AUC",
    cat_features=categorical_columns,
    random_seed=RANDOM_STATE,
    thread_count=NUMBER_OF_THREADS,
    verbose=100,
    allow_writing_files=False,
)


# Entraînement sur les 105 000 dossiers
print("\nEntraînement final : depth=7, 1 302 arbres")

# Il n'y a plus d'ensemble de validation ni d'arrêt anticipé :
# le nombre d'arbres a déjà été choisi avec V10.
final_model.fit(X, y)

print("\nNombre d'arbres entraînés :", final_model.tree_count_)


# Enregistrement du modèle et des importances
final_model.save_model(str(MODEL_PATH))

feature_importance = pd.DataFrame({
    "variable": X.columns,
    "importance": final_model.get_feature_importance(),
}).sort_values(
    by="importance",
    ascending=False,
).reset_index(drop=True)

feature_importance.to_csv(
    IMPORTANCE_PATH,
    index=False,
)

print("\nLes 20 variables les plus importantes :")
print(feature_importance.head(20).to_string(index=False))


# Prédiction sur les données de test Kaggle
test_probabilities = final_model.predict_proba(X_test)[:, 1]

print("\nVérification des prédictions")
print("Nombre de prédictions :", len(test_probabilities))
print("Probabilité minimale :", test_probabilities.min())
print("Probabilité maximale :", test_probabilities.max())
print("Valeurs manquantes :", np.isnan(test_probabilities).sum())
print("Valeurs infinies :", np.isinf(test_probabilities).sum())


# Construction de la soumission
predictions_by_id = pd.DataFrame({
    "id": test_ids,
    "accepted": test_probabilities,
})

# On part de sample_submission afin de conserver exactement l'ordre attendu.
submission = (
    sample_submission[["id"]]
    .merge(
        predictions_by_id,
        on="id",
        how="left",
        validate="one_to_one",
    )
)


# Contrôles avant l'enregistrement
expected_columns = ["id", "accepted"]

if list(submission.columns) != expected_columns:
    raise ValueError(
        f"Colonnes incorrectes : {list(submission.columns)}"
    )

if len(submission) != len(sample_submission):
    raise ValueError(
        "Le nombre de lignes ne correspond pas à sample_submission."
    )

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
        "L'ordre des identifiants diffère de sample_submission."
    )

print("\nDimensions de la soumission :", submission.shape)
print("\nPremières prédictions :")
print(submission.head())
print("\nNombre de valeurs manquantes :")
print(submission.isna().sum())
print(
    "\nToutes les probabilités sont entre 0 et 1 :",
    submission["accepted"].between(0, 1).all(),
)


# Enregistrement et vérification finale
submission.to_csv(
    SUBMISSION_PATH,
    index=False,
)

submission_check = pd.read_csv(SUBMISSION_PATH)

print("\nFichier créé :", SUBMISSION_PATH)
print("\nVérification du fichier enregistré :")
print(submission_check.shape)
print(submission_check.head())

print("\nAutres fichiers créés :")
print("-", MODEL_PATH)
print("-", IMPORTANCE_PATH)

