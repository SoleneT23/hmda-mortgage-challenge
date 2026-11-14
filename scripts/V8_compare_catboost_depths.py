"""Compare les profondeurs 5, 6 et 7 de CatBoost sur une validation fixe"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# Chemins et paramètres de l'expérience
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
INTERMEDIATE_DIR = PROJECT_DIR / "intermediate"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"

RANDOM_STATE = 42
MAX_ITERATIONS = 1000
NUMBER_OF_THREADS = 2


# Chargement des données
if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Le fichier de données est introuvable : {TRAIN_PATH}"
    )

train = pd.read_csv(TRAIN_PATH)

print("Dimensions du train :", train.shape)


# Création des trois variables supplémentaires

train["ltv_missing"] = (
    train["loan_to_value_ratio"].isna().astype(int)
)

train["ltv_above_100"] = (
    train["loan_to_value_ratio"] > 100
).astype(int)

train["loan_property_ratio"] = (
    train["loan_amount"]
    / train["property_value"]
    * 100
)

# Une division par zéro produirait une valeur infinie.
# CatBoost accepte les NaN numériques mais pas les valeurs infinies.
train["loan_property_ratio"] = (
    train["loan_property_ratio"]
    .replace([np.inf, -np.inf], np.nan)
)


# Séparation des variables explicatives et de la cible
X = train.drop(columns=["id", "accepted"])
y = train["accepted"]


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

print("Dimensions de X :", X.shape)
print("Dimensions de y :", y.shape)
print("Nombre de variables numériques :", len(numeric_columns))
print("Nombre de variables catégorielles :", len(categorical_columns))


# Préparation des données pour CatBoost
# Les colonnes catégorielles sont transformées en chaînes de caractères.
# Le texte __MISSING__ représente leurs valeurs manquantes.
for column in categorical_columns:
    X[column] = X[column].fillna("__MISSING__").astype(str)

# Toutes les valeurs infinies numériques deviennent des NaN.
# CatBoost sait traiter les NaN dans les variables numériques.
X[numeric_columns] = X[numeric_columns].replace(
    [np.inf, -np.inf],
    np.nan,
)


# Séparation fixe utilisée dans les autres expériences CatBoost
X_train, X_validation, y_train, y_validation = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE,
)

print("\nNombre de dossiers d'entraînement :", len(X_train))
print("Nombre de dossiers de validation :", len(X_validation))

print("\nProportions dans l'entraînement :")
print(y_train.value_counts(normalize=True))

print("\nProportions dans la validation :")
print(y_validation.value_counts(normalize=True))


# Profondeurs comparées
# 5 : modèle moins complexe que le modèle actuel.
# 6 : profondeur du modèle actuel.
# 7 : modèle plus complexe que le modèle actuel.
depths_to_test = [5, 6, 7]

results = []

validation_predictions = pd.DataFrame({
    "row_index": X_validation.index,
    "accepted": y_validation.to_numpy(),
})


# Entraînement d'un modèle pour chaque profondeur
for depth in depths_to_test:
    print(f"\nCatBoost avec depth = {depth}")

    model = CatBoostClassifier(
        iterations=MAX_ITERATIONS,
        depth=depth,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        cat_features=categorical_columns,
        random_seed=RANDOM_STATE,
        thread_count=NUMBER_OF_THREADS,
        verbose=100,
        allow_writing_files=False,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_validation, y_validation),
        early_stopping_rounds=50,
        use_best_model=True,
    )

    train_probabilities = model.predict_proba(X_train)[:, 1]
    validation_probabilities = model.predict_proba(X_validation)[:, 1]

    train_auc = roc_auc_score(
        y_train,
        train_probabilities,
    )

    validation_auc = roc_auc_score(
        y_validation,
        validation_probabilities,
    )

    best_iteration = model.get_best_iteration()
    number_of_trees = model.tree_count_

    print("\nRésultats pour depth =", depth)
    print("AUC entraînement :", train_auc)
    print("AUC validation :", validation_auc)
    print("Écart entraînement - validation :", train_auc - validation_auc)
    print("Meilleure itération :", best_iteration)
    print("Nombre d'arbres retenu :", number_of_trees)

    results.append({
        "depth": depth,
        "train_auc": train_auc,
        "validation_auc": validation_auc,
        "train_validation_gap": train_auc - validation_auc,
        "best_iteration": best_iteration,
        "number_of_trees": number_of_trees,
    })

    validation_predictions[
        f"probability_depth_{depth}"
    ] = validation_probabilities

    # Libère la mémoire avant d'entraîner le modèle suivant.
    del model
    del train_probabilities
    del validation_probabilities
    gc.collect()


# Comparaison des profondeurs
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="validation_auc",
    ascending=False,
).reset_index(drop=True)

print("\nComparaison des profondeurs")
print(results_df.to_string(index=False))

best_depth = int(results_df.loc[0, "depth"])
best_validation_auc = results_df.loc[0, "validation_auc"]

print("\nMeilleure profondeur sur cette validation :", best_depth)
print("Meilleure AUC de validation :", best_validation_auc)


# Enregistrement des résultats
comparison_path = RESULTS_DIR / "comparison_catboost_depths_5_6_7.csv"
predictions_path = (
    INTERMEDIATE_DIR
    / "validation_predictions_catboost_depths_5_6_7.csv"
)

results_df.to_csv(
    comparison_path,
    index=False,
)

validation_predictions.to_csv(
    predictions_path,
    index=False,
)

print("\nFichiers créés :")
print("-", comparison_path)
print("-", predictions_path)

