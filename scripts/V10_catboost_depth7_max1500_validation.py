"""Évalue CatBoost de profondeur 7 avec au plus 1 500 arbres."""

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
MAX_ITERATIONS = 1500
EARLY_STOPPING_ROUNDS = 100
NUMBER_OF_THREADS = 2


# Chargement des données
if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Le fichier de données est introuvable : {TRAIN_PATH}"
    )

train = pd.read_csv(TRAIN_PATH)

print("Dimensions du train avant feature engineering :", train.shape)


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

# Une éventuelle division par zéro produit une valeur infinie.
# CatBoost accepte les NaN numériques, mais pas les valeurs infinies.
train["loan_property_ratio"] = (
    train["loan_property_ratio"]
    .replace([np.inf, -np.inf], np.nan)
)


# Séparation des variables explicatives et de la cible
X = train.drop(columns=["id", "accepted"])
y = train["accepted"]

print("Dimensions de X :", X.shape)
print("Dimensions de y :", y.shape)


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
# Les variables catégorielles sont ensuite converties en chaînes de caractères.
for column in categorical_columns:
    X[column] = X[column].fillna("__MISSING__").astype(str)

# Par sécurité, toutes les valeurs infinies numériques deviennent des NaN.
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


# Modèle de profondeur 7 avec au plus 1 500 arbres
model = CatBoostClassifier(
    iterations=MAX_ITERATIONS,
    depth=7,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    cat_features=categorical_columns,
    random_seed=RANDOM_STATE,
    thread_count=NUMBER_OF_THREADS,
    verbose=100,
    allow_writing_files=False,
)


# Entraînement avec arrêt anticipé
print("\nCatBoost depth=7, maximum de 1 500 itérations")

model.fit(
    X_train,
    y_train,
    eval_set=(X_validation, y_validation),
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    use_best_model=True,
)


# Calcul des AUC
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
generalization_gap = train_auc - validation_auc


# Comparaison avec le modèle précédent limité à 1 000 arbres
# Résultats obtenus précédemment avec depth=7 et une limite de 1 000 arbres.
previous_train_auc = 0.9347569192506595
previous_validation_auc = 0.9112814738982734
previous_best_iteration = 996
previous_number_of_trees = 997

validation_difference = validation_auc - previous_validation_auc

print("\nRésultats")
print("AUC entraînement :", train_auc)
print("AUC validation :", validation_auc)
print("Écart entraînement - validation :", generalization_gap)
print("Meilleure itération :", best_iteration)
print("Nombre d'arbres retenu :", number_of_trees)

print("\nComparaison avec le modèle précédent :")
print("Ancienne AUC validation :", previous_validation_auc)
print("Nouvelle AUC validation :", validation_auc)
print("Différence d'AUC :", validation_difference)
print(
    "Différence en point d'AUC sur une échelle de 0 à 100 :",
    validation_difference * 100,
)

if validation_difference > 0:
    print("Conclusion locale : le nouveau modèle améliore l'AUC de validation.")
elif validation_difference < 0:
    print("Conclusion locale : le nouveau modèle diminue l'AUC de validation.")
else:
    print("Conclusion locale : les deux AUC de validation sont identiques.")

if best_iteration >= 1490:
    print(
        "Attention : la meilleure itération est encore très proche "
        "de la limite de 1 500."
    )
else:
    print(
        "La meilleure itération n'est pas à la limite de 1 500 : "
        "l'arrêt anticipé a pu rechercher un maximum au-delà de 1 000."
    )


# Enregistrement du tableau de comparaison
comparison = pd.DataFrame([
    {
        "experiment": "depth7_max1000_previous",
        "maximum_iterations": 1000,
        "best_iteration": previous_best_iteration,
        "number_of_trees": previous_number_of_trees,
        "train_auc": previous_train_auc,
        "validation_auc": previous_validation_auc,
        "train_validation_gap": (
            previous_train_auc - previous_validation_auc
        ),
    },
    {
        "experiment": "depth7_max1500",
        "maximum_iterations": 1500,
        "best_iteration": best_iteration,
        "number_of_trees": number_of_trees,
        "train_auc": train_auc,
        "validation_auc": validation_auc,
        "train_validation_gap": generalization_gap,
    },
])

comparison_path = (
    RESULTS_DIR / "comparison_catboost_depth7_1000_vs_1500.csv"
)
comparison.to_csv(comparison_path, index=False)

# Enregistrement des prédictions nécessaires au mélange avec LightGBM
validation_predictions = pd.DataFrame({
    "row_index": X_validation.index,
    "accepted": y_validation.to_numpy(),
    "catboost_probability": validation_probabilities,
})

predictions_path = (
    INTERMEDIATE_DIR
    / "validation_predictions_catboost_depth7_max1500.csv"
)
validation_predictions.to_csv(predictions_path, index=False)

print("\nTableau récapitulatif :")
print(comparison.to_string(index=False))

print("\nFichiers créés :")
print("-", comparison_path)
print("-", predictions_path)

