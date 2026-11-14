"""Compare plusieurs niveaux de régularisation L2 pour CatBoost."""

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
for column in categorical_columns:
    X[column] = X[column].fillna("__MISSING__").astype(str)

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


# Résultat de référence pour l2_leaf_reg=3
# CatBoost utilise l2_leaf_reg=3 par défaut.
# Ces valeurs viennent directement de V10. On ne réentraîne donc pas ce modèle.
baseline_l2 = 3
baseline_train_auc = 0.9384369547777102
baseline_validation_auc = 0.9118183305741654
baseline_best_iteration = 1301
baseline_number_of_trees = 1302
baseline_gap = baseline_train_auc - baseline_validation_auc

results = [
    {
        "l2_leaf_reg": baseline_l2,
        "train_auc": baseline_train_auc,
        "validation_auc": baseline_validation_auc,
        "difference_vs_l2_3": 0.0,
        "train_validation_gap": baseline_gap,
        "best_iteration": baseline_best_iteration,
        "number_of_trees": baseline_number_of_trees,
        "source": "previous_V10_result",
    }
]

print("\nRésultat de référence déjà connu :")
print("l2_leaf_reg :", baseline_l2)
print("AUC entraînement :", baseline_train_auc)
print("AUC validation :", baseline_validation_auc)
print("Écart entraînement - validation :", baseline_gap)
print("Nombre d'arbres retenu :", baseline_number_of_trees)


# Valeurs de régularisation testées
# Plus cette valeur est élevée, plus CatBoost réduit les valeurs extrêmes
# apprises dans les feuilles des arbres.
l2_values_to_test = [5, 10]

validation_predictions = pd.DataFrame({
    "row_index": X_validation.index,
    "accepted": y_validation.to_numpy(),
})


# Entraînement des deux nouvelles configurations
for l2_value in l2_values_to_test:
    print(f"\nCatBoost avec l2_leaf_reg = {l2_value}")

    model = CatBoostClassifier(
        iterations=MAX_ITERATIONS,
        depth=7,
        learning_rate=0.05,
        l2_leaf_reg=l2_value,
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
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
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
    generalization_gap = train_auc - validation_auc
    difference_vs_baseline = validation_auc - baseline_validation_auc

    print("\nRésultats pour l2_leaf_reg =", l2_value)
    print("AUC entraînement :", train_auc)
    print("AUC validation :", validation_auc)
    print("Différence par rapport à l2_leaf_reg=3 :", difference_vs_baseline)
    print(
        "Différence en point d'AUC sur une échelle de 0 à 100 :",
        difference_vs_baseline * 100,
    )
    print("Écart entraînement - validation :", generalization_gap)
    print("Meilleure itération :", best_iteration)
    print("Nombre d'arbres retenu :", number_of_trees)

    if best_iteration >= 1490:
        print(
            "Attention : la meilleure itération est proche de la limite "
            "de 1 500. Il pourra être nécessaire d'augmenter cette limite."
        )

    results.append({
        "l2_leaf_reg": l2_value,
        "train_auc": train_auc,
        "validation_auc": validation_auc,
        "difference_vs_l2_3": difference_vs_baseline,
        "train_validation_gap": generalization_gap,
        "best_iteration": best_iteration,
        "number_of_trees": number_of_trees,
        "source": "trained_in_V12",
    })

    validation_predictions[
        f"probability_l2_{l2_value}"
    ] = validation_probabilities

    # On libère la mémoire avant d'entraîner le modèle suivant.
    del model
    del train_probabilities
    del validation_probabilities
    gc.collect()


# Tableau récapitulatif
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="validation_auc",
    ascending=False,
).reset_index(drop=True)

print("\nComparaison des régularisations")
print(results_df.to_string(index=False))

best_l2 = results_df.loc[0, "l2_leaf_reg"]
best_validation_auc = results_df.loc[0, "validation_auc"]
best_number_of_trees = int(results_df.loc[0, "number_of_trees"])

print("\nMeilleure valeur de l2_leaf_reg :", best_l2)
print("Meilleure AUC de validation :", best_validation_auc)
print("Nombre d'arbres associé :", best_number_of_trees)

if best_l2 == baseline_l2:
    print(
        "Conclusion locale : aucune des nouvelles valeurs ne bat "
        "la valeur par défaut l2_leaf_reg=3."
    )
else:
    print(
        "Conclusion locale : une régularisation différente de la valeur "
        "par défaut améliore l'AUC sur cette validation."
    )


# Enregistrement des résultats
comparison_path = (
    RESULTS_DIR / "comparison_catboost_l2_leaf_reg_3_5_10.csv"
)
results_df.to_csv(comparison_path, index=False)

predictions_path = (
    INTERMEDIATE_DIR
    / "validation_predictions_catboost_l2_leaf_reg_5_10.csv"
)
validation_predictions.to_csv(predictions_path, index=False)

print("\nFichiers créés :")
print("-", comparison_path)
print("-", predictions_path)

