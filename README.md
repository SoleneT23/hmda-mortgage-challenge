# HMDA — Prédiction de l'octroi d'un prêt immobilier

Ce dépôt contient le code utilisé pour le data challenge Kaggle  
[Demandes de prêt immobilier aux États-Unis](https://www.kaggle.com/competitions/demandes-de-pret-immobilier-aux-etats-unis).

L'objectif est de prédire la probabilité qu'une demande de prêt immobilier soit
acceptée à partir des caractéristiques du prêt, du bien, de l'emprunteur et de
son environnement géographique.

## 1. Présentation du problème

Les données proviennent du registre public HMDA
(*Home Mortgage Disclosure Act*).

Chaque ligne représente une demande de prêt immobilier. La variable cible est
`accepted` :

- `accepted = 1` : le prêt a été accordé ;
- `accepted = 0` : le prêt a été refusé.

Il s'agit d'un problème de classification binaire déséquilibré. Environ 85 % des
demandes du jeu d'entraînement sont acceptées, contre 15 % de demandes refusées.

La métrique du challenge est l'aire sous la courbe ROC, ou AUC ROC. Les
prédictions soumises à Kaggle sont donc être des probabilités comprises entre
0 et 1.

Le jeu fourni comporte :

- environ 105 000 dossiers d'entraînement ;
- environ 45 000 dossiers de test ;
- 56 variables explicatives d'origine.

## 2. Variables supplémentaires

Trois variables ont été construites à partir des variables d'origine :

- `ltv_missing` : indique si le ratio LTV déclaré est manquant ;
- `ltv_above_100` : indique si le LTV déclaré est supérieur à 100 % ;
- `loan_property_ratio` : rapport entre le montant du prêt demandé et la valeur
  du bien.

Le LTV, ou *loan-to-value ratio*, mesure le rapport entre le montant garanti par
le bien et la valeur de celui-ci.

L'apport global de ces trois variables a été évalué par validation croisée à
trois folds avec une régression logistique. Leur ajout fait passer l'AUC moyenne
de 0,883853 à 0,888585, avec une amélioration observée sur chacun des trois
folds.

## 3. Modèles étudiés

Les modèles suivants ont été comparés :

1. régression logistique ;
2. CatBoost ;
3. LightGBM ;
4. mélange de CatBoost et LightGBM ;
5. forêt aléatoire.

La régression logistique constitue le modèle de référence linéaire.

CatBoost et LightGBM sont deux méthodes de gradient boosting capables de
modéliser des relations non linéaires et des interactions entre variables.

La forêt aléatoire a été utilisée afin de comparer le boosting à une méthode
fondée sur le bagging.

## 4. Installation

Le projet utilise Python et un environnement Conda.

Pour recréer l'environnement :

```bash
conda env create -f environment.yml
```

Puis l'activer :

```bash
conda activate hmda-catboost
```

Les principales bibliothèques utilisées sont :

- pandas ;
- NumPy ;
- scikit-learn ;
- CatBoost ;
- LightGBM ;
- Matplotlib.

## 5. Données

Les données ne sont pas incluses dans le dépôt.

Après les avoir téléchargées depuis Kaggle, placer les trois fichiers dans
le dossier `data/`.

## 6. Description des scripts

### Régression logistique

`V16_compare_logistic_three_features_cv.py`

Compare par validation croisée à trois folds :

- la régression logistique utilisant les 56 variables d'origine ;
- la régression logistique utilisant les variables d'origine et les trois
  variables construites.

Le script produit également les deux fichiers de soumission correspondants.

### Sélection des hyperparamètres de CatBoost

`V8_compare_catboost_depths.py`

Compare les profondeurs 5, 6 et 7 sur une même séparation
entraînement-validation.

`V10_catboost_depth7_max1500_validation.py`

Conserve une profondeur de 7, autorise jusqu'à 1 500 itérations et utilise un
arrêt anticipé. La meilleure itération observée est 1 301, soit 1 302 arbres
retenus.

Ce script crée également les prédictions de validation de CatBoost utilisées
plus tard pour sélectionner le poids du mélange.

`V12_compare_catboost_l2_leaf_reg.py`

Compare les valeurs 3, 5 et 10 du paramètre de régularisation
`l2_leaf_reg`.

### Modèle CatBoost final

`V11_catboost_depth7_1302_submission.py`

Entraîne le modèle CatBoost final sur l'ensemble des 105 000 dossiers
d'entraînement avec :

```text
iterations = 1302
depth = 7
learning_rate = 0.05
l2_leaf_reg = 3
```

Le script produit les probabilités pour les données de test et produit le
fichier de soumission Kaggle.

### Optimisation de LightGBM

`V19_lightgbm_grid_54_validation.py`

Teste une grille complète de 54 configurations LightGBM obtenues en combinant
plusieurs valeurs de :

- `learning_rate` ;
- `num_leaves` ;
- `min_child_samples` ;
- `reg_lambda`.

La meilleure configuration retenue est :

```text
learning_rate = 0.03
num_leaves = 63
min_child_samples = 20
reg_lambda = 1
n_estimators = 125
```

Le script sauvegarde les résultats complets de la grille ainsi que les
prédictions de validation du meilleur LightGBM.

### Mélange CatBoost-LightGBM

`V20_select_catboost_lightgbm_weight.py`

Compare plusieurs combinaisons linéaires des probabilités de CatBoost et de
LightGBM.

Ce script nécessite les prédictions de validation produites par
`V10_catboost_depth7_max1500_validation.py` et
`V19_lightgbm_grid_54_validation.py`.

Le meilleur poids obtenu sur la validation fixe est :

```text
CatBoost = 0.86
LightGBM = 0.14
```

`V21_plot_blend_weights.py`

Produit la courbe représentant l'AUC de validation en fonction du poids attribué
à CatBoost.

`V21_confirm_catboost_lightgbm_blend_5fold.py`

Compare CatBoost seul au mélange 86 % CatBoost et 14 % LightGBM par validation
croisée stratifiée à cinq folds.

`V22_catboost86_lightgbm14_submission.py`

Réentraîne CatBoost et LightGBM sur l'ensemble du jeu d'entraînement, mélange
leurs probabilités avec les poids 0,86 et 0,14, puis construit le fichier de
soumission Kaggle.

### Forêt aléatoire

`V24_tune_random_forest_limited_validation.py`

Compare six configurations de forêt aléatoire en faisant varier principalement :

- la profondeur maximale des arbres ;
- le nombre minimal d'observations dans une feuille.

## 7. Exécution et dépendances

La plupart des scripts sont indépendants.

La sélection du poids du mélange suit cependant la chaîne suivante :

```text
V10 + V19 → V20 → V21_plot
```
Plus précisément :

- `V10_catboost_depth7_max1500_validation.py` produit les prédictions de
  validation de CatBoost dans
  `intermediate/validation_predictions_catboost_depth7_max1500.csv` ;
- `V19_lightgbm_grid_54_validation.py` produit les prédictions de validation du
  meilleur LightGBM dans
  `intermediate/validation_predictions_lightgbm_grid_best.csv` ;
- `V20_select_catboost_lightgbm_weight.py` lit ces deux fichiers pour comparer
  les différents poids du mélange ;
- `V21_plot_blend_weights.py` utilise ensuite le tableau produit par `V20` pour
  tracer l'AUC en fonction du poids attribué à CatBoost.

Les fichiers générés dans intermediate/ sont ignorés par Git.

## 8. Principaux résultats

Le mélange CatBoost-LightGBM obtient un score Kaggle très légèrement supérieur à celui de
CatBoost seul. Cependant, la validation croisée à cinq folds ne confirme pas
l'existence d'un gain robuste du mélange par rapport à CatBoost.

CatBoost est donc retenu comme modèle principal en raison de sa meilleure
robustesse.

Les résultats détaillés des recherches d'hyperparamètres sont disponibles dans
le dossier `results/`.

## 9. Reproductibilité

Les séparations entraînement-validation sont stratifiées afin de conserver la
proportion de dossiers acceptés et refusés.

Les expériences utilisent :

```text
random_state = 42
```