"""
Train / Test preprocessing
==========================

Industrialisation du preprocessing validé dans :

notebooks/03_ml_preprocessing/03_train_test_preprocessing.ipynb

Le script :

1. charge le dataset issu du Feature Engineering ;
2. valide le schéma d'entrée ;
3. sépare les variables explicatives X et la cible y ;
4. exclut manufacturer_make de la modélisation ;
5. construit les jeux Train / Test ;
6. crée les indicateurs binaires de présence avant toute imputation ;
7. traite les NaN structurels, résiduels et informatifs ;
8. applique le One-Hot Encoding aux variables catégorielles nominales ;
9. standardise les variables numériques continues ;
10. valide les matrices finales ;
11. sauvegarde les datasets prétraités ;
12. sauvegarde les artefacts nécessaires à l'inférence ;
13. vérifie le rechargement et la cohérence des artefacts.

Tous les paramètres dépendant des données sont appris exclusivement sur Train
puis réutilisés sans réapprentissage sur Test.

Modes :

    --test
        100 000 premières observations.

    --full
        Dataset complet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =====================================================================
# Configuration
# =====================================================================

TARGET_COLUMN = "co2_wltp_g_km"

NROWS_TEST = 100_000
TEST_SIZE = 0.20
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Variable exclue de la modélisation
# ---------------------------------------------------------------------

EXCLUDED_FEATURES = [
    "manufacturer_make",
]


# ---------------------------------------------------------------------
# Variables explicatives utilisées pour la modélisation
# ---------------------------------------------------------------------

MODEL_FEATURE_COLUMNS = [
    "vehicle_category_type",
    "fuel_type",
    "fuel_mode",
    "mass_running_order_kg",
    "wltp_test_mass_kg",
    "engine_capacity_cm3",
    "engine_power_kw",
    "electric_energy_consumption_wh_km",
    "co2_reduction_wltp_g_km",
    "fuel_consumption",
    "electric_range_km",
    "registration_month_sin",
    "registration_month_cos",
]


# ---------------------------------------------------------------------
# Schéma attendu en entrée du script
# ---------------------------------------------------------------------

REQUIRED_COLUMNS = set(
    MODEL_FEATURE_COLUMNS
    + EXCLUDED_FEATURES
    + [TARGET_COLUMN]
)


# =====================================================================
# Règles métier
# =====================================================================

# Motorisations possédant une composante électrique.
ELECTRIC_FUEL_TYPES = {
    "electric",
    "diesel/electric",
    "petrol/electric",
}


# Motorisations possédant un moteur thermique.
THERMAL_ENGINE_FUEL_TYPES = {
    "diesel",
    "diesel/electric",
    "e85",
    "lpg",
    "ng",
    "petrol",
    "petrol/electric",
}


# Motorisations pour lesquelles la consommation de carburant
# est considérée comme applicable.
FUEL_CONSUMING_TYPES = {
    "diesel",
    "diesel/electric",
    "e85",
    "lpg",
    "petrol",
    "petrol/electric",
}


# ---------------------------------------------------------------------
# Indicateurs binaires créés avant imputation
# ---------------------------------------------------------------------

INDICATOR_COLUMNS = {
    "electric_energy_consumption_wh_km":
        "has_electric_energy_consumption_wh_km",

    "electric_range_km":
        "has_electric_range_km",

    "fuel_consumption":
        "has_fuel_consumption",

    "co2_reduction_wltp_g_km":
        "has_co2_reduction_wltp_g_km",
}


# ---------------------------------------------------------------------
# Variables catégorielles nominales encodées par One-Hot Encoding
# ---------------------------------------------------------------------

ONEHOT_COLUMNS = [
    "vehicle_category_type",
    "fuel_type",
    "fuel_mode",
]


# ---------------------------------------------------------------------
# Variables numériques avec NaN résiduels rares
# ---------------------------------------------------------------------

RARE_MEDIAN_COLUMNS = [
    "wltp_test_mass_kg",
    "engine_power_kw",
]


# ---------------------------------------------------------------------
# Variables à imputation conditionnelle
# ---------------------------------------------------------------------

CONDITIONAL_IMPUTATION_RULES = {
    "engine_capacity_cm3":
        THERMAL_ENGINE_FUEL_TYPES,

    "electric_energy_consumption_wh_km":
        ELECTRIC_FUEL_TYPES,

    "electric_range_km":
        ELECTRIC_FUEL_TYPES,

    "fuel_consumption":
        FUEL_CONSUMING_TYPES,
}


# =====================================================================
# Chargement
# =====================================================================

def load_data(
    input_path: Path,
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Charge le dataset issu du Feature Engineering.
    """

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Dataset de features introuvable : {input_path}"
        )

    print(f"Chargement : {input_path}")

    df = pd.read_csv(
        input_path,
        nrows=nrows,
        low_memory=False,
    )

    print(
        f"Dataset chargé : "
        f"{len(df):,} observations × "
        f"{df.shape[1]} variables"
    )

    return df


# =====================================================================
# Validation du schéma d'entrée
# =====================================================================

def validate_input_schema(
    df: pd.DataFrame,
) -> None:
    """
    Vérifie la conformité du dataset d'entrée.
    """

    missing_columns = sorted(
        REQUIRED_COLUMNS - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Variables indispensables absentes : "
            + ", ".join(missing_columns)
        )

    if not df.columns.is_unique:
        raise ValueError(
            "Les noms de colonnes ne sont pas uniques."
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"La cible '{TARGET_COLUMN}' contient des NaN."
        )

    print("✅ Schéma d'entrée valide.")


# =====================================================================
# Séparation X / y puis Train / Test
# =====================================================================

def split_train_test(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Construit X et y puis les jeux Train / Test.

    manufacturer_make est explicitement exclue de X.
    """

    X = df[MODEL_FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    if TARGET_COLUMN in X.columns:
        raise ValueError(
            f"La cible '{TARGET_COLUMN}' est encore présente dans X."
        )

    remaining_excluded_features = [
        column
        for column in EXCLUDED_FEATURES
        if column in X.columns
    ]

    if remaining_excluded_features:
        raise ValueError(
            "Des variables exclues sont encore présentes dans X : "
            + ", ".join(remaining_excluded_features)
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    if len(X_train) != len(y_train):
        raise ValueError(
            "Le nombre d'observations de X_train et y_train est différent."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "Le nombre d'observations de X_test et y_test est différent."
        )

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            "X_train et X_test ne possèdent pas le même nombre de variables."
        )

    print("\n=== Résultat de la séparation Train / Test ===")

    print(
        f"\nJeu d'entraînement (X_train) : "
        f"{X_train.shape[0]:,} observations × "
        f"{X_train.shape[1]} variables explicatives"
    )

    print(
        f"Jeu de test (X_test)         : "
        f"{X_test.shape[0]:,} observations × "
        f"{X_test.shape[1]} variables explicatives"
    )

    print(
        f"\nCible d'entraînement (y_train) : "
        f"{y_train.shape[0]:,} valeurs"
    )

    print(
        f"Cible de test (y_test)          : "
        f"{y_test.shape[0]:,} valeurs"
    )

    print(
        "\nRépartition appliquée : "
        "80 % des observations pour l'entraînement "
        "et 20 % pour le test."
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# =====================================================================
# Validation des variables catégorielles avant preprocessing
# =====================================================================

def validate_categorical_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    """
    Vérifie que les trois variables catégorielles attendues sont présentes
    et ne contiennent pas de valeur manquante.
    """

    missing_train_columns = [
        column
        for column in ONEHOT_COLUMNS
        if column not in X_train.columns
    ]

    missing_test_columns = [
        column
        for column in ONEHOT_COLUMNS
        if column not in X_test.columns
    ]

    if missing_train_columns or missing_test_columns:
        raise ValueError(
            "Certaines variables catégorielles sont absentes.\n"
            f"Absentes de Train : {missing_train_columns}\n"
            f"Absentes de Test : {missing_test_columns}"
        )

    train_nan_columns = [
        column
        for column in ONEHOT_COLUMNS
        if X_train[column].isna().any()
    ]

    test_nan_columns = [
        column
        for column in ONEHOT_COLUMNS
        if X_test[column].isna().any()
    ]

    if train_nan_columns or test_nan_columns:
        raise ValueError(
            "Des valeurs manquantes subsistent dans les variables "
            "catégorielles.\n"
            f"Train : {train_nan_columns}\n"
            f"Test : {test_nan_columns}"
        )

    print(
        "✅ Variables catégorielles conformes : "
        "3 variables nominales prêtes pour l'encodage."
    )


# =====================================================================
# Indicateurs binaires
# =====================================================================

def add_missing_indicators(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Crée les indicateurs binaires avant toute imputation.

    1 = donnée originale présente
    0 = donnée originale absente
    """

    X_train = X_train.copy()
    X_test = X_test.copy()

    for source_column, indicator_column in (
        INDICATOR_COLUMNS.items()
    ):

        X_train[indicator_column] = (
            X_train[source_column]
            .notna()
            .astype("int8")
        )

        X_test[indicator_column] = (
            X_test[source_column]
            .notna()
            .astype("int8")
        )

    print(
        f"✅ {len(INDICATOR_COLUMNS)} "
        "indicateurs binaires créés avant imputation."
    )

    return X_train, X_test


# =====================================================================
# Imputation conditionnelle
# =====================================================================

def apply_conditional_imputation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    column: str,
    applicable_fuel_types: set[str],
) -> tuple[float, dict[str, int]]:
    """
    Traite une variable comportant :

    - des NaN structurels -> 0 ;
    - des NaN résiduels -> médiane apprise exclusivement sur Train.
    """

    train_applicable = (
        X_train["fuel_type"]
        .isin(applicable_fuel_types)
    )

    test_applicable = (
        X_test["fuel_type"]
        .isin(applicable_fuel_types)
    )

    train_structural_nan = (
        ~train_applicable
        & X_train[column].isna()
    )

    train_residual_nan = (
        train_applicable
        & X_train[column].isna()
    )

    test_structural_nan = (
        ~test_applicable
        & X_test[column].isna()
    )

    test_residual_nan = (
        test_applicable
        & X_test[column].isna()
    )

    train_reference_values = X_train.loc[
        train_applicable
        & X_train[column].notna(),
        column,
    ]

    if train_reference_values.empty:
        raise ValueError(
            f"Aucune valeur Train disponible "
            f"pour calculer la médiane de '{column}'."
        )

    median_train = float(
        train_reference_values.median()
    )

    X_train.loc[
        train_structural_nan,
        column,
    ] = 0.0

    X_test.loc[
        test_structural_nan,
        column,
    ] = 0.0

    X_train.loc[
        train_residual_nan,
        column,
    ] = median_train

    X_test.loc[
        test_residual_nan,
        column,
    ] = median_train

    report = {
        "train_structural_nan":
            int(train_structural_nan.sum()),

        "train_residual_nan":
            int(train_residual_nan.sum()),

        "test_structural_nan":
            int(test_structural_nan.sum()),

        "test_residual_nan":
            int(test_residual_nan.sum()),
    }

    return median_train, report


# =====================================================================
# Traitement des NaN numériques
# =====================================================================

def impute_numeric_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float],
]:
    """
    Applique les règles métier validées dans le notebook 03.
    """

    X_train = X_train.copy()
    X_test = X_test.copy()

    imputation_values: dict[str, float] = {}

    print(
        "\n=== Traitement des valeurs manquantes numériques ==="
    )

    print(
        "\nTraitement conditionnel "
        "(NaN structurels -> 0 ; NaN résiduels -> médiane Train) :"
    )

    for column, applicable_types in (
        CONDITIONAL_IMPUTATION_RULES.items()
    ):

        median_train, report = (
            apply_conditional_imputation(
                X_train=X_train,
                X_test=X_test,
                column=column,
                applicable_fuel_types=applicable_types,
            )
        )

        imputation_values[column] = median_train

        print(
            f"  - {column}: "
            f"médiane Train={median_train:.4f} | "
            f"structurels Train="
            f"{report['train_structural_nan']:,} | "
            f"résiduels Train="
            f"{report['train_residual_nan']:,} | "
            f"structurels Test="
            f"{report['test_structural_nan']:,} | "
            f"résiduels Test="
            f"{report['test_residual_nan']:,}"
        )

    # -----------------------------------------------------------------
    # co2_reduction_wltp_g_km : absence informative -> 0
    # -----------------------------------------------------------------

    co2_reduction_column = "co2_reduction_wltp_g_km"

    train_co2_missing = int(
        X_train[co2_reduction_column]
        .isna()
        .sum()
    )

    test_co2_missing = int(
        X_test[co2_reduction_column]
        .isna()
        .sum()
    )

    X_train[co2_reduction_column] = (
        X_train[co2_reduction_column]
        .fillna(0.0)
    )

    X_test[co2_reduction_column] = (
        X_test[co2_reduction_column]
        .fillna(0.0)
    )

    print(
        f"\n  - {co2_reduction_column}: "
        f"absence informative -> 0 | "
        f"Train={train_co2_missing:,} | "
        f"Test={test_co2_missing:,}"
    )

    # -----------------------------------------------------------------
    # NaN résiduels rares -> médiane Train
    # -----------------------------------------------------------------

    print(
        "\nImputation des NaN résiduels rares "
        "par médiane apprise sur Train :"
    )

    for column in RARE_MEDIAN_COLUMNS:

        median_train = float(
            X_train[column].median()
        )

        if pd.isna(median_train):
            raise ValueError(
                f"Impossible de calculer "
                f"la médiane de '{column}'."
            )

        train_missing = int(
            X_train[column]
            .isna()
            .sum()
        )

        test_missing = int(
            X_test[column]
            .isna()
            .sum()
        )

        imputation_values[column] = median_train

        X_train[column] = (
            X_train[column]
            .fillna(median_train)
        )

        X_test[column] = (
            X_test[column]
            .fillna(median_train)
        )

        print(
            f"  - {column}: "
            f"médiane Train={median_train:.4f} | "
            f"NaN Train={train_missing:,} | "
            f"NaN Test={test_missing:,}"
        )

    remaining_train_nan = int(
        X_train
        .isna()
        .sum()
        .sum()
    )

    remaining_test_nan = int(
        X_test
        .isna()
        .sum()
        .sum()
    )

    if remaining_train_nan != 0 or remaining_test_nan != 0:
        raise ValueError(
            "Des valeurs manquantes subsistent après "
            "le traitement numérique : "
            f"Train={remaining_train_nan}, "
            f"Test={remaining_test_nan}."
        )

    print(
        "\n✅ Traitement numérique des valeurs manquantes terminé."
    )

    return (
        X_train,
        X_test,
        imputation_values,
    )


# =====================================================================
# One-Hot Encoding
# =====================================================================

def onehot_encode_categories(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    OneHotEncoder,
    list[str],
]:
    """
    Apprend le OneHotEncoder exclusivement sur Train,
    puis transforme Train et Test.
    """

    missing_train_columns = [
        column
        for column in ONEHOT_COLUMNS
        if column not in X_train.columns
    ]

    missing_test_columns = [
        column
        for column in ONEHOT_COLUMNS
        if column not in X_test.columns
    ]

    if missing_train_columns or missing_test_columns:
        raise ValueError(
            "Variables One-Hot absentes.\n"
            f"Train : {missing_train_columns}\n"
            f"Test : {missing_test_columns}"
        )

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.int8,
    )

    train_array = encoder.fit_transform(
        X_train[ONEHOT_COLUMNS]
    )

    test_array = encoder.transform(
        X_test[ONEHOT_COLUMNS]
    )

    onehot_feature_names = (
        encoder
        .get_feature_names_out(
            ONEHOT_COLUMNS
        )
        .tolist()
    )

    X_train_onehot = pd.DataFrame(
        train_array,
        columns=onehot_feature_names,
        index=X_train.index,
    )

    X_test_onehot = pd.DataFrame(
        test_array,
        columns=onehot_feature_names,
        index=X_test.index,
    )

    X_train = (
        X_train
        .drop(columns=ONEHOT_COLUMNS)
    )

    X_test = (
        X_test
        .drop(columns=ONEHOT_COLUMNS)
    )

    X_train = pd.concat(
        [
            X_train,
            X_train_onehot,
        ],
        axis=1,
    )

    X_test = pd.concat(
        [
            X_test,
            X_test_onehot,
        ],
        axis=1,
    )

    if (
        X_train.columns.tolist()
        != X_test.columns.tolist()
    ):
        raise ValueError(
            "Train et Test ne possèdent pas "
            "la même structure après One-Hot Encoding."
        )

    remaining_categorical = [
        column
        for column in ONEHOT_COLUMNS
        if (
            column in X_train.columns
            or column in X_test.columns
        )
    ]

    if remaining_categorical:
        raise ValueError(
            "Les variables catégorielles originales "
            "n'ont pas toutes été supprimées : "
            + ", ".join(remaining_categorical)
        )

    print(
        "✅ One-Hot Encoding terminé : "
        f"{len(onehot_feature_names)} "
        "variable(s) binaire(s) créée(s)."
    )

    return (
        X_train,
        X_test,
        encoder,
        onehot_feature_names,
    )


# =====================================================================
# Standardisation
# =====================================================================

def scale_numeric_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    onehot_feature_names: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    StandardScaler,
    list[str],
]:
    """
    Standardise les variables numériques continues.

    Les quatre indicateurs binaires et les variables issues du One-Hot
    Encoding restent en 0/1.
    """

    binary_indicator_columns = list(
        INDICATOR_COLUMNS.values()
    )

    excluded_from_scaling = set(
        binary_indicator_columns
        + onehot_feature_names
    )

    numeric_columns = (
        X_train
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    columns_to_scale = [
        column
        for column in numeric_columns
        if column not in excluded_from_scaling
    ]

    missing_test_columns = [
        column
        for column in columns_to_scale
        if column not in X_test.columns
    ]

    if missing_test_columns:
        raise ValueError(
            "Variables à standardiser absentes de Test : "
            + ", ".join(missing_test_columns)
        )

    if not columns_to_scale:
        raise ValueError(
            "Aucune variable numérique continue à standardiser."
        )

    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[columns_to_scale] = (
        scaler.fit_transform(
            X_train[columns_to_scale]
        )
    )

    X_test[columns_to_scale] = (
        scaler.transform(
            X_test[columns_to_scale]
        )
    )

    print(
        "✅ Standardisation terminée : "
        f"{len(columns_to_scale)} "
        "variable(s) apprise(s) sur Train."
    )

    return (
        X_train,
        X_test,
        scaler,
        columns_to_scale,
    )


# =====================================================================
# Validation finale
# =====================================================================

def validate_processed_datasets(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    Vérifie que les matrices finales sont prêtes pour la modélisation.
    """

    train_non_numeric = (
        X_train
        .select_dtypes(exclude="number")
        .columns
        .tolist()
    )

    test_non_numeric = (
        X_test
        .select_dtypes(exclude="number")
        .columns
        .tolist()
    )

    remaining_original_categorical = [
        column
        for column in ONEHOT_COLUMNS
        if (
            column in X_train.columns
            or column in X_test.columns
        )
    ]

    remaining_excluded_features = [
        column
        for column in EXCLUDED_FEATURES
        if (
            column in X_train.columns
            or column in X_test.columns
        )
    ]

    indicator_columns = list(
        INDICATOR_COLUMNS.values()
    )

    indicator_values_valid = all(
        (
            column in X_train.columns
            and column in X_test.columns
            and set(
                X_train[column]
                .dropna()
                .unique()
            ).issubset({0, 1})
            and set(
                X_test[column]
                .dropna()
                .unique()
            ).issubset({0, 1})
        )
        for column in indicator_columns
    )

    checks = {
        "Aucun NaN Train":
            not X_train.isna().any().any(),

        "Aucun NaN Test":
            not X_test.isna().any().any(),

        "Colonnes Train / Test identiques":
            X_train.columns.tolist()
            == X_test.columns.tolist(),

        "Aucune variable non numérique Train":
            len(train_non_numeric) == 0,

        "Aucune variable non numérique Test":
            len(test_non_numeric) == 0,

        "Variables catégorielles originales supprimées":
            len(remaining_original_categorical) == 0,

        "Variables exclues absentes":
            len(remaining_excluded_features) == 0,

        "Indicateurs binaires conformes":
            indicator_values_valid,

        "Cohérence X_train / y_train":
            len(X_train) == len(y_train),

        "Cohérence X_test / y_test":
            len(X_test) == len(y_test),

        "Aucune valeur infinie Train":
            not np.isinf(
                X_train.to_numpy()
            ).any(),

        "Aucune valeur infinie Test":
            not np.isinf(
                X_test.to_numpy()
            ).any(),
    }

    failed_checks = []

    print(
        "\n=== Validation finale du preprocessing ==="
    )

    print(
        f"\nX_train_processed : "
        f"{X_train.shape[0]:,} observations × "
        f"{X_train.shape[1]} variables"
    )

    print(
        f"X_test_processed  : "
        f"{X_test.shape[0]:,} observations × "
        f"{X_test.shape[1]} variables"
    )

    for name, result in checks.items():

        status = "✅" if result else "❌"

        print(
            f"{status} {name}"
        )

        if not result:
            failed_checks.append(name)

    if failed_checks:
        raise ValueError(
            "Validation finale échouée : "
            + ", ".join(failed_checks)
        )

    print(
        "✅ Dataset de preprocessing validé."
    )


# =====================================================================
# Sauvegarde des datasets
# =====================================================================

def save_datasets(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
    suffix: str = "",
) -> dict[str, Path]:
    """
    Sauvegarde les quatre jeux Train / Test au format Parquet.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "X_train":
            output_dir
            / f"X_train_processed{suffix}.parquet",

        "X_test":
            output_dir
            / f"X_test_processed{suffix}.parquet",

        "y_train":
            output_dir
            / f"y_train{suffix}.parquet",

        "y_test":
            output_dir
            / f"y_test{suffix}.parquet",
    }

    X_train.to_parquet(
        paths["X_train"],
        index=True,
    )

    X_test.to_parquet(
        paths["X_test"],
        index=True,
    )

    y_train.to_frame(
        name=TARGET_COLUMN
    ).to_parquet(
        paths["y_train"],
        index=True,
    )

    y_test.to_frame(
        name=TARGET_COLUMN
    ).to_parquet(
        paths["y_test"],
        index=True,
    )

    for path in paths.values():

        if not path.is_file():
            raise FileNotFoundError(
                f"Fichier non sauvegardé : {path}"
            )

    print(
        "\n✅ Datasets prétraités sauvegardés."
    )

    print(
        f"  - X_train : "
        f"{X_train.shape[0]:,} observations × "
        f"{X_train.shape[1]} variables"
    )

    print(
        f"  - X_test  : "
        f"{X_test.shape[0]:,} observations × "
        f"{X_test.shape[1]} variables"
    )

    print(
        f"  - y_train : "
        f"{len(y_train):,} observations"
    )

    print(
        f"  - y_test  : "
        f"{len(y_test):,} observations"
    )

    return paths


# =====================================================================
# Sauvegarde des artefacts
# =====================================================================

def save_preprocessing_artifacts(
    *,
    imputation_values: dict[str, float],
    onehot_encoder: OneHotEncoder,
    onehot_feature_names: list[str],
    standard_scaler: StandardScaler,
    columns_to_scale: list[str],
    final_features: list[str],
    output_dir: Path,
) -> dict[str, Path]:
    """
    Sauvegarde les paramètres appris exclusivement sur Train nécessaires
    pour reproduire le preprocessing en inférence.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    imputation_values_path = (
        output_dir
        / "imputation_values.joblib"
    )

    onehot_encoder_path = (
        output_dir
        / "onehot_encoder.joblib"
    )

    standard_scaler_path = (
        output_dir
        / "standard_scaler.joblib"
    )

    metadata_path = (
        output_dir
        / "preprocessing_metadata.json"
    )

    # -----------------------------------------------------------------
    # Sauvegarde des objets appris sur Train
    # -----------------------------------------------------------------

    joblib.dump(
        imputation_values,
        imputation_values_path,
    )

    joblib.dump(
        onehot_encoder,
        onehot_encoder_path,
    )

    joblib.dump(
        standard_scaler,
        standard_scaler_path,
    )

    # -----------------------------------------------------------------
    # Métadonnées
    # -----------------------------------------------------------------

    metadata = {
        "target_column":
            TARGET_COLUMN,

        "excluded_features":
            EXCLUDED_FEATURES,

        "model_input_features":
            MODEL_FEATURE_COLUMNS,

        "split": {
            "test_size":
                TEST_SIZE,

            "random_state":
                RANDOM_STATE,
        },

        "imputation": {
            "artifact":
                "imputation_values.joblib",

            "columns":
                list(imputation_values.keys()),

            "co2_reduction_wltp_g_km_missing_value":
                0.0,
        },

        "conditional_imputation": {
            "electric_fuel_types":
                sorted(ELECTRIC_FUEL_TYPES),

            "thermal_engine_fuel_types":
                sorted(THERMAL_ENGINE_FUEL_TYPES),

            "fuel_consuming_types":
                sorted(FUEL_CONSUMING_TYPES),
        },

        "binary_indicators":
            list(
                INDICATOR_COLUMNS.values()
            ),

        "onehot_encoding": {
            "source_columns":
                ONEHOT_COLUMNS,

            "output_columns":
                onehot_feature_names,

            "handle_unknown":
                "ignore",
        },

        "standardization": {
            "columns":
                columns_to_scale,
        },

        "final_features":
            final_features,
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )

    paths = {
        "imputation_values":
            imputation_values_path,

        "onehot_encoder":
            onehot_encoder_path,

        "standard_scaler":
            standard_scaler_path,

        "metadata":
            metadata_path,
    }

    for path in paths.values():

        if not path.is_file():
            raise FileNotFoundError(
                f"Artefact non sauvegardé : {path}"
            )

    print(
        "\n=== Sauvegarde des artefacts de preprocessing ==="
    )

    print(
        f"\nRépertoire de sauvegarde : {output_dir}"
    )

    print(
        f"\nNombre d'artefacts sauvegardés : "
        f"{len(paths)}"
    )

    print(
        "\n  - Valeurs d'imputation"
    )

    print(
        "  - OneHotEncoder"
    )

    print(
        "  - StandardScaler"
    )

    print(
        "  - Métadonnées preprocessing"
    )

    print(
        "\nValeurs d'imputation sauvegardées :"
    )

    for column, value in imputation_values.items():
        print(
            f"  - {column} : {value}"
        )

    print(
        f"\nNombre de variables finales enregistrées "
        f"dans les métadonnées : "
        f"{len(final_features)}"
    )

    print(
        "\n✅ Artefacts de preprocessing sauvegardés."
    )

    return paths


# =====================================================================
# Validation du rechargement des artefacts
# =====================================================================

def validate_saved_artifacts(
    artifact_paths: dict[str, Path],
    final_features: list[str],
) -> None:
    """
    Vérifie que les quatre artefacts sont rechargeables et cohérents
    avec la structure finale utilisée pour la modélisation.
    """

    loaded_imputation_values = joblib.load(
        artifact_paths["imputation_values"]
    )

    loaded_onehot_encoder = joblib.load(
        artifact_paths["onehot_encoder"]
    )

    loaded_standard_scaler = joblib.load(
        artifact_paths["standard_scaler"]
    )

    with open(
        artifact_paths["metadata"],
        "r",
        encoding="utf-8",
    ) as file:

        loaded_metadata = json.load(file)

    saved_final_features = loaded_metadata[
        "final_features"
    ]

    saved_imputation_columns = loaded_metadata[
        "imputation"
    ]["columns"]

    saved_onehot_columns = loaded_metadata[
        "onehot_encoding"
    ]["source_columns"]

    saved_onehot_output_columns = loaded_metadata[
        "onehot_encoding"
    ]["output_columns"]

    saved_scaling_columns = loaded_metadata[
        "standardization"
    ]["columns"]

    saved_binary_indicators = loaded_metadata[
        "binary_indicators"
    ]

    imputation_values_valid = (
        isinstance(
            loaded_imputation_values,
            dict,
        )
        and set(
            loaded_imputation_values.keys()
        )
        == set(
            saved_imputation_columns
        )
        and all(
            pd.notna(value)
            for value
            in loaded_imputation_values.values()
        )
    )

    onehot_encoder_valid = (
        hasattr(
            loaded_onehot_encoder,
            "categories_",
        )
        and len(
            loaded_onehot_encoder.categories_
        )
        == len(saved_onehot_columns)
        and len(
            saved_onehot_output_columns
        )
        == sum(
            len(categories)
            for categories
            in loaded_onehot_encoder.categories_
        )
    )

    standard_scaler_valid = (
        hasattr(
            loaded_standard_scaler,
            "mean_",
        )
        and hasattr(
            loaded_standard_scaler,
            "scale_",
        )
        and len(
            loaded_standard_scaler.mean_
        )
        == len(saved_scaling_columns)
        and len(
            loaded_standard_scaler.scale_
        )
        == len(saved_scaling_columns)
    )

    metadata_structure_valid = (
        isinstance(saved_final_features, list)
        and isinstance(saved_imputation_columns, list)
        and isinstance(saved_onehot_columns, list)
        and isinstance(saved_onehot_output_columns, list)
        and isinstance(saved_scaling_columns, list)
        and isinstance(saved_binary_indicators, list)
    )

    structure_valid = (
        saved_final_features
        == final_features
    )

    checks = {
        "Valeurs d'imputation rechargeables":
            imputation_values_valid,

        "OneHotEncoder rechargeable":
            onehot_encoder_valid,

        "StandardScaler rechargeable":
            standard_scaler_valid,

        "Métadonnées conformes":
            metadata_structure_valid,

        "Structure finale conforme":
            structure_valid,
    }

    failed_checks = []

    print(
        "\n=== Validation du rechargement des artefacts ==="
    )

    for name, result in checks.items():

        status = "✅" if result else "❌"

        print(
            f"{status} {name}"
        )

        if not result:
            failed_checks.append(name)

    if failed_checks:
        raise ValueError(
            "Validation des artefacts échouée : "
            + ", ".join(failed_checks)
        )

    print(
        "✅ Tous les artefacts sont rechargeables "
        "et cohérents avec la structure finale."
    )


# =====================================================================
# Pipeline principal
# =====================================================================

def run_preprocessing(
    input_path: Path,
    data_output_dir: Path,
    artifact_output_dir: Path,
    nrows: int | None = None,
    suffix: str = "",
) -> None:
    """
    Exécute le preprocessing Train / Test complet.
    """

    # -----------------------------------------------------------------
    # 1. Chargement et validation
    # -----------------------------------------------------------------

    df = load_data(
        input_path=input_path,
        nrows=nrows,
    )

    validate_input_schema(df)

    # -----------------------------------------------------------------
    # 2. Séparation X / y puis Train / Test
    # -----------------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_train_test(df)

    del df

    # -----------------------------------------------------------------
    # 3. Contrôle des variables catégorielles
    # -----------------------------------------------------------------

    validate_categorical_features(
        X_train=X_train,
        X_test=X_test,
    )

    # -----------------------------------------------------------------
    # 4. Indicateurs binaires
    # -----------------------------------------------------------------

    (
        X_train,
        X_test,
    ) = add_missing_indicators(
        X_train=X_train,
        X_test=X_test,
    )

    # -----------------------------------------------------------------
    # 5. Traitement des valeurs manquantes
    # -----------------------------------------------------------------

    (
        X_train,
        X_test,
        imputation_values,
    ) = impute_numeric_features(
        X_train=X_train,
        X_test=X_test,
    )

    # -----------------------------------------------------------------
    # 6. One-Hot Encoding
    # -----------------------------------------------------------------

    (
        X_train,
        X_test,
        onehot_encoder,
        onehot_feature_names,
    ) = onehot_encode_categories(
        X_train=X_train,
        X_test=X_test,
    )

    # -----------------------------------------------------------------
    # 7. Standardisation
    # -----------------------------------------------------------------

    (
        X_train,
        X_test,
        standard_scaler,
        columns_to_scale,
    ) = scale_numeric_features(
        X_train=X_train,
        X_test=X_test,
        onehot_feature_names=onehot_feature_names,
    )

    # -----------------------------------------------------------------
    # 8. Validation finale
    # -----------------------------------------------------------------

    validate_processed_datasets(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    print(
        f"\nStructure finale : "
        f"{X_train.shape[1]} variables"
    )

    # -----------------------------------------------------------------
    # 9. Sauvegarde des datasets
    # -----------------------------------------------------------------

    save_datasets(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        output_dir=data_output_dir,
        suffix=suffix,
    )

    # -----------------------------------------------------------------
    # 10. Sauvegarde des artefacts
    # -----------------------------------------------------------------

    artifact_paths = (
        save_preprocessing_artifacts(
            imputation_values=imputation_values,
            onehot_encoder=onehot_encoder,
            onehot_feature_names=onehot_feature_names,
            standard_scaler=standard_scaler,
            columns_to_scale=columns_to_scale,
            final_features=(
                X_train.columns.tolist()
            ),
            output_dir=artifact_output_dir,
        )
    )

    # -----------------------------------------------------------------
    # 11. Validation des artefacts
    # -----------------------------------------------------------------

    validate_saved_artifacts(
        artifact_paths=artifact_paths,
        final_features=(
            X_train.columns.tolist()
        ),
    )

    print(
        "\n✅ Pipeline Train / Test preprocessing "
        "terminé avec succès."
    )


# =====================================================================
# Interface CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    """
    Définit les modes TEST et FULL.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train / Test preprocessing du projet "
            "Vehicle Emissions Prediction MLOps."
        )
    )

    mode_group = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    mode_group.add_argument(
        "--test",
        action="store_true",
        help=(
            "Exécute le preprocessing sur "
            "les 100 000 premières observations."
        ),
    )

    mode_group.add_argument(
        "--full",
        action="store_true",
        help=(
            "Exécute le preprocessing "
            "sur l'intégralité du dataset."
        ),
    )

    return parser.parse_args()


# =====================================================================
# Point d'entrée
# =====================================================================

def main() -> None:
    """
    Point d'entrée du script.
    """

    args = parse_args()

    # -----------------------------------------------------------------
    # Racine du projet
    #
    # Fichier :
    # src/vehicle_emissions/preprocessing/train_test_preprocessing.py
    #
    # parents[3] = racine du projet
    # -----------------------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    # -----------------------------------------------------------------
    # Dataset source
    # -----------------------------------------------------------------

    input_path = (
        project_root
        / "data"
        / "interim"
        / "data_2024_features.csv"
    )

    # -----------------------------------------------------------------
    # Répertoire des datasets produits
    # -----------------------------------------------------------------

    data_output_dir = (
        project_root
        / "data"
        / "processed"
    )

    # -----------------------------------------------------------------
    # Modes
    # -----------------------------------------------------------------

    if args.test:

        mode = "TEST"
        nrows = NROWS_TEST
        suffix = "_test"

        artifact_output_dir = (
            project_root
            / "models"
            / "preprocessing"
            / "test"
        )

    else:

        mode = "COMPLET"
        nrows = None
        suffix = ""

        artifact_output_dir = (
            project_root
            / "models"
            / "preprocessing"
        )

    # -----------------------------------------------------------------
    # Informations d'exécution
    # -----------------------------------------------------------------

    print(
        f"Mode d'exécution : {mode}"
    )

    print(
        f"Fichier source   : {input_path}"
    )

    print(
        f"Sortie datasets  : {data_output_dir}"
    )

    print(
        f"Sortie artefacts : {artifact_output_dir}"
    )

    # -----------------------------------------------------------------
    # Exécution
    # -----------------------------------------------------------------

    run_preprocessing(
        input_path=input_path,
        data_output_dir=data_output_dir,
        artifact_output_dir=artifact_output_dir,
        nrows=nrows,
        suffix=suffix,
    )


if __name__ == "__main__":
    main()