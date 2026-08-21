"""
Train / Test preprocessing
==========================

Industrialisation du preprocessing validé dans :
notebooks/03_ml_preprocessing/03_train_test_preprocessing.ipynb

Principes :
- manufacturer_make est exclue de la modélisation ;
- les paramètres dépendant des données sont appris uniquement sur Train ;
- les mêmes paramètres sont appliqués à Test sans réapprentissage ;
- les artefacts nécessaires à l'inférence sont sauvegardés.

Modes :
    --test : 100 000 premières observations
    --full : dataset complet
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

EXCLUDED_FEATURES = [
    "manufacturer_make",
]

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

REQUIRED_COLUMNS = set(
    MODEL_FEATURE_COLUMNS
    + EXCLUDED_FEATURES
    + [TARGET_COLUMN]
)


# =====================================================================
# Règles métier
# =====================================================================

ELECTRIC_FUEL_TYPES = {
    "electric",
    "diesel/electric",
    "petrol/electric",
}

THERMAL_ENGINE_FUEL_TYPES = {
    "diesel",
    "diesel/electric",
    "e85",
    "lpg",
    "ng",
    "petrol",
    "petrol/electric",
}

FUEL_CONSUMING_TYPES = {
    "diesel",
    "diesel/electric",
    "e85",
    "lpg",
    "petrol",
    "petrol/electric",
}

CATEGORICAL_IMPUTATION_COLUMNS = [
    "vehicle_category_type",
    "fuel_mode",
]

ONEHOT_COLUMNS = [
    "vehicle_category_type",
    "fuel_type",
    "fuel_mode",
]

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

RARE_MEDIAN_COLUMNS = [
    "wltp_test_mass_kg",
    "engine_power_kw",
]

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
    Vérifie la conformité minimale du dataset d'entrée.
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
    Construit X, y puis les jeux Train / Test.
    """

    X = df[MODEL_FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    if TARGET_COLUMN in X.columns:
        raise ValueError(
            f"La cible '{TARGET_COLUMN}' est encore présente dans X."
        )

    remaining_excluded = [
        column
        for column in EXCLUDED_FEATURES
        if column in X.columns
    ]

    if remaining_excluded:
        raise ValueError(
            "Variables exclues encore présentes dans X : "
            + ", ".join(remaining_excluded)
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    if len(X_train) != len(y_train):
        raise ValueError(
            "Incohérence entre X_train et y_train."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "Incohérence entre X_test et y_test."
        )

    print(
        "\n=== Résultat de la séparation Train / Test ==="
    )

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
# Imputation catégorielle
# =====================================================================

def impute_categorical_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, str],
]:
    """
    Impute les NaN catégoriels par la modalité la plus fréquente
    apprise exclusivement sur Train.
    """

    X_train = X_train.copy()
    X_test = X_test.copy()

    imputation_values: dict[str, str] = {}

    print(
        "\n=== Imputation des valeurs manquantes catégorielles ==="
    )

    for column in CATEGORICAL_IMPUTATION_COLUMNS:

        train_mode = (
            X_train[column]
            .mode(dropna=True)
        )

        if train_mode.empty:
            raise ValueError(
                f"Impossible de déterminer la modalité "
                f"d'imputation pour '{column}'."
            )

        mode_value = str(
            train_mode.iloc[0]
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

        imputation_values[column] = mode_value

        X_train[column] = (
            X_train[column]
            .fillna(mode_value)
        )

        X_test[column] = (
            X_test[column]
            .fillna(mode_value)
        )

        print(
            f"  - {column}: "
            f"modalité Train='{mode_value}' | "
            f"NaN Train={train_missing:,} | "
            f"NaN Test={test_missing:,}"
        )

    return (
        X_train,
        X_test,
        imputation_values,
    )


# =====================================================================
# Validation catégorielle
# =====================================================================

def validate_categorical_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    """
    Vérifie que les variables catégorielles nécessaires
    au One-Hot Encoding existent et ne contiennent plus de NaN.
    """

    missing_train = [
        column
        for column in ONEHOT_COLUMNS
        if column not in X_train.columns
    ]

    missing_test = [
        column
        for column in ONEHOT_COLUMNS
        if column not in X_test.columns
    ]

    if missing_train or missing_test:
        raise ValueError(
            "Variables catégorielles absentes.\n"
            f"Train : {missing_train}\n"
            f"Test : {missing_test}"
        )

    train_nan = [
        column
        for column in ONEHOT_COLUMNS
        if X_train[column].isna().any()
    ]

    test_nan = [
        column
        for column in ONEHOT_COLUMNS
        if X_test[column].isna().any()
    ]

    if train_nan or test_nan:
        raise ValueError(
            "Des valeurs manquantes subsistent dans les variables "
            "catégorielles.\n"
            f"Train : {train_nan}\n"
            f"Test : {test_nan}"
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
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Crée les indicateurs binaires avant toute imputation numérique.

    1 = valeur originale présente
    0 = valeur originale absente
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
        "indicateurs binaires créés avant imputation numérique."
    )

    return (
        X_train,
        X_test,
    )


# =====================================================================
# Imputation conditionnelle numérique
# =====================================================================

def apply_conditional_imputation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    column: str,
    applicable_fuel_types: set[str],
) -> tuple[
    float,
    dict[str, int],
]:
    """
    Traite une variable numérique comportant :

    - des NaN structurels -> 0 ;
    - des NaN résiduels -> médiane apprise sur Train.
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

    return (
        median_train,
        report,
    )


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
    Applique les règles métier numériques validées.
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

    co2_reduction_column = (
        "co2_reduction_wltp_g_km"
    )

    train_co2_missing = int(
        X_train[
            co2_reduction_column
        ]
        .isna()
        .sum()
    )

    test_co2_missing = int(
        X_test[
            co2_reduction_column
        ]
        .isna()
        .sum()
    )

    X_train[
        co2_reduction_column
    ] = (
        X_train[
            co2_reduction_column
        ]
        .fillna(0.0)
    )

    X_test[
        co2_reduction_column
    ] = (
        X_test[
            co2_reduction_column
        ]
        .fillna(0.0)
    )

    print(
        f"\n  - {co2_reduction_column}: "
        f"absence informative -> 0 | "
        f"Train={train_co2_missing:,} | "
        f"Test={test_co2_missing:,}"
    )

    # -----------------------------------------------------------------
    # NaN rares -> médiane Train
    # -----------------------------------------------------------------

    print(
        "\nImputation des NaN résiduels rares "
        "par médiane apprise sur Train :"
    )

    for column in RARE_MEDIAN_COLUMNS:

        median_train = float(
            X_train[column]
            .median()
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

    if (
        remaining_train_nan != 0
        or remaining_test_nan != 0
    ):
        raise ValueError(
            "Des valeurs manquantes subsistent après "
            "le preprocessing numérique : "
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
    Apprend le OneHotEncoder sur Train puis transforme Train et Test.
    """

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

    feature_names = (
        encoder
        .get_feature_names_out(
            ONEHOT_COLUMNS
        )
        .tolist()
    )

    train_encoded = pd.DataFrame(
        train_array,
        columns=feature_names,
        index=X_train.index,
    )

    test_encoded = pd.DataFrame(
        test_array,
        columns=feature_names,
        index=X_test.index,
    )

    X_train = pd.concat(
        [
            X_train.drop(
                columns=ONEHOT_COLUMNS
            ),
            train_encoded,
        ],
        axis=1,
    )

    X_test = pd.concat(
        [
            X_test.drop(
                columns=ONEHOT_COLUMNS
            ),
            test_encoded,
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

    print(
        "✅ One-Hot Encoding terminé : "
        f"{len(feature_names)} variable(s) créée(s)."
    )

    return (
        X_train,
        X_test,
        encoder,
        feature_names,
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
    """

    excluded_from_scaling = set(
        list(
            INDICATOR_COLUMNS.values()
        )
        + onehot_feature_names
    )

    numeric_columns = (
        X_train
        .select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    columns_to_scale = [
        column
        for column in numeric_columns
        if column not in excluded_from_scaling
    ]

    if not columns_to_scale:
        raise ValueError(
            "Aucune variable numérique continue à standardiser."
        )

    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[
        columns_to_scale
    ] = (
        scaler.fit_transform(
            X_train[
                columns_to_scale
            ]
        )
    )

    X_test[
        columns_to_scale
    ] = (
        scaler.transform(
            X_test[
                columns_to_scale
            ]
        )
    )

    print(
        "✅ Standardisation terminée : "
        f"{len(columns_to_scale)} "
        "variable(s) standardisée(s)."
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
        .select_dtypes(
            exclude="number"
        )
        .columns
        .tolist()
    )

    test_non_numeric = (
        X_test
        .select_dtypes(
            exclude="number"
        )
        .columns
        .tolist()
    )

    indicator_columns = list(
        INDICATOR_COLUMNS.values()
    )

    indicators_valid = all(
        (
            column in X_train.columns
            and column in X_test.columns

            and set(
                X_train[
                    column
                ]
                .unique()
            ).issubset(
                {0, 1}
            )

            and set(
                X_test[
                    column
                ]
                .unique()
            ).issubset(
                {0, 1}
            )
        )
        for column
        in indicator_columns
    )

    checks = {
        "Aucun NaN Train":
            not X_train
            .isna()
            .any()
            .any(),

        "Aucun NaN Test":
            not X_test
            .isna()
            .any()
            .any(),

        "Colonnes Train / Test identiques":
            X_train.columns.tolist()
            == X_test.columns.tolist(),

        "Aucune variable non numérique Train":
            len(
                train_non_numeric
            ) == 0,

        "Aucune variable non numérique Test":
            len(
                test_non_numeric
            ) == 0,

        "Variables catégorielles originales supprimées":
            all(
                column not in X_train.columns
                and column not in X_test.columns
                for column
                in ONEHOT_COLUMNS
            ),

        "Variables exclues absentes":
            all(
                column not in X_train.columns
                and column not in X_test.columns
                for column
                in EXCLUDED_FEATURES
            ),

        "Indicateurs binaires conformes":
            indicators_valid,

        "Cohérence X_train / y_train":
            len(X_train)
            == len(y_train),

        "Cohérence X_test / y_test":
            len(X_test)
            == len(y_test),

        "Aucune valeur infinie Train":
            not np.isinf(
                X_train
                .to_numpy()
            ).any(),

        "Aucune valeur infinie Test":
            not np.isinf(
                X_test
                .to_numpy()
            ).any(),
    }

    failed = [
        name
        for name, result
        in checks.items()
        if not result
    ]

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

        status = (
            "✅"
            if result
            else "❌"
        )

        print(
            f"{status} {name}"
        )

    if failed:
        raise ValueError(
            "Validation finale échouée : "
            + ", ".join(failed)
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
) -> dict[
    str,
    Path,
]:
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
        paths[
            "X_train"
        ],
        index=True,
    )

    X_test.to_parquet(
        paths[
            "X_test"
        ],
        index=True,
    )

    y_train.to_frame(
        name=TARGET_COLUMN
    ).to_parquet(
        paths[
            "y_train"
        ],
        index=True,
    )

    y_test.to_frame(
        name=TARGET_COLUMN
    ).to_parquet(
        paths[
            "y_test"
        ],
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
    categorical_imputation_values: dict[
        str,
        str,
    ],
    numeric_imputation_values: dict[
        str,
        float,
    ],
    onehot_encoder: OneHotEncoder,
    onehot_feature_names: list[str],
    standard_scaler: StandardScaler,
    columns_to_scale: list[str],
    final_features: list[str],
    output_dir: Path,
) -> dict[
    str,
    Path,
]:
    """
    Sauvegarde les paramètres nécessaires à l'inférence.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "imputation_values":
            output_dir
            / "imputation_values.joblib",

        "onehot_encoder":
            output_dir
            / "onehot_encoder.joblib",

        "standard_scaler":
            output_dir
            / "standard_scaler.joblib",

        "metadata":
            output_dir
            / "preprocessing_metadata.json",
    }

    imputation_payload = {
        "categorical_imputation_values":
            categorical_imputation_values,

        "numeric_imputation_values":
            numeric_imputation_values,
    }

    joblib.dump(
        imputation_payload,
        paths[
            "imputation_values"
        ],
    )

    joblib.dump(
        onehot_encoder,
        paths[
            "onehot_encoder"
        ],
    )

    joblib.dump(
        standard_scaler,
        paths[
            "standard_scaler"
        ],
    )

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

        "categorical_imputation": {
            "columns":
                CATEGORICAL_IMPUTATION_COLUMNS,

            "artifact":
                "imputation_values.joblib",
        },

        "numeric_imputation": {
            "columns":
                list(
                    numeric_imputation_values.keys()
                ),

            "artifact":
                "imputation_values.joblib",

            "co2_reduction_wltp_g_km_missing_value":
                0.0,
        },

        "conditional_imputation": {
            "electric_fuel_types":
                sorted(
                    ELECTRIC_FUEL_TYPES
                ),

            "thermal_engine_fuel_types":
                sorted(
                    THERMAL_ENGINE_FUEL_TYPES
                ),

            "fuel_consuming_types":
                sorted(
                    FUEL_CONSUMING_TYPES
                ),
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
        paths[
            "metadata"
        ],
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )

    for path in paths.values():

        if not path.is_file():
            raise FileNotFoundError(
                f"Artefact non sauvegardé : {path}"
            )

    print(
        "\n=== Sauvegarde des artefacts de preprocessing ==="
    )

    print(
        f"Répertoire : "
        f"{output_dir}"
    )

    print(
        f"Nombre d'artefacts : "
        f"{len(paths)}"
    )

    print(
        "\nImputations catégorielles :"
    )

    for column, value in (
        categorical_imputation_values.items()
    ):

        print(
            f"  - {column} : {value}"
        )

    print(
        "\nImputations numériques :"
    )

    for column, value in (
        numeric_imputation_values.items()
    ):

        print(
            f"  - {column} : {value}"
        )

    print(
        f"\nNombre de variables finales : "
        f"{len(final_features)}"
    )

    print(
        "✅ Artefacts de preprocessing sauvegardés."
    )

    return paths


# =====================================================================
# Validation du rechargement des artefacts
# =====================================================================

def validate_saved_artifacts(
    artifact_paths: dict[
        str,
        Path,
    ],
    final_features: list[str],
) -> None:
    """
    Vérifie que les artefacts sauvegardés sont rechargeables.
    """

    imputation_payload = joblib.load(
        artifact_paths[
            "imputation_values"
        ]
    )

    onehot_encoder = joblib.load(
        artifact_paths[
            "onehot_encoder"
        ]
    )

    scaler = joblib.load(
        artifact_paths[
            "standard_scaler"
        ]
    )

    with open(
        artifact_paths[
            "metadata"
        ],
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    categorical_values = (
        imputation_payload.get(
            "categorical_imputation_values",
            {},
        )
    )

    numeric_values = (
        imputation_payload.get(
            "numeric_imputation_values",
            {},
        )
    )

    expected_numeric_columns = (
        set(
            CONDITIONAL_IMPUTATION_RULES
        )
        | set(
            RARE_MEDIAN_COLUMNS
        )
    )

    checks = {
        "Imputations catégorielles rechargeables":
            set(
                categorical_values
            )
            == set(
                CATEGORICAL_IMPUTATION_COLUMNS
            ),

        "Imputations numériques rechargeables":
            (
                set(
                    numeric_values
                )
                == expected_numeric_columns

                and all(
                    pd.notna(
                        value
                    )
                    for value
                    in numeric_values.values()
                )
            ),

        "OneHotEncoder rechargeable":
            (
                hasattr(
                    onehot_encoder,
                    "categories_",
                )

                and len(
                    onehot_encoder.categories_
                )
                == len(
                    ONEHOT_COLUMNS
                )
            ),

        "StandardScaler rechargeable":
            (
                hasattr(
                    scaler,
                    "mean_",
                )

                and hasattr(
                    scaler,
                    "scale_",
                )

                and len(
                    scaler.mean_
                )
                == len(
                    metadata[
                        "standardization"
                    ][
                        "columns"
                    ]
                )
            ),

        "Structure finale conforme":
            (
                metadata[
                    "final_features"
                ]
                == final_features
            ),
    }

    failed = [
        name
        for name, result
        in checks.items()
        if not result
    ]

    print(
        "\n=== Validation du rechargement des artefacts ==="
    )

    for name, result in checks.items():

        status = (
            "✅"
            if result
            else "❌"
        )

        print(
            f"{status} {name}"
        )

    if failed:
        raise ValueError(
            "Validation des artefacts échouée : "
            + ", ".join(failed)
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
    # 1. Chargement
    # -----------------------------------------------------------------

    df = load_data(
        input_path=input_path,
        nrows=nrows,
    )

    validate_input_schema(
        df
    )

    # -----------------------------------------------------------------
    # 2. Split Train / Test
    # -----------------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_train_test(
        df
    )

    del df

    # -----------------------------------------------------------------
    # 3. Imputation catégorielle
    # -----------------------------------------------------------------

    (
        X_train,
        X_test,
        categorical_imputation_values,
    ) = impute_categorical_features(
        X_train=X_train,
        X_test=X_test,
    )

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
    # 5. Imputation numérique
    # -----------------------------------------------------------------

    (
        X_train,
        X_test,
        numeric_imputation_values,
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
        onehot_feature_names=(
            onehot_feature_names
        ),
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
    # 9. Sauvegarde datasets
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
    # 10. Sauvegarde artefacts
    # -----------------------------------------------------------------

    artifact_paths = (
        save_preprocessing_artifacts(
            categorical_imputation_values=(
                categorical_imputation_values
            ),
            numeric_imputation_values=(
                numeric_imputation_values
            ),
            onehot_encoder=(
                onehot_encoder
            ),
            onehot_feature_names=(
                onehot_feature_names
            ),
            standard_scaler=(
                standard_scaler
            ),
            columns_to_scale=(
                columns_to_scale
            ),
            final_features=(
                X_train
                .columns
                .tolist()
            ),
            output_dir=(
                artifact_output_dir
            ),
        )
    )

    # -----------------------------------------------------------------
    # 11. Validation du rechargement
    # -----------------------------------------------------------------

    validate_saved_artifacts(
        artifact_paths=(
            artifact_paths
        ),
        final_features=(
            X_train
            .columns
            .tolist()
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
        parser
        .add_mutually_exclusive_group(
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

    project_root = (
        Path(
            __file__
        )
        .resolve()
        .parents[
            3
        ]
    )

    input_path = (
        project_root
        / "data"
        / "interim"
        / "data_2024_features.csv"
    )

    data_output_dir = (
        project_root
        / "data"
        / "processed"
    )

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

    print(
        f"Mode d'exécution : {mode}"
    )

    print(
        f"Fichier source   : "
        f"{input_path}"
    )

    print(
        f"Sortie datasets  : "
        f"{data_output_dir}"
    )

    print(
        f"Sortie artefacts : "
        f"{artifact_output_dir}"
    )

    run_preprocessing(
        input_path=input_path,
        data_output_dir=data_output_dir,
        artifact_output_dir=artifact_output_dir,
        nrows=nrows,
        suffix=suffix,
    )


if __name__ == "__main__":
    main()