"""
Train / Test preprocessing
==========================

Industrialisation du preprocessing validé dans :
notebooks/03_ml_preprocessing/03_train_test_preprocessing.ipynb

Principes :
- manufacturer_make est exclue de la modélisation ;
- les paramètres dépendant des données sont appris uniquement sur Train ;
- les mêmes paramètres sont appliqués à Test sans réapprentissage ;
- la gestion des NaN est robuste et indépendante de l'échantillon ;
- les artefacts nécessaires à l'inférence sont sauvegardés et validés.

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
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =====================================================================
# Configuration générale
# =====================================================================

TARGET_COLUMN = "co2_wltp_g_km"

NROWS_TEST = 100_000
TEST_SIZE = 0.20
RANDOM_STATE = 42

EXCLUDED_FEATURES = [
    "manufacturer_make",
]

CATEGORICAL_FEATURE_COLUMNS = [
    "vehicle_category_type",
    "fuel_type",
    "fuel_mode",
]

NUMERIC_FEATURE_COLUMNS = [
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

MODEL_FEATURE_COLUMNS = (
    CATEGORICAL_FEATURE_COLUMNS
    + NUMERIC_FEATURE_COLUMNS
)

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

INFORMATIVE_ZERO_COLUMNS = [
    "co2_reduction_wltp_g_km",
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

ONEHOT_COLUMNS = list(
    CATEGORICAL_FEATURE_COLUMNS
)

GENERAL_NUMERIC_IMPUTATION_COLUMNS = [
    column
    for column in NUMERIC_FEATURE_COLUMNS
    if (
        column not in CONDITIONAL_IMPUTATION_RULES
        and column not in INFORMATIVE_ZERO_COLUMNS
    )
]


# =====================================================================
# Chargement et validation du schéma
# =====================================================================

def load_data(
    input_path: Path,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Charge le dataset issu du Feature Engineering."""

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


def validate_input_schema(
    df: pd.DataFrame,
) -> None:
    """Vérifie la conformité du dataset d'entrée."""

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

    non_numeric_columns = [
        column
        for column in (
            NUMERIC_FEATURE_COLUMNS
            + [TARGET_COLUMN]
        )
        if not is_numeric_dtype(df[column])
    ]

    if non_numeric_columns:
        raise TypeError(
            "Les variables suivantes devraient être numériques : "
            + ", ".join(non_numeric_columns)
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"La cible '{TARGET_COLUMN}' contient des NaN."
        )

    target_values = df[
        TARGET_COLUMN
    ].to_numpy()

    if not np.isfinite(target_values).all():
        raise ValueError(
            f"La cible '{TARGET_COLUMN}' contient "
            "des valeurs infinies."
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
    """Construit X, y puis les jeux Train / Test."""

    X = df[
        MODEL_FEATURE_COLUMNS
    ].copy()

    y = df[
        TARGET_COLUMN
    ].copy()

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
# Imputation catégorielle robuste
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
    Impute toutes les variables catégorielles par leur modalité Train.

    La modalité est apprise pour chaque variable catégorielle, même si
    aucun NaN n'est présent dans l'échantillon courant. Elle est ainsi
    disponible pour l'inférence future.
    """

    X_train = X_train.copy()
    X_test = X_test.copy()

    imputation_values: dict[str, str] = {}

    print(
        "\n=== Traitement des valeurs manquantes catégorielles ==="
    )

    for column in CATEGORICAL_FEATURE_COLUMNS:

        train_mode = (
            X_train[column]
            .mode(dropna=True)
        )

        if train_mode.empty:
            raise ValueError(
                f"La variable catégorielle '{column}' est entièrement "
                "manquante dans X_train. "
                "Aucune modalité d'imputation ne peut être apprise."
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

        imputation_values[
            column
        ] = mode_value

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

    train_nan_report = (
        X_train[
            CATEGORICAL_FEATURE_COLUMNS
        ]
        .isna()
        .sum()
    )

    test_nan_report = (
        X_test[
            CATEGORICAL_FEATURE_COLUMNS
        ]
        .isna()
        .sum()
    )

    train_nan_report = (
        train_nan_report[
            train_nan_report > 0
        ]
    )

    test_nan_report = (
        test_nan_report[
            test_nan_report > 0
        ]
    )

    if (
        not train_nan_report.empty
        or not test_nan_report.empty
    ):
        raise ValueError(
            "Des NaN catégoriels subsistent après imputation.\n"
            f"Train : {train_nan_report.to_dict()}\n"
            f"Test : {test_nan_report.to_dict()}"
        )

    print(
        "✅ Toutes les valeurs manquantes catégorielles "
        "ont été traitées."
    )

    return (
        X_train,
        X_test,
        imputation_values,
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

        X_train[
            indicator_column
        ] = (
            X_train[
                source_column
            ]
            .notna()
            .astype("int8")
        )

        X_test[
            indicator_column
        ] = (
            X_test[
                source_column
            ]
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
# Imputation conditionnelle métier
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
    Applique la règle métier d'une variable numérique.

    - NaN structurel : 0 ;
    - NaN résiduel : médiane apprise exclusivement sur Train.
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

    train_reference_values = (
        X_train.loc[
            train_applicable,
            column,
        ]
        .dropna()
    )

    if train_reference_values.empty:
        raise ValueError(
            f"Aucune valeur Train disponible "
            f"pour calculer la médiane de '{column}'."
        )

    median_train = float(
        train_reference_values.median()
    )

    if not np.isfinite(
        median_train
    ):
        raise ValueError(
            f"Médiane invalide calculée pour '{column}' : "
            f"{median_train}"
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
            int(
                train_structural_nan.sum()
            ),

        "train_residual_nan":
            int(
                train_residual_nan.sum()
            ),

        "test_structural_nan":
            int(
                test_structural_nan.sum()
            ),

        "test_residual_nan":
            int(
                test_residual_nan.sum()
            ),
    }

    return (
        median_train,
        report,
    )


# =====================================================================
# Imputation numérique robuste
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
    Traite toutes les valeurs manquantes numériques.

    Ordre :
    1. règles métier conditionnelles ;
    2. absences informatives -> 0 ;
    3. médiane Train pour toutes les autres variables numériques ;
    4. validation stricte : aucun NaN numérique ne doit subsister.

    Les médianes des variables générales sont apprises même si elles ne
    contiennent aucun NaN dans l'échantillon courant. Elles restent ainsi
    disponibles pour l'inférence future.
    """

    X_train = X_train.copy()
    X_test = X_test.copy()

    imputation_values: dict[str, float] = {}

    print(
        "\n=== Traitement des valeurs manquantes numériques ==="
    )

    # -----------------------------------------------------------------
    # 1. Règles métier conditionnelles
    # -----------------------------------------------------------------

    print(
        "\nTraitement conditionnel métier :"
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

        imputation_values[
            column
        ] = median_train

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
    # 2. Absences informatives -> 0
    # -----------------------------------------------------------------

    print(
        "\nTraitement des absences informatives :"
    )

    for column in INFORMATIVE_ZERO_COLUMNS:

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

        X_train[column] = (
            X_train[column]
            .fillna(0.0)
        )

        X_test[column] = (
            X_test[column]
            .fillna(0.0)
        )

        print(
            f"  - {column}: "
            f"NaN -> 0 | "
            f"Train={train_missing:,} | "
            f"Test={test_missing:,}"
        )

    # -----------------------------------------------------------------
    # 3. Variables numériques générales -> médiane Train
    # -----------------------------------------------------------------

    print(
        "\nImputation générale par médiane apprise sur Train :"
    )

    for column in GENERAL_NUMERIC_IMPUTATION_COLUMNS:

        train_valid_values = (
            X_train[column]
            .dropna()
        )

        if train_valid_values.empty:
            raise ValueError(
                f"La variable numérique '{column}' est entièrement "
                "manquante dans X_train. "
                "Aucune médiane ne peut être apprise."
            )

        median_train = float(
            train_valid_values.median()
        )

        if not np.isfinite(
            median_train
        ):
            raise ValueError(
                f"Médiane invalide calculée pour '{column}' : "
                f"{median_train}"
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

        imputation_values[
            column
        ] = median_train

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

    # -----------------------------------------------------------------
    # 4. Filet de sécurité dynamique
    # -----------------------------------------------------------------

    numeric_columns = (
        X_train
        .select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    residual_columns = [
        column
        for column in numeric_columns
        if (
            X_train[column]
            .isna()
            .any()
            or X_test[column]
            .isna()
            .any()
        )
    ]

    if residual_columns:

        print(
            "\nFilet de sécurité : "
            "NaN numériques résiduels détectés."
        )

    for column in residual_columns:

        train_valid_values = (
            X_train[column]
            .dropna()
        )

        if train_valid_values.empty:
            raise ValueError(
                f"La variable numérique '{column}' est entièrement "
                "manquante dans X_train. "
                "Aucune médiane ne peut être apprise."
            )

        median_train = float(
            train_valid_values.median()
        )

        if not np.isfinite(
            median_train
        ):
            raise ValueError(
                f"Médiane invalide calculée pour '{column}' : "
                f"{median_train}"
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

        imputation_values[
            column
        ] = median_train

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

    # -----------------------------------------------------------------
    # 5. Validation stricte
    # -----------------------------------------------------------------

    train_nan_report = (
        X_train
        .isna()
        .sum()
    )

    test_nan_report = (
        X_test
        .isna()
        .sum()
    )

    train_nan_report = (
        train_nan_report[
            train_nan_report > 0
        ]
    )

    test_nan_report = (
        test_nan_report[
            test_nan_report > 0
        ]
    )

    if (
        not train_nan_report.empty
        or not test_nan_report.empty
    ):
        raise ValueError(
            "Des valeurs manquantes subsistent après "
            "le preprocessing numérique.\n"
            f"Train : {train_nan_report.to_dict()}\n"
            f"Test : {test_nan_report.to_dict()}"
        )

    print(
        "\n✅ Toutes les valeurs manquantes numériques "
        "ont été traitées."
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

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.int8,
    )

    train_array = (
        encoder.fit_transform(
            X_train[
                ONEHOT_COLUMNS
            ]
        )
    )

    test_array = (
        encoder.transform(
            X_test[
                ONEHOT_COLUMNS
            ]
        )
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

    Les indicateurs binaires et les variables issues du One-Hot Encoding
    ne sont pas standardisés.
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

    missing_test_columns = [
        column
        for column in columns_to_scale
        if column not in X_test.columns
    ]

    if missing_test_columns:
        raise ValueError(
            "Variables à standardiser absentes de X_test : "
            + ", ".join(missing_test_columns)
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
    """Vérifie que les matrices finales sont prêtes pour la modélisation."""

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
        in INDICATOR_COLUMNS.values()
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
    """Sauvegarde les quatre jeux Train / Test au format Parquet."""

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
    """Sauvegarde les paramètres nécessaires à l'inférence."""

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
        "schema_version":
            2,

        "target_column":
            TARGET_COLUMN,

        "excluded_features":
            EXCLUDED_FEATURES,

        "model_input_features":
            MODEL_FEATURE_COLUMNS,

        "categorical_features":
            CATEGORICAL_FEATURE_COLUMNS,

        "numeric_features":
            NUMERIC_FEATURE_COLUMNS,

        "split": {
            "test_size":
                TEST_SIZE,

            "random_state":
                RANDOM_STATE,
        },

        "imputation": {
            "artifact":
                "imputation_values.joblib",

            "categorical_columns":
                CATEGORICAL_FEATURE_COLUMNS,

            "conditional_numeric_columns":
                list(
                    CONDITIONAL_IMPUTATION_RULES.keys()
                ),

            "general_numeric_columns":
                GENERAL_NUMERIC_IMPUTATION_COLUMNS,

            "informative_zero_columns":
                INFORMATIVE_ZERO_COLUMNS,
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
    """Vérifie que les artefacts sauvegardés sont rechargeables."""

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
            CONDITIONAL_IMPUTATION_RULES.keys()
        )
        | set(
            GENERAL_NUMERIC_IMPUTATION_COLUMNS
        )
    )

    checks = {
        "Imputations catégorielles rechargeables":
            (
                isinstance(
                    categorical_values,
                    dict,
                )
                and set(
                    categorical_values.keys()
                )
                == set(
                    CATEGORICAL_FEATURE_COLUMNS
                )
                and all(
                    isinstance(
                        value,
                        str,
                    )
                    and value != ""
                    for value
                    in categorical_values.values()
                )
            ),

        "Imputations numériques rechargeables":
            (
                isinstance(
                    numeric_values,
                    dict,
                )
                and expected_numeric_columns.issubset(
                    set(
                        numeric_values.keys()
                    )
                )
                and all(
                    np.isfinite(
                        float(
                            value
                        )
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
                and len(
                    scaler.scale_
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
    """Exécute le preprocessing Train / Test complet."""

    df = load_data(
        input_path=input_path,
        nrows=nrows,
    )

    validate_input_schema(
        df
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_train_test(
        df
    )

    del df

    (
        X_train,
        X_test,
        categorical_imputation_values,
    ) = impute_categorical_features(
        X_train=X_train,
        X_test=X_test,
    )

    (
        X_train,
        X_test,
    ) = add_missing_indicators(
        X_train=X_train,
        X_test=X_test,
    )

    (
        X_train,
        X_test,
        numeric_imputation_values,
    ) = impute_numeric_features(
        X_train=X_train,
        X_test=X_test,
    )

    (
        X_train,
        X_test,
        onehot_encoder,
        onehot_feature_names,
    ) = onehot_encode_categories(
        X_train=X_train,
        X_test=X_test,
    )

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

    save_datasets(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        output_dir=data_output_dir,
        suffix=suffix,
    )

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
    """Définit les modes TEST et FULL."""

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
    """Point d'entrée du script."""

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