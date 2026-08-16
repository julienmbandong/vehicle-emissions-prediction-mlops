"""
Train / Test preprocessing
==========================

Industrialisation du preprocessing validé dans :

notebooks/03_ml_preprocessing/03_train_test_preprocessing.ipynb

Le script :

1. charge le dataset issu du Feature Engineering ;
2. sépare X / y ;
3. construit Train / Test ;
4. crée les indicateurs binaires de présence ;
5. traite les NaN structurels et résiduels ;
6. traite manufacturer_make ;
7. applique le Frequency Encoding ;
8. applique le One-Hot Encoding ;
9. standardise les variables numériques continues ;
10. valide les matrices finales ;
11. sauvegarde les datasets prétraités ;
12. sauvegarde les artefacts nécessaires à l'inférence ;
13. vérifie leur rechargement.

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
# Variables explicatives attendues
# ---------------------------------------------------------------------

FEATURE_COLUMNS = [
    "manufacturer_make",
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
    FEATURE_COLUMNS + [TARGET_COLUMN]
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
# Variables One-Hot
# ---------------------------------------------------------------------

ONEHOT_COLUMNS = [
    "vehicle_category_type",
    "fuel_type",
    "fuel_mode",
]


# ---------------------------------------------------------------------
# Variables numériques avec NaN rares
# ---------------------------------------------------------------------

RARE_MEDIAN_COLUMNS = [
    "wltp_test_mass_kg",
    "engine_power_kw",
]


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

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")
    print(f"y_train : {y_train.shape}")
    print(f"y_test  : {y_test.shape}")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
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
        "indicateurs binaires créés."
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


    # -----------------------------------------------------------------
    # Identification des NaN
    # -----------------------------------------------------------------

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


    # -----------------------------------------------------------------
    # Médiane apprise exclusivement sur Train
    # -----------------------------------------------------------------

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


    # -----------------------------------------------------------------
    # NaN structurels -> 0
    # -----------------------------------------------------------------

    X_train.loc[
        train_structural_nan,
        column,
    ] = 0.0

    X_test.loc[
        test_structural_nan,
        column,
    ] = 0.0


    # -----------------------------------------------------------------
    # NaN résiduels -> médiane Train
    # -----------------------------------------------------------------

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
    Applique les décisions métier validées dans le notebook 03.
    """

    conditional_medians: dict[str, float] = {}


    conditional_rules = {
        "engine_capacity_cm3":
            THERMAL_ENGINE_FUEL_TYPES,

        "electric_energy_consumption_wh_km":
            ELECTRIC_FUEL_TYPES,

        "electric_range_km":
            ELECTRIC_FUEL_TYPES,

        "fuel_consumption":
            FUEL_CONSUMING_TYPES,
    }


    print(
        "\nTraitement conditionnel des NaN :"
    )


    for column, applicable_types in (
        conditional_rules.items()
    ):

        median_train, report = (
            apply_conditional_imputation(
                X_train=X_train,
                X_test=X_test,
                column=column,
                applicable_fuel_types=applicable_types,
            )
        )

        conditional_medians[column] = (
            median_train
        )

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
    # co2_reduction_wltp_g_km
    # -----------------------------------------------------------------

    X_train["co2_reduction_wltp_g_km"] = (
        X_train["co2_reduction_wltp_g_km"]
        .fillna(0.0)
    )

    X_test["co2_reduction_wltp_g_km"] = (
        X_test["co2_reduction_wltp_g_km"]
        .fillna(0.0)
    )


    # -----------------------------------------------------------------
    # Variables avec NaN rares
    # -----------------------------------------------------------------

    rare_medians: dict[str, float] = {}

    for column in RARE_MEDIAN_COLUMNS:

        median_train = float(
            X_train[column].median()
        )

        if pd.isna(median_train):
            raise ValueError(
                f"Impossible de calculer "
                f"la médiane de '{column}'."
            )

        rare_medians[column] = median_train

        X_train[column] = (
            X_train[column]
            .fillna(median_train)
        )

        X_test[column] = (
            X_test[column]
            .fillna(median_train)
        )


    imputation_values = {
        **conditional_medians,
        **rare_medians,
    }


    print(
        "✅ Traitement numérique des NaN terminé."
    )


    return (
        X_train,
        X_test,
        imputation_values,
    )


# =====================================================================
# manufacturer_make : imputation catégorielle
# =====================================================================

def impute_manufacturer_make(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Impute manufacturer_make par sa modalité la plus fréquente
    apprise exclusivement sur Train.
    """

    train_mode = (
        X_train["manufacturer_make"]
        .mode(dropna=True)
    )

    if train_mode.empty:
        raise ValueError(
            "Impossible de déterminer la modalité "
            "la plus fréquente de manufacturer_make."
        )

    manufacturer_mode = str(
        train_mode.iloc[0]
    )


    X_train["manufacturer_make"] = (
        X_train["manufacturer_make"]
        .fillna(manufacturer_mode)
    )

    X_test["manufacturer_make"] = (
        X_test["manufacturer_make"]
        .fillna(manufacturer_mode)
    )


    print(
        "✅ manufacturer_make imputée avec "
        f"la modalité Train : {manufacturer_mode}"
    )


    return (
        X_train,
        X_test,
        manufacturer_mode,
    )


# =====================================================================
# Frequency Encoding
# =====================================================================

def frequency_encode_manufacturer(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
]:
    """
    Applique un Frequency Encoding à manufacturer_make.

    Les fréquences sont apprises exclusivement sur Train.
    """

    source_column = "manufacturer_make"

    encoded_column = (
        "manufacturer_make_frequency"
    )


    frequency_map = (
        X_train[source_column]
        .value_counts(normalize=True)
    )


    X_train[encoded_column] = (
        X_train[source_column]
        .map(frequency_map)
        .fillna(0.0)
    )

    X_test[encoded_column] = (
        X_test[source_column]
        .map(frequency_map)
        .fillna(0.0)
    )


    unknown_test_mask = (
        ~X_test[source_column]
        .isin(frequency_map.index)
    )

    unknown_test_categories = int(
        X_test.loc[
            unknown_test_mask,
            source_column,
        ]
        .nunique()
    )

    unknown_test_rows = int(
        unknown_test_mask.sum()
    )


    X_train.drop(
        columns=[source_column],
        inplace=True,
    )

    X_test.drop(
        columns=[source_column],
        inplace=True,
    )


    print(
        "✅ Frequency Encoding manufacturer_make : "
        f"{len(frequency_map):,} modalités apprises ; "
        f"{unknown_test_categories:,} modalité(s) "
        f"inconnue(s) dans Test ; "
        f"{unknown_test_rows:,} observation(s) concernée(s)."
    )


    return (
        X_train,
        X_test,
        frequency_map,
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
    Apprend le OneHotEncoder exclusivement sur Train.
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


    X_train.drop(
        columns=ONEHOT_COLUMNS,
        inplace=True,
    )

    X_test.drop(
        columns=ONEHOT_COLUMNS,
        inplace=True,
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


    print(
        "✅ One-Hot Encoding : "
        f"{len(onehot_feature_names)} "
        "variable(s) créée(s)."
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

    Les indicateurs binaires et variables One-Hot restent en 0/1.
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


    scaler = StandardScaler()


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
        "✅ Standardisation : "
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


    print("\nValidation finale :")


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

    y_train.to_frame().to_parquet(
        paths["y_train"],
        index=True,
    )

    y_test.to_frame().to_parquet(
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
    frequency_map: pd.Series,
    manufacturer_mode: str,
    imputation_values: dict[str, float],
    onehot_encoder: OneHotEncoder,
    onehot_feature_names: list[str],
    standard_scaler: StandardScaler,
    columns_to_scale: list[str],
    final_features: list[str],
    output_dir: Path,
) -> dict[str, Path]:
    """
    Sauvegarde tous les paramètres appris sur Train nécessaires
    pour reproduire le preprocessing en inférence.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    frequency_map_path = (
        output_dir
        / "manufacturer_frequency_map.joblib"
    )

    imputation_values_path = (
        output_dir
        / "imputation_values.joblib"
    )

    onehot_encoder_path = (
        output_dir
        / "onehot_encoder.joblib"
    )

    scaler_path = (
        output_dir
        / "standard_scaler.joblib"
    )

    metadata_path = (
        output_dir
        / "preprocessing_metadata.json"
    )


    # -----------------------------------------------------------------
    # Sauvegarde des objets Python
    # -----------------------------------------------------------------

    joblib.dump(
        frequency_map,
        frequency_map_path,
    )

    joblib.dump(
        {
            "manufacturer_make_mode":
                manufacturer_mode,

            "numeric_imputation_values":
                imputation_values,
        },
        imputation_values_path,
    )

    joblib.dump(
        onehot_encoder,
        onehot_encoder_path,
    )

    joblib.dump(
        standard_scaler,
        scaler_path,
    )


    # -----------------------------------------------------------------
    # Métadonnées
    # -----------------------------------------------------------------

    metadata = {
        "target_column":
            TARGET_COLUMN,

        "split": {
            "test_size":
                TEST_SIZE,

            "random_state":
                RANDOM_STATE,
        },

        "missing_indicators":
            list(
                INDICATOR_COLUMNS.values()
            ),

        "conditional_imputation": {
            "electric_fuel_types":
                sorted(ELECTRIC_FUEL_TYPES),

            "thermal_engine_fuel_types":
                sorted(THERMAL_ENGINE_FUEL_TYPES),

            "fuel_consuming_types":
                sorted(FUEL_CONSUMING_TYPES),
        },

        "frequency_encoding": {
            "source_column":
                "manufacturer_make",

            "output_column":
                "manufacturer_make_frequency",

            "unknown_value":
                0.0,
        },

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
        "frequency_map":
            frequency_map_path,

        "imputation_values":
            imputation_values_path,

        "onehot_encoder":
            onehot_encoder_path,

        "standard_scaler":
            scaler_path,

        "metadata":
            metadata_path,
    }


    for path in paths.values():

        if not path.is_file():
            raise FileNotFoundError(
                f"Artefact non sauvegardé : {path}"
            )


    print(
        "\n✅ Artefacts de preprocessing sauvegardés :"
    )

    print(
        "  - Frequency Encoding"
    )

    print(
        "  - Valeurs d'imputation"
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


    return paths


# =====================================================================
# Validation du rechargement des artefacts
# =====================================================================

def validate_saved_artifacts(
    artifact_paths: dict[str, Path],
    final_features: list[str],
) -> None:
    """
    Vérifie que les artefacts sont rechargeables et cohérents.
    """

    frequency_map = joblib.load(
        artifact_paths["frequency_map"]
    )

    imputation_values = joblib.load(
        artifact_paths["imputation_values"]
    )

    onehot_encoder = joblib.load(
        artifact_paths["onehot_encoder"]
    )

    scaler = joblib.load(
        artifact_paths["standard_scaler"]
    )


    with open(
        artifact_paths["metadata"],
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)


    checks = {
        "Frequency Encoding rechargeable":
            len(frequency_map) > 0,

        "Valeurs d'imputation rechargeables":
            bool(imputation_values),

        "OneHotEncoder rechargeable":
            hasattr(
                onehot_encoder,
                "categories_",
            ),

        "StandardScaler rechargeable":
            hasattr(
                scaler,
                "mean_",
            )
            and hasattr(
                scaler,
                "scale_",
            ),

        "Structure finale conforme":
            metadata["final_features"]
            == final_features,
    }


    failed_checks = []


    print(
        "\nValidation des artefacts :"
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
        "✅ Artefacts rechargés et validés."
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
    Exécute le pipeline complet Train / Test preprocessing.
    """

    # -----------------------------------------------------------------
    # Chargement
    # -----------------------------------------------------------------

    df = load_data(
        input_path=input_path,
        nrows=nrows,
    )

    validate_input_schema(df)


    # -----------------------------------------------------------------
    # Train / Test
    # -----------------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_train_test(df)


    del df


    # -----------------------------------------------------------------
    # Indicateurs binaires
    # -----------------------------------------------------------------

    (
        X_train,
        X_test,
    ) = add_missing_indicators(
        X_train=X_train,
        X_test=X_test,
    )


    # -----------------------------------------------------------------
    # Imputations numériques
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
    # manufacturer_make
    # -----------------------------------------------------------------

    (
        X_train,
        X_test,
        manufacturer_mode,
    ) = impute_manufacturer_make(
        X_train=X_train,
        X_test=X_test,
    )


    # -----------------------------------------------------------------
    # Frequency Encoding
    # -----------------------------------------------------------------

    (
        X_train,
        X_test,
        frequency_map,
    ) = frequency_encode_manufacturer(
        X_train=X_train,
        X_test=X_test,
    )


    # -----------------------------------------------------------------
    # One-Hot Encoding
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
    # StandardScaler
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
    # Validation finale
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
    # Sauvegarde des datasets
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
    # Sauvegarde des artefacts
    # -----------------------------------------------------------------

    artifact_paths = (
        save_preprocessing_artifacts(
            frequency_map=frequency_map,
            manufacturer_mode=manufacturer_mode,
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
    # Validation des artefacts
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