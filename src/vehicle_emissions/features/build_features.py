import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration générale
# ---------------------------------------------------------------------

TARGET_COLUMN = "co2_wltp_g_km"
DATE_COLUMN = "registration_date"

NROWS_TEST = 100_000

TEMPORAL_FEATURES = {
    "registration_month_sin",
    "registration_month_cos",
}


# ---------------------------------------------------------------------
# Schéma d'entrée attendu
# ---------------------------------------------------------------------
#
# REQUIRED_COLUMNS contient uniquement :
#
# 1. les variables nécessaires à une transformation du pipeline ;
# 2. les variables métier retenues dans le dataset final ;
# 3. la variable cible.
#
# Les identifiants, références techniques, variables redondantes ou autres
# variables destinées à être supprimées ne sont volontairement PAS déclarés
# comme obligatoires.
#
# Ainsi, le pipeline reste cohérent avec la sélection établie dans
# le notebook 02 de Feature Engineering.
# ---------------------------------------------------------------------

REQUIRED_COLUMNS = {
    # Variable temporelle nécessaire à la création des features cycliques
    DATE_COLUMN,

    # Variables catégorielles métier conservées
    "manufacturer_make",
    "vehicle_category_type",
    "fuel_type",
    "fuel_mode",

    # Variables numériques métier conservées
    "mass_running_order_kg",
    "wltp_test_mass_kg",
    "engine_capacity_cm3",
    "engine_power_kw",
    "electric_energy_consumption_wh_km",
    "co2_reduction_wltp_g_km",
    "fuel_consumption",
    "electric_range_km",

    # Variable cible
    TARGET_COLUMN,
}


# ---------------------------------------------------------------------
# Variables à exclure
# ---------------------------------------------------------------------
#
# Cette liste reprend les décisions finales du notebook 02.
#
# Ces colonnes sont supprimées lorsqu'elles sont présentes dans le dataset.
# Leur absence à l'entrée n'empêche donc pas le pipeline de fonctionner.
#
# registration_month n'apparaît pas ici :
# dans le script de production, le mois est manipulé comme une Series
# temporaire et n'est jamais ajouté au DataFrame.
# ---------------------------------------------------------------------

COLUMNS_TO_EXCLUDE = {
    # Variable de contexte géographique
    "country",

    # Identifiant technique
    "vehicle_record_id",

    # Références techniques ou administratives
    "vehicle_family_id",
    "type_approval_number",
    "vehicle_type",
    "vehicle_variant",
    "vehicle_version",
    "rlfi",

    # Variable textuelle composite à forte cardinalité
    "commercial_name",

    # Variables dont les modalités ne sont pas exploitables
    # de manière homogène dans l'étude
    "innovative_technology",
    "emission_standard",

    # Variables constructeur non retenues
    "manufacturer_pool",
    "manufacturer_name_eu",
    "manufacturer_name_oem",

    # Date brute remplacée par les features temporelles
    DATE_COLUMN,

    # Variable constante
    "vehicle_category",
}


# ---------------------------------------------------------------------
# Schéma final attendu
# ---------------------------------------------------------------------
#
# Ce schéma constitue le contrat de sortie du Feature Engineering.
#
# Il contient :
# - 12 variables explicatives métier issues du dataset nettoyé ;
# - 2 features temporelles créées par le pipeline ;
# - 1 variable cible.
#
# Total attendu : 15 variables.
# ---------------------------------------------------------------------

EXPECTED_OUTPUT_COLUMNS = {
    # Variables catégorielles conservées
    "manufacturer_make",
    "vehicle_category_type",
    "fuel_type",
    "fuel_mode",

    # Variables numériques conservées
    "mass_running_order_kg",
    "wltp_test_mass_kg",
    "engine_capacity_cm3",
    "engine_power_kw",
    "electric_energy_consumption_wh_km",
    "co2_reduction_wltp_g_km",
    "fuel_consumption",
    "electric_range_km",

    # Features temporelles
    "registration_month_sin",
    "registration_month_cos",

    # Variable cible
    TARGET_COLUMN,
}


# ---------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------

def load_data(
    input_path: Path,
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Charge le dataset nettoyé utilisé pour le Feature Engineering.

    Parameters
    ----------
    input_path : Path
        Chemin vers le dataset nettoyé.

    nrows : int | None
        Nombre maximal de lignes à charger.
        Si None, l'intégralité du dataset est chargée.

    Returns
    -------
    pd.DataFrame
        Dataset chargé.
    """

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Dataset nettoyé introuvable : {input_path}"
        )

    print(f"Chargement : {input_path}")

    df = pd.read_csv(
        input_path,
        nrows=nrows,
        low_memory=False,
        parse_dates=[DATE_COLUMN],
    )

    print(
        f"Dataset chargé : "
        f"{len(df):,} observations × "
        f"{df.shape[1]} variables"
    )

    return df


# ---------------------------------------------------------------------
# Validation du schéma d'entrée
# ---------------------------------------------------------------------

def validate_input_schema(
    df: pd.DataFrame,
) -> None:
    """
    Vérifie la conformité minimale du dataset d'entrée.

    Les contrôles portent uniquement sur les variables nécessaires
    au Feature Engineering et au dataset final.
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
            "Les noms de colonnes du dataset d'entrée "
            "ne sont pas uniques."
        )

    if not pd.api.types.is_datetime64_any_dtype(
        df[DATE_COLUMN]
    ):
        raise TypeError(
            f"La colonne '{DATE_COLUMN}' "
            "doit être au format datetime."
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"La variable cible '{TARGET_COLUMN}' "
            "contient des valeurs manquantes."
        )

    print("✅ Schéma d'entrée valide.")


# ---------------------------------------------------------------------
# Feature Engineering temporel
# ---------------------------------------------------------------------

def add_temporal_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Crée les représentations cycliques du mois d'immatriculation.

    Le mois brut est utilisé comme variable intermédiaire uniquement.
    Il n'est jamais ajouté au DataFrame.
    """

    registration_month = (
        df[DATE_COLUMN]
        .dt
        .month
    )

    if registration_month.isna().any():
        raise ValueError(
            "Des valeurs manquantes ont été détectées "
            "dans le mois d'immatriculation."
        )

    df = df.copy()

    df["registration_month_sin"] = np.sin(
        2
        * np.pi
        * (registration_month - 1)
        / 12
    )

    df["registration_month_cos"] = np.cos(
        2
        * np.pi
        * (registration_month - 1)
        / 12
    )

    print(
        "✅ Features temporelles créées : "
        "registration_month_sin, "
        "registration_month_cos."
    )

    return df


# ---------------------------------------------------------------------
# Sélection finale des variables
# ---------------------------------------------------------------------

def select_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Applique les décisions finales de sélection établies
    dans le notebook 02 de Feature Engineering.

    Les colonnes définies dans COLUMNS_TO_EXCLUDE sont supprimées
    lorsqu'elles sont présentes.

    Le dataset est ensuite limité explicitement au schéma final attendu.
    """

    columns_present_to_exclude = sorted(
        COLUMNS_TO_EXCLUDE.intersection(
            df.columns
        )
    )

    df_features = df.drop(
        columns=columns_present_to_exclude
    ).copy()

    print("Variables exclues :")

    for column in columns_present_to_exclude:
        print(f"  - {column}")

    # Vérification avant sélection stricte
    missing_expected_columns = sorted(
        EXPECTED_OUTPUT_COLUMNS
        - set(df_features.columns)
    )

    if missing_expected_columns:
        raise ValueError(
            "Variables attendues dans le dataset final absentes : "
            + ", ".join(missing_expected_columns)
        )

    # Conservation explicite du schéma final.
    #
    # L'ordre d'origine des colonnes est préservé afin de garder
    # un dataset lisible et stable.
    final_columns = [
        column
        for column in df_features.columns
        if column in EXPECTED_OUTPUT_COLUMNS
    ]

    df_features = df_features[
        final_columns
    ].copy()

    print(
        f"Dataset de features : "
        f"{len(df_features):,} observations × "
        f"{df_features.shape[1]} variables"
    )

    return df_features


# ---------------------------------------------------------------------
# Contrôles qualité du dataset final
# ---------------------------------------------------------------------

def validate_feature_dataset(
    df: pd.DataFrame,
) -> None:
    """
    Vérifie que le dataset final respecte le contrat de sortie
    défini par le Feature Engineering.
    """

    actual_columns = set(df.columns)

    quality_checks = {
        "Variable cible présente":
            TARGET_COLUMN in actual_columns,

        "Noms de colonnes uniques":
            df.columns.is_unique,

        "Toutes les variables exclues sont absentes":
            COLUMNS_TO_EXCLUDE.isdisjoint(
                actual_columns
            ),

        "Feature mois sinus présente":
            "registration_month_sin"
            in actual_columns,

        "Feature mois cosinus présente":
            "registration_month_cos"
            in actual_columns,

        "Features temporelles sans valeur manquante":
            not df[
                sorted(TEMPORAL_FEATURES)
            ]
            .isna()
            .any()
            .any(),

        "Schéma final conforme":
            actual_columns
            == EXPECTED_OUTPUT_COLUMNS,

        "Nombre final de variables conforme":
            df.shape[1]
            == len(EXPECTED_OUTPUT_COLUMNS),
    }

    failed_checks = []

    for check, result in quality_checks.items():
        status = "✅" if result else "❌"
        print(f"{status} {check}")

        if not result:
            failed_checks.append(check)

    if failed_checks:
        missing_columns = sorted(
            EXPECTED_OUTPUT_COLUMNS
            - actual_columns
        )

        unexpected_columns = sorted(
            actual_columns
            - EXPECTED_OUTPUT_COLUMNS
        )

        details = []

        if missing_columns:
            details.append(
                "variables manquantes : "
                + ", ".join(missing_columns)
            )

        if unexpected_columns:
            details.append(
                "variables inattendues : "
                + ", ".join(unexpected_columns)
            )

        message = (
            "Contrôles qualité échoués : "
            + ", ".join(failed_checks)
        )

        if details:
            message += " | " + " | ".join(details)

        raise ValueError(message)

    print("✅ Dataset de features validé.")


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------

def save_features(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Exporte le dataset final de Feature Engineering.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print("✅ Dataset de features exporté.")
    print(f"Fichier      : {output_path}")
    print(f"Observations : {len(df):,}")
    print(f"Variables    : {df.shape[1]}")


# ---------------------------------------------------------------------
# Pipeline de Feature Engineering
# ---------------------------------------------------------------------

def build_features(
    input_path: Path,
    output_path: Path,
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Exécute le pipeline complet de Feature Engineering.
    """

    df = load_data(
        input_path=input_path,
        nrows=nrows,
    )

    validate_input_schema(df)

    df = add_temporal_features(df)

    df_features = select_features(df)

    validate_feature_dataset(
        df_features
    )

    save_features(
        df=df_features,
        output_path=output_path,
    )

    return df_features


# ---------------------------------------------------------------------
# Interface CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Définit les arguments de ligne de commande.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Feature Engineering du dataset "
            "Vehicle Emissions 2024."
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
            "Exécute le Feature Engineering "
            "sur les 100 000 premières observations."
        ),
    )

    mode_group.add_argument(
        "--full",
        action="store_true",
        help=(
            "Exécute le Feature Engineering "
            "sur l'intégralité du dataset nettoyé."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------

def main() -> None:
    """
    Point d'entrée du pipeline de Feature Engineering.
    """

    args = parse_args()

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    input_path = (
        project_root
        / "data"
        / "processed"
        / "data_2024_cleaned.csv"
    )

    if args.test:
        output_path = (
            project_root
            / "data"
            / "interim"
            / "data_2024_features_test.csv"
        )

        nrows = NROWS_TEST
        mode = "TEST"

    else:
        output_path = (
            project_root
            / "data"
            / "interim"
            / "data_2024_features.csv"
        )

        nrows = None
        mode = "COMPLET"

    print(f"Mode d'exécution : {mode}")
    print(f"Fichier source : {input_path}")
    print(f"Fichier cible  : {output_path}")

    build_features(
        input_path=input_path,
        output_path=output_path,
        nrows=nrows,
    )


if __name__ == "__main__":
    main()