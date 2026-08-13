import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TARGET_COLUMN = "co2_wltp_g_km"
DATE_COLUMN = "registration_date"

NROWS_TEST = 100_000

TEMPORAL_FEATURES = {
    "registration_month_sin",
    "registration_month_cos",
}

# Colonnes nécessaires au fonctionnement du pipeline.
# Il s'agit des variables effectivement utilisées par les transformations
# ou indispensables aux contrôles d'entrée.
REQUIRED_COLUMNS = {
    "vehicle_record_id",
    "country",
    "manufacturer_make",
    "vehicle_category_type",
    "fuel_type",
    "fuel_mode",
    "mass_running_order_kg",
    "engine_capacity_cm3",
    "engine_power_kw",
    DATE_COLUMN,
    TARGET_COLUMN,
}

# Sélection finale établie dans le notebook 02.
#
# registration_month n'apparaît pas ici car, contrairement au notebook,
# le script de production ne l'ajoute jamais au DataFrame :
# le mois est manipulé comme Series temporaire.
COLUMNS_TO_EXCLUDE = {
    # Identifiant technique
    "vehicle_record_id",

    # Références techniques ou administratives
    "vehicle_family_id",
    "type_approval_number",
    "vehicle_variant",
    "vehicle_version",
    "rlfi",

    # Date brute remplacée par les composantes temporelles
    "registration_date",

    # Variable constante
    "vehicle_category",

    # Variables constructeur non retenues
    "manufacturer_pool",
    "manufacturer_name_eu",
    "manufacturer_name_oem",
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
        Chemin du dataset nettoyé.
    nrows : int | None
        Nombre maximal de lignes à charger.
        None signifie que l'intégralité du fichier est chargée.

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

    Le mois brut est conservé uniquement dans une Series temporaire.
    Il n'est pas ajouté au DataFrame de production.
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
    Applique la sélection finale des variables établie
    dans le notebook 02 de Feature Engineering.
    """

    columns_present_to_exclude = sorted(
        COLUMNS_TO_EXCLUDE.intersection(
            df.columns
        )
    )

    df = df.drop(
        columns=columns_present_to_exclude
    )

    print("Variables exclues :")

    for column in columns_present_to_exclude:
        print(f"  - {column}")

    print(
        f"Dataset de features : "
        f"{len(df):,} observations × "
        f"{df.shape[1]} variables"
    )

    return df


# ---------------------------------------------------------------------
# Contrôles qualité
# ---------------------------------------------------------------------

def validate_feature_dataset(
    df: pd.DataFrame,
) -> None:
    """
    Vérifie que le dataset final respecte les décisions
    de sélection établies dans le notebook 02.
    """

    quality_checks = {
        "Variable cible présente":
            TARGET_COLUMN in df.columns,

        "Noms de colonnes uniques":
            df.columns.is_unique,

        "Toutes les variables exclues sont absentes":
            COLUMNS_TO_EXCLUDE.isdisjoint(
                df.columns
            ),

        "Feature mois sinus présente":
            "registration_month_sin"
            in df.columns,

        "Feature mois cosinus présente":
            "registration_month_cos"
            in df.columns,

        "Features temporelles sans valeur manquante":
            not df[
                sorted(TEMPORAL_FEATURES)
            ]
            .isna()
            .any()
            .any(),
    }

    failed_checks = []

    for check, result in quality_checks.items():
        status = "✅" if result else "❌"
        print(f"{status} {check}")

        if not result:
            failed_checks.append(check)

    if failed_checks:
        raise ValueError(
            "Contrôles qualité échoués : "
            + ", ".join(failed_checks)
        )

    print("✅ Dataset de features validé.")


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------

def save_features(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Exporte le dataset produit par le Feature Engineering.
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

    df = select_features(df)

    validate_feature_dataset(df)

    save_features(
        df=df,
        output_path=output_path,
    )

    return df


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