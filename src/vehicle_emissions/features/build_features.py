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

REQUIRED_COLUMNS = {
    "vehicle_record_id",
    "country",
    "manufacturer_name_eu",
    "manufacturer_make",
    "vehicle_category_type",
    "fuel_type",
    "fuel_mode",
    "mass_running_order_kg",
    "engine_capacity_cm3",
    "engine_power_kw",
    "registration_date",
    TARGET_COLUMN,
}

COLUMNS_TO_EXCLUDE = {
    "vehicle_record_id",
    "registration_date",
    "vehicle_category",
}

TEMPORAL_FEATURES = {
    "registration_month_sin",
    "registration_month_cos",
}


# ---------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------

def load_data(
    input_path: Path,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Charge le dataset nettoyé utilisé pour le Feature Engineering."""

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
    """Vérifie la présence des variables indispensables."""

    missing_columns = sorted(
        REQUIRED_COLUMNS - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Variables attendues absentes : "
            + ", ".join(missing_columns)
        )

    if not pd.api.types.is_datetime64_any_dtype(
        df[DATE_COLUMN]
    ):
        raise TypeError(
            f"La colonne '{DATE_COLUMN}' "
            "doit être au format datetime."
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

    Le mois brut est utilisé uniquement comme variable intermédiaire
    et n'est pas conservé dans le dataset final.
    """

    registration_month = df[DATE_COLUMN].dt.month

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
# Sélection des variables
# ---------------------------------------------------------------------

def select_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Exclut les variables qui ne doivent pas alimenter
    le futur pipeline de modélisation.
    """

    columns_present_to_exclude = sorted(
        COLUMNS_TO_EXCLUDE.intersection(df.columns)
    )

    df.drop(
        columns=columns_present_to_exclude,
        inplace=True,
    )

    print("Variables exclues :")

    for column in columns_present_to_exclude:
        print(f"  - {column}")

    return df


# ---------------------------------------------------------------------
# Contrôles qualité
# ---------------------------------------------------------------------

def validate_feature_dataset(
    df: pd.DataFrame,
) -> None:
    """Vérifie la cohérence du dataset après Feature Engineering."""

    quality_checks = {
        "Variable cible présente":
            TARGET_COLUMN in df.columns,

        "Noms de colonnes uniques":
            df.columns.is_unique,

        "Identifiant technique supprimé":
            "vehicle_record_id"
            not in df.columns,

        "Date brute supprimée":
            DATE_COLUMN
            not in df.columns,

        "Variable constante vehicle_category supprimée":
            "vehicle_category"
            not in df.columns,

        "Feature mois sinus présente":
            "registration_month_sin"
            in df.columns,

        "Feature mois cosinus présente":
            "registration_month_cos"
            in df.columns,
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

    temporal_missing = (
        df[
            [
                "registration_month_sin",
                "registration_month_cos",
            ]
        ]
        .isna()
        .sum()
        .sum()
    )

    if temporal_missing != 0:
        raise ValueError(
            "Les features temporelles contiennent "
            "des valeurs manquantes."
        )

    print("✅ Dataset de features validé.")


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------

def save_features(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Exporte le dataset de Feature Engineering."""

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
# Pipeline
# ---------------------------------------------------------------------

def build_features(
    input_path: Path,
    output_path: Path,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Exécute le pipeline complet de Feature Engineering."""

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
    """Définit les arguments de ligne de commande."""

    parser = argparse.ArgumentParser(
        description=(
            "Feature Engineering du dataset "
            "Vehicle Emissions 2024."
        )
    )

    mode_group = parser.add_mutually_exclusive_group(
        required=True
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
    """Point d'entrée du pipeline de Feature Engineering."""

    args = parse_args()

    project_root = Path(__file__).resolve().parents[3]

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