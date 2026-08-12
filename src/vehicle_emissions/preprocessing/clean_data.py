from pathlib import Path

import pandas as pd
import argparse


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TARGET_COLUMN = "co2_wltp_g_km"
DATE_COLUMN = "registration_date"

COLUMN_MAPPING = {
    "ID": "vehicle_record_id",
    "Country": "country",
    "VFN": "vehicle_family_id",
    "Mp": "manufacturer_pool",
    "Mh": "manufacturer_name_eu",
    "Man": "manufacturer_name_oem",
    "Tan": "type_approval_number",
    "T": "vehicle_type",
    "Va": "vehicle_variant",
    "Ve": "vehicle_version",
    "Mk": "manufacturer_make",
    "Cn": "commercial_name",
    "Ct": "vehicle_category_type",
    "Cr": "vehicle_category",
    "r": "new_registrations",
    "m (kg)": "mass_running_order_kg",
    "Mt": "wltp_test_mass_kg",
    "Ewltp (g/km)": "co2_wltp_g_km",
    "Ft": "fuel_type",
    "Fm": "fuel_mode",
    "ec (cm3)": "engine_capacity_cm3",
    "ep (KW)": "engine_power_kw",
    "z (Wh/km)": "electric_energy_consumption_wh_km",
    "IT": "innovative_technology",
    "Erwltp (g/km)": "co2_reduction_wltp_g_km",
    "Status": "status",
    "year": "registration_year",
    "Date of registration": "registration_date",
    "Fuel consumption": "fuel_consumption",
    "ech": "emission_standard",
    "RLFI": "rlfi",
    "Electric range (km)": "electric_range_km",
}

REQUIRED_COLUMNS = {
    "vehicle_record_id",
    "country",
    "manufacturer_name_eu",
    "mass_running_order_kg",
    "co2_wltp_g_km",
    "fuel_type",
    "engine_capacity_cm3",
    "engine_power_kw",
    "registration_date",
}


# ---------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------

def load_data(
    input_path: Path,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Charge le dataset brut depuis un fichier CSV."""

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Fichier source introuvable : {input_path}"
        )

    print(f"Chargement : {input_path}")

    df = pd.read_csv(
        input_path,
        sep=",",
        nrows=nrows,
        low_memory=False,
    )

    print(
        f"Dataset chargé : "
        f"{len(df):,} observations × {df.shape[1]} variables"
    )

    return df


# ---------------------------------------------------------------------
# Nettoyage du schéma
# ---------------------------------------------------------------------

def remove_fully_empty_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Supprime les colonnes entièrement vides."""

    empty_columns = (
        df.columns[df.isna().all()].tolist()
    )

    if empty_columns:
        print(
            f"Colonnes entièrement vides supprimées : "
            f"{len(empty_columns)}"
        )

        df = df.drop(columns=empty_columns)

    return df


def normalize_column_names(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Normalise puis standardise les noms de colonnes."""

    df.columns = df.columns.str.strip()

    df = df.rename(
        columns=COLUMN_MAPPING
    )

    return df


# ---------------------------------------------------------------------
# Types et modalités
# ---------------------------------------------------------------------

def convert_data_types(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Convertit les variables nécessitant un type spécifique."""

    if DATE_COLUMN not in df.columns:
        raise KeyError(
            f"Colonne obligatoire absente : {DATE_COLUMN}"
        )

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce",
    )

    return df


def normalize_categorical_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Supprime les espaces parasites des variables catégorielles."""

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for column in categorical_columns:
        mask = df[column].notna()

        df.loc[mask, column] = (
            df.loc[mask, column]
            .astype(str)
            .str.strip()
        )

    return df


# ---------------------------------------------------------------------
# Variables non informatives
# ---------------------------------------------------------------------

def remove_constant_columns(
    df: pd.DataFrame,
    drop: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Détecte les variables constantes et les supprime si demandé."""

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    if not constant_columns:
        print("Aucune variable constante détectée.")
        return df, constant_columns

    print(
        f"Variables constantes détectées : "
        f"{len(constant_columns)}"
    )

    for column in constant_columns:
        print(f"  - {column}")

    if drop:
        df = df.drop(
            columns=constant_columns
        )

        print(
            "✅ Variables constantes supprimées."
        )
    else:
        print(
            "ℹ️ Variables constantes conservées "
            "pour cette exécution."
        )

    return df, constant_columns

# ---------------------------------------------------------------------
# Variable cible
# ---------------------------------------------------------------------

def remove_missing_target(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Supprime les observations dont la cible est absente."""

    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Variable cible absente : {TARGET_COLUMN}"
        )

    rows_before = len(df)

    df = df.dropna(
        subset=[TARGET_COLUMN]
    )

    rows_removed = rows_before - len(df)

    print(
        f"Observations sans cible supprimées : "
        f"{rows_removed:,}"
    )

    return df


# ---------------------------------------------------------------------
# Contrôles qualité
# ---------------------------------------------------------------------

def validate_cleaned_data(
    df: pd.DataFrame,
) -> None:
    """Vérifie les contraintes essentielles du dataset nettoyé."""

    missing_required_columns = sorted(
        REQUIRED_COLUMNS - set(df.columns)
    )

    if missing_required_columns:
        raise ValueError(
            "Variables indispensables manquantes : "
            + ", ".join(missing_required_columns)
        )

    if df["vehicle_record_id"].isna().any():
        raise ValueError(
            "Des identifiants techniques sont manquants."
        )

    if not df["vehicle_record_id"].is_unique:
        raise ValueError(
            "L'identifiant technique n'est pas unique."
        )

    if df.duplicated().any():
        raise ValueError(
            "Le dataset contient des doublons complets."
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            "La variable cible contient encore des valeurs manquantes."
        )

    if not pd.api.types.is_datetime64_any_dtype(
        df[DATE_COLUMN]
    ):
        raise TypeError(
            "registration_date n'est pas au format datetime."
        )

    print("✅ Contrôles qualité validés.")


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------

def save_data(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Exporte le dataset nettoyé au format CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"✅ Dataset exporté : {output_path}"
    )
    print(
        f"Observations : {len(df):,}"
    )
    print(
        f"Variables    : {df.shape[1]}"
    )


# ---------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------

def clean_data(
    input_path: Path,
    output_path: Path,
    nrows: int | None = None,
    drop_constant_columns: bool = True,
) -> pd.DataFrame:
    """Exécute le pipeline complet de nettoyage."""

    df = load_data(
        input_path=input_path,
        nrows=nrows,
    )

    df = remove_fully_empty_columns(df)
    df = normalize_column_names(df)
    df = convert_data_types(df)
    df = normalize_categorical_values(df)
    df, _ = remove_constant_columns(
        df,
        drop=drop_constant_columns,
    )
    df = remove_missing_target(df)

    validate_cleaned_data(df)

    save_data(
        df=df,
        output_path=output_path,
    )

    return df


def parse_args() -> argparse.Namespace:
    """Construit les arguments de ligne de commande."""

    parser = argparse.ArgumentParser(
        description=(
            "Nettoyage et prétraitement du dataset "
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
            "Exécute le pipeline sur 100 000 lignes "
            "et conserve les variables constantes."
        ),
    )

    mode_group.add_argument(
        "--full",
        action="store_true",
        help=(
            "Exécute le pipeline sur le dataset complet "
            "et supprime les variables constantes."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Point d'entrée principal du pipeline de nettoyage."""

    args = parse_args()

    project_root = Path(__file__).resolve().parents[3]

    input_path = (
        project_root
        / "data"
        / "raw"
        / "data_2024.csv"
    )

    if args.test:
        output_path = (
            project_root
            / "data"
            / "processed"
            / "data_2024_cleaned_test.csv"
        )

        nrows = 100_000
        drop_constants = False

        print("Mode d'exécution : TEST")

    else:
        output_path = (
            project_root
            / "data"
            / "processed"
            / "data_2024_cleaned.csv"
        )

        nrows = None
        drop_constants = True

        print("Mode d'exécution : COMPLET")

    print(f"Fichier source : {input_path}")
    print(f"Fichier cible  : {output_path}")

    clean_data(
        input_path=input_path,
        output_path=output_path,
        nrows=nrows,
        drop_constant_columns=drop_constants,
    )


if __name__ == "__main__":
    main()