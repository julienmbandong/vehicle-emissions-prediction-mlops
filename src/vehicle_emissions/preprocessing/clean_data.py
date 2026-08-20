from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TARGET_COLUMN = "co2_wltp_g_km"
DATE_COLUMN = "registration_date"

MANUFACTURER_RESOLUTION_MIN_SHARE = 0.95

CATEGORICAL_CODE_NORMALIZATION = {
    "vehicle_category_type": "upper",
    "fuel_mode": "upper",
}

MANUFACTURER_MAKE_ALIASES = {
    "VOLKSWAGEN VW": "VOLKSWAGEN",
}

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
    "manufacturer_name_oem",
    "manufacturer_make",
    "vehicle_category_type",
    "fuel_type",
    "fuel_mode",
    "mass_running_order_kg",
    "co2_wltp_g_km",
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

    df = df.copy()
    df.columns = df.columns.str.strip()

    return df.rename(columns=COLUMN_MAPPING)


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

    df = df.copy()

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce",
    )

    return df


def normalize_text_values(
    series: pd.Series,
) -> pd.Series:
    """
    Applique la normalisation technique générale d'une variable
    catégorielle en préservant les valeurs manquantes.
    """

    result = series.copy()
    mask = result.notna()

    result.loc[mask] = (
        result.loc[mask]
        .astype(str)
        .str.strip()
    )

    return result


def normalize_categorical_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalise les variables catégorielles selon deux niveaux :

    1. nettoyage technique général de toutes les variables catégorielles ;
    2. règles métier spécifiques aux codes dont la casse n'est pas
       porteuse d'information.
    """

    df = df.copy()

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for column in categorical_columns:
        df[column] = normalize_text_values(df[column])

    for column, normalization in CATEGORICAL_CODE_NORMALIZATION.items():
        if column not in df.columns:
            continue

        mask = df[column].notna()

        if normalization == "upper":
            df.loc[mask, column] = (
                df.loc[mask, column]
                .astype(str)
                .str.upper()
            )
        else:
            raise ValueError(
                f"Règle de normalisation inconnue pour '{column}' : "
                f"{normalization}"
            )

    print("✅ Normalisation des modalités catégorielles appliquée.")

    return df


# ---------------------------------------------------------------------
# Normalisation dédiée de manufacturer_make
# ---------------------------------------------------------------------


def build_manufacturer_comparison_key(
    series: pd.Series,
) -> pd.Series:
    """
    Construit une clé technique de comparaison permettant d'identifier
    les variantes typographiques d'une même marque.
    """

    result = (
        series
        .astype("string")
        .str.strip()
        .str.upper()
    )

    result = result.str.replace(
        r"[.\-_/]+",
        " ",
        regex=True,
    )

    result = result.str.replace(
        r"\s+",
        " ",
        regex=True,
    )

    return result.str.strip()


def normalize_manufacturer_make(
    df: pd.DataFrame,
    min_resolution_share: float = MANUFACTURER_RESOLUTION_MIN_SHARE,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """
    Normalise manufacturer_make selon une stratégie conservative.

    - regroupe les variantes typographiques équivalentes ;
    - retient la variante observée la plus fréquente comme canonique ;
    - résout les valeurs purement numériques à partir du couple
      manufacturer_name_eu / manufacturer_name_oem lorsque la marque
      dominante atteint le seuil de fiabilité ;
    - convertit les cas numériques non résolus en valeur manquante.
    """

    required_columns = {
        "manufacturer_make",
        "manufacturer_name_eu",
        "manufacturer_name_oem",
    }

    missing_columns = sorted(
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Colonnes indispensables absentes pour la normalisation "
            "de manufacturer_make : "
            + ", ".join(missing_columns)
        )

    if not 0 < min_resolution_share <= 1:
        raise ValueError(
            "min_resolution_share doit être compris "
            "dans l'intervalle ]0, 1]."
        )

    df = df.copy()

    manufacturer = (
        df["manufacturer_make"]
        .astype("string")
        .str.strip()
    )

    numeric_mask = manufacturer.str.fullmatch(
        r"\d+",
        na=False,
    )

    textual_mask = manufacturer.notna() & ~numeric_mask
    numeric_before = int(numeric_mask.sum())

    # -----------------------------------------------------------------
    # 1. Regroupement typographique des valeurs textuelles
    # -----------------------------------------------------------------

    comparison_key = build_manufacturer_comparison_key(
        manufacturer
    )

    textual_reference = pd.DataFrame(
        {
            "manufacturer_make": manufacturer[textual_mask],
            "comparison_key": comparison_key[textual_mask],
        }
    )

    variant_counts = (
        textual_reference
        .value_counts(
            ["comparison_key", "manufacturer_make"]
        )
        .rename("observations")
        .reset_index()
    )

    canonical_variants = (
        variant_counts
        .sort_values(
            by=[
                "comparison_key",
                "observations",
                "manufacturer_make",
            ],
            ascending=[True, False, True],
        )
        .drop_duplicates(
            subset="comparison_key",
            keep="first",
        )
        .set_index("comparison_key")["manufacturer_make"]
    )

    df.loc[textual_mask, "manufacturer_make"] = (
        comparison_key[textual_mask]
        .map(canonical_variants)
        .astype("string")
    )

    # -----------------------------------------------------------------
    # 2. Référentiel fiable (Mh, Man) -> manufacturer_make
    # -----------------------------------------------------------------

    textual_rows = df.loc[
        textual_mask,
        [
            "manufacturer_make",
            "manufacturer_name_eu",
            "manufacturer_name_oem",
        ],
    ].copy()

    for column in [
        "manufacturer_name_eu",
        "manufacturer_name_oem",
    ]:
        textual_rows[column] = (
            textual_rows[column]
            .astype("string")
            .str.strip()
            .str.upper()
        )

    valid_reference_mask = (
        textual_rows["manufacturer_name_eu"].notna()
        & textual_rows["manufacturer_name_oem"].notna()
    )

    reference_counts = (
        textual_rows.loc[valid_reference_mask]
        .value_counts(
            [
                "manufacturer_name_eu",
                "manufacturer_name_oem",
                "manufacturer_make",
            ]
        )
        .rename("observations")
        .reset_index()
    )

    pair_totals = (
        reference_counts
        .groupby(
            [
                "manufacturer_name_eu",
                "manufacturer_name_oem",
            ],
            as_index=False,
        )["observations"]
        .sum()
        .rename(
            columns={"observations": "pair_observations"}
        )
    )

    reference_counts = reference_counts.merge(
        pair_totals,
        on=[
            "manufacturer_name_eu",
            "manufacturer_name_oem",
        ],
        how="left",
    )

    reference_counts["share"] = (
        reference_counts["observations"]
        / reference_counts["pair_observations"]
    )

    dominant_reference = (
        reference_counts
        .sort_values(
            by=[
                "manufacturer_name_eu",
                "manufacturer_name_oem",
                "observations",
            ],
            ascending=[True, True, False],
        )
        .drop_duplicates(
            subset=[
                "manufacturer_name_eu",
                "manufacturer_name_oem",
            ],
            keep="first",
        )
    )

    reliable_reference = dominant_reference.loc[
        dominant_reference["share"] >= min_resolution_share
    ].copy()

    resolution_map = {
        (
            row["manufacturer_name_eu"],
            row["manufacturer_name_oem"],
        ): row["manufacturer_make"]
        for _, row in reliable_reference.iterrows()
    }

    # -----------------------------------------------------------------
    # 3. Résolution des valeurs numériques
    # -----------------------------------------------------------------

    numeric_rows = df.loc[
        numeric_mask,
        [
            "manufacturer_name_eu",
            "manufacturer_name_oem",
        ],
    ].copy()

    for column in [
        "manufacturer_name_eu",
        "manufacturer_name_oem",
    ]:
        numeric_rows[column] = (
            numeric_rows[column]
            .astype("string")
            .str.strip()
            .str.upper()
        )

    resolved_values = pd.Series(
        pd.NA,
        index=numeric_rows.index,
        dtype="string",
    )

    # Le volume des valeurs numériques est très faible par rapport au dataset
    # complet ; cette boucle ne porte donc que sur les anomalies à résoudre.
    for index, row in numeric_rows.iterrows():
        manufacturer_pair = (
            row["manufacturer_name_eu"],
            row["manufacturer_name_oem"],
        )

        resolved_values.loc[index] = resolution_map.get(
            manufacturer_pair,
            pd.NA,
        )

    numeric_resolved = int(resolved_values.notna().sum())
    numeric_unresolved = numeric_before - numeric_resolved

    df.loc[
        resolved_values.index,
        "manufacturer_make",
    ] = resolved_values

    final_manufacturer = (
        df["manufacturer_make"]
        .astype("string")
    )

    remaining_numeric = int(
        final_manufacturer
        .str.fullmatch(r"\d+", na=False)
        .sum()
    )

    if remaining_numeric != 0:
        raise ValueError(
            "Des valeurs numériques subsistent dans "
            "manufacturer_make après normalisation."
        )

    report: dict[str, int | float] = {
        "numeric_before": numeric_before,
        "numeric_resolved": numeric_resolved,
        "numeric_unresolved": numeric_unresolved,
        "typographic_keys": int(len(canonical_variants)),
        "reliable_manufacturer_pairs": int(len(reliable_reference)),
        "resolution_threshold": min_resolution_share,
    }

    print("✅ Normalisation de manufacturer_make appliquée.")
    print(
        "  - valeurs numériques avant traitement : "
        f"{numeric_before:,}"
    )
    print(
        "  - valeurs numériques résolues : "
        f"{numeric_resolved:,}"
    )
    print(
        "  - valeurs numériques non résolues -> NA : "
        f"{numeric_unresolved:,}"
    )
    print(
        "  - seuil de résolution : "
        f"{min_resolution_share:.0%}"
    )

    return df, report


# ---------------------------------------------------------------------
# Normalisation métier des alias de manufacturer_make
# ---------------------------------------------------------------------


def normalize_manufacturer_aliases(
    df: pd.DataFrame,
    aliases: dict[str, str] = MANUFACTURER_MAKE_ALIASES,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Remplace les variantes métier explicitement validées par leur
    marque canonique, indépendamment de la casse et des espaces
    périphériques.
    """

    if "manufacturer_make" not in df.columns:
        raise KeyError(
            "La colonne 'manufacturer_make' est absente."
        )

    df = df.copy()

    manufacturer_key = (
        df["manufacturer_make"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    normalized_aliases = {
        str(alias).strip().upper(): canonical
        for alias, canonical in aliases.items()
    }

    replacement_counts: dict[str, int] = {}

    for alias, canonical in aliases.items():
        alias_key = str(alias).strip().upper()

        mask = manufacturer_key.eq(
            alias_key
        ).fillna(False)

        replacement_counts[alias] = int(
            mask.sum()
        )

        if mask.any():
            df.loc[
                mask,
                "manufacturer_make",
            ] = normalized_aliases[alias_key]

    total_replaced = int(
        sum(replacement_counts.values())
    )

    print(
        "✅ Normalisation métier de manufacturer_make appliquée."
    )
    print(
        f"  - observations remplacées : {total_replaced:,}"
    )

    return df, replacement_counts


# ---------------------------------------------------------------------
# Variables non informatives
# ---------------------------------------------------------------------


def remove_constant_columns(
    df: pd.DataFrame,
    drop: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Détecte les variables constantes et supprime uniquement celles
    qui ne sont pas indispensables au contrat du dataset nettoyé.
    """

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    if not constant_columns:
        print("Aucune variable constante détectée.")
        return df, constant_columns

    # Les variables nécessaires au contrat du dataset nettoyé
    # ne doivent jamais être supprimées automatiquement.
    protected_columns = REQUIRED_COLUMNS | {
        TARGET_COLUMN,
    }

    protected_constant_columns = [
        column
        for column in constant_columns
        if column in protected_columns
    ]

    removable_constant_columns = [
        column
        for column in constant_columns
        if column not in protected_columns
    ]

    print(
        f"Variables constantes détectées : "
        f"{len(constant_columns)}"
    )

    for column in constant_columns:
        print(f"  - {column}")

    if protected_constant_columns:
        print(
            "ℹ️ Variables constantes protégées "
            "car indispensables :"
        )

        for column in protected_constant_columns:
            print(f"  - {column}")

    if drop and removable_constant_columns:
        df = df.drop(
            columns=removable_constant_columns
        )

        print(
            "✅ Variables constantes non indispensables supprimées."
        )

    elif drop:
        print(
            "ℹ️ Aucune variable constante supprimable après "
            "protection des variables indispensables."
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

    manufacturer_numeric = (
        df["manufacturer_make"]
        .dropna()
        .astype("string")
        .str.strip()
        .str.fullmatch(
            r"\d+",
            na=False,
        )
    )

    if manufacturer_numeric.any():
        raise ValueError(
            "manufacturer_make contient encore "
            "des valeurs numériques."
        )

    for column in CATEGORICAL_CODE_NORMALIZATION:
        if column not in df.columns:
            continue

        values = (
            df[column]
            .dropna()
            .astype("string")
        )

        if not values.eq(
            values.str.upper()
        ).all():
            raise ValueError(
                f"La colonne '{column}' contient encore "
                "des modalités dont la casse "
                "n'est pas normalisée."
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

    # Normalisation générale et codes catégoriels
    df = normalize_categorical_values(df)

    # Normalisation dédiée de la marque constructeur
    df, _ = normalize_manufacturer_make(df)
    df, _ = normalize_manufacturer_aliases(df)

    # Suppression des variables constantes non indispensables
    df, _ = remove_constant_columns(
        df,
        drop=drop_constant_columns,
    )

    # Suppression des observations sans variable cible
    df = remove_missing_target(df)

    # Contrôles qualité finaux
    validate_cleaned_data(df)

    # Export du dataset nettoyé
    save_data(
        df=df,
        output_path=output_path,
    )

    return df


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


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

    project_root = Path(
        __file__
    ).resolve().parents[3]

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