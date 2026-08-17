"""
Entraînement du modèle final de régression
==========================================

Ce script industrialise l'entraînement du modèle de régression sélectionné
dans le notebook :

notebooks/04_model_training/
04_regression_model_training_and_evaluation.ipynb

Le modèle retenu est un RandomForestRegressor.

Les hyperparamètres sont chargés depuis :

models/metadata/random_forest_regression_metadata.json

Deux modes d'exécution sont disponibles :

--test
    Entraînement sur un sous-échantillon du Train FULL.

--full
    Entraînement sur l'intégralité du Train FULL.

Le script :

1. charge X_train, y_train, X_test et y_test ;
2. charge les hyperparamètres sélectionnés lors du tuning ;
3. entraîne le Random Forest ;
4. évalue le modèle ;
5. sauvegarde le modèle entraîné ;
6. sauvegarde les métriques et métadonnées de l'entraînement ;
7. vérifie que le modèle sauvegardé est rechargeable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =====================================================================
# Configuration
# =====================================================================

TEST_SAMPLE_SIZE = 100_000
RANDOM_STATE = 42

MODEL_FILENAME = "random_forest_regressor.joblib"

TRAINING_METADATA_FILENAME = (
    "random_forest_regressor_training_metadata.json"
)


# =====================================================================
# Chargement des données
# =====================================================================

def load_processed_datasets(
    processed_data_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Charge les datasets prétraités produits par le notebook/script 03.
    """

    x_train_path = (
        processed_data_dir
        / "X_train_processed.parquet"
    )

    x_test_path = (
        processed_data_dir
        / "X_test_processed.parquet"
    )

    y_train_path = (
        processed_data_dir
        / "y_train.parquet"
    )

    y_test_path = (
        processed_data_dir
        / "y_test.parquet"
    )

    required_files = [
        x_train_path,
        x_test_path,
        y_train_path,
        y_test_path,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Fichiers prétraités manquants : "
            + ", ".join(missing_files)
        )

    print("Chargement des datasets prétraités...")

    X_train = pd.read_parquet(
        x_train_path
    )

    X_test = pd.read_parquet(
        x_test_path
    )

    y_train = (
        pd.read_parquet(y_train_path)
        .squeeze("columns")
    )

    y_test = (
        pd.read_parquet(y_test_path)
        .squeeze("columns")
    )

    print(
        f"X_train : "
        f"{X_train.shape[0]:,} × "
        f"{X_train.shape[1]}"
    )

    print(
        f"X_test  : "
        f"{X_test.shape[0]:,} × "
        f"{X_test.shape[1]}"
    )

    print(
        f"y_train : "
        f"{len(y_train):,}"
    )

    print(
        f"y_test  : "
        f"{len(y_test):,}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# =====================================================================
# Validation des datasets
# =====================================================================

def validate_datasets(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    Vérifie la cohérence minimale des datasets.
    """

    checks = {
        "Cohérence X_train / y_train":
            len(X_train) == len(y_train),

        "Cohérence X_test / y_test":
            len(X_test) == len(y_test),

        "Colonnes Train / Test identiques":
            X_train.columns.tolist()
            == X_test.columns.tolist(),

        "Aucun NaN Train":
            not X_train.isna().any().any(),

        "Aucun NaN Test":
            not X_test.isna().any().any(),

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

    print("\nValidation des datasets :")

    for name, result in checks.items():

        status = "✅" if result else "❌"

        print(
            f"{status} {name}"
        )

        if not result:
            failed_checks.append(name)

    if failed_checks:
        raise ValueError(
            "Validation des datasets échouée : "
            + ", ".join(failed_checks)
        )

    print(
        "✅ Datasets prêts pour l'entraînement."
    )


# =====================================================================
# Chargement des métadonnées du notebook 04
# =====================================================================

def load_regression_metadata(
    metadata_path: Path,
) -> dict:
    """
    Charge les résultats persistés du notebook 04.
    """

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Métadonnées introuvables : "
            f"{metadata_path}"
        )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    if (
        metadata.get("model_selected")
        != "RandomForestRegressor"
    ):
        raise ValueError(
            "Le modèle sélectionné dans les métadonnées "
            "n'est pas RandomForestRegressor."
        )

    best_params = metadata.get(
        "best_random_forest_params"
    )

    if not best_params:
        raise ValueError(
            "Les hyperparamètres Random Forest "
            "sont absents des métadonnées."
        )

    print(
        "\n✅ Métadonnées du notebook 04 chargées."
    )

    return metadata


# =====================================================================
# Préparation des hyperparamètres
# =====================================================================

def build_model_params(
    metadata: dict,
) -> dict:
    """
    Construit les paramètres du modèle final à partir du tuning.
    """

    model_params = (
        metadata[
            "best_random_forest_params"
        ]
        .copy()
    )

    model_params.update(
        {
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
    )

    print(
        "\nHyperparamètres du modèle :"
    )

    for parameter, value in (
        model_params.items()
    ):

        print(
            f"  - {parameter}: {value}"
        )

    return model_params


# =====================================================================
# Préparation du mode TEST
# =====================================================================

def build_test_sample(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extrait un sous-échantillon reproductible du Train FULL.
    """

    if TEST_SAMPLE_SIZE > len(X_train):
        raise ValueError(
            "TEST_SAMPLE_SIZE est supérieur "
            "au Train disponible."
        )

    sampled_indices = (
        X_train
        .sample(
            n=TEST_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )
        .index
    )

    X_train_sample = (
        X_train.loc[
            sampled_indices
        ]
        .copy()
    )

    y_train_sample = (
        y_train.loc[
            sampled_indices
        ]
        .copy()
    )

    return (
        X_train_sample,
        y_train_sample,
    )


# =====================================================================
# Entraînement
# =====================================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: dict,
) -> tuple[
    RandomForestRegressor,
    float,
]:
    """
    Entraîne le Random Forest final.
    """

    model = RandomForestRegressor(
        **model_params
    )

    print(
        "\nDébut de l'entraînement..."
    )

    start_time = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    training_time_seconds = (
        time.perf_counter()
        - start_time
    )

    print(
        "✅ Entraînement terminé."
    )

    print(
        f"Temps d'entraînement : "
        f"{training_time_seconds:.2f} secondes"
    )

    return (
        model,
        training_time_seconds,
    )


# =====================================================================
# Évaluation
# =====================================================================

def evaluate_model(
    model: RandomForestRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Calcule MAE, RMSE et R² sur Train et Test.
    """

    print(
        "\nCalcul des prédictions..."
    )

    y_train_pred = model.predict(
        X_train
    )

    y_test_pred = model.predict(
        X_test
    )

    metrics = {
        "mae_train":
            float(
                mean_absolute_error(
                    y_train,
                    y_train_pred,
                )
            ),

        "rmse_train":
            float(
                np.sqrt(
                    mean_squared_error(
                        y_train,
                        y_train_pred,
                    )
                )
            ),

        "r2_train":
            float(
                r2_score(
                    y_train,
                    y_train_pred,
                )
            ),

        "mae_test":
            float(
                mean_absolute_error(
                    y_test,
                    y_test_pred,
                )
            ),

        "rmse_test":
            float(
                np.sqrt(
                    mean_squared_error(
                        y_test,
                        y_test_pred,
                    )
                )
            ),

        "r2_test":
            float(
                r2_score(
                    y_test,
                    y_test_pred,
                )
            ),
    }

    print(
        "\nPERFORMANCES"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTrain"
    )

    print(
        f"  MAE  : "
        f"{metrics['mae_train']:.4f} g CO₂/km"
    )

    print(
        f"  RMSE : "
        f"{metrics['rmse_train']:.4f} g CO₂/km"
    )

    print(
        f"  R²   : "
        f"{metrics['r2_train']:.4f}"
    )

    print(
        "\nTest"
    )

    print(
        f"  MAE  : "
        f"{metrics['mae_test']:.4f} g CO₂/km"
    )

    print(
        f"  RMSE : "
        f"{metrics['rmse_test']:.4f} g CO₂/km"
    )

    print(
        f"  R²   : "
        f"{metrics['r2_test']:.4f}"
    )

    return metrics


# =====================================================================
# Sauvegarde du modèle
# =====================================================================

def save_model(
    model: RandomForestRegressor,
    output_dir: Path,
    suffix: str,
) -> Path:
    """
    Sauvegarde le modèle Random Forest.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        output_dir
        / (
            "random_forest_regressor"
            f"{suffix}.joblib"
        )
    )

    joblib.dump(
        model,
        model_path,
        compress=3,
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            "Le modèle n'a pas été sauvegardé."
        )

    print(
        f"\n✅ Modèle sauvegardé : "
        f"{model_path}"
    )

    return model_path


# =====================================================================
# Sauvegarde des métadonnées d'entraînement
# =====================================================================

def save_training_metadata(
    *,
    mode: str,
    model_path: Path,
    model_params: dict,
    metrics: dict[str, float],
    training_time_seconds: float,
    train_observations: int,
    test_observations: int,
    feature_count: int,
    output_dir: Path,
    suffix: str,
) -> Path:
    """
    Sauvegarde les informations de l'entraînement exécuté.
    """

    metadata = {
        "mode": mode,

        "model": "RandomForestRegressor",

        "target": "co2_wltp_g_km",

        "model_file":
            model_path.name,

        "train_observations":
            int(train_observations),

        "test_observations":
            int(test_observations),

        "features":
            int(feature_count),

        "training_time_seconds":
            float(training_time_seconds),

        "hyperparameters":
            model_params,

        "metrics":
            metrics,
    }

    metadata_path = (
        output_dir
        / (
            "random_forest_regressor"
            f"_training_metadata{suffix}.json"
        )
    )

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

    print(
        f"✅ Métadonnées sauvegardées : "
        f"{metadata_path}"
    )

    return metadata_path


# =====================================================================
# Validation du modèle sauvegardé
# =====================================================================

def validate_saved_model(
    model_path: Path,
    expected_features: int,
) -> None:
    """
    Vérifie que le modèle est rechargeable.
    """

    reloaded_model = joblib.load(
        model_path
    )

    if not isinstance(
        reloaded_model,
        RandomForestRegressor,
    ):
        raise TypeError(
            "Le modèle rechargé n'est pas "
            "un RandomForestRegressor."
        )

    if (
        hasattr(
            reloaded_model,
            "n_features_in_",
        )
        and
        reloaded_model.n_features_in_
        != expected_features
    ):
        raise ValueError(
            "Le nombre de variables du modèle "
            "rechargé est incohérent."
        )

    print(
        "✅ Modèle rechargé et validé."
    )


# =====================================================================
# Pipeline principal
# =====================================================================

def run_training(
    *,
    mode: str,
    processed_data_dir: Path,
    regression_metadata_path: Path,
    model_output_dir: Path,
) -> None:
    """
    Exécute le pipeline complet d'entraînement.
    """

    # -----------------------------------------------------------------
    # Chargement des données
    # -----------------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = load_processed_datasets(
        processed_data_dir
    )

    validate_datasets(
        X_train,
        X_test,
        y_train,
        y_test,
    )


    # -----------------------------------------------------------------
    # Chargement des hyperparamètres
    # -----------------------------------------------------------------

    metadata = load_regression_metadata(
        regression_metadata_path
    )

    model_params = build_model_params(
        metadata
    )


    # -----------------------------------------------------------------
    # Mode TEST / FULL
    # -----------------------------------------------------------------

    if mode == "TEST":

        (
            X_train_used,
            y_train_used,
        ) = build_test_sample(
            X_train,
            y_train,
        )

        suffix = "_test"

    else:

        X_train_used = X_train
        y_train_used = y_train

        suffix = ""


    print(
        f"\nMode d'exécution        : {mode}"
    )

    print(
        f"Observations Train      : "
        f"{len(X_train_used):,}"
    )

    print(
        f"Observations Test       : "
        f"{len(X_test):,}"
    )

    print(
        f"Nombre de variables     : "
        f"{X_train_used.shape[1]}"
    )


    # -----------------------------------------------------------------
    # Entraînement
    # -----------------------------------------------------------------

    (
        model,
        training_time_seconds,
    ) = train_model(
        X_train_used,
        y_train_used,
        model_params,
    )


    # -----------------------------------------------------------------
    # Évaluation
    # -----------------------------------------------------------------

    metrics = evaluate_model(
        model,
        X_train_used,
        y_train_used,
        X_test,
        y_test,
    )


    # -----------------------------------------------------------------
    # Sauvegarde
    # -----------------------------------------------------------------

    model_path = save_model(
        model,
        model_output_dir,
        suffix,
    )

    metadata_path = save_training_metadata(
        mode=mode,
        model_path=model_path,
        model_params=model_params,
        metrics=metrics,
        training_time_seconds=training_time_seconds,
        train_observations=len(
            X_train_used
        ),
        test_observations=len(
            X_test
        ),
        feature_count=X_train_used.shape[1],
        output_dir=model_output_dir,
        suffix=suffix,
    )


    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    validate_saved_model(
        model_path=model_path,
        expected_features=(
            X_train_used.shape[1]
        ),
    )


    print(
        "\n✅ Pipeline d'entraînement "
        f"{mode} terminé avec succès."
    )

    print(
        f"✅ Modèle       : {model_path}"
    )

    print(
        f"✅ Métadonnées  : {metadata_path}"
    )


# =====================================================================
# Arguments CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    """
    Définit les modes TEST et FULL.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Entraînement du modèle final "
            "de régression CO₂."
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
            "Entraîne le modèle sur un "
            "sous-échantillon du Train."
        ),
    )

    mode_group.add_argument(
        "--full",
        action="store_true",
        help=(
            "Entraîne le modèle sur "
            "l'intégralité du Train."
        ),
    )

    return parser.parse_args()


# =====================================================================
# Point d'entrée
# =====================================================================

def main() -> None:
    """
    Point d'entrée principal.
    """

    args = parse_args()

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    processed_data_dir = (
        project_root
        / "data"
        / "processed"
    )

    regression_metadata_path = (
        project_root
        / "models"
        / "metadata"
        / "random_forest_regression_metadata.json"
    )

    if args.test:

        mode = "TEST"

        model_output_dir = (
            project_root
            / "models"
            / "trained"
            / "test"
        )

    else:

        mode = "FULL"

        model_output_dir = (
            project_root
            / "models"
            / "trained"
        )


    print(
        f"Mode d'exécution : {mode}"
    )

    print(
        f"Données          : "
        f"{processed_data_dir}"
    )

    print(
        f"Métadonnées      : "
        f"{regression_metadata_path}"
    )

    print(
        f"Sortie modèle    : "
        f"{model_output_dir}"
    )


    run_training(
        mode=mode,
        processed_data_dir=processed_data_dir,
        regression_metadata_path=regression_metadata_path,
        model_output_dir=model_output_dir,
    )


if __name__ == "__main__":
    main()