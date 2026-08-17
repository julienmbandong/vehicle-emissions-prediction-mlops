"""
Entraînement du modèle final de classification
==============================================

Ce script industrialise l'entraînement du modèle de classification
sélectionné dans le notebook 05.

Le modèle retenu est un RandomForestClassifier.

Les hyperparamètres, les classes ADEME et les seuils de discrétisation
sont chargés depuis :

models/metadata/random_forest_classification_metadata.json

Deux modes d'exécution sont disponibles :

--test
    Entraînement sur un sous-échantillon stratifié du Train FULL.

--full
    Entraînement sur l'intégralité du Train FULL.

Le script :

1. charge X_train, X_test, y_train et y_test ;
2. charge les métadonnées de classification ;
3. reconstruit y_train_class et y_test_class ;
4. entraîne le Random Forest ;
5. évalue le modèle ;
6. sauvegarde le modèle ;
7. sauvegarde les métriques et métadonnées d'entraînement ;
8. vérifie que le modèle sauvegardé est rechargeable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


# =====================================================================
# Configuration
# =====================================================================

TEST_SAMPLE_SIZE = 100_000
RANDOM_STATE = 42


# =====================================================================
# Chargement des datasets prétraités
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
    Charge les datasets prétraités produits par le pipeline précédent.
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

    required_files = {
        "X_train_processed": x_train_path,
        "X_test_processed": x_test_path,
        "y_train": y_train_path,
        "y_test": y_test_path,
    }

    missing_files = [
        name
        for name, path in required_files.items()
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
    Vérifie la cohérence structurelle des datasets.
    """

    checks = {
        "Cohérence X_train / y_train":
            len(X_train) == len(y_train),

        "Cohérence X_test / y_test":
            len(X_test) == len(y_test),

        "Colonnes Train / Test identiques":
            X_train.columns.tolist()
            == X_test.columns.tolist(),

        "Aucun NaN dans X_train":
            not X_train.isna().any().any(),

        "Aucun NaN dans X_test":
            not X_test.isna().any().any(),

        "Aucun NaN dans y_train":
            not y_train.isna().any(),

        "Aucun NaN dans y_test":
            not y_test.isna().any(),

        "Aucune valeur infinie dans X_train":
            not np.isinf(
                X_train.to_numpy()
            ).any(),

        "Aucune valeur infinie dans X_test":
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
        "✅ Datasets prêts pour la classification."
    )


# =====================================================================
# Chargement des métadonnées du notebook 05
# =====================================================================

def load_classification_metadata(
    metadata_path: Path,
) -> dict:
    """
    Charge les métadonnées produites par le notebook 05.
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
        != "RandomForestClassifier"
    ):
        raise ValueError(
            "Le modèle sélectionné dans les métadonnées "
            "n'est pas RandomForestClassifier."
        )

    if not metadata.get(
        "best_random_forest_params"
    ):
        raise ValueError(
            "Les hyperparamètres Random Forest "
            "sont absents des métadonnées."
        )

    if not metadata.get(
        "classification_bins"
    ):
        raise ValueError(
            "Les seuils de classification "
            "sont absents des métadonnées."
        )

    if not metadata.get(
        "classes"
    ):
        raise ValueError(
            "Les classes sont absentes "
            "des métadonnées."
        )

    print(
        "\n✅ Métadonnées du notebook 05 chargées."
    )

    return metadata


# =====================================================================
# Reconstruction des seuils de classification
# =====================================================================

def restore_classification_bins(
    metadata: dict,
) -> list[float]:
    """
    Reconvertit les bornes sérialisées du JSON en valeurs numériques.
    """

    restored_bins = []

    for value in metadata[
        "classification_bins"
    ]:

        if value == "-inf":
            restored_bins.append(
                -float("inf")
            )

        elif value == "inf":
            restored_bins.append(
                float("inf")
            )

        else:
            restored_bins.append(
                float(value)
            )

    return restored_bins


# =====================================================================
# Création des cibles de classification
# =====================================================================

def build_classification_targets(
    y_train: pd.Series,
    y_test: pd.Series,
    metadata: dict,
) -> tuple[pd.Series, pd.Series]:
    """
    Construit y_train_class et y_test_class à partir des règles ADEME.
    """

    classification_bins = (
        restore_classification_bins(
            metadata
        )
    )

    class_labels = metadata[
        "classes"
    ]

    y_train_class = pd.cut(
        y_train,
        bins=classification_bins,
        labels=class_labels,
        include_lowest=True,
    )

    y_test_class = pd.cut(
        y_test,
        bins=classification_bins,
        labels=class_labels,
        include_lowest=True,
    )

    if y_train_class.isna().any():
        raise ValueError(
            "Certaines observations Train "
            "n'ont pas été classées."
        )

    if y_test_class.isna().any():
        raise ValueError(
            "Certaines observations Test "
            "n'ont pas été classées."
        )

    print(
        "\n✅ Cibles de classification reconstruites."
    )

    print(
        f"Classes : "
        f"{', '.join(class_labels)}"
    )

    return (
        y_train_class,
        y_test_class,
    )


# =====================================================================
# Construction des hyperparamètres du modèle
# =====================================================================

def build_model_params(
    metadata: dict,
) -> dict:
    """
    Construit les paramètres du Random Forest final.
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
    y_train_class: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    """
    Extrait un sous-échantillon stratifié du Train FULL.
    """

    if TEST_SAMPLE_SIZE > len(X_train):
        raise ValueError(
            "TEST_SAMPLE_SIZE est supérieur "
            "au nombre d'observations disponibles."
        )

    (
        X_train_sample,
        _,
        y_train_class_sample,
        _,
    ) = train_test_split(
        X_train,
        y_train_class,
        train_size=TEST_SAMPLE_SIZE,
        stratify=y_train_class,
        random_state=RANDOM_STATE,
    )

    return (
        X_train_sample,
        y_train_class_sample,
    )


# =====================================================================
# Entraînement
# =====================================================================

def train_model(
    X_train: pd.DataFrame,
    y_train_class: pd.Series,
    model_params: dict,
) -> tuple[
    RandomForestClassifier,
    float,
]:
    """
    Entraîne le RandomForestClassifier.
    """

    model = RandomForestClassifier(
        **model_params
    )

    print(
        "\nDébut de l'entraînement..."
    )

    start_time = time.perf_counter()

    model.fit(
        X_train,
        y_train_class,
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
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train_class: pd.Series,
    X_test: pd.DataFrame,
    y_test_class: pd.Series,
) -> dict[str, float]:
    """
    Calcule les métriques de classification sur Train et Test.
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
        "accuracy_train":
            float(
                accuracy_score(
                    y_train_class,
                    y_train_pred,
                )
            ),

        "balanced_accuracy_train":
            float(
                balanced_accuracy_score(
                    y_train_class,
                    y_train_pred,
                )
            ),

        "f1_macro_train":
            float(
                f1_score(
                    y_train_class,
                    y_train_pred,
                    average="macro",
                )
            ),

        "f1_weighted_train":
            float(
                f1_score(
                    y_train_class,
                    y_train_pred,
                    average="weighted",
                )
            ),

        "accuracy_test":
            float(
                accuracy_score(
                    y_test_class,
                    y_test_pred,
                )
            ),

        "balanced_accuracy_test":
            float(
                balanced_accuracy_score(
                    y_test_class,
                    y_test_pred,
                )
            ),

        "f1_macro_test":
            float(
                f1_score(
                    y_test_class,
                    y_test_pred,
                    average="macro",
                )
            ),

        "f1_weighted_test":
            float(
                f1_score(
                    y_test_class,
                    y_test_pred,
                    average="weighted",
                )
            ),
    }

    print(
        "\nPERFORMANCES"
    )

    print(
        "=" * 90
    )

    print(
        "\nTrain"
    )

    print(
        f"  Accuracy          : "
        f"{metrics['accuracy_train']:.4f}"
    )

    print(
        f"  Balanced Accuracy : "
        f"{metrics['balanced_accuracy_train']:.4f}"
    )

    print(
        f"  F1-macro          : "
        f"{metrics['f1_macro_train']:.4f}"
    )

    print(
        f"  F1-weighted       : "
        f"{metrics['f1_weighted_train']:.4f}"
    )

    print(
        "\nTest"
    )

    print(
        f"  Accuracy          : "
        f"{metrics['accuracy_test']:.4f}"
    )

    print(
        f"  Balanced Accuracy : "
        f"{metrics['balanced_accuracy_test']:.4f}"
    )

    print(
        f"  F1-macro          : "
        f"{metrics['f1_macro_test']:.4f}"
    )

    print(
        f"  F1-weighted       : "
        f"{metrics['f1_weighted_test']:.4f}"
    )

    return metrics


# =====================================================================
# Sauvegarde du modèle
# =====================================================================

def save_model(
    model: RandomForestClassifier,
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
            "random_forest_classifier"
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
    classification_metadata: dict,
    output_dir: Path,
    suffix: str,
) -> Path:
    """
    Sauvegarde les informations de l'entraînement exécuté.
    """

    training_metadata = {
        "mode": mode,

        "problem_type":
            "multiclass_classification",

        "model":
            "RandomForestClassifier",

        "target_original":
            classification_metadata[
                "target_original"
            ],

        "target_classification":
            classification_metadata[
                "target_classification"
            ],

        "classes":
            classification_metadata[
                "classes"
            ],

        "class_intervals_g_co2_km":
            classification_metadata[
                "class_intervals_g_co2_km"
            ],

        "class_business_labels":
            classification_metadata[
                "class_business_labels"
            ],

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
            "random_forest_classifier"
            f"_training_metadata{suffix}.json"
        )
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            training_metadata,
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
    Vérifie que le modèle sauvegardé est rechargeable.
    """

    reloaded_model = joblib.load(
        model_path
    )

    if not isinstance(
        reloaded_model,
        RandomForestClassifier,
    ):
        raise TypeError(
            "Le modèle rechargé n'est pas "
            "un RandomForestClassifier."
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
    classification_metadata_path: Path,
    model_output_dir: Path,
) -> None:
    """
    Exécute le pipeline complet d'entraînement classification.
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
    # Chargement des métadonnées
    # -----------------------------------------------------------------

    metadata = load_classification_metadata(
        classification_metadata_path
    )


    # -----------------------------------------------------------------
    # Reconstruction des cibles de classification
    # -----------------------------------------------------------------

    (
        y_train_class,
        y_test_class,
    ) = build_classification_targets(
        y_train,
        y_test,
        metadata,
    )


    # -----------------------------------------------------------------
    # Hyperparamètres
    # -----------------------------------------------------------------

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
            y_train_class,
        )

        suffix = "_test"

    else:

        X_train_used = X_train
        y_train_used = y_train_class

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

    print(
        f"Classes                 : "
        f"{', '.join(metadata['classes'])}"
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
        y_test_class,
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
        classification_metadata=metadata,
        output_dir=model_output_dir,
        suffix=suffix,
    )


    # -----------------------------------------------------------------
    # Validation du modèle sauvegardé
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
        f"✅ Modèle      : {model_path}"
    )

    print(
        f"✅ Métadonnées : {metadata_path}"
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
            "de classification des émissions CO₂."
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
            "sous-échantillon stratifié du Train."
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

    classification_metadata_path = (
        project_root
        / "models"
        / "metadata"
        / "random_forest_classification_metadata.json"
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
        f"{classification_metadata_path}"
    )

    print(
        f"Sortie modèle    : "
        f"{model_output_dir}"
    )


    run_training(
        mode=mode,
        processed_data_dir=processed_data_dir,
        classification_metadata_path=(
            classification_metadata_path
        ),
        model_output_dir=model_output_dir,
    )


if __name__ == "__main__":
    main()