"""
Entraînement du modèle de régression sélectionné
================================================

Industrialisation de la sélection réalisée dans le notebook :

notebooks/04_model_training/
04_regression_model_training_and_evaluation.ipynb

Le notebook réalise :
1. la comparaison des modèles ;
2. l'optimisation des hyperparamètres ;
3. la sélection du modèle candidat ;
4. l'enregistrement des hyperparamètres et métadonnées.

Ce script NE RELANCE PAS le tuning.

Il :
1. charge les datasets prétraités ;
2. valide leur cohérence ;
3. charge la configuration sélectionnée dans les métadonnées ;
4. construit le RandomForestRegressor avec les hyperparamètres retenus ;
5. entraîne le modèle sur le Train FULL ;
6. évalue le modèle sur le Test final ;
7. sauvegarde le modèle entraîné ;
8. sauvegarde les métadonnées d'entraînement ;
9. recharge les artefacts afin de vérifier leur cohérence.

Modes
-----
--test
    Lit uniquement les 100 000 premières observations des artefacts FULL.
    Ce mode permet de valider rapidement le pipeline technique sans charger
    plusieurs millions de lignes en mémoire.

--full
    Charge les datasets complets et entraîne le modèle destiné à
    l'industrialisation.

Important
---------
Le mode TEST ne modifie pas les artefacts FULL :
les sorties sont suffixées par "_test".
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)


# =====================================================================
# Configuration générale
# =====================================================================

TARGET_COLUMN = "co2_wltp_g_km"

RANDOM_STATE = 42

# Nombre maximal de lignes chargées par dataset en mode TEST.
NROWS_TEST = 100_000


# ---------------------------------------------------------------------
# Modèle attendu par le stage DVC actuel
# ---------------------------------------------------------------------

EXPECTED_MODEL_CLASS = "RandomForestRegressor"
EXPECTED_MODEL_NAME = "random_forest_regressor"


# ---------------------------------------------------------------------
# Noms des artefacts
# ---------------------------------------------------------------------

MODEL_FILENAME = "random_forest_regressor.joblib"

TRAINING_METADATA_FILENAME = (
    "random_forest_regressor_training_metadata.json"
)

SELECTION_METADATA_FILENAME = (
    "random_forest_regression_metadata.json"
)


# =====================================================================
# Utilitaires généraux
# =====================================================================

def to_python_scalar(value: Any) -> Any:
    """
    Convertit un éventuel scalaire NumPy en type Python natif.

    Cette conversion est utile avant sérialisation JSON.
    """

    return (
        value.item()
        if hasattr(value, "item")
        else value
    )


def load_json(
    path: Path,
) -> dict[str, Any]:
    """
    Charge un fichier JSON.

    Vérifie également que le contenu racine est bien un dictionnaire.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier JSON introuvable : {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        content = json.load(file)

    if not isinstance(content, dict):
        raise ValueError(
            "Le fichier JSON ne contient pas "
            f"un objet dictionnaire valide : {path}"
        )

    return content


def save_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    """
    Sauvegarde un dictionnaire au format JSON lisible.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )


# =====================================================================
# Racine du projet
# =====================================================================

def get_project_root() -> Path:
    """
    Détermine la racine du projet.

    Fichier attendu :

    src/
      vehicle_emissions/
        training/
          train_regression_model.py

    parents[3] correspond donc à la racine du projet.
    """

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    pyproject_path = (
        project_root
        / "pyproject.toml"
    )

    if not pyproject_path.is_file():
        raise FileNotFoundError(
            "Impossible de déterminer la racine du projet : "
            f"pyproject.toml absent de {project_root}."
        )

    return project_root


# =====================================================================
# Construction des chemins
# =====================================================================

def build_paths(
    project_root: Path,
    *,
    test_mode: bool,
) -> dict[str, Path]:
    """
    Construit les chemins d'entrée et de sortie.

    Les entrées TEST et FULL utilisent les mêmes artefacts prétraités FULL.

    En mode TEST, seule une partie des lignes est chargée.
    Les sorties TEST sont en revanche suffixées afin de ne jamais écraser
    les artefacts de production.
    """

    processed_dir = (
        project_root
        / "data"
        / "processed"
    )

    metadata_dir = (
        project_root
        / "models"
        / "metadata"
    )

    trained_dir = (
        project_root
        / "models"
        / "trained"
    )

    output_suffix = (
        "_test"
        if test_mode
        else ""
    )

    model_filename = (
        f"random_forest_regressor{output_suffix}.joblib"
        if test_mode
        else MODEL_FILENAME
    )

    training_metadata_filename = (
        "random_forest_regressor_training_metadata"
        f"{output_suffix}.json"
        if test_mode
        else TRAINING_METADATA_FILENAME
    )

    return {

        # -------------------------------------------------------------
        # Datasets
        # -------------------------------------------------------------

        "x_train": (
            processed_dir
            / "X_train_processed.parquet"
        ),

        "x_test": (
            processed_dir
            / "X_test_processed.parquet"
        ),

        "y_train": (
            processed_dir
            / "y_train.parquet"
        ),

        "y_test": (
            processed_dir
            / "y_test.parquet"
        ),

        # -------------------------------------------------------------
        # Métadonnées issues du notebook 04
        # -------------------------------------------------------------

        "selection_metadata": (
            metadata_dir
            / SELECTION_METADATA_FILENAME
        ),

        # -------------------------------------------------------------
        # Artefacts produits par ce script
        # -------------------------------------------------------------

        "model": (
            trained_dir
            / model_filename
        ),

        "training_metadata": (
            trained_dir
            / training_metadata_filename
        ),
    }


# =====================================================================
# Lecture Parquet
# =====================================================================

def read_parquet(
    path: Path,
    *,
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Charge un fichier Parquet.

    Mode FULL
    ---------
    pandas.read_parquet() charge le fichier complet.

    Mode TEST
    ---------
    PyArrow lit uniquement le premier batch demandé.

    Cela évite de charger plusieurs millions de lignes en mémoire
    uniquement pour vérifier le fonctionnement technique du pipeline.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier Parquet introuvable : {path}"
        )

    # -----------------------------------------------------------------
    # Mode FULL
    # -----------------------------------------------------------------

    if nrows is None:
        return pd.read_parquet(path)

    # -----------------------------------------------------------------
    # Mode TEST
    # -----------------------------------------------------------------

    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)

    batches = parquet_file.iter_batches(
        batch_size=nrows,
    )

    try:
        batch = next(batches)

    except StopIteration as exc:
        raise ValueError(
            f"Le fichier Parquet est vide : {path}"
        ) from exc

    return batch.to_pandas()


# =====================================================================
# Chargement de la cible
# =====================================================================

def load_target(
    path: Path,
    *,
    nrows: int | None = None,
) -> pd.Series:
    """
    Charge un fichier cible Parquet et retourne une Series pandas.
    """

    target_df = read_parquet(
        path,
        nrows=nrows,
    )

    if TARGET_COLUMN not in target_df.columns:
        raise ValueError(
            f"La cible '{TARGET_COLUMN}' "
            f"est absente de {path.name}."
        )

    if target_df.shape[1] != 1:
        raise ValueError(
            f"Le fichier {path.name} doit contenir "
            f"uniquement la cible '{TARGET_COLUMN}'."
        )

    return target_df[TARGET_COLUMN]


# =====================================================================
# Chargement des datasets
# =====================================================================

def load_datasets(
    paths: dict[str, Path],
    *,
    nrows: int | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Charge les jeux prétraités produits par l'étape de preprocessing.
    """

    print(
        "=" * 72
    )

    print(
        "CHARGEMENT DES DONNÉES PRÉTRAITÉES"
    )

    print(
        "=" * 72
    )

    X_train = read_parquet(
        paths["x_train"],
        nrows=nrows,
    )

    X_test = read_parquet(
        paths["x_test"],
        nrows=nrows,
    )

    y_train = load_target(
        paths["y_train"],
        nrows=nrows,
    )

    y_test = load_target(
        paths["y_test"],
        nrows=nrows,
    )

    print(
        f"\nTrain : "
        f"{X_train.shape[0]:,} observations × "
        f"{X_train.shape[1]} variables"
    )

    print(
        f"Test  : "
        f"{X_test.shape[0]:,} observations × "
        f"{X_test.shape[1]} variables"
    )

    print(
        f"Cible Train : {len(y_train):,} valeurs"
    )

    print(
        f"Cible Test  : {len(y_test):,} valeurs"
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
    Vérifie que les datasets sont prêts pour scikit-learn.
    """

    checks = {

        "Cohérence X_train / y_train":
            len(X_train) == len(y_train),

        "Cohérence X_test / y_test":
            len(X_test) == len(y_test),

        "Colonnes Train / Test identiques":
            X_train.columns.tolist()
            == X_test.columns.tolist(),

        "Cible absente de X_train":
            TARGET_COLUMN
            not in X_train.columns,

        "Cible absente de X_test":
            TARGET_COLUMN
            not in X_test.columns,

        "Aucun NaN X_train":
            not X_train
            .isna()
            .any()
            .any(),

        "Aucun NaN X_test":
            not X_test
            .isna()
            .any()
            .any(),

        "Aucun NaN y_train":
            not y_train
            .isna()
            .any(),

        "Aucun NaN y_test":
            not y_test
            .isna()
            .any(),

        "X_train entièrement numérique":
            X_train
            .select_dtypes(
                exclude="number"
            )
            .shape[1]
            == 0,

        "X_test entièrement numérique":
            X_test
            .select_dtypes(
                exclude="number"
            )
            .shape[1]
            == 0,

        "manufacturer_make exclue":
            all(
                "manufacturer_make"
                not in column.lower()

                for column
                in X_train.columns
            ),
    }

    failed_checks = [
        name
        for name, result
        in checks.items()
        if not result
    ]

    print(
        "\n"
        + "=" * 72
    )

    print(
        "VALIDATION DES DATASETS"
    )

    print(
        "=" * 72
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

    if failed_checks:
        raise ValueError(
            "Validation des datasets échouée : "
            + ", ".join(failed_checks)
        )

    print(
        "\n✅ Les datasets sont prêts "
        "pour l'entraînement."
    )


# =====================================================================
# Chargement de la configuration sélectionnée dans le notebook
# =====================================================================

def load_selected_model_configuration(
    metadata_path: Path,
    feature_names: list[str],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    """
    Charge les hyperparamètres sélectionnés dans le notebook 04.

    Le script ne refait donc ni :
    - comparaison de modèles ;
    - RandomizedSearchCV ;
    - sélection des hyperparamètres.

    Il applique la décision expérimentale déjà validée.
    """

    metadata = load_json(
        metadata_path
    )

    selected_model = metadata.get(
        "selected_model"
    )

    if not isinstance(
        selected_model,
        dict,
    ):
        raise ValueError(
            "La section 'selected_model' "
            "est absente des métadonnées."
        )

    model_class = selected_model.get(
        "model_class"
    )

    model_name = selected_model.get(
        "model_name"
    )

    best_params = selected_model.get(
        "best_params"
    )

    # -----------------------------------------------------------------
    # Le dvc.yaml actuel attend spécifiquement Random Forest
    # -----------------------------------------------------------------

    if model_class != EXPECTED_MODEL_CLASS:
        raise ValueError(
            "Le stage de régression actuel attend "
            f"'{EXPECTED_MODEL_CLASS}', "
            "mais les métadonnées sélectionnent "
            f"'{model_class}'."
        )

    if model_name != EXPECTED_MODEL_NAME:
        raise ValueError(
            "Nom de modèle inattendu dans les métadonnées : "
            f"{model_name!r}."
        )

    if (
        not isinstance(
            best_params,
            dict,
        )
        or not best_params
    ):
        raise ValueError(
            "Aucun hyperparamètre sélectionné "
            "n'est disponible."
        )

    # -----------------------------------------------------------------
    # Contrôle strict de la structure des features
    # -----------------------------------------------------------------

    expected_features = metadata.get(
        "features"
    )

    if expected_features != feature_names:
        raise ValueError(
            "La structure des variables du dataset "
            "ne correspond pas à celle enregistrée "
            "dans les métadonnées du notebook."
        )

    excluded_features = metadata.get(
        "excluded_features",
        [],
    )

    if "manufacturer_make" not in excluded_features:
        raise ValueError(
            "Les métadonnées ne confirment pas "
            "l'exclusion de manufacturer_make."
        )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "CONFIGURATION DU MODÈLE SÉLECTIONNÉ"
    )

    print(
        "=" * 72
    )

    print(
        f"\nClasse du modèle : {model_class}"
    )

    print(
        f"Nom du modèle    : {model_name}"
    )

    print(
        f"Nombre de variables : {len(feature_names)}"
    )

    print(
        "\nHyperparamètres sélectionnés :"
    )

    for parameter, value in best_params.items():
        print(
            f"  - {parameter}: {value}"
        )

    return (
        metadata,
        best_params,
    )


# =====================================================================
# Construction du modèle
# =====================================================================

def build_model(
    best_params: dict[str, Any],
) -> RandomForestRegressor:
    """
    Construit le RandomForestRegressor final.

    Les hyperparamètres métier/modèle proviennent du tuning du notebook.

    random_state et n_jobs sont ajoutés ici comme paramètres
    d'exécution et de reproductibilité.
    """

    params = {
        key: to_python_scalar(value)

        for key, value
        in best_params.items()
    }

    params["random_state"] = (
        RANDOM_STATE
    )

    params["n_jobs"] = -1

    model = RandomForestRegressor(
        **params
    )

    return model


# =====================================================================
# Évaluation
# =====================================================================

def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Calcule les métriques de régression avec les fonctions standards
    fournies par scikit-learn.
    """

    y_pred = model.predict(
        X_test
    )

    metrics = {

        "mae":
            float(
                mean_absolute_error(
                    y_test,
                    y_pred,
                )
            ),

        "rmse":
            float(
                root_mean_squared_error(
                    y_test,
                    y_pred,
                )
            ),

        "r2":
            float(
                r2_score(
                    y_test,
                    y_pred,
                )
            ),
    }

    return metrics


# =====================================================================
# Entraînement et évaluation
# =====================================================================

def train_and_evaluate(
    model: RandomForestRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[
    RandomForestRegressor,
    dict[str, float],
    float,
    float,
]:
    """
    Entraîne le modèle sur Train puis l'évalue sur Test.

    En mode FULL :
    - Train = Train FULL ;
    - Test = Test final.

    En mode TEST :
    - seule une fraction des mêmes artefacts est utilisée.
    """

    print(
        "\n"
        + "=" * 72
    )

    print(
        "ENTRAÎNEMENT DU MODÈLE DE RÉGRESSION"
    )

    print(
        "=" * 72
    )

    print(
        f"\nTrain : "
        f"{len(X_train):,} observations × "
        f"{X_train.shape[1]} variables"
    )

    print(
        f"Test  : "
        f"{len(X_test):,} observations"
    )

    # -----------------------------------------------------------------
    # Entraînement
    # -----------------------------------------------------------------

    training_start_time = (
        time.perf_counter()
    )

    model.fit(
        X_train,
        y_train,
    )

    training_time_seconds = (
        time.perf_counter()
        - training_start_time
    )

    # -----------------------------------------------------------------
    # Évaluation
    # -----------------------------------------------------------------

    evaluation_start_time = (
        time.perf_counter()
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    evaluation_time_seconds = (
        time.perf_counter()
        - evaluation_start_time
    )

    # -----------------------------------------------------------------
    # Rapport
    # -----------------------------------------------------------------

    print(
        "\n"
        + "=" * 72
    )

    print(
        "RÉSULTATS DE L'ÉVALUATION"
    )

    print(
        "=" * 72
    )

    print(
        f"\nMAE  : "
        f"{metrics['mae']:.4f} g CO₂/km"
    )

    print(
        f"RMSE : "
        f"{metrics['rmse']:.4f} g CO₂/km"
    )

    print(
        f"R²   : "
        f"{metrics['r2']:.4f}"
    )

    print(
        f"\nTemps d'entraînement : "
        f"{training_time_seconds:.2f} secondes"
    )

    print(
        f"Temps d'évaluation   : "
        f"{evaluation_time_seconds:.2f} secondes"
    )

    return (
        model,
        metrics,
        training_time_seconds,
        evaluation_time_seconds,
    )


# =====================================================================
# Sauvegarde des artefacts
# =====================================================================

def save_training_artifacts(
    *,
    model: RandomForestRegressor,
    selection_metadata: dict[str, Any],
    metrics: dict[str, float],
    training_time_seconds: float,
    evaluation_time_seconds: float,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    paths: dict[str, Path],
    mode: str,
) -> dict[str, Any]:
    """
    Sauvegarde :
    - le modèle Random Forest entraîné ;
    - les métadonnées de l'entraînement.
    """

    paths["model"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Sauvegarde du modèle
    # -----------------------------------------------------------------

    joblib.dump(
        model,
        paths["model"],
    )

    # -----------------------------------------------------------------
    # Construction des métadonnées
    # -----------------------------------------------------------------

    model_params = {
        key: to_python_scalar(value)

        for key, value
        in model
        .get_params(
            deep=False
        )
        .items()
    }

    training_metadata = {

        "mode":
            mode,

        "target":
            TARGET_COLUMN,

        "model": {

            "name":
                EXPECTED_MODEL_NAME,

            "class":
                EXPECTED_MODEL_CLASS,

            "artifact":
                paths["model"].name,

            "hyperparameters":
                model_params,
        },

        "data": {

            "train_observations":
                int(
                    len(X_train)
                ),

            "test_observations":
                int(
                    len(X_test)
                ),

            "feature_count":
                int(
                    X_train.shape[1]
                ),

            "features":
                X_train.columns.tolist(),

            "excluded_features": [
                "manufacturer_make"
            ],
        },

        "evaluation": {

            "dataset":
                (
                    "test_final"
                    if mode == "FULL"
                    else "test_sample"
                ),

            "mae":
                float(
                    metrics["mae"]
                ),

            "rmse":
                float(
                    metrics["rmse"]
                ),

            "r2":
                float(
                    metrics["r2"]
                ),
        },

        "timing": {

            "training_seconds":
                float(
                    training_time_seconds
                ),

            "evaluation_seconds":
                float(
                    evaluation_time_seconds
                ),
        },

        "selection_reference": {

            "metadata_file":
                paths[
                    "selection_metadata"
                ].name,

            "notebook_selected_model":
                selection_metadata.get(
                    "selected_model"
                ),

            "notebook_validation_metrics":
                selection_metadata.get(
                    "validation_metrics"
                ),

            "notebook_final_test_metrics":
                selection_metadata.get(
                    "final_test"
                ),
        },
    }

    # -----------------------------------------------------------------
    # Sauvegarde des métadonnées
    # -----------------------------------------------------------------

    save_json(
        training_metadata,
        paths[
            "training_metadata"
        ],
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "SAUVEGARDE DES ARTEFACTS"
    )

    print(
        "=" * 72
    )

    print(
        f"\nModèle : "
        f"{paths['model']}"
    )

    print(
        f"Métadonnées : "
        f"{paths['training_metadata']}"
    )

    print(
        "\n✅ Modèle sauvegardé."
    )

    print(
        "✅ Métadonnées d'entraînement sauvegardées."
    )

    return training_metadata


# =====================================================================
# Validation du rechargement des artefacts
# =====================================================================

def validate_saved_artifacts(
    *,
    paths: dict[str, Path],
    X_reference: pd.DataFrame,
    expected_metrics: dict[str, float],
) -> None:
    """
    Recharge le modèle et les métadonnées.

    Cette étape détecte immédiatement un artefact corrompu,
    incomplet ou incohérent avant son utilisation en inférence.
    """

    # -----------------------------------------------------------------
    # Existence
    # -----------------------------------------------------------------

    if not paths["model"].is_file():
        raise FileNotFoundError(
            "Modèle non sauvegardé : "
            f"{paths['model']}"
        )

    if not paths[
        "training_metadata"
    ].is_file():
        raise FileNotFoundError(
            "Métadonnées non sauvegardées : "
            f"{paths['training_metadata']}"
        )

    # -----------------------------------------------------------------
    # Rechargement
    # -----------------------------------------------------------------

    loaded_model = joblib.load(
        paths["model"]
    )

    loaded_metadata = load_json(
        paths[
            "training_metadata"
        ]
    )

    # -----------------------------------------------------------------
    # Contrôles
    # -----------------------------------------------------------------

    checks = {

        "Classe du modèle conforme":
            isinstance(
                loaded_model,
                RandomForestRegressor,
            ),

        "Modèle entraîné":
            hasattr(
                loaded_model,
                "estimators_",
            ),

        "Nombre de variables conforme":
            getattr(
                loaded_model,
                "n_features_in_",
                None,
            )
            == X_reference.shape[1],

        "Ordre des variables conforme":
            loaded_metadata
            .get(
                "data",
                {},
            )
            .get(
                "features"
            )
            == X_reference
            .columns
            .tolist(),

        "manufacturer_make exclue":
            all(
                "manufacturer_make"
                not in column.lower()

                for column
                in loaded_metadata
                .get(
                    "data",
                    {},
                )
                .get(
                    "features",
                    [],
                )
            ),

        "MAE sauvegardée":
            loaded_metadata
            .get(
                "evaluation",
                {},
            )
            .get(
                "mae"
            )
            == expected_metrics[
                "mae"
            ],

        "RMSE sauvegardée":
            loaded_metadata
            .get(
                "evaluation",
                {},
            )
            .get(
                "rmse"
            )
            == expected_metrics[
                "rmse"
            ],

        "R² sauvegardé":
            loaded_metadata
            .get(
                "evaluation",
                {},
            )
            .get(
                "r2"
            )
            == expected_metrics[
                "r2"
            ],
    }

    failed_checks = [
        name

        for name, result
        in checks.items()

        if not result
    ]

    print(
        "\n"
        + "=" * 72
    )

    print(
        "VALIDATION DU RECHARGEMENT DES ARTEFACTS"
    )

    print(
        "=" * 72
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

    if failed_checks:
        raise ValueError(
            "Validation des artefacts échouée : "
            + ", ".join(
                failed_checks
            )
        )

    print(
        "\n✅ Le modèle est rechargeable."
    )

    print(
        "✅ Les métadonnées sont cohérentes."
    )

    print(
        "✅ L'ordre des variables est conservé."
    )


# =====================================================================
# Pipeline principal
# =====================================================================

def run_training(
    *,
    project_root: Path,
    test_mode: bool,
) -> None:
    """
    Exécute le pipeline complet d'entraînement de régression.
    """

    mode = (
        "TEST"
        if test_mode
        else "FULL"
    )

    paths = build_paths(
        project_root,
        test_mode=test_mode,
    )

    # -----------------------------------------------------------------
    # Informations d'exécution
    # -----------------------------------------------------------------

    print(
        "=" * 72
    )

    print(
        "PIPELINE D'ENTRAÎNEMENT DU MODÈLE DE RÉGRESSION"
    )

    print(
        "=" * 72
    )

    print(
        f"\nMode               : {mode}"
    )

    print(
        f"Racine du projet   : {project_root}"
    )

    print(
        "Métadonnées source : "
        f"{paths['selection_metadata']}"
    )

    # -----------------------------------------------------------------
    # Limitation mémoire en mode TEST
    # -----------------------------------------------------------------

    nrows = (
        NROWS_TEST
        if test_mode
        else None
    )

    if test_mode:
        print(
            "\nMode TEST : "
            f"{NROWS_TEST:,} premières observations "
            "seront chargées par dataset."
        )

    # -----------------------------------------------------------------
    # Chargement
    # -----------------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = load_datasets(
        paths,
        nrows=nrows,
    )

    # -----------------------------------------------------------------
    # Validation des datasets
    # -----------------------------------------------------------------

    validate_datasets(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # -----------------------------------------------------------------
    # Chargement de la sélection effectuée dans le notebook
    # -----------------------------------------------------------------

    (
        selection_metadata,
        best_params,
    ) = load_selected_model_configuration(
        paths[
            "selection_metadata"
        ],
        X_train.columns.tolist(),
    )

    # -----------------------------------------------------------------
    # Construction du modèle
    # -----------------------------------------------------------------

    model = build_model(
        best_params
    )

    # -----------------------------------------------------------------
    # Entraînement + évaluation
    # -----------------------------------------------------------------

    (
        model,
        metrics,
        training_time_seconds,
        evaluation_time_seconds,
    ) = train_and_evaluate(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # -----------------------------------------------------------------
    # Sauvegarde
    # -----------------------------------------------------------------

    save_training_artifacts(
        model=model,
        selection_metadata=selection_metadata,
        metrics=metrics,
        training_time_seconds=training_time_seconds,
        evaluation_time_seconds=evaluation_time_seconds,
        X_train=X_train,
        X_test=X_test,
        paths=paths,
        mode=mode,
    )

    # -----------------------------------------------------------------
    # Rechargement et validation
    # -----------------------------------------------------------------

    validate_saved_artifacts(
        paths=paths,
        X_reference=X_train,
        expected_metrics=metrics,
    )

    # -----------------------------------------------------------------
    # Fin
    # -----------------------------------------------------------------

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PIPELINE TERMINÉ"
    )

    print(
        "=" * 72
    )

    print(
        "\n✅ Entraînement du modèle de régression "
        "terminé avec succès."
    )

    print(
        f"✅ Modèle : {paths['model']}"
    )

    print(
        "✅ Métadonnées : "
        f"{paths['training_metadata']}"
    )


# =====================================================================
# Interface CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    """
    Définit les modes d'exécution TEST et FULL.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Entraîne le modèle de régression sélectionné "
            "dans le notebook 04."
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
            "Valide le pipeline sur les "
            f"{NROWS_TEST:,} premières observations "
            "des artefacts prétraités FULL."
        ),
    )

    mode_group.add_argument(
        "--full",
        action="store_true",
        help=(
            "Entraîne le modèle sélectionné sur le "
            "Train FULL puis l'évalue sur le Test final."
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
        get_project_root()
    )

    run_training(
        project_root=project_root,
        test_mode=args.test,
    )


if __name__ == "__main__":
    main()