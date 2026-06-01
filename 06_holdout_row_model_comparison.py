"""
Train six tactile regression models with one held-out row per force file.

Each force-labeled npz file is split in the same way:
    9 rows for training + 1 row for test

Examples:
    python3 06_holdout_row_model_comparison.py --k 3
    python3 06_holdout_row_model_comparison.py --k 3 --test-index 4
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR / "Code"
sys.path.insert(0, str(CODE_DIR))

from bayesian_linear_regression import (  # noqa: E402
    MODEL_NAME as BAYESIAN_NAME,
    build_model as build_bayesian,
    predict_with_uncertainty as predict_bayesian,
)
from force_common import (  # noqa: E402
    extract_top_k_sensors,
    load_npz_array,
    parse_force_from_filename,
)
from gp_regression_tactile import build_model as build_gp  # noqa: E402
from mlp_deep_ensemble import (  # noqa: E402
    MODEL_NAME as ENSEMBLE_NAME,
    build_model as build_ensemble,
    predict_with_uncertainty as predict_ensemble,
)
from random_forest_regression import (  # noqa: E402
    MODEL_NAME as FOREST_NAME,
    build_model as build_forest,
    predict_with_uncertainty as predict_forest,
)
from random_forest_tree_variance import (  # noqa: E402
    MODEL_NAME as FOREST_VARIANCE_NAME,
    build_model as build_forest_variance,
    predict_with_uncertainty as predict_forest_variance,
)
from single_mlp_regression import (  # noqa: E402
    MODEL_NAME as MLP_NAME,
    build_model as build_mlp,
    predict_with_uncertainty as predict_mlp,
)


warnings.filterwarnings("ignore")

PredictionFunction = Callable[[Any, np.ndarray], tuple[np.ndarray, np.ndarray]]
EXPERIMENT_MODEL_FILES = {
    BAYESIAN_NAME: "bayesian_linear_regression.joblib",
    FOREST_NAME: "random_forest_regression.joblib",
    FOREST_VARIANCE_NAME: "random_forest_regression_with_tree_prediction_variance.joblib",
    MLP_NAME: "single_mlp_regression.joblib",
    ENSEMBLE_NAME: "mlp_deep_ensemble.joblib",
}


def load_row_holdout_dataset(
    data_dir: Path,
    test_index: int,
    k: int,
    sort_top_k: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    npz_files = sorted(data_dir.glob("*.npz"), key=parse_force_from_filename)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in: {data_dir.resolve()}")

    x_train_rows: list[np.ndarray] = []
    y_train_rows: list[float] = []
    x_test_rows: list[np.ndarray] = []
    y_test_rows: list[float] = []
    test_meta: list[dict[str, Any]] = []

    for path in npz_files:
        raw = load_npz_array(path)
        if not 0 <= test_index < len(raw):
            raise ValueError(
                f"--test-index {test_index} is out of range for {path.name}: "
                f"file contains {len(raw)} rows"
            )

        force = parse_force_from_filename(path)
        features = extract_top_k_sensors(raw, k=k, sort_values=sort_top_k)
        train_mask = np.ones(len(raw), dtype=bool)
        train_mask[test_index] = False

        x_train_rows.extend(features[train_mask])
        y_train_rows.extend([force] * int(train_mask.sum()))
        x_test_rows.append(features[test_index])
        y_test_rows.append(force)
        test_meta.append(
            {
                "file": path.name,
                "row_in_file": test_index,
                "force_N": force,
                "raw_values": raw[test_index].tolist(),
                "top_k_features": features[test_index].tolist(),
            }
        )

    return (
        np.asarray(x_train_rows, dtype=float),
        np.asarray(y_train_rows, dtype=float),
        np.asarray(x_test_rows, dtype=float),
        np.asarray(y_test_rows, dtype=float),
        pd.DataFrame(test_meta),
    )


def train_and_predict_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    random_state: int,
) -> list[tuple[str, Any, np.ndarray, np.ndarray]]:
    model_specs: list[tuple[str, Callable[[int], Any], PredictionFunction]] = [
        (BAYESIAN_NAME, build_bayesian, predict_bayesian),
        (FOREST_NAME, build_forest, predict_forest),
        (FOREST_VARIANCE_NAME, build_forest_variance, predict_forest_variance),
        (MLP_NAME, build_mlp, predict_mlp),
        (ENSEMBLE_NAME, build_ensemble, predict_ensemble),
    ]
    predictions: list[tuple[str, Any, np.ndarray, np.ndarray]] = []
    for model_name, build_model, predict in model_specs:
        model = build_model(random_state)
        model.fit(x_train, y_train)
        mean, std = predict(model, x_test)
        predictions.append((model_name, model, mean, std))
    return predictions


def train_and_predict_gp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    random_state: int,
) -> tuple[str, Any, StandardScaler, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    model = build_gp(random_state)
    model.fit(x_train_scaled, y_train)
    mean, std = model.predict(x_test_scaled, return_std=True)
    return "GP Regression", model, scaler, mean, std


def save_experiment_models(
    experiment_dir: Path,
    baseline_models: list[tuple[str, Any, np.ndarray, np.ndarray]],
    gp_result: tuple[str, Any, StandardScaler, np.ndarray, np.ndarray],
    k: int,
    test_index: int,
    sort_top_k: bool,
) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    for model_name, model, _, _ in baseline_models:
        bundle = {
            "model": model,
            "model_name": model_name,
            "k": k,
            "sort_top_k": sort_top_k,
        }
        joblib.dump(bundle, experiment_dir / EXPERIMENT_MODEL_FILES[model_name])

    _, gp_model, gp_scaler, _, _ = gp_result
    joblib.dump(gp_model, experiment_dir / "gp_model.pkl")
    joblib.dump(gp_scaler, experiment_dir / "scaler.pkl")
    config = {"k": k, "sort_top_k": sort_top_k}
    (experiment_dir / "gp_config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "k": k,
        "sort_top_k": sort_top_k,
        "test_index": test_index,
        "training_rule": "For every force npz file, train on all rows except test_index.",
    }
    (experiment_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def evaluate_holdout(
    data_dir: Path,
    test_index: int,
    k: int,
    sort_top_k: bool = False,
    random_state: int = 42,
    experiment_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train, y_train, x_test, y_test, meta = load_row_holdout_dataset(
        data_dir=data_dir,
        test_index=test_index,
        k=k,
        sort_top_k=sort_top_k,
    )
    baseline_models = train_and_predict_baselines(x_train, y_train, x_test, random_state)
    gp_result = train_and_predict_gp(x_train, y_train, x_test, random_state)
    if experiment_dir is not None:
        save_experiment_models(
            experiment_dir=experiment_dir,
            baseline_models=baseline_models,
            gp_result=gp_result,
            k=k,
            test_index=test_index,
            sort_top_k=sort_top_k,
        )
    predictions = [
        (model_name, mean, std)
        for model_name, _, mean, std in baseline_models
    ]
    predictions.append((gp_result[0], gp_result[3], gp_result[4]))

    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for model_name, mean, std in predictions:
        finite_std = np.where(np.isfinite(std), std, np.nan)
        for sample_index, (true, predicted, uncertainty) in enumerate(zip(y_test, mean, finite_std)):
            prediction_rows.append(
                {
                    "k": k,
                    "test_index": test_index,
                    "sample_index": sample_index,
                    "model": model_name,
                    "force_N": float(true),
                    "y_pred": float(predicted),
                    "uncertainty_std": float(uncertainty) if np.isfinite(uncertainty) else np.nan,
                    "absolute_error": float(abs(true - predicted)),
                    **meta.iloc[sample_index].to_dict(),
                }
            )

        summary_rows.append(
            {
                "k": k,
                "test_index": test_index,
                "model": model_name,
                "training_samples": len(y_train),
                "test_samples": len(y_test),
                "MAE": mean_absolute_error(y_test, mean),
                "RMSE": mean_squared_error(y_test, mean) ** 0.5,
                "R2": r2_score(y_test, mean),
            }
        )

    predictions_frame = pd.DataFrame(prediction_rows)
    summary_frame = pd.DataFrame(summary_rows).sort_values("RMSE").reset_index(drop=True)
    return predictions_frame, summary_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hold out one row from every force npz file and compare six regression models."
    )
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR / "Tactile_sensor_training_data")
    parser.add_argument("--test-index", type=int, default=0, help="Held-out row index in every npz file.")
    parser.add_argument("--k", type=int, default=3, help="Number of top sensor values.")
    parser.add_argument("--sort-top-k", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=CODE_DIR / "results" / "holdout")
    parser.add_argument("--experiment-root", type=Path, default=SCRIPT_DIR / "experiment_model")
    args = parser.parse_args()
    if not 1 <= args.k <= 9:
        parser.error("--k must be between 1 and 9")
    if args.test_index < 0:
        parser.error("--test-index must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    predictions, summary = evaluate_holdout(
        data_dir=args.data_dir,
        test_index=args.test_index,
        k=args.k,
        sort_top_k=args.sort_top_k,
        random_state=args.random_state,
        experiment_dir=args.experiment_root / f"k_{args.k}",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / f"holdout_row_{args.test_index}_k{args.k}"
    predictions_path = Path(f"{prefix}_predictions.csv")
    summary_path = Path(f"{prefix}_summary.csv")
    predictions.to_csv(predictions_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Hold-out rule: row {args.test_index} from every force file is test data")
    print(f"k={args.k}")
    print(f"Training samples: {int(summary.iloc[0]['training_samples'])}")
    print(f"Test samples: {int(summary.iloc[0]['test_samples'])}")
    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nPredictions:")
    print(
        predictions[
            ["model", "force_N", "y_pred", "uncertainty_std", "absolute_error"]
        ].to_string(index=False)
    )
    print(f"\nSaved: {predictions_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved experiment models: {args.experiment_root / f'k_{args.k}'}")


if __name__ == "__main__":
    main()
