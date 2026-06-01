"""
# 설명
- 저장된 모델(.joblib)을 불러와 실제 로봇손 test 데이터(.npz)에 대해 inference 수행
- 동료가 실제 로봇손 테스트에 사용하기 위한 단순 inference 코드
- 기본 출력 CSV에는 입력 센서값(sensor_0~sensor_8), 모델 이름, 예측 Force 값만 저장
- 필요하면 --include_top5_features, --include_uncertainty 옵션으로 추가 정보 저장 가능

# 실행 명령어
python Code/inference_saved_models.py --input_npz Data/Tactile_sensor_test_data/tac_finger_r_sensor1_10N.npz --model_path Code/saved_models/bayesian_linear_regression.joblib --output_csv Code/results/robot_inference_bayesian.csv
python Code/inference_saved_models.py --input_npz Data/Tactile_sensor_test_data/tac_finger_r_sensor1_10N.npz --all_models --output_csv Code/results/robot_inference_all_models.csv
python Code/inference_saved_models.py --input_npz Data/Tactile_sensor_test_data/tac_finger_r_sensor1_10N.npz --all_models --include_uncertainty
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from force_common import extract_top_k_sensors, load_model_bundle, load_npz_array


DEFAULT_MODEL_FILES = [
    "bayesian_linear_regression.joblib",
    "random_forest_regression.joblib",
    "random_forest_regression_with_tree_prediction_variance.joblib",
    "single_mlp_regression.joblib",
    "mlp_deep_ensemble.joblib",
]


def predict_model(
    model: Any,
    x_top: np.ndarray,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "predict_with_std"):
        return model.predict_with_std(x_top)

    if hasattr(model, "named_steps"):
        regressor = model.named_steps.get("regressor")

        if regressor is not None and regressor.__class__.__name__ == "BayesianRidge":
            x_scaled = model.named_steps["scaler"].transform(x_top)
            mean, std = regressor.predict(x_scaled, return_std=True)
            return mean, std

        if (
            regressor is not None
            and regressor.__class__.__name__ == "RandomForestRegressor"
            and "Tree Prediction Variance" in model_name
        ):
            x_scaled = model.named_steps["scaler"].transform(x_top)
            tree_preds = np.vstack([tree.predict(x_scaled) for tree in regressor.estimators_])
            mean = tree_preds.mean(axis=0)
            std = tree_preds.std(axis=0, ddof=1)
            return mean, std

    mean = model.predict(x_top)
    std = np.full(shape=mean.shape, fill_value=np.nan, dtype=float)
    return mean, std


def model_paths_from_args(args: argparse.Namespace) -> list[Path]:
    if args.model_path:
        return [Path(args.model_path)]

    model_dir = Path(args.model_dir)
    if args.all_models:
        return [model_dir / name for name in DEFAULT_MODEL_FILES]

    return [model_dir / "bayesian_linear_regression.joblib"]


def build_rows_for_model(
    model_path: Path,
    input_npz: Path,
    x_raw: np.ndarray,
    x_top: np.ndarray,
    include_top5_features: bool = False,
    include_uncertainty: bool = False,
) -> list[dict[str, Any]]:
    bundle = load_model_bundle(model_path)
    model = bundle["model"]
    model_name = bundle.get("model_name", model_path.stem)
    mean, std = predict_model(model, x_top, model_name)

    rows: list[dict[str, Any]] = []
    for sample_index in range(x_raw.shape[0]):
        row: dict[str, Any] = {
            "input_file": input_npz.name,
            "sample_index": sample_index,
            "model_name": model_name,
        }

        for sensor_index, value in enumerate(x_raw[sample_index]):
            row[f"sensor_{sensor_index}"] = float(value)

        if include_top5_features:
            for feature_index, value in enumerate(x_top[sample_index]):
                row[f"top5_feature_{feature_index}"] = float(value)

        row["predicted_force_N"] = float(mean[sample_index])

        if include_uncertainty:
            row["estimated_uncertainty_std"] = (
                float(std[sample_index]) if np.isfinite(std[sample_index]) else np.nan
            )

        rows.append(row)

    return rows


def main() -> None:
    code_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Load saved force regression model(s) and run simple inference on robot-hand test data."
    )
    parser.add_argument(
        "--input_npz",
        required=True,
        help="Input .npz file containing raw tactile sensor rows, e.g. shape (n_samples, 9).",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Single saved .joblib model bundle to load. If omitted, uses --model_dir.",
    )
    parser.add_argument(
        "--model_dir",
        default=str(code_dir / "saved_models"),
        help="Directory containing saved .joblib model bundles.",
    )
    parser.add_argument(
        "--all_models",
        action="store_true",
        help="Run inference with all default saved models in --model_dir.",
    )
    parser.add_argument(
        "--output_csv",
        default=str(code_dir / "robot_inference_results.csv"),
        help="CSV path for simple input/output inference results.",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of top sensor values to use.")
    parser.add_argument(
        "--sort_top_k",
        action="store_true",
        help="Sort top-k sensor values descending. Default matches training preprocessing.",
    )
    parser.add_argument(
        "--include_top5_features",
        action="store_true",
        help="Also save the top-k features used by the model.",
    )
    parser.add_argument(
        "--include_uncertainty",
        action="store_true",
        help="Also save estimated uncertainty when the loaded model supports it.",
    )
    args = parser.parse_args()

    input_npz = Path(args.input_npz)
    x_raw = load_npz_array(input_npz)
    x_top = extract_top_k_sensors(x_raw, k=args.k, sort_values=args.sort_top_k)

    rows: list[dict[str, Any]] = []
    for model_path in model_paths_from_args(args):
        if not model_path.exists():
            raise FileNotFoundError(f"Saved model not found: {model_path}")
        rows.extend(
            build_rows_for_model(
                model_path=model_path,
                input_npz=input_npz,
                x_raw=x_raw,
                x_top=x_top,
                include_top5_features=args.include_top5_features,
                include_uncertainty=args.include_uncertainty,
            )
        )

    results = pd.DataFrame(rows)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)

    print("Input:", input_npz)
    print("Output:", output_csv)
    print(results)


if __name__ == "__main__":
    main()
