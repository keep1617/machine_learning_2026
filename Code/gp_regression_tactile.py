"""
Train, save, and run inference with GP tactile force regression.

Examples:
    python3 Code/gp_regression_tactile.py --mode train_all --data_dir Tactile_sensor_training_data --k 3
    python3 Code/gp_regression_tactile.py --mode predict --input_npz Tactile_sensor_test_data/tac_finger_r_sensor1_10N.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from force_common import load_npz_features, load_tactile_dataset


CODE_DIR = Path(__file__).resolve().parent
ASSIGNMENT_DIR = CODE_DIR.parent
DEFAULT_MODEL_PATH = CODE_DIR / "gp_model.pkl"
DEFAULT_SCALER_PATH = CODE_DIR / "scaler.pkl"
DEFAULT_CONFIG_PATH = CODE_DIR / "gp_config.json"


def build_model(random_state: int) -> GaussianProcessRegressor:
    kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0)
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=True,
        random_state=random_state,
    )


def save_config(path: Path, k: int, sort_top_k: bool) -> None:
    path.write_text(
        json.dumps({"k": k, "sort_top_k": sort_top_k}, indent=2) + "\n",
        encoding="utf-8",
    )


def load_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"k": 5, "sort_top_k": False}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GP regression for tactile force estimation.")
    parser.add_argument("--mode", choices=["train_all", "predict"], default="train_all")
    parser.add_argument(
        "--data_dir",
        default=str(ASSIGNMENT_DIR / "Tactile_sensor_training_data"),
    )
    parser.add_argument("--input_npz", default=None)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--scaler_path", type=Path, default=DEFAULT_SCALER_PATH)
    parser.add_argument("--config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--k", type=int, default=None, help="Number of top sensor values.")
    parser.add_argument("--sort_top_k", action="store_true")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    if args.k is not None and not 1 <= args.k <= 9:
        parser.error("--k must be between 1 and 9")
    if args.mode == "predict" and args.input_npz is None:
        parser.error("--input_npz is required when --mode predict")
    return args


def main() -> None:
    args = parse_args()

    if args.mode == "train_all":
        k = args.k if args.k is not None else 5
        x, y, _ = load_tactile_dataset(args.data_dir, k=k, sort_top_k=args.sort_top_k)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        model = build_model(args.random_state)
        model.fit(x_scaled, y)
        pred, std = model.predict(x_scaled, return_std=True)

        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        args.scaler_path.parent.mkdir(parents=True, exist_ok=True)
        args.config_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model_path)
        joblib.dump(scaler, args.scaler_path)
        save_config(args.config_path, k=k, sort_top_k=args.sort_top_k)

        print("Dataset loaded")
        print("X shape:", x.shape)
        print("Force labels:", sorted(set(y.tolist())))
        print("Optimized kernel:", model.kernel_)
        print(
            "Training summary:",
            f"MAE={mean_absolute_error(y, pred):.6f}",
            f"RMSE={mean_squared_error(y, pred) ** 0.5:.6f}",
            f"R2={r2_score(y, pred):.6f}",
            f"mean_std={std.mean():.6f}",
        )
        print("Saved model:", args.model_path)
        print("Saved scaler:", args.scaler_path)
        print("Saved config:", args.config_path)
        return

    config = load_config(args.config_path)
    k = args.k if args.k is not None else int(config["k"])
    sort_top_k = args.sort_top_k or bool(config["sort_top_k"])
    model = joblib.load(args.model_path)
    scaler = joblib.load(args.scaler_path)
    x = load_npz_features(args.input_npz, k=k, sort_top_k=sort_top_k)
    pred, std = model.predict(scaler.transform(x), return_std=True)
    print(f"Input: {args.input_npz}")
    print(f"k={k}, sort_top_k={sort_top_k}")
    for index, (mean, uncertainty) in enumerate(zip(pred, std)):
        print(f"sample {index}: force={mean:.3f}N, std={uncertainty:.3f}N")


if __name__ == "__main__":
    main()
