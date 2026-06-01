"""
# 설명
- 모든 Baseline 모델이 공통으로 사용하는 Utils 코드
- .npz 데이터 로딩, Force label parsing, Top-5 센서 feature 추출 수행
- LOOCV 평가, 전체 데이터 학습(train_all), 모델 저장/로드, 결과 CSV 저장 기능 포함
- 보통 직접 실행하지 않고 각 Baseline 스크립트에서 import해서 사용

# 실행 명령어 예시
python Code/bayesian_linear_regression.py --mode train_all --data_dir Data/Tactile_sensor_test_data
python Code/bayesian_linear_regression.py --mode predict --input_npz Data/Tactile_sensor_test_data/tac_finger_r_sensor1_10N.npz
"""

from __future__ import annotations

import argparse
import re
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ModelBuilder = Callable[[int], Any]
PredictWithUncertainty = Callable[[Any, np.ndarray], tuple[np.ndarray, np.ndarray]]


def code_dir() -> Path:
    return Path(__file__).resolve().parent


def assignment_dir() -> Path:
    return code_dir().parent


def default_data_dir() -> Path:
    return assignment_dir() / "Data" / "Tactile_sensor_test_data"


def make_slug(model_name: str) -> str:
    return (
        model_name.lower()
        .replace("+", "plus")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def parse_force_from_filename(path: str | Path) -> float:
    path = Path(path)
    match = re.search(r"_(\d+(?:\.\d+)?)N\.npz$", path.name)
    if match is None:
        raise ValueError(f"Could not parse force label from file name: {path.name}")
    return float(match.group(1))


def extract_top_k_sensors(
    x: np.ndarray,
    k: int = 5,
    sort_values: bool = False,
) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D sensor array, got shape {x.shape}")
    if x.shape[1] < k:
        raise ValueError(f"Need at least {k} sensors, got shape {x.shape}")

    top_k = np.partition(x, -k, axis=1)[:, -k:]
    if sort_values:
        top_k = np.sort(top_k, axis=1)[:, ::-1]
    return top_k.astype(float)


def load_npz_array(npz_path: str | Path) -> np.ndarray:
    npz_path = Path(npz_path)
    loaded = np.load(npz_path)
    if "arr_0" in loaded.files:
        arr = loaded["arr_0"]
    elif len(loaded.files) == 1:
        arr = loaded[loaded.files[0]]
    else:
        raise ValueError(
            f"Expected arr_0 or a single array in {npz_path.name}, found {loaded.files}"
        )

    arr = arr.astype(float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array in {npz_path.name}, got shape {arr.shape}")
    return arr


def load_npz_features(
    npz_path: str | Path,
    k: int = 5,
    sort_top_k: bool = False,
) -> np.ndarray:
    arr = load_npz_array(npz_path)
    return extract_top_k_sensors(arr, k=k, sort_values=sort_top_k)


def load_tactile_dataset(
    data_dir: str | Path,
    k: int = 5,
    sort_top_k: bool = False,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    data_dir = Path(data_dir)
    npz_files = sorted(data_dir.glob("*.npz"), key=parse_force_from_filename)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in: {data_dir.resolve()}")

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []

    for path in npz_files:
        force = parse_force_from_filename(path)
        x_top = load_npz_features(path, k=k, sort_top_k=sort_top_k)
        y = np.full(shape=(x_top.shape[0],), fill_value=force, dtype=float)

        x_list.append(x_top)
        y_list.append(y)
        for row_idx in range(x_top.shape[0]):
            meta_rows.append(
                {
                    "file": path.name,
                    "row_in_file": row_idx,
                    "force_N": force,
                }
            )

    x = np.vstack(x_list)
    y = np.concatenate(y_list)
    meta = pd.DataFrame(meta_rows)
    return x, y, meta


def nan_std_like(mean: np.ndarray) -> np.ndarray:
    return np.full(shape=mean.shape, fill_value=np.nan, dtype=float)


def finite_or_nan(value: float) -> float:
    return float(value) if np.isfinite(value) else np.nan


def evaluate_loocv(
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    model_name: str,
    build_model: ModelBuilder,
    predict_with_uncertainty: PredictWithUncertainty,
    random_state: int = 42,
) -> pd.DataFrame:
    loo = LeaveOneOut()
    rows: list[dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(loo.split(x)):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = build_model(random_state + fold)
        model.fit(x_train, y_train)
        pred_mean, pred_std = predict_with_uncertainty(model, x_test)

        std_value = pred_std[0] if len(pred_std) else np.nan
        row = {
            "fold": fold,
            "test_index": int(test_idx[0]),
            "model": model_name,
            "y_true": float(y_test[0]),
            "y_pred": float(pred_mean[0]),
            "uncertainty_std": finite_or_nan(float(std_value)),
            "absolute_error": float(abs(y_test[0] - pred_mean[0])),
        }
        row.update(meta.iloc[test_idx[0]].to_dict())
        rows.append(row)

    return pd.DataFrame(rows)


def pearson_corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model_name, group in results.groupby("model"):
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group["y_pred"].to_numpy(dtype=float)
        std = group["uncertainty_std"].to_numpy(dtype=float)
        abs_err = group["absolute_error"].to_numpy(dtype=float)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        r2 = r2_score(y_true, y_pred)
        unc_err_corr = pearson_corr_safe(std, abs_err)

        if np.isfinite(std).any():
            lower = y_pred - 1.96 * std
            upper = y_pred + 1.96 * std
            coverage_95 = float(np.mean((y_true >= lower) & (y_true <= upper)))
            avg_interval_width_95 = float(np.nanmean(upper - lower))
        else:
            coverage_95 = np.nan
            avg_interval_width_95 = np.nan

        rows.append(
            {
                "model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "uncertainty_error_corr": unc_err_corr,
                "coverage_95": coverage_95,
                "avg_interval_width_95": avg_interval_width_95,
            }
        )

    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def build_labeled_prediction_frame(
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    model: Any,
    model_name: str,
    predict_with_uncertainty: PredictWithUncertainty,
) -> pd.DataFrame:
    pred_mean, pred_std = predict_with_uncertainty(model, x)
    rows: list[dict[str, Any]] = []

    for sample_index, (true, mean, std) in enumerate(zip(y, pred_mean, pred_std)):
        row = {
            "sample_index": sample_index,
            "model": model_name,
            "y_true": float(true),
            "y_pred": float(mean),
            "uncertainty_std": finite_or_nan(float(std)),
            "absolute_error": float(abs(true - mean)),
        }
        row.update(meta.iloc[sample_index].to_dict())
        rows.append(row)

    return pd.DataFrame(rows)


def collect_loss_diagnostics(model: Any) -> pd.DataFrame:
    if hasattr(model, "loss_diagnostics"):
        return model.loss_diagnostics()

    regressor = None
    if hasattr(model, "named_steps"):
        regressor = model.named_steps.get("regressor")

    if regressor is None or not hasattr(regressor, "loss_curve_"):
        return pd.DataFrame(
            [
                {
                    "member": "model",
                    "has_loss_curve": False,
                    "n_iter": np.nan,
                    "initial_loss": np.nan,
                    "final_loss": np.nan,
                    "best_loss": np.nan,
                    "loss_reduction": np.nan,
                    "convergence_note": "No iterative training loss is exposed for this model.",
                }
            ]
        )

    loss_curve = np.asarray(regressor.loss_curve_, dtype=float)
    initial_loss = loss_curve[0] if len(loss_curve) else np.nan
    final_loss = loss_curve[-1] if len(loss_curve) else np.nan
    best_loss = np.min(loss_curve) if len(loss_curve) else np.nan

    return pd.DataFrame(
        [
            {
                "member": "model",
                "has_loss_curve": True,
                "n_iter": getattr(regressor, "n_iter_", np.nan),
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "best_loss": best_loss,
                "loss_reduction": initial_loss - final_loss,
                "convergence_note": "Stopped when training loss did not improve beyond tol for n_iter_no_change epochs, or max_iter was reached.",
            }
        ]
    )


def save_model_bundle(
    path: str | Path,
    model: Any,
    model_name: str,
    k: int,
    sort_top_k: bool,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "model_name": model_name,
        "k": k,
        "sort_top_k": sort_top_k,
    }
    joblib.dump(bundle, path)


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    return joblib.load(Path(path))


def build_prediction_frame(
    model: Any,
    input_npz: str | Path,
    predict_with_uncertainty: PredictWithUncertainty,
    k: int,
    sort_top_k: bool,
) -> pd.DataFrame:
    x_test = load_npz_features(input_npz, k=k, sort_top_k=sort_top_k)
    pred_mean, pred_std = predict_with_uncertainty(model, x_test)

    rows: list[dict[str, Any]] = []
    for sample_index, (mean, std) in enumerate(zip(pred_mean, pred_std)):
        finite_std = np.isfinite(std)
        rows.append(
            {
                "input_file": Path(input_npz).name,
                "sample_index": sample_index,
                "y_pred": float(mean),
                "uncertainty_std": float(std) if finite_std else np.nan,
                "lower_95": float(mean - 1.96 * std) if finite_std else np.nan,
                "upper_95": float(mean + 1.96 * std) if finite_std else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_model_cli(
    model_name: str,
    build_model: ModelBuilder,
    predict_with_uncertainty: PredictWithUncertainty,
    description: str,
) -> None:
    slug = make_slug(model_name)
    default_model_path = code_dir() / "saved_models" / f"{slug}.joblib"
    default_save_prefix = code_dir() / "results" / slug

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--mode",
        choices=["train_eval", "train_all", "predict"],
        default="train_eval",
        help="Run LOOCV/train/save, train on all data only, or load a saved model and predict an npz file.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(default_data_dir()),
        help="Folder containing tac_finger_..._*N.npz files.",
    )
    parser.add_argument(
        "--input_npz",
        type=str,
        default=None,
        help="Input .npz file for --mode predict.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(default_model_path),
        help="Path used to save/load the final model.",
    )
    parser.add_argument(
        "--save_prefix",
        type=str,
        default=str(default_save_prefix),
        help="Prefix for output CSV files.",
    )
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--k", type=int, default=5, help="Number of top sensor values.")
    parser.add_argument(
        "--sort_top_k",
        action="store_true",
        help="Sort top-k sensor values descending. Default matches the original notebook.",
    )
    args = parser.parse_args()

    if args.mode in {"train_eval", "train_all"}:
        x, y, meta = load_tactile_dataset(args.data_dir, k=args.k, sort_top_k=args.sort_top_k)
        print("Dataset loaded")
        print("X shape:", x.shape)
        print("y shape:", y.shape)
        print("Force labels:", sorted(set(y.tolist())))

    if args.mode == "train_all":
        final_model = build_model(args.random_state)
        final_model.fit(x, y)

        train_results = build_labeled_prediction_frame(
            x=x,
            y=y,
            meta=meta,
            model=final_model,
            model_name=model_name,
            predict_with_uncertainty=predict_with_uncertainty,
        )
        train_summary = summarize_results(train_results)
        loss_diagnostics = collect_loss_diagnostics(final_model)

        train_results_path = Path(f"{args.save_prefix}_train_predictions.csv")
        train_summary_path = Path(f"{args.save_prefix}_train_summary.csv")
        loss_path = Path(f"{args.save_prefix}_loss_diagnostics.csv")
        train_results_path.parent.mkdir(parents=True, exist_ok=True)
        train_results.to_csv(train_results_path, index=False)
        train_summary.to_csv(train_summary_path, index=False)
        loss_diagnostics.to_csv(loss_path, index=False)

        save_model_bundle(
            path=args.model_path,
            model=final_model,
            model_name=model_name,
            k=args.k,
            sort_top_k=args.sort_top_k,
        )

        print("\nTraining-set prediction results:")
        print(
            train_results[
                [
                    "sample_index",
                    "force_N",
                    "model",
                    "y_true",
                    "y_pred",
                    "uncertainty_std",
                    "absolute_error",
                ]
            ]
        )
        print("\nTraining summary:")
        print(train_summary)
        print("\nLoss diagnostics:")
        print(loss_diagnostics)
        print(f"\nSaved: {train_results_path}")
        print(f"Saved: {train_summary_path}")
        print(f"Saved: {loss_path}")
        print(f"Saved model: {args.model_path}")
        return

    if args.mode == "train_eval":
        results = evaluate_loocv(
            x=x,
            y=y,
            meta=meta,
            model_name=model_name,
            build_model=build_model,
            predict_with_uncertainty=predict_with_uncertainty,
            random_state=args.random_state,
        )
        summary = summarize_results(results)

        results_path = Path(f"{args.save_prefix}_loocv_predictions.csv")
        summary_path = Path(f"{args.save_prefix}_summary.csv")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(results_path, index=False)
        summary.to_csv(summary_path, index=False)

        final_model = build_model(args.random_state)
        final_model.fit(x, y)
        save_model_bundle(
            path=args.model_path,
            model=final_model,
            model_name=model_name,
            k=args.k,
            sort_top_k=args.sort_top_k,
        )

        print("\nLOOCV prediction results:")
        print(
            results[
                [
                    "test_index",
                    "force_N",
                    "model",
                    "y_true",
                    "y_pred",
                    "uncertainty_std",
                    "absolute_error",
                ]
            ]
        )
        print("\nSummary:")
        print(summary)
        print(f"\nSaved: {results_path}")
        print(f"Saved: {summary_path}")
        print(f"Saved model: {args.model_path}")
        return

    if args.input_npz is None:
        parser.error("--input_npz is required when --mode predict")

    bundle = load_model_bundle(args.model_path)
    predictions = build_prediction_frame(
        model=bundle["model"],
        input_npz=args.input_npz,
        predict_with_uncertainty=predict_with_uncertainty,
        k=int(bundle.get("k", args.k)),
        sort_top_k=bool(bundle.get("sort_top_k", args.sort_top_k)),
    )

    predict_path = Path(f"{args.save_prefix}_predictions.csv")
    predict_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predict_path, index=False)

    print("Loaded model:", bundle.get("model_name", model_name))
    print("Input npz:", args.input_npz)
    print("\nPredictions:")
    print(predictions)
    print(f"\nSaved: {predict_path}")


class MLPDeepEnsembleRegressor:
    def __init__(
        self,
        n_members: int = 3,
        hidden_layer_sizes: tuple[int, ...] = (8,),
        activation: str = "relu",
        solver: str = "adam",
        learning_rate_init: float = 1e-2,
        alpha: float = 1e-2,
        max_iter: int = 300,
        tol: float = 1e-5,
        n_iter_no_change: int = 50,
        bootstrap: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_members = n_members
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.models_: list[Pipeline] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MLPDeepEnsembleRegressor":
        rng = np.random.default_rng(self.random_state)
        self.models_ = []

        for member_idx in range(self.n_members):
            if self.bootstrap:
                sampled_idx = rng.choice(len(x), size=len(x), replace=True)
                x_member = x[sampled_idx]
                y_member = y[sampled_idx]
            else:
                x_member = x
                y_member = y

            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "regressor",
                        MLPRegressor(
                            hidden_layer_sizes=self.hidden_layer_sizes,
                            activation=self.activation,
                            solver=self.solver,
                            learning_rate_init=self.learning_rate_init,
                            alpha=self.alpha,
                            max_iter=self.max_iter,
                            tol=self.tol,
                            n_iter_no_change=self.n_iter_no_change,
                            random_state=self.random_state + member_idx,
                        ),
                    ),
                ]
            )
            model.fit(x_member, y_member)
            self.models_.append(model)

        return self

    def predict_member_matrix(self, x: np.ndarray) -> np.ndarray:
        if not self.models_:
            raise ValueError("MLPDeepEnsembleRegressor must be fitted before prediction.")
        return np.vstack([model.predict(x) for model in self.models_])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_member_matrix(x).mean(axis=0)

    def predict_with_std(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds = self.predict_member_matrix(x)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0, ddof=1) if preds.shape[0] > 1 else nan_std_like(mean)
        return mean, std

    def loss_diagnostics(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        for member_idx, model in enumerate(self.models_):
            regressor = model.named_steps["regressor"]
            loss_curve = np.asarray(regressor.loss_curve_, dtype=float)
            initial_loss = loss_curve[0] if len(loss_curve) else np.nan
            final_loss = loss_curve[-1] if len(loss_curve) else np.nan
            best_loss = np.min(loss_curve) if len(loss_curve) else np.nan
            rows.append(
                {
                    "member": member_idx,
                    "has_loss_curve": True,
                    "n_iter": getattr(regressor, "n_iter_", np.nan),
                    "initial_loss": initial_loss,
                    "final_loss": final_loss,
                    "best_loss": best_loss,
                    "loss_reduction": initial_loss - final_loss,
                    "convergence_note": "Stopped when training loss did not improve beyond tol for n_iter_no_change epochs, or max_iter was reached.",
                }
            )

        return pd.DataFrame(rows)
