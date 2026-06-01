"""
Visualize hold-out regression performance while changing Top-K from 1 to 9.

Examples:
    python3 07_visualize_k_holdout_comparison.py
    python3 07_visualize_k_holdout_comparison.py --test-index 4
    python3 07_visualize_k_holdout_comparison.py --k-values 1 3 5 7 9
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
HOLDOUT_SCRIPT = SCRIPT_DIR / "06_holdout_row_model_comparison.py"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "Code" / "results" / "holdout"
DEFAULT_EXPERIMENT_ROOT = SCRIPT_DIR / "experiment_model"


def load_holdout_module() -> Any:
    spec = importlib.util.spec_from_file_location("holdout_row_model_comparison", HOLDOUT_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper module: {HOLDOUT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


holdout = load_holdout_module()


def run_k_comparison(
    data_dir: Path,
    test_index: int,
    k_values: list[int],
    sort_top_k: bool,
    random_state: int,
    experiment_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for k in k_values:
        print(f"Evaluating k={k}...", flush=True)
        predictions, summary = holdout.evaluate_holdout(
            data_dir=data_dir,
            test_index=test_index,
            k=k,
            sort_top_k=sort_top_k,
            random_state=random_state,
            experiment_dir=experiment_root / f"k_{k}",
        )
        prediction_frames.append(predictions)
        summary_frames.append(summary)
    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(summary_frames, ignore_index=True),
    )


def plot_metric_lines(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for model_name, group in summary.groupby("model"):
        ordered = group.sort_values("k")
        axes[0].plot(ordered["k"], ordered["RMSE"], marker="o", linewidth=2, label=model_name)
        axes[1].plot(ordered["k"], ordered["MAE"], marker="o", linewidth=2, label=model_name)

    axes[0].set_title("Hold-out RMSE by Top-K")
    axes[1].set_title("Hold-out MAE by Top-K")
    for axis in axes:
        axis.set_xlabel("Top-K sensor values")
        axis.set_ylabel("Error (N)")
        axis.set_xticks(sorted(summary["k"].unique()))
        axis.grid(True, alpha=0.3)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_rmse_heatmap(summary: pd.DataFrame, output_path: Path) -> None:
    pivot = summary.pivot(index="model", columns="k", values="RMSE")
    fig, axis = plt.subplots(figsize=(12, 5.5))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu_r")
    axis.set_title("Hold-out RMSE Heatmap")
    axis.set_xlabel("Top-K sensor values")
    axis.set_ylabel("Model")
    axis.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns)
    axis.set_yticks(np.arange(len(pivot.index)), labels=pivot.index, fontsize=8)
    for row_index in range(len(pivot.index)):
        for column_index in range(len(pivot.columns)):
            value = pivot.iloc[row_index, column_index]
            axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, label="RMSE (N)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def summarize_force_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["k", "test_index", "model", "force_N"]
    for keys, group in predictions.groupby(group_columns, sort=True):
        k, test_index, model_name, force = keys
        errors = group["y_pred"].to_numpy(dtype=float) - group["force_N"].to_numpy(dtype=float)
        uncertainty = group["uncertainty_std"].to_numpy(dtype=float)
        finite_uncertainty = uncertainty[np.isfinite(uncertainty)]
        rows.append(
            {
                "k": int(k),
                "test_index": int(test_index),
                "model": model_name,
                "force_N": float(force),
                "test_samples": len(group),
                "MSE": float(np.mean(errors**2)),
                "RMSE": float(np.sqrt(np.mean(errors**2))),
                "mean_uncertainty_std": (
                    float(np.mean(finite_uncertainty)) if len(finite_uncertainty) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def select_best_k_force_metrics(
    summary: pd.DataFrame,
    force_metrics: pd.DataFrame,
) -> pd.DataFrame:
    best_rows = summary.loc[summary.groupby("model")["RMSE"].idxmin(), ["model", "k"]]
    best_rows = best_rows.rename(columns={"k": "best_k"})
    selected = force_metrics.merge(best_rows, left_on=["model", "k"], right_on=["model", "best_k"])
    return selected.drop(columns=["best_k"]).sort_values(["model", "force_N"]).reset_index(drop=True)


def plot_force_metric(
    force_metrics: pd.DataFrame,
    value_column: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    model_names = sorted(force_metrics["model"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
    for axis, model_name in zip(axes.flat, model_names):
        model_frame = force_metrics[force_metrics["model"] == model_name]
        has_values = False
        for k, group in model_frame.groupby("k"):
            ordered = group.sort_values("force_N")
            values = ordered[value_column].to_numpy(dtype=float)
            if not np.isfinite(values).any():
                continue
            has_values = True
            axis.plot(
                ordered["force_N"],
                values,
                marker="o",
                linewidth=1.5,
                label=f"k={k}",
            )
        axis.set_title(model_name, fontsize=10)
        axis.set_xlabel("Force label (N)")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        if has_values:
            axis.legend(fontsize=7, ncol=3)
        else:
            axis.text(0.5, 0.5, "No uncertainty output", ha="center", va="center")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_best_k_force_metric(
    force_metrics: pd.DataFrame,
    value_column: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(11, 6.5))
    for model_name, group in force_metrics.groupby("model"):
        ordered = group.sort_values("force_N")
        values = ordered[value_column].to_numpy(dtype=float)
        if not np.isfinite(values).any():
            continue
        k = int(ordered["k"].iloc[0])
        axis.plot(
            ordered["force_N"],
            values,
            marker="o",
            linewidth=2,
            label=f"{model_name} (k={k})",
        )
    axis.set_title(title)
    axis.set_xlabel("Force label (N)")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def print_best_k(summary: pd.DataFrame) -> None:
    best_rows = summary.loc[summary.groupby("model")["RMSE"].idxmin()]
    best_rows = best_rows.sort_values("RMSE").reset_index(drop=True)
    print("\nBest k for each model:")
    print(best_rows[["model", "k", "MAE", "RMSE", "R2"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize six-model hold-out performance by Top-K.")
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR / "Tactile_sensor_training_data")
    parser.add_argument("--test-index", type=int, default=0, help="Held-out row index in every npz file.")
    parser.add_argument("--k-values", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--sort-top-k", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    args = parser.parse_args()
    if args.test_index < 0:
        parser.error("--test-index must be non-negative")
    if not args.k_values or any(not 1 <= k <= 9 for k in args.k_values):
        parser.error("--k-values must contain integers between 1 and 9")
    args.k_values = sorted(set(args.k_values))
    return args


def main() -> None:
    args = parse_args()
    predictions, summary = run_k_comparison(
        data_dir=args.data_dir,
        test_index=args.test_index,
        k_values=args.k_values,
        sort_top_k=args.sort_top_k,
        random_state=args.random_state,
        experiment_root=args.experiment_root,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / f"holdout_row_{args.test_index}_k_comparison"
    predictions_path = Path(f"{prefix}_predictions.csv")
    summary_path = Path(f"{prefix}_summary.csv")
    metrics_plot_path = Path(f"{prefix}_metrics.png")
    heatmap_path = Path(f"{prefix}_rmse_heatmap.png")
    force_metrics_path = Path(f"{prefix}_force_metrics.csv")
    force_mse_plot_path = Path(f"{prefix}_force_mse.png")
    force_rmse_plot_path = Path(f"{prefix}_force_rmse.png")
    force_uncertainty_plot_path = Path(f"{prefix}_force_uncertainty.png")
    best_k_force_metrics_path = Path(f"{prefix}_best_k_force_metrics.csv")
    best_k_force_mse_plot_path = Path(f"{prefix}_best_k_force_mse.png")
    best_k_force_rmse_plot_path = Path(f"{prefix}_best_k_force_rmse.png")
    best_k_force_uncertainty_plot_path = Path(f"{prefix}_best_k_force_uncertainty.png")
    force_metrics = summarize_force_metrics(predictions)
    best_k_force_metrics = select_best_k_force_metrics(summary, force_metrics)

    predictions.to_csv(predictions_path, index=False)
    summary.to_csv(summary_path, index=False)
    force_metrics.to_csv(force_metrics_path, index=False)
    best_k_force_metrics.to_csv(best_k_force_metrics_path, index=False)
    plot_metric_lines(summary, metrics_plot_path)
    plot_rmse_heatmap(summary, heatmap_path)
    plot_force_metric(
        force_metrics,
        value_column="MSE",
        title="Hold-out MSE by Force Label",
        ylabel="MSE (N^2)",
        output_path=force_mse_plot_path,
    )
    plot_force_metric(
        force_metrics,
        value_column="RMSE",
        title="Hold-out RMSE by Force Label",
        ylabel="RMSE (N)",
        output_path=force_rmse_plot_path,
    )
    plot_force_metric(
        force_metrics,
        value_column="mean_uncertainty_std",
        title="Hold-out Mean Uncertainty by Force Label",
        ylabel="Estimated uncertainty std (N)",
        output_path=force_uncertainty_plot_path,
    )
    plot_best_k_force_metric(
        best_k_force_metrics,
        value_column="MSE",
        title="Hold-out MSE by Force Label with Per-Model Best K",
        ylabel="MSE (N^2)",
        output_path=best_k_force_mse_plot_path,
    )
    plot_best_k_force_metric(
        best_k_force_metrics,
        value_column="RMSE",
        title="Hold-out RMSE by Force Label with Per-Model Best K",
        ylabel="RMSE (N)",
        output_path=best_k_force_rmse_plot_path,
    )
    plot_best_k_force_metric(
        best_k_force_metrics,
        value_column="mean_uncertainty_std",
        title="Hold-out Mean Uncertainty by Force Label with Per-Model Best K",
        ylabel="Estimated uncertainty std (N)",
        output_path=best_k_force_uncertainty_plot_path,
    )
    print_best_k(summary)

    print(f"\nSaved: {predictions_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {metrics_plot_path}")
    print(f"Saved: {heatmap_path}")
    print(f"Saved: {force_metrics_path}")
    print(f"Saved: {force_mse_plot_path}")
    print(f"Saved: {force_rmse_plot_path}")
    print(f"Saved: {force_uncertainty_plot_path}")
    print(f"Saved: {best_k_force_metrics_path}")
    print(f"Saved: {best_k_force_mse_plot_path}")
    print(f"Saved: {best_k_force_rmse_plot_path}")
    print(f"Saved: {best_k_force_uncertainty_plot_path}")
    print(f"Saved experiment models: {args.experiment_root}")


if __name__ == "__main__":
    main()
