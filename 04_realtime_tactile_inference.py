"""
Subscribe to tactile sensor data and continuously estimate force with saved models.

Examples:
    source /opt/ros/humble/setup.bash
    python3 04_realtime_tactile_inference.py
    python3 04_realtime_tactile_inference.py --sensor-index 2 --all-models
    python3 04_realtime_tactile_inference.py --gp-only
    python3 04_realtime_tactile_inference.py --output-csv Code/results/realtime_inference.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import rclpy
from control_msgs.msg import DynamicJointState
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR / "Code"
DEFAULT_MODEL_DIR = CODE_DIR / "saved_models"
DEFAULT_GP_MODEL_PATH = CODE_DIR / "gp_model.pkl"
DEFAULT_GP_SCALER_PATH = CODE_DIR / "scaler.pkl"
DEFAULT_GP_CONFIG_PATH = CODE_DIR / "gp_config.json"
DEFAULT_MODEL_FILES = [
    "bayesian_linear_regression.joblib",
    "random_forest_regression.joblib",
    "random_forest_regression_with_tree_prediction_variance.joblib",
    "single_mlp_regression.joblib",
    "mlp_deep_ensemble.joblib",
]
SENSOR_NAMES = [
    "finger_r_sensor1",
    "finger_r_sensor2",
    "finger_r_sensor3",
    "finger_r_sensor4",
    "finger_r_sensor5",
    "finger_l_sensor1",
    "finger_l_sensor2",
    "finger_l_sensor3",
    "finger_l_sensor4",
    "finger_l_sensor5",
]
CSV_FIELDS = [
    "timestamp_ns",
    "message_count",
    "sensor_name",
    "model_name",
    "raw_values",
    "top_k_features",
    "predicted_force_N",
    "estimated_uncertainty_std",
]


def predict_model(
    model: Any,
    x_top: np.ndarray,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Match the prediction behavior in Code/inference_saved_models.py."""
    if model_name == "GP Regression":
        mean, std = model.predict(x_top, return_std=True)
        return mean, std

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
            return tree_preds.mean(axis=0), tree_preds.std(axis=0, ddof=1)

    mean = model.predict(x_top)
    return mean, np.full(shape=mean.shape, fill_value=np.nan, dtype=float)


def extract_top_k_sensors(
    x_raw: np.ndarray,
    k: int,
    sort_values: bool,
) -> np.ndarray:
    if x_raw.ndim != 2:
        raise ValueError(f"Expected a 2D sensor array, got shape {x_raw.shape}")
    if x_raw.shape[1] < k:
        raise ValueError(f"Need at least {k} sensor values, got shape {x_raw.shape}")

    top_k = np.partition(x_raw, -k, axis=1)[:, -k:]
    if sort_values:
        top_k = np.sort(top_k, axis=1)[:, ::-1]
    return top_k.astype(float)


def load_models(model_paths: list[Path]) -> list[dict[str, Any]]:
    # Custom estimator classes in saved bundles were defined in Code/force_common.py.
    sys.path.insert(0, str(CODE_DIR))

    bundles: list[dict[str, Any]] = []
    for model_path in model_paths:
        if not model_path.exists():
            raise FileNotFoundError(f"Saved model not found: {model_path}")

        bundle = joblib.load(model_path)
        bundle["model_path"] = model_path
        bundles.append(bundle)
    return bundles


def load_gp_bundle(
    model_path: Path,
    scaler_path: Path,
    config_path: Path = DEFAULT_GP_CONFIG_PATH,
) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Saved GP model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Saved GP scaler not found: {scaler_path}")

    config = {"k": 5, "sort_top_k": False}
    if config_path.exists():
        config.update(json.loads(config_path.read_text(encoding="utf-8")))

    return {
        "model": joblib.load(model_path),
        "scaler": joblib.load(scaler_path),
        "model_name": "GP Regression",
        "model_path": model_path,
        "k": int(config["k"]),
        "sort_top_k": bool(config["sort_top_k"]),
    }


def override_bundle_k(bundles: list[dict[str, Any]], k: int | None) -> None:
    if k is None:
        return
    for bundle in bundles:
        bundle["k"] = k


class RealtimeTactileInference(Node):
    def __init__(
        self,
        topic: str,
        sensor_name: str,
        bundles: list[dict[str, Any]],
        print_every: int,
        rate_hz: float,
        status_interval: float,
        output_csv: Path | None,
    ) -> None:
        super().__init__("realtime_tactile_inference")
        self.sensor_name = sensor_name
        self.bundles = bundles
        self.print_every = print_every
        self.message_count = 0
        self.sample_count = 0
        self.prediction_count = 0
        self.inference_count = 0
        self.last_inferred_sample_count = 0
        self.latest_raw_values: np.ndarray | None = None
        self.latest_timestamp_ns = 0
        self.csv_file = None
        self.csv_writer = None

        if output_csv is not None:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            self.csv_file = output_csv.open("a", newline="", encoding="utf-8")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=CSV_FIELDS)
            if output_csv.stat().st_size == 0:
                self.csv_writer.writeheader()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.subscription = self.create_subscription(
            DynamicJointState,
            topic,
            self.listener_callback,
            qos,
        )
        self.inference_timer = self.create_timer(1.0 / rate_hz, self.run_inference)
        self.status_timer = self.create_timer(status_interval, self.log_status)

        model_names = ", ".join(bundle["model_name"] for bundle in bundles)
        self.get_logger().info(f"Listening: topic={topic}, sensor={sensor_name}, inference_rate={rate_hz:g}Hz")
        self.get_logger().info(f"Loaded model(s): {model_names}")
        if output_csv is not None:
            self.get_logger().info(f"Appending inference rows to: {output_csv.resolve()}")

    def listener_callback(self, msg: DynamicJointState) -> None:
        self.message_count += 1

        if self.sensor_name not in msg.joint_names:
            self.get_logger().warning(
                f"Sensor '{self.sensor_name}' is absent. Available names: {list(msg.joint_names)}",
                throttle_duration_sec=5.0,
            )
            return

        sensor_index = msg.joint_names.index(self.sensor_name)
        self.latest_raw_values = np.asarray(msg.interface_values[sensor_index].values, dtype=float)
        self.latest_timestamp_ns = self.get_clock().now().nanoseconds
        self.sample_count += 1

    def run_inference(self) -> None:
        if self.latest_raw_values is None or self.sample_count == self.last_inferred_sample_count:
            return

        raw_values = self.latest_raw_values
        x_raw = raw_values.reshape(1, -1)
        timestamp_ns = self.latest_timestamp_ns
        self.last_inferred_sample_count = self.sample_count
        self.inference_count += 1

        for bundle in self.bundles:
            model_name = bundle.get("model_name", bundle["model_path"].stem)
            k = int(bundle.get("k", 5))
            sort_top_k = bool(bundle.get("sort_top_k", False))

            try:
                x_top = extract_top_k_sensors(x_raw, k=k, sort_values=sort_top_k)
                if model_name == "GP Regression":
                    x_top = bundle["scaler"].transform(x_top)
                mean, std = predict_model(bundle["model"], x_top, model_name)
            except Exception as exc:
                self.get_logger().error(
                    f"Inference failed for {model_name}: {type(exc).__name__}: {exc}",
                    throttle_duration_sec=5.0,
                )
                continue

            predicted_force = float(mean[0])
            uncertainty = float(std[0]) if np.isfinite(std[0]) else np.nan
            self.prediction_count += 1

            if self.inference_count % self.print_every == 0:
                uncertainty_text = f", std={uncertainty:.3f}N" if np.isfinite(uncertainty) else ""
                self.get_logger().info(
                    f"#{self.message_count} {self.sensor_name} [{model_name}]: "
                    f"force={predicted_force:.3f}N{uncertainty_text}, raw={raw_values.tolist()}"
                )

            if self.csv_writer is not None:
                self.csv_writer.writerow(
                    {
                        "timestamp_ns": timestamp_ns,
                        "message_count": self.message_count,
                        "sensor_name": self.sensor_name,
                        "model_name": model_name,
                        "raw_values": json.dumps(raw_values.tolist()),
                        "top_k_features": json.dumps(x_top[0].tolist()),
                        "predicted_force_N": predicted_force,
                        "estimated_uncertainty_std": uncertainty,
                    }
                )
                self.csv_file.flush()

    def log_status(self) -> None:
        if self.message_count == 0:
            self.get_logger().warning("Waiting for /dynamic_joint_states messages. No data received yet.")
            return
        self.get_logger().info(
            f"Status: received_messages={self.message_count}, predictions={self.prediction_count}"
        )

    def close(self) -> None:
        if self.csv_file is not None:
            self.csv_file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time force inference from tactile ROS 2 data.")
    parser.add_argument("--topic", default="/dynamic_joint_states")
    parser.add_argument("--sensor-index", type=int, choices=range(1, 11), default=1)
    parser.add_argument(
        "--sensor-name",
        default="finger_l_sensor1",
        help="Override --sensor-index with an exact sensor joint name.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Use one saved model bundle. Default: Bayesian linear regression.",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument(
        "--gp-only",
        action="store_true",
        help="Run inference with only Code/gp_model.pkl and Code/scaler.pkl.",
    )
    parser.add_argument("--gp-model-path", type=Path, default=DEFAULT_GP_MODEL_PATH)
    parser.add_argument("--gp-scaler-path", type=Path, default=DEFAULT_GP_SCALER_PATH)
    parser.add_argument("--gp-config-path", type=Path, default=DEFAULT_GP_CONFIG_PATH)
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Override the saved top-k feature count for every loaded model.",
    )
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--status-interval", type=float, default=5.0)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    if args.print_every < 1:
        parser.error("--print-every must be at least 1")
    if args.status_interval <= 0:
        parser.error("--status-interval must be positive")
    if args.rate_hz <= 0:
        parser.error("--rate-hz must be positive")
    if args.k is not None and not 1 <= args.k <= 9:
        parser.error("--k must be between 1 and 9")
    selected_modes = sum([args.model_path is not None, args.all_models, args.gp_only])
    if selected_modes > 1:
        parser.error("Use only one of --model-path, --all-models, or --gp-only")
    return args


def model_paths_from_args(args: argparse.Namespace) -> list[Path]:
    if args.gp_only:
        return []
    if args.model_path is not None:
        return [args.model_path]
    if args.all_models:
        return [args.model_dir / filename for filename in DEFAULT_MODEL_FILES]
    return [args.model_dir / "bayesian_linear_regression.joblib"]


def main() -> None:
    args = parse_args()
    sensor_name = args.sensor_name or SENSOR_NAMES[args.sensor_index - 1]
    bundles = load_models(model_paths_from_args(args))
    if args.all_models or args.gp_only:
        bundles.append(load_gp_bundle(args.gp_model_path, args.gp_scaler_path, args.gp_config_path))
    override_bundle_k(bundles, args.k)

    rclpy.init()
    node = RealtimeTactileInference(
        topic=args.topic,
        sensor_name=sensor_name,
        bundles=bundles,
        print_every=args.print_every,
        rate_hz=args.rate_hz,
        status_interval=args.status_interval,
        output_csv=args.output_csv,
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
