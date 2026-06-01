"""
Load hold-out experiment models for a selected Top-K and show web inference.

Create experiment models first:
    python3 06_holdout_row_model_comparison.py --k 3 --test-index 0

Run the dashboard:
    source /opt/ros/humble/setup.bash
    python3 08_experiment_model_dashboard.py --k 3
    python3 08_experiment_model_dashboard.py --k 3 --demo
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import threading
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DASHBOARD_SCRIPT = SCRIPT_DIR / "05_realtime_tactile_dashboard.py"
DEFAULT_EXPERIMENT_ROOT = SCRIPT_DIR / "experiment_model"


def load_dashboard_module() -> Any:
    spec = importlib.util.spec_from_file_location("realtime_tactile_dashboard", DASHBOARD_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper module: {DASHBOARD_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dashboard = load_dashboard_module()


def load_experiment_bundles(experiment_root: Path, k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    experiment_dir = experiment_root / f"k_{k}"
    metadata_path = experiment_dir / "experiment_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Experiment models not found for k={k}: {experiment_dir}\n"
            f"Run: python3 06_holdout_row_model_comparison.py --k {k}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_paths = [
        experiment_dir / filename
        for filename in dashboard.realtime.DEFAULT_MODEL_FILES
    ]
    bundles = dashboard.realtime.load_models(model_paths)
    bundles.append(
        dashboard.realtime.load_gp_bundle(
            experiment_dir / "gp_model.pkl",
            experiment_dir / "scaler.pkl",
            experiment_dir / "gp_config.json",
        )
    )
    return bundles, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show real-time web inference with hold-out experiment models."
    )
    parser.add_argument("--k", type=int, required=True, help="Experiment Top-K model folder to load.")
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--topic", default="/dynamic_joint_states")
    parser.add_argument("--sensor-name", default="finger_l_sensor1")
    parser.add_argument("--rate-hz", type=float, default=5.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--demo", action="store_true", help="Preview with a saved test npz.")
    args = parser.parse_args()
    if not 1 <= args.k <= 9:
        parser.error("--k must be between 1 and 9")
    if args.rate_hz <= 0:
        parser.error("--rate-hz must be positive")
    return args


def main() -> None:
    args = parse_args()
    bundles, metadata = load_experiment_bundles(args.experiment_root, args.k)
    state = dashboard.DashboardState(
        sensor_name=args.sensor_name,
        rate_hz=args.rate_hz,
        top_k=args.k,
    )
    app = dashboard.create_app(state)
    web_thread = threading.Thread(
        target=app.run,
        kwargs={"host": args.host, "port": args.port, "debug": False, "use_reloader": False},
        daemon=True,
    )
    web_thread.start()
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(
        f"Loaded experiment models: k={args.k}, "
        f"held-out row={metadata['test_index']}"
    )

    if args.demo:
        try:
            dashboard.run_demo(state, bundles, args.rate_hz)
        except KeyboardInterrupt:
            pass
        return

    dashboard.realtime.rclpy.init()
    node = dashboard.DashboardNode(
        args.topic,
        args.sensor_name,
        bundles,
        state,
        args.rate_hz,
    )
    try:
        dashboard.realtime.rclpy.spin(node)
    except (KeyboardInterrupt, dashboard.realtime.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if dashboard.realtime.rclpy.ok():
            dashboard.realtime.rclpy.shutdown()


if __name__ == "__main__":
    main()
