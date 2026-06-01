"""
Show six tactile force regression models in a local real-time dashboard.

Examples:
    source /opt/ros/humble/setup.bash
    python3 05_realtime_tactile_dashboard.py
    python3 05_realtime_tactile_dashboard.py --sensor-name finger_r_sensor1 --rate-hz 5
    python3 05_realtime_tactile_dashboard.py --demo
"""

from __future__ import annotations

import argparse
import importlib.util
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template_string


SCRIPT_DIR = Path(__file__).resolve().parent
REALTIME_SCRIPT = SCRIPT_DIR / "04_realtime_tactile_inference.py"
DEMO_NPZ = SCRIPT_DIR / "Tactile_sensor_test_data" / "tac_finger_r_sensor1_10N.npz"

warnings.filterwarnings("ignore", message="Trying to unpickle estimator")


def load_realtime_module() -> Any:
    spec = importlib.util.spec_from_file_location("realtime_tactile_inference", REALTIME_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper module: {REALTIME_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


realtime = load_realtime_module()


class DashboardState:
    def __init__(self, sensor_name: str, rate_hz: float, top_k: int | str = "saved") -> None:
        self.lock = threading.Lock()
        self.data: dict[str, Any] = {
            "connected": False,
            "sensor_name": sensor_name,
            "rate_hz": rate_hz,
            "top_k": top_k,
            "message_count": 0,
            "inference_count": 0,
            "updated_at": None,
            "raw_values": [],
            "models": [],
        }

    def update(self, **values: Any) -> None:
        with self.lock:
            self.data.update(values)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.data)


def load_all_bundles(k: int | None = None) -> list[dict[str, Any]]:
    paths = [realtime.DEFAULT_MODEL_DIR / filename for filename in realtime.DEFAULT_MODEL_FILES]
    bundles = realtime.load_models(paths)
    bundles.append(
        realtime.load_gp_bundle(
            realtime.DEFAULT_GP_MODEL_PATH,
            realtime.DEFAULT_GP_SCALER_PATH,
        )
    )
    realtime.override_bundle_k(bundles, k)
    return bundles


def run_models(bundles: list[dict[str, Any]], raw_values: np.ndarray) -> list[dict[str, Any]]:
    x_raw = raw_values.reshape(1, -1)
    results: list[dict[str, Any]] = []

    for bundle in bundles:
        model_name = bundle.get("model_name", bundle["model_path"].stem)
        k = int(bundle.get("k", 5))
        sort_top_k = bool(bundle.get("sort_top_k", False))
        x_top = realtime.extract_top_k_sensors(x_raw, k=k, sort_values=sort_top_k)
        if model_name == "GP Regression":
            x_top = bundle["scaler"].transform(x_top)
        mean, std = realtime.predict_model(bundle["model"], x_top, model_name)
        uncertainty = float(std[0]) if np.isfinite(std[0]) else None
        results.append(
            {
                "name": model_name,
                "force": round(float(mean[0]), 3),
                "uncertainty": round(uncertainty, 3) if uncertainty is not None else None,
            }
        )
    return results


HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tactile Force Dashboard</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #07111f;
      --panel: rgba(16, 33, 54, 0.88);
      --line: rgba(148, 180, 214, 0.18);
      --text: #f1f7ff;
      --muted: #91a6bd;
      --accent: #55d8c1;
      --warm: #ffb86b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(circle at 15% 5%, rgba(45, 104, 160, 0.34), transparent 32rem),
        radial-gradient(circle at 92% 90%, rgba(31, 139, 125, 0.20), transparent 34rem),
        var(--bg);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main { width: min(1480px, 94vw); margin: 0 auto; padding: 34px 0 30px; }
    header { display: flex; justify-content: space-between; gap: 20px; align-items: end; margin-bottom: 24px; }
    .eyebrow { color: var(--accent); font-size: 12px; font-weight: 800; letter-spacing: 0.2em; text-transform: uppercase; }
    h1 { margin: 7px 0 0; font-size: clamp(28px, 4vw, 46px); letter-spacing: -0.055em; }
    .status { display: flex; gap: 10px; align-items: center; color: var(--muted); font-size: 14px; }
    .dot { width: 10px; height: 10px; border-radius: 50%; background: #62758c; box-shadow: 0 0 0 4px rgba(98,117,140,.16); }
    .dot.live { background: var(--accent); box-shadow: 0 0 18px rgba(85,216,193,.8); }
    .cards { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 15px; }
    .card, .footer-panel {
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: 0 18px 60px rgba(0,0,0,.2);
      backdrop-filter: blur(14px);
    }
    .card { min-height: 222px; padding: 21px; border-radius: 20px; display: flex; flex-direction: column; justify-content: space-between; }
    .card:nth-child(6) { border-color: rgba(85,216,193,.48); background: linear-gradient(145deg, rgba(22,55,70,.95), rgba(14,39,56,.92)); }
    .model-name { min-height: 42px; color: #b8c8d8; font-size: 14px; font-weight: 750; line-height: 1.4; letter-spacing: .03em; text-transform: uppercase; }
    .reading { display: flex; align-items: baseline; gap: 7px; }
    .force { font-size: clamp(55px, 6vw, 82px); font-weight: 820; letter-spacing: -.085em; line-height: .96; }
    .unit { color: var(--accent); font-size: 22px; font-weight: 800; }
    .uncertainty { min-height: 22px; margin-top: 12px; color: var(--muted); font-size: 14px; }
    .uncertainty strong { color: var(--warm); font-weight: 750; }
    .footer-panel { margin-top: 16px; border-radius: 18px; padding: 17px 20px; display: grid; grid-template-columns: 1fr auto; gap: 18px; align-items: center; }
    .meta { display: flex; flex-wrap: wrap; gap: 15px 24px; color: var(--muted); font-size: 13px; }
    .meta strong { color: var(--text); font-weight: 720; }
    .raw { display: grid; grid-template-columns: repeat(9, 36px); gap: 5px; }
    .raw span { padding: 7px 2px; border-radius: 8px; background: rgba(4,14,27,.64); color: #b8e9df; font: 12px/1 ui-monospace, SFMono-Regular, Menlo, monospace; text-align: center; }
    @media (max-width: 900px) { .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); } .footer-panel { grid-template-columns: 1fr; } }
    @media (max-width: 560px) { header { display: block; } .status { margin-top: 14px; } .cards { grid-template-columns: 1fr; } .raw { grid-template-columns: repeat(9, minmax(0, 1fr)); } }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <div class="eyebrow">Robotic tactile sensing</div>
        <h1>Force Regression Monitor</h1>
      </div>
      <div class="status"><span id="dot" class="dot"></span><span id="connection">Waiting for tactile data</span></div>
    </header>
    <section id="cards" class="cards"></section>
    <section class="footer-panel">
      <div class="meta">
        <span>Sensor <strong id="sensor">-</strong></span>
        <span>Inference rate <strong id="rate">-</strong></span>
        <span>Top-K <strong id="top-k">-</strong></span>
        <span>Messages <strong id="messages">0</strong></span>
        <span>Updates <strong id="updates">0</strong></span>
        <span>Last update <strong id="updated">-</strong></span>
      </div>
      <div id="raw" class="raw"></div>
    </section>
  </main>
  <script>
    const modelNames = [
      "Bayesian Linear Regression",
      "Random Forest Regression",
      "Random Forest Regression with Tree Prediction Variance",
      "Single MLP Regression",
      "MLP Deep Ensemble",
      "GP Regression"
    ];
    const cards = document.getElementById("cards");
    cards.innerHTML = modelNames.map((name, index) => `
      <article class="card">
        <div class="model-name">${index + 1}. ${name}</div>
        <div>
          <div class="reading"><span class="force" data-name="${name}">--</span><span class="unit">N</span></div>
          <div class="uncertainty" data-std="${name}">Waiting for inference</div>
        </div>
      </article>`).join("");

    function render(state) {
      document.getElementById("dot").classList.toggle("live", state.connected);
      document.getElementById("connection").textContent = state.connected ? "Live tactile stream" : "Waiting for tactile data";
      document.getElementById("sensor").textContent = state.sensor_name;
      document.getElementById("rate").textContent = `${state.rate_hz} Hz`;
      document.getElementById("top-k").textContent = state.top_k;
      document.getElementById("messages").textContent = state.message_count;
      document.getElementById("updates").textContent = state.inference_count;
      document.getElementById("updated").textContent = state.updated_at || "-";
      document.getElementById("raw").innerHTML = (state.raw_values || []).map(v => `<span>${Math.round(v)}</span>`).join("");
      (state.models || []).forEach(model => {
        document.querySelector(`[data-name="${model.name}"]`).textContent = model.force.toFixed(2);
        document.querySelector(`[data-std="${model.name}"]`).innerHTML =
          model.uncertainty == null ? "Point estimate" : `Uncertainty <strong>± ${model.uncertainty.toFixed(2)} N</strong>`;
      });
    }
    async function refresh() {
      try { render(await (await fetch("/api/state")).json()); } catch (_) {}
    }
    refresh();
    setInterval(refresh, 200);
  </script>
</body>
</html>
"""


def create_app(state: DashboardState) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return render_template_string(HTML)

    @app.get("/api/state")
    def api_state() -> Any:
        return jsonify(state.snapshot())

    return app


class DashboardNode(realtime.Node):
    def __init__(
        self,
        topic: str,
        sensor_name: str,
        bundles: list[dict[str, Any]],
        state: DashboardState,
        rate_hz: float,
    ) -> None:
        super().__init__("realtime_tactile_dashboard")
        self.sensor_name = sensor_name
        self.bundles = bundles
        self.state = state
        self.message_count = 0
        self.sample_count = 0
        self.last_inferred_sample_count = 0
        self.latest_raw_values: np.ndarray | None = None

        qos = realtime.QoSProfile(
            reliability=realtime.ReliabilityPolicy.BEST_EFFORT,
            history=realtime.HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.subscription = self.create_subscription(
            realtime.DynamicJointState,
            topic,
            self.listener_callback,
            qos,
        )
        self.inference_timer = self.create_timer(1.0 / rate_hz, self.run_inference)
        self.get_logger().info(f"Dashboard listening: topic={topic}, sensor={sensor_name}, rate={rate_hz:g}Hz")

    def listener_callback(self, msg: Any) -> None:
        self.message_count += 1
        if self.sensor_name not in msg.joint_names:
            self.get_logger().warning(
                f"Sensor '{self.sensor_name}' is absent. Available names: {list(msg.joint_names)}",
                throttle_duration_sec=5.0,
            )
            return
        sensor_index = msg.joint_names.index(self.sensor_name)
        self.latest_raw_values = np.asarray(msg.interface_values[sensor_index].values, dtype=float)
        self.sample_count += 1

    def run_inference(self) -> None:
        if self.latest_raw_values is None or self.sample_count == self.last_inferred_sample_count:
            return
        self.last_inferred_sample_count = self.sample_count
        results = run_models(self.bundles, self.latest_raw_values)
        snapshot = self.state.snapshot()
        self.state.update(
            connected=True,
            message_count=self.message_count,
            inference_count=int(snapshot["inference_count"]) + 1,
            updated_at=time.strftime("%H:%M:%S"),
            raw_values=self.latest_raw_values.tolist(),
            models=results,
        )


def run_demo(state: DashboardState, bundles: list[dict[str, Any]], rate_hz: float) -> None:
    raw_rows = np.load(DEMO_NPZ)["arr_0"]
    index = 0
    while True:
        raw_values = raw_rows[index % len(raw_rows)]
        snapshot = state.snapshot()
        state.update(
            connected=True,
            message_count=int(snapshot["message_count"]) + 1,
            inference_count=int(snapshot["inference_count"]) + 1,
            updated_at=time.strftime("%H:%M:%S"),
            raw_values=raw_values.tolist(),
            models=run_models(bundles, raw_values),
        )
        index += 1
        time.sleep(1.0 / rate_hz)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show six tactile force models in a local web dashboard.")
    parser.add_argument("--topic", default="/dynamic_joint_states")
    parser.add_argument("--sensor-name", default="finger_l_sensor1")
    parser.add_argument("--rate-hz", type=float, default=5.0)
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Override the saved top-k feature count for every loaded model.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--demo", action="store_true", help="Preview the dashboard with a saved test npz.")
    args = parser.parse_args()
    if args.rate_hz <= 0:
        parser.error("--rate-hz must be positive")
    if args.k is not None and not 1 <= args.k <= 9:
        parser.error("--k must be between 1 and 9")
    return args


def main() -> None:
    args = parse_args()
    bundles = load_all_bundles(k=args.k)
    state = DashboardState(sensor_name=args.sensor_name, rate_hz=args.rate_hz, top_k=args.k or "saved")
    app = create_app(state)
    web_thread = threading.Thread(
        target=app.run,
        kwargs={"host": args.host, "port": args.port, "debug": False, "use_reloader": False},
        daemon=True,
    )
    web_thread.start()
    print(f"Dashboard: http://{args.host}:{args.port}")

    if args.demo:
        try:
            run_demo(state, bundles, args.rate_hz)
        except KeyboardInterrupt:
            pass
        return

    realtime.rclpy.init()
    node = DashboardNode(args.topic, args.sensor_name, bundles, state, args.rate_hz)
    try:
        realtime.rclpy.spin(node)
    except (KeyboardInterrupt, realtime.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if realtime.rclpy.ok():
            realtime.rclpy.shutdown()


if __name__ == "__main__":
    main()
