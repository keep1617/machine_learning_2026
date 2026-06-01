"""
# 설명
- Code 폴더 안의 모든 Baseline 모델을 한 번에 실행하는 코드
- 전체 데이터로 모델을 학습하고 저장한 뒤, 저장된 모델을 다시 불러와 inference 테스트 수행
- --skip_predict 옵션을 주면 학습/저장만 수행

# 실행 명령어
python3 Code/run_all_models.py
python3 Code/run_all_models.py --skip_predict
python3 Code/run_all_models.py --skip_predict --k 3
python3 Code/run_all_models.py --input_npz Tactile_sensor_test_data/tac_finger_r_sensor1_10N.npz
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    code_dir = Path(__file__).resolve().parent
    assignment_dir = code_dir.parent

    parser = argparse.ArgumentParser(description="Run all force regression baselines.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(assignment_dir / "Tactile_sensor_training_data"),
        help="Folder containing force-labeled .npz files.",
    )
    parser.add_argument(
        "--input_npz",
        type=str,
        default=str(assignment_dir / "Tactile_sensor_test_data" / "tac_finger_r_sensor1_10N.npz"),
        help="Input .npz used for saved-model inference test.",
    )
    parser.add_argument(
        "--skip_predict",
        action="store_true",
        help="Only train/save all models; do not run predict mode.",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of top sensor values.")
    parser.add_argument(
        "--sort_top_k",
        action="store_true",
        help="Sort top-k sensor values descending before training and prediction.",
    )
    args = parser.parse_args()

    scripts = [
        "bayesian_linear_regression.py",
        "random_forest_regression.py",
        "random_forest_tree_variance.py",
        "single_mlp_regression.py",
        "mlp_deep_ensemble.py",
        "gp_regression_tactile.py",
    ]

    for script in scripts:
        script_path = code_dir / script
        print(f"\n=== train_all: {script} ===", flush=True)
        command = [
            sys.executable,
            str(script_path),
            "--mode",
            "train_all",
            "--data_dir",
            args.data_dir,
            "--k",
            str(args.k),
        ]
        if args.sort_top_k:
            command.append("--sort_top_k")
        subprocess.run(
            command,
            cwd=assignment_dir,
            check=True,
        )

    if args.skip_predict:
        return

    for script in scripts:
        script_path = code_dir / script
        print(f"\n=== predict: {script} ===", flush=True)
        command = [
            sys.executable,
            str(script_path),
            "--mode",
            "predict",
            "--input_npz",
            args.input_npz,
        ]
        subprocess.run(
            command,
            cwd=assignment_dir,
            check=True,
        )


if __name__ == "__main__":
    main()
