"""
# 설명
- Bayesian Linear Regression 학습/평가/저장을 위한 코드
- StandardScaler + BayesianRidge 사용
- 출력: 예측 Force 평균값 + Bayesian predictive uncertainty(std)

# 실행 명령어
python Code/bayesian_linear_regression.py --mode train_all --data_dir Data/Tactile_sensor_test_data
python Code/bayesian_linear_regression.py --mode train_eval --data_dir Data/Tactile_sensor_test_data
python Code/bayesian_linear_regression.py --mode predict --input_npz Data/Tactile_sensor_test_data/tac_finger_r_sensor1_10N.npz
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from force_common import run_model_cli


MODEL_NAME = "Bayesian Linear Regression"


def build_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", BayesianRidge()),
        ]
    )


def predict_with_uncertainty(
    model: Pipeline,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_scaled = model.named_steps["scaler"].transform(x_test)
    regressor = model.named_steps["regressor"]
    mean, std = regressor.predict(x_scaled, return_std=True)
    return mean, std


if __name__ == "__main__":
    run_model_cli(
        model_name=MODEL_NAME,
        build_model=build_model,
        predict_with_uncertainty=predict_with_uncertainty,
        description="Bayesian Linear Regression for tactile force estimation.",
    )
