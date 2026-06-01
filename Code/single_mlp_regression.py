"""
# 설명
- Single MLP Regression 학습/평가/저장을 위한 코드
- StandardScaler + MLPRegressor 사용
- training loss가 충분히 줄어들고 개선이 멈추면 자동 종료
- 출력: 예측 Force 평균값만 사용하며 uncertainty는 제공하지 않음

# 실행 명령어
python Code/single_mlp_regression.py --mode train_all --data_dir Data/Tactile_sensor_test_data
python Code/single_mlp_regression.py --mode train_eval --data_dir Data/Tactile_sensor_test_data
python Code/single_mlp_regression.py --mode predict --input_npz Data/Tactile_sensor_test_data/tac_finger_r_sensor1_10N.npz
"""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from force_common import nan_std_like, run_model_cli


MODEL_NAME = "Single MLP Regression"


def build_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                MLPRegressor(
                    hidden_layer_sizes=(8,),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=1e-2,
                    alpha=1e-2,
                    max_iter=5000,
                    tol=1e-5,
                    n_iter_no_change=50,
                    random_state=random_state,
                ),
            ),
        ]
    )


def predict_with_uncertainty(
    model: Pipeline,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = model.predict(x_test)
    return mean, nan_std_like(mean)


if __name__ == "__main__":
    run_model_cli(
        model_name=MODEL_NAME,
        build_model=build_model,
        predict_with_uncertainty=predict_with_uncertainty,
        description="Single MLP regression for tactile force estimation.",
    )
