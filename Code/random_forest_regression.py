"""
# 설명
- Random Forest Regression 학습/평가/저장을 위한 코드
- StandardScaler + RandomForestRegressor 사용
- 출력: 예측 Force 평균값만 사용하며 uncertainty는 제공하지 않음

# 실행 명령어
python Code/random_forest_regression.py --mode train_all --data_dir Data/Tactile_sensor_test_data
python Code/random_forest_regression.py --mode train_eval --data_dir Data/Tactile_sensor_test_data
python Code/random_forest_regression.py --mode predict --input_npz Data/Tactile_sensor_test_data/tac_finger_r_sensor1_10N.npz
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from force_common import nan_std_like, run_model_cli


MODEL_NAME = "Random Forest Regression"


def build_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_leaf=1,
                    random_state=random_state,
                    bootstrap=True,
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
        description="Random Forest point regression for tactile force estimation.",
    )
