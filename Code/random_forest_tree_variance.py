"""
# 설명
- Random Forest Regression with Tree Prediction Variance 학습/평가/저장을 위한 코드
- Random Forest의 각 tree 예측값 평균을 최종 예측값으로 사용
- 각 tree 예측값의 표준편차를 estimated uncertainty로 사용

# 실행 명령어
python Code/random_forest_tree_variance.py --mode train_all --data_dir Data/Tactile_sensor_test_data
python Code/random_forest_tree_variance.py --mode train_eval --data_dir Data/Tactile_sensor_test_data
python Code/random_forest_tree_variance.py --mode predict --input_npz Data/Tactile_sensor_test_data/tac_finger_r_sensor1_10N.npz
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from force_common import run_model_cli


MODEL_NAME = "Random Forest Regression with Tree Prediction Variance"


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
    x_scaled = model.named_steps["scaler"].transform(x_test)
    forest = model.named_steps["regressor"]
    tree_preds = np.vstack([tree.predict(x_scaled) for tree in forest.estimators_])
    mean = tree_preds.mean(axis=0)
    std = tree_preds.std(axis=0, ddof=1)
    return mean, std


if __name__ == "__main__":
    run_model_cli(
        model_name=MODEL_NAME,
        build_model=build_model,
        predict_with_uncertainty=predict_with_uncertainty,
        description="Random Forest regression with tree prediction variance.",
    )
