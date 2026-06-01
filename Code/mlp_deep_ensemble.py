"""
# 설명
- MLP Deep Ensemble 학습/평가/저장을 위한 코드
- 여러 개의 MLPRegressor를 bootstrap sampling으로 학습
- Ensemble 평균을 최종 예측값으로 사용
- Ensemble 멤버 간 예측값 표준편차를 estimated uncertainty로 사용

# 실행 명령어
python Code/mlp_deep_ensemble.py --mode train_all --data_dir Data/Tactile_sensor_test_data
python Code/mlp_deep_ensemble.py --mode train_eval --data_dir Data/Tactile_sensor_test_data
python Code/mlp_deep_ensemble.py --mode predict --input_npz Data/Tactile_sensor_test_data/tac_finger_r_sensor1_10N.npz
"""

from __future__ import annotations

import numpy as np

from force_common import MLPDeepEnsembleRegressor, run_model_cli


MODEL_NAME = "MLP Deep Ensemble"


def build_model(random_state: int = 42) -> MLPDeepEnsembleRegressor:
    return MLPDeepEnsembleRegressor(
        n_members=3,
        hidden_layer_sizes=(8,),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-2,
        alpha=1e-2,
        max_iter=5000,
        tol=1e-5,
        n_iter_no_change=50,
        bootstrap=True,
        random_state=random_state,
    )


def predict_with_uncertainty(
    model: MLPDeepEnsembleRegressor,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return model.predict_with_std(x_test)


if __name__ == "__main__":
    run_model_cli(
        model_name=MODEL_NAME,
        build_model=build_model,
        predict_with_uncertainty=predict_with_uncertainty,
        description="MLP deep ensemble regression for tactile force estimation.",
    )
