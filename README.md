# Tactile Force Regression

3x3 tactile sensor 값으로 접촉 force를 추정하는 회귀 모델 모음입니다.

## Demo

[실시간 tactile force 대시보드 영상 보기](./tactile_force_dashboard_demo.mp4)

[실시간 tactile force 대시보드 screencast 보기](./tactile_force_dashboard_screencast.webm)

지원 모델:

| 모델 | 저장 파일 | Uncertainty |
|---|---|---|
| Bayesian Linear Regression | `Code/saved_models/bayesian_linear_regression.joblib` | 지원 |
| Random Forest Regression | `Code/saved_models/random_forest_regression.joblib` | 미지원 |
| Random Forest Tree Variance | `Code/saved_models/random_forest_regression_with_tree_prediction_variance.joblib` | 지원 |
| Single MLP Regression | `Code/saved_models/single_mlp_regression.joblib` | 미지원 |
| MLP Deep Ensemble | `Code/saved_models/mlp_deep_ensemble.joblib` | 지원 |
| GP Regression | `Code/gp_model.pkl`, `Code/scaler.pkl`, `Code/gp_config.json` | 지원 |

## 1. 환경 준비

프로젝트 폴더로 이동하고 ROS 2 환경을 불러옵니다.

```bash
cd /home/adfa5456/Desktop/YC_ws/ri_motion_0512/project/TBD_2026_BYP
source /opt/ros/humble/setup.bash
```

이 문서의 실행 예시는 현재 활성화된 Python 3 환경을 사용합니다.

```bash
python3
```

## 2. 학습 데이터

학습 데이터는 `Tactile_sensor_training_data/` 폴더에 둡니다.

```text
Tactile_sensor_training_data/
  tac_finger_l_sensor1_0N.npz
  tac_finger_l_sensor1_5N.npz
  tac_finger_l_sensor1_7N.npz
  tac_finger_l_sensor1_10N.npz
  ...
```

파일명 끝의 `0N`, `5N`, `10N` 값을 force label로 사용합니다. 각 `.npz` 파일의 `arr_0` 배열은 다음 shape이어야 합니다.

```text
(샘플 수, 9)
```

한 행은 3x3 tactile sensor의 raw 값 9개입니다.

```text
[sensor_0, sensor_1, ..., sensor_8]
```

모델은 각 행에서 값이 큰 `k`개 센서만 feature로 사용합니다. 예를 들어 `--k 3`이면 Top-3 값만 사용합니다.

### 데이터 수집

ROS tactile 데이터를 직접 수집하려면:

```bash
python3 01_sub_tactile.py
```

실행 후 입력:

| 입력 | 의미 |
|---|---|
| `1` ~ `10` | 관측할 tactile sensor 선택 |
| `f0`, `f5`, `f10` | 현재 force label 선택 |
| `a` | 현재 tactile 값을 한 번 저장 |
| `s` | 수집한 값을 `.npz` 파일로 저장 |

학습에 사용할 파일은 `Tactile_sensor_training_data/` 폴더로 이동합니다.

무접촉 상태를 정확히 예측하려면 `0N` 데이터가 반드시 필요합니다.

## 3. 전체 모델 학습

권장 설정인 Top-3로 6개 모델을 모두 학습합니다.

```bash
python3 \
  Code/run_all_models.py \
  --skip_predict \
  --k 3
```

이 명령은 기존 5개 baseline 모델과 GP Regression을 순서대로 학습합니다.

다른 `k`를 시험하려면 숫자만 변경합니다.

```bash
python3 \
  Code/run_all_models.py \
  --skip_predict \
  --k 5
```

학습 데이터 폴더를 직접 지정할 수도 있습니다.

```bash
python3 \
  Code/run_all_models.py \
  --data_dir Tactile_sensor_training_data \
  --skip_predict \
  --k 3
```

### GP만 학습

GP Regression만 다시 학습하려면:

```bash
python3 \
  Code/gp_regression_tactile.py \
  --mode train_all \
  --data_dir Tactile_sensor_training_data \
  --k 3
```

GP 학습 결과:

```text
Code/gp_model.pkl
Code/scaler.pkl
Code/gp_config.json
```

`gp_config.json`에는 학습에 사용한 `k` 값이 저장됩니다.

### GP Notebook

실험 과정과 그래프를 보려면 `Code/gp_regression_tactile.ipynb`를 사용합니다.

첫 번째 코드 셀에서 값을 변경합니다.

```python
K = 3
```

위에서부터 순서대로 실행하면 GP 모델, scaler, config가 저장됩니다. 반복 학습과 자동 실행에는 Python CLI 사용을 권장합니다.

## 4. 저장 모델 파일 Inference

GP 모델을 저장된 test `.npz` 파일에 적용하려면:

```bash
python3 \
  Code/gp_regression_tactile.py \
  --mode predict \
  --input_npz Tactile_sensor_test_data/tac_finger_r_sensor1_10N.npz
```

저장된 `gp_config.json`의 `k`를 자동으로 사용합니다.

## 5. ROS 실시간 Inference

### 전체 모델

6개 모델을 모두 실행하고 5Hz로 출력합니다.

```bash
python3 \
  04_realtime_tactile_inference.py \
  --all-models \
  --rate-hz 5
```

학습 모델에 저장된 `k` 값을 자동으로 사용합니다. 모든 모델의 `k`를 강제로 덮어쓰려면:

```bash
python3 \
  04_realtime_tactile_inference.py \
  --all-models \
  --rate-hz 5 \
  --k 3
```

### GP만 실행

```bash
python3 \
  04_realtime_tactile_inference.py \
  --gp-only \
  --rate-hz 5
```

### Sensor 선택

기본 sensor는 `finger_l_sensor1`입니다. 다른 sensor를 사용하려면:

```bash
python3 \
  04_realtime_tactile_inference.py \
  --all-models \
  --sensor-name finger_r_sensor1 \
  --rate-hz 5
```

### CSV 저장

```bash
python3 \
  04_realtime_tactile_inference.py \
  --all-models \
  --rate-hz 5 \
  --output-csv Code/results/realtime_all_models.csv
```

## 6. 웹 대시보드

6개 모델의 force 값을 2x3 카드 화면으로 확인합니다.

```bash
python3 \
  05_realtime_tactile_dashboard.py \
  --sensor-name finger_l_sensor1 \
  --rate-hz 5
```

브라우저에서 접속합니다.

```text
http://127.0.0.1:8050
```

모든 모델의 Top-K 값을 강제로 지정하려면:

```bash
python3 \
  05_realtime_tactile_dashboard.py \
  --sensor-name finger_l_sensor1 \
  --rate-hz 5 \
  --k 3
```

ROS 없이 저장된 test 데이터로 화면만 확인하려면:

```bash
python3 \
  05_realtime_tactile_dashboard.py \
  --demo \
  --rate-hz 5
```

## 7. 행 단위 Hold-out 평가

각 force `.npz` 파일에는 보통 10개 행이 있습니다. `06_holdout_row_model_comparison.py`는 각 파일에서 동일한 인덱스의 행 하나를 test로 제외하고, 나머지 9개 행만으로 6개 모델을 학습합니다.

현재 8개 force 파일이 있다면:

```text
학습: 8 force x 9행 = 72행
테스트: 8 force x 1행 = 8행
```

기본값은 각 파일의 첫 번째 행인 `--test-index 0`을 test로 사용합니다.

```bash
python3 \
  06_holdout_row_model_comparison.py \
  --k 3 \
  --test-index 0
```

다른 행을 test로 사용하려면:

```bash
python3 \
  06_holdout_row_model_comparison.py \
  --k 3 \
  --test-index 4
```

결과는 다음 폴더에 저장됩니다.

```text
Code/results/holdout/
```

또한 hold-out 학습 모델은 선택한 `k`별 폴더에 저장됩니다.

```text
experiment_model/k_3/
```

이 폴더에는 6개 모델 파일과 `experiment_metadata.json`이 포함됩니다. metadata에는 학습에 사용한 `k`와 제외한 test 행 인덱스가 기록됩니다. production 모델 파일은 덮어쓰지 않습니다.

## 8. Top-K 시각화

`07_visualize_k_holdout_comparison.py`는 `k=1~9`에 대해 위 hold-out 실험을 반복하고 모델별 MAE, RMSE를 그래프로 저장합니다. 각 `k`로 학습한 모델도 `experiment_model/k_1/`부터 `experiment_model/k_9/`까지 함께 저장합니다.

```bash
python3 07_visualize_k_holdout_comparison.py
```

특정 `k`만 비교하려면:

```bash
python3 \
  07_visualize_k_holdout_comparison.py \
  --k-values 1 3 5 7 9 \
  --test-index 0
```

출력 파일:

```text
Code/results/holdout/holdout_row_0_k_comparison_summary.csv
Code/results/holdout/holdout_row_0_k_comparison_predictions.csv
Code/results/holdout/holdout_row_0_k_comparison_metrics.png
Code/results/holdout/holdout_row_0_k_comparison_rmse_heatmap.png
Code/results/holdout/holdout_row_0_k_comparison_force_metrics.csv
Code/results/holdout/holdout_row_0_k_comparison_force_mse.png
Code/results/holdout/holdout_row_0_k_comparison_force_rmse.png
Code/results/holdout/holdout_row_0_k_comparison_force_uncertainty.png
Code/results/holdout/holdout_row_0_k_comparison_best_k_force_metrics.csv
Code/results/holdout/holdout_row_0_k_comparison_best_k_force_mse.png
Code/results/holdout/holdout_row_0_k_comparison_best_k_force_rmse.png
Code/results/holdout/holdout_row_0_k_comparison_best_k_force_uncertainty.png
```

force별 그래프는 모델별 2x3 화면으로 구성되며, 각 패널에서 `k`에 따른 차이를 확인할 수 있습니다. 현재 hold-out 방식은 force별 test 행이 하나이므로 각 `k`, 모델, force 조합에서 `RMSE`는 absolute error와 같고 `MSE`는 squared error입니다.

`best_k_force_*` 파일은 전체 hold-out RMSE가 가장 낮은 `k`를 모델마다 따로 선택합니다. 예를 들어 GP는 `k=3`, Random Forest는 `k=2`처럼 서로 다른 `k`를 사용한 상태에서 모델별 force 오차와 uncertainty를 한 그래프에서 비교합니다.

## 9. 실험 모델 웹 대시보드

`08_experiment_model_dashboard.py`는 production 모델이 아니라 `experiment_model/` 아래의 hold-out 학습 모델을 읽습니다. `--k`로 불러올 폴더를 선택합니다.

```bash
python3 \
  08_experiment_model_dashboard.py \
  --k 3 \
  --sensor-name finger_l_sensor1 \
  --rate-hz 5
```

브라우저에서 접속합니다.

```text
http://127.0.0.1:8050
```

ROS 없이 저장된 test 데이터로 화면만 확인하려면:

```bash
python3 \
  08_experiment_model_dashboard.py \
  --k 3 \
  --demo \
  --rate-hz 5
```

다른 `--test-index`로 `06` 또는 `07`을 다시 실행하면 해당 `experiment_model/k_*` 폴더가 새 실험 모델로 갱신됩니다.

## 10. 주요 파일

| 파일 | 역할 |
|---|---|
| `01_sub_tactile.py` | ROS tactile 데이터 수집 |
| `04_realtime_tactile_inference.py` | ROS 실시간 터미널 inference |
| `05_realtime_tactile_dashboard.py` | ROS 실시간 웹 대시보드 |
| `06_holdout_row_model_comparison.py` | force별 1행 hold-out 모델 비교 및 단일 `k` 실험 모델 저장 |
| `07_visualize_k_holdout_comparison.py` | `k=1~9` hold-out 성능 시각화 및 실험 모델 일괄 저장 |
| `08_experiment_model_dashboard.py` | 선택한 `k`의 hold-out 실험 모델 웹 대시보드 |
| `Code/run_all_models.py` | 6개 모델 전체 학습 |
| `Code/gp_regression_tactile.py` | GP 학습 및 `.npz` inference |
| `Code/gp_regression_tactile.ipynb` | GP 실험 및 시각화 notebook |
| `Code/force_common.py` | 공통 데이터 로딩, feature 추출, 평가 함수 |

## 11. 주의사항

### 학습과 inference의 `k` 값

원칙적으로 학습과 inference에서 같은 `k`를 사용해야 합니다. 저장 모델에는 학습 당시 `k` 값이 기록되므로 보통 `--k` override가 필요하지 않습니다.

다른 `k`를 비교 실험할 때만 실시간 inference의 `--k` 옵션을 사용합니다.

### Sensor별 분포 차이

왼손 sensor로 학습하고 오른손 sensor에 적용하면 오차가 증가할 수 있습니다. 정확도가 중요하면 실제 사용할 sensor별로 학습 데이터를 수집합니다.

### 고하중 구간

raw tactile 값이 `255`에 도달하면 sensor가 포화된 상태입니다. 이 구간에서는 정확한 force 회귀가 어렵습니다. 특히 `20N` 부근에서는 수치를 절대값으로 신뢰하기보다 고하중 상태로 해석해야 합니다.

### 무접촉 상태

`0N` 학습 데이터가 없으면 무접촉 상태에서도 모델이 가장 낮은 학습 label을 예측합니다. `0N` 데이터를 반드시 포함합니다.
