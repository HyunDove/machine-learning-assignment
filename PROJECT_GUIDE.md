# 프로젝트 소스 가이드

> 이 파일은 ml_project의 전체 소스 구조와 각 파일의 역할을 설명합니다.

---

## 전체 구조 한눈에 보기

```
ml_project/
│
├── streamlit_app.py          ← 웹 UI (메인 진입점)
├── requirements.txt          ← Python 패키지 목록
├── packages.txt              ← OS 패키지 목록 (Streamlit Cloud용)
│
├── ml/                       ← ML 파이프라인 (전처리 → 학습 → 평가)
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── data/
│   ├── raw/                  ← 원본 CSV (Kaggle)
│   └── processed/            ← 전처리 완료 CSV
│
├── models/                   ← 학습된 모델 pkl 파일
│   ├── loan_rate_model.pkl       (363KB — GradientBoostingRegressor)
│   └── loan_status_model.pkl     (17MB  — RandomForestClassifier)
│
├── notebooks/                ← 분석 과정 Jupyter 노트북 4개
│
├── backend/                  ← Flask REST API
│   ├── main.py
│   └── app/
│       ├── routes/
│       ├── services/
│       ├── models/           ← Pydantic 스키마
│       └── utils/
│
└── reports/                  ← 과제 보고서 및 시각화 이미지
```

---

## 데이터 흐름

```
원본 CSV (Kaggle, 32,581행)
   ↓ ml/preprocessing.py
전처리 CSV
   ↓ ml/train.py
모델 pkl 파일
   ↓
   ├── streamlit_app.py  (웹 UI — Streamlit Cloud 배포)
   └── backend/          (REST API — 로컬 or 별도 서버)
```

---

## 파일별 상세 설명

---

### `ml/preprocessing.py` — 데이터 전처리

원본 CSV를 모델에 넣을 수 있는 형태로 가공합니다.

```python
def load_raw()         # data/raw/ 에서 원본 CSV 로드
def preprocess(df)     # 결측치·이상치 처리 + 범주형 인코딩
def save_processed(df) # data/processed/ 에 저장
```

**처리 순서:**

| 단계 | 내용 |
|------|------|
| 결측치 제거 | `loan_int_rate`(예측 대상)가 NaN인 행 삭제 |
| 결측치 대체 | `person_emp_length`(재직기간) NaN → 중앙값으로 채움 |
| 이상치 제거 | 나이 > 100, 재직기간 > 60인 행 제거 |
| 범주형 인코딩 | 문자열 컬럼 4개를 숫자로 변환 (LabelEncoder, 알파벳 순) |

인코딩 결과:

| 컬럼 | 매핑 |
|------|------|
| `person_home_ownership` | MORTGAGE=0, OTHER=1, OWN=2, RENT=3 |
| `loan_intent` | DEBTCONSOLIDATION=0, EDUCATION=1, HOMEIMPROVEMENT=2, MEDICAL=3, PERSONAL=4, VENTURE=5 |
| `loan_grade` | A=0, B=1, C=2, D=3, E=4, F=5, G=6 |
| `cb_person_default_on_file` | N=0, Y=1 |

> **중요**: 이 매핑은 `streamlit_app.py`와 `backend/prediction_service.py`에도 동일하게 하드코딩되어 있습니다. 셋이 불일치하면 예측값이 틀립니다.

---

### `ml/train.py` — 모델 학습

전처리된 데이터로 두 가지 모델을 학습하고 pkl 파일로 저장합니다.

```python
def train()  # 학습 실행 → (regressor, classifier, X_test, y_test) 반환
```

**학습하는 모델:**

| 구분 | 알고리즘 | 예측 대상 | 저장 경로 |
|------|---------|----------|----------|
| 회귀 | GradientBoostingRegressor (n_estimators=200) | `loan_int_rate` — 대출 금리(%) | `models/loan_rate_model.pkl` |
| 분류 | RandomForestClassifier (n_estimators=50) | `loan_status` — 정상/부도 여부 | `models/loan_status_model.pkl` |

**사용 피처 (11개):**

```
person_age, person_income, person_home_ownership, person_emp_length,
loan_intent, loan_grade, loan_amnt, loan_status,
loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length
```

> 분류 모델은 `loan_status` 자체를 예측하므로 해당 컬럼을 피처에서 제외(10개 사용).

---

### `ml/evaluate.py` — 모델 성능 평가

학습된 회귀 모델의 성능 지표를 계산합니다. (테스트셋 20% 기준)

| 지표 | 값 | 의미 |
|------|----|------|
| RMSE | 1.0079 | 예측 금리와 실제 금리의 평균 오차 (단위: %) |
| MAE | 0.7899 | 절대 오차 평균 |
| R² | 0.9041 | 분산의 90.4%를 설명 (1에 가까울수록 좋음) |

> 이 수치가 `streamlit_app.py`의 `MODEL_RESULTS` 딕셔너리에 하드코딩되어 있습니다. 재학습해도 자동 반영되지 않으므로 수동 갱신이 필요합니다.

---

### `streamlit_app.py` — 웹 UI

Streamlit으로 만든 웹 애플리케이션입니다.

#### 앱 초기화 흐름

```
앱 시작
  ↓ _set_korean_font()     # matplotlib 한글 폰트 설정
  ↓ load_models()          # 모델 pkl 로드 (캐시)
      └─ pkl 없으면 ensure_models() 자동 실행
             → preprocessing.py + train.py 순서로 실행 (최초 1회)
```

#### Tab 1 — 금리 예측

```
입력 폼 (11개 필드, 2열 그리드)
  ↓ loan_percent_income 자동 계산 (loan_amnt / person_income)
  ↓ [예측하기] 버튼 클릭
  ↓ predict(raw)
      ├── encode_input()              # 문자열 → 숫자 변환
      ├── regressor.predict()         # 금리(float) 반환
      └── classifier.predict_proba() # 부도 확률(0~1) 반환
  ↓ @st.dialog 모달로 결과 표시
```

`@st.dialog`는 Streamlit 1.36.0에서 추가된 오버레이 팝업입니다. 함수에 데코레이터로 선언한 뒤 호출하면 X 버튼이 달린 모달 창이 열립니다.

#### Tab 2 — 모델 성능

`MODEL_RESULTS` 딕셔너리(하드코딩)의 값으로 4개 모델 비교표와 RMSE/MAE/R² 막대그래프를 그립니다.

#### Tab 3 — 데이터 인사이트

`data/processed/credit_risk_cleaned.csv`를 직접 읽어 6개 변수(등급, 나이, 소득, 대출금액, 재직기간, 신용이력)별 평균 금리를 bar chart로 표시합니다.

#### 캐싱 전략

```python
@st.cache_resource  # 모델: 앱 전체에서 딱 1번만 로드 (재실행해도 유지)
def load_models(): ...

@st.cache_data      # 데이터: CSV 변경 전까지 캐시 (재실행 시 재로드 안 함)
def load_data(): ...
```

---

### `backend/` — Flask REST API

웹 UI 없이 API 형태로 예측 결과를 받을 때 사용합니다.

#### 엔드포인트

```
POST /api/predictions/
Content-Type: application/json
```

**요청 예시:**
```json
{
  "person_age": 30,
  "person_income": 60000,
  "person_home_ownership": "RENT",
  "person_emp_length": 3.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "C",
  "loan_amnt": 10000,
  "loan_status": 0,
  "loan_percent_income": 0.1667,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 5
}
```

**응답 예시:**
```json
{
  "loan_int_rate": 11.4200,
  "loan_status_prob": 0.1800,
  "loan_status_label": "정상"
}
```

#### 계층 구조

```
routes/prediction.py          # HTTP 수신·응답만 담당
   ↓
services/prediction_service.py  # 인코딩 + 예측 로직
   ↓
utils/model_loader.py           # 싱글톤 패턴으로 모델 1회만 로드
```

#### `backend/app/models/schemas.py` — Pydantic 스키마

요청 데이터의 타입과 허용값을 정의합니다. 잘못된 값이 오면 자동으로 400을 반환합니다.

```python
class PredictionRequest(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    # ... 나머지 11개 필드
```

#### `backend/app/errors.py` — 에러 처리

```python
class ApiError(Exception):
    def __init__(self, status: int, message: str): ...

# 사용 예
raise ApiError(400, "잘못된 등급입니다.")
# → {"error": "잘못된 등급입니다."} 400 응답 자동 반환
```

---

### `backend/tests/` — 테스트

pytest 기반 6개 테스트 케이스입니다.

| 테스트 함수 | 검증 내용 |
|------------|---------|
| `test_predict_success` | 정상 요청 → 200, 3개 응답 필드 존재 |
| `test_predict_missing_field` | 빈 JSON → 400 |
| `test_predict_invalid_grade` | 존재하지 않는 등급("Z") → 400 |
| `test_predict_invalid_home_ownership` | 잘못된 주거형태 → 400 |
| `test_predict_high_risk_profile` | 등급G + 연체 + 과거부도 → 금리 > 10% |
| `test_predict_low_risk_profile` | 등급A + 정상 + 신용양호 → 금리 < 15% |

```bash
# 실행
cd backend && pytest tests/ -v
```

---

### `notebooks/` — 분석 노트북

| 파일 | 내용 |
|------|------|
| `01_eda.ipynb` | 분포, 결측치, 상관관계 탐색 |
| `02_preprocessing.ipynb` | 전처리 과정 단계별 확인 |
| `03_modeling.ipynb` | 4개 모델 학습 및 성능 비교 실험 |
| `04_evaluation.ipynb` | 최종 모델 상세 평가 (ROC, Feature Importance 등) |

> 노트북에서 실험한 결과 수치가 `streamlit_app.py`의 `MODEL_RESULTS`에 하드코딩되어 있습니다.

---

### `requirements.txt` / `packages.txt` — 의존성

**`requirements.txt`** (Python 패키지):

| 패키지 | 버전 | 비고 |
|--------|------|------|
| streamlit | >=1.36.0 | @st.dialog 사용을 위해 1.36 이상 필요 |
| pandas | >=2.2.0 | |
| numpy | >=1.26.0 | |
| scikit-learn | ==1.4.0 | **버전 고정** — 다른 버전이면 pkl 로드 시 역직렬화 오류 발생 |
| joblib | >=1.3.0 | pkl 파일 저장/로드 |
| matplotlib | >=3.8.0 | 차트 시각화 |

**`packages.txt`** (Streamlit Cloud OS 패키지):

```
fonts-nanum    # Ubuntu 서버에 나눔 한글 폰트 설치
```

---

## 자주 헷갈리는 포인트

### 인코딩이 세 곳에 중복 정의됨

범주형 인코딩 매핑이 아래 세 파일에 각각 독립적으로 정의되어 있습니다:

- `ml/preprocessing.py` — LabelEncoder 학습 결과
- `streamlit_app.py` — `ENCODINGS` 딕셔너리
- `backend/app/services/prediction_service.py` — `_ENCODINGS` 딕셔너리

세 곳이 반드시 일치해야 합니다. 새 범주값이 생기면 세 파일 모두 수정해야 합니다.

### MODEL_RESULTS는 정적 값

`streamlit_app.py`의 `MODEL_RESULTS`는 노트북 실험 결과를 수동으로 복사한 값입니다. 모델을 재학습해도 자동으로 갱신되지 않습니다.

### 분류 모델의 피처 수가 다름

회귀 모델은 피처 11개, 분류 모델은 10개(`loan_status` 제외)를 사용합니다. `predict()` 함수에서 각각 별도의 DataFrame을 만드는 이유입니다.

```python
df_reg = encode_input(raw)[FEATURES]           # 11개
df_cls = encode_input(raw)[FEATURES_NO_STATUS] # 10개 (loan_status 제외)
```

### Streamlit Cloud 첫 실행

모델 pkl 파일이 git에 포함되어 있어 배포 후 재학습 없이 바로 예측이 가능합니다. `ensure_models()`는 pkl이 없을 때만 실행됩니다.
