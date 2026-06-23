# 💳 대출 금리 예측 머신러닝 과제

> Kaggle **Credit Risk Dataset** 기반으로 대출 금리를 예측하는 회귀 모델을 구현한 머신러닝 과제입니다.

---

## 📋 과제 개요

| 항목 | 내용 |
|---|---|
| 🎯 **주제** | 대출 금리 예측 모델 구현 |
| 🔍 **문제 유형** | 회귀 (`loan_int_rate`) |
| 📦 **라이브러리** | Python · scikit-learn 1.4.0 · Flask · Streamlit · Plotly · Pydantic |
| 📊 **데이터** | Kaggle — Credit Risk Dataset (32,581행, 12컬럼) |
| 📏 **평가지표** | RMSE, MAE, R² |
| 🚀 **결과물** | Streamlit 웹앱 (Cloud 배포) + Flask REST API + Jupyter 노트북 4종 |

---

## 🌐 Streamlit 웹앱

**탭 구성:**

| 탭 | 내용 |
|---|---|
| 🔮 **금리 예측** | 신청자 정보 입력 → 예측 금리 모달 출력 (등급 내 위치 비교 + 전체 입력값 10개 vs 데이터 평균 비교) |
| 📊 **모델 성능** | 4개 회귀 모델 RMSE·MAE·R² 비교 표 + Plotly 막대 그래프 (다크/라이트 테마 자동 대응) |
| 📈 **데이터 인사이트** | 등급·나이·소득·대출금액 등 변수별 평균 금리 시각화 |

---

## 📁 프로젝트 구조

```
ml_project/
│
├── 📄 credit_risk_dataset.csv      # 원본 데이터 (Kaggle)
├── 📄 streamlit_app.py             # Streamlit 웹앱 (3탭 구성)
├── 📄 requirements.txt             # Python 의존성 (scikit-learn==1.4.0 고정)
├── 📄 packages.txt                 # Streamlit Cloud OS 패키지 (fonts-nanum 한글 폰트)
│
├── 📂 data/
│   └── processed/
│       └── credit_risk_cleaned.csv # 전처리 완료 CSV (git 추적)
│
├── 📂 ml/                          # ML 파이프라인
│   ├── preprocessing.py            # 결측치·이상치 처리, LabelEncoding
│   ├── train.py                    # 모델 학습 (RF 회귀, RandomizedSearchCV 튜닝)
│   └── evaluate.py                 # RMSE, MAE, R² 평가
│
├── 📂 models/                      # 학습된 모델 파일 (로컬 시연용 — git 추적 제외)
│   └── loan_rate_model.pkl         # RandomForestRegressor n_estimators=200 (compress=3)
│
├── 📂 notebooks/                   # Jupyter 노트북 (순서대로 실행)
│   ├── 01_eda.ipynb                # EDA — 결측치·분포·상관관계 탐색
│   ├── 02_preprocessing.ipynb      # 전처리 단계별 확인 및 CSV 저장
│   ├── 03_modeling.ipynb           # 4개 모델 비교 학습 + pkl 저장
│   └── 04_evaluation.ipynb         # 잔차·피처 중요도·ROC 평가
│
├── 📂 backend/                     # Flask REST API
│   ├── main.py                     # 앱 엔트리포인트
│   ├── requirements.txt            # 의존성
│   └── app/
│       ├── routes/prediction.py    # POST /api/predictions/
│       ├── services/               # 비즈니스 로직
│       ├── models/schemas.py       # Pydantic Request/Response
│       └── utils/model_loader.py   # 모델 로더
│
└── 📂 reports/
    ├── AI개발_수행내역서.md         # 과제 보고서 (한국어)
    └── figures/                    # 시각화 이미지 21종 (개별 변수 분포 11a~11h 포함)
```

---

## 📊 데이터셋

**Kaggle — [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)**

### 입력 피처 (10개)

| 컬럼명 | 설명 | 타입 |
|---|---|---|
| `person_age` | 나이 | 수치 |
| `person_income` | 연 소득 (달러) | 수치 |
| `person_home_ownership` | 주거 형태 (RENT/OWN/MORTGAGE/OTHER) | 범주 |
| `person_emp_length` | 재직 기간 (연) | 수치 |
| `loan_intent` | 대출 목적 | 범주 |
| `loan_grade` | 대출 등급 (A~G) | 범주 |
| `loan_amnt` | 대출 금액 (달러) | 수치 |
| `loan_percent_income` | 소득 대비 대출 비율 | 수치 |
| `cb_person_default_on_file` | 과거 부도 이력 (Y/N) | 범주 |
| `cb_person_cred_hist_length` | 신용 이력 기간 (연) | 수치 |

### 목표 변수

| 컬럼명 | 설명 | 용도 |
|---|---|---|
| `loan_int_rate` | 대출 금리 (%) | 🔵 회귀 타겟 |

---

## 🤖 모델

### 🔵 회귀 — `loan_int_rate` 예측

| 모델 | R² | RMSE | MAE |
|---|---|---|---|
| 선형 회귀 | 0.8692 | 1.1770 | 0.9173 |
| 릿지 회귀 | 0.8692 | 1.1770 | 0.9173 |
| GradientBoostingRegressor | 0.9038 | 1.0093 | 0.7906 |
| ✅ **RandomForestRegressor (튜닝)** | **0.9053** | **1.0017** | **0.7826** |

> 최종 배포 채택: **RandomForestRegressor** (RandomizedSearchCV 튜닝: n_estimators=200, max_depth=15, max_features=0.5, min_samples_leaf=1, compress=3)  
> `models/*.pkl`은 로컬 시연 전용으로 git 추적 제외 (`.gitignore`)

---

## 📈 평가 결과

> **RandomForestRegressor** 기준 (n_estimators=200, max_depth=15, max_features=0.5, min_samples_leaf=1, test set 20%, random_state=42)

| 지표 | 값 |
|---|---|
| 📉 **RMSE** | `1.0017` |
| 📉 **MAE** | `0.7826` |
| 📈 **R²** | `0.9053` |

R² **0.9053** — 모델이 대출 금리 분산의 90.5%를 설명합니다.

---

## ⚙️ 실행 방법

### 1️⃣ 의존성 설치

```bash
pip install -r requirements.txt
pip install jupyter
```

### 2️⃣ ML 파이프라인 실행

```bash
# 데이터 전처리
python ml/preprocessing.py

# 모델 학습
python ml/train.py

# 성능 평가
python ml/evaluate.py
```

> `models/*.pkl`은 git에서 제외되어 있습니다(로컬 시연 전용).  
> pkl이 없으면 `ensure_models()`가 자동으로 전처리·학습을 실행합니다.  
> `data/processed/*.csv`는 git에 포함되어 있으므로 별도 전처리 없이 바로 사용 가능합니다.

### 3️⃣ Jupyter 노트북 실행

```bash
jupyter notebook notebooks/
```

> `01_eda.ipynb` → `02_preprocessing.ipynb` → `03_modeling.ipynb` → `04_evaluation.ipynb` 순서로 실행

### 4️⃣ Streamlit 웹앱 실행

```bash
streamlit run streamlit_app.py
```

### 5️⃣ Flask API 실행

```bash
cd backend
flask run
```

### 6️⃣ API 요청 예시

```bash
curl -X POST http://localhost:5000/api/predictions/ \
  -H "Content-Type: application/json" \
  -d '{
    "person_age": 28,
    "person_income": 65000,
    "person_home_ownership": "RENT",
    "person_emp_length": 3.0,
    "loan_intent": "PERSONAL",
    "loan_grade": "C",
    "loan_amnt": 10000,
    "loan_percent_income": 0.15,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 5
  }'
```

**응답 예시:**
```json
{
  "loan_int_rate": 12.45
}
```

---

## 🗺️ 전체 파이프라인

```
📥 Kaggle CSV (credit_risk_dataset.csv)
     ↓
🔍 EDA (01_eda.ipynb) — 분포·결측치·상관관계 탐색
     ↓
🔧 전처리 (02_preprocessing.ipynb / ml/preprocessing.py)
   결측치 제거 · 이상치 필터 · LabelEncoding → CSV 저장
     ↓
🤖 모델 학습 (03_modeling.ipynb / ml/train.py)
   RF 회귀 (RandomizedSearchCV 튜닝) → .pkl 저장 (compress=3)
     ↓
📏 성능 평가 (04_evaluation.ipynb / ml/evaluate.py)
   RMSE / MAE / R²
     ↓
         ┌──────────────────┬─────────────────┐
    🌐 Flask API        💻 Streamlit 웹앱   📊 시각화 보고서
   (backend/)         (streamlit_app.py)  (reports/figures/)
                      ☁️ Streamlit Cloud 배포
```

---

## 🛠️ Streamlit Cloud 배포 설정

| 파일 | 역할 |
|---|---|
| `requirements.txt` | Python 패키지 (`scikit-learn==1.4.0` 고정 — pkl 역직렬화 버전 호환) |
| `packages.txt` | OS 패키지 (`fonts-nanum` — 차트 한글 폰트) |
| `models/*.pkl` | git 제외 (로컬 시연 전용) — Cloud에서는 `ensure_models()`가 자동 생성 |
| `data/processed/*.csv` | git 추적 — 데이터 인사이트 탭 즉시 표시 |

> `scikit-learn` 버전 불일치 시 `__pyx_unpickle AttributeError` 발생 → 반드시 버전 고정 필요

---

## ✅ 진행 현황

- [x] 데이터셋 선정 (Credit Risk Dataset)
- [x] 프로젝트 구조 스캐폴딩
- [x] 데이터 전처리 (`ml/preprocessing.py`)
- [x] 모델 학습 (`ml/train.py`)
- [x] 성능 평가 (`ml/evaluate.py`) — R² 0.9053 (RandomizedSearchCV 튜닝 RF)
- [x] EDA Jupyter 노트북 (`notebooks/01_eda.ipynb`)
- [x] 전처리 노트북 (`notebooks/02_preprocessing.ipynb`)
- [x] 모델 학습 노트북 (`notebooks/03_modeling.ipynb`)
- [x] 평가 노트북 (`notebooks/04_evaluation.ipynb`)
- [x] 시각화 이미지 생성 (`reports/figures/` — 21종, 개별 변수 분포 11a~11h 추가)
- [x] 과제 보고서 작성 (`reports/AI개발_수행내역서.md`)
- [x] Streamlit 웹앱 3탭 구성 (`streamlit_app.py`)
- [x] Streamlit Cloud 배포 (한글 폰트·버전 호환 처리)
- [x] 예측 모달 개선 — 등급 내 금리 위치 비교 + 전체 입력값 10개 vs 데이터 평균 수치 비교
- [x] Plotly 전환 — 다크/라이트 테마 자동 대응
- [x] Flask API 테스트 (`backend/tests/` — 6/6 PASS)
- [ ] HWP 보고서 최종 제출
