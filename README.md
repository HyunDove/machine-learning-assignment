# 💳 대출 금리 예측 머신러닝 과제

> Kaggle **Credit Risk Dataset** 기반으로 대출 금리를 예측하는 회귀 모델과  
> 대출 부도 여부를 예측하는 분류 모델을 구현한 머신러닝 과제입니다.

---

## 📋 과제 개요

| 항목 | 내용 |
|---|---|
| 🎯 **주제** | 대출 금리 예측 모델 구현 |
| 🔍 **문제 유형** | 회귀 (`loan_int_rate`) + 분류 (`loan_status`) |
| 📦 **라이브러리** | Python · scikit-learn · Flask · Streamlit · Pydantic |
| 📊 **데이터** | Kaggle — Credit Risk Dataset (32,581행, 12컬럼) |
| 📏 **평가지표** | RMSE, MAE, R² |
| 🚀 **결과물** | Streamlit 웹앱 + Flask API + Jupyter 노트북 + 시각화 보고서 |

---

## 📁 프로젝트 구조

```
ml_project/
│
├── 📄 credit_risk_dataset.csv      # 원본 데이터 (Kaggle)
├── 📄 streamlit_app.py             # Streamlit 웹앱
├── 📄 requirements.txt             # Streamlit Cloud 의존성
│
├── 📂 data/
│   └── processed/                  # 전처리 완료 CSV
│
├── 📂 ml/                          # ML 파이프라인
│   ├── preprocessing.py            # 결측치·이상치 처리, LabelEncoding
│   ├── train.py                    # 모델 학습 (회귀 + 분류)
│   └── evaluate.py                 # RMSE, MAE, R² 평가
│
├── 📂 models/                      # 학습된 모델 파일 (.pkl, gitignore)
│   ├── loan_rate_model.pkl         # GradientBoostingRegressor
│   └── loan_status_model.pkl       # RandomForestClassifier
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
    └── figures/                    # 시각화 이미지 10종
```

---

## 📊 데이터셋

**Kaggle — [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)**

### 입력 피처 (11개)

| 컬럼명 | 설명 | 타입 |
|---|---|---|
| `person_age` | 나이 | 수치 |
| `person_income` | 연 소득 (달러) | 수치 |
| `person_home_ownership` | 주거 형태 (RENT/OWN/MORTGAGE/OTHER) | 범주 |
| `person_emp_length` | 재직 기간 (연) | 수치 |
| `loan_intent` | 대출 목적 | 범주 |
| `loan_grade` | 대출 등급 (A~G) | 범주 |
| `loan_amnt` | 대출 금액 (달러) | 수치 |
| `loan_status` | 대출 상태 (0=정상, 1=부도) | 범주 |
| `loan_percent_income` | 소득 대비 대출 비율 | 수치 |
| `cb_person_default_on_file` | 과거 부도 이력 (Y/N) | 범주 |
| `cb_person_cred_hist_length` | 신용 이력 기간 (연) | 수치 |

### 목표 변수

| 컬럼명 | 설명 | 용도 |
|---|---|---|
| `loan_int_rate` | 대출 금리 (%) | 🔵 회귀 타겟 |
| `loan_status` | 대출 부도 여부 | 🟠 분류 타겟 |

---

## 🤖 모델

### 🔵 회귀 — `loan_int_rate` 예측

| 모델 | 선택 이유 |
|---|---|
| ✅ **GradientBoostingRegressor** | 비선형 관계 포착, 피처 중요도 제공 |
| RandomForestRegressor | 앙상블, 이상치 강건 (비교 기준) |
| Ridge / Lasso | 선형 베이스라인 |

### 🟠 분류 — `loan_status` 예측

| 모델 | 선택 이유 |
|---|---|
| ✅ **RandomForestClassifier** | 불균형 데이터에 강건, 해석 용이 |
| GradientBoostingClassifier | 높은 정밀도 (비교 기준) |
| LogisticRegression | 선형 베이스라인 |

---

## 📈 평가 결과

> **GradientBoostingRegressor** 기준 (test set 20%, random_state=42)

| 지표 | 값 |
|---|---|
| 📉 **RMSE** | `1.0079` |
| 📉 **MAE** | `0.7899` |
| 📈 **R²** | `0.9041` |

R² **0.90** — 모델이 대출 금리 분산의 90%를 설명합니다.

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

### 3️⃣ Jupyter 노트북 실행 (EDA → 전처리 → 학습 → 평가)

```bash
jupyter notebook notebooks/
```

> `01_eda.ipynb` → `02_preprocessing.ipynb` → `03_modeling.ipynb` → `04_evaluation.ipynb` 순서로 실행

### 4️⃣ Streamlit 웹앱 실행

```bash
streamlit run streamlit_app.py
```

> 모델 파일(.pkl)이 없으면 앱 실행 시 자동으로 학습합니다.

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
    "loan_status": 0,
    "loan_percent_income": 0.15,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 5
  }'
```

**응답 예시:**
```json
{
  "loan_int_rate": 12.45,
  "loan_status_prob": 0.21,
  "loan_status_label": "정상"
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
   GBR 회귀 + RF 분류 → .pkl 저장
     ↓
📏 성능 평가 (04_evaluation.ipynb / ml/evaluate.py)
   RMSE / MAE / R² / AUC-ROC
     ↓
         ┌──────────────────┬─────────────────┐
    🌐 Flask API        💻 Streamlit 웹앱   📊 시각화 보고서
   (backend/)         (streamlit_app.py)  (reports/figures/)
```

---

## ✅ 진행 현황

- [x] 데이터셋 선정 (Credit Risk Dataset)
- [x] 프로젝트 구조 스캐폴딩
- [x] 데이터 전처리 (`ml/preprocessing.py`)
- [x] 모델 학습 (`ml/train.py`)
- [x] 성능 평가 (`ml/evaluate.py`) — R² 0.9041
- [x] EDA Jupyter 노트북 (`notebooks/01_eda.ipynb`)
- [x] 전처리 노트북 (`notebooks/02_preprocessing.ipynb`)
- [x] 모델 학습 노트북 (`notebooks/03_modeling.ipynb`)
- [x] 평가 노트북 (`notebooks/04_evaluation.ipynb`)
- [x] 시각화 이미지 생성 (`reports/figures/` — 10종)
- [x] 과제 보고서 작성 (`reports/AI개발_수행내역서.md`)
- [x] Streamlit 웹앱 (`streamlit_app.py`)
- [x] Flask API 테스트 (`backend/tests/` — 6/6 PASS)
- [ ] HWP 보고서 최종 제출
