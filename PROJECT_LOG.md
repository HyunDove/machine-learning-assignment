# ML 프로젝트 진행 기록

> 최초 작성: 2026-06-19

---

## 과제 개요

| 항목 | 내용 |
|---|---|
| **주제** | 대출 금리 예측 모델 구현 |
| **문제 유형** | 회귀 (`loan_int_rate`) + 분류 (`loan_status`) |
| **라이브러리** | Python · scikit-learn · Flask · Pydantic |
| **데이터** | Kaggle — Credit Risk Dataset (32,581행, 12컬럼) |
| **평가지표** | RMSE, MAE, R² |
| **결과물** | 웹 API + 시각화 + HWP 보고서 |

---

## 진행 작업 내역

### 2026-06-19

#### 1. 데이터셋 확정
- Kaggle `laotse/credit-risk-dataset` 선택
- 파일 위치: `credit_risk_dataset.csv` (프로젝트 루트)
- 입력 피처 11개, 타겟 변수 2개 확정

#### 2. 프로젝트 구조 스캐폴딩
- 전체 디렉토리 및 파일 생성 완료
- Flask 앱 구조 (팩토리 패턴, Blueprint, Pydantic 스키마)
- ML 파이프라인 스크립트 (전처리 → 학습 → 평가)

---

## 프로젝트 구조

```
ml_project/
│
├── credit_risk_dataset.csv          # 원본 데이터 (Kaggle)
├── ml_assignment_notes.md           # 과제 노트
├── PROJECT_LOG.md                   # 이 파일
├── .gitignore
│
├── data/
│   ├── raw/                         # 원본 CSV 보관
│   └── processed/                   # 전처리 후 CSV 저장
│
├── ml/                              # ML 파이프라인
│   ├── preprocessing.py             # 결측치·이상치 처리, LabelEncoding
│   ├── train.py                     # 모델 학습 (회귀 + 분류)
│   └── evaluate.py                  # RMSE, MAE, R² 평가
│
├── models/                          # 학습된 모델 파일 (.pkl)
│   ├── loan_rate_model.pkl          # GradientBoostingRegressor (생성 예정)
│   └── loan_status_model.pkl        # RandomForestClassifier (생성 예정)
│
├── notebooks/                       # Jupyter 노트북
│   ├── README.txt                   # 실행 순서 안내
│   ├── 01_eda.ipynb                 # EDA (생성 예정)
│   ├── 02_preprocessing.ipynb       # 전처리 확인 (생성 예정)
│   ├── 03_modeling.ipynb            # 모델 학습 (생성 예정)
│   └── 04_evaluation.ipynb          # 평가 및 시각화 (생성 예정)
│
├── backend/                         # Flask REST API
│   ├── main.py                      # 앱 엔트리포인트
│   ├── requirements.txt             # 의존성
│   ├── .env.example                 # 환경변수 템플릿
│   └── app/
│       ├── __init__.py              # create_app (팩토리 패턴)
│       ├── config.py                # 환경변수 설정
│       ├── errors.py                # ApiError + 글로벌 핸들러
│       ├── routes/
│       │   └── prediction.py        # POST /api/predictions/
│       ├── services/
│       │   └── prediction_service.py
│       ├── models/
│       │   └── schemas.py           # Pydantic Request/Response
│       └── utils/
│           └── model_loader.py      # joblib lazy-load
│
├── backend/tests/
│   ├── conftest.py
│   └── test_prediction.py
│
└── reports/
    └── figures/                     # 시각화 이미지 저장
```

---

## 주요 파일 역할

### ML 파이프라인

| 파일 | 역할 | 실행 방법 |
|---|---|---|
| `ml/preprocessing.py` | 결측치 처리, 이상치 제거, LabelEncoding | `python ml/preprocessing.py` |
| `ml/train.py` | GBR(회귀) + RF(분류) 학습 후 `.pkl` 저장 | `python ml/train.py` |
| `ml/evaluate.py` | RMSE, MAE, R² 출력 | `python ml/evaluate.py` |

### Flask API

| 엔드포인트 | 메서드 | 설명 |
|---|---|---|
| `/api/predictions/` | POST | 입력값 → 금리 예측 + 부도 확률 반환 |

**요청 예시:**

```json
{
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
}
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

## 추천 모델 근거

### 회귀 — `loan_int_rate` 예측

| 모델 | 순위 | 선택 이유 |
|---|---|---|
| `GradientBoostingRegressor` | 1순위 | 비선형 관계 포착, 피처 중요도 제공 |
| `RandomForestRegressor` | 2순위 | 앙상블, 이상치 강건 |
| `Ridge / Lasso` | 베이스라인 | 선형 모델 비교 기준 |

### 분류 — `loan_status` 예측

| 모델 | 순위 | 선택 이유 |
|---|---|---|
| `RandomForestClassifier` | 1순위 | 불균형 데이터에 강건, 해석 용이 |
| `GradientBoostingClassifier` | 2순위 | 높은 정밀도 |
| `LogisticRegression` | 베이스라인 | 선형 비교 기준 |

---

## 남은 작업 (TODO)

- [ ] `data/raw/`에 CSV 복사 후 `ml/preprocessing.py` 실행
- [ ] `ml/train.py` 로 모델 학습
- [ ] `ml/evaluate.py` 로 RMSE / MAE / R² 확인
- [ ] Jupyter 노트북 EDA 작성 (`notebooks/01_eda.ipynb`)
- [ ] Flask API 기동 및 테스트 (`cd backend && flask run`)
- [ ] 시각화 구현 (예측값 분포, 피처 중요도 그래프)
- [ ] HWP 보고서 작성

---

## 의존성 설치

```bash
cd backend
pip install -r requirements.txt
pip install jupyter
```
