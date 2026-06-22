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

### 2026-06-20

#### 1. ML 파이프라인 구현 완료
- `ml/preprocessing.py`: 결측치 제거, 이상치 필터(나이>100·재직기간>60 제거), LabelEncoding → `data/processed/credit_risk_cleaned.csv` 저장
- `ml/train.py`: GBR(회귀) + RF(분류) 학습 후 `models/*.pkl` 저장
- `ml/evaluate.py`: RMSE / MAE / R² / AUC-ROC 출력

#### 2. Jupyter 노트북 4종 작성
- `01_eda.ipynb`: 분포·결측치·상관관계 탐색 (히트맵, 박스플롯, 등급별 금리)
- `02_preprocessing.ipynb`: 단계별 전처리 확인 및 CSV 저장
- `03_modeling.ipynb`: 4개 모델 비교 학습 + pkl 저장
- `04_evaluation.ipynb`: 잔차분포·피처중요도·ROC 평가 + 시각화 13종

#### 3. Flask REST API 완성
- `POST /api/predictions/` 구현 및 테스트 6/6 PASS
- Pydantic Request/Response 스키마, 글로벌 에러 핸들러

#### 4. 과제 보고서 초안 작성
- `reports/AI개발_수행내역서.md`: 수행 내역·모델 비교·평가 결과·구현 산출물 표

---

### 2026-06-21

#### 1. Streamlit 웹앱 전면 개편 (3탭 구성)
- **탭 1 🔮 금리 예측**: 사이드바 입력 → 예측 금리 + 부도 확률 카드 출력
- **탭 2 📊 모델 성능**: 4개 회귀 모델 RMSE·MAE·R² 비교 표 + 막대 그래프
- **탭 3 📈 데이터 인사이트**: 등급·나이·소득·대출금액·목적·부도이력별 평균 금리 시각화 (6개 차트)
- `@st.cache_resource` / `@st.cache_data` 적용
- 앱 시작 시 `load_models()` 호출 → 데이터 인사이트 탭 즉시 표시

#### 2. Streamlit Cloud 배포 이슈 해결
| 이슈 | 원인 | 해결 |
|---|---|---|
| `ImportError: matplotlib` | requirements.txt 누락 | `matplotlib>=3.8.0` 추가 |
| 한글 깨짐 (matplotlib 차트) | Linux 환경, Malgun Gothic 없음 | `packages.txt`(fonts-nanum) + `fm.fontManager.addfont()` 직접 경로 |
| `전처리된 데이터 파일이 없습니다` | CSV .gitignore 제외 + 버튼 클릭 시만 생성 | `.gitignore`에서 제거 + 앱 시작 시 `load_models()` |
| `__pyx_unpickle AttributeError` | Streamlit Cloud scikit-learn 버전 불일치 | `scikit-learn==1.4.0` 고정 |
| `loan_status_model.pkl` 67MB (GitHub 경고) | RF n_estimators=200 | n_estimators=200→50, pkl 17MB로 경량화 |

#### 3. 파일 변경 내역
- `streamlit_app.py`: 3탭 + 이모지 UI + 한글 폰트 함수 `_set_korean_font()` 추가
- `requirements.txt`: `matplotlib>=3.8.0` 추가, `scikit-learn==1.4.0` 버전 고정
- `packages.txt` (신규): `fonts-nanum` (Streamlit Cloud OS 패키지)
- `.gitignore`: `models/*.pkl` 및 `data/processed/*.csv` git 추적으로 전환
- `ml/train.py`: `RandomForestClassifier(n_estimators=50)` 변경 후 재학습
- `reports/AI개발_수행내역서.md`: RF 경량화·Streamlit 3탭 내용 추가
- `README.md`: 전면 최신화 (Streamlit Cloud 배포 섹션·배포 설정 표·진행현황 갱신)

---

### 2026-06-22

#### 1. UI 개편 — 중앙 입력 폼 + @st.dialog 결과 모달
- 사이드바 입력 폼 제거, `initial_sidebar_state="collapsed"`
- Tab 1: 2열 그리드 입력 폼 (col_a 신청자정보, col_b 대출정보, col_c 신용정보, col_d 자동계산)
- 예측 결과: `@st.dialog("🔮 예측 결과", width="large")` 오버레이 모달로 표시 (Streamlit 1.36.0+)
- `requirements.txt`: `streamlit>=1.35.0` → `>=1.36.0`

#### 2. 회귀 모델 GBR → RandomForestRegressor 교체 + compress=3 적용
| 항목 | 변경 전 | 변경 후 |
|---|---|---|
| 회귀 알고리즘 | GradientBoostingRegressor (n_estimators=200) | RandomForestRegressor (n_estimators=100, max_depth=15) |
| compress | 없음 | compress=3 (무손실 압축) |
| loan_rate_model.pkl | 363KB | 13.82MB |
| loan_status_model.pkl | 17MB | 2.92MB (compress=3 적용) |
| R² | 0.9041 | 0.9011 |
| RMSE | 1.0079 | 1.0235 |
| MAE | 0.7899 | 0.7941 |

#### 3. 문서 최신화
- `README.md`: 모델 비교표·평가 결과·파이프라인 섹션·체크리스트 갱신
- `ml_assignment_notes.md`: 모델 성능표 RF 최종채택으로 변경
- `reports/AI개발_수행내역서.md`: 3절 모델 선정·4절 예측 결과·6절 요약 RF 기준으로 갱신
- `PROJECT_GUIDE.md` (신규): 전체 소스 구조·데이터 흐름·파일별 상세 설명

#### 4. 파일 변경 내역
- `streamlit_app.py`: 사이드바 제거, 2열 그리드 입력, `@st.dialog` 모달 추가
- `ml/train.py`: RandomForestRegressor(n_estimators=100, max_depth=15) + compress=3 후 재학습
- `requirements.txt`: `streamlit>=1.36.0`으로 상향
- `PROJECT_GUIDE.md` (신규 생성)

---

## 프로젝트 구조 (최종)

```
ml_project/
│
├── credit_risk_dataset.csv          # 원본 데이터 (Kaggle)
├── streamlit_app.py                 # Streamlit 웹앱 (3탭 구성)
├── requirements.txt                 # Python 의존성 (scikit-learn==1.4.0 고정)
├── packages.txt                     # Streamlit Cloud OS 패키지 (fonts-nanum)
├── ml_assignment_notes.md           # 과제 노트
├── PROJECT_LOG.md                   # 이 파일
├── README.md                        # 프로젝트 소개
├── .gitignore
│
├── data/
│   ├── raw/                         # 원본 CSV 보관
│   └── processed/
│       └── credit_risk_cleaned.csv  # 전처리 완료 CSV (git 추적)
│
├── ml/                              # ML 파이프라인
│   ├── preprocessing.py             # 결측치·이상치 처리, LabelEncoding
│   ├── train.py                     # 모델 학습 (회귀 + 분류)
│   └── evaluate.py                  # RMSE, MAE, R² 평가
│
├── models/                          # 학습된 모델 파일 (git 추적)
│   ├── loan_rate_model.pkl          # RandomForestRegressor n_estimators=100, compress=3 (13.82MB)
│   └── loan_status_model.pkl        # RandomForestClassifier n_estimators=50, compress=3 (2.92MB)
│
├── notebooks/                       # Jupyter 노트북
│   ├── 01_eda.ipynb                 # EDA — 분포·결측치·상관관계
│   ├── 02_preprocessing.ipynb       # 전처리 단계별 확인
│   ├── 03_modeling.ipynb            # 4개 모델 비교 학습
│   └── 04_evaluation.ipynb          # 잔차·피처중요도·ROC 평가
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
│   └── test_prediction.py           # 6/6 PASS
│
└── reports/
    ├── AI개발_수행내역서.md          # 과제 보고서 (한국어)
    └── figures/                     # 시각화 이미지 13종
```

---

## 주요 파일 역할

### ML 파이프라인

| 파일 | 역할 | 실행 방법 |
|---|---|---|
| `ml/preprocessing.py` | 결측치 처리, 이상치 제거, LabelEncoding | `python ml/preprocessing.py` |
| `ml/train.py` | RF(회귀, n_estimators=100, max_depth=15, compress=3) + RF(분류, n_estimators=50, compress=3) 학습 후 `.pkl` 저장 | `python ml/train.py` |
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

- [x] 데이터 전처리 (`ml/preprocessing.py`)
- [x] 모델 학습 (`ml/train.py`) — RF 회귀 R² 0.9011 / RF 분류 AUC 0.9397
- [x] 성능 평가 (`ml/evaluate.py`)
- [x] Jupyter 노트북 4종 (`notebooks/01~04`)
- [x] Flask API 테스트 (`backend/tests/` — 6/6 PASS)
- [x] 시각화 13종 (`reports/figures/`)
- [x] Streamlit 웹앱 3탭 + Streamlit Cloud 배포
- [x] 과제 보고서 (`reports/AI개발_수행내역서.md`)
- [x] README 최신화
- [ ] HWP 보고서 최종 제출 (수동)

---

## 의존성 설치

```bash
cd backend
pip install -r requirements.txt
pip install jupyter
```
