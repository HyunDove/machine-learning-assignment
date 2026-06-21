# 머신러닝 과제 노트

> 최초 작성: 2026-06-18

---

## 과제 개요

| 항목 | 내용 |
|---|---|
| **주제** | 대출 금리 예측 모델 구현 |
| **문제 유형** | 회귀 + 분류 (지도학습) |
| **라이브러리** | Python + scikit-learn |
| **데이터** | Kaggle - Credit Risk Dataset |
| **평가지표** | RMSE, MAE, R² |
| **결과물** | 웹 API + 시각화 + HWP 보고서 |

---

## 과제 요구사항 (1.2 과제 범위)

### AI
- 원시 데이터 수집 및 데이터셋 구축
- 데이터 전처리, 표준화, 상관관계 분석 (EDA 도구 활용)
- 예측모델 선정 및 학습
- RMSE, MAE 등 평가지표를 활용한 모델 성능 평가
- 웹 API 및 프로토타입 구축
- 예측모델 웹/앱 시스템 구축
- 테스트

### 시각화
- 사용자 입력 구현
- MSE, RMSE, R² 기반 성능 평가
- 예측 결과 기반 원환산값 제공
- 테스트

---

## CSV 컬럼 설계 (실제 데이터셋 기준)

### 입력 변수 (Feature) — 11개

| 컬럼명 | 한국어 | 타입 | 예시값 |
|---|---|---|---|
| `person_age` | 나이 | 수치 | 22 |
| `person_income` | 연 소득 | 수치 | 59000 |
| `person_home_ownership` | 주거 형태 | 범주 | RENT |
| `person_emp_length` | 재직 기간 | 수치 | 5.0 |
| `loan_intent` | 대출 목적 | 범주 | PERSONAL |
| `loan_grade` | 대출 등급 | 범주 | D |
| `loan_amnt` | 대출 금액 | 수치 | 35000 |
| `loan_status` | 대출 상태 | 범주 | 1 |
| `loan_percent_income` | 소득 대비 대출 비율 | 수치 | 0.59 |
| `cb_person_default_on_file` | 과거 부도 이력 | 범주 | Y |
| `cb_person_cred_hist_length` | 신용 이력 기간 | 수치 | 3 |

### 목표 변수 (Target)

| 컬럼명 | 한국어 | 타입 | 예시값 | 용도 |
|---|---|---|---|---|
| `loan_int_rate` | 대출 금리 | 수치 | 16.02 | 회귀 타겟 |
| `loan_status` | 대출 부도 여부 | 범주 | 0(정상) / 1(부도) | 분류 타겟 |

### 컬럼 간 관계

```
loan_grade 높을수록 (A→G)  → 금리 ↑
person_income 높을수록     → 금리 ↓
loan_amnt 높을수록         → 금리 ↑
cb_person_default_on_file Y → 금리 ↑
person_emp_length 길수록   → 금리 ↓
```

### 데이터 예시 (1행)

```csv
person_age,person_income,person_home_ownership,person_emp_length,loan_intent,loan_grade,loan_amnt,loan_int_rate,loan_status,loan_percent_income,cb_person_default_on_file,cb_person_cred_hist_length
22,59000,RENT,123.0,PERSONAL,D,35000,16.02,1,0.59,Y,3
```

---

## 전체 파이프라인

```
[Kaggle CSV] → [EDA/전처리] → [모델 학습] → [성능 평가] → [웹 API] → [시각화]
```

### 추천 모델

**[회귀] loan_int_rate 예측**
- `GradientBoostingRegressor` (1순위)
- `RandomForestRegressor`
- `Ridge` / `Lasso`

**[분류] loan_status 예측**
- `RandomForestClassifier` (1순위)
- `GradientBoostingClassifier`
- `LogisticRegression`

---

## 배경 조사 메모

### 딥러닝 vs 머신러닝 (게임 매크로 기준)
- 실무: 딥러닝 주류 (화면=이미지 → CNN 필수)
- 이번 과제: 이미지→수치화 단계를 수작업으로 대체하는 구조
- 딥러닝이 자동으로 하는 피처 추출을 사람이 직접 설계

### 검토했던 다른 주제들

| 주제 | 탈락 이유 |
|---|---|
| 게임 매크로 (메이플스토리 플래닛) | 과제 요구사항과 불일치 |
| FPS 에임봇 탐지 | RMSE/MAE/R² 적용 어려움 (이상탐지 지표 상이) |
| FPS 플레이어 MMR 예측 | 가능하나 금융권으로 방향 전환 |
| 보험료 예측 | 가능하나 대출 금리가 더 실용적 |

---

## 데이터셋 정보 (Credit Risk Dataset)

- **Kaggle**: laotse/credit-risk-dataset
- **파일 경로**: `D:\AiWorkspace\individual\ml_project\credit_risk_dataset.csv`
- **크기**: 32,581행, 12개 컬럼
- **목표변수**: `loan_int_rate` (대출 금리 %) — 회귀 / `loan_status` (대출 부도 여부) — 분류

### 컬럼 상세 설명

| 컬럼명 | 한국어 | 타입 | 설명 | 예시값 |
|---|---|---|---|---|
| `person_age` | 나이 | 수치 | 대출 신청자 나이 | 22 |
| `person_income` | 연 소득 | 수치 | 연간 소득 (달러) | 59000 |
| `person_home_ownership` | 주거 형태 | 범주 | RENT / OWN / MORTGAGE / OTHER | RENT |
| `person_emp_length` | 재직 기간 | 수치 | 현재 직장 근무 연수 | 5.0 |
| `loan_intent` | 대출 목적 | 범주 | PERSONAL / EDUCATION / MEDICAL / VENTURE / HOMEIMPROVEMENT / DEBTCONSOLIDATION | PERSONAL |
| `loan_grade` | 대출 등급 | 범주 | A(우량) ~ G(불량) | D |
| `loan_amnt` | 대출 금액 | 수치 | 신청 대출 금액 (달러) | 35000 |
| `loan_int_rate` | **대출 금리** | 수치 | **목표변수** — 적용 금리 (%) | 16.02 |
| `loan_status` | 대출 상태 | 범주 | 0(정상) / 1(부도) | 1 |
| `loan_percent_income` | 소득 대비 대출 비율 | 수치 | loan_amnt / person_income | 0.59 |
| `cb_person_default_on_file` | 과거 부도 이력 | 범주 | Y(있음) / N(없음) | Y |
| `cb_person_cred_hist_length` | 신용 이력 기간 | 수치 | 신용 기록 보유 연수 | 3 |

### 범주형 컬럼 고유값

| 컬럼명 | 고유값 |
|---|---|
| `person_home_ownership` | RENT, OWN, MORTGAGE, OTHER |
| `loan_intent` | PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION |
| `loan_grade` | A, B, C, D, E, F, G |
| `loan_status` | 0 (정상), 1 (부도) |
| `cb_person_default_on_file` | Y (있음), N (없음) |

---

## TODO

- [x] Kaggle 대출 데이터셋 선정 (Credit Risk Dataset)
- [ ] EDA 진행 (결측치, 이상치, 상관관계)
- [ ] 모델 선정 및 학습
- [ ] RMSE, MAE, R² 평가
- [ ] 웹 API 구축 (Flask 등)
- [ ] 시각화 구현
- [ ] HWP 보고서 작성
