"""
보고서 추가 시각화 이미지 생성 (11~13번)
- 11_eda_distributions.png : 수치형·범주형 변수 분포
- 12_classification_evaluation.png : Confusion Matrix + ROC 곡선
- 13_residual_analysis.png : 잔차 분석
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score,
)

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

ROOT      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_PATH = os.path.join(ROOT, "data", "processed", "credit_risk_cleaned.csv")
OUT_DIR   = os.path.join(ROOT, "reports", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

df   = pd.read_csv(DATA_PATH)
SEED = 42

FEATURES = [
    "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "loan_intent", "loan_grade", "loan_amnt", "loan_status",
    "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length",
]
FEATURES_CLS = [f for f in FEATURES if f != "loan_status"]

X  = df[FEATURES];  y  = df["loan_int_rate"]
Xc = df[FEATURES_CLS]; yc = df["loan_status"]

X_train,  X_test,  y_train,  y_test  = train_test_split(X,  y,  test_size=0.2, random_state=SEED)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=SEED)

print("모델 학습 중...")
reg = GradientBoostingRegressor(n_estimators=200, random_state=SEED)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

clf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
clf.fit(Xc_train, yc_train)
yc_pred      = clf.predict(Xc_test)
yc_pred_prob = clf.predict_proba(Xc_test)[:, 1]
print("완료")

# ── 11. 수치형·범주형 변수 분포 ─────────────────────────────────────────────
num_cols   = ["person_age", "person_income", "person_emp_length",
              "loan_amnt", "loan_percent_income", "cb_person_cred_hist_length"]
num_labels = ["나이 (세)", "연소득 (달러)", "재직기간 (년)",
              "대출금액 (달러)", "소득대비비율", "신용이력 (년)"]

fig = plt.figure(figsize=(16, 12))
gs  = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.38)

for i, (col, label) in enumerate(zip(num_cols, num_labels)):
    ax   = fig.add_subplot(gs[i // 3, i % 3])
    data = df[col].dropna()
    ax.hist(data, bins=35, color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax.axvline(data.mean(), color="red", linestyle="--", lw=1.3,
               label=f"평균 {data.mean():.1f}")
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    if col == "person_income":
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

# 대출 목적 분포
ax7 = fig.add_subplot(gs[2, 1])
intent_labels = {
    "DEBTCONSOLIDATION": "부채통합", "EDUCATION": "교육",
    "HOMEIMPROVEMENT": "주거개선", "MEDICAL": "의료",
    "PERSONAL": "개인", "VENTURE": "창업",
}
vc7   = df["loan_intent"].value_counts()
vc7.index = [intent_labels.get(x, x) for x in vc7.index]
colors7 = sns.color_palette("Set2", len(vc7))
bars7   = ax7.bar(vc7.index, vc7.values, color=colors7, edgecolor="white")
ax7.set_title("대출 목적 분포", fontsize=11, fontweight="bold")
ax7.set_xticklabels(vc7.index, rotation=20, ha="right", fontsize=8)
for bar, val in zip(bars7, vc7.values):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f"{val:,}", ha="center", fontsize=7)

# 대출 등급 분포
ax8 = fig.add_subplot(gs[2, 2])
vc8     = df["loan_grade"].value_counts().reindex([0,1,2,3,4,5,6])
palette = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, 7))
bars8   = ax8.bar(["A","B","C","D","E","F","G"], vc8.values, color=palette, edgecolor="white")
ax8.set_title("대출 등급 분포", fontsize=11, fontweight="bold")
for bar, val in zip(bars8, vc8.values):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             f"{val:,}", ha="center", fontsize=7)

fig.suptitle("주요 변수별 데이터 분포 (수치형 6개 + 범주형 2개)",
             fontsize=13, fontweight="bold", y=1.01)
plt.savefig(os.path.join(OUT_DIR, "11_eda_distributions.png"), bbox_inches="tight")
plt.close()
print("11_eda_distributions.png 저장 완료")

# ── 12. 분류 모델 평가 (Confusion Matrix + ROC) ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm = confusion_matrix(yc_test, yc_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["예측 정상", "예측 부도"],
            yticklabels=["실제 정상", "실제 부도"],
            annot_kws={"size": 14})
axes[0].set_title("혼동 행렬 (RandomForestClassifier)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("실제 값", fontsize=11)
axes[0].set_xlabel("예측 값", fontsize=11)

tn, fp, fn, tp = cm.ravel()
axes[0].text(0.5, -0.18,
    f"정밀도: {tp/(tp+fp):.3f}   재현율: {tp/(tp+fn):.3f}   F1: {2*tp/(2*tp+fp+fn):.3f}",
    ha="center", transform=axes[0].transAxes, fontsize=9, color="#333")

fpr, tpr, _ = roc_curve(yc_test, yc_pred_prob)
auc_score   = roc_auc_score(yc_test, yc_pred_prob)
axes[1].plot(fpr, tpr, color="#4C72B0", lw=2.5,
             label=f"RandomForest (AUC = {auc_score:.4f})")
axes[1].plot([0, 1], [0, 1], "r--", lw=1.5, label="Random Classifier (AUC = 0.5)")
axes[1].fill_between(fpr, tpr, alpha=0.08, color="#4C72B0")
axes[1].set_xlabel("False Positive Rate (FPR)", fontsize=11)
axes[1].set_ylabel("True Positive Rate (TPR)", fontsize=11)
axes[1].set_title("ROC 곡선 — 부도 예측 분류 모델", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.suptitle("부도 예측 분류 모델 (RandomForestClassifier) 평가",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "12_classification_evaluation.png"), bbox_inches="tight")
plt.close()
print("12_classification_evaluation.png 저장 완료")

# ── 13. 잔차 분석 ─────────────────────────────────────────────────────────────
residuals = y_test.values - y_pred
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(residuals, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.3)
axes[0].axvline(0, color="red", linestyle="--", lw=1.8, label="잔차 0 기준선")
axes[0].axvline(residuals.mean(), color="orange", linestyle="--", lw=1.3,
                label=f"평균 {residuals.mean():.3f}")
axes[0].set_title("잔차 분포 (히스토그램)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("잔차 (실제값 − 예측값)")
axes[0].set_ylabel("빈도")
axes[0].legend(fontsize=8)

axes[1].scatter(y_pred, residuals, alpha=0.2, s=6, color="#4C72B0")
axes[1].axhline(0, color="red", linestyle="--", lw=1.8)
axes[1].set_xlabel("예측 금리 (%)")
axes[1].set_ylabel("잔차")
axes[1].set_title("잔차 vs 예측값", fontsize=12, fontweight="bold")

axes[2].scatter(y_test, np.abs(residuals), alpha=0.2, s=6, color="#C44E52")
axes[2].axhline(mae, color="orange", linestyle="--", lw=1.5,
                label=f"MAE = {mae:.4f}")
axes[2].set_xlabel("실제 금리 (%)")
axes[2].set_ylabel("|잔차|")
axes[2].set_title("절대 잔차 vs 실제값", fontsize=12, fontweight="bold")
axes[2].legend(fontsize=9)

plt.suptitle(
    f"잔차 분석 — GradientBoostingRegressor  (RMSE={rmse:.4f} · MAE={mae:.4f} · R²={r2:.4f})",
    fontsize=12, fontweight="bold", y=1.02,
)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "13_residual_analysis.png"), bbox_inches="tight")
plt.close()
print("13_residual_analysis.png 저장 완료")

print("\n[분류 모델 상세 성능]")
print(classification_report(yc_test, yc_pred, target_names=["정상(0)", "부도(1)"]))
print(f"AUC-ROC: {auc_score:.4f}")
print("\n모든 추가 이미지 생성 완료!")
