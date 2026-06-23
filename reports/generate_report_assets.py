"""
보고서용 시각화 이미지 및 통계 일괄 생성 스크립트
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

ROOT          = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW_PATH      = os.path.join(ROOT, "credit_risk_dataset.csv")
PROCESSED_PATH = os.path.join(ROOT, "data", "processed", "credit_risk_cleaned.csv")
OUT_DIR       = os.path.join(ROOT, "reports", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

df_raw = pd.read_csv(RAW_PATH)
df     = pd.read_csv(PROCESSED_PATH)

FEATURES = [
    "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "loan_intent", "loan_grade", "loan_amnt",
    "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length",
]
TARGET = "loan_int_rate"

X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 모델 학습 ─────────────────────────────────────────────────────────────────
models = {
    "선형 회귀":        LinearRegression(),
    "릿지 회귀":        Ridge(alpha=1.0),
    "그래디언트 부스팅": GradientBoostingRegressor(n_estimators=200, random_state=42),
    "랜덤 포레스트":    RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
}
results = {}
preds   = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    preds[name] = y_pred
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    results[name] = {"RMSE": round(rmse,4), "MAE": round(mae,4), "R2": round(r2,4)}
    print(f"[{name}] RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

rf_model = models["랜덤 포레스트"]

# ── 01. 결측치 통계 ───────────────────────────────────────────────────────────
miss = df_raw.isnull().sum()
miss = miss[miss > 0].sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(miss.index, miss.values, color="#4C72B0", width=0.5)
ax.set_title("결측치 개수", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("결측치 수")
ax.set_xlabel("컬럼명")
for bar, val in zip(bars, miss.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_missing_values.png"), bbox_inches="tight")
plt.close()
print("01_missing_values.png 저장 완료")

# ── 02. 상관관계 히트맵 ───────────────────────────────────────────────────────
corr_cols = FEATURES + [TARGET]
corr = df[corr_cols].corr()

col_labels_kr = {
    "person_age": "나이", "person_income": "연소득",
    "person_home_ownership": "주거형태", "person_emp_length": "재직기간",
    "loan_intent": "대출목적", "loan_grade": "대출등급",
    "loan_amnt": "대출금액",
    "loan_percent_income": "소득대비비율", "cb_person_default_on_file": "부도이력",
    "cb_person_cred_hist_length": "신용이력", "loan_int_rate": "대출금리",
}
corr.rename(index=col_labels_kr, columns=col_labels_kr, inplace=True)

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(
    corr, annot=True, fmt=".2f", cmap="RdYlBu_r",
    linewidths=0.4, linecolor="white", ax=ax,
    annot_kws={"size": 8}, vmin=-1, vmax=1,
    cbar_kws={"shrink": 0.8},
)
ax.set_title("피처 간 상관관계 히트맵 (다중공선성 분석)", fontsize=13, fontweight="bold", pad=14)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_correlation_heatmap.png"), bbox_inches="tight")
plt.close()
print("02_correlation_heatmap.png 저장 완료")

# ── 03. 주요 변수 분포 ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.scatter(df["loan_grade"], df[TARGET], alpha=0.15, s=10, color="#4C72B0")
ax.set_xticks(range(7))
ax.set_xticklabels(["A","B","C","D","E","F","G"])
ax.set_xlabel("대출 등급")
ax.set_ylabel("대출 금리 (%)")
ax.set_title("대출 등급 vs 대출 금리", fontsize=12, fontweight="bold")

ax2 = axes[1]
ax2.hist(df[TARGET], bins=40, color="#4C72B0", edgecolor="white", linewidth=0.4)
ax2.set_xlabel("대출 금리 (%)")
ax2.set_ylabel("빈도")
ax2.set_title("대출 금리 분포", fontsize=12, fontweight="bold")
ax2.axvline(df[TARGET].mean(), color="red", linestyle="--", linewidth=1.5,
            label=f"평균 {df[TARGET].mean():.2f}%")
ax2.legend()

plt.suptitle("주요 변수별 데이터 분포", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_feature_distribution.png"), bbox_inches="tight")
plt.close()
print("03_feature_distribution.png 저장 완료")

# ── 04~07. 모델별 예측 결과 ───────────────────────────────────────────────────
model_file_map = {
    "선형 회귀":        "04_model_linear.png",
    "릿지 회귀":        "05_model_ridge.png",
    "그래디언트 부스팅": "06_model_gbr.png",
    "랜덤 포레스트":    "07_model_rf.png",
}
grade_test = X_test["loan_grade"].values

for name, fname in model_file_map.items():
    y_pred = preds[name]
    r2 = results[name]["R2"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(grade_test, y_test.values, alpha=0.25, s=12, color="#4C72B0", label="실제값")
    ax.scatter(grade_test, y_pred,        alpha=0.25, s=12, color="#DD3333", label="예측값")
    ax.set_xticks(range(7))
    ax.set_xticklabels(["A","B","C","D","E","F","G"])
    ax.set_xlabel("대출 등급")
    ax.set_ylabel("대출 금리 (%)")
    ax.set_title(f"{name} — 대출 등급 vs 대출 금리 예측 결과\n(R² = {r2})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), bbox_inches="tight")
    plt.close()
    print(f"{fname} 저장 완료")

# ── 08. 모델 성능 비교 ────────────────────────────────────────────────────────
model_names = list(results.keys())
rmse_vals = [results[m]["RMSE"] for m in model_names]
mae_vals  = [results[m]["MAE"]  for m in model_names]
r2_vals   = [results[m]["R2"]   for m in model_names]
colors    = ["#4C72B0","#55A868","#C44E52","#8172B2"]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, vals, title, ylabel, lower_is_better in zip(
    axes,
    [rmse_vals, mae_vals, r2_vals],
    ["RMSE 비교 (낮을수록 좋음)", "MAE 비교 (낮을수록 좋음)", "R² 비교 (높을수록 좋음)"],
    ["RMSE", "MAE", "R²"],
    [True, True, False],
):
    bars = ax.bar(model_names, vals, color=colors, width=0.5, edgecolor="white")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
    best_idx = int(np.argmin(vals)) if lower_is_better else int(np.argmax(vals))
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

plt.suptitle("모델별 성능 지표 비교", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "08_model_comparison.png"), bbox_inches="tight")
plt.close()
print("08_model_comparison.png 저장 완료")

# ── 09. 피처 중요도 ───────────────────────────────────────────────────────────
feat_imp = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values()
feat_labels_kr = {
    "person_age": "나이", "person_income": "연소득",
    "person_home_ownership": "주거형태", "person_emp_length": "재직기간",
    "loan_intent": "대출목적", "loan_grade": "대출등급",
    "loan_amnt": "대출금액", "loan_status": "대출상태",
    "loan_percent_income": "소득대비비율", "cb_person_default_on_file": "부도이력",
    "cb_person_cred_hist_length": "신용이력",
}
feat_imp.index = [feat_labels_kr.get(i, i) for i in feat_imp.index]

fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = ["#DD3333" if v == feat_imp.max() else "#4C72B0" for v in feat_imp.values]
ax.barh(feat_imp.index, feat_imp.values, color=bar_colors, edgecolor="white")
ax.set_xlabel("중요도")
ax.set_title("랜덤 포레스트 — 피처 중요도 (Feature Importance)", fontsize=12, fontweight="bold")
for i, (idx, val) in enumerate(feat_imp.items()):
    ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "09_feature_importance.png"), bbox_inches="tight")
plt.close()
print("09_feature_importance.png 저장 완료")

# ── 10. 최종 실제값 vs 예측값 ─────────────────────────────────────────────────
y_pred_rf = preds["랜덤 포레스트"]
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, y_pred_rf, alpha=0.25, s=12, color="#4C72B0")
lims = [min(y_test.min(), y_pred_rf.min()) - 0.5,
        max(y_test.max(), y_pred_rf.max()) + 0.5]
ax.plot(lims, lims, "r--", linewidth=1.5, label="완벽한 예측선 (y=x)")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("실제 대출 금리 (%)")
ax.set_ylabel("예측 대출 금리 (%)")
r2 = results["랜덤 포레스트"]["R2"]
ax.set_title(f"랜덤 포레스트 — 실제값 vs 예측값 (R² = {r2})", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "10_final_actual_vs_pred.png"), bbox_inches="tight")
plt.close()
print("10_final_actual_vs_pred.png 저장 완료")

# ── 통계 출력 ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"원본 데이터: {len(df_raw):,}행 x {len(df_raw.columns)}컬럼")
print(f"전처리 후:  {len(df):,}행")
print("\n[결측치]\n" + df_raw.isnull().sum().to_string())

key_cols = ["person_age","person_income","loan_amnt","loan_int_rate",
            "person_emp_length","loan_percent_income","cb_person_cred_hist_length"]
print("\n[주요 컬럼 기술통계]\n" + df[key_cols].describe().round(4).to_string())

print("\n[모델 성능 최종 비교]")
for name, r in results.items():
    print(f"  {name:16s} | RMSE={r['RMSE']:.4f} | MAE={r['MAE']:.4f} | R2={r['R2']:.4f}")

print("\n모든 리소스 생성 완료!")
