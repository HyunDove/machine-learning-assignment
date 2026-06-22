"""
reports/figures/ 의 모델 관련 이미지 재생성 스크립트
- 04~08: 4개 비교 모델 재학습 (RF는 배포 하이퍼파라미터 n_estimators=100 기준)
- 09, 10, 13: 배포 pkl 로드해서 생성
- EDA/분류 이미지(01~03, 11, 12)는 변경 없으므로 유지
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor,
    GradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score,
)
from xgboost import XGBClassifier

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 110

ROOT      = os.path.dirname(os.path.abspath(__file__))
FIG_DIR   = os.path.join(ROOT, 'reports', 'figures')
RATE_PATH = os.path.join(ROOT, 'models', 'loan_rate_model.pkl')
DATA_PATH = os.path.join(ROOT, 'data', 'processed', 'credit_risk_cleaned.csv')

SEED = 42
FEATURES = [
    'person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
    'loan_intent', 'loan_grade', 'loan_amnt', 'loan_status',
    'loan_percent_income', 'cb_person_default_on_file', 'cb_person_cred_hist_length',
]

FEATURES_CLS = [f for f in FEATURES if f != 'loan_status']

LABELS_KR = {
    'person_age': '나이', 'person_income': '연소득',
    'person_home_ownership': '주거형태', 'person_emp_length': '재직기간',
    'loan_intent': '대출목적', 'loan_grade': '대출등급',
    'loan_amnt': '대출금액', 'loan_status': '대출상태',
    'loan_percent_income': '소득대비비율',
    'cb_person_default_on_file': '부도이력',
    'cb_person_cred_hist_length': '신용이력',
}


def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  저장: {name}')


def main():
    print('=== figures 재생성 시작 ===')

    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURES]
    y  = df['loan_int_rate']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # ── 비교 실험 재학습 ──────────────────────────────────────────────────
    print('\n[1/3] 비교 모델 학습 중...')
    reg_models = {
        '선형 회귀':        LinearRegression(),
        '릿지 회귀':        Ridge(alpha=1.0),
        '그래디언트 부스팅': GradientBoostingRegressor(n_estimators=200, random_state=SEED),
        '랜덤 포레스트':    RandomForestRegressor(
                                n_estimators=100, max_depth=15,
                                max_features='sqrt', random_state=SEED, n_jobs=-1),
    }
    results = {}
    preds   = {}
    for name, model in reg_models.items():
        print(f'  {name} ...', end=' ', flush=True)
        model.fit(X_train, y_train)
        yp = model.predict(X_test)
        preds[name] = yp
        results[name] = {
            'RMSE': round(float(np.sqrt(mean_squared_error(y_test, yp))), 4),
            'MAE':  round(float(mean_absolute_error(y_test, yp)), 4),
            'R2':   round(float(r2_score(y_test, yp)), 4),
        }
        print(f'R²={results[name]["R2"]:.4f}')

    # ── 04~07 개별 모델 scatter plot ────────────────────────────────────
    print('\n[2/3] 개별 모델 scatter 및 비교 차트 저장...')
    fig_names = {
        '선형 회귀':        '04_model_linear.png',
        '릿지 회귀':        '05_model_ridge.png',
        '그래디언트 부스팅': '06_model_gbr.png',
        '랜덤 포레스트':    '07_model_rf.png',
    }
    for name, fname in fig_names.items():
        yp = preds[name]
        r2 = results[name]['R2']
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, yp, alpha=0.2, s=8, color='#4C72B0')
        lims = [min(float(y_test.min()), float(yp.min())) - 0.5,
                max(float(y_test.max()), float(yp.max())) + 0.5]
        ax.plot(lims, lims, 'r--', lw=1.5, label='완벽한 예측선 (y=x)')
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('실제 대출 금리 (%)'); ax.set_ylabel('예측 대출 금리 (%)')
        ax.set_title(f'{name} (R² = {r2:.4f})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        plt.tight_layout()
        savefig(fname)

    # ── 08 모델 비교 막대그래프 ─────────────────────────────────────────
    model_names = list(results.keys())
    colors      = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric, label, lower in zip(
        axes,
        ['RMSE', 'MAE', 'R2'],
        ['RMSE (낮을수록 좋음)', 'MAE (낮을수록 좋음)', 'R² (높을수록 좋음)'],
        [True, True, False],
    ):
        vals     = [results[m][metric] for m in model_names]
        bars     = ax.bar(model_names, vals, color=colors, width=0.5, edgecolor='white')
        best_idx = int(np.argmin(vals)) if lower else int(np.argmax(vals))
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002, f'{v:.4f}', ha='center', fontsize=8)
    plt.suptitle('회귀 모델별 성능 비교', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    savefig('08_model_comparison.png')

    # ── 배포 pkl 로드 → 09, 10, 13 ──────────────────────────────────────
    print('\n[3/3] 배포 pkl → 09/10/13 생성...')
    regressor = joblib.load(RATE_PATH)
    print(f'  로드: {type(regressor).__name__}')
    y_pred    = regressor.predict(X_test)
    residuals = y_test.values - y_pred
    r2_dep    = r2_score(y_test, y_pred)

    # 09 피처 중요도
    feat_imp = pd.Series(regressor.feature_importances_, index=FEATURES).sort_values()
    feat_imp.index = [LABELS_KR.get(i, i) for i in feat_imp.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = ['#C44E52' if v == feat_imp.max() else '#4C72B0' for v in feat_imp.values]
    ax.barh(feat_imp.index, feat_imp.values, color=bar_colors, edgecolor='white')
    ax.set_xlabel('중요도')
    ax.set_title(f'{type(regressor).__name__} — 피처 중요도', fontsize=12, fontweight='bold')
    for i, (_, val) in enumerate(feat_imp.items()):
        ax.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    savefig('09_feature_importance.png')

    # 10 실제값 vs 예측값
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.2, s=10, color='#4C72B0')
    lims = [min(float(y_test.min()), float(y_pred.min())) - 0.5,
            max(float(y_test.max()), float(y_pred.max())) + 0.5]
    ax.plot(lims, lims, 'r--', lw=1.5, label='완벽한 예측선 (y=x)')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('실제 대출 금리 (%)'); ax.set_ylabel('예측 대출 금리 (%)')
    ax.set_title(f'실제값 vs 예측값 (R² = {r2_dep:.4f})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig('10_final_actual_vs_pred.png')

    # 13 잔차 분석
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].hist(residuals, bins=50, color='#4C72B0', edgecolor='white', linewidth=0.3)
    axes[0].axvline(0, color='red', linestyle='--', lw=1.5)
    axes[0].set_title('잔차 분포', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('잔차 (실제 - 예측)'); axes[0].set_ylabel('빈도')
    axes[1].scatter(y_pred, residuals, alpha=0.2, s=6, color='#4C72B0')
    axes[1].axhline(0, color='red', linestyle='--', lw=1.5)
    axes[1].set_title('잔차 vs 예측값', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('예측 금리 (%)'); axes[1].set_ylabel('잔차')
    axes[2].scatter(y_test, np.abs(residuals), alpha=0.2, s=6, color='#C44E52')
    axes[2].set_title('절대 잔차 vs 실제값', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('실제 금리 (%)'); axes[2].set_ylabel('|잔차|')
    plt.suptitle('잔차 분석', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    savefig('13_residual_analysis.png')

    # ── 분류 모델 4종 비교 → 14, 15 ────────────────────────────────────
    print('\n[4/4] 분류 4종 비교 → 14/15 생성...')
    Xc = df[FEATURES_CLS]
    yc = df['loan_status']
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        Xc, yc, test_size=0.2, random_state=SEED
    )

    clf_models = {
        '로지스틱 회귀':       LogisticRegression(max_iter=1000, random_state=SEED),
        'RandomForest':        RandomForestClassifier(n_estimators=50, random_state=SEED, n_jobs=-1),
        'GradientBoosting':    GradientBoostingClassifier(n_estimators=100, random_state=SEED),
        'XGBoost':             XGBClassifier(n_estimators=100, random_state=SEED,
                                             eval_metric='logloss', verbosity=0),
    }
    clf_results = {}
    clf_probs   = {}
    for name, model in clf_models.items():
        print(f'  {name} ...', end=' ', flush=True)
        model.fit(Xc_train, yc_train)
        prob = model.predict_proba(Xc_test)[:, 1]
        pred = model.predict(Xc_test)
        clf_probs[name] = prob
        clf_results[name] = {
            'AUC-ROC':  round(float(roc_auc_score(yc_test, prob)), 4),
            'F1':       round(float(f1_score(yc_test, pred)), 4),
            'Precision': round(float(precision_score(yc_test, pred)), 4),
            'Recall':   round(float(recall_score(yc_test, pred)), 4),
        }
        print(f'AUC={clf_results[name]["AUC-ROC"]:.4f}')

    # 14 분류 모델 성능 비교 막대그래프
    clf_names  = list(clf_results.keys())
    clf_colors = ['#4C72B0', '#8172B2', '#C44E52', '#DD8452']
    metrics_to_plot = ['AUC-ROC', 'F1', 'Precision', 'Recall']
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, metric in zip(axes, metrics_to_plot):
        vals     = [clf_results[m][metric] for m in clf_names]
        bars     = ax.bar(clf_names, vals, color=clf_colors, width=0.5, edgecolor='white')
        best_idx = int(np.argmax(vals))
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        ax.set_title(f'{metric} (높을수록 좋음)', fontsize=10, fontweight='bold')
        ax.set_xticklabels(clf_names, rotation=15, ha='right', fontsize=8)
        ax.set_ylim(0, 1.1)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f'{v:.4f}', ha='center', fontsize=8)
    plt.suptitle('분류 모델별 성능 비교 (부도 예측)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    savefig('14_clf_comparison.png')

    # 15 ROC 곡선 오버레이
    fig, ax = plt.subplots(figsize=(7, 6))
    line_styles = ['-', '--', '-.', ':']
    for (name, prob), ls in zip(clf_probs.items(), line_styles):
        fpr, tpr, _ = roc_curve(yc_test, prob)
        auc = clf_results[name]['AUC-ROC']
        ax.plot(fpr, tpr, lw=2, linestyle=ls, label=f'{name} (AUC={auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.5)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC 곡선 비교 — 4개 분류 모델', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    savefig('15_roc_curves.png')

    print('\n=== 완료 ===')
    rmse_dep = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae_dep  = float(mean_absolute_error(y_test, y_pred))
    print(f'회귀 배포 모델 — RMSE: {rmse_dep:.4f}  MAE: {mae_dep:.4f}  R2: {r2_dep:.4f}')
    print('분류 비교 결과:')
    for name, r in clf_results.items():
        print(f'  {name}: AUC={r["AUC-ROC"]}  F1={r["F1"]}  P={r["Precision"]}  R={r["Recall"]}')


if __name__ == '__main__':
    main()
