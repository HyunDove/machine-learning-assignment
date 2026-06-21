import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

ROOT              = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR         = os.path.join(ROOT, "models")
RATE_MODEL_PATH   = os.path.join(MODEL_DIR, "loan_rate_model.pkl")
STATUS_MODEL_PATH = os.path.join(MODEL_DIR, "loan_status_model.pkl")
PROCESSED_PATH    = os.path.join(ROOT, "data", "processed", "credit_risk_cleaned.csv")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

ENCODINGS = {
    "person_home_ownership": {"MORTGAGE": 0, "OTHER": 1, "OWN": 2, "RENT": 3},
    "loan_intent": {
        "DEBTCONSOLIDATION": 0, "EDUCATION": 1, "HOMEIMPROVEMENT": 2,
        "MEDICAL": 3, "PERSONAL": 4, "VENTURE": 5,
    },
    "loan_grade": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6},
    "cb_person_default_on_file": {"N": 0, "Y": 1},
}

FEATURES = [
    "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "loan_intent", "loan_grade", "loan_amnt", "loan_status",
    "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length",
]
FEATURES_NO_STATUS = [f for f in FEATURES if f != "loan_status"]

MODEL_RESULTS = {
    "선형 회귀":        {"RMSE": 1.1725, "MAE": 0.9139, "R²": 0.8702},
    "릿지 회귀":        {"RMSE": 1.1725, "MAE": 0.9140, "R²": 0.8702},
    "그래디언트 부스팅": {"RMSE": 1.0079, "MAE": 0.7899, "R²": 0.9041},
    "랜덤 포레스트":    {"RMSE": 0.9967, "MAE": 0.7735, "R²": 0.9062},
}


# ── 모델 로드 ────────────────────────────────────────────────────────────────

def ensure_models():
    if os.path.exists(RATE_MODEL_PATH) and os.path.exists(STATUS_MODEL_PATH):
        return
    with st.spinner("모델 파일이 없어 자동으로 학습 중입니다... (최초 1회)"):
        sys.path.insert(0, ROOT)
        from ml.preprocessing import load_raw, preprocess, save_processed
        from ml.train import train
        df = load_raw()
        df = preprocess(df)
        save_processed(df)
        train()
    st.success("모델 학습 완료!")


@st.cache_resource
def load_models():
    ensure_models()
    return joblib.load(RATE_MODEL_PATH), joblib.load(STATUS_MODEL_PATH)


@st.cache_data
def load_data():
    if not os.path.exists(PROCESSED_PATH):
        return None
    return pd.read_csv(PROCESSED_PATH)


# ── 예측 ─────────────────────────────────────────────────────────────────────

def encode_input(raw: dict) -> pd.DataFrame:
    encoded = raw.copy()
    for col, mapping in ENCODINGS.items():
        encoded[col] = mapping[encoded[col]]
    return pd.DataFrame([encoded])


def predict(raw: dict):
    regressor, classifier = load_models()
    df_reg = encode_input(raw)[FEATURES]
    df_cls = encode_input(raw)[FEATURES_NO_STATUS]
    rate = float(regressor.predict(df_reg)[0])
    prob = float(classifier.predict_proba(df_cls)[0][1])
    return round(rate, 2), round(prob * 100, 1)


# ── 페이지 설정 ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="대출 금리 예측",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #f5f7fa; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }

    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 14px; padding: 24px 28px;
        color: white; text-align: center; margin-bottom: 14px;
    }
    .result-label { font-size: 0.88rem; opacity: 0.85; margin-bottom: 4px; }
    .result-value { font-size: 2.8rem; font-weight: 800; line-height: 1.1; }
    .result-unit  { font-size: 0.95rem; opacity: 0.8; }

    .status-safe {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 14px; padding: 18px 28px; color: white; text-align: center;
    }
    .status-risk {
        background: linear-gradient(135deg, #f7971e 0%, #e53935 100%);
        border-radius: 14px; padding: 18px 28px; color: white; text-align: center;
    }
    .status-icon { font-size: 2.2rem; }
    .status-text { font-size: 1.3rem; font-weight: 700; margin-top: 4px; }

    .info-card {
        background: #f8f9ff; border-left: 4px solid #667eea;
        border-radius: 8px; padding: 12px 16px; margin-top: 10px;
        font-size: 0.88rem; color: #444;
    }
    .section-title {
        font-size: 0.75rem; font-weight: 700; color: #888;
        letter-spacing: 0.08em; text-transform: uppercase; margin: 18px 0 6px;
    }
    [data-testid="metric-container"] {
        background: #f8f9ff; border-radius: 10px;
        padding: 12px 16px; border: 1px solid #e8ecf0;
    }
</style>
""", unsafe_allow_html=True)


# ── 사이드바 입력 ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 대출 신청 정보")

    st.markdown('<p class="section-title">신청자 정보</p>', unsafe_allow_html=True)
    person_age    = st.slider("나이", 18, 100, 30)
    person_income = st.number_input("연 소득 (달러)", min_value=1_000, max_value=10_000_000,
                                    value=60_000, step=1_000)
    person_home_ownership = st.selectbox(
        "주거 형태",
        ["RENT", "OWN", "MORTGAGE", "OTHER"],
        format_func=lambda x: {"RENT": "임대", "OWN": "자가", "MORTGAGE": "담보대출", "OTHER": "기타"}[x],
    )
    person_emp_length = st.slider("재직 기간 (년)", 0.0, 60.0, 3.0, step=0.5)

    st.markdown('<p class="section-title">대출 정보</p>', unsafe_allow_html=True)
    loan_amnt = st.number_input("대출 금액 (달러)", min_value=500, max_value=500_000,
                                 value=10_000, step=500)
    loan_intent = st.selectbox(
        "대출 목적",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        format_func=lambda x: {
            "PERSONAL": "개인 용도", "EDUCATION": "교육", "MEDICAL": "의료",
            "VENTURE": "창업", "HOMEIMPROVEMENT": "주거 개선", "DEBTCONSOLIDATION": "부채 통합",
        }[x],
    )
    loan_grade = st.select_slider(
        "대출 등급 (A=우량 / G=불량)",
        options=["A", "B", "C", "D", "E", "F", "G"], value="C",
    )
    loan_status = st.radio(
        "현재 대출 상태", [0, 1],
        format_func=lambda x: "정상 상환 중" if x == 0 else "연체 / 부도",
        horizontal=True,
    )

    st.markdown('<p class="section-title">신용 정보</p>', unsafe_allow_html=True)
    cb_person_default_on_file = st.radio(
        "과거 부도 이력", ["N", "Y"],
        format_func=lambda x: "없음" if x == "N" else "있음",
        horizontal=True,
    )
    cb_person_cred_hist_length = st.slider("신용 이력 기간 (년)", 0, 30, 5)

    loan_percent_income = round(loan_amnt / person_income, 4) if person_income > 0 else 0.0
    st.info(f"소득 대비 대출 비율: **{loan_percent_income:.2%}**")
    predict_btn = st.button("예측하기", type="primary", use_container_width=True)


# ── 메인 ─────────────────────────────────────────────────────────────────────

st.title("대출 금리 예측 시스템")
st.caption("Credit Risk Dataset 기반 머신러닝 모델 · GradientBoostingRegressor + RandomForestClassifier")
st.divider()

tab1, tab2, tab3 = st.tabs(["금리 예측", "모델 성능", "데이터 인사이트"])


# ── Tab 1: 금리 예측 ──────────────────────────────────────────────────────────

with tab1:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("예측 결과")

        if predict_btn:
            raw = {
                "person_age": person_age,
                "person_income": person_income,
                "person_home_ownership": person_home_ownership,
                "person_emp_length": person_emp_length,
                "loan_intent": loan_intent,
                "loan_grade": loan_grade,
                "loan_amnt": loan_amnt,
                "loan_status": loan_status,
                "loan_percent_income": loan_percent_income,
                "cb_person_default_on_file": cb_person_default_on_file,
                "cb_person_cred_hist_length": cb_person_cred_hist_length,
            }
            with st.spinner("예측 중..."):
                rate, prob = predict(raw)
            st.session_state["result"] = (rate, prob)

        if "result" in st.session_state:
            rate, prob = st.session_state["result"]
            is_safe = prob < 50

            r1, r2 = st.columns(2)
            with r1:
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-label">예측 대출 금리</div>
                    <div class="result-value">{rate}</div>
                    <div class="result-unit">% per annum</div>
                </div>""", unsafe_allow_html=True)
            with r2:
                css   = "status-safe" if is_safe else "status-risk"
                icon  = "✅" if is_safe else "⚠️"
                label = "정상" if is_safe else "부도 위험"
                st.markdown(f"""
                <div class="{css}">
                    <div class="status-icon">{icon}</div>
                    <div class="result-label" style="color:white;opacity:.85;">부도 판정</div>
                    <div class="status-text">{label}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"#### 부도 확률: **{prob}%**")
            st.progress(prob / 100)

            grade_comment = {
                "A": "최우량 등급 — 낮은 금리가 적용될 가능성이 높습니다.",
                "B": "우량 등급 — 평균 이하의 금리를 기대할 수 있습니다.",
                "C": "양호한 등급 — 평균 수준의 금리가 적용됩니다.",
                "D": "보통 등급 — 다소 높은 금리가 적용될 수 있습니다.",
                "E": "주의 등급 — 상당히 높은 금리가 적용됩니다.",
                "F": "불량 등급 — 높은 금리와 심사 제한이 있을 수 있습니다.",
                "G": "최하위 등급 — 대출 승인이 어려울 수 있습니다.",
            }
            st.markdown(f"""
            <div class="info-card">
                등급 {loan_grade} 해석: {grade_comment[loan_grade]}
            </div>""", unsafe_allow_html=True)

        else:
            st.info("왼쪽 사이드바에서 정보를 입력하고 예측하기 버튼을 눌러주세요.")

    with col_right:
        st.subheader("입력 요약")
        summary = {
            "나이": f"{person_age}세",
            "연 소득": f"${person_income:,}",
            "주거 형태": person_home_ownership,
            "재직 기간": f"{person_emp_length}년",
            "대출 금액": f"${loan_amnt:,}",
            "대출 목적": loan_intent,
            "대출 등급": loan_grade,
            "현재 상태": "정상" if loan_status == 0 else "연체/부도",
            "소득 대비 비율": f"{loan_percent_income:.2%}",
            "과거 부도 이력": "없음" if cb_person_default_on_file == "N" else "있음",
            "신용 이력": f"{cb_person_cred_hist_length}년",
        }
        for k, v in summary.items():
            c1, c2 = st.columns([1, 1])
            c1.markdown(f"**{k}**")
            c2.markdown(v)


# ── Tab 2: 모델 성능 ──────────────────────────────────────────────────────────

with tab2:
    st.subheader("모델별 성능 비교")
    st.caption("테스트 세트 기준 (전체의 20%, random_state=42)")

    df_metrics = pd.DataFrame(MODEL_RESULTS).T.reset_index()
    df_metrics.columns = ["모델", "RMSE (낮을수록 좋음)", "MAE (낮을수록 좋음)", "R² (높을수록 좋음)"]

    st.dataframe(
        df_metrics.style
            .highlight_max(subset=["R² (높을수록 좋음)"], color="#c6f6d5")
            .highlight_min(subset=["RMSE (낮을수록 좋음)", "MAE (낮을수록 좋음)"], color="#c6f6d5")
            .format({
                "RMSE (낮을수록 좋음)": "{:.4f}",
                "MAE (낮을수록 좋음)": "{:.4f}",
                "R² (높을수록 좋음)": "{:.4f}",
            }),
        use_container_width=True, hide_index=True,
    )

    st.divider()

    model_names = list(MODEL_RESULTS.keys())
    base_colors = ["#94a3b8", "#94a3b8", "#667eea", "#764ba2"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.patch.set_facecolor("#ffffff")

    for ax, metric, lower in zip(axes, ["RMSE", "MAE", "R²"], [True, True, False]):
        vals   = [MODEL_RESULTS[m][metric] for m in model_names]
        colors = list(base_colors)
        best   = int(np.argmin(vals)) if lower else int(np.argmax(vals))
        colors[best] = "#f59e0b"
        bars = ax.bar(model_names, vals, color=colors, width=0.55, edgecolor="white", linewidth=0.5)
        ax.set_title(f"{metric}  ({'낮을수록 좋음' if lower else '높을수록 좋음'})",
                     fontsize=11, fontweight="bold", pad=10)
        ax.set_facecolor("#ffffff")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{v:.4f}", ha="center", fontsize=8.5, fontweight="bold")

    plt.suptitle("회귀 모델 성능 비교  (최우수 = 노란색)", fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.divider()
    st.subheader("최종 채택 모델 — GradientBoostingRegressor")
    m1, m2, m3 = st.columns(3)
    m1.metric("RMSE", "1.0079", delta="-0.1646 vs 선형회귀", delta_color="inverse")
    m2.metric("MAE",  "0.7899", delta="-0.1240 vs 선형회귀", delta_color="inverse")
    m3.metric("R²",   "0.9041", delta="+0.0339 vs 선형회귀")
    st.caption("분류 모델(RandomForestClassifier) AUC-ROC: **0.9397**")


# ── Tab 3: 데이터 인사이트 ────────────────────────────────────────────────────

with tab3:
    df_data = load_data()

    if df_data is None:
        st.warning("전처리된 데이터 파일이 없습니다. `python ml/preprocessing.py`를 먼저 실행해주세요.")
    else:
        GRADE_MAP = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G"}
        df = df_data.copy()
        df["loan_grade_label"] = df["loan_grade"].map(GRADE_MAP)

        st.subheader("주요 변수별 평균 대출 금리")
        st.caption(f"전처리 데이터 기준 ({len(df):,}행)")

        palette_grade = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, 7))

        # 행 1: 대출 등급 / 나이 구간
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 대출 등급별 평균 금리")
            grade_avg = (
                df.groupby("loan_grade_label")["loan_int_rate"]
                .mean()
                .reindex(["A", "B", "C", "D", "E", "F", "G"])
            )
            fig, ax = plt.subplots(figsize=(6, 3.8))
            fig.patch.set_facecolor("#ffffff")
            bars = ax.bar(grade_avg.index, grade_avg.values,
                          color=palette_grade, edgecolor="white", width=0.6)
            ax.set_ylabel("평균 금리 (%)", fontsize=10)
            ax.set_facecolor("#ffffff")
            ax.spines[["top", "right"]].set_visible(False)
            for bar, val in zip(bars, grade_avg.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{val:.2f}%", ha="center", fontsize=9, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col2:
            st.markdown("#### 나이 구간별 평균 금리")
            bins_age   = [18, 25, 30, 35, 40, 50, 200]
            labels_age = ["18-24", "25-29", "30-34", "35-39", "40-49", "50+"]
            df["age_group"] = pd.cut(df["person_age"], bins=bins_age,
                                     labels=labels_age, right=False)
            age_avg = df.groupby("age_group", observed=True)["loan_int_rate"].mean()
            fig, ax = plt.subplots(figsize=(6, 3.8))
            fig.patch.set_facecolor("#ffffff")
            bars = ax.bar(age_avg.index.astype(str), age_avg.values,
                          color="#667eea", edgecolor="white", width=0.6, alpha=0.85)
            ax.set_ylabel("평균 금리 (%)", fontsize=10)
            ax.set_facecolor("#ffffff")
            ax.spines[["top", "right"]].set_visible(False)
            for bar, val in zip(bars, age_avg.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{val:.2f}%", ha="center", fontsize=9, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # 행 2: 소득 구간 / 대출금액 구간
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("#### 연소득 구간별 평균 금리")
            bins_inc   = [0, 30_000, 60_000, 100_000, 200_000, float("inf")]
            labels_inc = ["<$30K", "$30K-60K", "$60K-100K", "$100K-200K", ">$200K"]
            df["income_group"] = pd.cut(df["person_income"], bins=bins_inc,
                                        labels=labels_inc, right=False)
            inc_avg = df.groupby("income_group", observed=True)["loan_int_rate"].mean()
            fig, ax = plt.subplots(figsize=(6, 3.8))
            fig.patch.set_facecolor("#ffffff")
            bars = ax.bar(inc_avg.index.astype(str), inc_avg.values,
                          color="#764ba2", edgecolor="white", width=0.6, alpha=0.85)
            ax.set_ylabel("평균 금리 (%)", fontsize=10)
            ax.set_facecolor("#ffffff")
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticklabels(inc_avg.index.astype(str), rotation=15, ha="right", fontsize=9)
            for bar, val in zip(bars, inc_avg.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{val:.2f}%", ha="center", fontsize=9, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col4:
            st.markdown("#### 대출금액 구간별 평균 금리")
            bins_amnt   = [0, 5_000, 10_000, 20_000, 35_001]
            labels_amnt = ["<$5K", "$5K-10K", "$10K-20K", ">$20K"]
            df["amnt_group"] = pd.cut(df["loan_amnt"], bins=bins_amnt,
                                      labels=labels_amnt, right=False)
            amnt_avg = df.groupby("amnt_group", observed=True)["loan_int_rate"].mean()
            fig, ax = plt.subplots(figsize=(6, 3.8))
            fig.patch.set_facecolor("#ffffff")
            bars = ax.bar(amnt_avg.index.astype(str), amnt_avg.values,
                          color="#11998e", edgecolor="white", width=0.6, alpha=0.85)
            ax.set_ylabel("평균 금리 (%)", fontsize=10)
            ax.set_facecolor("#ffffff")
            ax.spines[["top", "right"]].set_visible(False)
            for bar, val in zip(bars, amnt_avg.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{val:.2f}%", ha="center", fontsize=9, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # 행 3: 재직기간 / 신용이력
        col5, col6 = st.columns(2)

        with col5:
            st.markdown("#### 재직기간 구간별 평균 금리")
            bins_emp   = [0, 2, 5, 10, 20, float("inf")]
            labels_emp = ["<2년", "2-4년", "5-9년", "10-19년", "20년+"]
            df["emp_group"] = pd.cut(df["person_emp_length"], bins=bins_emp,
                                     labels=labels_emp, right=False)
            emp_avg = df.groupby("emp_group", observed=True)["loan_int_rate"].mean()
            fig, ax = plt.subplots(figsize=(6, 3.8))
            fig.patch.set_facecolor("#ffffff")
            bars = ax.bar(emp_avg.index.astype(str), emp_avg.values,
                          color="#f59e0b", edgecolor="white", width=0.6, alpha=0.85)
            ax.set_ylabel("평균 금리 (%)", fontsize=10)
            ax.set_facecolor("#ffffff")
            ax.spines[["top", "right"]].set_visible(False)
            for bar, val in zip(bars, emp_avg.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{val:.2f}%", ha="center", fontsize=9, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col6:
            st.markdown("#### 신용이력 기간별 평균 금리")
            bins_cred   = [0, 3, 6, 10, 15, float("inf")]
            labels_cred = ["<3년", "3-5년", "6-9년", "10-14년", "15년+"]
            df["cred_group"] = pd.cut(df["cb_person_cred_hist_length"], bins=bins_cred,
                                      labels=labels_cred, right=False)
            cred_avg = df.groupby("cred_group", observed=True)["loan_int_rate"].mean()
            fig, ax = plt.subplots(figsize=(6, 3.8))
            fig.patch.set_facecolor("#ffffff")
            bars = ax.bar(cred_avg.index.astype(str), cred_avg.values,
                          color="#c44e52", edgecolor="white", width=0.6, alpha=0.85)
            ax.set_ylabel("평균 금리 (%)", fontsize=10)
            ax.set_facecolor("#ffffff")
            ax.spines[["top", "right"]].set_visible(False)
            for bar, val in zip(bars, cred_avg.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{val:.2f}%", ha="center", fontsize=9, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # 요약 통계
        st.divider()
        st.subheader("주요 통계 요약")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("전체 평균 금리",    f"{df['loan_int_rate'].mean():.2f}%")
        s2.metric("중앙값 금리",       f"{df['loan_int_rate'].median():.2f}%")
        s3.metric("최저 평균 (A등급)", f"{df[df['loan_grade'] == 0]['loan_int_rate'].mean():.2f}%")
        s4.metric("최고 평균 (G등급)", f"{df[df['loan_grade'] == 6]['loan_int_rate'].mean():.2f}%")
