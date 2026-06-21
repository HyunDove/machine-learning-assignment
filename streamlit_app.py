import os
import sys
import joblib
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
RATE_MODEL_PATH = os.path.join(MODEL_DIR, "loan_rate_model.pkl")
STATUS_MODEL_PATH = os.path.join(MODEL_DIR, "loan_status_model.pkl")

# LabelEncoder는 알파벳 순 정렬로 동일하게 인코딩됨
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


# ── 모델 자동 학습 ─────────────────────────────────────────────────────────────

def ensure_models():
    if os.path.exists(RATE_MODEL_PATH) and os.path.exists(STATUS_MODEL_PATH):
        return
    with st.spinner("🔧 모델 파일이 없어 자동으로 학습 중입니다... (최초 1회)"):
        sys.path.insert(0, ROOT)
        from ml.preprocessing import load_raw, preprocess, save_processed
        from ml.train import train
        df = load_raw()
        df = preprocess(df)
        save_processed(df)
        train()
    st.success("✅ 모델 학습 완료!")


@st.cache_resource
def load_models():
    ensure_models()
    regressor = joblib.load(RATE_MODEL_PATH)
    classifier = joblib.load(STATUS_MODEL_PATH)
    return regressor, classifier


# ── 예측 ───────────────────────────────────────────────────────────────────────

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


# ── UI ─────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="💳 대출 금리 예측",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 28px 32px;
        color: white;
        text-align: center;
        margin-bottom: 16px;
    }
    .result-label { font-size: 0.95rem; opacity: 0.85; margin-bottom: 4px; }
    .result-value { font-size: 3rem; font-weight: 800; line-height: 1.1; }
    .result-unit  { font-size: 1.1rem; opacity: 0.8; }

    .status-safe {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 16px;
        padding: 20px 32px;
        color: white;
        text-align: center;
    }
    .status-risk {
        background: linear-gradient(135deg, #f7971e 0%, #e53935 100%);
        border-radius: 16px;
        padding: 20px 32px;
        color: white;
        text-align: center;
    }
    .status-icon  { font-size: 2.5rem; }
    .status-text  { font-size: 1.4rem; font-weight: 700; margin-top: 4px; }

    .info-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 12px;
        font-size: 0.9rem;
        color: #444;
    }
    .section-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #888;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 20px 0 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── 헤더 ──────────────────────────────────────────────────────────────────────
st.title("💳 대출 금리 예측 시스템")
st.markdown("**Credit Risk Dataset** 기반 머신러닝 모델로 대출 금리와 부도 위험을 예측합니다.")
st.divider()

# ── 사이드바 입력 ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📝 대출 신청 정보 입력")

    st.markdown('<p class="section-title">👤 신청자 정보</p>', unsafe_allow_html=True)
    person_age = st.slider("나이", 18, 100, 30)
    person_income = st.number_input("연 소득 (달러)", min_value=1000, max_value=10_000_000, value=60000, step=1000)
    person_home_ownership = st.selectbox(
        "주거 형태",
        ["RENT", "OWN", "MORTGAGE", "OTHER"],
        format_func=lambda x: {"RENT": "🏠 임대 (RENT)", "OWN": "🏡 자가 (OWN)",
                                "MORTGAGE": "🏦 담보대출 (MORTGAGE)", "OTHER": "기타 (OTHER)"}[x],
    )
    person_emp_length = st.slider("재직 기간 (년)", 0.0, 60.0, 3.0, step=0.5)

    st.markdown('<p class="section-title">💰 대출 정보</p>', unsafe_allow_html=True)
    loan_amnt = st.number_input("대출 금액 (달러)", min_value=500, max_value=500_000, value=10000, step=500)
    loan_intent = st.selectbox(
        "대출 목적",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        format_func=lambda x: {
            "PERSONAL": "💼 개인 용도", "EDUCATION": "🎓 교육",
            "MEDICAL": "🏥 의료", "VENTURE": "🚀 창업",
            "HOMEIMPROVEMENT": "🔨 주거 개선", "DEBTCONSOLIDATION": "🔄 부채 통합",
        }[x],
    )
    loan_grade = st.select_slider(
        "대출 등급 (A=우량 · G=불량)",
        options=["A", "B", "C", "D", "E", "F", "G"],
        value="C",
    )
    loan_status = st.radio(
        "현재 대출 상태",
        [0, 1],
        format_func=lambda x: "✅ 정상 상환 중" if x == 0 else "⚠️ 연체 / 부도",
        horizontal=True,
    )

    st.markdown('<p class="section-title">📋 신용 정보</p>', unsafe_allow_html=True)
    cb_person_default_on_file = st.radio(
        "과거 부도 이력",
        ["N", "Y"],
        format_func=lambda x: "없음 ✅" if x == "N" else "있음 ⚠️",
        horizontal=True,
    )
    cb_person_cred_hist_length = st.slider("신용 이력 기간 (년)", 0, 30, 5)

    loan_percent_income = round(loan_amnt / person_income, 4) if person_income > 0 else 0.0
    st.info(f"📊 소득 대비 대출 비율: **{loan_percent_income:.2%}** (자동 계산)")

    predict_btn = st.button("🔍 예측하기", type="primary", use_container_width=True)

# ── 메인 영역 ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📊 예측 결과")

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
                <div class="result-label">📈 예측 대출 금리</div>
                <div class="result-value">{rate}</div>
                <div class="result-unit">% per annum</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            status_class = "status-safe" if is_safe else "status-risk"
            icon = "✅" if is_safe else "⚠️"
            label = "정상" if is_safe else "부도 위험"
            st.markdown(f"""
            <div class="{status_class}">
                <div class="status-icon">{icon}</div>
                <div class="result-label" style="color:white;opacity:0.85;">부도 판정</div>
                <div class="status-text">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"#### 📉 부도 확률: **{prob}%**")
        st.progress(prob / 100)

        grade_comment = {
            "A": "최우량 등급입니다. 낮은 금리가 적용될 가능성이 높습니다.",
            "B": "우량 등급입니다. 평균 이하의 금리를 기대할 수 있습니다.",
            "C": "양호한 등급입니다. 평균 수준의 금리가 적용됩니다.",
            "D": "보통 등급입니다. 다소 높은 금리가 적용될 수 있습니다.",
            "E": "주의 등급입니다. 상당히 높은 금리가 적용됩니다.",
            "F": "불량 등급입니다. 높은 금리와 심사 제한이 있을 수 있습니다.",
            "G": "최하위 등급입니다. 대출 승인이 어려울 수 있습니다.",
        }
        st.markdown(f"""
        <div class="info-card">
            💡 <b>등급 {loan_grade} 해석:</b> {grade_comment[loan_grade]}
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("👈 왼쪽 사이드바에서 대출 정보를 입력하고 **예측하기** 버튼을 눌러주세요.")

with col_right:
    st.subheader("📋 입력 요약")
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
    st.divider()
    st.caption("🤖 모델: GradientBoostingRegressor (회귀) · RandomForestClassifier (분류)")
    st.caption("📊 학습 데이터: Credit Risk Dataset (29,459행)")
    st.caption("📏 회귀 성능: RMSE 1.008 · MAE 0.790 · R² 0.904")
