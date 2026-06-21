VALID_PAYLOAD = {
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
    "cb_person_cred_hist_length": 5,
}


def test_predict_success(client):
    resp = client.post("/api/predictions/", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    data = resp.get_json()
    assert "loan_int_rate" in data
    assert "loan_status_prob" in data
    assert "loan_status_label" in data
    assert isinstance(data["loan_int_rate"], float)
    assert 0.0 <= data["loan_status_prob"] <= 1.0
    assert data["loan_status_label"] in ("정상", "부도")


def test_predict_missing_field(client):
    resp = client.post("/api/predictions/", json={})
    assert resp.status_code == 400


def test_predict_invalid_grade(client):
    payload = {**VALID_PAYLOAD, "loan_grade": "Z"}
    resp = client.post("/api/predictions/", json=payload)
    assert resp.status_code == 400


def test_predict_invalid_home_ownership(client):
    payload = {**VALID_PAYLOAD, "person_home_ownership": "UNKNOWN"}
    resp = client.post("/api/predictions/", json=payload)
    assert resp.status_code == 400


def test_predict_high_risk_profile(client):
    payload = {
        **VALID_PAYLOAD,
        "loan_grade": "G",
        "loan_status": 1,
        "cb_person_default_on_file": "Y",
        "loan_percent_income": 0.8,
    }
    resp = client.post("/api/predictions/", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["loan_int_rate"] > 10.0


def test_predict_low_risk_profile(client):
    payload = {
        **VALID_PAYLOAD,
        "loan_grade": "A",
        "loan_status": 0,
        "cb_person_default_on_file": "N",
        "loan_percent_income": 0.05,
    }
    resp = client.post("/api/predictions/", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["loan_int_rate"] < 15.0
