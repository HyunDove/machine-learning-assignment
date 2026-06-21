def test_predict_missing_field(client):
    resp = client.post("/api/predictions/", json={})
    assert resp.status_code == 400
