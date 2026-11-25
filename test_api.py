from fastapi.testclient import TestClient
from api import app

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

def test_predict_normal():
    device_data = {
        "device_type": "L",
        "air_temperature": 300.2,
        "process_temperature": 310.5,
        "rotational_speed": 1500,
        "torque": 42.8,
        "tool_wear": 180
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=device_data)
        assert response.status_code == 200
        result = response.json()
        assert 0 <= result["failure_probability"] <= 1
        assert result["predicted_failure"] == 0
        assert result["threshold"] == 0.1

def test_predict_high_risk():
    device_data = {
        "device_type": "L",
        "air_temperature": 304,
        "process_temperature": 312,
        "rotational_speed": 1300,
        "torque": 65,
        "tool_wear": 240
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=device_data)
        assert response.status_code == 200
        result = response.json()
        assert 0 <= result["failure_probability"] <= 1
        assert result["predicted_failure"] == 1
        assert result["threshold"] == 0.1

def test_predict_invalid_type():
    device_data = {
        "device_type": "Z",
        "air_temperature": 300.2,
        "process_temperature": 310.5,
        "rotational_speed": 1500,
        "torque": 42.8,
        "tool_wear": 180
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=device_data)
        assert response.status_code == 422
        result = response.json()
        assert "detail" in result
        assert result["detail"][0]["type"] == "literal_error"
        assert result["detail"][0]["loc"] == ["body", "device_type"]