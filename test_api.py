"""使用发布产物验证 FastAPI 的健康检查、端到端预测和输入校验。"""

from fastapi.testclient import TestClient
from api import app

def test_health():
    """断言健康端点返回 HTTP 200 和约定的响应体。"""
    # 上下文管理器触发应用的启动和关闭流程，使测试覆盖实际的 Lifespan。
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

def test_predict_normal():
    """验证常规工况样本返回正常类别、合法概率和发布阈值。"""
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
        # 检查概率范围和最终决策，并核对响应使用的是发布产物中的阈值。
        assert 0 <= result["failure_probability"] <= 1
        assert result["predicted_failure"] == 0
        assert result["threshold"] == 0.1

def test_predict_high_risk():
    """验证高风险样本能经完整推理链路返回故障类别，并保留同一阈值。"""
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
    """验证类型枚举拒绝非法设备类型，并返回可定位到请求字段的校验错误。"""
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
        # 422 表示请求未通过校验。错误类型和字段位置共同确认是 device_type 触发的枚举校验。
        assert response.status_code == 422
        result = response.json()
        assert "detail" in result
        assert result["detail"][0]["type"] == "literal_error"
        assert result["detail"][0]["loc"] == ["body", "device_type"]