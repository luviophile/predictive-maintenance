"""提供设备故障推理接口，并管理模型产物在服务中的加载与释放。"""

from pydantic import BaseModel
from typing import Literal
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI
from persistence import load_artifacts
from preprocessing import preprocess_input
from model import predict_torch_proba
from evaluation import predict_with_threshold

@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理 FastAPI 应用的推理资源生命周期。

    参数
        app（FastAPI）为当前应用实例，使用 app.state 在请求之间共享推理资源。

    产出
        None。yield 前加载模型、标准化器和配置，应用关闭时移除这些引用。
    """
    print("服务器正在启动")
    model, scaler, metadata = load_artifacts()
    # 将启动时加载的资源绑定到应用实例，避免每次预测请求都从磁盘重新读取。
    app.state.model = model
    app.state.scaler = scaler
    app.state.metadata = metadata
    try:
        yield
    finally:
        del app.state.model
        del app.state.scaler
        del app.state.metadata
        print("服务器正在关闭")

app = FastAPI(
    title="工业设备智能故障预测系统",
    lifespan=lifespan
)



class DeviceInput(BaseModel):
    """描述单条预测请求，并由 Pydantic 在路由执行前校验请求体。

    字段
        device_type（str）为设备类型，允许值为 L、M、H。
        air_temperature、process_temperature（float）分别为空气温度、工艺温度，单位为 K。
        rotational_speed（int）为转速，单位为 rpm。
        torque（float）为扭矩，单位为 N·m。
        tool_wear（int）为刀具磨损时间，单位为分钟。
    """
    device_type: Literal["L", "M", "H"]
    air_temperature: float
    process_temperature: float
    rotational_speed: int
    torque: float
    tool_wear: int



@app.get("/health")
def health():
    """返回服务存活状态。

    返回
        dict[str, str]，包含固定的 ok 状态。
    """
    return {"status": "ok"}

@app.post("/predict")
def predict(device: DeviceInput):
    """对单台设备执行预处理、概率推理和阈值分类。

    参数
        device（DeviceInput）为 FastAPI 解析并校验后的单条设备记录。

    返回
        dict，包含故障概率（float）、类别（int，0 表示正常，1 表示故障）
        及本次分类采用的阈值（float）。
    """
    # 将 API 字段映射到训练时的原始列名，构造形状为（1，6）的表以复用批量预处理。
    raw_df = pd.DataFrame([{
        "Type": device.device_type,
        "Torque": device.torque,
        "Air temperature": device.air_temperature,
        "Process temperature": device.process_temperature,
        "Rotational speed": device.rotational_speed,
        "Tool wear": device.tool_wear
    }])
    processed_df = preprocess_input(raw_df, app.state.scaler, app.state.metadata)
    failure_probs = predict_torch_proba(app.state.model,processed_df)
    # 使用与发布权重一同保存的阈值，沿用验证集上的业务成本决策。
    threshold = app.state.metadata["threshold"]
    y_pred = predict_with_threshold(failure_probs, threshold)
    # 从单条批量结果取出标量并转为 Python 原生数值，供 FastAPI 序列化。
    return {
        "failure_probability": float(failure_probs[0]),
        "predicted_failure": int(y_pred[0]),
        "threshold": threshold
    }
