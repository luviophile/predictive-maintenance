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
    print("服务器正在启动")
    model, scaler, metadata = load_artifacts()
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
    device_type: Literal["L", "M", "H"]
    air_temperature: float
    process_temperature: float
    rotational_speed: int
    torque: float
    tool_wear: int



@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(device: DeviceInput):
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
    threshold = app.state.metadata["threshold"]
    y_pred = predict_with_threshold(failure_probs, threshold)
    return {
        "failure_probability": float(failure_probs[0]),
        "predicted_failure": int(y_pred[0]),
        "threshold": threshold
    }
